/**
 * @file defect_map_pipeline_node.cpp
 * @brief Implementation of the modular defect_map_pipeline ROS 2 node.
 * @author Alessio Lovato
 */
#include "defect_map_pipeline/defect_map_pipeline_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <future>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2/exceptions.h>
#include <tf2/time.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <opencv2/imgproc.hpp>

#include "rclcpp_components/register_node_macro.hpp"

namespace defect_map_pipeline
{
namespace
{

/**
 * @brief Read metric depth value from a depth image pixel.
 * @param depth Depth image matrix (16UC1 in mm or 32FC1 in m).
 * @param y Pixel row.
 * @param x Pixel column.
 * @return Depth in meters, or NaN when invalid/out-of-range.
 */
double readDepthMeters(const cv::Mat & depth, int y, int x)
{
  if (y < 0 || x < 0 || y >= depth.rows || x >= depth.cols) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  if (depth.type() == CV_16UC1) {
    const auto d = depth.at<uint16_t>(y, x);
    if (d == 0U) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return static_cast<double>(d) * 0.001;
  }

  if (depth.type() == CV_32FC1) {
    const auto d = depth.at<float>(y, x);
    if (!std::isfinite(d) || d <= 0.0F) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return static_cast<double>(d);
  }

  return std::numeric_limits<double>::quiet_NaN();
}

/**
 * @brief Locally normalize a single-channel float map.
 * @param input Input map.
 * @param sigma Gaussian sigma used for local statistics.
 * @return Locally normalized output map.
 */
cv::Mat localNormalize(const cv::Mat & input, double sigma)
{
  cv::Mat mu;
  cv::GaussianBlur(input, mu, cv::Size(0, 0), sigma, sigma);

  cv::Mat sq = input.mul(input);
  cv::Mat mu2;
  cv::GaussianBlur(sq, mu2, cv::Size(0, 0), sigma, sigma);

  cv::Mat var;
  cv::max(mu2 - mu.mul(mu), 0.0, var);

  cv::Mat stddev;
  cv::sqrt(var, stddev);

  cv::Mat normalized;
  normalized = (input - mu) / (stddev + 1e-6);
  return normalized;
}

/**
 * @brief Clip a float map to [-clip, clip] and map it to uint8.
 * @param input Input map.
 * @param clip Symmetric clipping magnitude.
 * @return Clipped and linearly mapped uint8 image.
 */
cv::Mat clipMapToU8(const cv::Mat & input, double clip)
{
  cv::Mat clipped;
  cv::min(cv::max(input, -clip), clip, clipped);
  cv::Mat mapped = (clipped + clip) * (255.0 / (2.0 * clip));
  cv::Mat out;
  mapped.convertTo(out, CV_8UC1);
  return out;
}

}  // namespace

/**
 * @brief Construct the pipeline node and initialize all internal modules.
 * @param options ROS node options.
 */
DefectMapPipelineNode::DefectMapPipelineNode(const rclcpp::NodeOptions & options)
: Node("defect_map_pipeline", options)
{
  // Core topics/services.
  declare_parameter<std::string>("rgb_topic", "/camera/camera/color/image_raw");
  declare_parameter<std::string>("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw");
  declare_parameter<std::string>("camera_info_topic", "/camera/camera/color/camera_info");
  declare_parameter<std::string>("prediction_service_name", "/defect_map_prediction/segment_image");
  declare_parameter<std::string>("map_add_defects_service_name", "/defect_map/add_defects");

  // Frames/runtime.
  declare_parameter<std::string>("base_frame", "world");
  declare_parameter<std::string>("camera_frame", "camera_color_optical_frame");
  declare_parameter<int>("crop_width", 512);
  declare_parameter<int>("crop_height", 512);
  declare_parameter<int>("expected_shots_per_image", 4);
  declare_parameter<int>("max_queue_size", 16);
  declare_parameter<int>("worker_threads", 1);
  declare_parameter<int>("prediction_timeout_ms", 5000);
  declare_parameter<int>("map_write_timeout_ms", 2000);
  declare_parameter<int>("tf_lookup_timeout_ms", 2000);
  declare_parameter<int>("capture_next_frame_timeout_ms", 1000);
  declare_parameter<int>("sync_queue_size", 30);
  declare_parameter<bool>("tf_preflight_enabled", true);

  // Voxelization + preprocessing.
  declare_parameter<double>("voxel_size_m", 0.01);
  declare_parameter<std::string>("preprocess_mode", "composite");
  declare_parameter<std::vector<std::string>>(
    "light_order", std::vector<std::string>{"left", "bottom", "right", "top"});
  declare_parameter<double>("ps_curv_sigma", 60.0);
  declare_parameter<double>("ps_height_sigma", 40.0);
  declare_parameter<double>("ps_encode_clip", 4.0);

  rgb_topic_ = get_parameter("rgb_topic").as_string();
  depth_topic_ = get_parameter("depth_topic").as_string();
  camera_info_topic_ = get_parameter("camera_info_topic").as_string();
  prediction_service_name_ = get_parameter("prediction_service_name").as_string();
  map_add_defects_service_name_ = get_parameter("map_add_defects_service_name").as_string();

  base_frame_ = get_parameter("base_frame").as_string();
  camera_frame_ = get_parameter("camera_frame").as_string();
  crop_width_ = get_parameter("crop_width").as_int();
  crop_height_ = get_parameter("crop_height").as_int();
  expected_shots_per_image_ = get_parameter("expected_shots_per_image").as_int();
  max_queue_size_ = get_parameter("max_queue_size").as_int();
  worker_threads_ = static_cast<int>(std::max<int64_t>(1, get_parameter("worker_threads").as_int())); // ROS 2 returns int64_t, thus need to cast '1'.
  prediction_timeout_ms_ = get_parameter("prediction_timeout_ms").as_int();
  map_write_timeout_ms_ = get_parameter("map_write_timeout_ms").as_int();
  tf_lookup_timeout_ms_ = get_parameter("tf_lookup_timeout_ms").as_int();
  capture_next_frame_timeout_ms_ = get_parameter("capture_next_frame_timeout_ms").as_int();
  sync_queue_size_ = static_cast<int>(std::max<int64_t>(1, get_parameter("sync_queue_size").as_int()));
  tf_preflight_enabled_ = get_parameter("tf_preflight_enabled").as_bool();

  voxel_size_m_ = get_parameter("voxel_size_m").as_double();
  preprocess_mode_ = toUpper(get_parameter("preprocess_mode").as_string());
  light_order_ = get_parameter("light_order").as_string_array();
  ps_curv_sigma_ = get_parameter("ps_curv_sigma").as_double();
  ps_height_sigma_ = get_parameter("ps_height_sigma").as_double();
  ps_encode_clip_ = get_parameter("ps_encode_clip").as_double();

  if (expected_shots_per_image_ <= 0) {
    throw std::invalid_argument("Parameter expected_shots_per_image must be > 0");
  }

  const auto expected_size = static_cast<size_t>(expected_shots_per_image_);
  if (light_order_.size() != expected_size) {
    throw std::invalid_argument(
            "Parameter light_order size (" + std::to_string(light_order_.size()) +
            ") must match expected_shots_per_image (" +
            std::to_string(expected_shots_per_image_) + ")");
  }

  // TF preflight module.
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  if (tf_preflight_enabled_) {
    tf_preflight_start_ = now();
    tf_preflight_timer_ = create_wall_timer(
      std::chrono::milliseconds(200),
      std::bind(&DefectMapPipelineNode::tfPreflightTick, this));
  } else {
    tf_ready_ = true;
    RCLCPP_WARN(
      get_logger(),
      "TF preflight disabled by parameter tf_preflight_enabled=false. Capture will not be blocked on startup TF availability.");
  }

  // Frame synchronization module.
  const auto sensor_qos = rclcpp::SensorDataQoS();
  const auto rmw_sensor_qos = sensor_qos.get_rmw_qos_profile();
  rgb_sub_.subscribe(this, rgb_topic_, rmw_sensor_qos);
  depth_sub_.subscribe(this, depth_topic_, rmw_sensor_qos);
  camera_info_sub_.subscribe(this, camera_info_topic_, rmw_sensor_qos);
  frame_sync_ = std::make_shared<FrameSynchronizer>(
    FrameSyncPolicy(sync_queue_size_), rgb_sub_, depth_sub_, camera_info_sub_);
  frame_sync_->registerCallback(std::bind(
      &DefectMapPipelineNode::onSynchronizedFrame,
      this,
      std::placeholders::_1,
      std::placeholders::_2,
      std::placeholders::_3));

  // Debug publishers.
  const auto debug_cloud_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
  raw_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    "~/debug/raw_defects_cloud", debug_cloud_qos);
  clustered_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    "~/debug/clustered_defects_cloud", debug_cloud_qos);
  const auto debug_image_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
  preprocessed_image_pub_ = create_publisher<sensor_msgs::msg::Image>(
    "~/debug/preprocessed_image", debug_image_qos);

  // Services owned by the pipeline node.
  capture_service_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  capture_service_ = create_service<defect_map_interfaces::srv::CaptureShot>(
    "~/capture_shot",
    std::bind(&DefectMapPipelineNode::onCaptureShot, this, std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(),
    capture_service_callback_group_);

  // Clients: predictor + map writer.
  prediction_client_ = create_client<defect_map_interfaces::srv::SegmentImage>(prediction_service_name_);
  add_defects_client_ = create_client<defect_map_interfaces::srv::AddDefects>(map_add_defects_service_name_);

  // Queue/worker orchestration module.
  for (int i = 0; i < worker_threads_; ++i) {
    workers_.emplace_back(std::bind(&DefectMapPipelineNode::workerLoop, this));
  }

  RCLCPP_INFO(
    get_logger(),
    "Pipeline started. preprocess_mode=%s expected_shots=%d max_queue=%d sync=ExactTime(queue=%d)",
    preprocess_mode_.c_str(), expected_shots_per_image_, max_queue_size_, sync_queue_size_);
  RCLCPP_INFO(get_logger(), "Prediction service endpoint: %s", prediction_service_name_.c_str());
  RCLCPP_INFO(get_logger(), "Map writer endpoint (AddDefects): %s", map_add_defects_service_name_.c_str());
}

/**
 * @brief Stop workers and release synchronization resources.
 */
DefectMapPipelineNode::~DefectMapPipelineNode()
{
  {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    capture_waiting_for_next_frame_ = false;
    capture_frame_ready_ = false;
  }
  frame_cv_.notify_all();

  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    stop_worker_ = true;
  }
  queue_cv_.notify_all();
  for (auto & worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

/**
 * @brief Check TF availability during startup preflight.
 */
void DefectMapPipelineNode::tfPreflightTick()
{
  if (tf_ready_ || tf_permanent_error_) {
    return;
  }

  try {
    (void)tf_buffer_->lookupTransform(
      base_frame_, camera_frame_, tf2::TimePointZero,
      tf2::durationFromSec(static_cast<double>(tf_lookup_timeout_ms_) / 1000.0));
    tf_ready_ = true;
    tf_preflight_timer_->cancel();
    RCLCPP_INFO(get_logger(), "TF preflight succeeded: %s <- %s", base_frame_.c_str(), camera_frame_.c_str());
    return;
  } catch (const tf2::TransformException &) {
    // Keep trying until timeout.
  }

  const auto elapsed_ms = (now() - tf_preflight_start_).nanoseconds() / 1000000LL;
  if (elapsed_ms > tf_lookup_timeout_ms_) {
    tf_permanent_error_ = true;
    tf_preflight_timer_->cancel();
    RCLCPP_ERROR(
      get_logger(), "TF preflight failed for %s <- %s within %d ms",
      base_frame_.c_str(), camera_frame_.c_str(), tf_lookup_timeout_ms_);
  }
}

/**
 * @brief Latch one synchronized frame when a capture request is waiting.
 * @param rgb Synchronized RGB image.
 * @param depth Synchronized depth image.
 * @param info Synchronized camera info.
 */
void DefectMapPipelineNode::onSynchronizedFrame(
  const sensor_msgs::msg::Image::ConstSharedPtr & rgb,
  const sensor_msgs::msg::Image::ConstSharedPtr & depth,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & info)
{
  bool captured_for_request = false;
  {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (capture_waiting_for_next_frame_ && !capture_frame_ready_ && rgb && depth && info) {
      latest_frame_.rgb = rgb;
      latest_frame_.depth = depth;
      latest_frame_.info = info;
      latest_frame_.valid = true;
      capture_frame_ready_ = true;
      capture_waiting_for_next_frame_ = false;
      captured_for_request = true;
    }
  }

  if (captured_for_request) {
    frame_cv_.notify_one();
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Captured requested sync frame: rgb=%ux%u enc=%s depth=%ux%u enc=%s cam_info=%ux%u stamp=%u.%u",
      rgb->width, rgb->height, rgb->encoding.c_str(),
      depth->width, depth->height, depth->encoding.c_str(),
      info->width, info->height,
      rgb->header.stamp.sec, rgb->header.stamp.nanosec);
  }
}

/**
 * @brief Insert one shot into per-image buffer and emit a ready job when complete.
 * @return Detailed append/enqueue result for service response handling.
 */
DefectMapPipelineNode::AppendShotResult DefectMapPipelineNode::appendShotAndMaybeBuildJob(
  const ShotData & shot)
{
  AppendShotResult result;
  CaptureJob ready_job;

  std::lock_guard<std::mutex> shot_lock(shot_buffer_mutex_);
  // Default-construct shot map for this image_id if not present.
  auto & img_shots = shot_buffer_[shot.image_id];

  if (img_shots.find(shot.shot_id) != img_shots.end()) {
    result.status_code = "DUPLICATE_SHOT";
    result.status_message = "Shot already captured for this image_id";
    return result;
  }

  const bool will_complete =
    (static_cast<int>(img_shots.size()) + 1) >= expected_shots_per_image_;

  if (will_complete) {
    // Reserve queue capacity and enqueue atomically with completed-shot assembly,
    // so a completed job is never erased from the shot buffer without being queued.
    std::lock_guard<std::mutex> queue_lock(queue_mutex_);
    if (static_cast<int>(job_queue_.size()) >= max_queue_size_) {
      result.status_code = "QUEUE_FULL";
      result.status_message = "Queue full while assembling completed shot set";
      return result;
    }

    // Insert the final shot and build deterministic ordered job.
    img_shots[shot.shot_id] = shot;
    ready_job.image_id = shot.image_id;
    for (const auto & kv : img_shots) {
      ready_job.shots.push_back(kv.second);
    }

    job_queue_.push(std::move(ready_job));
    const auto queue_depth = static_cast<uint32_t>(job_queue_.size());
    queue_cv_.notify_one();
    shot_buffer_.erase(shot.image_id);
    result.accepted = true;
    result.job_ready = true;

    if (static_cast<int>(queue_depth) >= max_queue_size_) {
      result.status_code = "ACCEPTED_QUEUE_SATURATED";
      result.status_message = "Shot accepted and job enqueued; queue is now saturated";
    } else {
      result.status_code = "ACCEPTED";
      result.status_message = "Shot accepted";
    }
    return result;
  }

  // Non-completing shot: keep buffering.
  img_shots[shot.shot_id] = shot;
  result.accepted = true;
  result.status_code = "ACCEPTED";
  result.status_message = "Shot accepted";
  return result;
}

/**
 * @brief Block until work is available or shutdown is requested.
 * @return True when a job is popped, false on shutdown.
 */
bool DefectMapPipelineNode::dequeueJob(CaptureJob & job)
{
  std::unique_lock<std::mutex> lock(queue_mutex_);
  queue_cv_.wait(lock, [this]() { return stop_worker_ || !job_queue_.empty(); });

  if (stop_worker_) {
    return false;
  }

  job = std::move(job_queue_.front());
  job_queue_.pop();
  return true;
}

/**
 * @brief Capture service callback entrypoint.
 */
void DefectMapPipelineNode::onCaptureShot(
  const std::shared_ptr<defect_map_interfaces::srv::CaptureShot::Request> request,
  std::shared_ptr<defect_map_interfaces::srv::CaptureShot::Response> response)
{
  response->accepted = false;
  response->queue_depth = 0U;
  response->accepted_stamp = now();

  if (!tf_ready_ || tf_permanent_error_) {
    response->status_code = "TF_UNAVAILABLE";
    response->message = "TF preflight not ready";
    return;
  }

  if (request->image_id.empty() || request->shot_id == 0U) {
    response->status_code = "INVALID_REQUEST";
    response->message = "image_id and shot_id must be valid";
    return;
  }

  if (static_cast<int>(request->shot_id) > expected_shots_per_image_) {
    response->status_code = "INVALID_REQUEST";
    response->message = "shot_id out of configured range";
    return;
  }

  FrameSnapshot snapshot;
  {
    std::unique_lock<std::mutex> frame_lock(frame_mutex_);
    if (capture_waiting_for_next_frame_) {
      response->status_code = "BUSY";
      response->message = "Another capture request is already waiting for next sync frame";
      return;
    }

    capture_waiting_for_next_frame_ = true;
    capture_frame_ready_ = false;
    latest_frame_ = FrameSnapshot{};

    const bool got_frame = frame_cv_.wait_for(
      frame_lock,
      std::chrono::milliseconds(capture_next_frame_timeout_ms_),
      [this]() { return capture_frame_ready_; });

    if (!got_frame || !latest_frame_.valid) {
      capture_waiting_for_next_frame_ = false;
      capture_frame_ready_ = false;
      response->status_code = "NO_FRAME";
      response->message = "Timed out waiting for next synchronized frame";
      return;
    }

    snapshot = latest_frame_;
    capture_frame_ready_ = false;
  }

  ShotData shot;
  shot.image_id = request->image_id;
  shot.shot_id = request->shot_id;
  shot.rgb = snapshot.rgb;
  shot.depth = snapshot.depth;
  shot.info = snapshot.info;
  if (shot.rgb) {
    response->accepted_stamp = shot.rgb->header.stamp;
  }

  const auto append_result = appendShotAndMaybeBuildJob(shot);
  {
    std::lock_guard<std::mutex> queue_lock(queue_mutex_);
    response->queue_depth = static_cast<uint32_t>(job_queue_.size());
  }

  if (!append_result.accepted) {
    response->status_code = append_result.status_code;
    response->message = append_result.status_message;
    return;
  }

  if (append_result.job_ready) {
    RCLCPP_INFO(
      get_logger(), "Enqueued complete shot set image_id=%s queue_depth=%u",
      request->image_id.c_str(), response->queue_depth);
  } else {
    RCLCPP_INFO(
      get_logger(), "Accepted shot image_id=%s shot_id=%u queue_depth=%u",
      request->image_id.c_str(), request->shot_id, response->queue_depth);
  }

  response->accepted = true;
  response->status_code = append_result.status_code;
  response->message = append_result.status_message;
}

/**
 * @brief Worker thread loop executing queued jobs.
 */
void DefectMapPipelineNode::workerLoop()
{
  // Workers run independently from ROS callbacks to keep capture latency low.
  while (rclcpp::ok()) {
    CaptureJob job;
    if (!dequeueJob(job)) {
      return;
    }
    processJob(job);
  }
}

/**
 * @brief Call external SegmentImage service with processed image.
 * @return True when a response object is received before timeout.
 */
bool DefectMapPipelineNode::callPrediction(
  const cv::Mat & processed,
  defect_map_interfaces::srv::SegmentImage::Response::SharedPtr & response_out)
{
  if (!prediction_client_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_ERROR(get_logger(), "Prediction service unavailable");
    return false;
  }

  auto request = std::make_shared<defect_map_interfaces::srv::SegmentImage::Request>();
  request->score_threshold_override = -1.0F;
  request->roi_rgb_processed =
    *cv_bridge::CvImage(std_msgs::msg::Header(), sensor_msgs::image_encodings::BGR8, processed).toImageMsg();

  auto future = prediction_client_->async_send_request(request);
  if (future.wait_for(std::chrono::milliseconds(prediction_timeout_ms_)) != std::future_status::ready) {
    return false;
  }

  response_out = future.get();
  return static_cast<bool>(response_out);
}

/**
 * @brief Project mask pixels with valid depth to base-frame 3D points.
 * @return True when at least one valid transformed point is produced.
 */
bool DefectMapPipelineNode::projectMaskToBasePoints(
  const cv::Mat & mask,
  const sensor_msgs::msg::Image::ConstSharedPtr & depth,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & info,
  const std::string & from_frame,
  const builtin_interfaces::msg::Time & stamp,
  std::vector<geometry_msgs::msg::Point> & points_base) const
{
  points_base.clear();
  if (!depth || !info || info->k[0] <= 0.0 || info->k[4] <= 0.0 || mask.empty()) {
    return false;
  }

  cv_bridge::CvImageConstPtr depth_cv;
  try {
    depth_cv = cv_bridge::toCvShare(depth);
  } catch (const std::exception &) {
    return false;
  }

  const cv::Mat depth_mat = depth_cv->image;
  if (mask.cols > depth_mat.cols || mask.rows > depth_mat.rows) {
    return false;
  }

  const int x0 = std::max(0, (depth_mat.cols - mask.cols) / 2);
  const int y0 = std::max(0, (depth_mat.rows - mask.rows) / 2);

  const double fx = info->k[0];
  const double fy = info->k[4];
  const double cx = info->k[2];
  const double cy = info->k[5];

  std::vector<geometry_msgs::msg::Point> points_cam;
  points_cam.reserve(static_cast<size_t>(mask.cols * mask.rows / 4));

  for (int v = 0; v < mask.rows; ++v) {
    const uint8_t * row_ptr = mask.ptr<uint8_t>(v);
    for (int u = 0; u < mask.cols; ++u) {
      if (row_ptr[u] == 0U) {
        continue;
      }

      const int du = x0 + u;
      const int dv = y0 + v;
      const double z = readDepthMeters(depth_mat, dv, du);
      if (!std::isfinite(z) || z <= 0.0) {
        continue;
      }

      geometry_msgs::msg::Point point;
      point.x = (static_cast<double>(du) - cx) / fx * z;
      point.y = (static_cast<double>(dv) - cy) / fy * z;
      point.z = z;
      points_cam.push_back(point);
    }
  }

  if (points_cam.empty()) {
    return false;
  }

  return transformPointsToBase(points_cam, from_frame, stamp, points_base);
}

/**
 * @brief Transform a batch of points to base_frame using one TF lookup.
 * @return True on successful transform.
 */
bool DefectMapPipelineNode::transformPointsToBase(
  const std::vector<geometry_msgs::msg::Point> & points,
  const std::string & from_frame,
  const builtin_interfaces::msg::Time & stamp,
  std::vector<geometry_msgs::msg::Point> & transformed_points) const
{
  transformed_points.clear();
  transformed_points.reserve(points.size());

  const std::string effective_from = from_frame.empty() ? camera_frame_ : from_frame;

  geometry_msgs::msg::TransformStamped tf_msg;
  try {
    // Single TF lookup for the full point batch keeps projection cost predictable.
    tf_msg = tf_buffer_->lookupTransform(
      base_frame_, effective_from, stamp,
      tf2::durationFromSec(static_cast<double>(tf_lookup_timeout_ms_) / 1000.0));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_ERROR(
      get_logger(), "TF transform lookup failed (%s -> %s): %s",
      effective_from.c_str(), base_frame_.c_str(), ex.what());
    return false;
  }

  for (const auto & p : points) {
    geometry_msgs::msg::PointStamped in_pt;
    in_pt.header.frame_id = effective_from;
    in_pt.header.stamp = stamp;
    in_pt.point = p;

    geometry_msgs::msg::PointStamped out_pt;
    tf2::doTransform(in_pt, out_pt, tf_msg);
    transformed_points.push_back(out_pt.point);
  }

  return !transformed_points.empty();
}

/**
 * @brief Generate next local outgoing UID for defect entries.
 * @return Monotonic UID value.
 */
uint64_t DefectMapPipelineNode::nextUid()
{
  return next_uid_.fetch_add(1U, std::memory_order_relaxed);
}

/**
 * @brief Convert base-frame points to deduplicated voxel index arrays.
 */
void DefectMapPipelineNode::pointsToVoxels(
  const std::vector<geometry_msgs::msg::Point> & points_base,
  std::vector<int32_t> & voxel_ix,
  std::vector<int32_t> & voxel_iy,
  std::vector<int32_t> & voxel_iz) const
{
  voxel_ix.clear();
  voxel_iy.clear();
  voxel_iz.clear();

  if (voxel_size_m_ <= 0.0) {
    return;
  }

  // Deduplicate voxels per defect before writing the contract payload.
  std::set<std::tuple<int32_t, int32_t, int32_t>> unique_voxels;
  for (const auto & p : points_base) {
    const auto ix = static_cast<int32_t>(std::floor(p.x / voxel_size_m_));
    const auto iy = static_cast<int32_t>(std::floor(p.y / voxel_size_m_));
    const auto iz = static_cast<int32_t>(std::floor(p.z / voxel_size_m_));
    unique_voxels.emplace(ix, iy, iz);
  }

  voxel_ix.reserve(unique_voxels.size());
  voxel_iy.reserve(unique_voxels.size());
  voxel_iz.reserve(unique_voxels.size());
  for (const auto & v : unique_voxels) {
    voxel_ix.push_back(std::get<0>(v));
    voxel_iy.push_back(std::get<1>(v));
    voxel_iz.push_back(std::get<2>(v));
  }
}

/**
 * @brief Send AddDefects request and apply UID resync feedback when needed.
 * @return True when map writer accepts the batch.
 */
bool DefectMapPipelineNode::sendDefectsToMap(
  const std::vector<defect_map_interfaces::msg::DefectEntry> & defects,
  std::string & status_code,
  std::string & status_message)
{
  if (defects.empty()) {
    status_code = "NO_DATA";
    status_message = "No defects to write";
    return true;
  }

  if (!add_defects_client_->wait_for_service(std::chrono::seconds(1))) {
    status_code = "SERVICE_UNAVAILABLE";
    status_message = "AddDefects service unavailable";
    return false;
  }

  auto request = std::make_shared<defect_map_interfaces::srv::AddDefects::Request>();
  request->defects = defects;

  auto future = add_defects_client_->async_send_request(request);
  if (future.wait_for(std::chrono::milliseconds(map_write_timeout_ms_)) != std::future_status::ready) {
    status_code = "TIMEOUT";
    status_message = "AddDefects response timeout";
    return false;
  }

  auto response = future.get();
  if (!response) {
    status_code = "NO_RESPONSE";
    status_message = "AddDefects response is null";
    return false;
  }

  status_code = response->status_code;
  status_message = response->message;

  if (!response->accepted) {
    if (response->status_code == "UID_OUT_OF_SYNC") {
      // Keep pipeline writer counter aligned with map-owner feedback.
      const uint64_t target = response->latest_uid + 1U;
      uint64_t current = next_uid_.load(std::memory_order_relaxed);
      while (current < target &&
        !next_uid_.compare_exchange_weak(current, target, std::memory_order_relaxed)) {}
    }
    return false;
  }

  const uint64_t target = response->latest_uid + 1U;
  uint64_t current = next_uid_.load(std::memory_order_relaxed);
  while (current < target &&
    !next_uid_.compare_exchange_weak(current, target, std::memory_order_relaxed)) {}

  return true;
}

/**
 * @brief Process one capture job from preprocessing to map-write request.
 * @param job Complete shot set for one image.
 */
void DefectMapPipelineNode::processJob(const CaptureJob & job)
{
  if (job.shots.empty()) {
    return;
  }

  RCLCPP_INFO(
    get_logger(), "Processing image_id=%s with %zu shots", job.image_id.c_str(), job.shots.size());

  std::vector<cv::Mat> shot_rgbs;
  shot_rgbs.reserve(job.shots.size());

  for (const auto & shot : job.shots) {
    try {
      auto cv_img = cv_bridge::toCvCopy(shot.rgb, sensor_msgs::image_encodings::BGR8);
      shot_rgbs.push_back(cropCenter(cv_img->image));
      RCLCPP_INFO(
        get_logger(), "Shot converted image_id=%s shot_id=%u rgb=%dx%d->crop=%dx%d",
        job.image_id.c_str(), shot.shot_id,
        cv_img->image.cols, cv_img->image.rows,
        shot_rgbs.back().cols, shot_rgbs.back().rows);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "RGB conversion failed for image_id=%s: %s", job.image_id.c_str(), e.what());
      return;
    }
  }

  if (shot_rgbs.empty()) {
    RCLCPP_ERROR(get_logger(), "No valid RGB shots for image_id=%s", job.image_id.c_str());
    return;
  }

  const cv::Mat processed = preprocessShots(shot_rgbs);
  if (processed.empty()) {
    RCLCPP_ERROR(get_logger(), "Preprocessing returned empty image for image_id=%s", job.image_id.c_str());
    return;
  }

  std_msgs::msg::Header processed_header;
  if (job.shots.front().rgb) {
    processed_header = job.shots.front().rgb->header;
  } else {
    processed_header.stamp = now();
    processed_header.frame_id = camera_frame_;
  }

  preprocessed_image_pub_->publish(
    *cv_bridge::CvImage(processed_header, sensor_msgs::image_encodings::BGR8, processed).toImageMsg());

  RCLCPP_INFO(
    get_logger(),
    "Published preprocessed image image_id=%s size=%dx%d subs=%zu topic=~/debug/preprocessed_image",
    job.image_id.c_str(), processed.cols, processed.rows,
    preprocessed_image_pub_->get_subscription_count());

  defect_map_interfaces::srv::SegmentImage::Response::SharedPtr prediction_response;
  if (!callPrediction(processed, prediction_response)) {
    RCLCPP_ERROR(get_logger(), "Prediction request failed for image_id=%s", job.image_id.c_str());
    return;
  }

  if (!prediction_response->success) {
    RCLCPP_ERROR(
      get_logger(), "Prediction failed image_id=%s [%s]: %s",
      job.image_id.c_str(), prediction_response->status_code.c_str(), prediction_response->message.c_str());
    return;
  }

  const auto & depth_ref = job.shots.front().depth;
  const auto & info_ref = job.shots.front().info;
  const auto depth_frame = (depth_ref && !depth_ref->header.frame_id.empty()) ?
    depth_ref->header.frame_id : camera_frame_;
  const auto capture_stamp = depth_ref ? depth_ref->header.stamp : processed_header.stamp;

  std::vector<defect_map_interfaces::msg::DefectEntry> defects;
  defects.reserve(prediction_response->instances.size());

  std::vector<geometry_msgs::msg::Point> debug_points;

  for (const auto & inst : prediction_response->instances) {
    cv::Mat mask;
    try {
      mask = cv_bridge::toCvCopy(inst.mask, sensor_msgs::image_encodings::MONO8)->image;
    } catch (const std::exception & e) {
      RCLCPP_WARN(get_logger(), "Mask conversion failed for image_id=%s: %s", job.image_id.c_str(), e.what());
      continue;
    }

    std::vector<geometry_msgs::msg::Point> points_base;
    if (!projectMaskToBasePoints(mask, depth_ref, info_ref, depth_frame, capture_stamp, points_base)) {
      continue;
    }

    std::vector<int32_t> voxel_ix;
    std::vector<int32_t> voxel_iy;
    std::vector<int32_t> voxel_iz;
    pointsToVoxels(points_base, voxel_ix, voxel_iy, voxel_iz);
    if (voxel_ix.empty()) {
      continue;
    }

    defect_map_interfaces::msg::DefectEntry entry;
    entry.uid = nextUid();
    entry.cluster = false;
    entry.zone_id = job.image_id;
    entry.label = inst.label;
    entry.score = inst.score;
    entry.voxel_ix = std::move(voxel_ix);
    entry.voxel_iy = std::move(voxel_iy);
    entry.voxel_iz = std::move(voxel_iz);
    defects.push_back(std::move(entry));

    debug_points.insert(debug_points.end(), points_base.begin(), points_base.end());
  }

  if (defects.empty()) {
    RCLCPP_WARN(get_logger(), "No valid defects generated for image_id=%s", job.image_id.c_str());
    return;
  }

  std::string status_code;
  std::string status_message;
  if (!sendDefectsToMap(defects, status_code, status_message)) {
    RCLCPP_ERROR(
      get_logger(), "AddDefects failed image_id=%s [%s]: %s",
      job.image_id.c_str(), status_code.c_str(), status_message.c_str());
  } else {
    RCLCPP_INFO(
      get_logger(), "AddDefects accepted image_id=%s defects=%zu",
      job.image_id.c_str(), defects.size());
  }

  const auto cloud = makeCloudFromPoints(debug_points);
  raw_cloud_pub_->publish(cloud);
  // Clustered debug topic is kept for compatibility until the map node is introduced.
  clustered_cloud_pub_->publish(cloud);
}

/**
 * @brief Crop a centered ROI using configured dimensions.
 */
cv::Mat DefectMapPipelineNode::cropCenter(const cv::Mat & input) const
{
  const int w = std::min(crop_width_, input.cols);
  const int h = std::min(crop_height_, input.rows);
  const int x = std::max(0, (input.cols - w) / 2);
  const int y = std::max(0, (input.rows - h) / 2);
  return input(cv::Rect(x, y, w, h)).clone();
}

/**
 * @brief Dispatch preprocessing mode.
 */
cv::Mat DefectMapPipelineNode::preprocessShots(const std::vector<cv::Mat> & shot_rgbs) const
{
  if (shot_rgbs.empty()) {
    return {};
  }

  if (preprocess_mode_ == "NONE") {
    return shot_rgbs.front().clone();
  }

  if (preprocess_mode_ == "NORMAL") {
    return preprocessNormal(shot_rgbs);
  }

  if (preprocess_mode_ == "CURVATURE") {
    return preprocessCurvature(shot_rgbs);
  }

  return preprocessComposite(shot_rgbs);
}

/**
 * @brief Build normal-map encoding from 4 directional shots.
 */
cv::Mat DefectMapPipelineNode::preprocessNormal(const std::vector<cv::Mat> & shot_rgbs) const
{
  if (shot_rgbs.size() < 4) {
    return shot_rgbs.front().clone();
  }

  std::vector<cv::Mat> gray(4);
  for (size_t i = 0; i < 4; ++i) {
    cv::cvtColor(shot_rgbs[i], gray[i], cv::COLOR_BGR2GRAY);
    gray[i].convertTo(gray[i], CV_32F, 1.0 / 255.0);
  }

  cv::Mat nx = gray[0] - gray[2];
  cv::Mat ny = gray[3] - gray[1];
  cv::Mat nz = cv::Mat::ones(nx.size(), CV_32F);

  cv::Mat norm;
  cv::magnitude(nx, ny, norm);
  const cv::Mat norm_sq = norm.mul(norm) + nz.mul(nz);
  cv::sqrt(norm_sq, norm);
  norm += 1e-6;

  nx = nx / norm;
  ny = ny / norm;
  nz = nz / norm;

  cv::Mat r = (nx + 1.0) * 127.5;
  cv::Mat g = (ny + 1.0) * 127.5;
  cv::Mat b = (nz + 1.0) * 127.5;

  cv::Mat r8, g8, b8;
  r.convertTo(r8, CV_8UC1);
  g.convertTo(g8, CV_8UC1);
  b.convertTo(b8, CV_8UC1);

  cv::Mat out;
  cv::merge(std::vector<cv::Mat>{b8, g8, r8}, out);
  return out;
}

/**
 * @brief Build curvature/height/albedo composite encoding.
 */
cv::Mat DefectMapPipelineNode::preprocessComposite(const std::vector<cv::Mat> & shot_rgbs) const
{
  if (shot_rgbs.size() < 4) {
    return shot_rgbs.front().clone();
  }

  std::vector<cv::Mat> gray(4);
  for (size_t i = 0; i < 4; ++i) {
    cv::cvtColor(shot_rgbs[i], gray[i], cv::COLOR_BGR2GRAY);
    gray[i].convertTo(gray[i], CV_32F, 1.0 / 255.0);
  }

  const cv::Mat albedo = (gray[0] + gray[1] + gray[2] + gray[3]) * 0.25;

  const cv::Mat nx = gray[0] - gray[2];
  const cv::Mat ny = gray[3] - gray[1];
  cv::Mat curv;
  cv::Laplacian(nx + ny, curv, CV_32F, 3);
  const cv::Mat curv_n = localNormalize(curv, ps_curv_sigma_);

  cv::Mat gx, gy;
  cv::Sobel(nx, gx, CV_32F, 1, 0, 3);
  cv::Sobel(ny, gy, CV_32F, 0, 1, 3);
  const cv::Mat height = gx + gy;
  const cv::Mat height_n = localNormalize(height, ps_height_sigma_);

  const cv::Mat r = clipMapToU8(curv_n, ps_encode_clip_);
  const cv::Mat g = clipMapToU8(height_n, ps_encode_clip_);
  cv::Mat b;
  cv::normalize(albedo, b, 0, 255, cv::NORM_MINMAX);
  b.convertTo(b, CV_8UC1);

  cv::Mat out;
  cv::merge(std::vector<cv::Mat>{b, g, r}, out);
  return out;
}

/**
 * @brief Build multi-scale curvature encoding.
 */
cv::Mat DefectMapPipelineNode::preprocessCurvature(const std::vector<cv::Mat> & shot_rgbs) const
{
  if (shot_rgbs.size() < 4) {
    return shot_rgbs.front().clone();
  }

  std::vector<cv::Mat> gray(4);
  for (size_t i = 0; i < 4; ++i) {
    cv::cvtColor(shot_rgbs[i], gray[i], cv::COLOR_BGR2GRAY);
    gray[i].convertTo(gray[i], CV_32F, 1.0 / 255.0);
  }

  cv::Mat base;
  cv::Laplacian((gray[0] + gray[1] + gray[2] + gray[3]) * 0.25, base, CV_32F, 3);

  const double s1 = std::max(3.0, ps_curv_sigma_ / 12.0);
  const double s2 = std::max(8.0, ps_curv_sigma_ / 3.0);
  const double s3 = std::max(15.0, ps_curv_sigma_);

  const cv::Mat c1 = clipMapToU8(localNormalize(base, s1), ps_encode_clip_);
  const cv::Mat c2 = clipMapToU8(localNormalize(base, s2), ps_encode_clip_);
  const cv::Mat c3 = clipMapToU8(localNormalize(base, s3), ps_encode_clip_);

  cv::Mat out;
  cv::merge(std::vector<cv::Mat>{c3, c2, c1}, out);
  return out;
}

/**
 * @brief Build PointCloud2 debug message from point list.
 */
sensor_msgs::msg::PointCloud2 DefectMapPipelineNode::makeCloudFromPoints(
  const std::vector<geometry_msgs::msg::Point> & points) const
{
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.frame_id = base_frame_;
  cloud.header.stamp = now();
  cloud.height = 1;
  cloud.width = static_cast<uint32_t>(points.size());
  cloud.is_dense = false;
  cloud.is_bigendian = false;

  sensor_msgs::PointCloud2Modifier modifier(cloud);
  modifier.setPointCloud2FieldsByString(1, "xyz");
  modifier.resize(points.size());

  sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");

  for (const auto & p : points) {
    *iter_x = static_cast<float>(p.x);
    *iter_y = static_cast<float>(p.y);
    *iter_z = static_cast<float>(p.z);
    ++iter_x;
    ++iter_y;
    ++iter_z;
  }

  return cloud;
}

/**
 * @brief Convert string to uppercase.
 */
std::string DefectMapPipelineNode::toUpper(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return value;
}

}  // namespace defect_map_pipeline

RCLCPP_COMPONENTS_REGISTER_NODE(defect_map_pipeline::DefectMapPipelineNode)
