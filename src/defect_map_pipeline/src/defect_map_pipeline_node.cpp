#include "defect_map_pipeline/defect_map_pipeline_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <limits>
#include <set>
#include <sstream>
#include <utility>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2/exceptions.h>
#include <tf2/time.h>

#include <opencv2/imgproc.hpp>

#include "rclcpp_components/register_node_macro.hpp"

namespace defect_map_pipeline
{
namespace
{

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

DefectMapPipelineNode::DefectMapPipelineNode(const rclcpp::NodeOptions & options)
: Node("defect_map_pipeline", options)
{
  declare_parameter<std::string>("rgb_topic", "/camera/camera/color/image_raw");
  declare_parameter<std::string>("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw");
  declare_parameter<std::string>("camera_info_topic", "/camera/camera/color/camera_info");
  declare_parameter<std::string>("base_frame", "world");
  declare_parameter<std::string>("camera_frame", "camera_color_optical_frame");

  declare_parameter<int>("crop_width", 512);
  declare_parameter<int>("crop_height", 512);
  declare_parameter<int>("expected_shots_per_image", 4);
  declare_parameter<int>("max_queue_size", 16);
  declare_parameter<int>("worker_threads", 1);
  declare_parameter<int>("inference_timeout_ms", 5000);
  declare_parameter<int>("tf_lookup_timeout_ms", 2000);

  declare_parameter<bool>("cluster_default_enabled", true);
  declare_parameter<double>("cluster_voxel_size_m", 0.01);
  declare_parameter<double>("cluster_neighbor_distance_m", 0.02);
  declare_parameter<int>("cluster_min_points", 5);

  declare_parameter<std::string>("preprocess_mode", "composite");
  declare_parameter<std::vector<std::string>>(
    "light_order", std::vector<std::string>{"left", "bottom", "right", "top"});
  declare_parameter<double>("ps_curv_sigma", 60.0);
  declare_parameter<double>("ps_height_sigma", 40.0);
  declare_parameter<double>("ps_encode_clip", 4.0);

  rgb_topic_ = get_parameter("rgb_topic").as_string();
  depth_topic_ = get_parameter("depth_topic").as_string();
  camera_info_topic_ = get_parameter("camera_info_topic").as_string();
  base_frame_ = get_parameter("base_frame").as_string();
  camera_frame_ = get_parameter("camera_frame").as_string();

  crop_width_ = get_parameter("crop_width").as_int();
  crop_height_ = get_parameter("crop_height").as_int();
  expected_shots_per_image_ = get_parameter("expected_shots_per_image").as_int();
  max_queue_size_ = get_parameter("max_queue_size").as_int();
  worker_threads_ = static_cast<int>(std::max<int64_t>(1, get_parameter("worker_threads").as_int()));
  inference_timeout_ms_ = get_parameter("inference_timeout_ms").as_int();
  tf_lookup_timeout_ms_ = get_parameter("tf_lookup_timeout_ms").as_int();

  cluster_default_enabled_ = get_parameter("cluster_default_enabled").as_bool();
  cluster_voxel_size_m_ = get_parameter("cluster_voxel_size_m").as_double();
  cluster_neighbor_distance_m_ = get_parameter("cluster_neighbor_distance_m").as_double();
  cluster_min_points_ = get_parameter("cluster_min_points").as_int();

  preprocess_mode_ = toUpper(get_parameter("preprocess_mode").as_string());
  light_order_ = get_parameter("light_order").as_string_array();
  if (light_order_.size() < static_cast<size_t>(expected_shots_per_image_)) {
    RCLCPP_WARN(get_logger(), "light_order has fewer entries than expected_shots_per_image; using defaults");
    light_order_ = {"left", "bottom", "right", "top"};
  }

  ps_curv_sigma_ = get_parameter("ps_curv_sigma").as_double();
  ps_height_sigma_ = get_parameter("ps_height_sigma").as_double();
  ps_encode_clip_ = get_parameter("ps_encode_clip").as_double();

  rgb_sub_ = create_subscription<sensor_msgs::msg::Image>(
    rgb_topic_, rclcpp::SensorDataQoS(),
    std::bind(&DefectMapPipelineNode::onRgb, this, std::placeholders::_1));
  depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
    depth_topic_, rclcpp::SensorDataQoS(),
    std::bind(&DefectMapPipelineNode::onDepth, this, std::placeholders::_1));
  camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
    camera_info_topic_, rclcpp::SensorDataQoS(),
    std::bind(&DefectMapPipelineNode::onCameraInfo, this, std::placeholders::_1));

  raw_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    "~/debug/raw_defects_cloud", rclcpp::SensorDataQoS());
  clustered_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    "~/debug/clustered_defects_cloud", rclcpp::SensorDataQoS());

  capture_service_ = create_service<defect_map_interfaces::srv::CaptureShot>(
    "~/capture_shot",
    std::bind(&DefectMapPipelineNode::onCaptureShot, this, std::placeholders::_1, std::placeholders::_2));
  build_map_service_ = create_service<defect_map_interfaces::srv::BuildMap>(
    "~/build_map",
    std::bind(&DefectMapPipelineNode::onBuildMap, this, std::placeholders::_1, std::placeholders::_2));
  get_map_service_ = create_service<defect_map_interfaces::srv::GetDefectMap>(
    "~/get_defect_map",
    std::bind(&DefectMapPipelineNode::onGetDefectMap, this, std::placeholders::_1, std::placeholders::_2));

  inference_client_ = create_client<defect_map_interfaces::srv::SegmentImage>(
    "/defect_map_inference/segment_image");

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  tf_preflight_start_ = now();
  tf_preflight_timer_ = create_wall_timer(
    std::chrono::milliseconds(200),
    std::bind(&DefectMapPipelineNode::tfPreflightTick, this));

  for (int i = 0; i < worker_threads_; ++i) {
    workers_.emplace_back(std::bind(&DefectMapPipelineNode::workerLoop, this));
  }

  RCLCPP_INFO(
    get_logger(), "Pipeline started. preprocess_mode=%s expected_shots=%d max_queue=%d",
    preprocess_mode_.c_str(), expected_shots_per_image_, max_queue_size_);
}

DefectMapPipelineNode::~DefectMapPipelineNode()
{
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
    // keep trying until timeout
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

void DefectMapPipelineNode::onRgb(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(frame_mutex_);
  latest_frame_.rgb = msg;
  latest_frame_.valid = latest_frame_.rgb && latest_frame_.depth && latest_frame_.info;
}

void DefectMapPipelineNode::onDepth(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(frame_mutex_);
  latest_frame_.depth = msg;
  latest_frame_.valid = latest_frame_.rgb && latest_frame_.depth && latest_frame_.info;
}

void DefectMapPipelineNode::onCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(frame_mutex_);
  latest_frame_.info = msg;
  latest_frame_.valid = latest_frame_.rgb && latest_frame_.depth && latest_frame_.info;
}

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
    std::lock_guard<std::mutex> lock(frame_mutex_);
    snapshot = latest_frame_;
  }

  if (!snapshot.valid) {
    response->status_code = "NO_FRAME";
    response->message = "No synchronized frame snapshot available";
    return;
  }

  ShotData shot;
  shot.image_id = request->image_id;
  shot.shot_id = request->shot_id;
  shot.rgb = snapshot.rgb;
  shot.depth = snapshot.depth;
  shot.info = snapshot.info;

  bool enqueue_ready_job = false;
  CaptureJob new_job;

  {
    std::lock_guard<std::mutex> lock(shot_buffer_mutex_);
    auto & img_shots = shot_buffer_[request->image_id];
    if (img_shots.find(request->shot_id) != img_shots.end()) {
      response->status_code = "DUPLICATE_SHOT";
      response->message = "Shot already captured for this image_id";
      return;
    }

    const bool will_complete =
      (static_cast<int>(img_shots.size()) + 1) >= expected_shots_per_image_;

    if (will_complete) {
      std::lock_guard<std::mutex> qlock(queue_mutex_);
      if (static_cast<int>(job_queue_.size()) >= max_queue_size_) {
        response->status_code = "QUEUE_FULL";
        response->message = "Queue full while assembling completed shot set";
        response->queue_depth = static_cast<uint32_t>(job_queue_.size());
        return;
      }
    }

    img_shots[request->shot_id] = shot;

    if (static_cast<int>(img_shots.size()) >= expected_shots_per_image_) {
      new_job.image_id = request->image_id;
      for (const auto & kv : img_shots) {
        new_job.shots.push_back(kv.second);
      }
      shot_buffer_.erase(request->image_id);
      enqueue_ready_job = true;
    }
  }

  if (enqueue_ready_job) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (static_cast<int>(job_queue_.size()) >= max_queue_size_) {
      response->status_code = "QUEUE_FULL";
      response->message = "Queue full";
      response->queue_depth = static_cast<uint32_t>(job_queue_.size());
      return;
    }
    job_queue_.push(std::move(new_job));
    response->queue_depth = static_cast<uint32_t>(job_queue_.size());
    queue_cv_.notify_one();
  } else {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    response->queue_depth = static_cast<uint32_t>(job_queue_.size());
  }

  response->accepted = true;
  response->status_code = "ACCEPTED";
  response->message = "Shot accepted";
}

void DefectMapPipelineNode::onBuildMap(
  const std::shared_ptr<defect_map_interfaces::srv::BuildMap::Request> request,
  std::shared_ptr<defect_map_interfaces::srv::BuildMap::Response> response)
{
  if (!tf_ready_ || tf_permanent_error_) {
    response->success = false;
    response->status_code = "TF_UNAVAILABLE";
    response->message = "TF preflight not ready";
    response->raw_entry_count = 0U;
    response->map_entry_count = 0U;
    response->clustering_applied = false;
    return;
  }

  std::vector<defect_map_interfaces::msg::DefectEntry> raw;
  {
    std::lock_guard<std::mutex> lock(entries_mutex_);
    raw = raw_entries_;
  }

  if (raw.empty()) {
    response->success = false;
    response->status_code = "NO_DATA";
    response->message = "No raw entries available";
    response->raw_entry_count = 0U;
    response->map_entry_count = 0U;
    response->clustering_applied = false;
    return;
  }

  ClusterSettings settings;
  settings.voxel_size = request->voxel_size_m > 0.0F ? request->voxel_size_m : cluster_voxel_size_m_;
  settings.neighbor_distance =
    request->neighbor_distance_m > 0.0F ? request->neighbor_distance_m : cluster_neighbor_distance_m_;
  settings.min_cluster_points =
    request->min_cluster_points > 0U ? request->min_cluster_points : static_cast<uint32_t>(cluster_min_points_);

  bool apply_clustering = request->enable_clustering;
  if (!request->enable_clustering && cluster_default_enabled_) {
    apply_clustering = false;
  }

  std::vector<defect_map_interfaces::msg::DefectEntry> clustered =
    apply_clustering ? buildClusteredMap(raw, settings) : raw;

  {
    std::lock_guard<std::mutex> lock(entries_mutex_);
    latest_map_raw_ = raw;
    latest_map_clustered_ = clustered;
  }

  raw_cloud_pub_->publish(makeCloud(raw));
  clustered_cloud_pub_->publish(makeCloud(clustered));

  response->success = true;
  response->status_code = "OK";
  response->message = apply_clustering ? "Map built with clustering" : "Map built without clustering";
  response->raw_entry_count = static_cast<uint32_t>(raw.size());
  response->map_entry_count = static_cast<uint32_t>(clustered.size());
  response->clustering_applied = apply_clustering;
}

void DefectMapPipelineNode::onGetDefectMap(
  const std::shared_ptr<defect_map_interfaces::srv::GetDefectMap::Request> request,
  std::shared_ptr<defect_map_interfaces::srv::GetDefectMap::Response> response)
{
  std::vector<defect_map_interfaces::msg::DefectEntry> current;
  {
    std::lock_guard<std::mutex> lock(entries_mutex_);
    current = request->clustered_view ? latest_map_clustered_ : latest_map_raw_;
  }

  if (current.empty()) {
    response->success = false;
    response->status_code = "NO_MAP";
    response->message = "No map available. Call build_map first.";
    return;
  }

  if (!request->label_filter.empty()) {
    std::vector<defect_map_interfaces::msg::DefectEntry> filtered;
    filtered.reserve(current.size());
    for (const auto & entry : current) {
      if (entry.label == request->label_filter) {
        filtered.push_back(entry);
      }
    }
    current = std::move(filtered);
  }

  response->entries = std::move(current);
  response->success = true;
  response->status_code = "OK";
  response->message = "Map returned";
}

void DefectMapPipelineNode::workerLoop()
{
  while (rclcpp::ok()) {
    CaptureJob job;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock, [this]() { return stop_worker_ || !job_queue_.empty(); });
      if (stop_worker_) {
        return;
      }
      job = std::move(job_queue_.front());
      job_queue_.pop();
    }

    processJob(job);
  }
}

void DefectMapPipelineNode::processJob(const CaptureJob & job)
{
  if (job.shots.empty()) {
    return;
  }

  std::vector<cv::Mat> shot_rgbs;
  std::vector<uint32_t> shot_ids;
  shot_rgbs.reserve(job.shots.size());
  shot_ids.reserve(job.shots.size());

  for (const auto & shot : job.shots) {
    try {
      auto cv_img = cv_bridge::toCvCopy(shot.rgb, sensor_msgs::image_encodings::BGR8);
      shot_rgbs.push_back(cropCenter(cv_img->image));
      shot_ids.push_back(shot.shot_id);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "RGB conversion failed for image_id=%s: %s", job.image_id.c_str(), e.what());
      return;
    }
  }

  if (shot_rgbs.empty()) {
    return;
  }

  cv::Mat processed = preprocessShots(shot_rgbs);
  if (processed.empty()) {
    RCLCPP_ERROR(get_logger(), "Preprocessing returned empty image for image_id=%s", job.image_id.c_str());
    return;
  }

  if (!inference_client_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_ERROR(get_logger(), "Inference service unavailable");
    return;
  }

  auto request = std::make_shared<defect_map_interfaces::srv::SegmentImage::Request>();
  request->image_id = job.image_id;
  request->shot_ids = shot_ids;
  request->score_threshold_override = -1.0F;
  request->roi_rgb_processed =
    *cv_bridge::CvImage(std_msgs::msg::Header(), sensor_msgs::image_encodings::BGR8, processed).toImageMsg();

  auto future = inference_client_->async_send_request(request);
  if (future.wait_for(std::chrono::milliseconds(inference_timeout_ms_)) != std::future_status::ready) {
    RCLCPP_ERROR(get_logger(), "Inference timeout for image_id=%s", job.image_id.c_str());
    return;
  }

  auto response = future.get();
  if (!response || !response->success) {
    const auto code = response ? response->status_code : std::string("NO_RESPONSE");
    const auto msg = response ? response->message : std::string("inference response is null");
    RCLCPP_ERROR(get_logger(), "Inference failed [%s]: %s", code.c_str(), msg.c_str());
    return;
  }

  const auto & depth_ref = job.shots.front().depth;
  const auto & info_ref = job.shots.front().info;
  const std::string depth_frame = depth_ref ? depth_ref->header.frame_id : camera_frame_;

  std::vector<defect_map_interfaces::msg::DefectEntry> new_entries;
  new_entries.reserve(response->instances.size());

  for (const auto & inst : response->instances) {
    cv::Mat mask;
    try {
      mask = cv_bridge::toCvCopy(inst.mask, sensor_msgs::image_encodings::MONO8)->image;
    } catch (const std::exception & e) {
      RCLCPP_WARN(get_logger(), "Mask conversion failed for image_id=%s: %s", job.image_id.c_str(), e.what());
      continue;
    }

    uint32_t support_points = 0U;
    std::vector<geometry_msgs::msg::Point> support_points_xyz;
    std::vector<int32_t> voxel_ix;
    std::vector<int32_t> voxel_iy;
    std::vector<int32_t> voxel_iz;
    bool ok = false;
    auto centroid_cam = projectMaskCentroid(
      mask, depth_ref, info_ref, support_points,
      support_points_xyz, voxel_ix, voxel_iy, voxel_iz, ok);
    if (!ok) {
      continue;
    }

    auto centroid_base = transformPointToBase(centroid_cam, depth_frame, ok);
    if (!ok) {
      continue;
    }

    defect_map_interfaces::msg::DefectEntry entry;
    entry.image_id = job.image_id;
    entry.shot_id = job.shots.front().shot_id;
    entry.instance_id = inst.instance_id;
    entry.label = inst.label;
    entry.class_id = inst.class_id;
    entry.score = inst.score;
    entry.centroid = centroid_base;
    entry.support_points = support_points;
    entry.support_points_xyz = support_points_xyz;
    entry.voxel_ix = voxel_ix;
    entry.voxel_iy = voxel_iy;
    entry.voxel_iz = voxel_iz;
    entry.cluster_id = 0U;
    new_entries.push_back(entry);
  }

  if (!new_entries.empty()) {
    std::lock_guard<std::mutex> lock(entries_mutex_);
    raw_entries_.insert(raw_entries_.end(), new_entries.begin(), new_entries.end());
  }
}

cv::Mat DefectMapPipelineNode::cropCenter(const cv::Mat & input) const
{
  const int w = std::min(crop_width_, input.cols);
  const int h = std::min(crop_height_, input.rows);
  const int x = std::max(0, (input.cols - w) / 2);
  const int y = std::max(0, (input.rows - h) / 2);
  return input(cv::Rect(x, y, w, h)).clone();
}

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

  cv::Mat nx = gray[0] - gray[2];      // left-right
  cv::Mat ny = gray[3] - gray[1];      // top-bottom
  cv::Mat nz = cv::Mat::ones(nx.size(), CV_32F);

  cv::Mat norm;
  cv::magnitude(nx, ny, norm);
  cv::Mat norm_sq = norm.mul(norm) + nz.mul(nz);
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

  cv::Mat albedo = (gray[0] + gray[1] + gray[2] + gray[3]) * 0.25;

  cv::Mat nx = gray[0] - gray[2];
  cv::Mat ny = gray[3] - gray[1];
  cv::Mat curv;
  cv::Laplacian(nx + ny, curv, CV_32F, 3);
  cv::Mat curv_n = localNormalize(curv, ps_curv_sigma_);

  cv::Mat gx, gy;
  cv::Sobel(nx, gx, CV_32F, 1, 0, 3);
  cv::Sobel(ny, gy, CV_32F, 0, 1, 3);
  cv::Mat height = gx + gy;
  cv::Mat height_n = localNormalize(height, ps_height_sigma_);

  cv::Mat r = clipMapToU8(curv_n, ps_encode_clip_);
  cv::Mat g = clipMapToU8(height_n, ps_encode_clip_);
  cv::Mat b;
  cv::normalize(albedo, b, 0, 255, cv::NORM_MINMAX);
  b.convertTo(b, CV_8UC1);

  cv::Mat out;
  cv::merge(std::vector<cv::Mat>{b, g, r}, out);
  return out;
}

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

  cv::Mat c1 = clipMapToU8(localNormalize(base, s1), ps_encode_clip_);
  cv::Mat c2 = clipMapToU8(localNormalize(base, s2), ps_encode_clip_);
  cv::Mat c3 = clipMapToU8(localNormalize(base, s3), ps_encode_clip_);

  cv::Mat out;
  cv::merge(std::vector<cv::Mat>{c3, c2, c1}, out);
  return out;
}

geometry_msgs::msg::Point DefectMapPipelineNode::projectMaskCentroid(
  const cv::Mat & mask,
  const sensor_msgs::msg::Image::ConstSharedPtr & depth,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & info,
  uint32_t & support_points,
  std::vector<geometry_msgs::msg::Point> & support_points_xyz,
  std::vector<int32_t> & voxel_ix,
  std::vector<int32_t> & voxel_iy,
  std::vector<int32_t> & voxel_iz,
  bool & ok) const
{
  geometry_msgs::msg::Point out;
  support_points = 0U;
  support_points_xyz.clear();
  voxel_ix.clear();
  voxel_iy.clear();
  voxel_iz.clear();
  ok = false;

  if (!depth || !info || info->k[0] <= 0.0 || info->k[4] <= 0.0) {
    return out;
  }

  cv_bridge::CvImageConstPtr depth_cv;
  try {
    depth_cv = cv_bridge::toCvShare(depth);
  } catch (const std::exception &) {
    return out;
  }

  const cv::Mat depth_mat = depth_cv->image;
  const int x0 = std::max(0, (depth_mat.cols - mask.cols) / 2);
  const int y0 = std::max(0, (depth_mat.rows - mask.rows) / 2);

  const double fx = info->k[0];
  const double fy = info->k[4];
  const double cx = info->k[2];
  const double cy = info->k[5];

  double sx = 0.0;
  double sy = 0.0;
  double sz = 0.0;

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
      const double x = (static_cast<double>(du) - cx) / fx * z;
      const double y = (static_cast<double>(dv) - cy) / fy * z;
      sx += x;
      sy += y;
      sz += z;
      geometry_msgs::msg::Point point;
      point.x = x;
      point.y = y;
      point.z = z;
      support_points_xyz.push_back(point);
      if (cluster_voxel_size_m_ > 0.0) {
        voxel_ix.push_back(static_cast<int32_t>(std::floor(x / cluster_voxel_size_m_)));
        voxel_iy.push_back(static_cast<int32_t>(std::floor(y / cluster_voxel_size_m_)));
        voxel_iz.push_back(static_cast<int32_t>(std::floor(z / cluster_voxel_size_m_)));
      } else {
        voxel_ix.push_back(0);
        voxel_iy.push_back(0);
        voxel_iz.push_back(0);
      }
      ++support_points;
    }
  }

  if (support_points == 0U) {
    return out;
  }

  out.x = sx / static_cast<double>(support_points);
  out.y = sy / static_cast<double>(support_points);
  out.z = sz / static_cast<double>(support_points);
  ok = true;
  return out;
}

geometry_msgs::msg::Point DefectMapPipelineNode::transformPointToBase(
  const geometry_msgs::msg::Point & p,
  const std::string & from_frame,
  bool & ok) const
{
  ok = false;
  geometry_msgs::msg::Point out;

  geometry_msgs::msg::PointStamped in_pt;
  in_pt.header.frame_id = from_frame;
  in_pt.header.stamp = now();
  in_pt.point = p;

  geometry_msgs::msg::PointStamped out_pt;
  try {
    tf_buffer_->transform(in_pt, out_pt, base_frame_, tf2::durationFromSec(0.2));
    out = out_pt.point;
    ok = true;
  } catch (const tf2::TransformException & ex) {
    RCLCPP_ERROR(get_logger(), "TF transform failed (%s -> %s): %s", from_frame.c_str(), base_frame_.c_str(), ex.what());
  }

  return out;
}

sensor_msgs::msg::PointCloud2 DefectMapPipelineNode::makeCloud(
  const std::vector<defect_map_interfaces::msg::DefectEntry> & entries) const
{
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.frame_id = base_frame_;
  cloud.header.stamp = now();
  cloud.height = 1;
  cloud.width = static_cast<uint32_t>(entries.size());
  cloud.is_dense = false;
  cloud.is_bigendian = false;

  sensor_msgs::PointCloud2Modifier modifier(cloud);
  modifier.setPointCloud2FieldsByString(1, "xyz");
  modifier.resize(entries.size());

  sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");

  for (const auto & entry : entries) {
    *iter_x = static_cast<float>(entry.centroid.x);
    *iter_y = static_cast<float>(entry.centroid.y);
    *iter_z = static_cast<float>(entry.centroid.z);
    ++iter_x;
    ++iter_y;
    ++iter_z;
  }

  return cloud;
}

bool DefectMapPipelineNode::isAdjacent(
  const defect_map_interfaces::msg::DefectEntry & a,
  const defect_map_interfaces::msg::DefectEntry & b,
  const ClusterSettings & settings) const
{
  const double dx = a.centroid.x - b.centroid.x;
  const double dy = a.centroid.y - b.centroid.y;
  const double dz = a.centroid.z - b.centroid.z;
  const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);

  if (distance <= settings.neighbor_distance) {
    return true;
  }

  if (settings.voxel_size > 0.0) {
    const auto ax = static_cast<int>(std::floor(a.centroid.x / settings.voxel_size));
    const auto ay = static_cast<int>(std::floor(a.centroid.y / settings.voxel_size));
    const auto az = static_cast<int>(std::floor(a.centroid.z / settings.voxel_size));
    const auto bx = static_cast<int>(std::floor(b.centroid.x / settings.voxel_size));
    const auto by = static_cast<int>(std::floor(b.centroid.y / settings.voxel_size));
    const auto bz = static_cast<int>(std::floor(b.centroid.z / settings.voxel_size));

    return std::abs(ax - bx) <= 1 && std::abs(ay - by) <= 1 && std::abs(az - bz) <= 1;
  }

  return false;
}

std::vector<defect_map_interfaces::msg::DefectEntry> DefectMapPipelineNode::buildClusteredMap(
  const std::vector<defect_map_interfaces::msg::DefectEntry> & raw,
  const ClusterSettings & settings) const
{
  std::vector<defect_map_interfaces::msg::DefectEntry> output;
  output.reserve(raw.size());

  std::unordered_map<std::string, std::vector<size_t>> by_label;
  for (size_t i = 0; i < raw.size(); ++i) {
    by_label[raw[i].label].push_back(i);
  }

  uint32_t cluster_id = 1U;

  for (const auto & kv : by_label) {
    const auto & indices = kv.second;
    std::vector<bool> visited(indices.size(), false);

    for (size_t i = 0; i < indices.size(); ++i) {
      if (visited[i]) {
        continue;
      }

      std::vector<size_t> component;
      std::queue<size_t> q;
      q.push(i);
      visited[i] = true;

      while (!q.empty()) {
        const size_t cur = q.front();
        q.pop();
        component.push_back(indices[cur]);

        for (size_t j = 0; j < indices.size(); ++j) {
          if (visited[j]) {
            continue;
          }
          if (isAdjacent(raw[indices[cur]], raw[indices[j]], settings)) {
            visited[j] = true;
            q.push(j);
          }
        }
      }

      if (component.empty()) {
        continue;
      }

      if (component.size() < settings.min_cluster_points) {
        for (const auto idx : component) {
          auto copy = raw[idx];
          copy.cluster_id = cluster_id++;
          output.push_back(copy);
        }
        continue;
      }

      defect_map_interfaces::msg::DefectEntry merged;
      merged.image_id = raw[component.front()].image_id;
      merged.shot_id = raw[component.front()].shot_id;
      merged.instance_id = raw[component.front()].instance_id;
      merged.label = raw[component.front()].label;
      merged.class_id = raw[component.front()].class_id;
      merged.cluster_id = cluster_id++;

      double sx = 0.0;
      double sy = 0.0;
      double sz = 0.0;
      double sw = 0.0;
      float best_score = 0.0F;
      uint32_t support = 0U;

      for (const auto idx : component) {
        const auto & e = raw[idx];
        const double w = std::max<uint32_t>(1U, e.support_points);
        sx += e.centroid.x * w;
        sy += e.centroid.y * w;
        sz += e.centroid.z * w;
        sw += w;
        best_score = std::max(best_score, e.score);
        support += e.support_points;
        merged.support_points_xyz.insert(
          merged.support_points_xyz.end(),
          e.support_points_xyz.begin(),
          e.support_points_xyz.end());
        merged.voxel_ix.insert(merged.voxel_ix.end(), e.voxel_ix.begin(), e.voxel_ix.end());
        merged.voxel_iy.insert(merged.voxel_iy.end(), e.voxel_iy.begin(), e.voxel_iy.end());
        merged.voxel_iz.insert(merged.voxel_iz.end(), e.voxel_iz.begin(), e.voxel_iz.end());
      }

      merged.centroid.x = sx / std::max(1.0, sw);
      merged.centroid.y = sy / std::max(1.0, sw);
      merged.centroid.z = sz / std::max(1.0, sw);
      merged.score = best_score;
      merged.support_points = support;
      output.push_back(merged);
    }
  }

  return output;
}

std::string DefectMapPipelineNode::toUpper(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return value;
}

}  // namespace defect_map_pipeline

RCLCPP_COMPONENTS_REGISTER_NODE(defect_map_pipeline::DefectMapPipelineNode)
