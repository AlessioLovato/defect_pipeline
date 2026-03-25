/**
 * @file defect_localization_node.hpp
 * @brief Declaration of the modular defect_localization ROS 2 node.
 * @author Alessio Lovato
 */
#ifndef DEFECT_LOCALIZATION__DEFECT_LOCALIZATION_NODE_HPP_
#define DEFECT_LOCALIZATION__DEFECT_LOCALIZATION_NODE_HPP_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <message_filters/subscriber.hpp>
#include <message_filters/sync_policies/exact_time.hpp>
#include <message_filters/synchronizer.hpp>
#include <opencv2/core.hpp>

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "defect_map_interfaces/msg/defect_entry.hpp"
#include "defect_map_interfaces/srv/add_defects.hpp"
#include "defect_map_interfaces/srv/capture_shot.hpp"
#include "defect_map_interfaces/srv/segment_image.hpp"

namespace defect_localization
{

/**
 * @brief ROS 2 pipeline node that orchestrates capture, preprocessing, prediction, and map writing.
 *
 * Expected runtime behavior:
 * - Subscribes to RGB, depth, and camera info and keeps an exact-time synchronized frame cache.
 * - Exposes @c ~/capture_shot to collect multi-shot sets by image_id and queue processing jobs.
 * - Runs worker threads that preprocess shot bundles and call external @c SegmentImage prediction.
 * - Projects mask pixels to 3D, transforms points to @c base_frame at capture timestamp,
 *   voxelizes/deduplicates them, and sends @c DefectEntry batches to @c AddDefects.
 * - Does not own persistent map storage; authoritative map state lives in the map node.
 */
class DefectLocalizationNode : public rclcpp::Node
{
public:
  /**
   * @brief Construct and wire all runtime modules (sync, queue, workers, clients).
   * @param options ROS node options.
   */
  explicit DefectLocalizationNode(const rclcpp::NodeOptions & options);

  /**
   * @brief Stop worker threads and release resources.
   */
  ~DefectLocalizationNode() override;

private:
  using FrameSyncPolicy = message_filters::sync_policies::ExactTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::Image,
    sensor_msgs::msg::CameraInfo>;
  using FrameSynchronizer = message_filters::Synchronizer<FrameSyncPolicy>;

  /**
   * @brief One synchronized frame tuple used by request-driven capture.
   */
  struct FrameSnapshot
  {
    sensor_msgs::msg::Image::ConstSharedPtr rgb;
    sensor_msgs::msg::Image::ConstSharedPtr depth;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr info;
    bool valid{false};
  };

  /**
   * @brief Stored shot payload used to assemble complete jobs.
   */
  struct ShotData
  {
    std::string image_id;
    uint32_t shot_id{0U};
    sensor_msgs::msg::Image::ConstSharedPtr rgb;
    sensor_msgs::msg::Image::ConstSharedPtr depth;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr info;
  };

  /**
   * @brief Work item consumed by worker threads.
   */
  struct CaptureJob
  {
    std::string image_id;
    std::vector<ShotData> shots;
  };

  /**
   * @brief One debug cloud point with a preselected RGB color.
   */
  struct DebugCloudPoint
  {
    geometry_msgs::msg::Point point;
    uint8_t r{255U};
    uint8_t g{255U};
    uint8_t b{255U};
  };

  /**
   * @brief Result of shot-buffer insertion and optional job enqueue.
   */
  struct AppendShotResult
  {
    bool accepted{false};
    bool job_ready{false};
    std::string status_code;
    std::string status_message;
  };

  /**
   * @brief Frame sync callback that fulfills pending capture requests.
   * @param rgb Synchronized RGB image.
   * @param depth Synchronized depth image.
   * @param info Synchronized camera info.
   */
  void onSynchronizedFrame(
    const sensor_msgs::msg::Image::ConstSharedPtr & rgb,
    const sensor_msgs::msg::Image::ConstSharedPtr & depth,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & info);

  /**
   * @brief Capture service callback that validates and buffers a shot.
   * @param request CaptureShot request.
   * @param response CaptureShot response.
   */
  void onCaptureShot(
    const std::shared_ptr<defect_map_interfaces::srv::CaptureShot::Request> request,
    std::shared_ptr<defect_map_interfaces::srv::CaptureShot::Response> response);

  /**
   * @brief Worker loop: pop jobs and execute full processing pipeline.
   */
  void workerLoop();

  /**
   * @brief Execute one queued job end-to-end.
   * @param job Fully assembled shot set for one image_id.
   */
  void processJob(const CaptureJob & job);

  /**
   * @brief Periodic startup TF preflight check.
   */
  void tfPreflightTick();

  /**
   * @brief Insert one shot and atomically enqueue the complete set when ready.
   * @param shot Incoming shot payload.
   * @return Detailed append/enqueue result with acceptance and status fields.
   */
  AppendShotResult appendShotAndMaybeBuildJob(const ShotData & shot);

  /**
   * @brief Wait and pop next job from queue.
   * @param job Output dequeued job.
   * @return False when shutdown is requested, true otherwise.
   */
  bool dequeueJob(CaptureJob & job);

  /**
   * @brief Crop center ROI used as preprocessing input.
   * @param input Source image.
   * @return Cropped clone.
   */
  cv::Mat cropCenter(const cv::Mat & input) const;

  /**
   * @brief Select configured preprocessing mode.
   * @param shot_rgbs Ordered shot RGB images.
   * @return Preprocessed BGR image for prediction.
   */
  cv::Mat preprocessShots(const std::vector<cv::Mat> & shot_rgbs) const;

  /**
   * @brief Normal-map photometric stereo encoding.
   * @param shot_rgbs Ordered shot RGB images.
   * @return BGR normal-map encoded image.
   */
  cv::Mat preprocessNormal(const std::vector<cv::Mat> & shot_rgbs) const;

  /**
   * @brief Composite curvature/height/albedo encoding.
   * @param shot_rgbs Ordered shot RGB images.
   * @return Composite encoded image.
   */
  cv::Mat preprocessComposite(const std::vector<cv::Mat> & shot_rgbs) const;

  /**
   * @brief Multi-scale curvature encoding.
   * @param shot_rgbs Ordered shot RGB images.
   * @return Curvature encoded image.
   */
  cv::Mat preprocessCurvature(const std::vector<cv::Mat> & shot_rgbs) const;

  /**
   * @brief Invoke external SegmentImage service.
   * @param processed Preprocessed image.
   * @param response_out Prediction response output.
   * @return True when response was received, false on timeout/unavailability.
   */
  bool callPrediction(
    const cv::Mat & processed,
    defect_map_interfaces::srv::SegmentImage::Response::SharedPtr & response_out);

  /**
   * @brief Convert mask pixels to base-frame 3D points.
   * @param mask Mono8 instance mask.
   * @param depth Depth image aligned to RGB.
   * @param info Camera model.
   * @param from_frame Frame used by depth points.
   * @param stamp Capture timestamp used for TF lookup.
   * @param points_base Output transformed base-frame points.
   * @return True when at least one valid point was projected.
   */
  bool projectMaskToBasePoints(
    const cv::Mat & mask,
    const sensor_msgs::msg::Image::ConstSharedPtr & depth,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & info,
    const std::string & from_frame,
    const builtin_interfaces::msg::Time & stamp,
    std::vector<geometry_msgs::msg::Point> & points_base) const;

  /**
   * @brief Transform a batch of points to base_frame using one TF lookup.
   * @param points Input points in @p from_frame.
   * @param from_frame Source frame id.
   * @param stamp Timestamp for TF lookup.
   * @param transformed_points Output points in base_frame.
   * @return True on successful transform.
   */
  bool transformPointsToBase(
    const std::vector<geometry_msgs::msg::Point> & points,
    const std::string & from_frame,
    const builtin_interfaces::msg::Time & stamp,
    std::vector<geometry_msgs::msg::Point> & transformed_points) const;

  /**
   * @brief Generate next outgoing UID for defect writer messages.
   * @return Monotonic UID.
   */
  uint64_t nextUid();

  /**
   * @brief Voxelize and deduplicate base-frame points.
   * @param points_base Input base-frame points.
   * @param voxel_ix Output voxel x indices.
   * @param voxel_iy Output voxel y indices.
   * @param voxel_iz Output voxel z indices.
   */
  void pointsToVoxels(
    const std::vector<geometry_msgs::msg::Point> & points_base,
    std::vector<int32_t> & voxel_ix,
    std::vector<int32_t> & voxel_iy,
    std::vector<int32_t> & voxel_iz) const;

  /**
   * @brief Send defects to map node via AddDefects service.
   * @param defects Defect batch to write.
   * @param status_code Returned writer status code.
   * @param status_message Returned writer status text.
   * @return True when accepted by map node.
   */
  bool sendDefectsToMap(
    const std::vector<defect_map_interfaces::msg::DefectEntry> & defects,
    std::string & status_code,
    std::string & status_message);

  /**
   * @brief Build debug point cloud from base-frame points.
   * @param points Input points.
   * @return PointCloud2 message.
   */
  sensor_msgs::msg::PointCloud2 makeCloudFromPoints(
    const std::vector<DebugCloudPoint> & points) const;

  /**
   * @brief Uppercase helper for parameter normalization.
   * @param value Input string.
   * @return Uppercased string.
   */
  static std::string toUpper(std::string value);

  // Parameters.
  std::string rgb_topic_;
  std::string depth_topic_;
  std::string camera_info_topic_;
  std::string prediction_service_name_;
  std::string map_add_defects_service_name_;
  std::string base_frame_;
  std::string camera_frame_;
  std::string preprocess_mode_;
  std::vector<std::string> light_order_;

  int crop_width_{512};
  int crop_height_{512};
  int crop_offset_x_{0};
  int crop_offset_y_{0};
  int expected_shots_per_image_{4};
  int max_queue_size_{16};
  int worker_threads_{1};
  int prediction_timeout_ms_{5000};
  int map_write_timeout_ms_{2000};
  int tf_lookup_timeout_ms_{2000};
  int capture_next_frame_timeout_ms_{1000};
  int sync_queue_size_{30};
  bool tf_preflight_enabled_{true};

  double voxel_size_m_{0.01};
  double ps_curv_sigma_{60.0};
  double ps_height_sigma_{40.0};
  double ps_encode_clip_{4.0};

  // Subscribers/pubs/srvs/clients.
  message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_;
  std::shared_ptr<FrameSynchronizer> frame_sync_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr raw_cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr preprocessed_image_pub_;

  rclcpp::Service<defect_map_interfaces::srv::CaptureShot>::SharedPtr capture_service_;
  rclcpp::CallbackGroup::SharedPtr capture_service_callback_group_;

  rclcpp::Client<defect_map_interfaces::srv::SegmentImage>::SharedPtr prediction_client_;
  rclcpp::Client<defect_map_interfaces::srv::AddDefects>::SharedPtr add_defects_client_;

  // TF.
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::TimerBase::SharedPtr tf_preflight_timer_;
  rclcpp::Time tf_preflight_start_;
  bool tf_ready_{false};
  bool tf_permanent_error_{false};

  // Frame synchronization module state.
  mutable std::mutex frame_mutex_;
  std::condition_variable frame_cv_;
  bool capture_waiting_for_next_frame_{false};
  bool capture_frame_ready_{false};
  FrameSnapshot latest_frame_;

  // Shot buffering module state.
  mutable std::mutex shot_buffer_mutex_;
  std::unordered_map<std::string, std::map<uint32_t, ShotData>> shot_buffer_;

  // Queue/worker orchestration module state.
  mutable std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::queue<CaptureJob> job_queue_;
  bool stop_worker_{false};
  std::vector<std::thread> workers_;

  // Map-writer UID state.
  std::atomic<uint64_t> next_uid_{1U};
};

}  // namespace defect_localization

#endif  // DEFECT_LOCALIZATION__DEFECT_LOCALIZATION_NODE_HPP_
