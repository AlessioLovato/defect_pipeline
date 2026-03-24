#ifndef DEFECT_MAP_PIPELINE__DEFECT_MAP_PIPELINE_NODE_HPP_
#define DEFECT_MAP_PIPELINE__DEFECT_MAP_PIPELINE_NODE_HPP_

#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
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
#include "defect_map_interfaces/msg/segmented_instance.hpp"
#include "defect_map_interfaces/srv/build_map.hpp"
#include "defect_map_interfaces/srv/capture_shot.hpp"
#include "defect_map_interfaces/srv/clear_defect_map.hpp"
#include "defect_map_interfaces/srv/get_defect_map.hpp"
#include "defect_map_interfaces/srv/segment_image.hpp"

namespace defect_map_pipeline
{

class DefectMapPipelineNode : public rclcpp::Node
{
public:
  explicit DefectMapPipelineNode(const rclcpp::NodeOptions & options);
  ~DefectMapPipelineNode() override;

private:
  using FrameSyncPolicy = message_filters::sync_policies::ExactTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::Image,
    sensor_msgs::msg::CameraInfo>;
  using FrameSynchronizer = message_filters::Synchronizer<FrameSyncPolicy>;

  struct FrameSnapshot
  {
    sensor_msgs::msg::Image::ConstSharedPtr rgb;
    sensor_msgs::msg::Image::ConstSharedPtr depth;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr info;
    bool valid{false};
  };

  struct ShotData
  {
    std::string image_id;
    uint32_t shot_id{0U};
    sensor_msgs::msg::Image::ConstSharedPtr rgb;
    sensor_msgs::msg::Image::ConstSharedPtr depth;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr info;
  };

  struct CaptureJob
  {
    std::string image_id;
    std::vector<ShotData> shots;
  };

  struct ClusterSettings
  {
    double voxel_size{0.01};
    double neighbor_distance{0.02};
    uint32_t min_cluster_points{5U};
  };

  // ROS callbacks
  void onSynchronizedFrame(
    const sensor_msgs::msg::Image::ConstSharedPtr & rgb,
    const sensor_msgs::msg::Image::ConstSharedPtr & depth,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & info);

  void onCaptureShot(
    const std::shared_ptr<defect_map_interfaces::srv::CaptureShot::Request> request,
    std::shared_ptr<defect_map_interfaces::srv::CaptureShot::Response> response);

  void onBuildMap(
    const std::shared_ptr<defect_map_interfaces::srv::BuildMap::Request> request,
    std::shared_ptr<defect_map_interfaces::srv::BuildMap::Response> response);

  void onGetDefectMap(
    const std::shared_ptr<defect_map_interfaces::srv::GetDefectMap::Request> request,
    std::shared_ptr<defect_map_interfaces::srv::GetDefectMap::Response> response);

  void onClearDefectMap(
    const std::shared_ptr<defect_map_interfaces::srv::ClearDefectMap::Request> request,
    std::shared_ptr<defect_map_interfaces::srv::ClearDefectMap::Response> response);

  // Worker + processing
  void workerLoop();
  void processJob(const CaptureJob & job);

  // TF preflight
  void tfPreflightTick();

  // Helpers
  cv::Mat cropCenter(const cv::Mat & input) const;
  cv::Mat preprocessShots(const std::vector<cv::Mat> & shot_rgbs) const;
  cv::Mat preprocessNormal(const std::vector<cv::Mat> & shot_rgbs) const;
  cv::Mat preprocessComposite(const std::vector<cv::Mat> & shot_rgbs) const;
  cv::Mat preprocessCurvature(const std::vector<cv::Mat> & shot_rgbs) const;

  geometry_msgs::msg::Point projectMaskCentroid(
    const cv::Mat & mask,
    const sensor_msgs::msg::Image::ConstSharedPtr & depth,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & info,
    uint32_t & support_points,
    std::vector<geometry_msgs::msg::Point> & support_points_xyz,
    std::vector<int32_t> & voxel_ix,
    std::vector<int32_t> & voxel_iy,
    std::vector<int32_t> & voxel_iz,
    bool & ok) const;

  geometry_msgs::msg::Point transformPointToBase(
    const geometry_msgs::msg::Point & p,
    const std::string & from_frame,
    bool & ok) const;

  sensor_msgs::msg::PointCloud2 makeCloud(
    const std::vector<defect_map_interfaces::msg::DefectEntry> & entries,
    bool use_support_points) const;

  std::vector<defect_map_interfaces::msg::DefectEntry> buildClusteredMap(
    const std::vector<defect_map_interfaces::msg::DefectEntry> & raw,
    const ClusterSettings & settings) const;

  bool isAdjacent(
    const defect_map_interfaces::msg::DefectEntry & a,
    const defect_map_interfaces::msg::DefectEntry & b,
    const ClusterSettings & settings) const;

  static std::string toUpper(std::string value);

  // Parameters
  std::string rgb_topic_;
  std::string depth_topic_;
  std::string camera_info_topic_;
  std::string prediction_service_name_;
  std::string base_frame_;
  std::string camera_frame_;
  std::string preprocess_mode_;
  std::vector<std::string> light_order_;

  int crop_width_{512};
  int crop_height_{512};
  int expected_shots_per_image_{4};
  int max_queue_size_{16};
  int worker_threads_{1};
  int prediction_timeout_ms_{5000};
  int tf_lookup_timeout_ms_{2000};
  int sync_queue_size_{30};
  bool tf_preflight_enabled_{true};

  double ps_curv_sigma_{60.0};
  double ps_height_sigma_{40.0};
  double ps_encode_clip_{4.0};

  bool cluster_default_enabled_{true};
  double cluster_voxel_size_m_{0.01};
  double cluster_neighbor_distance_m_{0.02};
  int cluster_min_points_{5};

  // Subscribers/pubs/srvs
  message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_;
  std::shared_ptr<FrameSynchronizer> frame_sync_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr raw_cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr clustered_cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr preprocessed_image_pub_;

  rclcpp::Service<defect_map_interfaces::srv::CaptureShot>::SharedPtr capture_service_;
  rclcpp::Service<defect_map_interfaces::srv::BuildMap>::SharedPtr build_map_service_;
  rclcpp::Service<defect_map_interfaces::srv::GetDefectMap>::SharedPtr get_map_service_;
  rclcpp::Service<defect_map_interfaces::srv::ClearDefectMap>::SharedPtr clear_map_service_;

  rclcpp::Client<defect_map_interfaces::srv::SegmentImage>::SharedPtr prediction_client_;

  // TF
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::TimerBase::SharedPtr tf_preflight_timer_;
  rclcpp::Time tf_preflight_start_;
  bool tf_ready_{false};
  bool tf_permanent_error_{false};

  // Shared state
  mutable std::mutex frame_mutex_;
  FrameSnapshot latest_frame_;

  mutable std::mutex shot_buffer_mutex_;
  std::unordered_map<std::string, std::map<uint32_t, ShotData>> shot_buffer_;

  mutable std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::queue<CaptureJob> job_queue_;
  bool stop_worker_{false};
  std::vector<std::thread> workers_;

  mutable std::mutex entries_mutex_;
  std::vector<defect_map_interfaces::msg::DefectEntry> raw_entries_;
  std::vector<defect_map_interfaces::msg::DefectEntry> latest_map_raw_;
  std::vector<defect_map_interfaces::msg::DefectEntry> latest_map_clustered_;
};

}  // namespace defect_map_pipeline

#endif  // DEFECT_MAP_PIPELINE__DEFECT_MAP_PIPELINE_NODE_HPP_
