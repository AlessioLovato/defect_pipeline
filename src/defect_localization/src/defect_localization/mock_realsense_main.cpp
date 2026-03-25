/**
 * @file mock_realsense_main.cpp
 * @brief Mock RealSense publisher for snapshot-driven pipeline tests.
 */
#include "defect_localization/mock_image_catalog.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2_ros/transform_broadcaster.hpp>

namespace defect_localization
{
namespace
{

struct LoadedShot
{
  cv::Mat rgb;
  sensor_msgs::msg::CameraInfo camera_info;
};

using LoadedShotMap = std::map<std::string, std::map<uint32_t, LoadedShot>>;

class MockRealsenseNode : public rclcpp::Node
{
public:
  MockRealsenseNode()
  : Node("mock_realsense")
  {
    declare_parameter<std::string>("images_root", defaultMockImagesRoot().string());
    declare_parameter<int>("expected_shots_per_image", 4);
    declare_parameter<std::string>("active_image_id", "");
    declare_parameter<int>("active_shot_id", 1);
    declare_parameter<double>("publish_rate_hz", 5.0);
    declare_parameter<double>("depth_meters", 1.0);
    declare_parameter<double>("days_between_image_ids", 0.0);
    declare_parameter<double>("meters_left_per_image_id", 0.26);
    declare_parameter<double>("camera_fx", 0.0);
    declare_parameter<double>("camera_fy", 0.0);
    declare_parameter<std::string>("base_frame", "world");
    declare_parameter<std::string>("frame_id", "camera_color_optical_frame");
    declare_parameter<std::string>("rgb_topic", "/camera/camera/color/image_raw");
    declare_parameter<std::string>(
      "depth_topic", "/camera/camera/aligned_depth_to_color/image_raw");
    declare_parameter<std::string>("camera_info_topic", "/camera/camera/color/camera_info");

    const auto images_root = std::filesystem::path(get_parameter("images_root").as_string());
    const auto expected_shots = static_cast<int>(get_parameter("expected_shots_per_image").as_int());
    publish_rate_hz_ = get_parameter("publish_rate_hz").as_double();
    depth_meters_ = get_parameter("depth_meters").as_double();
    days_between_image_ids_ = get_parameter("days_between_image_ids").as_double();
    meters_left_per_image_id_ = get_parameter("meters_left_per_image_id").as_double();
    camera_fx_ = get_parameter("camera_fx").as_double();
    camera_fy_ = get_parameter("camera_fy").as_double();
    base_frame_ = get_parameter("base_frame").as_string();
    frame_id_ = get_parameter("frame_id").as_string();

    if (publish_rate_hz_ <= 0.0) {
      throw std::invalid_argument("Parameter publish_rate_hz must be > 0");
    }
    if (depth_meters_ <= 0.0) {
      throw std::invalid_argument("Parameter depth_meters must be > 0");
    }
    if (days_between_image_ids_ < 0.0) {
      throw std::invalid_argument("Parameter days_between_image_ids must be >= 0");
    }
    if (meters_left_per_image_id_ < 0.0) {
      throw std::invalid_argument("Parameter meters_left_per_image_id must be >= 0");
    }

    const auto catalog = loadMockCatalog(images_root, expected_shots);
    if (!catalog) {
      throw std::runtime_error(catalog.error());
    }
    assignImageIndices(*catalog);
    loadImages(*catalog);
    mock_time_origin_ = now();

    active_image_id_ = get_parameter("active_image_id").as_string();
    if (active_image_id_.empty()) {
      active_image_id_ = loaded_shots_.begin()->first;
    }
    active_shot_id_ = static_cast<uint32_t>(std::max<int64_t>(
      1, get_parameter("active_shot_id").as_int()));
    validateSelection(active_image_id_, active_shot_id_);

    const auto sensor_qos = rclcpp::SensorDataQoS();
    rgb_pub_ = create_publisher<sensor_msgs::msg::Image>(
      get_parameter("rgb_topic").as_string(), sensor_qos);
    depth_pub_ = create_publisher<sensor_msgs::msg::Image>(
      get_parameter("depth_topic").as_string(), sensor_qos);
    camera_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
      get_parameter("camera_info_topic").as_string(), sensor_qos);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    parameter_callback_handle_ = add_on_set_parameters_callback(
      std::bind(&MockRealsenseNode::onParametersSet, this, std::placeholders::_1));

    const auto period = std::chrono::duration<double>(1.0 / publish_rate_hz_);
    publish_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::milliseconds>(period),
      std::bind(&MockRealsenseNode::publishFrame, this));

    RCLCPP_INFO(
      get_logger(),
      "Mock RealSense ready. images_root=%s active_image_id=%s active_shot_id=%u",
      images_root.string().c_str(),
      active_image_id_.c_str(),
      active_shot_id_);
  }

private:
  void assignImageIndices(const MockCatalog & catalog)
  {
    for (size_t index = 0; index < catalog.size(); ++index) {
      image_index_by_id_.emplace(catalog[index].image_id, index);
    }
  }

  void loadImages(const MockCatalog & catalog)
  {
    for (const auto & image_set : catalog) {
      auto & shots = loaded_shots_[image_set.image_id];
      for (const auto & shot_spec : image_set.shots) {
        const auto rgb = cv::imread(shot_spec.file_path.string(), cv::IMREAD_COLOR);
        if (rgb.empty()) {
          throw std::runtime_error("Failed to load mock RGB image: " + shot_spec.file_path.string());
        }

        shots.emplace(
          shot_spec.shot_id,
          LoadedShot{
            .rgb = rgb,
            .camera_info = buildCameraInfoTemplate(rgb.cols, rgb.rows)});
      }
    }
  }

  sensor_msgs::msg::CameraInfo buildCameraInfoTemplate(int width, int height) const
  {
    const double fx = camera_fx_ > 0.0 ? camera_fx_ : static_cast<double>(std::max(width, height));
    const double fy = camera_fy_ > 0.0 ? camera_fy_ : static_cast<double>(std::max(width, height));
    const double cx = (static_cast<double>(width) - 1.0) * 0.5;
    const double cy = (static_cast<double>(height) - 1.0) * 0.5;

    sensor_msgs::msg::CameraInfo info;
    info.width = static_cast<uint32_t>(width);
    info.height = static_cast<uint32_t>(height);
    info.distortion_model = "plumb_bob";
    info.d = {0.0, 0.0, 0.0, 0.0, 0.0};
    info.k = {fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0};
    info.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    info.p = {fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0};
    return info;
  }

  void validateSelection(const std::string & image_id, uint32_t shot_id) const
  {
    const auto image_it = loaded_shots_.find(image_id);
    if (image_it == loaded_shots_.end()) {
      throw std::runtime_error("Unknown mock image_id: " + image_id);
    }
    if (image_it->second.count(shot_id) == 0U) {
      throw std::runtime_error(
        "Unknown shot_id " + std::to_string(shot_id) + " for image_id " + image_id);
    }
  }

  const LoadedShot & activeShot() const
  {
    const auto image_it = loaded_shots_.find(active_image_id_);
    return image_it->second.at(active_shot_id_);
  }

  rcl_interfaces::msg::SetParametersResult onParametersSet(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    std::lock_guard<std::mutex> lock(selection_mutex_);

    auto next_image_id = active_image_id_;
    auto next_shot_id = active_shot_id_;

    for (const auto & parameter : parameters) {
      if (parameter.get_name() == "active_image_id") {
        next_image_id = parameter.as_string();
      } else if (parameter.get_name() == "active_shot_id") {
        next_shot_id = static_cast<uint32_t>(std::max<int64_t>(1, parameter.as_int()));
      }
    }

    rcl_interfaces::msg::SetParametersResult result;
    const auto image_it = loaded_shots_.find(next_image_id);
    if (image_it == loaded_shots_.end()) {
      result.successful = false;
      result.reason = "Unknown mock image_id: " + next_image_id;
      return result;
    }
    if (image_it->second.count(next_shot_id) == 0U) {
      result.successful = false;
      result.reason =
        "Unknown shot_id " + std::to_string(next_shot_id) + " for image_id " + next_image_id;
      return result;
    }

    if (next_image_id != active_image_id_ || next_shot_id != active_shot_id_) {
      RCLCPP_INFO(
        get_logger(),
        "Switching mock frame to image_id=%s shot_id=%u",
        next_image_id.c_str(),
        next_shot_id);
    }
    active_image_id_ = std::move(next_image_id);
    active_shot_id_ = next_shot_id;
    result.successful = true;
    return result;
  }

  void publishFrame()
  {
    std::lock_guard<std::mutex> lock(selection_mutex_);
    const auto & shot = activeShot();

    const auto image_index_it = image_index_by_id_.find(active_image_id_);
    const auto image_index =
      image_index_it != image_index_by_id_.end() ? image_index_it->second : size_t{0U};
    const auto image_offset_seconds = static_cast<int64_t>(
      days_between_image_ids_ * 24.0 * 60.0 * 60.0 * static_cast<double>(image_index));
    const auto publish_offset_seconds = static_cast<double>(publish_sequence_) / publish_rate_hz_;
    const auto stamp =
      mock_time_origin_ +
      rclcpp::Duration::from_seconds(
        static_cast<double>(image_offset_seconds) + publish_offset_seconds);
    ++publish_sequence_;

    geometry_msgs::msg::TransformStamped camera_tf;
    camera_tf.header.stamp = stamp;
    camera_tf.header.frame_id = base_frame_;
    camera_tf.child_frame_id = frame_id_;
    // Shift each new image_id 26 cm to the left. With the current frame convention
    // we model "left" as negative X in the parent frame.
    camera_tf.transform.translation.x =
      -meters_left_per_image_id_ * static_cast<double>(image_index);
    camera_tf.transform.translation.y = 0.0;
    camera_tf.transform.translation.z = 0.0;
    camera_tf.transform.rotation.x = 0.0;
    camera_tf.transform.rotation.y = 0.0;
    camera_tf.transform.rotation.z = 0.0;
    camera_tf.transform.rotation.w = 1.0;

    std_msgs::msg::Header header;
    header.stamp = stamp;
    header.frame_id = frame_id_;

    auto rgb_msg =
      cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, shot.rgb).toImageMsg();

    cv::Mat depth_image(
      shot.rgb.rows,
      shot.rgb.cols,
      CV_32FC1,
      cv::Scalar(depth_meters_));
    auto depth_msg =
      cv_bridge::CvImage(header, sensor_msgs::image_encodings::TYPE_32FC1, depth_image)
      .toImageMsg();

    auto camera_info = shot.camera_info;
    camera_info.header = header;

    tf_broadcaster_->sendTransform(camera_tf);
    rgb_pub_->publish(*rgb_msg);
    depth_pub_->publish(*depth_msg);
    camera_info_pub_->publish(camera_info);
  }

  double publish_rate_hz_{5.0};
  double depth_meters_{1.0};
  double days_between_image_ids_{0.0};
  double meters_left_per_image_id_{0.26};
  double camera_fx_{0.0};
  double camera_fy_{0.0};
  std::string base_frame_;
  std::string frame_id_;
  rclcpp::Time mock_time_origin_{0, 0, RCL_ROS_TIME};

  LoadedShotMap loaded_shots_;
  std::map<std::string, size_t> image_index_by_id_;
  uint64_t publish_sequence_{0U};

  mutable std::mutex selection_mutex_;
  std::string active_image_id_;
  uint32_t active_shot_id_{1U};

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;
};

}  // namespace
}  // namespace defect_localization

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<defect_localization::MockRealsenseNode>();
  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
