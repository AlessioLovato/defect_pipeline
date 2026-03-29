#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

class MockCameraInfoPublisher : public rclcpp::Node
{
public:
  MockCameraInfoPublisher()
  : Node("mock_camera_info_pub")
  {
    const auto topic = declare_parameter<std::string>("topic", "/camera/camera_info");
    const auto frame_id = declare_parameter<std::string>("frame_id", "camera_color_optical_frame");
    const auto width = declare_parameter<int>("width", 640);
    const auto height = declare_parameter<int>("height", 480);
    const auto fx = declare_parameter<double>("fx", 615.0);
    const auto fy = declare_parameter<double>("fy", 615.0);
    const auto cx = declare_parameter<double>("cx", static_cast<double>(width) / 2.0);
    const auto cy = declare_parameter<double>("cy", static_cast<double>(height) / 2.0);
    const auto rate_hz = declare_parameter<double>("rate_hz", 5.0);
    const auto distortion_model = declare_parameter<std::string>("distortion_model", "plumb_bob");
    const auto distortion_coeffs = declare_parameter<std::vector<double>>(
      "distortion_coeffs", std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0});

    publisher_ = create_publisher<sensor_msgs::msg::CameraInfo>(topic, 10);
    message_ = build_message(
      frame_id, width, height, fx, fy, cx, cy, distortion_model, distortion_coeffs);

    const auto period = std::chrono::duration<double>(1.0 / std::max(rate_hz, 1e-3));
    timer_ = create_wall_timer(period, [this]() { publish_once(); });

    RCLCPP_INFO(
      get_logger(),
      "Publishing mock CameraInfo on %s (%dx%d, fx=%.3f, fy=%.3f, cx=%.3f, cy=%.3f, frame_id=%s)",
      topic.c_str(), static_cast<int>(width), static_cast<int>(height), fx, fy, cx, cy,
      frame_id.c_str());
  }

private:
  sensor_msgs::msg::CameraInfo build_message(
    const std::string & frame_id,
    int width,
    int height,
    double fx,
    double fy,
    double cx,
    double cy,
    const std::string & distortion_model,
    const std::vector<double> & distortion_coeffs) const
  {
    sensor_msgs::msg::CameraInfo msg;
    msg.header.frame_id = frame_id;
    msg.width = static_cast<uint32_t>(std::max(width, 1));
    msg.height = static_cast<uint32_t>(std::max(height, 1));
    msg.distortion_model = distortion_model;
    msg.d = distortion_coeffs;

    // Publish a conventional pinhole camera model with identity rectification.
    msg.k = {
      fx, 0.0, cx,
      0.0, fy, cy,
      0.0, 0.0, 1.0
    };
    msg.r = {
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0
    };
    msg.p = {
      fx, 0.0, cx, 0.0,
      0.0, fy, cy, 0.0,
      0.0, 0.0, 1.0, 0.0
    };
    return msg;
  }

  void publish_once()
  {
    message_.header.stamp = now();
    publisher_->publish(message_);
  }

  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  sensor_msgs::msg::CameraInfo message_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MockCameraInfoPublisher>());
  rclcpp::shutdown();
  return 0;
}
