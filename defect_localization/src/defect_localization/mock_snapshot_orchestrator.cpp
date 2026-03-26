/**
 * @file mock_snapshot_orchestrator.cpp
 * @brief Drive mock snapshot capture over the folder-based mock image catalog.
 */
#include "defect_localization/mock_image_catalog.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/parameter_client.hpp>
#include <rclcpp/rclcpp.hpp>

#include "defect_map_interfaces/srv/capture_shot.hpp"

namespace defect_localization
{
namespace
{

class MockSnapshotOrchestrator : public rclcpp::Node
{
public:
  MockSnapshotOrchestrator()
  : Node("mock_snapshot_orchestrator")
  {
    declare_parameter<std::string>("images_root", defaultMockImagesRoot().string());
    declare_parameter<int>("expected_shots_per_image", 4);
    declare_parameter<int>("max_images", 2);
    declare_parameter<std::string>("mock_camera_node_name", "/mock_realsense");
    declare_parameter<std::string>("capture_service_name", "/defect_localization/capture_shot");
    declare_parameter<int>("publish_settle_ms", 500);
    declare_parameter<int>("image_id_delay_ms", 2000);
    declare_parameter<int>("request_timeout_ms", 3000);
    declare_parameter<int>("service_wait_timeout_ms", 5000);
    declare_parameter<int>("max_retries_per_shot", 3);

    const auto images_root = std::filesystem::path(get_parameter("images_root").as_string());
    const auto expected_shots = static_cast<int>(get_parameter("expected_shots_per_image").as_int());
    max_images_ = static_cast<int>(get_parameter("max_images").as_int());
    publish_settle_ms_ = static_cast<int>(get_parameter("publish_settle_ms").as_int());
    image_id_delay_ms_ = static_cast<int>(get_parameter("image_id_delay_ms").as_int());
    request_timeout_ms_ = static_cast<int>(get_parameter("request_timeout_ms").as_int());
    service_wait_timeout_ms_ =
      static_cast<int>(get_parameter("service_wait_timeout_ms").as_int());
    max_retries_per_shot_ = static_cast<int>(get_parameter("max_retries_per_shot").as_int());

    const auto catalog = loadMockCatalog(images_root, expected_shots);
    if (!catalog) {
      throw std::runtime_error(catalog.error());
    }
    catalog_ = *catalog;

    mock_camera_node_name_ = get_parameter("mock_camera_node_name").as_string();
    capture_service_name_ = get_parameter("capture_service_name").as_string();
    parameter_client_ = std::make_shared<rclcpp::AsyncParametersClient>(this, mock_camera_node_name_);
    capture_client_ = create_client<defect_map_interfaces::srv::CaptureShot>(capture_service_name_);

    RCLCPP_INFO(
      get_logger(),
      "Mock snapshot orchestrator ready. sets=%zu mock_camera=%s capture_service=%s",
      catalog_.size(),
      mock_camera_node_name_.c_str(),
      capture_service_name_.c_str());
  }

  int run()
  {
    if (!waitForDependencies()) {
      return 1;
    }

    const auto image_count = max_images_ > 0 ?
      std::min<size_t>(catalog_.size(), static_cast<size_t>(max_images_)) :
      catalog_.size();
    if (max_images_ > 0 && static_cast<size_t>(max_images_) > catalog_.size()) {
      RCLCPP_WARN(
        get_logger(),
        "Requested max_images=%d but only %zu image sets are available",
        max_images_,
        catalog_.size());
    }

    for (size_t image_index = 0; image_index < image_count; ++image_index) {
      const auto & image_set = catalog_[image_index];
      for (const auto & shot : image_set.shots) {
        bool captured = false;
        for (int attempt = 1; attempt <= max_retries_per_shot_; ++attempt) {
          if (!selectShot(image_set.image_id, shot.shot_id)) {
            return 1;
          }

          rclcpp::sleep_for(std::chrono::milliseconds(publish_settle_ms_));

          if (captureShot(image_set.image_id, shot.shot_id)) {
            captured = true;
            break;
          }

          RCLCPP_WARN(
            get_logger(),
            "Retrying image_id=%s shot_id=%u (attempt %d/%d)",
            image_set.image_id.c_str(),
            shot.shot_id,
            attempt + 1,
            max_retries_per_shot_);
        }

        if (!captured) {
          RCLCPP_ERROR(
            get_logger(),
            "Failed to capture image_id=%s shot_id=%u after %d attempts",
            image_set.image_id.c_str(),
            shot.shot_id,
            max_retries_per_shot_);
          return 1;
        }
      }

      if (image_id_delay_ms_ > 0 && image_index + 1 < image_count) {
        RCLCPP_INFO(
          get_logger(),
          "Waiting %d ms before next image_id after %s",
          image_id_delay_ms_,
          image_set.image_id.c_str());
        rclcpp::sleep_for(std::chrono::milliseconds(image_id_delay_ms_));
      }
    }

    RCLCPP_INFO(
      get_logger(),
      "Completed mock snapshot orchestration for %zu image sets",
      image_count);
    return 0;
  }

private:
  bool waitForDependencies()
  {
    const auto timeout = std::chrono::milliseconds(service_wait_timeout_ms_);
    if (!capture_client_->wait_for_service(timeout)) {
      RCLCPP_ERROR(
        get_logger(),
        "Capture service unavailable: %s",
        capture_service_name_.c_str());
      return false;
    }
    if (!parameter_client_->wait_for_service(timeout)) {
      RCLCPP_ERROR(
        get_logger(),
        "Mock camera parameter service unavailable: %s",
        mock_camera_node_name_.c_str());
      return false;
    }
    return true;
  }

  bool selectShot(const std::string & image_id, uint32_t shot_id)
  {
    std::vector<rclcpp::Parameter> parameters;
    parameters.emplace_back("active_image_id", image_id);
    parameters.emplace_back("active_shot_id", static_cast<int>(shot_id));

    auto future = parameter_client_->set_parameters(parameters);
    const auto result = rclcpp::spin_until_future_complete(
      get_node_base_interface(),
      future,
      std::chrono::milliseconds(request_timeout_ms_));
    if (result != rclcpp::FutureReturnCode::SUCCESS) {
      RCLCPP_ERROR(
        get_logger(),
        "Timed out while selecting image_id=%s shot_id=%u on mock camera",
        image_id.c_str(),
        shot_id);
      return false;
    }

    for (const auto & set_result : future.get()) {
      if (!set_result.successful) {
        RCLCPP_ERROR(
          get_logger(),
          "Mock camera rejected image_id=%s shot_id=%u: %s",
          image_id.c_str(),
          shot_id,
          set_result.reason.c_str());
        return false;
      }
    }
    return true;
  }

  bool captureShot(const std::string & image_id, uint32_t shot_id)
  {
    auto request = std::make_shared<defect_map_interfaces::srv::CaptureShot::Request>();
    request->image_id = image_id;
    request->shot_id = shot_id;

    auto future = capture_client_->async_send_request(request);
    const auto result = rclcpp::spin_until_future_complete(
      get_node_base_interface(),
      future,
      std::chrono::milliseconds(request_timeout_ms_));
    if (result != rclcpp::FutureReturnCode::SUCCESS) {
      RCLCPP_ERROR(
        get_logger(),
        "Timed out waiting for capture_shot response image_id=%s shot_id=%u",
        image_id.c_str(),
        shot_id);
      return false;
    }

    const auto response = future.get();
    if (!response || !response->accepted) {
      RCLCPP_WARN(
        get_logger(),
        "Capture rejected image_id=%s shot_id=%u [%s]: %s",
        image_id.c_str(),
        shot_id,
        response ? response->status_code.c_str() : "NO_RESPONSE",
        response ? response->message.c_str() : "Missing response");
      return false;
    }

    RCLCPP_INFO(
      get_logger(),
      "Captured image_id=%s shot_id=%u queue_depth=%u",
      image_id.c_str(),
      shot_id,
      response->queue_depth);
    return true;
  }

  int publish_settle_ms_{250};
  int image_id_delay_ms_{0};
  int request_timeout_ms_{3000};
  int service_wait_timeout_ms_{5000};
  int max_retries_per_shot_{3};
  int max_images_{0};

  MockCatalog catalog_;
  std::string mock_camera_node_name_;
  std::string capture_service_name_;

  std::shared_ptr<rclcpp::AsyncParametersClient> parameter_client_;
  rclcpp::Client<defect_map_interfaces::srv::CaptureShot>::SharedPtr capture_client_;
};

}  // namespace
}  // namespace defect_localization

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<defect_localization::MockSnapshotOrchestrator>();
  const int exit_code = node->run();

  rclcpp::shutdown();
  return exit_code;
}
