#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/pose_array.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/parameter_client.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "wall_patch_planner/srv/filter_wall_poses.hpp"

namespace
{

struct WallPlanCache
{
  geometry_msgs::msg::PoseArray poses;
  visualization_msgs::msg::MarkerArray markers;
};

class WallPatchFilterDemoNode : public rclcpp::Node
{
public:
  WallPatchFilterDemoNode()
  : Node("wall_patch_filter_demo")
  {
    declare_parameter<std::string>("planner_node_name", "/wall_patch_planner");
    declare_parameter<std::string>("planner_service_name", "/wall_patch_planner/plan_patches");
    declare_parameter<std::string>("planner_pose_topic", "/wall_patch_planner/poses");
    declare_parameter<std::string>("planner_marker_topic", "/wall_patch_planner/debug_markers");
    declare_parameter<std::string>("filtered_pose_topic", "/wall_patch_planner/filter_demo/poses");
    declare_parameter<std::string>("filtered_marker_topic", "/wall_patch_planner/filter_demo/markers");
    declare_parameter<int>("request_timeout_ms", 5000);
    declare_parameter<int>("planner_wait_timeout_ms", 5000);
    declare_parameter<int>("topic_wait_timeout_ms", 5000);

    load_parameters();

    planner_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    rclcpp::SubscriptionOptions subscription_options;
    subscription_options.callback_group = planner_cb_group_;

    planner_pose_sub_ = create_subscription<geometry_msgs::msg::PoseArray>(
      planner_pose_topic_, rclcpp::QoS(10),
      std::bind(&WallPatchFilterDemoNode::planner_pose_callback, this, std::placeholders::_1),
      subscription_options);
    planner_marker_sub_ = create_subscription<visualization_msgs::msg::MarkerArray>(
      planner_marker_topic_, rclcpp::QoS(10),
      std::bind(&WallPatchFilterDemoNode::planner_marker_callback, this, std::placeholders::_1),
      subscription_options);

    filtered_pose_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(filtered_pose_topic_, 10);
    filtered_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(filtered_marker_topic_, 10);

    planner_parameter_client_ = std::make_shared<rclcpp::AsyncParametersClient>(
      this, planner_node_name_, rclcpp::ParametersQoS(), planner_cb_group_);
    planner_trigger_client_ = create_client<std_srvs::srv::Trigger>(
      planner_service_name_, rclcpp::ServicesQoS(), planner_cb_group_);

    filter_service_ = create_service<wall_patch_planner::srv::FilterWallPoses>(
      "filter_wall_poses",
      std::bind(
        &WallPatchFilterDemoNode::handle_filter_request, this, std::placeholders::_1,
        std::placeholders::_2));

    RCLCPP_INFO(
      get_logger(),
      "wall_patch_filter_demo ready. planner=%s trigger=%s",
      planner_node_name_.c_str(),
      planner_service_name_.c_str());
  }

private:
  using FilterRequest = wall_patch_planner::srv::FilterWallPoses::Request;
  using FilterResponse = wall_patch_planner::srv::FilterWallPoses::Response;

  void load_parameters()
  {
    planner_node_name_ = get_parameter("planner_node_name").as_string();
    planner_service_name_ = get_parameter("planner_service_name").as_string();
    planner_pose_topic_ = get_parameter("planner_pose_topic").as_string();
    planner_marker_topic_ = get_parameter("planner_marker_topic").as_string();
    filtered_pose_topic_ = get_parameter("filtered_pose_topic").as_string();
    filtered_marker_topic_ = get_parameter("filtered_marker_topic").as_string();
    request_timeout_ms_ = static_cast<int>(get_parameter("request_timeout_ms").as_int());
    planner_wait_timeout_ms_ = static_cast<int>(get_parameter("planner_wait_timeout_ms").as_int());
    topic_wait_timeout_ms_ = static_cast<int>(get_parameter("topic_wait_timeout_ms").as_int());
  }

  void planner_pose_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
  {
    {
      std::lock_guard<std::mutex> lock(planner_data_mutex_);
      latest_planner_poses_ = *msg;
      ++planner_pose_seq_;
    }
    planner_data_cv_.notify_all();
  }

  void planner_marker_callback(const visualization_msgs::msg::MarkerArray::SharedPtr msg)
  {
    {
      std::lock_guard<std::mutex> lock(planner_data_mutex_);
      latest_planner_markers_ = *msg;
      ++planner_marker_seq_;
    }
    planner_data_cv_.notify_all();
  }

  bool validate_request(const FilterRequest & request, std::string & reason) const
  {
    const std::size_t window_count = request.center_x.size();
    if (window_count == 0U) {
      reason = "At least one filter window is required";
      return false;
    }
    if (
      request.center_y.size() != window_count || request.center_z.size() != window_count ||
      request.range.size() != window_count)
    {
      reason = "center_x, center_y, center_z, and range must have the same length";
      return false;
    }
    if (request.wall_id < 0) {
      reason = "wall_id must be >= 0";
      return false;
    }
    for (std::size_t idx = 0; idx < window_count; ++idx) {
      if (!(request.range[idx] >= 0.0)) {
        std::ostringstream oss;
        oss << "range[" << idx << "] must be >= 0";
        reason = oss.str();
        return false;
      }
    }
    return true;
  }

  bool wait_for_planner_dependencies(std::string & reason) const
  {
    const auto timeout = std::chrono::milliseconds(planner_wait_timeout_ms_);
    if (!planner_parameter_client_->wait_for_service(timeout)) {
      reason = "Planner parameter service is unavailable";
      return false;
    }
    if (!planner_trigger_client_->wait_for_service(timeout)) {
      reason = "Planner trigger service is unavailable";
      return false;
    }
    return true;
  }

  bool set_planner_wall_id(int wall_id, std::string & reason) const
  {
    std::vector<rclcpp::Parameter> parameters;
    parameters.emplace_back("selected_wall_ids", std::vector<int64_t>{wall_id});

    auto future = planner_parameter_client_->set_parameters(parameters);
    if (future.wait_for(std::chrono::milliseconds(request_timeout_ms_)) != std::future_status::ready) {
      reason = "Timed out while setting selected_wall_ids on the planner";
      return false;
    }

    for (const auto & result : future.get()) {
      if (!result.successful) {
        reason = result.reason.empty() ? "Planner rejected selected_wall_ids" : result.reason;
        return false;
      }
    }
    return true;
  }

  bool refresh_wall_cache(int wall_id, WallPlanCache & cache, std::string & reason)
  {
    if (!wait_for_planner_dependencies(reason)) {
      return false;
    }
    if (!set_planner_wall_id(wall_id, reason)) {
      return false;
    }

    std::size_t pose_seq_before = 0U;
    std::size_t marker_seq_before = 0U;
    {
      std::lock_guard<std::mutex> lock(planner_data_mutex_);
      pose_seq_before = planner_pose_seq_;
      marker_seq_before = planner_marker_seq_;
    }

    auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
    auto future = planner_trigger_client_->async_send_request(request);
    if (future.wait_for(std::chrono::milliseconds(request_timeout_ms_)) != std::future_status::ready) {
      reason = "Timed out while waiting for the planner trigger response";
      return false;
    }

    const auto response = future.get();
    if (!response) {
      reason = "Planner returned an empty trigger response";
      return false;
    }
    if (!response->success) {
      reason = response->message.empty() ? "Planner failed to produce wall poses" : response->message;
      return false;
    }

    std::unique_lock<std::mutex> lock(planner_data_mutex_);
    const bool received_updates = planner_data_cv_.wait_for(
      lock, std::chrono::milliseconds(topic_wait_timeout_ms_),
      [&]() {
        return planner_pose_seq_ > pose_seq_before && planner_marker_seq_ > marker_seq_before;
      });
    if (!received_updates) {
      reason = "Planner responded, but no fresh poses/markers arrived on the subscribed topics";
      return false;
    }

    cache.poses = latest_planner_poses_;
    cache.markers = latest_planner_markers_;
    wall_plan_cache_[wall_id] = cache;
    return true;
  }

  bool pose_matches_any_window(
    const geometry_msgs::msg::Pose & pose,
    const FilterRequest & request) const
  {
    for (std::size_t idx = 0; idx < request.center_x.size(); ++idx) {
      const double dx = pose.position.x - request.center_x[idx];
      const double dy = pose.position.y - request.center_y[idx];
      const double dz = pose.position.z - request.center_z[idx];
      const double distance_sq = dx * dx + dy * dy + dz * dz;
      const double radius_sq = request.range[idx] * request.range[idx];
      if (distance_sq <= radius_sq) {
        return true;
      }
    }
    return false;
  }

  visualization_msgs::msg::MarkerArray build_filtered_markers(
    const visualization_msgs::msg::MarkerArray & source,
    const std::vector<std::size_t> & matched_indices) const
  {
    visualization_msgs::msg::MarkerArray filtered;

    // Clear previous selection in RViz before publishing the new subset.
    visualization_msgs::msg::Marker clear_marker;
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    filtered.markers.push_back(clear_marker);

    for (std::size_t output_index = 0; output_index < matched_indices.size(); ++output_index) {
      const std::size_t source_index = matched_indices[output_index];
      if (source_index >= source.markers.size()) {
        continue;
      }

      auto marker = source.markers[source_index];
      marker.id = static_cast<int32_t>(output_index);
      marker.ns = "filtered_roi_rectangles";
      marker.header.stamp = now();
      filtered.markers.push_back(std::move(marker));
    }

    return filtered;
  }

  void handle_filter_request(
    const std::shared_ptr<FilterRequest> request,
    std::shared_ptr<FilterResponse> response)
  {
    std::string reason;
    if (!validate_request(*request, reason)) {
      response->success = false;
      response->message = reason;
      return;
    }

    WallPlanCache cache;
    if (!refresh_wall_cache(request->wall_id, cache, reason)) {
      response->success = false;
      response->message = reason;
      return;
    }

    geometry_msgs::msg::PoseArray filtered_poses;
    filtered_poses.header = cache.poses.header;
    filtered_poses.header.stamp = now();

    std::vector<std::size_t> matched_indices;
    matched_indices.reserve(cache.poses.poses.size());

    // Preserve the planner ordering so marker-to-pose correspondence stays stable.
    for (std::size_t idx = 0; idx < cache.poses.poses.size(); ++idx) {
      const auto & pose = cache.poses.poses[idx];
      if (!pose_matches_any_window(pose, *request)) {
        continue;
      }

      filtered_poses.poses.push_back(pose);
      matched_indices.push_back(idx);
      response->pose_indices.push_back(static_cast<int32_t>(idx));
      response->x.push_back(pose.position.x);
      response->y.push_back(pose.position.y);
      response->z.push_back(pose.position.z);
    }

    filtered_pose_pub_->publish(filtered_poses);
    filtered_marker_pub_->publish(build_filtered_markers(cache.markers, matched_indices));

    std::ostringstream oss;
    oss << "Wall " << request->wall_id << ": matched " << matched_indices.size()
        << " pose(s) across " << request->center_x.size() << " window(s)";
    response->success = true;
    response->message = oss.str();
  }

  std::string planner_node_name_;
  std::string planner_service_name_;
  std::string planner_pose_topic_;
  std::string planner_marker_topic_;
  std::string filtered_pose_topic_;
  std::string filtered_marker_topic_;
  int request_timeout_ms_{5000};
  int planner_wait_timeout_ms_{5000};
  int topic_wait_timeout_ms_{5000};

  mutable std::mutex planner_data_mutex_;
  std::condition_variable planner_data_cv_;
  geometry_msgs::msg::PoseArray latest_planner_poses_;
  visualization_msgs::msg::MarkerArray latest_planner_markers_;
  std::size_t planner_pose_seq_{0U};
  std::size_t planner_marker_seq_{0U};
  std::unordered_map<int, WallPlanCache> wall_plan_cache_;

  rclcpp::CallbackGroup::SharedPtr planner_cb_group_;

  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr planner_pose_sub_;
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr planner_marker_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr filtered_pose_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr filtered_marker_pub_;
  std::shared_ptr<rclcpp::AsyncParametersClient> planner_parameter_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr planner_trigger_client_;
  rclcpp::Service<wall_patch_planner::srv::FilterWallPoses>::SharedPtr filter_service_;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<WallPatchFilterDemoNode>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
