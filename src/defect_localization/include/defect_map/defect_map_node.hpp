/**
 * @file defect_map_node.hpp
 * @brief Declaration of the in-package standalone defect map ROS 2 node.
 */
#ifndef DEFECT_LOCALIZATION__DEFECT_MAP__DEFECT_MAP_NODE_HPP_
#define DEFECT_LOCALIZATION__DEFECT_MAP__DEFECT_MAP_NODE_HPP_

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "defect_map_interfaces/srv/add_defects.hpp"
#include "defect_map_interfaces/srv/clear_defect_map.hpp"
#include "defect_map_interfaces/srv/get_defect_map.hpp"
#include "defect_map_interfaces/srv/get_defects.hpp"
#include "defect_map_interfaces/srv/load_defect_map.hpp"
#include "defect_map_interfaces/srv/process_clusters.hpp"
#include "defect_map_interfaces/srv/remove_defects.hpp"
#include "defect_map_interfaces/srv/save_defect_map.hpp"
#include "defect_map/json_persistence.hpp"
#include "defect_map/map_store.hpp"

namespace defect_localization
{

/**
 * @brief Standalone ROS 2 node that owns the authoritative defect map state.
 *
 * The node exposes write, query, cluster, and persistence services while the
 * mutable map state itself remains isolated inside @c defect_map::MapStore.
 */
class DefectMapNode : public rclcpp::Node
{
public:
  /**
   * @brief Construct the defect map node and advertise all services.
   * @param options ROS node options.
   */
  explicit DefectMapNode(const rclcpp::NodeOptions & options);

private:
  /**
   * @brief Resolve a save/load request path or fall back to the default path.
   * @param requested_path Path from the service request.
   * @return Resolved absolute path or a detailed error string.
   */
  defect_map::Expected<std::filesystem::path> resolveMapPath(
    std::string_view requested_path) const;

  /**
   * @brief Load the configured startup map snapshot when enabled.
   */
  void attemptAutoload();

  /**
   * @brief Optionally log snapshot counts after successful mutating operations.
   * @param context Short operation label used in the log line.
   */
  void maybeLogSnapshotMetadata(const std::string & context) const;

  /**
   * @brief Publish the current raw defect-map snapshot as a debug point cloud.
   * @param context Short operation label used in logs.
   */
  void publishRawMapDebugCloud(const std::string & context);

  /**
   * @brief AddDefects service callback.
   */
  void onAddDefects(
    const std::shared_ptr<defect_map_interfaces::srv::AddDefects::Request> request,
    std::shared_ptr<defect_map_interfaces::srv::AddDefects::Response> response);

  /**
   * @brief RemoveDefects service callback.
   */
  void onRemoveDefects(
    const std::shared_ptr<defect_map_interfaces::srv::RemoveDefects::Request> request,
    std::shared_ptr<defect_map_interfaces::srv::RemoveDefects::Response> response);

  /**
   * @brief GetDefects service callback.
   */
  void onGetDefects(
    const std::shared_ptr<defect_map_interfaces::srv::GetDefects::Request> request,
    std::shared_ptr<defect_map_interfaces::srv::GetDefects::Response> response);

  /**
   * @brief GetDefectMap compatibility service callback.
   */
  void onGetDefectMap(
    const std::shared_ptr<defect_map_interfaces::srv::GetDefectMap::Request> request,
    std::shared_ptr<defect_map_interfaces::srv::GetDefectMap::Response> response);

  /**
   * @brief ProcessClusters service callback.
   */
  void onProcessClusters(
    const std::shared_ptr<defect_map_interfaces::srv::ProcessClusters::Request> request,
    std::shared_ptr<defect_map_interfaces::srv::ProcessClusters::Response> response);

  /**
   * @brief SaveDefectMap service callback.
   */
  void onSaveDefectMap(
    const std::shared_ptr<defect_map_interfaces::srv::SaveDefectMap::Request> request,
    std::shared_ptr<defect_map_interfaces::srv::SaveDefectMap::Response> response);

  /**
   * @brief LoadDefectMap service callback.
   */
  void onLoadDefectMap(
    const std::shared_ptr<defect_map_interfaces::srv::LoadDefectMap::Request> request,
    std::shared_ptr<defect_map_interfaces::srv::LoadDefectMap::Response> response);

  /**
   * @brief ClearDefectMap service callback.
   */
  void onClearDefectMap(
    const std::shared_ptr<defect_map_interfaces::srv::ClearDefectMap::Request> request,
    std::shared_ptr<defect_map_interfaces::srv::ClearDefectMap::Response> response);

  std::shared_ptr<defect_map::MapStore> map_store_;
  defect_map::JsonPersistence persistence_;

  std::string default_map_path_;
  bool autoload_on_startup_{false};
  bool save_pretty_json_{true};
  bool log_snapshot_sizes_{false};
  double voxel_size_m_{0.01};
  std::vector<std::string> class_names_;
  std::string debug_cloud_default_frame_id_{"world"};
  std::string last_debug_cloud_frame_id_;
  int max_service_wait_ms_{2000};

  rclcpp::CallbackGroup::SharedPtr service_callback_group_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr raw_map_debug_cloud_publisher_;
  rclcpp::Service<defect_map_interfaces::srv::AddDefects>::SharedPtr add_defects_service_;
  rclcpp::Service<defect_map_interfaces::srv::RemoveDefects>::SharedPtr remove_defects_service_;
  rclcpp::Service<defect_map_interfaces::srv::GetDefects>::SharedPtr get_defects_service_;
  rclcpp::Service<defect_map_interfaces::srv::GetDefectMap>::SharedPtr get_defect_map_service_;
  rclcpp::Service<defect_map_interfaces::srv::ProcessClusters>::SharedPtr process_clusters_service_;
  rclcpp::Service<defect_map_interfaces::srv::SaveDefectMap>::SharedPtr save_defect_map_service_;
  rclcpp::Service<defect_map_interfaces::srv::LoadDefectMap>::SharedPtr load_defect_map_service_;
  rclcpp::Service<defect_map_interfaces::srv::ClearDefectMap>::SharedPtr clear_defect_map_service_;
};

}  // namespace defect_localization

#endif  // DEFECT_LOCALIZATION__DEFECT_MAP__DEFECT_MAP_NODE_HPP_
