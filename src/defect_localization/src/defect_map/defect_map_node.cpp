/**
 * @file defect_map_node.cpp
 * @brief ROS service adapter for the in-package standalone defect map node.
 */
#include "defect_map/defect_map_node.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/point_field.hpp"

namespace defect_localization
{
namespace
{

constexpr int kDefaultServiceWaitMs = 2000;
constexpr double kDefaultVoxelSizeM = 0.01;

struct ColorRgb
{
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

constexpr std::array<ColorRgb, 10> kDebugPalette{{
    {255, 0, 0},
    {0, 255, 0},
    {0, 0, 255},
    {255, 255, 0},
    {255, 0, 255},
    {0, 255, 255},
    {255, 128, 0},
    {128, 0, 255},
    {255, 255, 255},
    {255, 64, 160},
  }};

uint32_t packRgb(const ColorRgb & color)
{
  return
    (static_cast<uint32_t>(color.r) << 16) |
    (static_cast<uint32_t>(color.g) << 8) |
    static_cast<uint32_t>(color.b);
}

ColorRgb colorForIndex(size_t index)
{
  return kDebugPalette[index];
}

}  // namespace

DefectMapNode::DefectMapNode(const rclcpp::NodeOptions & options)
: Node("defect_map", options),
  map_store_(std::make_shared<defect_map::MapStore>())
{
  declare_parameter<std::string>("default_map_path", "");
  declare_parameter<bool>("autoload_on_startup", false);
  declare_parameter<bool>("save_pretty_json", true);
  declare_parameter<bool>("log_snapshot_sizes", false);
  declare_parameter<double>("voxel_size_m", kDefaultVoxelSizeM);
  declare_parameter<std::vector<std::string>>(
    "class_names",
    std::vector<std::string>{"positive", "negative", "line", "crack", "wood", "dent"});
  declare_parameter<std::string>("debug_cloud_default_frame_id", "world");
  declare_parameter<int>("max_service_wait_ms", kDefaultServiceWaitMs);

  default_map_path_ = get_parameter("default_map_path").as_string();
  autoload_on_startup_ = get_parameter("autoload_on_startup").as_bool();
  save_pretty_json_ = get_parameter("save_pretty_json").as_bool();
  log_snapshot_sizes_ = get_parameter("log_snapshot_sizes").as_bool();
  voxel_size_m_ = get_parameter("voxel_size_m").as_double();
  class_names_ = get_parameter("class_names").as_string_array();
  debug_cloud_default_frame_id_ = get_parameter("debug_cloud_default_frame_id").as_string();
  max_service_wait_ms_ = static_cast<int>(
    std::max<int64_t>(1, get_parameter("max_service_wait_ms").as_int()));
  if (voxel_size_m_ <= 0.0) {
    RCLCPP_WARN(
      get_logger(),
      "Invalid voxel_size_m=%.6f; falling back to %.6f",
      voxel_size_m_,
      kDefaultVoxelSizeM);
    voxel_size_m_ = kDefaultVoxelSizeM;
  }
  if (debug_cloud_default_frame_id_.empty()) {
    RCLCPP_WARN(
      get_logger(),
      "Empty debug_cloud_default_frame_id; falling back to 'world'");
    debug_cloud_default_frame_id_ = "world";
  }

  service_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  raw_map_debug_cloud_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    "~/debug/raw_defect_map_cloud",
    rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());

  add_defects_service_ = create_service<defect_map_interfaces::srv::AddDefects>(
    "~/add_defects",
    std::bind(&DefectMapNode::onAddDefects, this, std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(),
    service_callback_group_);
  remove_defects_service_ = create_service<defect_map_interfaces::srv::RemoveDefects>(
    "~/remove_defects",
    std::bind(
      &DefectMapNode::onRemoveDefects, this, std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(),
    service_callback_group_);
  get_defects_service_ = create_service<defect_map_interfaces::srv::GetDefects>(
    "~/get_defects",
    std::bind(&DefectMapNode::onGetDefects, this, std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(),
    service_callback_group_);
  get_defect_map_service_ = create_service<defect_map_interfaces::srv::GetDefectMap>(
    "~/get_defect_map",
    std::bind(
      &DefectMapNode::onGetDefectMap, this, std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(),
    service_callback_group_);
  process_clusters_service_ = create_service<defect_map_interfaces::srv::ProcessClusters>(
    "~/process_clusters",
    std::bind(
      &DefectMapNode::onProcessClusters, this, std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(),
    service_callback_group_);
  save_defect_map_service_ = create_service<defect_map_interfaces::srv::SaveDefectMap>(
    "~/save_defect_map",
    std::bind(
      &DefectMapNode::onSaveDefectMap, this, std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(),
    service_callback_group_);
  load_defect_map_service_ = create_service<defect_map_interfaces::srv::LoadDefectMap>(
    "~/load_defect_map",
    std::bind(
      &DefectMapNode::onLoadDefectMap, this, std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(),
    service_callback_group_);
  clear_defect_map_service_ = create_service<defect_map_interfaces::srv::ClearDefectMap>(
    "~/clear_defect_map",
    std::bind(
      &DefectMapNode::onClearDefectMap, this, std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(),
    service_callback_group_);

  attemptAutoload();
  publishRawMapDebugCloud("startup");
  RCLCPP_INFO(get_logger(), "defect_map node ready");
}

defect_map::Expected<std::filesystem::path> DefectMapNode::resolveMapPath(
  std::string_view requested_path) const
{
  const auto candidate =
    requested_path.empty() ? default_map_path_ : std::string(requested_path);
  if (candidate.empty()) {
    return std::unexpected(
      std::string("No map file path was provided and default_map_path is empty"));
  }

  try {
    return std::filesystem::absolute(std::filesystem::path(candidate));
  } catch (const std::exception & ex) {
    return std::unexpected(
      std::string("Failed to resolve map path '") + candidate + "': " + ex.what());
  }
}

void DefectMapNode::attemptAutoload()
{
  if (!autoload_on_startup_) {
    return;
  }

  const auto resolved_path = resolveMapPath("");
  if (!resolved_path) {
    RCLCPP_WARN(get_logger(), "Startup autoload skipped: %s", resolved_path.error().c_str());
    return;
  }

  if (!std::filesystem::exists(*resolved_path)) {
    RCLCPP_WARN(
      get_logger(),
      "Startup autoload skipped because map file does not exist: %s",
      resolved_path->string().c_str());
    return;
  }

  const auto loaded_state = persistence_.load(*resolved_path);
  if (!loaded_state) {
    RCLCPP_ERROR(
      get_logger(),
      "Failed to autoload map from %s: %s",
      resolved_path->string().c_str(), loaded_state.error().c_str());
    return;
  }

  const auto result = map_store_->replaceState(
    *loaded_state,
    std::chrono::milliseconds(max_service_wait_ms_));
  if (!result.success) {
    RCLCPP_ERROR(
      get_logger(),
      "Autoload rejected by map store: %s (%s)",
      result.status_code.c_str(), result.message.c_str());
    return;
  }

  RCLCPP_INFO(
    get_logger(),
    "Loaded %u entries from %s at startup",
    result.loaded_entries, resolved_path->string().c_str());
  maybeLogSnapshotMetadata("autoload");
}

void DefectMapNode::maybeLogSnapshotMetadata(const std::string & context) const
{
  if (!log_snapshot_sizes_) {
    return;
  }

  const auto metadata = map_store_->snapshotMetadata();
  RCLCPP_INFO(
    get_logger(),
    "[%s] raw=%zu clustered=%zu latest_uid=%" PRIu64 " cluster_epoch=%" PRIu64,
    context.c_str(),
    metadata.raw_count,
    metadata.clustered_count,
    metadata.latest_uid,
    metadata.cluster_epoch);
}

void DefectMapNode::publishRawMapDebugCloud(const std::string & context)
{
  const auto persistence_state = map_store_->capturePersistenceState();

  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.stamp = now();
  cloud.height = 1U;
  cloud.is_bigendian = false;
  cloud.is_dense = true;
  cloud.point_step = 16U;
  cloud.fields.resize(4U);
  cloud.fields[0].name = "x";
  cloud.fields[0].offset = 0U;
  cloud.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud.fields[0].count = 1U;
  cloud.fields[1].name = "y";
  cloud.fields[1].offset = 4U;
  cloud.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud.fields[1].count = 1U;
  cloud.fields[2].name = "z";
  cloud.fields[2].offset = 8U;
  cloud.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud.fields[2].count = 1U;
  cloud.fields[3].name = "rgb";
  cloud.fields[3].offset = 12U;
  cloud.fields[3].datatype = sensor_msgs::msg::PointField::UINT32;
  cloud.fields[3].count = 1U;

  size_t point_count = 0U;
  std::string frame_id;
  for (const auto & raw_record : persistence_state.raw_defects) {
    point_count += raw_record.voxels.size();
    if (frame_id.empty() && !raw_record.frame_id.empty()) {
      frame_id = raw_record.frame_id;
    }
  }

  if (frame_id.empty()) {
    frame_id = last_debug_cloud_frame_id_.empty() ?
      debug_cloud_default_frame_id_ :
      last_debug_cloud_frame_id_;
  }
  cloud.header.frame_id = frame_id;
  last_debug_cloud_frame_id_ = frame_id;

  cloud.width = static_cast<uint32_t>(point_count);
  cloud.row_step = cloud.point_step * cloud.width;
  cloud.data.resize(static_cast<size_t>(cloud.row_step));

  size_t point_index = 0U;
  for (const auto & raw_record : persistence_state.raw_defects) {
    if (!raw_record.frame_id.empty() && raw_record.frame_id != frame_id) {
      RCLCPP_WARN(
        get_logger(),
        "Skipping uid=%" PRIu64 " in frame '%s' because the debug cloud frame is '%s'",
        raw_record.uid,
        raw_record.frame_id.c_str(),
        frame_id.c_str());
      continue;
    }

    size_t palette_index = std::hash<std::string>{}(raw_record.label) % kDebugPalette.size();
    const auto class_it = std::find(class_names_.begin(), class_names_.end(), raw_record.label);
    if (class_it != class_names_.end()) {
      palette_index = static_cast<size_t>(std::distance(class_names_.begin(), class_it)) %
        kDebugPalette.size();
    }

    const auto color = packRgb(colorForIndex(palette_index));
    for (const auto & voxel : raw_record.voxels) {
      const auto offset = point_index * cloud.point_step;
      const float x = static_cast<float>((static_cast<double>(voxel.x) + 0.5) * voxel_size_m_);
      const float y = static_cast<float>((static_cast<double>(voxel.y) + 0.5) * voxel_size_m_);
      const float z = static_cast<float>((static_cast<double>(voxel.z) + 0.5) * voxel_size_m_);
      std::memcpy(&cloud.data[offset + 0U], &x, sizeof(float));
      std::memcpy(&cloud.data[offset + 4U], &y, sizeof(float));
      std::memcpy(&cloud.data[offset + 8U], &z, sizeof(float));
      std::memcpy(&cloud.data[offset + 12U], &color, sizeof(uint32_t));
      ++point_index;
    }
  }

  cloud.width = static_cast<uint32_t>(point_index);
  cloud.row_step = cloud.point_step * cloud.width;
  cloud.data.resize(static_cast<size_t>(cloud.row_step));
  raw_map_debug_cloud_publisher_->publish(cloud);

  RCLCPP_INFO(
    get_logger(),
    "[%s] Published raw defect-map cloud points=%zu defects=%zu frame=%s subscribers=%zu",
    context.c_str(),
    point_index,
    persistence_state.raw_defects.size(),
    frame_id.c_str(),
    raw_map_debug_cloud_publisher_->get_subscription_count());
}

void DefectMapNode::onAddDefects(
  const std::shared_ptr<defect_map_interfaces::srv::AddDefects::Request> request,
  std::shared_ptr<defect_map_interfaces::srv::AddDefects::Response> response)
{
  const auto result = map_store_->addDefects(
    request->defects,
    std::chrono::milliseconds(max_service_wait_ms_));

  response->accepted = result.success;
  response->status_code = result.status_code;
  response->message = result.message;
  response->latest_uid = result.latest_uid;
  response->accepted_count = result.accepted_count;
  response->rejected_count = result.rejected_count;
  response->retry_after_ms = result.retry_after_ms;

  if (result.success) {
    maybeLogSnapshotMetadata("add_defects");
    publishRawMapDebugCloud("add_defects");
  }
}

void DefectMapNode::onRemoveDefects(
  const std::shared_ptr<defect_map_interfaces::srv::RemoveDefects::Request> request,
  std::shared_ptr<defect_map_interfaces::srv::RemoveDefects::Response> response)
{
  const auto result = map_store_->removeDefects(
    request->uids,
    request->clusters,
    std::chrono::milliseconds(max_service_wait_ms_));

  response->success = result.success;
  response->status_code = result.status_code;
  response->message = result.message;
  response->removed_count = result.removed_count;
  response->not_found_uids = result.not_found_uids;

  if (result.success) {
    maybeLogSnapshotMetadata("remove_defects");
    publishRawMapDebugCloud("remove_defects");
  }
}

void DefectMapNode::onGetDefects(
  const std::shared_ptr<defect_map_interfaces::srv::GetDefects::Request> request,
  std::shared_ptr<defect_map_interfaces::srv::GetDefects::Response> response)
{
  const auto result = map_store_->getDefects(
    request->clustered_view,
    request->zone_id,
    request->label_filter);

  response->success = result.success;
  response->status_code = result.status_code;
  response->message = result.message;
  response->entries = result.entries;
  response->cluster_epoch = result.cluster_epoch;
}

void DefectMapNode::onGetDefectMap(
  const std::shared_ptr<defect_map_interfaces::srv::GetDefectMap::Request> request,
  std::shared_ptr<defect_map_interfaces::srv::GetDefectMap::Response> response)
{
  const auto result = map_store_->getDefects(
    request->clustered_view,
    "",
    request->label_filter);

  response->entries = result.entries;
  response->message = result.message;
  if (result.success) {
    response->success = true;
    response->status_code = "OK";
  } else if (result.status_code == "NO_DATA") {
    response->success = false;
    response->status_code = "NO_MAP";
    response->message = "No defect map entries match the requested filters";
  } else {
    response->success = false;
    response->status_code = "INTERNAL_ERROR";
  }
}

void DefectMapNode::onProcessClusters(
  const std::shared_ptr<defect_map_interfaces::srv::ProcessClusters::Request> request,
  std::shared_ptr<defect_map_interfaces::srv::ProcessClusters::Response> response)
{
  const auto result = map_store_->processClusters(
    request->force_recompute,
    std::chrono::milliseconds(max_service_wait_ms_));

  response->success = result.success;
  response->status_code = result.status_code;
  response->message = result.message;
  response->cluster_epoch = result.cluster_epoch;
  response->cluster_count = result.cluster_count;

  if (result.success) {
    maybeLogSnapshotMetadata("process_clusters");
  }
}

void DefectMapNode::onSaveDefectMap(
  const std::shared_ptr<defect_map_interfaces::srv::SaveDefectMap::Request> request,
  std::shared_ptr<defect_map_interfaces::srv::SaveDefectMap::Response> response)
{
  const auto resolved_path = resolveMapPath(request->file_path);
  if (!resolved_path) {
    response->success = false;
    response->status_code = "INVALID_PATH";
    response->message = resolved_path.error();
    return;
  }

  const auto persistence_state = map_store_->capturePersistenceState();
  const auto save_result = persistence_.save(*resolved_path, persistence_state, save_pretty_json_);
  if (!save_result) {
    response->success = false;
    response->status_code = "IO_ERROR";
    response->message = save_result.error();
    response->resolved_file_path = resolved_path->string();
    return;
  }

  response->success = true;
  response->status_code = "OK";
  response->message = "Defect map saved";
  response->resolved_file_path = resolved_path->string();
  response->saved_entries = static_cast<uint32_t>(persistence_state.raw_defects.size());
}

void DefectMapNode::onLoadDefectMap(
  const std::shared_ptr<defect_map_interfaces::srv::LoadDefectMap::Request> request,
  std::shared_ptr<defect_map_interfaces::srv::LoadDefectMap::Response> response)
{
  const auto resolved_path = resolveMapPath(request->file_path);
  if (!resolved_path) {
    response->success = false;
    response->status_code = "FILE_NOT_FOUND";
    response->message = resolved_path.error();
    return;
  }

  if (!std::filesystem::exists(*resolved_path)) {
    response->success = false;
    response->status_code = "FILE_NOT_FOUND";
    response->message = "Map file does not exist";
    response->resolved_file_path = resolved_path->string();
    return;
  }

  const auto loaded_state = persistence_.load(*resolved_path);
  if (!loaded_state) {
    response->success = false;
    response->status_code = "PARSE_ERROR";
    response->message = loaded_state.error();
    response->resolved_file_path = resolved_path->string();
    return;
  }

  const auto result = map_store_->replaceState(
    *loaded_state,
    std::chrono::milliseconds(max_service_wait_ms_));

  response->success = result.success;
  response->status_code = result.status_code;
  response->message = result.message;
  response->resolved_file_path = resolved_path->string();
  response->loaded_entries = result.loaded_entries;
  response->latest_uid = result.latest_uid;

  if (result.success) {
    maybeLogSnapshotMetadata("load_defect_map");
    publishRawMapDebugCloud("load_defect_map");
  }
}

void DefectMapNode::onClearDefectMap(
  const std::shared_ptr<defect_map_interfaces::srv::ClearDefectMap::Request> request,
  std::shared_ptr<defect_map_interfaces::srv::ClearDefectMap::Response> response)
{
  (void)request;

  const auto result = map_store_->clear(std::chrono::milliseconds(max_service_wait_ms_));

  response->success = result.success;
  response->status_code = result.status_code;
  response->message = result.message;
  response->cleared_raw_entries = result.cleared_raw_entries;
  response->cleared_latest_raw_entries = result.cleared_latest_raw_entries;
  response->cleared_latest_clustered_entries = result.cleared_latest_clustered_entries;
  response->cleared_pending_images = result.cleared_pending_images;
  response->cleared_queued_jobs = result.cleared_queued_jobs;

  if (result.success) {
    maybeLogSnapshotMetadata("clear_defect_map");
    publishRawMapDebugCloud("clear_defect_map");
  }
}

}  // namespace defect_localization

RCLCPP_COMPONENTS_REGISTER_NODE(defect_localization::DefectMapNode)
