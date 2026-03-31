#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace
{
constexpr double kEps = 1e-9;

struct PointXYZL
{
  float x{0.0F};
  float y{0.0F};
  float z{0.0F};
  int32_t label{0};
};

struct CameraFootprint
{
  double full_x_min{0.0};
  double full_x_max{0.0};
  double full_y_min{0.0};
  double full_y_max{0.0};
  double roi_x_min{0.0};
  double roi_x_max{0.0};
  double roi_y_min{0.0};
  double roi_y_max{0.0};
  double roi_width{0.0};
  double roi_height{0.0};
  double roi_center_x{0.0};
  double roi_center_y{0.0};
  double ee_width{0.0};
  double ee_height{0.0};
  double hard_width{0.0};
  double hard_height{0.0};
};

struct PatchPose
{
  geometry_msgs::msg::Pose pose;
  int wall_id{0};
  int patch_index{0};
  double axis_u{0.0};  // optical-axis intersection on wall plane
  double axis_v{0.0};
  double roi_u_min{0.0};
  double roi_u_max{0.0};
  double roi_v_min{0.0};
  double roi_v_max{0.0};
};

struct RectangleUV
{
  double center_u{0.0};
  double center_v{0.0};
  double width{0.0};
  double height{0.0};
  double offset_u{0.0};
  double offset_v{0.0};
};

struct CandidatePatch
{
  int wall_id{0};
  double center_u{0.0};
  double center_v{0.0};
  RectangleUV hard_polygon;
  RectangleUV roi_polygon;
  std::vector<int> covered_cells;
  double coverage_ratio{0.0};
};

struct SelectedPatchResult
{
  std::vector<CandidatePatch> selected_patches;
  std::vector<uint8_t> uncovered_mask;
};

struct WallGridData
{
  double min_u{0.0};
  double max_u{0.0};
  double min_v{0.0};
  double max_v{0.0};
  double grid_resolution{0.01};
  int size_u{0};
  int size_v{0};
  std::vector<uint8_t> coverage_mask;
  std::vector<uint8_t> outer_envelope;
};

struct WallModel
{
  int wall_id{0};
  Eigen::Vector3d centroid{Eigen::Vector3d::Zero()};
  Eigen::Vector3d normal{Eigen::Vector3d::UnitX()};
  Eigen::Vector3d axis_u{Eigen::Vector3d::UnitY()};
  Eigen::Vector3d axis_v{Eigen::Vector3d::UnitZ()};
  double min_u{0.0};
  double max_u{0.0};
  double min_v{0.0};
  double max_v{0.0};
  std::vector<PointXYZL> points;
  std::vector<Eigen::Vector2d> uv_points;
};

struct LabelLayout
{
  bool has_label{false};
  uint32_t label_offset{0U};
  uint8_t label_datatype{0U};
};

LabelLayout find_label_layout(const sensor_msgs::msg::PointCloud2 & cloud)
{
  LabelLayout result;
  for (const auto & field : cloud.fields) {
    if (field.name == "label") {
      result.has_label = true;
      result.label_offset = field.offset;
      result.label_datatype = field.datatype;
      return result;
    }
  }
  return result;
}

int32_t read_label(const sensor_msgs::msg::PointCloud2 & cloud, std::size_t byte_offset, const LabelLayout & layout)
{
  const std::size_t idx = byte_offset + layout.label_offset;
  switch (layout.label_datatype) {
    case sensor_msgs::msg::PointField::INT32: {
      int32_t value;
      std::memcpy(&value, &cloud.data[idx], sizeof(int32_t));
      return value;
    }
    case sensor_msgs::msg::PointField::UINT32: {
      uint32_t value;
      std::memcpy(&value, &cloud.data[idx], sizeof(uint32_t));
      return static_cast<int32_t>(value);
    }
    case sensor_msgs::msg::PointField::INT16: {
      int16_t value;
      std::memcpy(&value, &cloud.data[idx], sizeof(int16_t));
      return static_cast<int32_t>(value);
    }
    case sensor_msgs::msg::PointField::UINT16: {
      uint16_t value;
      std::memcpy(&value, &cloud.data[idx], sizeof(uint16_t));
      return static_cast<int32_t>(value);
    }
    case sensor_msgs::msg::PointField::INT8: {
      int8_t value;
      std::memcpy(&value, &cloud.data[idx], sizeof(int8_t));
      return static_cast<int32_t>(value);
    }
    case sensor_msgs::msg::PointField::UINT8: {
      uint8_t value;
      std::memcpy(&value, &cloud.data[idx], sizeof(uint8_t));
      return static_cast<int32_t>(value);
    }
    case sensor_msgs::msg::PointField::FLOAT32: {
      float value;
      std::memcpy(&value, &cloud.data[idx], sizeof(float));
      return static_cast<int32_t>(std::lround(value));
    }
    case sensor_msgs::msg::PointField::FLOAT64: {
      double value;
      std::memcpy(&value, &cloud.data[idx], sizeof(double));
      return static_cast<int32_t>(std::llround(value));
    }
    default:
      return 0;
  }
}

bool is_finite_point(float x, float y, float z)
{
  return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

std::vector<PointXYZL> extract_points_with_label(const sensor_msgs::msg::PointCloud2 & cloud)
{
  const auto layout = find_label_layout(cloud);
  if (!layout.has_label) {
    throw std::runtime_error("Input cloud is missing 'label' field");
  }

  std::vector<PointXYZL> points;
  points.reserve(static_cast<std::size_t>(cloud.width) * static_cast<std::size_t>(cloud.height));

  sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iter_y(cloud, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iter_z(cloud, "z");

  for (std::size_t i = 0; i < static_cast<std::size_t>(cloud.width) * static_cast<std::size_t>(cloud.height);
       ++i, ++iter_x, ++iter_y, ++iter_z)
  {
    const float x = *iter_x;
    const float y = *iter_y;
    const float z = *iter_z;
    if (!is_finite_point(x, y, z)) {
      continue;
    }

    const std::size_t byte_offset = i * cloud.point_step;
    const int32_t label = read_label(cloud, byte_offset, layout);
    points.push_back(PointXYZL{x, y, z, label});
  }

  return points;
}

CameraFootprint compute_camera_footprint(
  const sensor_msgs::msg::CameraInfo & info,
  double distance,
  double roi_u_min,
  double roi_u_max,
  double roi_v_min,
  double roi_v_max,
  double ee_size_x,
  double ee_size_y)
{
  const double fx = info.k[0];
  const double fy = info.k[4];
  const double cx = info.k[2];
  const double cy = info.k[5];
  const double width = static_cast<double>(info.width);
  const double height = static_cast<double>(info.height);

  if (fx <= 0.0 || fy <= 0.0 || width <= 0.0 || height <= 0.0) {
    throw std::runtime_error("Invalid CameraInfo intrinsics");
  }

  const double u0 = roi_u_min * width;
  const double u1 = roi_u_max * width;
  const double v0 = roi_v_min * height;
  const double v1 = roi_v_max * height;

  CameraFootprint fp;
  fp.full_x_min = (0.0 - cx) * distance / fx;
  fp.full_x_max = (width - cx) * distance / fx;
  fp.full_y_min = (cy - height) * distance / fy;
  fp.full_y_max = (cy - 0.0) * distance / fy;

  fp.roi_x_min = (u0 - cx) * distance / fx;
  fp.roi_x_max = (u1 - cx) * distance / fx;
  fp.roi_y_min = (cy - v1) * distance / fy;
  fp.roi_y_max = (cy - v0) * distance / fy;

  fp.roi_width = fp.roi_x_max - fp.roi_x_min;
  fp.roi_height = fp.roi_y_max - fp.roi_y_min;
  fp.roi_center_x = 0.5 * (fp.roi_x_min + fp.roi_x_max);
  fp.roi_center_y = 0.5 * (fp.roi_y_min + fp.roi_y_max);
  fp.ee_width = ee_size_x;
  fp.ee_height = ee_size_y;
  fp.hard_width = std::max(fp.roi_width, ee_size_x);
  fp.hard_height = std::max(fp.roi_height, ee_size_y);
  return fp;
}

geometry_msgs::msg::Quaternion quaternion_from_basis(
  const Eigen::Vector3d & x_axis,
  const Eigen::Vector3d & y_axis,
  const Eigen::Vector3d & z_axis)
{
  Eigen::Matrix3d rot;
  rot.col(0) = x_axis.normalized();
  rot.col(1) = y_axis.normalized();
  rot.col(2) = z_axis.normalized();

  tf2::Matrix3x3 tf_rot(
    rot(0, 0), rot(0, 1), rot(0, 2),
    rot(1, 0), rot(1, 1), rot(1, 2),
    rot(2, 0), rot(2, 1), rot(2, 2));
  tf2::Quaternion q;
  tf_rot.getRotation(q);
  q.normalize();

  geometry_msgs::msg::Quaternion out;
  out.x = q.x();
  out.y = q.y();
  out.z = q.z();
  out.w = q.w();
  return out;
}

std::array<std::uint8_t, 3> color_for_index(std::size_t index)
{
  static const std::array<std::array<std::uint8_t, 3>, 12> palette{{
    {{230, 25, 75}}, {{60, 180, 75}}, {{255, 225, 25}}, {{0, 130, 200}},
    {{245, 130, 48}}, {{145, 30, 180}}, {{70, 240, 240}}, {{240, 50, 230}},
    {{210, 245, 60}}, {{250, 190, 190}}, {{0, 128, 128}}, {{170, 110, 40}}
  }};
  const auto & c = palette[index % palette.size()];
  return {{c[0], c[1], c[2]}};
}

int grid_index(int u_idx, int v_idx, int size_u)
{
  return v_idx * size_u + u_idx;
}

bool is_valid_cell(int u_idx, int v_idx, int size_u, int size_v)
{
  return u_idx >= 0 && u_idx < size_u && v_idx >= 0 && v_idx < size_v;
}

std::vector<uint8_t> dilate_mask(
  const std::vector<uint8_t> & mask, int size_u, int size_v, int radius_u, int radius_v)
{
  std::vector<uint8_t> out(mask.size(), 0U);
  for (int v = 0; v < size_v; ++v) {
    for (int u = 0; u < size_u; ++u) {
      bool occupied = false;
      for (int dv = -radius_v; dv <= radius_v && !occupied; ++dv) {
        for (int du = -radius_u; du <= radius_u; ++du) {
          const int nu = u + du;
          const int nv = v + dv;
          if (!is_valid_cell(nu, nv, size_u, size_v)) {
            continue;
          }
          if (mask[grid_index(nu, nv, size_u)] != 0U) {
            occupied = true;
            break;
          }
        }
      }
      out[grid_index(u, v, size_u)] = occupied ? 1U : 0U;
    }
  }
  return out;
}

std::vector<uint8_t> erode_mask(
  const std::vector<uint8_t> & mask, int size_u, int size_v, int radius_u, int radius_v)
{
  std::vector<uint8_t> out(mask.size(), 0U);
  for (int v = 0; v < size_v; ++v) {
    for (int u = 0; u < size_u; ++u) {
      bool occupied = true;
      for (int dv = -radius_v; dv <= radius_v && occupied; ++dv) {
        for (int du = -radius_u; du <= radius_u; ++du) {
          const int nu = u + du;
          const int nv = v + dv;
          if (!is_valid_cell(nu, nv, size_u, size_v) || mask[grid_index(nu, nv, size_u)] == 0U) {
            occupied = false;
            break;
          }
        }
      }
      out[grid_index(u, v, size_u)] = occupied ? 1U : 0U;
    }
  }
  return out;
}

std::vector<uint8_t> close_mask(
  const std::vector<uint8_t> & mask, int size_u, int size_v, int radius_u, int radius_v)
{
  return erode_mask(dilate_mask(mask, size_u, size_v, radius_u, radius_v), size_u, size_v, radius_u, radius_v);
}

std::vector<uint8_t> largest_connected_component(
  const std::vector<uint8_t> & mask, int size_u, int size_v)
{
  std::vector<uint8_t> visited(mask.size(), 0U);
  std::vector<int> best_component;

  for (int v = 0; v < size_v; ++v) {
    for (int u = 0; u < size_u; ++u) {
      const int start_idx = grid_index(u, v, size_u);
      if (mask[start_idx] == 0U || visited[start_idx] != 0U) {
        continue;
      }

      std::vector<int> component;
      std::vector<int> stack{start_idx};
      visited[start_idx] = 1U;
      while (!stack.empty()) {
        const int idx = stack.back();
        stack.pop_back();
        component.push_back(idx);

        const int cu = idx % size_u;
        const int cv = idx / size_u;
        constexpr std::array<std::array<int, 2>, 4> kNeighbors{{{{1, 0}}, {{-1, 0}}, {{0, 1}}, {{0, -1}}}};
        for (const auto & delta : kNeighbors) {
          const int nu = cu + delta[0];
          const int nv = cv + delta[1];
          if (!is_valid_cell(nu, nv, size_u, size_v)) {
            continue;
          }
          const int n_idx = grid_index(nu, nv, size_u);
          if (mask[n_idx] == 0U || visited[n_idx] != 0U) {
            continue;
          }
          visited[n_idx] = 1U;
          stack.push_back(n_idx);
        }
      }

      if (component.size() > best_component.size()) {
        best_component = std::move(component);
      }
    }
  }

  std::vector<uint8_t> out(mask.size(), 0U);
  for (const int idx : best_component) {
    out[idx] = 1U;
  }
  return out;
}

std::vector<uint8_t> fill_internal_holes(
  const std::vector<uint8_t> & mask, int size_u, int size_v)
{
  std::vector<uint8_t> visited(mask.size(), 0U);
  std::vector<int> stack;

  auto push_if_empty = [&](int u, int v) {
    if (!is_valid_cell(u, v, size_u, size_v)) {
      return;
    }
    const int idx = grid_index(u, v, size_u);
    if (mask[idx] != 0U || visited[idx] != 0U) {
      return;
    }
    visited[idx] = 1U;
    stack.push_back(idx);
  };

  for (int u = 0; u < size_u; ++u) {
    push_if_empty(u, 0);
    push_if_empty(u, size_v - 1);
  }
  for (int v = 0; v < size_v; ++v) {
    push_if_empty(0, v);
    push_if_empty(size_u - 1, v);
  }

  while (!stack.empty()) {
    const int idx = stack.back();
    stack.pop_back();
    const int cu = idx % size_u;
    const int cv = idx / size_u;
    constexpr std::array<std::array<int, 2>, 4> kNeighbors{{{{1, 0}}, {{-1, 0}}, {{0, 1}}, {{0, -1}}}};
    for (const auto & delta : kNeighbors) {
      push_if_empty(cu + delta[0], cv + delta[1]);
    }
  }

  std::vector<uint8_t> out = mask;
  for (std::size_t idx = 0; idx < out.size(); ++idx) {
    if (out[idx] == 0U && visited[idx] == 0U) {
      out[idx] = 1U;
    }
  }
  return out;
}

std::size_t count_occupied_cells(const std::vector<uint8_t> & mask)
{
  return static_cast<std::size_t>(
    std::count(mask.begin(), mask.end(), static_cast<uint8_t>(1U)));
}

}  // namespace

class WallPatchPlannerNode : public rclcpp::Node
{
public:
  WallPatchPlannerNode()
  : Node("wall_patch_planner"),
    tf_broadcaster_(std::make_unique<tf2_ros::TransformBroadcaster>(*this))
  {
    declare_parameter<std::string>("cloud_topic", "/ransac/walls");
    declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");
    declare_parameter<std::string>("output_pose_topic", "/wall_patch_planner/poses");
    declare_parameter<std::string>("output_debug_cloud_topic", "/wall_patch_planner/debug_cloud");
    declare_parameter<std::string>("output_marker_topic", "/wall_patch_planner/debug_markers");
    declare_parameter<std::string>("world_frame", "map");
    declare_parameter<std::string>("camera_frame_prefix", "wall_patch");
    declare_parameter<int>("selected_room_id", 0);
    declare_parameter<std::vector<int64_t>>("selected_wall_ids", {0});
    declare_parameter<int>("plane_label_stride", 1000);
    declare_parameter<double>("fov_distance", 0.35);
    declare_parameter<double>("wall_distance", 0.0);
    declare_parameter<double>("overlap", 0.15);
    declare_parameter<double>("ee_size_x", 0.30);
    declare_parameter<double>("ee_size_y", 0.30);
    declare_parameter<double>("grid_resolution", 0.02);
    declare_parameter<double>("min_valid_roi_ratio", 0.65);
    declare_parameter<double>("min_new_coverage_ratio", 0.05);
    declare_parameter<double>("roi_width_px", -1.0);
    declare_parameter<double>("roi_height_px", -1.0);
    declare_parameter<double>("roi_center_u_offset_px", 0.0);
    declare_parameter<double>("roi_center_v_offset_px", 0.0);
    declare_parameter<int>("max_tf_frames", 512);

    load_parameters();

    pose_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(output_pose_topic_, 10);
    debug_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_debug_cloud_topic_, 10);
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(output_marker_topic_, 10);

    plan_service_ = create_service<std_srvs::srv::Trigger>(
      "plan_patches",
      std::bind(
        &WallPatchPlannerNode::handle_plan_request, this, std::placeholders::_1,
        std::placeholders::_2));

    // CameraInfo is effectively static in this workflow, so caching the latest
    // message is more robust than timestamp-based synchronization.
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic_, rclcpp::QoS(10),
      std::bind(&WallPatchPlannerNode::cloud_callback, this, std::placeholders::_1));
    cam_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_, rclcpp::QoS(10),
      std::bind(&WallPatchPlannerNode::camera_info_callback, this, std::placeholders::_1));

    on_set_params_handle_ = add_on_set_parameters_callback(
      std::bind(&WallPatchPlannerNode::on_parameters_set, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "wall_patch_planner is caching the latest cloud and CameraInfo");
  }

private:
  void load_parameters()
  {
    cloud_topic_ = get_parameter("cloud_topic").as_string();
    camera_info_topic_ = get_parameter("camera_info_topic").as_string();
    output_pose_topic_ = get_parameter("output_pose_topic").as_string();
    output_debug_cloud_topic_ = get_parameter("output_debug_cloud_topic").as_string();
    output_marker_topic_ = get_parameter("output_marker_topic").as_string();
    world_frame_ = get_parameter("world_frame").as_string();
    camera_frame_prefix_ = get_parameter("camera_frame_prefix").as_string();
    selected_room_id_ = get_parameter("selected_room_id").as_int();
    plane_label_stride_ = get_parameter("plane_label_stride").as_int();
    fov_distance_ = get_parameter("fov_distance").as_double();
    wall_distance_ = get_parameter("wall_distance").as_double();
    ee_size_x_ = get_parameter("ee_size_x").as_double();
    ee_size_y_ = get_parameter("ee_size_y").as_double();
    grid_resolution_ = get_parameter("grid_resolution").as_double();
    min_valid_roi_ratio_ = get_parameter("min_valid_roi_ratio").as_double();
    min_new_coverage_ratio_ = get_parameter("min_new_coverage_ratio").as_double();
    roi_width_px_ = get_parameter("roi_width_px").as_double();
    roi_height_px_ = get_parameter("roi_height_px").as_double();
    roi_center_u_offset_px_ = get_parameter("roi_center_u_offset_px").as_double();
    roi_center_v_offset_px_ = get_parameter("roi_center_v_offset_px").as_double();
    overlap_ = get_parameter("overlap").as_double();
    max_tf_frames_ = get_parameter("max_tf_frames").as_int();

    selected_wall_ids_.clear();
    // Materialize the parameter array before iterating to avoid binding a
    // range-for loop to a temporary returned by get_parameter(...).
    const auto selected_wall_ids_param = get_parameter("selected_wall_ids").as_integer_array();
    for (const auto value : selected_wall_ids_param) {
      selected_wall_ids_.push_back(static_cast<int>(value));
    }
  }

  rcl_interfaces::msg::SetParametersResult on_parameters_set(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    for (const auto & param : parameters) {
      if (param.get_name() == "selected_room_id") {
        selected_room_id_ = param.as_int();
      } else if (param.get_name() == "selected_wall_ids") {
        selected_wall_ids_.clear();
        for (const auto value : param.as_integer_array()) {
          selected_wall_ids_.push_back(static_cast<int>(value));
        }
      } else if (param.get_name() == "fov_distance") {
        fov_distance_ = param.as_double();
      } else if (param.get_name() == "wall_distance") {
        wall_distance_ = param.as_double();
      } else if (param.get_name() == "ee_size_x") {
        ee_size_x_ = param.as_double();
      } else if (param.get_name() == "ee_size_y") {
        ee_size_y_ = param.as_double();
      } else if (param.get_name() == "grid_resolution") {
        grid_resolution_ = param.as_double();
      } else if (param.get_name() == "min_valid_roi_ratio") {
        min_valid_roi_ratio_ = param.as_double();
      } else if (param.get_name() == "min_new_coverage_ratio") {
        min_new_coverage_ratio_ = param.as_double();
      } else if (param.get_name() == "roi_width_px") {
        roi_width_px_ = param.as_double();
      } else if (param.get_name() == "roi_height_px") {
        roi_height_px_ = param.as_double();
      } else if (param.get_name() == "roi_center_u_offset_px") {
        roi_center_u_offset_px_ = param.as_double();
      } else if (param.get_name() == "roi_center_v_offset_px") {
        roi_center_v_offset_px_ = param.as_double();
      } else if (param.get_name() == "overlap") {
        overlap_ = param.as_double();
      } else if (param.get_name() == "plane_label_stride") {
        plane_label_stride_ = param.as_int();
      }
    }

    rcl_interfaces::msg::SetParametersResult result;
    result.successful = validate_configuration(result.reason);
    return result;
  }

  std::array<double, 4> resolve_roi_bounds(const sensor_msgs::msg::CameraInfo & info) const
  {
    const double image_width = static_cast<double>(info.width);
    const double image_height = static_cast<double>(info.height);
    if (image_width <= 0.0 || image_height <= 0.0) {
      throw std::runtime_error("CameraInfo width/height must be positive for pixel ROI");
    }
    if (!(roi_width_px_ > 0.0 && roi_height_px_ > 0.0)) {
      throw std::runtime_error("roi_width_px and roi_height_px must both be > 0 when using pixel ROI");
    }

    const double center_u = 0.5 * image_width + roi_center_u_offset_px_;
    const double center_v = 0.5 * image_height + roi_center_v_offset_px_;
    const double u0 = center_u - 0.5 * roi_width_px_;
    const double u1 = center_u + 0.5 * roi_width_px_;
    const double v0 = center_v - 0.5 * roi_height_px_;
    const double v1 = center_v + 0.5 * roi_height_px_;

    if (!(0.0 <= u0 && u0 < u1 && u1 <= image_width)) {
      throw std::runtime_error("Pixel ROI width/center produces bounds outside the image");
    }
    if (!(0.0 <= v0 && v0 < v1 && v1 <= image_height)) {
      throw std::runtime_error("Pixel ROI height/center produces bounds outside the image");
    }

    return {
      u0 / image_width,
      u1 / image_width,
      v0 / image_height,
      v1 / image_height
    };
  }

  bool validate_configuration(std::string & reason) const
  {
    if (plane_label_stride_ <= 0) {
      reason = "plane_label_stride must be > 0";
      return false;
    }
    if (fov_distance_ <= 0.0) {
      reason = "fov_distance must be > 0";
      return false;
    }
    if (wall_distance_ < 0.0) {
      reason = "wall_distance must be >= 0";
      return false;
    }
    if (ee_size_x_ <= 0.0 || ee_size_y_ <= 0.0) {
      reason = "ee_size_x and ee_size_y must be > 0";
      return false;
    }
    if (grid_resolution_ <= 0.0) {
      reason = "grid_resolution must be > 0";
      return false;
    }
    if (!(0.0 < min_valid_roi_ratio_ && min_valid_roi_ratio_ <= 1.0)) {
      reason = "min_valid_roi_ratio must be in (0,1]";
      return false;
    }
    if (!(0.0 < min_new_coverage_ratio_ && min_new_coverage_ratio_ <= 1.0)) {
      reason = "min_new_coverage_ratio must be in (0,1]";
      return false;
    }
    if (overlap_ < 0.0 || overlap_ >= 1.0) {
      reason = "overlap must be in [0,1)";
      return false;
    }
    if (!(roi_width_px_ > 0.0 && roi_height_px_ > 0.0)) {
      reason = "roi_width_px and roi_height_px must both be > 0";
      return false;
    }
    if (selected_wall_ids_.empty()) {
      reason = "selected_wall_ids cannot be empty";
      return false;
    }
    return true;
  }

  std::optional<Eigen::Vector3d> compute_room_centroid(const std::vector<PointXYZL> & points) const
  {
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    std::size_t count = 0;
    for (const auto & point : points) {
      const int room_id = point.label / plane_label_stride_;
      if (room_id != selected_room_id_) {
        continue;
      }
      centroid += Eigen::Vector3d(point.x, point.y, point.z);
      ++count;
    }

    if (count == 0U) {
      return std::nullopt;
    }
    return centroid / static_cast<double>(count);
  }

  void cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud)
  {
    latest_cloud_ = cloud;
  }

  void camera_info_callback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info)
  {
    latest_camera_info_ = camera_info;
  }

  void handle_plan_request(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    if (!latest_cloud_ || !latest_camera_info_) {
      response->success = false;
      response->message = "No cloud/camera_info available yet";
      return;
    }

    std::string reason;
    if (!validate_configuration(reason)) {
      response->success = false;
      response->message = reason;
      return;
    }

    try {
      const auto points = extract_points_with_label(*latest_cloud_);
      const auto room_centroid = compute_room_centroid(points);
      if (!room_centroid.has_value()) {
        response->success = false;
        response->message = "No points matched selected room id";
        return;
      }

      const auto walls = build_selected_walls(points, *room_centroid);
      if (walls.empty()) {
        response->success = false;
        response->message = "No points matched selected room/wall ids";
        return;
      }

      const auto roi_bounds = resolve_roi_bounds(*latest_camera_info_);
      const CameraFootprint footprint = compute_camera_footprint(
        *latest_camera_info_, fov_distance_,
        roi_bounds[0], roi_bounds[1], roi_bounds[2], roi_bounds[3], ee_size_x_, ee_size_y_);
      RCLCPP_INFO(
        get_logger(),
        "Planning request: room=%d walls=%zu roi_bounds=[%.3f, %.3f, %.3f, %.3f] fov_distance=%.3f wall_distance=%.3f roi_size=(%.3f, %.3f)m hard_size=(%.3f, %.3f)m",
        selected_room_id_, selected_wall_ids_.size(), roi_bounds[0], roi_bounds[1], roi_bounds[2], roi_bounds[3],
        fov_distance_, wall_distance_, footprint.roi_width, footprint.roi_height, footprint.hard_width, footprint.hard_height);

      geometry_msgs::msg::PoseArray pose_array;
      pose_array.header.stamp = now();
      pose_array.header.frame_id = world_frame_;

      visualization_msgs::msg::MarkerArray markers;
      std::vector<PatchPose> all_patches;
      int marker_id = 0;

      for (const auto & wall : walls) {
        RCLCPP_INFO(
          get_logger(),
          "Wall %d: input_points=%zu extents_u=[%.3f, %.3f] extents_v=[%.3f, %.3f]",
          wall.wall_id, wall.points.size(), wall.min_u, wall.max_u, wall.min_v, wall.max_v);
        const auto wall_grid = build_wall_2d_representation(wall);
        RCLCPP_INFO(
          get_logger(),
          "Wall %d: grid=%dx%d res=%.3f occupied=%zu envelope=%zu",
          wall.wall_id, wall_grid.size_u, wall_grid.size_v, wall_grid.grid_resolution,
          count_occupied_cells(wall_grid.coverage_mask), count_occupied_cells(wall_grid.outer_envelope));
        const auto candidates = generate_valid_candidate_centers(wall, wall_grid, footprint);
        RCLCPP_INFO(
          get_logger(),
          "Wall %d: valid_candidates=%zu",
          wall.wall_id, candidates.size());
        const auto selected_result = select_patches_for_coverage(candidates, wall_grid);
        RCLCPP_INFO(
          get_logger(),
          "Wall %d: selected_patches=%zu remaining_uncovered=%zu",
          wall.wall_id, selected_result.selected_patches.size(), count_occupied_cells(selected_result.uncovered_mask));
        const auto wall_patches = convert_selected_patches_to_poses(selected_result, wall);
        if (wall_patches.empty()) {
          RCLCPP_WARN(
            get_logger(),
            "Wall %d produced no poses. Check valid_candidates, min_valid_roi_ratio=%.3f, min_new_coverage_ratio=%.3f, grid_resolution=%.3f",
            wall.wall_id, min_valid_roi_ratio_, min_new_coverage_ratio_, grid_resolution_);
        }
        for (const auto & patch : wall_patches) {
          pose_array.poses.push_back(patch.pose);
          all_patches.push_back(patch);
          append_markers_for_patch(markers, wall, patch, footprint, marker_id++);
        }
      }

      if (pose_array.poses.empty()) {
        response->success = false;
        response->message = "No valid patch poses could be generated for the selected wall(s)";
        return;
      }

      pose_pub_->publish(pose_array);
      marker_pub_->publish(markers);
      debug_cloud_pub_->publish(build_debug_cloud(all_patches, walls));
      publish_patch_transforms(all_patches);

      std::ostringstream oss;
      oss << "Published " << pose_array.poses.size() << " patch poses for room " << selected_room_id_
          << " and " << walls.size() << " wall(s)";
      response->success = true;
      response->message = oss.str();
      RCLCPP_INFO(get_logger(), "%s", response->message.c_str());
    } catch (const std::exception & e) {
      response->success = false;
      response->message = e.what();
      RCLCPP_ERROR(get_logger(), "Patch planning failed: %s", e.what());
    }
  }

  std::vector<WallModel> build_selected_walls(
    const std::vector<PointXYZL> & points,
    const Eigen::Vector3d & room_centroid) const
  {
    std::unordered_map<int, std::vector<PointXYZL>> grouped;
    for (const auto & point : points) {
      const int room_id = point.label / plane_label_stride_;
      const int wall_id = point.label % plane_label_stride_;
      if (room_id != selected_room_id_) {
        continue;
      }
      if (std::find(selected_wall_ids_.begin(), selected_wall_ids_.end(), wall_id) == selected_wall_ids_.end()) {
        continue;
      }
      grouped[wall_id].push_back(point);
    }

    std::vector<WallModel> walls;
    walls.reserve(grouped.size());
    for (auto & entry : grouped) {
      if (entry.second.size() < 10U) {
        continue;
      }
      walls.push_back(fit_wall(entry.first, entry.second, room_centroid));
    }
    std::sort(walls.begin(), walls.end(), [](const WallModel & a, const WallModel & b) {return a.wall_id < b.wall_id;});
    return walls;
  }

  WallModel fit_wall(
    int wall_id,
    const std::vector<PointXYZL> & points,
    const Eigen::Vector3d & room_centroid) const
  {
    WallModel wall;
    wall.wall_id = wall_id;
    wall.points = points;

    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (const auto & p : points) {
      centroid += Eigen::Vector3d(p.x, p.y, p.z);
    }
    centroid /= static_cast<double>(points.size());
    wall.centroid = centroid;

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto & p : points) {
      const Eigen::Vector3d d = Eigen::Vector3d(p.x, p.y, p.z) - centroid;
      cov += d * d.transpose();
    }
    cov /= std::max<std::size_t>(1U, points.size() - 1U);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Failed to compute wall eigensystem");
    }

    Eigen::Vector3d normal = solver.eigenvectors().col(0).normalized();
    const Eigen::Vector3d world_up = Eigen::Vector3d::UnitZ();
    Eigen::Vector3d interior_hint = room_centroid - centroid;
    interior_hint.z() = 0.0;
    if (interior_hint.squaredNorm() > kEps) {
      if (normal.dot(interior_hint) < 0.0) {
        normal = -normal;
      }
    } else if (normal.dot(Eigen::Vector3d::UnitX()) < 0.0) {
      normal = -normal;
    }

    Eigen::Vector3d axis_u = world_up.cross(normal);
    if (axis_u.norm() < kEps) {
      axis_u = solver.eigenvectors().col(2);
    }
    axis_u.normalize();
    Eigen::Vector3d axis_v = normal.cross(axis_u).normalized();
    if (axis_v.dot(world_up) < 0.0) {
      axis_v = -axis_v;
      axis_u = -axis_u;
    }

    wall.normal = normal;
    wall.axis_u = axis_u;
    wall.axis_v = axis_v;

    wall.min_u = std::numeric_limits<double>::infinity();
    wall.max_u = -std::numeric_limits<double>::infinity();
    wall.min_v = std::numeric_limits<double>::infinity();
    wall.max_v = -std::numeric_limits<double>::infinity();
    wall.uv_points.reserve(points.size());
    for (const auto & p : points) {
      const Eigen::Vector3d d = Eigen::Vector3d(p.x, p.y, p.z) - centroid;
      const double u = d.dot(axis_u);
      const double v = d.dot(axis_v);
      wall.uv_points.emplace_back(u, v);
      wall.min_u = std::min(wall.min_u, u);
      wall.max_u = std::max(wall.max_u, u);
      wall.min_v = std::min(wall.min_v, v);
      wall.max_v = std::max(wall.max_v, v);
    }
    return wall;
  }

  WallGridData build_wall_2d_representation(const WallModel & wall) const
  {
    WallGridData grid;
    grid.grid_resolution = grid_resolution_;
    const double margin = 2.0 * grid_resolution_;
    grid.min_u = wall.min_u - margin;
    grid.max_u = wall.max_u + margin;
    grid.min_v = wall.min_v - margin;
    grid.max_v = wall.max_v + margin;
    grid.size_u = std::max(1, static_cast<int>(std::ceil((grid.max_u - grid.min_u) / grid.grid_resolution)) + 1);
    grid.size_v = std::max(1, static_cast<int>(std::ceil((grid.max_v - grid.min_v) / grid.grid_resolution)) + 1);
    grid.coverage_mask.assign(static_cast<std::size_t>(grid.size_u * grid.size_v), 0U);

    for (const auto & uv : wall.uv_points) {
      const int cell_u = static_cast<int>(std::floor((uv.x() - grid.min_u) / grid.grid_resolution));
      const int cell_v = static_cast<int>(std::floor((uv.y() - grid.min_v) / grid.grid_resolution));
      if (!is_valid_cell(cell_u, cell_v, grid.size_u, grid.size_v)) {
        continue;
      }
      grid.coverage_mask[grid_index(cell_u, cell_v, grid.size_u)] = 1U;
    }

    const auto closed_mask = close_mask(grid.coverage_mask, grid.size_u, grid.size_v, 1, 1);
    const auto largest_component = largest_connected_component(closed_mask, grid.size_u, grid.size_v);
    grid.coverage_mask = largest_component;
    grid.outer_envelope = fill_internal_holes(largest_component, grid.size_u, grid.size_v);
    return grid;
  }

  RectangleUV build_rectangle_polygon(
    double center_u,
    double center_v,
    double width,
    double height,
    double offset_u,
    double offset_v) const
  {
    RectangleUV rect;
    rect.center_u = center_u;
    rect.center_v = center_v;
    rect.width = width;
    rect.height = height;
    rect.offset_u = offset_u;
    rect.offset_v = offset_v;
    return rect;
  }

  std::vector<int> cells_inside_rectangle(
    const WallGridData & grid,
    const RectangleUV & rect) const
  {
    const double cu = rect.center_u + rect.offset_u;
    const double cv = rect.center_v + rect.offset_v;
    const double min_u = cu - 0.5 * rect.width;
    const double max_u = cu + 0.5 * rect.width;
    const double min_v = cv - 0.5 * rect.height;
    const double max_v = cv + 0.5 * rect.height;

    const int min_cell_u = std::max(0, static_cast<int>(std::floor((min_u - grid.min_u) / grid.grid_resolution)));
    const int max_cell_u = std::min(
      grid.size_u - 1, static_cast<int>(std::ceil((max_u - grid.min_u) / grid.grid_resolution)));
    const int min_cell_v = std::max(0, static_cast<int>(std::floor((min_v - grid.min_v) / grid.grid_resolution)));
    const int max_cell_v = std::min(
      grid.size_v - 1, static_cast<int>(std::ceil((max_v - grid.min_v) / grid.grid_resolution)));

    std::vector<int> cells;
    for (int cell_v = min_cell_v; cell_v <= max_cell_v; ++cell_v) {
      for (int cell_u = min_cell_u; cell_u <= max_cell_u; ++cell_u) {
        const double center_cell_u = grid.min_u + (static_cast<double>(cell_u) + 0.5) * grid.grid_resolution;
        const double center_cell_v = grid.min_v + (static_cast<double>(cell_v) + 0.5) * grid.grid_resolution;
        if (center_cell_u < min_u - 1e-6 || center_cell_u > max_u + 1e-6 ||
            center_cell_v < min_v - 1e-6 || center_cell_v > max_v + 1e-6)
        {
          continue;
        }
        cells.push_back(grid_index(cell_u, cell_v, grid.size_u));
      }
    }
    return cells;
  }

  bool rectangle_fully_inside_mask(
    const RectangleUV & rect,
    const WallGridData & grid,
    const std::vector<uint8_t> & mask) const
  {
    const auto cells = cells_inside_rectangle(grid, rect);
    if (cells.empty()) {
      return false;
    }
    for (const int idx : cells) {
      if (mask[static_cast<std::size_t>(idx)] == 0U) {
        return false;
      }
    }
    return true;
  }

  std::vector<CandidatePatch> generate_valid_candidate_centers(
    const WallModel & wall,
    const WallGridData & grid,
    const CameraFootprint & fp) const
  {
    const int kernel_half_u = std::max(0, static_cast<int>(std::ceil((0.5 * fp.hard_width) / grid.grid_resolution)));
    const int kernel_half_v = std::max(0, static_cast<int>(std::ceil((0.5 * fp.hard_height) / grid.grid_resolution)));
    const auto valid_center_region = erode_mask(
      grid.outer_envelope, grid.size_u, grid.size_v, kernel_half_u, kernel_half_v);
    const auto valid_center_count = count_occupied_cells(valid_center_region);
    if (valid_center_count == 0U) {
      RCLCPP_WARN(
        get_logger(),
        "Wall %d: valid center region is empty after erosion. hard_size=(%.3f, %.3f)m grid_res=%.3f envelope_cells=%zu",
        wall.wall_id, fp.hard_width, fp.hard_height, grid.grid_resolution,
        count_occupied_cells(grid.outer_envelope));
    } else {
      RCLCPP_INFO(
        get_logger(),
        "Wall %d: valid center cells after erosion=%zu",
        wall.wall_id, valid_center_count);
    }

    const double step_u = std::max(fp.roi_width * (1.0 - overlap_), grid.grid_resolution);
    const double step_v = std::max(fp.roi_height * (1.0 - overlap_), grid.grid_resolution);
    const double max_shift_u =
      0.5 * std::max(fp.hard_width - fp.roi_width, 0.0) + overlap_ * fp.roi_width;
    const double max_shift_v =
      0.5 * std::max(fp.hard_height - fp.roi_height, 0.0) + overlap_ * fp.roi_height;

    std::vector<CandidatePatch> candidates;
    std::size_t rejected_hard_boundary = 0U;
    std::size_t rejected_empty_roi = 0U;
    std::size_t rejected_low_coverage = 0U;
    std::size_t rejected_no_feasible_center = 0U;
    for (double roi_center_v = wall.min_v; roi_center_v <= wall.max_v + 1e-6; roi_center_v += step_v) {
      for (double roi_center_u = wall.min_u; roi_center_u <= wall.max_u + 1e-6; roi_center_u += step_u) {
        const double nominal_hard_center_u = roi_center_u - fp.roi_center_x;
        const double nominal_hard_center_v = roi_center_v - fp.roi_center_y;

        double chosen_hard_center_u = 0.0;
        double chosen_hard_center_v = 0.0;
        double best_distance_sq = std::numeric_limits<double>::infinity();

        for (int cell_v = 0; cell_v < grid.size_v; ++cell_v) {
          for (int cell_u = 0; cell_u < grid.size_u; ++cell_u) {
            const int idx = grid_index(cell_u, cell_v, grid.size_u);
            if (valid_center_region[static_cast<std::size_t>(idx)] == 0U) {
              continue;
            }

            const double hard_center_u = grid.min_u + (static_cast<double>(cell_u) + 0.5) * grid.grid_resolution;
            const double hard_center_v = grid.min_v + (static_cast<double>(cell_v) + 0.5) * grid.grid_resolution;
            const double delta_u = std::abs(hard_center_u - nominal_hard_center_u);
            const double delta_v = std::abs(hard_center_v - nominal_hard_center_v);
            if (delta_u > max_shift_u + 1e-6 || delta_v > max_shift_v + 1e-6) {
              continue;
            }

            const double distance_sq = delta_u * delta_u + delta_v * delta_v;
            if (distance_sq < best_distance_sq) {
              best_distance_sq = distance_sq;
              chosen_hard_center_u = hard_center_u;
              chosen_hard_center_v = hard_center_v;
            }
          }
        }

        if (!std::isfinite(best_distance_sq)) {
          ++rejected_no_feasible_center;
          continue;
        }

        CandidatePatch candidate;
        candidate.wall_id = wall.wall_id;
        candidate.center_u = chosen_hard_center_u;
        candidate.center_v = chosen_hard_center_v;
        candidate.hard_polygon = build_rectangle_polygon(
          candidate.center_u, candidate.center_v, fp.hard_width, fp.hard_height, 0.0, 0.0);
        candidate.roi_polygon = build_rectangle_polygon(
          roi_center_u, roi_center_v, fp.roi_width, fp.roi_height, 0.0, 0.0);

        if (!rectangle_fully_inside_mask(candidate.hard_polygon, grid, grid.outer_envelope)) {
          ++rejected_hard_boundary;
          continue;
        }

        const auto roi_cells = cells_inside_rectangle(grid, candidate.roi_polygon);
        if (roi_cells.empty()) {
          ++rejected_empty_roi;
          continue;
        }

        int valid_count = 0;
        for (const int roi_idx : roi_cells) {
          if (grid.coverage_mask[static_cast<std::size_t>(roi_idx)] != 0U) {
            candidate.covered_cells.push_back(roi_idx);
            ++valid_count;
          }
        }
        candidate.coverage_ratio = static_cast<double>(valid_count) / static_cast<double>(roi_cells.size());
        if (candidate.coverage_ratio < min_valid_roi_ratio_) {
          ++rejected_low_coverage;
          continue;
        }
        candidates.push_back(std::move(candidate));
      }
    }
    RCLCPP_INFO(
      get_logger(),
      "Wall %d: candidate rejection stats no_feasible_center=%zu hard_boundary=%zu empty_roi=%zu low_coverage=%zu accepted=%zu",
      wall.wall_id, rejected_no_feasible_center, rejected_hard_boundary, rejected_empty_roi, rejected_low_coverage,
      candidates.size());
    return candidates;
  }

  SelectedPatchResult select_patches_for_coverage(
    const std::vector<CandidatePatch> & candidates,
    const WallGridData & grid) const
  {
    SelectedPatchResult result;
    result.uncovered_mask = grid.coverage_mask;
    const double total_wall_cells = static_cast<double>(
      std::count(grid.coverage_mask.begin(), grid.coverage_mask.end(), static_cast<uint8_t>(1U)));

    if (total_wall_cells <= 0.0) {
      return result;
    }

    std::vector<uint8_t> candidate_disabled(candidates.size(), 0U);
    while (true) {
      int best_index = -1;
      double best_gain_ratio = 0.0;

      for (std::size_t idx = 0; idx < candidates.size(); ++idx) {
        if (candidate_disabled[idx] != 0U) {
          continue;
        }

        int gain = 0;
        for (const int cell_idx : candidates[idx].covered_cells) {
          if (result.uncovered_mask[static_cast<std::size_t>(cell_idx)] != 0U) {
            ++gain;
          }
        }
        const double gain_ratio = static_cast<double>(gain) / total_wall_cells;
        if (gain_ratio > best_gain_ratio) {
          best_gain_ratio = gain_ratio;
          best_index = static_cast<int>(idx);
        }
      }

      if (best_index < 0 || best_gain_ratio < min_new_coverage_ratio_) {
        if (best_index < 0) {
          RCLCPP_INFO(get_logger(), "Coverage selection stopped: no remaining candidate adds new coverage");
        } else {
          RCLCPP_INFO(
            get_logger(),
            "Coverage selection stopped: best_gain_ratio=%.4f below threshold=%.4f",
            best_gain_ratio, min_new_coverage_ratio_);
        }
        break;
      }

      result.selected_patches.push_back(candidates[static_cast<std::size_t>(best_index)]);
      candidate_disabled[static_cast<std::size_t>(best_index)] = 1U;
      for (const int cell_idx : candidates[static_cast<std::size_t>(best_index)].covered_cells) {
        result.uncovered_mask[static_cast<std::size_t>(cell_idx)] = 0U;
      }

      if (std::none_of(
            result.uncovered_mask.begin(), result.uncovered_mask.end(),
            [](uint8_t value) {return value != 0U;}))
      {
        break;
      }
    }
    return result;
  }

  std::vector<PatchPose> convert_selected_patches_to_poses(
    const SelectedPatchResult & selected_result,
    const WallModel & wall) const
  {
    std::vector<PatchPose> poses;
    poses.reserve(selected_result.selected_patches.size());

    int patch_idx = 0;
    for (const auto & selected_patch : selected_result.selected_patches) {
      const Eigen::Vector3d wall_point =
        wall.centroid + selected_patch.center_u * wall.axis_u + selected_patch.center_v * wall.axis_v;
      const Eigen::Vector3d position = wall_point + wall.normal * (fov_distance_ + wall_distance_);
      const Eigen::Vector3d forward = -wall.normal.normalized();
      const Eigen::Vector3d right = wall.axis_u.normalized();
      const Eigen::Vector3d up = wall.axis_v.normalized();

      PatchPose patch;
      patch.wall_id = wall.wall_id;
      patch.patch_index = patch_idx++;
      patch.axis_u = selected_patch.center_u;
      patch.axis_v = selected_patch.center_v;
      patch.roi_u_min = selected_patch.roi_polygon.center_u + selected_patch.roi_polygon.offset_u -
        0.5 * selected_patch.roi_polygon.width;
      patch.roi_u_max = selected_patch.roi_polygon.center_u + selected_patch.roi_polygon.offset_u +
        0.5 * selected_patch.roi_polygon.width;
      patch.roi_v_min = selected_patch.roi_polygon.center_v + selected_patch.roi_polygon.offset_v -
        0.5 * selected_patch.roi_polygon.height;
      patch.roi_v_max = selected_patch.roi_polygon.center_v + selected_patch.roi_polygon.offset_v +
        0.5 * selected_patch.roi_polygon.height;
      patch.pose.position.x = position.x();
      patch.pose.position.y = position.y();
      patch.pose.position.z = position.z();
      patch.pose.orientation = quaternion_from_basis(right, up, forward);
      poses.push_back(patch);
    }

    return poses;
  }

  void append_markers_for_patch(
    visualization_msgs::msg::MarkerArray & markers,
    const WallModel & wall,
    const PatchPose & patch,
    const CameraFootprint & /*fp*/,
    int marker_id) const
  {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = world_frame_;
    marker.header.stamp = now();
    marker.ns = "roi_rectangles";
    marker.id = marker_id;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.01;
    const auto rgb = color_for_index(static_cast<std::size_t>(patch.patch_index));
    marker.color.r = rgb[0] / 255.0F;
    marker.color.g = rgb[1] / 255.0F;
    marker.color.b = rgb[2] / 255.0F;
    marker.color.a = 0.85F;
    marker.lifetime = rclcpp::Duration::from_seconds(0.0);

    const auto corner = [&](double u, double v) {
      geometry_msgs::msg::Point p;
      const Eigen::Vector3d xyz = wall.centroid + u * wall.axis_u + v * wall.axis_v;
      p.x = xyz.x();
      p.y = xyz.y();
      p.z = xyz.z();
      return p;
    };

    marker.points.push_back(corner(patch.roi_u_min, patch.roi_v_min));
    marker.points.push_back(corner(patch.roi_u_max, patch.roi_v_min));
    marker.points.push_back(corner(patch.roi_u_max, patch.roi_v_max));
    marker.points.push_back(corner(patch.roi_u_min, patch.roi_v_max));
    marker.points.push_back(corner(patch.roi_u_min, patch.roi_v_min));
    markers.markers.push_back(marker);
  }

  sensor_msgs::msg::PointCloud2 build_debug_cloud(
    const std::vector<PatchPose> & patches,
    const std::vector<WallModel> & walls) const
  {
    std::unordered_map<int, std::vector<PatchPose>> by_wall;
    for (const auto & patch : patches) {
      by_wall[patch.wall_id].push_back(patch);
    }

    std::size_t total_points = 0;
    for (const auto & wall : walls) {
      total_points += wall.points.size();
    }

    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.frame_id = world_frame_;
    cloud.header.stamp = now();
    cloud.height = 1;
    cloud.width = static_cast<uint32_t>(total_points);
    cloud.is_bigendian = false;
    cloud.is_dense = true;

    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
    modifier.resize(total_points);

    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(cloud, "r");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(cloud, "g");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(cloud, "b");

    for (const auto & wall : walls) {
      const auto patch_it = by_wall.find(wall.wall_id);
      for (const auto & p : wall.points) {
        *iter_x = p.x;
        *iter_y = p.y;
        *iter_z = p.z;

        std::array<std::uint8_t, 3> color{{180, 180, 180}};
        if (patch_it != by_wall.end() && !patch_it->second.empty()) {
          const Eigen::Vector3d d = Eigen::Vector3d(p.x, p.y, p.z) - wall.centroid;
          const double u = d.dot(wall.axis_u);
          const double v = d.dot(wall.axis_v);

          int assigned = -1;
          for (std::size_t idx = 0; idx < patch_it->second.size(); ++idx) {
            const auto & patch = patch_it->second[idx];
            if (u >= patch.roi_u_min - 1e-6 && u <= patch.roi_u_max + 1e-6 &&
                v >= patch.roi_v_min - 1e-6 && v <= patch.roi_v_max + 1e-6)
            {
              assigned = static_cast<int>(idx);
              break;
            }
          }
          if (assigned >= 0) {
            const auto rgb = color_for_index(static_cast<std::size_t>(assigned));
            color = {rgb[0], rgb[1], rgb[2]};
          }
        }

        *iter_r = color[0];
        *iter_g = color[1];
        *iter_b = color[2];
        ++iter_x; ++iter_y; ++iter_z; ++iter_r; ++iter_g; ++iter_b;
      }
    }

    return cloud;
  }

  void publish_patch_transforms(const std::vector<PatchPose> & patches)
  {
    const auto stamp = now();
    const std::size_t num_frames = std::min<std::size_t>(patches.size(), static_cast<std::size_t>(max_tf_frames_));
    for (std::size_t i = 0; i < num_frames; ++i) {
      geometry_msgs::msg::TransformStamped tf;
      tf.header.frame_id = world_frame_;
      tf.header.stamp = stamp;
      std::ostringstream frame_name;
      frame_name << camera_frame_prefix_ << "_room_" << selected_room_id_ << "_wall_" << patches[i].wall_id
                 << "_patch_" << patches[i].patch_index;
      tf.child_frame_id = frame_name.str();
      tf.transform.translation.x = patches[i].pose.position.x;
      tf.transform.translation.y = patches[i].pose.position.y;
      tf.transform.translation.z = patches[i].pose.position.z;
      tf.transform.rotation = patches[i].pose.orientation;
      tf_broadcaster_->sendTransform(tf);
    }
  }

  std::string cloud_topic_;
  std::string camera_info_topic_;
  std::string output_pose_topic_;
  std::string output_debug_cloud_topic_;
  std::string output_marker_topic_;
  std::string world_frame_;
  std::string camera_frame_prefix_;
  int selected_room_id_{0};
  std::vector<int> selected_wall_ids_;
  int plane_label_stride_{1000};
  double fov_distance_{0.35};
  double wall_distance_{0.0};
  double overlap_{0.15};
  double ee_size_x_{0.30};
  double ee_size_y_{0.30};
  double grid_resolution_{0.02};
  double min_valid_roi_ratio_{0.65};
  double min_new_coverage_ratio_{0.05};
  double roi_width_px_{-1.0};
  double roi_height_px_{-1.0};
  double roi_center_u_offset_px_{0.0};
  double roi_center_v_offset_px_{0.0};
  int max_tf_frames_{512};

  sensor_msgs::msg::PointCloud2::ConstSharedPtr latest_cloud_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr latest_camera_info_;

  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pose_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_cloud_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr plan_service_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_sub_;

  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr on_set_params_handle_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<WallPatchPlannerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
