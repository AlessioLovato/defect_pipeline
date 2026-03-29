#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

class RansacNode : public rclcpp::Node
{
public:
  using PointT = pcl::PointXYZL;
  using DebugPointT = pcl::PointXYZRGB;

  RansacNode() : Node("ransac_node")
  {
    input_topic_ = declare_parameter<std::string>("input_topic", "~/input_cloud");
    output_topic_ = declare_parameter<std::string>("output_topic", "~/wall_planes_cloud");
    debug_rgb_topic_ = declare_parameter<std::string>("debug_rgb_topic", "~/wall_planes_cloud_rgb");
    publish_debug_rgb_ = declare_parameter<bool>("publish_debug_rgb", true);
    segment_per_input_label_ = declare_parameter<bool>("segment_per_input_label", false);
    selected_room_id_ = declare_parameter<int>("selected_room_id", -1);
    plane_label_stride_ = declare_parameter<int>("plane_label_stride", 1000);
    target_semantic_label_ = declare_parameter<int>("target_semantic_label", 1);
    max_iterations_ = declare_parameter<int>("max_iterations", 500);
    distance_threshold_ = declare_parameter<double>("distance_threshold", 0.03);
    min_plane_points_ = declare_parameter<int>("min_plane_points", 100);
    remaining_ratio_stop_ = declare_parameter<double>("remaining_ratio_stop", 0.5);

    publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, 10);
    debug_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(debug_rgb_topic_, 10);
    subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, 10,
      std::bind(&RansacNode::cloudCallback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "ransac_node ready");
  }

private:
  static std::array<std::uint8_t, 3> colorForPlaneId(int plane_id)
  {
    static const std::array<std::array<std::uint8_t, 3>, 50> kPalette = {{
      {{230, 25, 75}}, {{60, 180, 75}}, {{255, 225, 25}}, {{0, 130, 200}}, {{245, 130, 48}},
      {{145, 30, 180}}, {{70, 240, 240}}, {{240, 50, 230}}, {{210, 245, 60}}, {{250, 190, 190}},
      {{0, 128, 128}}, {{230, 190, 255}}, {{170, 110, 40}}, {{255, 250, 200}}, {{128, 0, 0}},
      {{170, 255, 195}}, {{128, 128, 0}}, {{255, 215, 180}}, {{0, 0, 128}}, {{128, 128, 128}},
      {{255, 99, 71}}, {{124, 252, 0}}, {{255, 140, 0}}, {{30, 144, 255}}, {{218, 112, 214}},
      {{154, 205, 50}}, {{255, 182, 193}}, {{32, 178, 170}}, {{255, 160, 122}}, {{95, 158, 160}},
      {{127, 255, 212}}, {{216, 191, 216}}, {{173, 255, 47}}, {{255, 105, 180}}, {{72, 209, 204}},
      {{176, 196, 222}}, {{238, 130, 238}}, {{0, 191, 255}}, {{255, 127, 80}}, {{100, 149, 237}},
      {{152, 251, 152}}, {{135, 206, 250}}, {{221, 160, 221}}, {{46, 139, 87}}, {{244, 164, 96}},
      {{123, 104, 238}}, {{60, 179, 113}}, {{255, 69, 0}}, {{0, 206, 209}}, {{199, 21, 133}}
    }};
    return kPalette[static_cast<std::size_t>((plane_id - 1) % static_cast<int>(kPalette.size()))];
  }

  std::uint32_t encodePlaneLabel(std::uint32_t input_label, int local_plane_id) const
  {
    if (!segment_per_input_label_) {
      return static_cast<std::uint32_t>(local_plane_id);
    }

    const std::uint32_t stride = plane_label_stride_ > 0
      ? static_cast<std::uint32_t>(plane_label_stride_)
      : 1000u;
    return input_label * stride + static_cast<std::uint32_t>(local_plane_id);
  }

  sensor_msgs::msg::PointCloud2 toDebugRos(
    const pcl::PointCloud<PointT> & cloud,
    const std_msgs::msg::Header & header) const
  {
    pcl::PointCloud<DebugPointT> debug_cloud;
    debug_cloud.reserve(cloud.size());
    for (const auto & p : cloud.points) {
      DebugPointT q;
      q.x = p.x;
      q.y = p.y;
      q.z = p.z;
      const auto color = colorForPlaneId(static_cast<int>(p.label));
      q.r = color[0];
      q.g = color[1];
      q.b = color[2];
      debug_cloud.push_back(q);
    }

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(debug_cloud, msg);
    msg.header = header;
    return msg;
  }

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    pcl::PointCloud<PointT>::Ptr input(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*msg, *input);

    pcl::PointCloud<PointT> labeled_planes;

    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(max_iterations_);
    seg.setDistanceThreshold(distance_threshold_);

    pcl::ExtractIndices<PointT> extract;

    auto segment_group =
      [&](const pcl::PointCloud<PointT>::Ptr & source_cloud, std::uint32_t input_label)
      {
        if (source_cloud->empty()) {
          return;
        }

        const std::size_t original_size = source_cloud->size();
        pcl::PointCloud<PointT>::Ptr remaining(new pcl::PointCloud<PointT>(*source_cloud));
        int local_plane_id = 1;

        while (!remaining->empty()) {
          const double remaining_ratio =
            static_cast<double>(remaining->size()) / static_cast<double>(original_size);
          if (remaining_ratio < remaining_ratio_stop_) {
            break;
          }

          pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
          pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
          seg.setInputCloud(remaining);
          seg.segment(*inliers, *coefficients);

          if (inliers->indices.empty() || static_cast<int>(inliers->indices.size()) < min_plane_points_) {
            break;
          }

          pcl::PointCloud<PointT>::Ptr plane_cloud(new pcl::PointCloud<PointT>());
          extract.setInputCloud(remaining);
          extract.setIndices(inliers);
          extract.setNegative(false);
          extract.filter(*plane_cloud);

          for (auto & p : plane_cloud->points) {
            p.label = encodePlaneLabel(input_label, local_plane_id);
            labeled_planes.push_back(p);
          }

          pcl::PointCloud<PointT>::Ptr next_remaining(new pcl::PointCloud<PointT>());
          extract.setNegative(true);
          extract.filter(*next_remaining);
          remaining = next_remaining;
          ++local_plane_id;
        }
      };

    if (segment_per_input_label_) {
      std::map<std::uint32_t, pcl::PointCloud<PointT>::Ptr> groups;
      for (const auto & p : input->points) {
        auto & group = groups[p.label];
        if (!group) {
          group.reset(new pcl::PointCloud<PointT>());
        }
        group->push_back(p);
      }

      if (groups.empty()) {
        RCLCPP_WARN(get_logger(), "No labeled clusters available for per-cluster RANSAC");
        return;
      }

      for (const auto & [cluster_id, group_cloud] : groups) {
        if (selected_room_id_ >= 0 && static_cast<int>(cluster_id) != selected_room_id_) {
          continue;
        }
        RCLCPP_INFO(get_logger(), "Running RANSAC on cluster_id=%u with %zu points", cluster_id, group_cloud->size());
        segment_group(group_cloud, cluster_id);
      }
    } else {
      pcl::PointCloud<PointT>::Ptr wall_points(new pcl::PointCloud<PointT>());
      wall_points->reserve(input->size());
      for (const auto & p : input->points) {
        if (static_cast<int>(p.label) == target_semantic_label_) {
          wall_points->push_back(p);
        }
      }

      if (wall_points->empty()) {
        RCLCPP_WARN(get_logger(), "No points with target_semantic_label=%d", target_semantic_label_);
        return;
      }

      segment_group(wall_points, 0u);
    }

    sensor_msgs::msg::PointCloud2 out;
    pcl::toROSMsg(labeled_planes, out);
    out.header = msg->header;
    publisher_->publish(out);
    if (publish_debug_rgb_) {
      debug_publisher_->publish(toDebugRos(labeled_planes, msg->header));
    }
  }

  std::string input_topic_;
  std::string output_topic_;
  std::string debug_rgb_topic_;
  int target_semantic_label_;
  int max_iterations_;
  double distance_threshold_;
  int min_plane_points_;
  double remaining_ratio_stop_;
  bool publish_debug_rgb_;
  bool segment_per_input_label_;
  int selected_room_id_;
  int plane_label_stride_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_publisher_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RansacNode>());
  rclcpp::shutdown();
  return 0;
}
