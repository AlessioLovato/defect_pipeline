#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

class JsonExportNode : public rclcpp::Node
{
public:
  using PointT = pcl::PointXYZL;

  JsonExportNode() : Node("json_export_node")
  {
    input_topic_ = declare_parameter<std::string>("input_topic", "~/input_cloud");
    output_directory_ = declare_parameter<std::string>("output_directory", "");
    file_prefix_ = declare_parameter<std::string>("file_prefix", "wall_planes");
    export_on_receive_ = declare_parameter<bool>("export_on_receive", true);

    if (output_directory_.empty()) {
      output_directory_ = std::filesystem::current_path().string();
    }

    subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, 10,
      std::bind(&JsonExportNode::cloudCallback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "json_export_node ready. Writing to %s", output_directory_.c_str());
  }

private:
  std::string nowStamp() const
  {
    const auto now = std::chrono::system_clock::now();
    const auto secs = std::chrono::time_point_cast<std::chrono::seconds>(now);
    const auto value = secs.time_since_epoch().count();
    return std::to_string(value);
  }

  void writeJson(const pcl::PointCloud<PointT>::Ptr & cloud, const std_msgs::msg::Header & header)
  {
    std::map<std::uint32_t, std::vector<PointT>> grouped;
    for (const auto & p : cloud->points) {
      grouped[p.label].push_back(p);
    }

    std::filesystem::create_directories(output_directory_);
    const auto path = std::filesystem::path(output_directory_) /
      (file_prefix_ + std::string("_") + nowStamp() + ".json");
    std::ofstream out(path);
    if (!out.is_open()) {
      throw std::runtime_error("Could not open JSON output file: " + path.string());
    }

    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"frame_id\": \"" << header.frame_id << "\",\n";
    out << "  \"stamp_sec\": " << header.stamp.sec << ",\n";
    out << "  \"stamp_nanosec\": " << header.stamp.nanosec << ",\n";
    out << "  \"planes\": [\n";

    bool first_plane = true;
    for (const auto & [plane_id, points] : grouped) {
      if (!first_plane) {
        out << ",\n";
      }
      first_plane = false;
      out << "    {\n";
      out << "      \"plane_id\": " << plane_id << ",\n";
      out << "      \"num_points\": " << points.size() << ",\n";
      out << "      \"points\": [\n";

      for (std::size_t i = 0; i < points.size(); ++i) {
        const auto & p = points[i];
        out << "        {\"x\": " << p.x << ", \"y\": " << p.y << ", \"z\": " << p.z << "}";
        if (i + 1 != points.size()) {
          out << ",";
        }
        out << "\n";
      }
      out << "      ]\n";
      out << "    }";
    }

    out << "\n  ]\n";
    out << "}\n";
    RCLCPP_INFO(get_logger(), "Exported %zu plane groups to %s", grouped.size(), path.c_str());
  }

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (!export_on_receive_) {
      return;
    }
    pcl::PointCloud<PointT>::Ptr input(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*msg, *input);
    writeJson(input, msg->header);
  }

  std::string input_topic_;
  std::string output_directory_;
  std::string file_prefix_;
  bool export_on_receive_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<JsonExportNode>());
  rclcpp::shutdown();
  return 0;
}
