#include <memory>
#include <string>
#include <vector>

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "interior_wall_extractor/srv/process_point_cloud.hpp"

class PreprocessNode : public rclcpp::Node
{
public:
  using ProcessSrv = interior_wall_extractor::srv::ProcessPointCloud;
  using PointT = pcl::PointXYZ;

  PreprocessNode() : Node("preprocess_node")
  {
    voxel_leaf_size_ = declare_parameter<double>("voxel_leaf_size", 0.05);
    sor_mean_k_ = declare_parameter<int>("sor_mean_k", 30);
    sor_stddev_mul_ = declare_parameter<double>("sor_stddev_mul", 1.0);
    input_topic_ = declare_parameter<std::string>("input_topic", "~/input_cloud");
    output_topic_ = declare_parameter<std::string>("output_topic", "~/filtered_cloud");
    service_name_ = declare_parameter<std::string>("process_service", "~/process_cloud");

    publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, 10);
    subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, 10,
      std::bind(&PreprocessNode::cloudCallback, this, std::placeholders::_1));
    service_ = create_service<ProcessSrv>(
      service_name_,
      std::bind(&PreprocessNode::serviceCallback, this, std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(get_logger(), "preprocess_node ready");
  }

private:
  pcl::PointCloud<PointT>::Ptr rosToPcl(const sensor_msgs::msg::PointCloud2 & msg) const
  {
    auto cloud = std::make_shared<pcl::PointCloud<PointT>>();
    pcl::fromROSMsg(msg, *cloud);
    return cloud;
  }

  sensor_msgs::msg::PointCloud2 pclToRos(
    const pcl::PointCloud<PointT>::Ptr & cloud,
    const std_msgs::msg::Header & header) const
  {
    sensor_msgs::msg::PointCloud2 out;
    pcl::toROSMsg(*cloud, out);
    out.header = header;
    return out;
  }

  pcl::PointCloud<PointT>::Ptr preprocess(const pcl::PointCloud<PointT>::Ptr & input) const
  {
    auto voxelized = std::make_shared<pcl::PointCloud<PointT>>();
    pcl::VoxelGrid<PointT> voxel_grid;
    voxel_grid.setInputCloud(input);
    voxel_grid.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
    voxel_grid.filter(*voxelized);

    auto filtered = std::make_shared<pcl::PointCloud<PointT>>();
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(voxelized);
    sor.setMeanK(sor_mean_k_);
    sor.setStddevMulThresh(sor_stddev_mul_);
    sor.filter(*filtered);
    return filtered;
  }

  void publishProcessed(const sensor_msgs::msg::PointCloud2 & msg)
  {
    auto cloud = rosToPcl(msg);
    auto filtered = preprocess(cloud);
    publisher_->publish(pclToRos(filtered, msg.header));
  }

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    publishProcessed(*msg);
  }

  void serviceCallback(
    const std::shared_ptr<ProcessSrv::Request> request,
    std::shared_ptr<ProcessSrv::Response> response)
  {
    (void)request->save_debug_outputs;
    (void)request->request_id;
    try {
      publishProcessed(request->cloud);
      response->success = true;
      response->message = "Preprocessing completed and published.";
    } catch (const std::exception & ex) {
      response->success = false;
      response->message = ex.what();
      RCLCPP_ERROR(get_logger(), "Preprocess service failed: %s", ex.what());
    }
  }

  double voxel_leaf_size_;
  int sor_mean_k_;
  double sor_stddev_mul_;
  std::string input_topic_;
  std::string output_topic_;
  std::string service_name_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Service<ProcessSrv>::SharedPtr service_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PreprocessNode>());
  rclcpp::shutdown();
  return 0;
}
