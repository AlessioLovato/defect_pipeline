#include <memory>

#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>

#include "defect_localization/defect_localization_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<defect_localization::DefectLocalizationNode>(rclcpp::NodeOptions{});
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions{}, 2U);
  executor.add_node(node);
  executor.spin();
  executor.remove_node(node);

  rclcpp::shutdown();
  return 0;
}
