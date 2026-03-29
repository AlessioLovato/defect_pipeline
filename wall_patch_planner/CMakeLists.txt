cmake_minimum_required(VERSION 3.10)
project(wall_patch_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(wall_patch_planner_node src/wall_patch_planner_node.cpp)
target_compile_features(wall_patch_planner_node PUBLIC cxx_std_17)
target_include_directories(wall_patch_planner_node PRIVATE ${EIGEN3_INCLUDE_DIRS})
ament_target_dependencies(wall_patch_planner_node
  rclcpp sensor_msgs geometry_msgs visualization_msgs std_srvs tf2 tf2_ros)

add_executable(mock_camera_info_pub src/mock_camera_info_pub.cpp)
target_compile_features(mock_camera_info_pub PUBLIC cxx_std_17)
ament_target_dependencies(mock_camera_info_pub
  rclcpp sensor_msgs)

install(TARGETS
  wall_patch_planner_node
  mock_camera_info_pub
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

install(PROGRAMS scripts/trigger_plan.sh
  DESTINATION share/${PROJECT_NAME}/scripts)

ament_package()
