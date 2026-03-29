cmake_minimum_required(VERSION 3.8)
project(interior_wall_extractor)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters segmentation search)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/ProcessPointCloud.srv"
  DEPENDENCIES sensor_msgs std_msgs
)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(preprocess_node src/preprocess_node.cpp)
ament_target_dependencies(preprocess_node rclcpp sensor_msgs pcl_conversions std_srvs)
target_link_libraries(preprocess_node ${PCL_LIBRARIES})
rosidl_target_interfaces(preprocess_node ${PROJECT_NAME} "rosidl_typesupport_cpp")

add_executable(ransac_node src/ransac_node.cpp)
ament_target_dependencies(ransac_node rclcpp sensor_msgs geometry_msgs pcl_conversions)
target_link_libraries(ransac_node ${PCL_LIBRARIES})

add_executable(json_export_node src/json_export_node.cpp)
ament_target_dependencies(json_export_node rclcpp sensor_msgs pcl_conversions ament_index_cpp)
target_link_libraries(json_export_node ${PCL_LIBRARIES})

install(TARGETS
  preprocess_node
  ransac_node
  json_export_node
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/dbscan_node.py
  # scripts/randla_bridge.py
  DESTINATION lib/${PROJECT_NAME}
)

# install(DIRECTORY scripts/third_party
#   DESTINATION lib/${PROJECT_NAME}
# )

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  package.xml
  DESTINATION share/${PROJECT_NAME}
)

# install(FILES
#   resource/${PROJECT_NAME}
#   DESTINATION share/ament_index/resource_index/packages
# )

ament_export_dependencies(rosidl_default_runtime)
ament_package()
