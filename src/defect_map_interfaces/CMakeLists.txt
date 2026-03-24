cmake_minimum_required(VERSION 3.16)
project(defect_map_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

set(msg_files
  "msg/DefectEntry.msg"
  "msg/SegmentedInstance.msg"
)

set(srv_files
  "srv/CaptureShot.srv"
  "srv/BuildMap.srv"
  "srv/GetDefectMap.srv"
  "srv/ClearDefectMap.srv"
  "srv/SegmentImage.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES builtin_interfaces geometry_msgs sensor_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
