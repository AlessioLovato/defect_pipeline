cmake_minimum_required(VERSION 3.16)
project(defect_localization)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(rclcpp)
find_package(rcl_interfaces REQUIRED)
find_package(rclpy)
find_package(builtin_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(defect_map_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

find_package(OpenCV REQUIRED)


# Detectron2 installation with colcon build if not already pre-installed
option(
  DEFECT_LOCALIZATION_INSTALL_DETECTRON2
  "Install bundled detectron2 with pip during colcon build when missing"
  OFF
)
set(
  TORCH_VERSION
  "2.10"
  CACHE STRING
  "Torch version required before installing detectron2"
)
set(
  TORCHVISION_VERSION
  "0.25"
  CACHE STRING
  "Torch version required before installing detectron2"
)

# Set test option
option(
  BUILD_TESTING
  "Build and install tests"
  OFF
)

if(DEFECT_LOCALIZATION_INSTALL_DETECTRON2)
  set(DETECTRON2_BUNDLED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/defect_localization/detectron2")
  if(EXISTS "${DETECTRON2_BUNDLED_DIR}/setup.py")
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import detectron2"
      RESULT_VARIABLE DETECTRON2_IMPORT_RESULT
      OUTPUT_QUIET
      ERROR_QUIET
    )
    if(NOT DETECTRON2_IMPORT_RESULT EQUAL 0)
      message(STATUS "detectron2 not found. Installing from ${DETECTRON2_BUNDLED_DIR}")
      execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import sys; import torch; sys.exit(0 if torch.__version__.startswith('${TORCH_VERSION}') else 1)"
        RESULT_VARIABLE TORCH_VERSION_OK
        OUTPUT_QUIET
        ERROR_QUIET
      )
      if(NOT TORCH_VERSION_OK EQUAL 0)
        message(STATUS "Installing torch==${TORCH_VERSION} before detectron2")
        execute_process(
          COMMAND ${Python3_EXECUTABLE} -m pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
          RESULT_VARIABLE TORCH_INSTALL_RESULT
        )
        if(NOT TORCH_INSTALL_RESULT EQUAL 0)
          message(FATAL_ERROR "Failed to install torch==${TORCH_VERSION}")
        endif()
      endif()
      execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pip install -e "${DETECTRON2_BUNDLED_DIR}" --no-build-isolation
        RESULT_VARIABLE DETECTRON2_INSTALL_RESULT
      )
      if(NOT DETECTRON2_INSTALL_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to install detectron2 from bundled source")
      endif()
    else()
      message(STATUS "detectron2 already available in current Python environment")
    endif()
  else()
    message(WARNING "Bundled detectron2 source not found at ${DETECTRON2_BUNDLED_DIR}")
  endif()
endif()

ament_python_install_package(${PROJECT_NAME})

add_library(defect_localization_component SHARED
  src/defect_localization/defect_localization_node.cpp
)

add_library(defect_localization_mock_support STATIC
  src/defect_localization/mock_image_catalog.cpp
)

target_include_directories(defect_localization_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(defect_localization_mock_support PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(defect_localization_component
  ${OpenCV_LIBS}
)

ament_target_dependencies(defect_localization_component
  ament_index_cpp
  builtin_interfaces
  cv_bridge
  defect_map_interfaces
  geometry_msgs
  message_filters
  rclcpp
  rclcpp_components
  rcl_interfaces
  sensor_msgs
  std_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
)

ament_target_dependencies(defect_localization_mock_support
  ament_index_cpp
)

rclcpp_components_register_node(defect_localization_component
  PLUGIN "defect_localization::DefectLocalizationNode"
  EXECUTABLE defect_localization_component_node
)

add_executable(defect_localization_node
  src/defect_localization/defect_localization_main.cpp
)
target_link_libraries(defect_localization_node
  defect_localization_component
)
ament_target_dependencies(defect_localization_node
  rclcpp
)

add_executable(mock_realsense_node
  src/defect_localization/mock_realsense_main.cpp
)
target_link_libraries(mock_realsense_node
  defect_localization_mock_support
  ${OpenCV_LIBS}
)
ament_target_dependencies(mock_realsense_node
  cv_bridge
  geometry_msgs
  rclcpp
  rcl_interfaces
  sensor_msgs
  std_msgs
  tf2_ros
)

add_executable(mock_snapshot_orchestrator
  src/defect_localization/mock_snapshot_orchestrator.cpp
)
target_link_libraries(mock_snapshot_orchestrator
  defect_localization_mock_support
)
ament_target_dependencies(mock_snapshot_orchestrator
  defect_map_interfaces
  rclcpp
)

install(TARGETS
  defect_localization_component
  defect_localization_mock_support
  defect_localization_node
  mock_realsense_node
  mock_snapshot_orchestrator
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(PROGRAMS
  scripts/detectron2_predictor_node.py
  scripts/prediction_visualizer_node.py
  examples/request_prediction.py
  examples/capture_four_shots.sh
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY test/mock_images
  DESTINATION share/${PROJECT_NAME}/test
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
