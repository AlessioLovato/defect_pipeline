cmake_minimum_required(VERSION 3.16)
project(defect_map_pipeline)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(rclcpp)
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
  DEFECT_MAP_PIPELINE_INSTALL_DETECTRON2
  "Install bundled detectron2 with pip during colcon build when missing"
  ON
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

if(DEFECT_MAP_PIPELINE_INSTALL_DETECTRON2)
  set(DETECTRON2_BUNDLED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/defect_map_pipeline/detectron2")
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

add_library(defect_map_pipeline_component SHARED
  src/defect_map_pipeline_node.cpp
)

target_include_directories(defect_map_pipeline_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(defect_map_pipeline_component
  ${OpenCV_LIBS}
)

ament_target_dependencies(defect_map_pipeline_component
  builtin_interfaces
  cv_bridge
  defect_map_interfaces
  geometry_msgs
  message_filters
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
)

rclcpp_components_register_node(defect_map_pipeline_component
  PLUGIN "defect_map_pipeline::DefectMapPipelineNode"
  EXECUTABLE defect_map_pipeline_node
)

install(TARGETS defect_map_pipeline_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(PROGRAMS
  scripts/detectron2_predictor_node.py
  scripts/prediction_visualizer_node.py
  examples/request_prediction.py
  examples/capture_four_shots.sh
  examples/build_and_get_map.sh
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
