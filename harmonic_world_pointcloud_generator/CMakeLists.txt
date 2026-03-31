cmake_minimum_required(VERSION 3.16)
project(harmonic_world_pointcloud_generator)

####################
# NOTE: dummy CMakeLists for forest compatibility. Package if fully python3
####################

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/generator_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
