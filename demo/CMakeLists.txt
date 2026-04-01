cmake_minimum_required(VERSION 3.16)
project(demo)

####################
# NOTE: dummy CMakeLists for forest compatibility. Package if fully python3
####################

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/orchestrator_node
  scripts/cartesio_notebook_client_node
  scripts/joint_command_mux_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
