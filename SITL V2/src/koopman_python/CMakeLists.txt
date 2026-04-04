cmake_minimum_required(VERSION 3.0.2)
project(koopman_python)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

install(DIRECTORY koopman_python
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)

