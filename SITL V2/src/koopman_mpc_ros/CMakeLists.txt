cmake_minimum_required(VERSION 3.0.2)
project(koopman_mpc_ros)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  mav_msgs
  nav_msgs
  rospy
  std_msgs
  trajectory_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/koopman_mpc_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

