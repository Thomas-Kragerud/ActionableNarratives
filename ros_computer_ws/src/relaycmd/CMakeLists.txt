cmake_minimum_required(VERSION 3.0.2)
project(relaycmd)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  tf
  actionlib
  move_base_msgs
  std_srvs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

###
# You're right, the catkin_install_python command is not strictly necessary for your package to function. It is used to install Python scripts from your package to the system's bin directory, making them accessible from anywhere.
#If you have another package that doesn't include the catkin_install_python command in its CMakeLists.txt file, it means that the Python scripts in that package are not being installed system-wide. However, you can still run those scripts using rosrun as long as you have sourced the setup file for your workspace.
#In your case, if you don't need to install the relay_node.py script system-wide, you can remove the catkin_install_python command from your CMakeLists.txt file.


# catkin_install_python(PROGRAMS
#   scripts/relay_node.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )