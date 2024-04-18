cmake_minimum_required(VERSION 3.0.2)
project(pose_sub)


find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  std_msgs
  airsim_ros
  bbox_ex_msgs
)
find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES pose_sub
  CATKIN_DEPENDS geometry_msgs roscpp std_msgs airsim_ros bbox_ex_msgs
  DEPENDS system_lib

)


include_directories(
  include
  # ~/pose_sub/src/pose_sub
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  
)


 add_executable(global_circle_sub_node src/pose_sub.cpp)


 target_link_libraries(global_circle_sub_node
   ${catkin_LIBRARIES}
   Eigen3::Eigen
 )


