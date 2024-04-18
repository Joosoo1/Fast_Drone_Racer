cmake_minimum_required(VERSION 3.0.2)
project(detect_node)

FIND_PACKAGE(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  message_filters
  bbox_ex_msgs
  cv_bridge
)

catkin_package(
 INCLUDE_DIRS include
#  LIBRARIES detect_node
#  CATKIN_DEPENDS roscpp std_msgs
#  DEPENDS system_lib
)

include_directories(3dparty/rknpu2/include/)
# include_directories(3dparty/rga/include/)

link_directories(3dparty/rknpu2/lib/)
# link_directories(3dparty/rga/libs/Linux/gcc-aarch64/)

include_directories(
  include
  include/detect_node
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

set(APP_NAME "detect_frame")

# 指定源代码路径
set(SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")
# 收集源文件
aux_source_directory(${SRC_PATH} CXX_SOURCES)
add_executable(${APP_NAME} ${CXX_SOURCES})
add_dependencies(${APP_NAME} bbox_ex_msgs_generate_messages_cpp)

target_link_libraries(${APP_NAME} 
                      ${OpenCV_LIBS} 
                      ${catkin_LIBRARIES} 
                      rknn_api 
                      # rga
                      )
