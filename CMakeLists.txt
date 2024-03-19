cmake_minimum_required(VERSION 3.0)
project(eye_tracking)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
SET(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(catkin REQUIRED COMPONENTS 
             roscpp 
             std_msgs
             cv_bridge
             image_transport
             message_generation
             rosneuro_msgs
             sensor_msgs
             )

find_package(OpenCV REQUIRED)

find_package(dlib REQUIRED)

add_message_files(
  FILES
  pupil.msg
)

generate_messages(
  DEPENDENCIES
    geometry_msgs
    std_msgs
    sensor_msgs
)
             
catkin_package(
  INCLUDE_DIRS 
    include
  LIBRARIES 
    ${PROJECT_NAME}
  CATKIN_DEPENDS
    roscpp
    std_msgs
    message_runtime
    rosneuro_msgs
    sensor_msgs
  DEPENDS
)


###########
## Build ##
###########

#include(/home/paolo/Local/dlib/dlib/cmake)

#set(USE_AVX_INSTRUCTIONS 1)

include_directories(include 
                    ${catkin_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    )

add_library(${PROJECT_NAME}    
            src/GazeTracking.cpp
            )

target_link_libraries(${PROJECT_NAME} 
                      ${catkin_LIBRARIES}
                      ${OpenCV_LIBRARIES}
                      )

add_dependencies(${PROJECT_NAME} 
                 ${catkin_EXPORTED_TARGETS}
)

#################
## Executables ##
#################
add_executable(gaze_node src/gaze_node.cpp)
target_link_libraries(gaze_node  ${PROJECT_NAME}
                                 ${catkin_LIBRARIES}
                                 dlib::dlib
                   )
