#include <ros/ros.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>

#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include "eye_tracking/GazeTracking.h"

int main(int argc, char** argv) {

    // ros initialization
    ros::init(argc, argv, "gaze_node");
    ros::NodeHandle nh;

    GazeTracking gaze_tracking;
    if(!gaze_tracking.configure()){
        ROS_ERROR("Error in gaze tracking configuration");
        ros::shutdown();
        return 0;
    }

    gaze_tracking.run();

    ros::shutdown();

    return 0;
    
    
}
