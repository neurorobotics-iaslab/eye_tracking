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
    gaze_tracking.set_up();

    cv::VideoCapture cap;
    cap.open(0);

    bool calibration = true;

    while (ros::ok()) {
        cv::Mat frame;
        cap >> frame;
        
        if (frame.empty()) {
            ROS_ERROR("Camera frame is empty");
            break;
        }

        cv::imshow("Camera", frame);
        cv::waitKey(1);

        if(!gaze_tracking.run(frame)){
            ROS_WARN("Error in gaze tracking");
        }
        ros::spinOnce();

    }

    ros::shutdown();

    return 0;
    
    
}
