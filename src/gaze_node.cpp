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
    ros::init(argc, argv, "gaze_tracking_node");
    ros::NodeHandle nh;

    std::string model_path = "/home/paolo/rosneuro_ws/src/eye_cvsa/trained_models/shape_predictor_68_face_landmarks.dat";
    dlib::frontal_face_detector face_detector = dlib::frontal_face_detector(dlib::get_frontal_face_detector());
    dlib::shape_predictor predictor;
    dlib::deserialize(model_path) >> predictor;

    GazeTracking gaze_tracking;
    gaze_tracking.set_up(face_detector, predictor);

    cv::VideoCapture cap;
    cap.open(0);

    bool calibration = true;

    while (ros::ok()) {
        cv::Mat frame;
        cap >> frame;

        /*if(calibration){
            if (frame.empty()) {
                ROS_ERROR("Camera frame is empty");
                break;
            }
            Calibration* calib_left = new Calibration();
            Calibration* calib_right = new Calibration();
        }*/

        
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
