#ifndef EYE_TRACKING_GAZETRACKING1_H_
#define EYE_TRACKING_GAZETRACKING1_H_

#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <vector>
#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "cv_bridge/cv_bridge.h"

#include "eye_tracking/pupil.h"
#include <rosneuro_msgs/NeuroEvent.h>

class GazeTracking{

private:
    cv::Mat frame_;
    cv::Mat face_frame_;
    dlib::frontal_face_detector face_detector_; 
    dlib::shape_predictor predictor_;
    dlib::rectangle face_box_;
    std::vector<cv::Rect> eyes_box_;
    std::vector<cv::Point> eye_centers_;
    bool show_face_pupil_detect_;

    int id = 0;

    ros::NodeHandle nh_;
    ros::NodeHandle p_nh_;
    ros::Publisher pub_;

    eye_tracking::pupil msg_;

public:
    GazeTracking();
    ~GazeTracking();

    bool set_up();
    bool run(cv::Mat frame);

private:
    bool detect_face();
    bool detect_eyes();
    bool detect_pupil();

    cv::Mat gradient_as_matlab(const cv::Mat &mat);
    cv::Mat compute_magnitude(const cv::Mat &matX, const cv::Mat &matY);
    cv::Mat normalize_gradient(cv::Mat &gradient, cv::Mat &magnitude);
    double compute_dynamic_threshold(const cv::Mat &mat, double stdDevFactor);

    cv::Point algorithm_Timm_Barth(cv::Mat &gradientX, cv::Mat &gradientY, cv::Mat &weight);
    void test_centers(int x, int y, const cv::Mat &weight,double gx, double gy, cv::Mat &out);
    cv::Mat flood_kill_edges(cv::Mat &mat);
    void show();

    void publish_msgs();
};

#endif