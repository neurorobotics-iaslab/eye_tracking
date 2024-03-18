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

class GazeTracking{

private:
    cv::Mat frame_;
    cv::Mat face_frame_;
    dlib::frontal_face_detector face_detector_; 
    dlib::shape_predictor predictor_;
    dlib::rectangle face_box_;
    std::vector<cv::Rect> eyes_box_;
    std::vector<cv::Point> eye_centers_;

public:
    GazeTracking();
    ~GazeTracking();

    bool set_up(dlib::frontal_face_detector face_detector, dlib::shape_predictor predictor);
    bool run(cv::Mat frame);

private:
    bool detect_face();
    bool detect_eyes();
    bool detect_pupil();

    cv::Mat gradient_as_matlab(const cv::Mat &mat);
    cv::Mat compute_magnitude(const cv::Mat &matX, const cv::Mat &matY);
    
    
    
    double computeDynamicThreshold(const cv::Mat &mat, double stdDevFactor);

};

#endif