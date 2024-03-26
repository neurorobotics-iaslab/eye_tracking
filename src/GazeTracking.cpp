#ifndef EYE_TRACKING_GAZETRACKING1_CPP_
#define EYE_TRACKING_GAZETRACKING1_CPP_

#include "eye_tracking/GazeTracking.h"

GazeTracking::GazeTracking() : p_nh_("~"){
    this->pub_ = this->nh_.advertise<eye_tracking::pupil>("/cvsa/pupils", 1);
}

GazeTracking::~GazeTracking() {
}

bool GazeTracking::configure() {
  
    std::string model_path;
    if(this->p_nh_.getParam("model_path", model_path) == false) {
        ROS_ERROR("Parameter 'model_path' is mandatory");
        return false;
    }
    this->face_detector_ = dlib::frontal_face_detector(dlib::get_frontal_face_detector());
    dlib::deserialize(model_path) >> this->predictor_;

    this->p_nh_.param("show_face_pupil_detect", this->show_face_pupil_detect_, false);

    this->p_nh_.param("show_camera", this->show_camera_, false);

    this->p_nh_.param("rate", this->rate_, 256);

    this->p_nh_.param("camera_open", this->camera_open_, 0);

    return true;
}

void GazeTracking::run() {


    cv::VideoCapture cap;
    cap.open(this->camera_open_);

    ros::Rate r(this->rate_);

    while (ros::ok()) {
        cap >> this->frame_;
        
        if (this->frame_.empty()) {
            ROS_ERROR("Camera frame is empty");
            break;
        }

        // show the frame
        if(this->show_camera_){
            cv::imshow("frame", this->frame_);
            cv::waitKey(1);
        }
        
        // convert to gray scale
        cv::cvtColor(this->frame_, this->frame_, cv::COLOR_BGR2GRAY);

        if(detect_face()){
            if(detect_eyes()){
                if(detect_pupil()){
                    if(this->show_face_pupil_detect_){
                        show();
                    }
                    //publish message
                    publish_msgs();
                }else{
                    //ROS_WARN("Pupils not detected");
                }
            }else{
                //ROS_WARN("Eyes not detected");
            }
        }else{
            //ROS_WARN("Face not detected");
        }
        r.sleep();
        ros::spinOnce();
    }
}

bool GazeTracking::detect_face() {
    // transform to gray scale to detect faces
    std::vector<dlib::rectangle> faces = this->face_detector_(dlib::cv_image<unsigned char>(this->frame_));

    // take only the first face
    if(faces.size() >= 1){
        // face found and save the image with only the face
        this->face_box_ = faces[0];
        cv::Rect rect(face_box_.left(), face_box_.top(), face_box_.width(), face_box_.height());
        try{
            this->face_frame_ = this->frame_(rect).clone();
            return true;
        }
        catch(...){
            ROS_WARN("Error in cropping the face");
            return false;
        }

        return true;
    }

    return false;
}

bool GazeTracking::detect_eyes(){

    dlib::full_object_detection landmarks = this->predictor_(dlib::cv_image<unsigned char>(this->frame_), this->face_box_);
    std::vector<cv::Point> landmarks_points_left, landmarks_points_right;
    for(int i = 0; i < left_eye_region_landmarks.size(); i++){
        landmarks_points_left.push_back(cv::Point(landmarks.part(left_eye_region_landmarks[i]).x(), landmarks.part(left_eye_region_landmarks[i]).y()));
    }
    for(int i = 0; i < right_eye_region_landmarks.size(); i++){
        landmarks_points_right.push_back(cv::Point(landmarks.part(right_eye_region_landmarks[i]).x(), landmarks.part(right_eye_region_landmarks[i]).y()));
    }

    cv::Rect left_eye_region(landmarks_points_left[0].x - this->face_box_.tl_corner().x(), landmarks_points_left[1].y - this->face_box_.tl_corner().y(), 
                             landmarks_points_left[2].x - landmarks_points_left[0].x, landmarks_points_left[2].y - landmarks_points_left[1].y);
    cv::Rect right_eye_region(landmarks_points_right[2].x  - this->face_box_.tl_corner().x(), landmarks_points_right[1].y - this->face_box_.tl_corner().y(),
                             landmarks_points_right[0].x - landmarks_points_right[2].x, landmarks_points_right[2].y - landmarks_points_right[1].y);

    this->eyes_box_ = {left_eye_region, right_eye_region};
    
    /*
    // works -> position of the eyes zones are hardcoded
    int eye_region_width = this->face_box_.width() * 0.30;
    int eye_region_height = this->face_box_.width() * 0.25;
    int eye_region_top = this->face_box_.height() * 0.20;

    cv::Rect left_eye_region(this->face_box_.width() * 0.15, eye_region_top, eye_region_width, eye_region_height);
    cv::Rect right_eye_region(this->face_box_.width() - eye_region_width - this->face_box_.width() * 0.15, 
                              eye_region_top, eye_region_width, eye_region_height);

    */

    return true;
}

cv::Mat GazeTracking::gradient_as_matlab(const cv::Mat &mat) {
    cv::Mat out(mat.rows,mat.cols,CV_64F);
  
    for (int y = 0; y < mat.rows; ++y) {
        const uchar *Mr = mat.ptr<uchar>(y);
        double *Or = out.ptr<double>(y);

        Or[0] = Mr[1] - Mr[0];
        for (int x = 1; x < mat.cols - 1; ++x) {
            Or[x] = (Mr[x+1] - Mr[x-1])/2.0;
        }
        Or[mat.cols-1] = Mr[mat.cols-1] - Mr[mat.cols-2];
    }
    
    return out;
}

cv::Mat GazeTracking::compute_magnitude(const cv::Mat &matX, const cv::Mat &matY) {
    cv::Mat magnitude;
    magnitude = matX.mul(matX) + matY.mul(matY);
    sqrt(magnitude, magnitude);
    return magnitude;
}

double GazeTracking::compute_dynamic_threshold(const cv::Mat &mat, double stdDevFactor) {
    cv::Scalar stdMagnGrad, meanMagnGrad;
    cv::meanStdDev(mat, meanMagnGrad, stdMagnGrad);
    double stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
    return stdDevFactor * stdDev + meanMagnGrad[0];
}

cv::Mat GazeTracking::flood_kill_edges(cv::Mat &mat) {
    rectangle(mat,cv::Rect(0,0,mat.cols,mat.rows),255);
    
    cv::Mat mask(mat.rows, mat.cols, CV_8U, 255);
    std::queue<cv::Point> toDo;
    toDo.push(cv::Point(0,0));
    while (!toDo.empty()) {
        cv::Point p = toDo.front();
        toDo.pop();
        if (mat.at<float>(p) == 0.0f) {
            continue;
        }
        // add in every direction
        cv::Point np(p.x + 1, p.y); // right
        if (np.x >= 0 && np.x < mat.cols && np.y >= 0 && np.y < mat.rows) toDo.push(np);
        np.x = p.x - 1; np.y = p.y; // left
        if (np.x >= 0 && np.x < mat.cols && np.y >= 0 && np.y < mat.rows) toDo.push(np);
        np.x = p.x; np.y = p.y + 1; // down
        if (np.x >= 0 && np.x < mat.cols && np.y >= 0 && np.y < mat.rows) toDo.push(np);
        np.x = p.x; np.y = p.y - 1; // up
        if (np.x >= 0 && np.x < mat.cols && np.y >= 0 && np.y < mat.rows) toDo.push(np);
        // kill it
        mat.at<float>(p) = 0.0f;
        mask.at<uchar>(p) = 0;
    }
    return mask;
}

void GazeTracking::test_centers(int x, int y, const cv::Mat &weight,double gx, double gy, cv::Mat &out){
    // for all possible centers
    for (int cy = 0; cy < out.rows; ++cy) {
        double *Or = out.ptr<double>(cy);
        const unsigned char *Wr = weight.ptr<unsigned char>(cy);
        for (int cx = 0; cx < out.cols; ++cx) {
            if (x == cx && y == cy) {
                continue;
            }
            // create a vector from the possible center to the gradient origin
            double dx = x - cx;
            double dy = y - cy;
            // normalize d
            double magnitude = sqrt((dx * dx) + (dy * dy));
            dx = dx / magnitude;
            dy = dy / magnitude;
            double dotProduct = dx*gx + dy*gy;
            dotProduct = std::max(0.0,dotProduct);
            // square and multiply by the weight
            if (true) {
                Or[cx] += dotProduct * dotProduct * (Wr[cx]/1.0f);
            } else {
                Or[cx] += dotProduct * dotProduct;
            }
        }
    }
}

cv::Mat GazeTracking::normalize_gradient(cv::Mat &gradient, cv::Mat &magnitude){
    // get the threshold
    double gradientThresh = compute_dynamic_threshold(magnitude, 50);

    // normalize
    for (int y = 0; y < gradient.rows; ++y) {
        double *Xr = gradient.ptr<double>(y);
        const double *Mr = magnitude.ptr<double>(y);
        for (int x = 0; x < gradient.cols; ++x) {
            double gX = Xr[x];
            double mag = Mr[x];
            if (mag > gradientThresh) {
                Xr[x] = gX/mag;
            } else {
                Xr[x] = 0.0;
            }
        }
    }

    return gradient;
}

cv::Point GazeTracking::algorithm_Timm_Barth(cv::Mat &gradientX, cv::Mat &gradientY, cv::Mat &weight) {
    cv::Mat outSum = cv::Mat::zeros(gradientX.rows, gradientX.cols,CV_64F);
    // it evaluates every possible center for each gradient location
    for (int y = 0; y < weight.rows; ++y) {
        const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
        for (int x = 0; x < weight.cols; ++x) {
            double gX = Xr[x], gY = Yr[x];
            if (gX == 0.0 && gY == 0.0) {
                continue;
            }
            test_centers(x, y, weight, gX, gY, outSum);
        }
    }
    // scale all the values down, basically averaging them
    double numGradients = (weight.rows*weight.cols);
    cv::Mat out;
    outSum.convertTo(out, CV_32F,1.0/numGradients);

    // find the maximum point
    cv::Point maxP;
    double maxVal;
    cv::minMaxLoc(out, NULL,&maxVal,NULL,&maxP);
    // flood fill the edges
    if(true) {
        cv::Mat floodClone;
        double floodThresh = maxVal * 0.97;
        cv::threshold(out, floodClone, floodThresh, 0.0f, cv::THRESH_TOZERO);

        cv::Mat mask = flood_kill_edges(floodClone);
        cv::minMaxLoc(out, NULL,&maxVal,NULL,&maxP,mask);
    }

    return maxP;
}

bool GazeTracking::detect_pupil() { 
    this->eye_centers_ = {};
    for(int side = 0; side < this->eyes_box_.capacity(); side++){
        // take both eyes from boxes and face_frame_
        try{
            cv::Mat eye = this->face_frame_(this->eyes_box_[side]);
            cv::resize(eye, eye, cv::Size(50,(((float)50)/eye.cols) * eye.rows));

            // Find the gradient -> implemented to be similar to the matlab code and in the paper of Timm and Barth
            cv::Mat gradientX = gradient_as_matlab(eye);
            cv::Mat gradientY = gradient_as_matlab(eye.t()).t();

            // normalize and threshold the gradient
            cv::Mat mags = compute_magnitude(gradientX, gradientY);

            gradientX = normalize_gradient(gradientX, mags);
            gradientY = normalize_gradient(gradientY, mags);

            // create a blurred and inverted image for weighting
            cv::Mat weight;
            GaussianBlur(eye, weight, cv::Size( 5, 5 ), 0, 0 );
            for (int y = 0; y < weight.rows; ++y) {
                unsigned char *row = weight.ptr<unsigned char>(y);
                for (int x = 0; x < weight.cols; ++x) {
                    row[x] = (255 - row[x]);
                }
            }

            // run Timm and Barth algorithm
            cv::Point maxP = algorithm_Timm_Barth(gradientX, gradientY, weight);

            // unscaled the point
            float ratio = (((float)50)/this->eyes_box_[side].width);
            int x = round(maxP.x / ratio);
            int y = round(maxP.y / ratio);
            this->eye_centers_.push_back(cv::Point(x,y));
        }catch(...){
            ROS_WARN("Error in cropping the eyes");
            return false;
        } 
    } 

   
    // check if both pupils are detected
    if(this->eye_centers_.size() != 2){
        ROS_WARN("Error not both eyes are computed");
        return false;
    }
    
    // update according to the eye_box which is taken from the face_frame_
    this->eye_centers_[0].x += this->eyes_box_[0].x;
    this->eye_centers_[0].y += this->eyes_box_[0].y;
    this->eye_centers_[1].x += this->eyes_box_[1].x;
    this->eye_centers_[1].y += this->eyes_box_[1].y;
    return true;
}

void GazeTracking::show(){
    cv::circle(this->face_frame_, this->eye_centers_[0], 3, cv::Scalar(0, 255, 0), 2);
    cv::circle(this->face_frame_, this->eye_centers_[1], 3, cv::Scalar(0, 255, 0), 2);
    cv::imshow("Face with eyes", this->face_frame_);
    cv::waitKey(1);
}

void GazeTracking::publish_msgs() {
    eye_tracking::pupil msg;

    // pupil information
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = "face_image";
    msg.header.seq = this->id++;

    msg.left_pupil.x = this->eye_centers_[0].x;
    msg.left_pupil.y = this->eye_centers_[0].y;
    msg.right_pupil.x = this->eye_centers_[1].x;
    msg.right_pupil.y = this->eye_centers_[1].y;

    // image information
    sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->face_frame_).toImageMsg();
    msg.face_image = *img_msg;
    msg.face_image.header.stamp = msg.header.stamp;
    msg.face_image.header.frame_id = "face_image";
    msg.face_image.header.seq = this->id;

    this->pub_.publish(msg);
}

#endif  




