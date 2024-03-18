#include "eye_tracking/GazeTracking.h"

GazeTracking::GazeTracking() {
}

GazeTracking::~GazeTracking() {
}

bool GazeTracking::set_up(dlib::frontal_face_detector face_detector, dlib::shape_predictor predictor) {
    this->face_detector_ = face_detector;
    this->predictor_ = predictor;
    return true;
}

bool GazeTracking::run(cv::Mat frame) {
    cv::cvtColor(frame, this->frame_, cv::COLOR_BGR2GRAY);

    // TODO: mettere gli altri detect
    if(detect_face()){
        if(detect_eyes()){
          if(detect_pupil()){
            this->eye_centers_[0].x += this->eyes_box_[0].x;
            this->eye_centers_[0].y += this->eyes_box_[0].y;
            this->eye_centers_[1].x += this->eyes_box_[1].x;
            this->eye_centers_[1].y += this->eyes_box_[1].y;

            cv::circle(this->face_frame_, this->eye_centers_[0], 3, cv::Scalar(0, 255, 0), 2);
            cv::circle(this->face_frame_, this->eye_centers_[1], 3, cv::Scalar(0, 255, 0), 2);
            cv::imshow("Face with eyes", this->face_frame_);
            cv::waitKey(1);
            return true;
          }
        }
    }
    return false;
}

bool GazeTracking::detect_face() {
    // transform to gray scale to detect faces
    std::vector<dlib::rectangle> faces = this->face_detector_(dlib::cv_image<unsigned char>(this->frame_));

    // take only the first face
    if(faces.size() >= 1){
      // face found and save the image with only the face
      this->face_box_ = faces[0];
      cv::Rect rect(face_box_.left(), face_box_.top(), face_box_.width(), face_box_.height());
      this->face_frame_ = this->frame_(rect).clone();
      
      return true;
    }else{
      this->face_box_ = dlib::rectangle(0, 0, 0, 0);
      this->face_frame_ = cv::Mat();

      return false;
    }
}

bool GazeTracking::detect_eyes(){
  int eye_region_width = this->face_box_.width() * 0.30;
  int eye_region_height = this->face_box_.width() * 0.25;
  int eye_region_top = this->face_box_.height() * 0.20;

  cv::Rect left_eye_region(this->face_box_.width() * 0.15, eye_region_top, eye_region_width, eye_region_height);
  cv::Rect right_eye_region(this->face_box_.width() - eye_region_width - this->face_box_.width() * 0.15, 
                            eye_region_top, eye_region_width, eye_region_height);
  
  this->eyes_box_ = {left_eye_region, right_eye_region};

  /* debug to see where are the box eyes with respect to the face
  cv::Mat tmp;
  this->face_frame_.copyTo(tmp);

  cv::rectangle(tmp, left_eye_region, cv::Scalar(0, 255, 0), 2);
  cv::rectangle(tmp, right_eye_region, cv::Scalar(0, 255, 0), 2);

  cv::imshow("boh", tmp);*/

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

double GazeTracking::computeDynamicThreshold(const cv::Mat &mat, double stdDevFactor) {
  cv::Scalar stdMagnGrad, meanMagnGrad;
  cv::meanStdDev(mat, meanMagnGrad, stdMagnGrad);
  double stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
  return stdDevFactor * stdDev + meanMagnGrad[0];
}

bool inMat(cv::Point p,int rows,int cols) {
  return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
}

bool floodShouldPushPoint(const cv::Point &np, const cv::Mat &mat) {
  return inMat(np, mat.rows, mat.cols);
}

cv::Mat floodKillEdges(cv::Mat &mat) {
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
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x - 1; np.y = p.y; // left
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x; np.y = p.y + 1; // down
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x; np.y = p.y - 1; // up
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    // kill it
    mat.at<float>(p) = 0.0f;
    mask.at<uchar>(p) = 0;
  }
  return mask;
}

void testPossibleCentersFormula(int x, int y, const cv::Mat &weight,double gx, double gy, cv::Mat &out) {
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

bool GazeTracking::detect_pupil() { 
  this->eye_centers_ = {};
  for(int side = 0; side < this->eyes_box_.capacity(); side++){
    // take both eyes from boxes and face_frame_
    cv::Mat eye = this->face_frame_(this->eyes_box_[side]);
    cv::resize(eye, eye, cv::Size(50,(((float)50)/eye.cols) * eye.rows));

    // Find the gradient -> implemented to be similar to the matlab code and in the paper of Timm and Barth
    cv::Mat gradientX = gradient_as_matlab(eye);
    cv::Mat gradientY = gradient_as_matlab(eye.t()).t();

    // normalize and threshold the gradient
    cv::Mat mags = compute_magnitude(gradientX, gradientY);

    //compute the threshold
    double gradientThresh = computeDynamicThreshold(mags, 50);

    // normalize
    for (int y = 0; y < eye.rows; ++y) {
      double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
      const double *Mr = mags.ptr<double>(y);

      for (int x = 0; x < eye.cols; ++x) {
        double gX = Xr[x], gY = Yr[x];
        double magnitude = Mr[x];
        if (magnitude > gradientThresh) {
          Xr[x] = gX/magnitude;
          Yr[x] = gY/magnitude;
        } else {
          Xr[x] = 0.0;
          Yr[x] = 0.0;
        }
      }
    }

    //-- Create a blurred and inverted image for weighting
    cv::Mat weight;
    GaussianBlur(eye, weight, cv::Size( 5, 5 ), 0, 0 );
    for (int y = 0; y < weight.rows; ++y) {
      unsigned char *row = weight.ptr<unsigned char>(y);
      for (int x = 0; x < weight.cols; ++x) {
        row[x] = (255 - row[x]);
      }
    }

    //-- Run the algorithm!
    cv::Mat outSum = cv::Mat::zeros(eye.rows,eye.cols,CV_64F);
    // for each possible gradient location
    // Note: these loops are reversed from the way the paper does them
    // it evaluates every possible center for each gradient location instead of
    // every possible gradient location for every center.
    for (int y = 0; y < weight.rows; ++y) {
      const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
      for (int x = 0; x < weight.cols; ++x) {
        double gX = Xr[x], gY = Yr[x];
        if (gX == 0.0 && gY == 0.0) {
          continue;
        }
        testPossibleCentersFormula(x, y, weight, gX, gY, outSum);
      }
    }
    // scale all the values down, basically averaging them
    double numGradients = (weight.rows*weight.cols);
    cv::Mat out;
    outSum.convertTo(out, CV_32F,1.0/numGradients);

    //-- Find the maximum point
    cv::Point maxP;
    double maxVal;
    cv::minMaxLoc(out, NULL,&maxVal,NULL,&maxP);
    //-- Flood fill the edges
    if(true) {
      cv::Mat floodClone;
      //double floodThresh = computeDynamicThreshold(out, 1.5);
      double floodThresh = maxVal * 0.97;
      cv::threshold(out, floodClone, floodThresh, 0.0f, cv::THRESH_TOZERO);

      cv::Mat mask = floodKillEdges(floodClone);
      // redo max
      cv::minMaxLoc(out, NULL,&maxVal,NULL,&maxP,mask);
    }

    // unscaled the point
    float ratio = (((float)50)/this->eyes_box_[side].width);
    int x = round(maxP.x / ratio);
    int y = round(maxP.y / ratio);
    this->eye_centers_.push_back(cv::Point(x,y));
  } 

  if(this->eye_centers_.size() != 2){
    return false;
  }else{
    return true;
  }
}






