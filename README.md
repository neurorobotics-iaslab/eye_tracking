# eye_tracking

this package followed the paper: ''ACCURATE EYE CENTRE LOCALISATION BY MEANS OF GRADIENTS'' by F. Timm and E. Barth

# required
OpenCV 4.2
dlib -> used to detect the face
shape_predictor_68_face_landmarks.dat used to obrain the eyes zones

# parameters
model_path: path to shape_predictor_68_face_landmarks.dat
show_face_pupil_detect: true if you want to see the detected pupils (default: false)
show_camera: true if you want to see what camera sees (default: false)
rate: ros rate (default: 256)
camera_open: id of the camera to open (default: 0)
