# robohead_project

Project for hackerspace MakeITLab

Task: find methods to detect faces of human-beings and algorithms for gesture recognition. I made research on detection precision of different methods. It turned out that different methods have different results, the problem is solved with big approximation in all cases. It has been observed that the camera resolution and lighting highly affect the face location detection in a frame: in case of bad lighting face detection methods are not efficient. 

Regarding algorithms for gesture detection using convex hull allows to find contours of fingers although it works well only when background is monotonous without light patterns.

Stack: Python (OpenCV)
Hardware: Arduino, USB camera
