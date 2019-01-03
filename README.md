# SimpleObjectDetectionAndTracking
Tool to try simple object detection and tracking methods as well as different parameters using OpenCV with Python and Tkinter

It was a good starting point to gain insight multiple object detection and tracking. To detect and track many vehicles and pedestrians in crowded scenes (or even with an moving cam) you will need more advanced methods using convolutional neural networks (CNN) for detection and tracking as well as tracking-by-detection approaches. See for example [here](https://github.com/mavoll/MotionPathsExtraction).

To run the tool:

* Install prerequisites and run the python script (simple_detection_and_tracking.py), or
* just run the executable file (.exe file for windows; .app for mac will follow) 

### Following methods can be tried:
* Haar-cascade Detection
  * Parameters and haar-cascades (from [opencv](https://github.com/opencv/opencv/tree/master/data/haarcascades) and [Andrews Sobral](https://github.com/andrewssobral/vehicle_detection_haarcascades)):
    * scaleFactor
    * minNeighbors
    * minSize
    * maxSize
    * haarcascade_frontalface_default.xml 
    * haarcascade_eye.xml    
    * haarcascade_mcs_mouth.xml  
    * haarcascade_upperbody.xml
    * haarcascade_lowerbody.xml
    * haarcascade_fullbody.xml
    * haarcascade_eye_tree_eyeglasses.xml
    * haarcascade_mcs_upperbody.xml
    * haarcascade_profileface.xml
    * haarcascade_car.xml
    * haarcascade_head.xml
    * haarcascade_people.xml
* HOG-SVN Detection (cv2.HOGDescriptor_getDefaultPeopleDetector())
  * Parameters:
    * winStride
    * padding
    * scale
* Camshift Tracking
  * Termination criteria:
    * max_iter (maximum iterations)
    * min_movement (minimum movement at pixel)
    * Grouping
* MIL Tracking
* KCF Tracking
* CSRT Tracking
* CSRT Boosting
* TLD Boosting
* MedianFlow Boosting
* MOSSE Boosting
* Non-Maximum Suppression (for detection and tracking)
  * Parameters:
    * overlapThreshold

## Prerequisites and used versions

* Python 3.6
* OpenCV 3.4 
* OpenCV's extra modules(opencv_contrib)
* NumPy 1.12.1
* Imutils 0.4.3
* Tkinter 8.6

## Usage

1. Start tool, open a video file or use a webcam by id.

<p align="center">
  <img src="/images/gui.png" width="600" align="middle">
</p>

2. Select an region of interest to detect within.

<p align="center">
  <img src="/images/gui1.png" width="600" align="middle">
</p>

3. Release the mouse button and start tracking for detected objects.

<p align="center">
  <img src="/images/gui2.png" width="600" align="middle">
</p>

4. Check masks (HSV, RGB and Lab based) and play around with threshold values. 

<p align="center">
  <img src="/images/gui4.png" width="600" align="middle">
</p>

5. Check the masks in conjunction with the source video.

<p align="center">
  <img src="/images/gui3.png" width="600" align="middle">
</p>

6. Check histogram of selected ROI (HSV, RGB and Lab based).

<p align="center">
  <img src="/images/gui5.png" width="600" align="middle">
</p>

7. Choose what kind of detectors and trackers you want to try and play with some parameters. 

<p align="center">
  <img src="/images/gui6.png" width="600" align="middle">
</p>

## Authors

* **Marc-André Vollstedt** - marc.vollstedt@gmail.com

## Acknowledgments
