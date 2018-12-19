# SimpleObjectDetectionAndTracking
Tool to try simple object detection and tracking methods as well as different parameters using OpenCV with Python and Tkinter

It was a good starting point to gain insight multiple object detection and tracking. To detect and track many vehicles and pedestrians in crowded scenes (or even with an moving cam) you will need more advanced methods using convolutional neural networks (CNN) for detection and tracking as well as tracking-by-detection approaches. See for example [here](https://github.com/mavoll/MotionPathsExtraction).

To run the tool:

* Install prerequisites and run the python script (simpke_detection_and_tracking.py), or
* just run the executable file (.exe file for windows; .app for mac will follow) 

### Following methods can be tried:
* Haar-cascade Detection
  * Parameters:
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
    * meanShift
* Camshift Tracking
  * Termination criteria:
    * max_iter (maximum iterations)
    * min_movement (minimum movement at pixel)
* MIL Tracking
* KCF Tracking
* (AruCo Marker Detection)

## Prerequisites and used versions

* Python 3.6
* OpenCV 3.2
* NumPy 1.12.1
* Imutils 0.4.3
* Tkinter 8.6

## Usage

* For just trying out, you can find an image and tracking result files [here](/test_data/cam_01/recording_day/time_slice/).

1. Start tool, select an image representing the cams perspective and set parameters. Default parameters are taken from the image (created at) and its path using python´s os module.

<p align="center">
  <img src="/images/set_image_parameter.jpg" width="600" align="middle">
</p>

2. Select tracking result files and choose the object classes you want to count for.

<p align="center">
  <img src="/images/select_tracking_files.jpg" width="450" align="middle">
  <img src="/images/select_classes.jpg" width="150" align="middle">
</p>

3. Draw all tracks (related to former chosen object classes). Drawing all relevant tracks helps to find best positions for the counting lines. 

<p align="center">
  <img src="/images/draw_all_tracks.jpg" width="600" align="middle">
</p>

4. Draw counting lines into image.

* point p1 is the point of the line where you have started to draw the line. Accordingly p2 is the point where you have released the left mouse button. 
* if you look from point p1 to point p2 you will always find B at the left hand side and A at the right hand side

<p align="center">
  <img src="/images/counting_line.jpg" width="600" align="middle">
  <img src="/images/draw_counting_lines.jpg" width="600" align="middle">
</p>

5. Start counting (can take a while depending off number of tracks and counting lines).

<p align="center">
  <img src="/images/start_counting.jpg" width="600" align="middle">
</p>

6. Choose export granularities and save results to excel. 

<p align="center">
  <img src="/images/save_results_to_excel.jpg" width="300" align="middle">
  <img src="/images/excel_results.jpg" width="600" align="middle">
</p>


## Authors

* **Marc-André Vollstedt** - marc.vollstedt@gmail.com

## Acknowledgments
