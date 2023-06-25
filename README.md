# Balls Tracking

A Python script to detect and track yellow and white balls in a video stream. The script takes advantage of OpenCV library for computer vision tasks.


## Requirements

    Python 3.x
    OpenCV
    Numpy

## Usage
The script can be run from the terminal with the following command:

bash

`python ball_tracking.py --input <path_to_input_file_or_camera_index> --z <height_of_table> --show <True_or_False> --print_centers <True_or_False> --record <True_or_False>`

The following arguments are required:

    --input: path to input video file or camera index (default is 0)
    --z: height of table in cm (default is 50)

The following arguments are optional:

    --show: set to True to show the window with the ball detection, False otherwise (default is False)
    --print_centers: set to True to print the coordinates of each detected ball center on the console, False otherwise (default is False)
    --record: set to True to record an output video file, False otherwise (default is False)


## Functionality
The script first applies a Gaussian blur to the input frame, then converts it to the HSV color space. Yellow and white balls are detected separately by thresholding the image using specified lower and higher HSV color values. Contours are found in the resulting binary images, and bounding boxes are drawn around the contours that have an area within a specified range.

The script can also record the output frames to a video file if specified.
