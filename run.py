'''

Go to cmd/terminal and write following commands to run this script.

1. If you want to detect keypoints from video with threshold value and output path:
   python run.py --path <Path of the video> --threshold_value <Enter threshold value between 0 and 1> --output <Specify output path where video will be stored>
   Example:
   python run.py --path 'Input_Video/video.mp4' --threshold_value 0.4 --output 'Output_Video/output_video.avi'

2. If you want to detect keypoints from video without threshold value(default value is 0.4) and output path:
   python run.py --path <Path of the video>
   Example:
   python run.py --path 'Input_Video/video.mp4'

3. If you want to detect keypoints from video with output path and without threshold value(default value is 0.4):
   python run.py --path <Path of the video> --output <Specify output path where video will be stored>
   Example:
   python run.py --path 'Input_Video/video.mp4' --output 'Output_Video/output_video.avi'

4. If you want to detect keypoints from webcam with threshold value and output path:
   python run.py --threshold_value <Enter threshold value between 0 and 1> --output <Specify output path where video will be stored>
   Example:
   python run.py --threshold_value 0.4 --output 'Output_Video/output_webcam.avi'

5. If you want to detect keypoints from webcam without threshold value(default value is 0.4) and output path:
   python run.py
   Example:
   python run.py

6. If you want to detect keypoints from webcam with output path and without threshold value(default value is 0.4):
   python run.py --output <Specify output path where video will be stored>
   Example:
   python run.py --output 'Output_Video/output_webcam.avi'


'''


'''

Before running this code create virtual environment and install all the following required dependencies.

1. pip install tensorflow
2. pip install tensorflow-gpu (It is not mandatory but if you have gpu in your machine you can install it)
3. pip install opencv-python
4. pip install argparse

Or

Just execute requirements.txt file and run following command:
pip install -r requirements.txt

'''


# Import Required Libraries
import argparse
import tensorflow as tf
from detect_keypoints import DetectKeypoints


# Load Model
interpreter = tf.lite.Interpreter(model_path='model/lite-model_movenet_singlepose_lightning_3.tflite')
interpreter.allocate_tensors()


# Used to draw connection by joining the keypoints
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


if __name__ == '__main__':
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument('--path', type=str, default=0, help="If it is webcam don't pass this argument(default value is 0) otherwise enter path if it is video")
    ap.add_argument('--threshold_value', type=float, default=0.4, help="Enter threshold value between 0 and 1. If you don't enter any value by default it will take 0.4")
    ap.add_argument('--output', type=str, default=None, help="Store the output video on the specified location")

    # Parse the argument
    parsed_args = ap.parse_args()

    # Store parsed arguments into an variable
    path, Threshold_Value, output_path = parsed_args.path, parsed_args.threshold_value, parsed_args.output

    # Create DetectKeypoints() class object, initialize values and call make_detection() to make detection
    detect_keypoints_obj = DetectKeypoints(path, Threshold_Value, output_path, interpreter, EDGES)
    detect_keypoints_obj.make_detection()
    

# -------------------------------------------------------------------------------END---------------------------------------------------------------------------------------