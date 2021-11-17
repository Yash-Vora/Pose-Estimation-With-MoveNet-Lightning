'''

Go to cmd/terminal and write following commands to run this script.

1. If you want to detect keypoints from video with threshold value and output path:
   python detect.py --path <Path of the video> --threshold_value <Enter threshold value between 0 and 1> --output <Specify output path where video will be stored>
   Example:
   python detect.py --path 'Input_Video/video.mp4' --threshold_value 0.4 --output 'Output_Video/output_video.avi'

2. If you want to detect keypoints from video without threshold value(default value is 0.4) and output path:
   python detect.py --path <Path of the video>
   Example:
   python detect.py --path 'Input_Video/video.mp4'

3. If you want to detect keypoints from video with output path and without threshold value(default value is 0.4):
   python detect.py --path <Path of the video> --output <Specify output path where video will be stored>
   Example:
   python detect.py --path 'Input_Video/video.mp4' --output 'Output_Video/output_video.avi'

4. If you want to detect keypoints from webcam with threshold value and output path:
   python detect.py --threshold_value <Enter threshold value between 0 and 1> --output <Specify output path where video will be stored>
   Example:
   python detect.py --threshold_value 0.4 --output 'Output_Video/output_webcam.avi'

5. If you want to detect keypoints from webcam without threshold value(default value is 0.4) and output path:
   python detect.py
   Example:
   python detect.py

6. If you want to detect keypoints from webcam with output path and without threshold value(default value is 0.4):
   python detect.py --output <Specify output path where video will be stored>
   Example:
   python detect.py --output 'Output_Video/output_webcam.avi'


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
import tensorflow as tf
import numpy as np
import cv2
import argparse


# Draw Keypoints
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape # frame.shape = (480,640,3)
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, k_conf = kp
        
        if k_conf > confidence_threshold:
            cv2.circle(frame, (round(kx),round(ky)), 4, (255,0,0), -1)


# Draw Connections
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape # frame.shape = (480,640,3)
    shaped = np.squeeze(np.multiply(keypoints,[y,x,1]))
    
    for edge,color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (round(x1),round(y1)), (round(x2),round(y2)), (0,255,0), 2)


# Make Detection
def make_detection(path, Threshold_Value, output_path):
    # Used for Downloading Video
    if output_path != None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (192,192))

    # WebCam/Video (If path=0 it is WebCam and If path='/File Path' it is a video)
    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        # Read from WebCam/Video by frames
        ret,frame = cap.read()

        if ret == True:        
            # Reshape Image - Because MoveNet is accepting input in 192x192x3 and data type must be float32
            img = frame.copy()
            img = tf.image.resize_with_pad(np.expand_dims(img,axis=0), 192, 192)
            input_image = tf.cast(img, dtype=tf.float32)
            
            
            # Set Input and Output
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Make Predictions
            interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
            interpreter.invoke()
            keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
            print(keypoints_with_scores)
            
            # Draw Connections
            draw_connections(frame, keypoints_with_scores, EDGES, Threshold_Value)
            
            # Draw Keypoints
            draw_keypoints(frame, keypoints_with_scores, Threshold_Value)

            # Download Video
            if output_path != None:
                out.write(frame)
            
            # Show WebCam/Video
            cv2.imshow('MoveNet Lightning', frame)
            
            # Stop the WebCam/Video when 'q' is pressed on the keyboard
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        else:
            break

    # Release the WebCam/Video
    cap.release()

    # Release Storing Video
    if output_path != None:
        out.release()

    # Destroy all windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
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

    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument('--path', type=str, default=0, help="If it is webcam don't pass this argument(default value is 0) otherwise enter path if it is video")
    ap.add_argument('--threshold_value', type=float, default=0.4, help="Enter threshold value between 0 and 1. If you don't enter any value by default it will take 0.4")
    ap.add_argument('--output', type=str, default=None, help="Store the output video on the specified location")

    # Parse the argument
    parsed_args = ap.parse_args()

    # Take parsed arguments and pass to make_detection() function to make detection
    path, Threshold_Value, output_path = parsed_args.path, parsed_args.threshold_value, parsed_args.output
    make_detection(path, Threshold_Value, output_path)


# -------------------------------------------------------------------------------END---------------------------------------------------------------------------------------