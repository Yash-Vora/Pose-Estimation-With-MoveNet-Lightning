# Pose Estimation With MoveNet Lightning

MoveNet is the TensorFlow pre-trained model that identifies **17** different key points of the human body. It is the fastest model that detects the key points at a speed **>50fps**. 

For more information just visit the following link:\
[Click Here](https://www.tensorflow.org/lite/examples/pose_estimation/overview)

## Demo
<p align="center" width="100%">
  <img alt='Animation Showing Pose Estimation' src="gifs/movenet_gif.gif">
</p>

<p align="center" width="100%">
  <img alt='Animation Showing Pose Estimation With My Web Camera' src="gifs/my_gif.gif">
</p>

## Download MoveNet Lighting Model(Single-Pose)
Click on this link to download the model from Tensorflow Hub:
[Download Model](https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/3)

## Cloning
Just clone this repository to get the code by using the following command.

```powershell
git clone https://github.com/Yash-Vora/Pose-Estimation-With-MoveNet-Lightining.git
```

## Installation
Create a Virtual Environment & Install all the following required dependencies.

```powershell
pip install tensorflow
pip install tensorflow-gpu
pip install opencv-python
pip install argparse
```
_Note:_ It is not mandatory to install tensorflow-gpu but if you have GPU on your machine you can install it.

**OR**

Just execute the requirements.txt file by running the following command.

```powershell
pip install -r requirements.txt
```

## Usage
There are three arguments that you can pass from the cmd/terminal:\
`--path - Path of video`\
`--threshold_value - Pass threshold value between 0 to 1`\
`--output - Path to store the output of video/webcam`

Go to cmd/terminal/powershell and write the following commands to run this script.

1. It will detect key points from the video with a threshold value and output path.
```powershell
python run.py --path 'Input_Video/video.mp4' --threshold_value 0.4 --output 'Output_Video/output_video.avi'
```

2. It will detect key points from the video without threshold value(default value is 0.4) and output path.
```powershell
python run.py --path 'Input_Video/video.mp4'
```

3. It will detect key points from the video with output path and without threshold value(default value is 0.4).
```powershell
python run.py --path 'Input_Video/video.mp4' --output 'Output_Video/output_video.avi'
```

4. It will detect key points from the webcam with a threshold value and output path.
```powershell
python run.py --threshold_value 0.4 --output 'Output_Video/output_webcam.avi'
```

5. It will detect key points from the webcam without threshold value(default value is 0.4) and output path.
```powershell
python run.py
```

6. It will detect key points from webcam with output path and without threshold value(default value is 0.4).
```powershell
python run.py --output 'Output_Video/output_webcam.avi'
```
