Driver Drowsiness Detection System

Overview
This repository contains a real-time Driver Drowsiness Detection System that leverages machine learning and computer vision to monitor drivers and identify signs of drowsiness. The system processes real-time video feed from a camera and uses facial landmarks to detect drowsiness, alerting the driver to ensure safety.

Features
Real-Time Detection: Captures video feed from a camera and processes frames in real-time.
Facial Landmark Detection: Identifies key facial features such as eyes and mouth.
Drowsiness Indicators: Measures metrics like Eye Aspect Ratio (EAR) and Yawn Frequency to detect:
Prolonged eye closure.
Frequent yawning.
Audio Alert: Triggers an alarm when drowsiness is detected.
Customizable Thresholds: Allows tuning of thresholds for EAR and other metrics to adapt to different scenarios.

System Architecture
Video Capture: Captures real-time video using a webcam or external camera.
Preprocessing: Detects face and landmarks using pre-trained models.

Drowsiness Detection:
Calculates Eye Aspect Ratio (EAR).
Detects yawning based on mouth aspect ratio.
Alert Mechanism: Issues an alarm if drowsiness thresholds are exceeded.

Dataset
The model is trained on a custom dataset comprising labeled images/videos of drowsy and non-drowsy states. Additional publicly available datasets like MRL Eye Dataset were used to enhance model performance.

Usage
Connect a webcam or external camera to your system.
Run the application using the above command.
The system will start processing the video feed and display the detection results in real-time.
If drowsiness is detected, an audio alert will sound.

Requirements
Python 3.7+
OpenCV
dlib
imutils
numpy
scipy

Future Improvements
Enhance detection accuracy in low-light environments.
Add support for multi-camera setups.
Implement cloud-based monitoring and analytics
