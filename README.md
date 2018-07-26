# **udacity-term1-project3**
# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./figures/PilotNet_NVIDIA.JPG "PilotNet from NVIDIA"
[image2]: ./figures/model_loss.png "Model Loss"
[image3]: ./figures/center_image.jpg "Center Image"
[image4]: ./figures/left_image.jpg "Left Image"
[image5]: ./figures/right_image.jpg "Right Image"
[image6]: ./figures/center_image_flip.png "Flipped Image"
[image7]: ./figures/center_image_crop.png "Cropped Image"

## Rubric Points
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` A Python script to create and train the model
* `model.h5` A trained convolution neural network 
* 'drive.py' A python script to control the vehicle in the simultor
* `Video.mp4` A video file showing a car driving autonomously for more than one lap
* `README.md` A report to summarize the results

#### 2. Submission includes functional code
Using the simulator and drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
