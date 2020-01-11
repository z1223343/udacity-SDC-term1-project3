# **Project 3: Behavioral Cloning**

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
[image3]: ./figures/image1.jpg "Center Image"
[image4]: ./figures/image2.jpg "Left Image"
[image5]: ./figures/image3.jpg "Right Image"
[image6]: ./figures/image4.jpg "Flipped center Image"
[image8]: ./figures/image5.jpg "Flipped left Image"
[image9]: ./figures/image6.jpg "Flipped right Image"
[image7]: ./figures/corp_image.jpg "Cropped Image"

## Rubric Points
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` A Python script to create and train the model
* `model.h5` A trained convolution neural network 
* `drive.py` A python script to control the vehicle in the simultor
* `Video.mp4` A video file showing a car driving autonomously for more than one lap
* `README.md` A report to summarize the results

#### 2. Submission includes functional code
Using the simulator and drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolutional neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the structure of the PilotNet from Nvidia which is presented in their paper by NVIDIA, as shown in the following figure. To aviod overfitting, I added a few dropout layers and reduced one fully connected layer from PilotNet, and it turns out to be able to improve the validation accuracy a lot during the tests.

![PilotNet from NVIDIA][image1]

My model's structure is described as follows:
1. A Lambda layer to normalize the data.
2. Five convolution layers, each followed by a RELU activation function.
3. A Flatten layer to make the data into one row in preparation of the fully connected layers.
4. Three fully connected layers, with two Dropout layers in between.

#### 2. Attempts to reduce overfitting in the model

1. Contain two dropout layers in order to reduce overfitting with 0.25 `keep_prob`.

2. Reduce one fully connected layer from PilotNet.

3. Make data set as large as possible. Use left and right camera images with `steering_correction` and flip images of center, left and right cameras.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Closewise & anticlockwise driving, combination of center lane driving and recovering from the left and right sides of the road can be used when collecting data set. In this project I also used sample data directly.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to prevent overfitting and reduce the loss of validation accuracy.

My first step was to use a convolutional neural network model similar to PilotNet by Nvidia, I thought this model might be appropriate because it has been proved to be successful in similar task.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a high mean squared error on both the training set and the validation set. This implied that the model was not fit for the task and somehow overfitting.

To combat the the problem, I modified the model so that it is simple enough to be fit for this task. I also enlarge my data set by using flip images of three cameras to deal with overfitting.

Finally, my model turn out to produce a relatively low mean squared error on training set and validation set. Although the loss of validation set doesn't reduce as training set during the 10 epochs, it is stable and low compared with my first model. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a deep neural network with multiple convolusion and fully connected layers, as described in details in section "Model Architecture and Training Strategy".

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps driving respectively in clockwise direction and in anticlockwise direction on the first track using center lane driving. Here is three example images from center, left and right cameras respectively:

![Center Image][image3]
![Left Image][image4]
![Right Image][image5]

To augment the data sat, I also flipped images from center, left and right cameras. For example, here is an image that has then been flipped:

![Flipped Center Image][image6]
![Flipped Left Image][image8]
![Flipped Right Image][image9]

I then preprocessed this data by cropping and normalizing the images. For example, here is an image that has been cropped:

![Cropped Center Image][image7]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

In total, my data set contains 38,568 images.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was determined to be 10 as a compromise between accuracy and time efficiency. I used an adam optimizer so that manually training the learning rate wasn't necessary. The following figure shows the loss evolution for both training and validation sets during 10 epochs.

![Model Loss][image2]


