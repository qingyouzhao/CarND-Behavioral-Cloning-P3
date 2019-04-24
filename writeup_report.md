# **Behavioral Cloning** 

## Writeup 
by Qingyou Zhao


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided [simulator](todo(qingouz:simulatorlink)) and my drive.py file, the car can be driven autonomously around the track by executing 
```ps
python drive.py model.h5
```
#### 3. Submission code is usable and readable

The model.py file contains the final code for training and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
The playground.ipynb is the file used for prototypical works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Based on [nvidia deep learning model for self-driving cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The training data used the project default data set while validation used cusom data collected. Overfitting can be observed by having small training loss but large validation loss. This was not observed during the training process. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track based on as visualized in **video.mp4**

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road provided in the project default data set.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to prototype from simple model to more complicated model. I evaluated the model by checking the training error, validation error and test with the simulator.

My first step was to use a convolution neural network model similar to the LeNet-5. I thought this model might be appropriate because it is a simple yet powerful image classification model. The result was alright until the car sees the water section in the track and drove off.

Then I tested with some models built into Keras, like VGG-16 and InceptionV3. Then tried to apply transfer learning technique to that. I was not sure if the transfer learning implementation was fully correct so I chose another route due to time constraint.

Then I switched to using the [nvidia deep learning model for self-driving cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). This time the model worked really well after training 2 epochs.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture(function`nvidia_net` in `model.py`) consisted of a convolution neural network with the following layers and layer sizes where the final output corresponds to the steering angle.

| Layers                               | Sizes  |
| ------------------------------------ | ------ |
| Lambda for normalizing image         |        |
| Lambda for corpping top of the image |        |
| Conv2D                               | 24,5,5 |
| Conv2D                               | 36,5,5 |
| Conv2D                               | 48,5,5 |
| Conv2D                               | 64,3,3 |
| Conv2D                               | 64,3,3 |
| Flatten                              |        |
| Dense                                | 100    |
| Dense                                | 50     |
| Dense                                | 10     |
| Dense                                | 1      |



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to respond to driving off different kinds of tracks. However the training data I first collected was not good enough because using keyboard input resulted in binary inputs of either 0 angle or max steering angle. 

With time limit, I fall back to the project default data set but still used the data I collected as validation. An example is here

![Example of center](.\examples\center_2016_12_01_13_33_07_936.jpg)

![Example of left](.\examples\left_2019_04_07_18_13_15_260.jpg)

![right_2016_12_01_13_43_02_568](.\examples\right_2016_12_01_13_43_02_568.jpg)



To augment the data sat, I also flipped images and angles thinking that this would tackle the problem of the training data collected being all driving anti-clockwise. I also used the left and right camera image with and additional sterring angle of 0.2 to augment the training data available.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
