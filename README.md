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

[image1]: ./writeup_images/cnn-architecture-624x890.png "Nvidia CNN"
[image2]: ./writeup_images/cnn-architecture-624x890_cropped.png "Nvidia CNN modified"
[image6]: ./writeup_images/2019_02_17_22_44_41_736.jpg "Normal Image"
[image7]: ./writeup_images/2019_02_17_22_44_41_736_flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model architecture that was employed for this project is based on the Nvidia self driving car paper.
https://devblogs.nvidia.com/deep-learning-self-driving-cars/

This model utilizes the following architecture:

![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

The model had the original images of size 160x320 cropped down to reduce background noise overfitting on non relevant information. This was added to the Nvidia CNN design after the normalization and input planes.  The images were cropped by 50 pixels on the top and 20 pixels on the bottom with no crop on the left or right of the image.  This resulted in a 110x320 image for processing.
 
The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 100). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 98) and further utilized a loss function of the mean squared error for the adam optimizer. 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving with the provided steering measurements.  After some initial attempts around the track with almost successfully completing the track some adjustments had to be made.  The first adjustment was augmenting the data with a reverse of the course so the car stopped pulling to the left of the track.
The second adjustment was adding in the left and right images and creating a correction factor for the steering for those images based on the center image.  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

After going through the coursework and seeing the Nvidia CNN for their self driving car I decided to try and apply that to this project.  The first step was to implment their model using Keras.  This was successfully done and when the original training data was run through the model and then the model tested, it was able to drive around the first curve and then proceeded to pull left and run off the track.

The next step was then improving the model.  From observing the pull to the left I thought to correct this by augmenting the dataset by flipping all of the images and steering values to create a right track bias to the system.  When the model was then retrained and run again it was able to make it around the turn.  After that it came to straight away bridge and then failed to turn quick enough for the S turns in the track.

Going back to drawing board I decided to take the left and right camera angles and incorporate the image data from them to help with the course.  This data was also augmented to create a reverse right track to balance out the left track.  A correction factor was applied to the left and right of 0.2 plus or minus from the center steering value.  When the model was trained and run it was able to then successfully navigate the course.

#### 2. Final Model Architecture

The final model architecture (model.py lines 85-96) consisted of a convolution neural network with the following layers and layer sizes, here is a visualization of the architecture.

![alt text][image2]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first tried to capture my own data but found that this was not ideal with my setup.  After further review I found provided test data for the first track and utilized that as my training data.

To augment the dataset I flipped the images to create a reverse track and flipped the steering measurements as well.  Next 

![alt text][image6]
![alt text][image7]

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by after applying all of the augmented data and retaining the model was able to successfully navigate the course, so this parameter did not have to tuned further. I used an adam optimizer so that manually training the learning rate wasn't necessary.


#### 4. Final Result Video
<img src="final_video/output_video.gif?raw=true" width="720px">

### Lessons Learned
In this project I had to fix a few things on my local environment in order to get the project to connect and run the provided Unity simulation.  After this was done and some modifications to the Keras version I was able to build the model and augment the data without any problems or road blocks.  In the future I would like to incorporate some of the other fields to the model that were provided in the data file (braking, speed, throttle) and see how the model performs.  Also, testing the second track the model with using the first track training data is able to navigate until the first major turn and then it leaves the road.