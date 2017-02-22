# **Behavioral Cloning**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points][1] individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup\_report.md or writeup\_report.pdf summarizing the results

#### 2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed
My model consists of 3 convolution layers  and 3 fully connected layers:

1. **Layer 1**: convolutional layer with 24  5x5 filters, subsample 2 x 2 , RELU activation
2. **Layer 2**: convolutional layer with 36 5x5 filters, subsample 2 x 2, RELU activation 
3. **Layer 3**: convolutional layer with 64 3x3 filters, RELU activation,
	  followed by MaxPooling 2 x2  
	  Flatten 
4. **Layer 4**: Fully connected layer with 1000 neurons,and RELU activation  
	 Dropout(0.5) 
5. **Layer 5**: Fully connected layer with 500 neurons,  and ELU activation
	 Dropout(0.5)
6. **Layer 6**: Fully connected layer with 10 neurons,  and ELU activation


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 87,89). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 93).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image-1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image-2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steering angle correction.These images show what a recovery looks like starting from ... :

![alt text][image-3]
![alt text][image-4]
![alt text][image-5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would simulate turing right and also reverse the steering angle. For example, here is an image that has then been flipped:

![alt text][image-6]
![alt text][image-7]

After the collection process, I had 24108 number of data points. I then preprocessed this data by cropping operation. I remove 50 pixel of the original image from the top and 25 pixel from bottom.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

[1]:	https://review.udacity.com/#!/rubrics/432/view

[image-1]:	./examples/placeholder.png "Model Visualization"
[image-2]:	./examples/placeholder.png "Grayscaling"
[image-3]:	./examples/placeholder_small.png "Recovery Image"
[image-4]:	./examples/placeholder_small.png "Recovery Image"
[image-5]:	./examples/placeholder_small.png "Recovery Image"
[image-6]:	./examples/placeholder_small.png "Normal Image"
[image-7]:	./examples/placeholder_small.png "Flipped Image"