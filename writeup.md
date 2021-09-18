# Behavioral Cloning Project

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

In the following I give a more detailed description and reflection on the project. The main development was carried out using the [self-driving car simulator][simulator], using a local installation, and a Jupyter [notebook](./ModelBuilding.ipynb). Note that in order to create all required files mentioned in the project rubic, the file ``model.py`` is just an automatically created version of this notebook.

[simulator]: https://github.com/udacity/self-driving-car-sim
[simulator_screenshot]: ./examples/simulator.jpg
[orig_images_0]: ./examples/orig_images_0.jpg "Original data - Sample 1"
[orig_images_1]: ./examples/orig_images_1.jpg "Original data - Sample 2"
[cropped]: ./examples/cropped_image.jpg "Cropped image"
[history]: ./examples/history.jpg "Fitting history"
[video]: video.mp4 "Video output"
[video_add]: video_additional_simulator.mp4 "Additional video from simulator"
[nvidia_paper]: ./literature/end-to-end-dl-using-px.pdf

----


# Description & Reflection

## 1. Setup

The simulator was installed following the description on the [self-driving car simulator][simulator] GitHub page. 

Local environment was created via the modified ``environment_mod_gpu.yml``. Note that ``cudnn=6`` is still missing for proper usage of TensorFlow GPU 1.3, which does not seem to be provided anywhere on the common conda channels. Thus, download directly from NVIDIA and extract its content to the ``Library`` folder of the environment folder. 


## 2. Data

This section covers:
* Use the simulator to collect data of good driving behavior

Data is collected using the simulator in training mode. The simulator is like a video game, where one steers a car using either keyboard and/or mouse inputs. A screenshot of the simulator is shown below.

![][simulator_screenshot]

The recorded data are not screenshots from the simulator, but rather images taken from 3 different cameras mounted in the car, showing left, center and right views. Besides those images, most importantly, also the steering angle is recorded. Examples of those data can be seen in the following examples:

![][orig_images_0]
![][orig_images_1]

The model should output the steering angle from the center facing image. A steering angle is good if it __behaves like__ the steering angle of a good human driver. Thus it is essential that the recorded data contains good human driving behavior.

Via the simulator the following datasets were collected:

* data_run1: Track 1 original direction
* data_run2: Track 2 original direction
* data_run3: Track 3 reverse direction (first picture above is from reverse direction knowing the track)

Each set consists of a single lap where I tried to drive center focused. I had to restart this a couple of times because in particular in the curves this was challenging, and I did not want to create bad data. For steering, I was using the mouse to get smoother steering targets. Note those data files are not part of the repository. Ultimately for the final model I also only used the data from run2 and run3, because it showed the best performance.

In order to create the training data let me comment on the following points:

* The center images are taken directly.
* Having records also from left and right views are very important additional data, because they allow the model to learn to steer back to center. For them the steering targets must be corrected to represent good behavior, and the argument is as follows: In the above second image, the center image clearly signals to steer to the left. However, using the left image and assuming it to be center view, the steering direction towards the left is slightly reduced, while from the right view it is slightly enlarged. This correction factor is one of the main hyperparameters that I had to tune. 
* A simple data augmentation is flipping images left and right, and inverting the steering angle.
* Data augmentation on image quality did not sound reasonable to me in such simulated cases, but might be very important on real data.
* Note, images and measurement still fit easily in memory. So I did not see any point using a custom-made generator.


## 3. Model

This section covers:

* Build, a convolution neural network in Keras that predicts steering angles from images

Following the course material I was slowly progressing thru different network architectures, starting with a simple feed-forward network, and then moving to LeNet5. Finally, I used the network from the mentioned [NVIDIA paper][nvidia_paper] (I included a copy in this repository). Additionally, I cropped the input to focus only on the road, an example given below. This was already providing excellent results.

![][cropped]

For completeness let me include the Python code creating the model using the Kears Sequential API; this should suffice to understand the model architecture.

```python
from keras import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, AvgPool2D, Cropping2D, Dropout

# Keras model: ./literature/end-to-end-dl-using-px.pdf
keras.backend.clear_session()
model = Sequential()
model.add(Cropping2D(cropping=CROPPING, input_shape=INPUT_SHAPE))
model.add(Lambda(lambda x: x / 255. - 0.5))
model.add(Conv2D(24, 5, strides=2, activation='relu'))
model.add(Conv2D(36, 5, strides=2, activation='relu'))
model.add(Conv2D(48, 5, strides=2, activation='relu'))
model.add(Conv2D(64, 3, activation='relu'))
model.add(Conv2D(64, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()
```


## 4. Training & Outcome

This subsection includes:

* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

Training is carried out on the mean-squared-error loss function and using the Adam optimizer with default setting. A certain random fraction (finally 20%) of data is used as validation set. On this I observe that only a few epochs are required before degradation. I employ the Keras model checkpointer callback to only save the best model while fitting. The learning curve for the trained model is shown below. 

![][history]

After that the model can be utilized in the autonomous mode of the simulator. For this one needs to start the prediction engine
```cmd
python drive.py model.h5 [image_folder]
```
using the trained model file of the Keras model ``model.h5`` and starting autonomous mode in the simulator. Specifying an image folder captures the front view during autonoums driving, which later can be turned into a video using the script ``video.py``.

The output of driving one lap as requested by the project rubic can be found  [here][video].

Since I really liked also having a view from the autonomous mode I included a second video that shows the [full simulator in autonomous mode][video_add]. This includes about 3 laps: The first being a full autonomous run going in the original direction, the second lap is autonomously going in reverse direction. In the third lap I manually intervened and steered the car towards the side of the track to see how the autonomous mode would correct this. The interaction can be seen on the top left side of the simulator via the _Mode: Manual_ instances. I was pretty satisfied with its performance.
