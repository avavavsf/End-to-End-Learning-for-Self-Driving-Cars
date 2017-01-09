# CarND-Behavioral-Cloning
My behavioral cloning project from Udactiy's Self-Driving Car Nanodegree. </br>
Project video here: https://youtu.be/ERMdxHlE2js </br>

Simulator can be found at:

[Windows 64 bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)</br>
[Windows 32 bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip)</br>
[macOS](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip)</br>
[Linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip)</br>

Model can be run as:
```
python drive.py model.json
```

Software requirements:
* numpy
* flask-socketio
* eventlet
* pillow
* h5py
* keras

## Data
I collected [data provided by Udacity](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip). I collected more data around tricky corners since most of the tracks were straight or soft curves.

I preprocessed the data for training by resizing to 66x200 pixel images and converting to YUV color scale, for using in the NVIDIA model. I also cropped the bottom 25 pix and top 10% of the image to remove unwanted noise. I used a generator to read images, perform the preprocessing and randomly augment the images and yield images and steering angles. For augmentation, I flipped random images horizontally and changed the brightness of the images randomly. I didn't create a different testing set because the real testing could be done by running the simulator in autonomous mode to get qualitative results.

## Model
For the model architecture I chose the [NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). It consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers. The convolutional layers are designed to perform feature extraction, and are chosen empirically through a series of experiments that vary layer configurations. We then use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, and a non-strided convolution with a 3×3 kernel size in the final two convolutional layers. A fully connected layer of 1 neuron was used at the end to output the steering angles. I used "relu" activations between the layers. 
I used a Lambda layer as the first layer. This ensured that the data fed into the NVIDIA model was normalized and there was no overfitting involved in the model. Using dropout gave poor results because the model is already very trim, so I avoided using it.

I optimized the model with an Adam optimizer over MSE loss.

## Hyperparameters
I used the basic NVIDIA model without any change. I tried ELU, tanh and RELU for activations and got the best results using RELU activations. I tried the [comma.ai model](https://github.com/commaai/research/blob/master/train_steering_model.py) which had the same input size as our images and made the most sense to apply but unfortunately could not get satisfactory results in the simulation. The values for epochs and batch size for training was obtained by experimenation. Batch size of 256 made sense because it was roughly 1% of the data (I also tried 128, 512, 1024 but did not get better results). After 15 epochs, the performance did not increase much so I chose that as the number of epochs.

Final results: I used the default parameters with Adam optimizer. I trained for 15 epochs and a batch size of 256 for training and 20 for validation. The number of samples per epoch for training was 20224 and 1000 for validation. 

## Results
Since the generator was randomly picking images, the training and validation accuracy were not consistent. Both training and validation accuracy were in the range of 0.02. The model can be evaluated qualitatively on the track where it drives the car well on track 1 without ever crashing or venturing into dangerous areas. However, it fails to generalize on track 2, possibly because it hasn't been trained for darker environments and slopes.

## Sample Images from camera view

#### Right image
![](/images/right_image.png "Right image")

#### Center image 
![](/images/image_center.png "Center image")

#### Left image
![](/images/image_left.png "Left image")
