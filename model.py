
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree project 3
# 
# Project3 : Behavior Cloning
# End to End Leanring to train a CNN from a single camera to steering angles.
# 

# Image shape:(160, 320, 3)

print("Start import software module")
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.convolutional import Convolution2D
from keras.models import model_from_json
from keras import backend as K
#import pickle
#import random
import numpy as np
import tensorflow as tf
#import sklearn
#import cv2
from sklearn.model_selection import train_test_split
#from tensorflow.contrib.layers import flatten
import matplotlib.image as mpimg
print("Finish software module")

# Load datadata
print("Start data loading")
# need to revise on OSU OSC
features_directory = './training_data/'
#labels_file= './training_data/test.csv'
labels_file= './training_data/driving_log.csv'
labels = []
features = []
with open(labels_file, mode='r') as f:
    reader = csv.reader(f, delimiter=',')
    headers = next(reader)
    for row in reader:
        labels.append(float(row[3]))
        #only use the center images
        features_file = features_directory + row[0]
        img=mpimg.imread(features_file)
        features.append(img)
print("Finish data loading")
#print(labels)
#print(features[0])
print("Start training/validation/testing splitting")
# split the data into 70% training, 15% validation, and 15% testing
train_features, validation_test_features, train_labels, validation_test_labels = train_test_split(
   features,
   labels,
   test_size=0.2,
   random_state=26746
)
del features, labels
test_features, validation_features, test_labels, validation_labels = train_test_split(
   validation_test_features,
   validation_test_labels,
   test_size=0.5,
   random_state=38562
)
del validation_test_features, validation_test_labels
train_features = np.array(train_features)
validation_features = np.array(validation_features)
test_features = np.array(test_features)
print("Number of training examples =", len(train_features))
print("Number of validation examples =", len(validation_features))
print("Number of testing examples =", len(test_features))
print("Image data shape =", train_features[0].shape)
print("Training data shape =", train_features.shape)

print("Finish training/validation/testing splitting")


print("Start preprocess")
#define a min-max scaling function used to normalize the image data
def normalize_inputimage(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    a = 0.1
    b = 0.9
    greyscale_min = 0
    greyscale_max = 255
    return a + ( ( (image_data - greyscale_min)*(b - a) )/( greyscale_max - greyscale_min ) )

def transform_image(img,ang_range):
    #1- Image
    #2- ang_range: Range of angles for rotation   

    # Rotation
    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)
        
    img = cv2.warpAffine(img,Rot_M,(cols,rows))

    return img


#generate new training data to make the data balance
#inputs_per_class = np.bincount(y_train)
# each class will have 2800-3200 training examples 
#for i in range(len(inputs_per_class)):
#    add_number = 3000 + random.randint(-200, 200) - inputs_per_class[i]
#    
#    new_features = []
#    new_labels = []
#    mask = np.where(y_train == i)
#    features = X_train[mask]
#    for j in range(add_number):
#        index = random.randint(0, inputs_per_class[i] - 1)
#        new_features.append(transform_image(features[index],20))
#        new_labels.append(i)
#    X_train = np.append(X_train, new_features, axis=0)
#    y_train = np.append(y_train, new_labels, axis=0)
#del new_features, new_labels

#Normorlization, scale the image data from [0 255] to [0.1 0.9]
#train_features = normalize_inputimage(train_features)
#validation_features = normalize_inputimage(validation_features)
#test_features = normalize_inputimage(test_features)

print('Finish preprocess')




def Steering_Model(cameraFormat=(3, 160, 320)):
    """
    The intent is a scaled down version of the model from "End to End Learning
    for Self-Driving Cars": https://arxiv.org/abs/1604.07316.
    Args:
      cameraFormat: (3-tuple) Ints to specify the input dimensions (color
          channels, rows, columns).
    Returns:
      A compiled Keras model.
    """
    print('Start building model')

    ch, row, col = cameraFormat 

    model = Sequential() 

    # Use a lambda layer to normalize the input data
    model.add(Lambda(
        lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch))
    )
    '''
    In tensorflow

    For the VALID padding, the output height and width are computed as:

    out_height = ceil(float(in_height - filter_height + 1) / float(strides1))

    out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))
    '''

    #filter(24, 5, 5), valid padding, stride 2, input(3, 160, 320), output(24, 78, 158)
    model.add(Convolution2D(24, 5, 5, init = 'normal', subsample= (2, 2), name='conv1_1'))
    model.add(Activation('relu'))
    #filter(36, 5, 5), valid padding, stride 2, input(24, 78, 158), output(36, 37, 77)
    model.add(Convolution2D(36, 5, 5, init = 'normal', subsample= (2, 2), name='conv2_1'))
    model.add(Activation('relu'))
    #filter(48, 5, 5), valid padding, stride 2, input(36, 37, 77), output(48, 17, 37)
    model.add(Convolution2D(48, 5, 5, init = 'normal', subsample= (2, 2), name='conv3_1'))
    model.add(Activation('relu'))
    #filter(64, 3, 3), valid padding, stride 1, input(48, 17, 37), output(64, 15, 35)
    model.add(Convolution2D(64, 3, 3, init = 'normal', subsample= (1, 1), name='conv4_1'))
    model.add(Activation('relu'))
    #filter(64, 3, 3), valid padding, stride 1, input(64, 15, 35), output(64, 13, 33)
    model.add(Convolution2D(64, 3, 3, init = 'normal', subsample= (1, 1), name='conv4_2'))
    model.add(Activation('relu'))
    #input(64, 13, 33),output 27456
    model.add(Flatten())
    #input 27456,output 1164
    model.add(Dense(1164, init = 'normal', name = "dense_0"))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Dense(100, init = 'normal',  name = "dense_1"))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Dense(50, init = 'normal', name = "dense_2"))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Dense(10, init = 'normal', name = "dense_3"))
    model.add(Activation('relu'))
    model.add(Dense(1, init = 'normal', name = "dense_4"))

    model.summary()
    # Adam optimizer is a standard, efficient SGD optimization method
     # Loss function is mean squared error, standard for regression problems
    model.compile(optimizer="adam", loss="mse")

    return model

print('Finish building model')
print('Start training model')

#train
model = Steering_Model()
history = model.fit(train_features, train_labels,
                    batch_size=128, nb_epoch=1,
                    verbose=1, validation_data=(validation_features, validation_labels))
print(history.history)
print('Finish training model')
print('Start save model architecture and weights')
#save the model architecture
json_model = model.to_json()
with open("model.json", "w") as f:
    f.write(json_model)
#save the model weights
model.save_weights("model.h5")
print('Finish save model architecture and weights')


#test after satisfying with the validation accuracy.
#Returns the loss value and metrics values for the model in test mode
#model.evaluate(test_features, test_labels, batch_size=128)