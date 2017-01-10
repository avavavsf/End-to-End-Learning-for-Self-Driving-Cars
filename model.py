
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
import pandas as pd
import numpy as np
import tensorflow as tf
#import sklearn
import cv2
from sklearn.model_selection import train_test_split
#from tensorflow.contrib.layers import flatten
print("Finish software module")

# Load datadata
print("Start data loading")
# need to revise on OSU OSC
#features_directory = '/users/PAS0947/osu8077/new/udacity/P3-CarND-Behavioral-Cloning/training_data/'
#labels_file= '/users/PAS0947/osu8077/new/udacity/P3-CarND-Behavioral-Cloning/training_data/driving_log.csv'
features_directory = './training_data/'
labels_file= './training_data/driving_log.csv'

## split the data into 80% training, 20% validation
n_ep = 0
with open(labels_file, mode='r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        n_ep = n_ep + 1
n_ep = n_ep - 1        
print('the number of epoch is ', n_ep)    
arr = np.arange(n_ep)
np.random.shuffle(arr)
t = arr[:int(n_ep * 0.8)]
v = arr[int(n_ep * 0.8):]

def read_image(file_name):
    img = cv2.imread(file_name)
    img = cv2.resize(img, (200, 66), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

# the generator
def generator(train_or_valid,batch_size):
    labels = []
    features = []
    while 1:
        if train_or_valid == "train":
            ibatch = np.random.choice(t, batch_size)
        elif train_or_valid == "valid":
            ibatch = np.random.choice(v, batch_size)
        else:
            print('wrong key words, sould be either train or valid ')

        df=pd.read_csv(labels_file, sep=',',header=None)
        for index in ibatch:
            camera_index = np.random.randint(3)
            features_file = features_directory + df[camera_index][index + 1].strip()
            img = read_image(features_file)
            steer = float(df[3][index + 1])
            #center camera
            if camera_index == 0:
                flip_index = np.random.randint(2)
                if flip_index == 0:
                    features.append(img)
                    labels.append(steer)
                else:
                    img = cv2.flip(img,1)
                    features.append(cv2.flip(img,1))
                    labels.append(steer * (-1.0))
            #left camera       
            elif camera_index == 1:
                flip_index = np.random.randint(2)
                if flip_index == 0:
                    features.append(img)
                    labels.append(steer + 0.25)
                else:
                    img = cv2.flip(img,1)
                    features.append(cv2.flip(img,1))
                    labels.append((steer + 0.25)* (-1.0))
            #right camera
            else:
                flip_index = np.random.randint(2)
                if flip_index == 0:
                    labels.append(steer - 0.25)
                    features.append(cv2.flip(img,1))
                else:
                    img = cv2.flip(img,1)
                    features.append(cv2.flip(img,1))
                    labels.append((steer - 0.25) * (-1.0))
    	yield features, labels                                                       

def Steering_Model(cameraFormat=(3, 66, 200)):
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

    return model

print('Finish building model')
print('Start training model')

#train
model = Steering_Model()
# Adam optimizer is a standard, efficient SGD optimization method
# Loss function is mean squared error, standard for regression problems
#adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer="adam", loss="mse")
#8036*3*2*0.8/256=151
#8036*3*2*0.2/128=76
history = model.fit_generator(generator('train',256),
        samples_per_epoch=151, nb_epoch=7,validation_data=generator('valid',128),
                    nb_val_samples=76)

print('Finish training model')
print('Start save model architecture and weights')
#save the model architecture
json_model = model.to_json()
#with open("/users/PAS0947/osu8077/new/udacity/P3-CarND-Behavioral-Cloning/model.json", "w") as f:
#    f.write(json_model)
with open("./model.json", "w") as f:
    f.write(json_model)
#save the model weights
#model.save_weights("/users/PAS0947/osu8077/new/udacity/P3-CarND-Behavioral-Cloning/model.h5")
model.save_weights("./model.h5")
print('Finish save model architecture and weights')

