
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
from keras.optimizers import Adam
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
features_directory = '/users/PAS0947/osu8077/new/udacity/P3-CarND-Behavioral-Cloning/training_data/'
labels_file= '/users/PAS0947/osu8077/new/udacity/P3-CarND-Behavioral-Cloning/training_data/driving_log.csv'
#features_directory = './training_data/'
#labels_file= './training_data/test.csv'
#define the input image shape
row = 66
col = 200
ch = 3 

def bright_augment(img):
    img1 = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    #print(random_bright)
    #print(img1[:,:,2])
    img1[:,:,2] = img1[:,:,2]*random_bright
    img1 = cv2.cvtColor(img1,cv2.COLOR_HSV2RGB)
    return img1

# This removes most of the area above the road and small amount below including the hood
def chop_image(img):
    n_row,n_col, n_ch = img.shape
    img1 = img[40:135, :]
    return img1
    
def resize_image(img):
    img1 = cv2.resize(img, (200,66), interpolation=cv2.INTER_AREA)
    return img1

def read_image(file_name):
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = chop_image(img)
    img = resize_image(img)
    img = bright_augment(img)
    return img

def model_save(model_json,model_h5):
    json_model = model.to_json()
    with open(model_json, "w") as f:
        f.write(json_model)
    model.save_weights(model_h5)

## split the data into 80% training, 20% validation
training_percent = 0.85
n_ep = 0
with open(labels_file, mode='r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        n_ep = n_ep + 1
n_ep = n_ep - 1        
print('the number of epoch is ', n_ep)    
arr = np.arange(n_ep)
np.random.shuffle(arr)
t = arr[:int(n_ep * training_percent)]
v = arr[int(n_ep * training_percent):]

# the generator
def data_generator(train_or_valid,batch_size):
    #labels = np.zeros(batch_size)
    #features = np.zeros((batch_size, row, col, 3))
    labels = []
    features = []
    ajusteed_angle = 0.25
    df=pd.read_csv(labels_file, sep=',',header=None)
    while True:
        n_sample = 0
        while n_sample < batch_size:
            #decide if train batch or valid batch
            if train_or_valid == "train":
                index = np.random.choice(t, 1)
            elif train_or_valid == "valid":
                index = np.random.choice(v, 1)
            else:
                print('wrong key words, sould be either train or valid ')
            #only choose the image 
            steer = float(df[3][index+1])
            camera_index = np.random.randint(3)
            #camera_index = 0
            features_file = features_directory + df.values[index+1][0][camera_index].strip()
            img = read_image(features_file)
            #center camera
            if camera_index == 0:
                flip_index = np.random.randint(2)
                if flip_index == 0:
                    img = cv2.flip(img,1)
                    steer = steer * (-1.0)
            #left camera       
            elif camera_index == 1:
                flip_index = np.random.randint(2)
                if flip_index == 0:
                    img = cv2.flip(img,1)
                    steer = (steer + ajusteed_angle) * (-1.0)
            #right camera
            else:               
                flip_index = np.random.randint(2)
                if flip_index == 0:
                    img = cv2.flip(img,1)
                    steer = (steer - ajusteed_angle) * (-1.0)
            features.append(img)
            labels.append(steer)                    
            n_sample = n_sample + 1    
            #if abs(labels[-1]) > 0.05:
            #    n_sample = n_sample + 1
            #else:
            #    features.pop()
            #    labels.pop()
        yield np.array(features), np.array(labels)                                                       

def Steering_Model(cameraFormat=(3, 66, 200)):
    """
    The intent is a scaled down version of the model from "End to End Learning
    for Self-Driving Cars": https://arxiv.org/abs/1604.07316.

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
    model.add(Convolution2D(24, 5, 5, init = 'he_normal', subsample= (2, 2), border_mode="valid", name='conv1_1'))
    model.add(ELU())
    #filter(36, 5, 5), valid padding, stride 2, input(24, 78, 158), output(36, 37, 77)
    model.add(Convolution2D(36, 5, 5, init = 'he_normal', subsample= (2, 2), border_mode="valid", name='conv2_1'))
    model.add(ELU())
    #filter(48, 5, 5), valid padding, stride 2, input(36, 37, 77), output(48, 17, 37)
    model.add(Convolution2D(48, 5, 5, init = 'he_normal', subsample= (2, 2), border_mode="valid", name='conv3_1'))
    model.add(ELU())
    #filter(64, 3, 3), valid padding, stride 1, input(48, 17, 37), output(64, 15, 35)
    model.add(Convolution2D(64, 3, 3, init = 'he_normal', subsample= (1, 1), border_mode="valid", name='conv4_1'))
    model.add(ELU())
    #filter(64, 3, 3), valid padding, stride 1, input(64, 15, 35), output(64, 13, 33)
    model.add(Convolution2D(64, 3, 3, init = 'he_normal', subsample= (1, 1), border_mode="valid", name='conv4_2'))
    model.add(ELU())
    #input(64, 13, 33),output 27456
    model.add(Flatten())
    #input 27456,output 1164
    model.add(Dense(1164, init = 'he_normal', name = "dense_0"))
    model.add(ELU())
    #model.add(Dropout(p))
    model.add(Dense(100, init = 'he_normal',  name = "dense_1"))
    model.add(ELU())
    #model.add(Dropout(p))
    model.add(Dense(50, init = 'he_normal', name = "dense_2"))
    model.add(ELU())
    #model.add(Dropout(p))
    model.add(Dense(10, init = 'he_normal', name = "dense_3"))
    model.add(ELU())
    model.add(Dense(1, init = 'he_normal', name = "dense_4"))

    model.summary()    

    return model

print('Finish building model')

model = Steering_Model()
# Adam optimizer is a standard, efficient SGD optimization method
# Loss function is mean squared error, standard for regression problems
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss="mse")

#training and saving model
val_best = 9999
train_best_index = 0
for i_train in range(20):

    history = model.fit_generator(data_generator('train',128),
            samples_per_epoch=8036*3*2*training_percent, nb_epoch=1,validation_data=data_generator('valid',64),
                        nb_val_samples=8036*3*2*(1.0-training_percent))

    model_json = '/users/PAS0947/osu8077/new/udacity/P3-CarND-Behavioral-Cloning/model_' + str(i_train) + '.json'
    model_h5 = '/users/PAS0947/osu8077/new/udacity/P3-CarND-Behavioral-Cloning/model_' + str(i_train) + '.h5'
    model_save(model_json,model_h5)

    val_loss = history.history['val_loss'][0]
    if val_loss < val_best:
        train_best_index = i_train 
        val_best = val_loss
        model_json = '/users/PAS0947/osu8077/new/udacity/P3-CarND-Behavioral-Cloning/model_best.json'
        model_h5 = '/users/PAS0947/osu8077/new/udacity/P3-CarND-Behavioral-Cloning/model_best.h5'
        model_save(model_json,model_h5)
print('Best model found at iteration # ' + str(train_best_index))
print('Best Validation score : ' + str(np.round(val_best,4)))
print('Finish save model architecture and weights')

