import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math

# from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from tensorflow.keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def dice_loss(y_true, y_pred, smooth=1e-12):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    mseloss = K.sum(K.square(y_true_f-y_pred_f))
    
    intersection = K.sum(y_true_f * y_pred_f)
    dice = K.mean( (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth) )
    return 0.5*(1-dice)+1.5*K.mean(mseloss)



def plot_sample(X, preds,output_dir, ix=None ,im_width=128):
    lung_area = 0
    ild_area = 0
    
    for i in range(im_width):
        for j in range(im_width):
            if X[ix][i][j] > 0.03:
                lung_area+=1
    
    
    output = preds[ix].squeeze()
    new = []
    for i in range(im_width):
        new.append([])    
        for j in range(im_width):
            new[i].append(output[i][j])

    
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    I = np.dstack([X[ix, ..., 0], X[ix, ..., 0], X[ix, ..., 0]])
    for i in range(im_width):
        for j in range(im_width):
            new[i][j] = sigmoid(new[i][j])
            if new[i][j] > 0.57:
                new[i][j] = 255
                I[i, j, :] = [1, 0, 0]
                ild_area+=1
            else:
                new[i][j] = 0
                
    ax.set_title('overlap')    
    ax.imshow(X[ix, ..., 0],cmap='gray')
    ax.imshow(I,alpha=0.5)

    fig.savefig(output_dir+str(ix)+'.png')
    rv = ild_area/lung_area
    return rv

def pred_ild(file_list,path_data,output_dir):
    #scale data's height and width
    im_width = 128
    im_height = 128
    index = 0
    rf_list = []
    rv_list = []
    
    #preprocess the data
    x_new = np.zeros(( len(file_list), im_height, im_width, 1), dtype=np.float32)
    for fname in file_list:
        img = load_img(path_data + fname, grayscale=True)
        x_img = img_to_array(img)
        x_img = resize(x_img, (im_width, im_height, 1), mode='constant', preserve_range=True)
        # Save images
        x_new[index, ..., 0] = x_img.squeeze() / 255
        index += 1
    #declare the model
    input_img = Input((im_height, im_width, 1), name='img')
    model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
    model.compile(optimizer=Adam(), loss=dice_loss, metrics=["accuracy"])
    #load weight 
    model.load_weights('model-tgs-salt.h5')
    #predict the data
    preds_val = model.predict(x_new,batch_size=1, verbose=1)
    #plot the data
    for i in range(len(file_list)):
        t = plot_sample(x_new, preds_val,output_dir=output_dir,ix=i)
        rf_list.append(output_dir+str(i)+'.png')
        rv_list.append(t)
    return rf_list,rv_list

if __name__ == "__main__":
    file_list = ['ILD05_18.jpg','ILD05_39.jpg','ILD05_80.jpg']
    path_data='../../../Desktop/new5/img/'
    output_dir='../../output_pic_unet2/'
    
    return_file_list , return_value_list = pred_ild(file_list,path_data,output_dir)
    print(return_file_list)
    print(return_value_list)