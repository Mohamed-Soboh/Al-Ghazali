# -*- coding: utf-8 -*-
"""
Created on Sat May  9 13:06:17 2020

@author: ZEEV
"""
# import pickle
# import re, os
import numpy as np
# from os.path import join
# import shutil
from tensorflow import keras
from keras.optimizers import SGD
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import MaxPooling1D, Conv1D
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, concatenate
from keras.layers import *
from keras.models import Model
from keras import regularizers
from keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from prepare_input import prepare_input


def train_model(path1,
                path2,
                emb_dim, 
                data_batch_size, 
                kernel_size , 
                nb_filter, 
                pool_size , 
                dense_outputs = 256, 
                cat_output = 10, 
                learning_rate = 0.001, 
                momentum = 0.9, 
                decay = 0, 
                nb_epoch = 12,
                usePreL=0,
                mod_name="model_eng"):
    
    np.random.seed(0)
    print('Loading and preparing data...')
    
    if(usePreL==0):
     X, Y = prepare_input(path1, path2, emb_dim,data_batch_size, number_of_batches = None, usePreLoadedData = False)
     np.save('XX',X,allow_pickle = True)
     np.save('YY',Y,allow_pickle = True)
    else:
     X = np.load('XX.npy')
     Y = np.load('YY.npy')
     
    print('Data:')
    print(X.shape)
    print(Y.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1)
    print('Testing data: ' + str(len(X_test)))
    print('Training data: ' + str(X_train.shape) + ' ' + str(Y_train.shape))
    
    # Creating a model
    model = Sequential()
    model.add(Conv1D(filters = nb_filter, kernel_size = kernel_size[0], padding = 'valid', activation = 'relu', input_shape = (data_batch_size, emb_dim)))
              # ,kernel_regularizer=regularizers.l2(0.01))
    model.add(MaxPooling1D(pool_size = pool_size))
    model.add(Conv1D(filters = nb_filter, kernel_size = kernel_size[1], padding = 'valid', activation = 'relu', input_shape = (data_batch_size, emb_dim)))
              # ,kernel_regularizer=regularizers.l2(0.01))
    model.add(MaxPooling1D(pool_size = pool_size))
    model.add(Conv1D(filters = nb_filter, kernel_size = kernel_size[2], padding = 'valid', activation = 'relu', input_shape = (data_batch_size, emb_dim)))
              # ,kernel_regularizer=regularizers.l2(0.01))
    model.add(MaxPooling1D(pool_size = pool_size))
    model.add(Flatten())
    model.add(Dense(units=dense_outputs, activation='relu'))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(cat_output, activation = 'softmax', name = 'output'))
    # kernel_regularizer=regularizers.l1(0.001),activity_regularizer=regularizers.l1(0.001),bias_regularizer=l2(0.001)))
              # ,
              #   activity_regularizer=regularizers.l1(0.005)))
    
    model.compile(loss = keras.losses.categorical_crossentropy, optimizer = 'adam', metrics = ['accuracy'])
    # print(model.summary())
    print('Fit model...')
    
    Y_train = keras.utils.to_categorical(Y_train)
    history = model.fit(X_train, Y_train, validation_split=0.25,epochs = nb_epoch, verbose = 2)
    
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()  
    
    model.save(mod_name+".h5")
    return model,history 
