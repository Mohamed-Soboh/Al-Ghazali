# -*- coding: utf-8 -*-
"""
Created on Sat May  9 13:06:17 2020

@author: ZEEV
"""
# import pickle
import re, os
import numpy as np
from os.path import join
import shutil

def prepare_input(path1, path2, emb_dim, data_batch_size, number_of_batches = None, usePreLoadedData = False):

    calculate_number_of_batches = False
    if number_of_batches is None:
        calculate_number_of_batches = True

    if usePreLoadedData:
        print('Using existing data')
        # Full data, 2 authors ~21k
        X = np.load(path1)
        Y = np.load(path2)
        return X, Y
    else:
        print('Preparing data')

        print('Batch size: ' + str(data_batch_size))
        print('Number of batches: ' + str(number_of_batches))

        dataset_path = path1
        labels = os.listdir(dataset_path)
        if 'tmp' in labels:
            labels.remove('tmp')
        #print(labels)
        # classifications = np.empty()
        index = 0
        Y1 = []
        X1 = np.empty((0, data_batch_size, emb_dim))
        # iterating over authors
        # Creating temporary dir to save data
        tempDirectory = os.path.join('tmp')
        if not os.path.exists(tempDirectory):
                os.makedirs(tempDirectory)
        for label in labels:
            # print(label) 
            # if label.endswith!=('.npy'):continue
            file=join(dataset_path, label)

        
            
            numpy_data = np.load(file)

            if calculate_number_of_batches:
               number_of_batches = len(numpy_data)//data_batch_size

            if number_of_batches > 0 :     # The book might not have enough parts
             
                    split_up_data = np.asarray(np.split(numpy_data[:number_of_batches*data_batch_size], number_of_batches))
                    y_ = np.zeros(number_of_batches)
                    Y1 = np.concatenate((Y1, y_), axis=0)
                    X1 = np.concatenate((X1, split_up_data), axis=0)
           
        dataset_path = path2
        labels = os.listdir(dataset_path)
        index += 1
        Y2 = []
        X2 = np.empty((0, data_batch_size, emb_dim))
        # iterating over authors
        # Creating temporary dir to save data
        tempDirectory = os.path.join('tmp')
        if not os.path.exists(tempDirectory):
                os.makedirs(tempDirectory)
        for label in labels:
            # print(label) 
            # if label.endswith!=('.npy'):continue
            file=join(dataset_path, label)

            
            numpy_data = np.load(file)

            if calculate_number_of_batches:
               number_of_batches = len(numpy_data)//data_batch_size

            if number_of_batches>0 :     # The book might not have enough parts
             
                    split_up_data = np.asarray(np.split(numpy_data[:number_of_batches*data_batch_size], number_of_batches))
                    y_ = np.zeros(number_of_batches)+1
                    Y2 = np.concatenate((Y2, y_), axis=0)
                    X2 = np.concatenate((X2, split_up_data), axis=0)
           
        X=np.concatenate((X1, X2), axis=0)
        Y=np.concatenate((Y1, Y2), axis=0)

        return X, Y
    