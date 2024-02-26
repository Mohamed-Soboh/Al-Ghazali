# -*- coding: utf-8 -*-
"""
Created on Sat May  9 13:06:17 2020

@author: ZEEV
"""
import pickle
import re, os
import numpy as np
from os.path import join
import shutil

def embedding_train_data(model_path, train_neg_dir_path, train_pos_dir_path):
    ##############################
    # Converting text to vectors #
    ##############################
    
    # t_model = gensim.models.Word2Vec.load(model_path)
    # t_model=gensim.models.Word2Vec.load(model_path)
    with open(model_path, 'rb') as f:  # Python 3: open(..., 'wb')
     t_model = pickle.load(f)

    

    # pickle.dump(emb_dim, open('save.p', 'wb'))
    
    # Path to dir with data
    dataset_path = train_neg_dir_path
    
    # Path to temporary dir to save data
    save_dir = join(dataset_path, 'save_dir/')
    
    # Deleting temporary dir to save data
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir, ignore_errors = True)
    
    # List of file names
    labels = os.listdir(dataset_path)
    labels = [label for label in labels if label.endswith(".txt")]
    labels.sort()
    
    # Creating temporary dir to save data
    os.mkdir(save_dir)
    
    for label in labels:
        text_file_path = join(dataset_path, label) # Path to file
        print(text_file_path)
        vectors = convert_text_from_file(text_file_path, emb_dim, t_model) # Convert file to vectors
        asp = np.asarray(vectors)
        save_text_file_path = join(save_dir, label.replace('.txt', '')) # Remove extention from file name
        save_ndarray(asp, save_text_file_path) # Save a file as array of vectors

    # Path to dir with data 
    dataset_path = train_pos_dir_path
    
    # Path to temporary dir to save data
    save_dir = join(dataset_path, 'save_dir/')
    
    # Deleting temporary dir to save data
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir, ignore_errors = True)
    
    # List of file names
    labels = os.listdir(dataset_path)
    labels = [label for label in labels if label.endswith(".txt")]
    labels.sort()
    
    # Creating temporary dir to save data
    os.mkdir(save_dir)    
    
    for label in labels:
        text_file_path = join(dataset_path, label) # Path to file
        print(text_file_path)
        vectors = convert_text_from_file(text_file_path, emb_dim, t_model) # Convert file to vectors
        asp = np.asarray(vectors)
        save_text_file_path = join(save_dir, label.replace('.txt', '')) # Remove extention from file name
        save_ndarray(asp, save_text_file_path) # Save a file as array of vectors 

