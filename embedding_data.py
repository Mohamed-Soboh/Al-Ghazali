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
from convert_text_from_file import convert_text_from_file
from save_ndarray import save_ndarray

def embedding_data(model_path, dataset_path, emb_dim):
    ##############################
    # Converting text to vectors #
    ##############################
    
    # t_model = gensim.models.Word2Vec.load(model_path)
    # t_model=gensim.models.Word2Vec.load(model_path)
    with open(model_path, 'rb') as f:  # Python 3: open(..., 'wb')
     t_model = pickle.load(f)



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
    print(dataset_path)
    for label in labels:
        if label.endswith(".txt"):
            text_file_path = join(dataset_path, label) # Path to file
    #        print(text_file_path)
            vectors = convert_text_from_file(text_file_path, emb_dim, t_model) # Convert file to vectors
            asp = np.asarray(vectors)
            save_text_file_path = join(save_dir, label.replace('.txt', '')) # Remove extention from file name
            save_ndarray(asp, save_text_file_path) # Save a file as array of vectors

    return(save_dir)
