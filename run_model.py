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

import os
import numpy as np
from os.path import join
def run_model(path, model,data_batch_size,emb_dim):
  
    X1 = np.empty((0, data_batch_size, emb_dim))
    labels = os.listdir(path)
    labels = [label for label in labels if label.endswith(".npy")]
    labels.sort()
    s1 = []
    
    for label in labels:
        # print(label)
        file = join(path, label)  
        numpy_data = np.load(file)
        number_of_batches = len(numpy_data)//data_batch_size    
        if(number_of_batches > 0):
            split_up_data = np.asarray(np.split(numpy_data[: number_of_batches * data_batch_size], number_of_batches))
            s1.append(model.predict_classes(split_up_data))
            # with open('outfile', 'wb') as fp:
            #     pickle.dump(s1, fp)
            X1 = np.concatenate((X1, split_up_data), axis = 0)
  
    s2 = [s.mean() for s in s1]
    labels_init = [l[0 : 3] for l in labels]
    labels_uniqe = list(set(labels_init))
    labels_uniqe.sort()
    labels_nums = [labels_init.count(l) for l in labels_uniqe]

    pos = 0
    means = []
    for l in labels_nums:
        means.append(s2[pos : pos + l])
        pos += l 
  
    classes = model.predict_classes(X1)
    
    return (means, classes)
