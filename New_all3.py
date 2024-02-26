# -*- coding: utf-8 -*-
"""
Created on Sat May  9 12:33:30 2020

@author: ZEEV
"""


from __future__ import print_function
from __future__ import division

# import gensim
import shutil
import pickle
import numpy as np
import os
from os.path import join
import sys
# import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew

currentDirectory = os.getcwd()
pr_dir=join(currentDirectory,'pr_functions')

sys.path.append(pr_dir)
currentDirectory = os.getcwd()

# from clean_str import clean_str
# from convert_text_from_file import convert_text_from_file
# from divide_text import divide_text
from divide_texts_in_dir import divide_texts_in_dir
from embedding_data import embedding_data
# from get_vec import get_vec
# from prepare_input import prepare_input
from runningMean import runningMean
from run_model3 import run_model
# from save_ndarray import save_ndarray
from train_model3 import train_model




str0='\\angl'
# train_neg_dir_path = './s1/save_dir/'
# train_pos_dir_path = './s2/save_dir/'
model_path = './angl/eng_textT.pkl'
emb_dim = 300
n=512*16
fact=3
# Embedding params

data_batch_size = 64
# Model params for convolution
kernel_size =[3,5,16]
# Filters for conv layers
nb_filter = 500
# Number of units in the dense layer - number of neurons in the dense layer
dense_outputs = 256
# Number of units in the final output layer. Number of classes.
cat_output = 2  # - we have 2 authors
# size of the max pooling window
pool_size = 1
# Optimizer params
learning_rate = 0.01
momentum = 0.9
decay = 1
nb_epoch = 10
i1=1
if __name__ == '__main__':
    dirtext=[]
    dirnpy=[]
    labels0=[]
    currentDirectory = os.getcwd()
    dataset_path=currentDirectory
    for jj in list(range(1,3)):
     print(jj)
     save_dir= divide_texts_in_dir(currentDirectory+str0+'\\s'+str(jj),n)
     labels0.append(len(os.listdir(save_dir)))
     dirtext.append(save_dir+'\\save_dir')
     save_dir=embedding_data(model_path, save_dir, emb_dim)
     dirnpy.append(save_dir)
    
    i1=labels0.index(max(labels0))
    i2=labels0.index(min(labels0))
     
    niter=1
    print(niter)
    save_dir=divide_texts_in_dir(currentDirectory+str0+'\\s3', n)
    dir_source=embedding_data(model_path, save_dir, emb_dim)
    # fact=2
    path0=dirtext[i1]+'\\temp'
    path1=dirtext[i2]+'\\temp'
    train_neg_dir_path=path0
    train_pos_dir_path=path1
     
    mean0=[]
    meaena=[]
    cl0=[]
    cl0a=[]
    hist0=[]
    quality=[]
    cl0N=[]
    sw=[]
            
              
    if os.path.isdir(path0):         
         files=os.listdir(path0)
         for f in files:
          os.remove(path0 +'\\'+ f)
    else:         
         os.mkdir(path0)  
         
         
         
    if os.path.isdir(path1):         
         files=os.listdir(path1)
         for f in files:
          os.remove(path1 +'\\'+ f)
    else:         
         os.mkdir(path1) 
              
    for filename in os.listdir(dirnpy[i1]): 
      if filename.endswith(".npy"):
         for jjk in range(0,fact):
           shutil.copy(dirnpy[i1]+'\\'+filename, path0+'\\'+str(jjk)+filename)
           
    for filename in os.listdir(dirnpy[i2]): 
      if filename.endswith(".npy"):
         for jjk in range(0,int(fact*max(labels0)/min(labels0))):
           shutil.copy(dirnpy[i2]+'\\'+filename, path1+'\\'+str(jjk)+filename)
     
    for jnum  in range(niter):
     print(jnum) 
     model, history = train_model(
            path0,
            path1,
            emb_dim, 
            data_batch_size, 
            kernel_size, 
            nb_filter, 
            pool_size, 
            dense_outputs, 
            cat_output, 
            learning_rate, 
            momentum, 
            decay, 
            nb_epoch,0)
     
     means, classes = run_model(dir_source, model,data_batch_size,emb_dim)
     
     print(i1)
     if(i1==1): classes=1-classes
     cl0N.append(np.mean(classes))
     print(cl0N)
     cl0.append(classes)
     
     
     sw.append(skew(classes))
     print(sw)
     # print(1-sum(np.array(sw)<0)/len(sw))
     hist0.append(history)
     quality.append(history.history['val_accuracy'][-1])
     # print(quality)
     np.save('mean0'+'+srt0+ '+str(n)+' '+str(kernel_size)+'.npy', mean0, allow_pickle = True)
     np.save('cl0'+' +srt0+ '+str(n)+' '+str(kernel_size)+'.npy', cl0, allow_pickle = True)
     np.save('hist0'+' +srt0+ '+str(n)+' '+str(kernel_size)+'.npy', hist0, allow_pickle = True)
     
     
     _ = plt.hist(classes, bins='auto')
     plt.show()
     
     asdf=runningMean(classes,data_batch_size*2)
     fig, ax = plt.subplots(1, 1, figsize=(6, 4))
     ax.plot(range(len(asdf)), asdf, '--k')
     ll=[]
     ll.append(n)
     ll.append(emb_dim)
     ll.append(data_batch_size)
     ll.append(kernel_size)
     ll.append(model_path)
     ll.append(i1)
     
     with open('parametrs1', 'wb') as fp:
         pickle.dump(ll,fp)
         
         # pickle.dump(n,emb_dim,data_batch_size,kernel_size,model_path,i1, fp)
         
     # os.remove(path0)
    # np.savetxt("means.csv",means, delimiter=",", fmt='%s')
# ZZ=Z*(Z>(Z.max()*0.1))
    
# 