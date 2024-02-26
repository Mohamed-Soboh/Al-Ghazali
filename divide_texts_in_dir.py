# -*- coding: utf-8 -*-
"""
Created on Sat May  9 13:06:17 2020

@author: ZEEV
"""
# import pickle
import os
from clean_str import clean_str
import numpy as np
from os.path import join
import shutil
from divide_text import divide_text

def divide_texts_in_dir(dataset_path,n):
    # Path to temporary dir to save data
    save_dir = join(dataset_path, 'save_dir/')
#    print(save_dir)
    try:
      os.remove(save_dir)
    except:  
        print(save_dir)
    # Deleting temporary dir to save data
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir, ignore_errors = True)
    
    # List of file names
    labels = os.listdir(dataset_path)

    # Creating temporary dir to save data
    
    try:
     os.mkdir(save_dir)
    except:
     print(save_dir+" The directory existes")

    for label in labels:
        if label.endswith(".txt"):
            text_file_path = join(dataset_path, label) # Path to file
            file = open(text_file_path)
                        # , encoding = 'utf-8')
            text = file.read()
            L = divide_text(text, n)
            i = 0
            for l in L:
                save_file_name = join(save_dir, label.replace('.txt', '')) + '_' + str(i) + '.txt'
#                save_file = codecs.open(save_file_name, "w", "utf-8")
                

                save_file = open(save_file_name, 'w', encoding = 'utf-8')
#                save_file.write(l'\ufeff')
                save_file.write(l)
                save_file.close()
                i += 1
    return save_dir