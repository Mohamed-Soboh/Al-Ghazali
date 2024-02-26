# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:56:03 2020

@author: ZEEV
"""
import numpy as np

def save_ndarray(ndarray, filename):
#    Store a single NumPy array on an .npy binary file.
#    Args:
#        ndarray (ndarray): Array to be stored on disk.
#        filename (str): The file's name on disk. No need for .npy extension
#    Returns:
#        None
    
    if ~filename.endswith('.npy'):
        filename += '.npy'

    try:
        np.save(filename, ndarray, allow_pickle = False)
    except ValueError:
        np.save(filename, ndarray, allow_pickle = True)

