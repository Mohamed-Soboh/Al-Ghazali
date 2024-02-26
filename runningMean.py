# -*- coding: utf-8 -*-
"""
Created on Sat May  9 12:31:06 2020

@author: ZEEV
"""
import numpy as np
def runningMean(x, N):
    y = np.zeros((len(x),))
    for ctr in range(len(x)):
         y[ctr] = np.sum(x[ctr:(ctr+N)])
    return y/N