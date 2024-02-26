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
from clean_str import clean_str

def divide_text(text, n):
    T = clean_str(text)
    words = T.split(' ')
    L = [words[i : i + n] for i in range(0, len(words), n)]
    L = [' '.join(l) for l in L]
    return L