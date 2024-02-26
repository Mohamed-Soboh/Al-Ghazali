# -*- coding: utf-8 -*-
"""
Created on Sat May  9 13:04:28 2020

@author: ZEEV
"""
import numpy as np
def get_vec(n_model, dim, token):
    vec = np.zeros(dim)
    # print(token)
    if token not in n_model.wv:
        _count = 0
        for w in token.split('_'):
            if w in n_model.wv:
                print(w)
                _count += 1
                vec += n_model.wv[w]
        if _count > 0:
            vec = vec / _count
    else:
        vec = n_model.wv[token]
    return vec