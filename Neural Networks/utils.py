# -*- coding: utf-8 -*-
"""
Some helper functions for the neural network implementation

@author: Yoshihiro Obata
"""
import numpy as np

# forward pass helper functions
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# 
def shuffle(train, labels):
    idx = np.random.choice(np.arange(len(train)), len(train),
                               replace=False)
    shuff_data = train[idx,:]
    shuff_lab = labels[idx]
    return shuff_data, shuff_lab

def schedule(lr0, d0, t):
    return lr0/(1 + (lr0*t)/d0)

def linear(x, w):
    return np.sum(x*w)

def node(x, w, verbose=False, desc=None):
    summation = linear(x, w)
    if verbose:
        summary = f""" {desc}
        features = {x}
        weights = {w}
        linear = {summation}
        sigmoid = {sigmoid(summation)}
        """
        print(summary)
    return sigmoid(summation)