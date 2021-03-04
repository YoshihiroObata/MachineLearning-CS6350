# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 19:10:39 2021

@author: Yoshihiro Obata
"""
# %% importing packages
import pandas as pd
import numpy as np
from LinearRegression import LMS, run_LMS_GD, run_LMS_SGD
import matplotlib.pyplot as plt
# %% importing data
cols = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr',
        'y']
train = pd.read_csv('concrete/concrete/train.csv', names=cols)
test = pd.read_csv('concrete/concrete/test.csv', names=cols)

# %% testing
lms = LMS(train, 1000, 0.001, verbose=True)
w_final = run_LMS_GD(lms)

# %%
# lrs = 1/(2**np.arange(10))
lrs = np.linspace(0.02, 0.01, 10)
for lr in lrs:
    bgd = LMS(train, 6000, lr)
    w = run_LMS_GD(bgd)
    if bgd.converged:
        break

# %%
# lrs = np.linspace(0.02, 0.01, 10)
lrs = [0.1, 0.05, 0.01, 0.005, 0.001]
for lr in lrs:
    sgd = LMS(train, 3000, lr)
    w = run_LMS_SGD(sgd)
    plt.figure()
    plt.plot(np.arange(3000),sgd.J)
    if sgd.converged:
        break
    
# %%
J = bgd.J[lms.J != 0]
plt.plot(np.arange(len(J)), J)
