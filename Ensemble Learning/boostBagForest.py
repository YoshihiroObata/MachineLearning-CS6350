# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:52:18 2021

@author: Yoshihiro Obata
"""
# %%
import pandas as pd
import numpy as np
from AdaBoost import AdaBoost, run_adaBoost
from Bagging import Bagging, run_bagging, apply_bagging

# %%
cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
        'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
        'previous', 'poutcome', 'y']
train = pd.read_csv('bank/train.csv', names=cols)
test = pd.read_csv('bank/test.csv', names=cols)

# %% boosting
adaInit = AdaBoost(train, 5)
learners, alphas = run_adaBoost(adaInit)

# %% bagging
key = {'no':-1, 'yes':1}
bagInit = Bagging(100, 10, train, numerical=True, key=key)
# trees, votes = run_bagging(bagInit)
run_bagging(bagInit)
err = apply_bagging(bagInit, train)

# %%
