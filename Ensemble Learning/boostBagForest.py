# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:52:18 2021

@author: Yoshihiro Obata
"""
# %%
import pandas as pd
import numpy as np
from AdaBoost import AdaBoost, run_adaBoost, apply_adaBoost
from Bagging import Bagging, run_bagging, apply_bagging
import matplotlib.pyplot as plt
import time

# %%
cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
        'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
        'previous', 'poutcome', 'y']
train = pd.read_csv('bank/train.csv', names=cols)
test = pd.read_csv('bank/test.csv', names=cols)

# %% boosting
tic = time.perf_counter()
T = 500
key = {'no':-1, 'yes':1}
adaInit = AdaBoost(train, T, key=key)
run_adaBoost(adaInit)

err_AdaTrain = apply_adaBoost(adaInit, train)
stump_err_train = adaInit.errs
toc = time.perf_counter()
print('Time to train and apply AdaBoost with {} trees was {:0.4f} seconds.\n'.format(T, toc-tic))

tic = time.perf_counter()
err_AdaTest = apply_adaBoost(adaInit, test)
stump_err_test = adaInit.errs
toc = time.perf_counter()
print('Time to apply AdaBoost to test set with {} trees was {:0.4f} seconds.\n'.format(T, toc-tic))

# %% plotting adaboost
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(np.arange(T), err_AdaTrain, 'k', label='Training', linewidth=2)
ax.plot(np.arange(T), err_AdaTest, 'r--', label='Test', linewidth=2)
ax.tick_params(labelsize = 16, size = 10, width = 2)
for spine in ax.spines:
    ax.spines[spine].set_linewidth(2)
plt.xlim([0,adaInit.T])
plt.xlabel('Iterations', fontsize=18)
plt.ylabel('Error', fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)

fig,ax = plt.subplots(figsize=(10,5))
plt.plot(np.arange(T), stump_err_train, 'k', label='Training', linewidth=2)
plt.plot(np.arange(T), stump_err_test, 'r--', label='Test', linewidth=2)
ax.tick_params(labelsize = 16, size = 10, width = 2)
for spine in ax.spines:
    ax.spines[spine].set_linewidth(2)
plt.xlim([0,adaInit.T])
plt.xlabel('Iterations', fontsize=18)
plt.ylabel('Stump Error', fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)
if T == 500:
    plt.savefig('P2AdaBoost500.png', dpi=150, bbox_inches='tight')
else:
    plt.savefig('P2AdaBoost.png', dpi=150, bbox_inches='tight')

# %% bagging
tic = time.perf_counter()
m = 1000
T_bag = 500
bagInit = Bagging(m, T_bag, train, numerical=True, key=key)
run_bagging(bagInit)
# run_bagging_parallel(bagInit)

err_bag_train = apply_bagging(bagInit, train)
toc = time.perf_counter()
print('Time to train and apply Bagging with {} trees was {:0.4f} seconds.\n'.format(T_bag, toc-tic))

tic = time.perf_counter()
err_bag_test = apply_bagging(bagInit, test)
toc = time.perf_counter()
print('Time to apply Bagging to test set with {} trees was {:0.4f} seconds.\n'.format(T_bag, toc-tic))

# %%
fig, ax = plt.subplots(figsize=(10,5))
plt.plot(np.arange(T_bag), err_bag_train, 'k', label='Training', linewidth=2)
plt.plot(np.arange(T_bag), err_bag_test, 'r--', label='Test', linewidth=2)
ax.tick_params(labelsize = 16, size = 10, width = 2)
for spine in ax.spines:
    ax.spines[spine].set_linewidth(2)
plt.xlim([0,adaInit.T])
plt.xlabel('Number of Trees', fontsize=18)
plt.ylabel('Error', fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)
if T_bag == 500:
    plt.savefig('P2Bagging500.png', dpi=150, bbox_inches='tight')
else:
    plt.savefig('P2Bagging.png', dpi=150, bbox_inches='tight')

# %% 
# N = 10
# single_trees = []
# all_errs_bag = []
# for n in range(N):
#     bagInit = Bagging(m, T_bag, train, numerical=True, key=key)
#     run_bagging(bagInit)
#     single_trees.append(bagInit.treesInit[0].tree)
#     all_errs_bag.append(apply_bagging(bagInit, test))

# %% random forest
err_forest_train = []
err_forest_test = []
for Gsize in [2, 4, 6]:
    tic = time.perf_counter()
    forestInit = Bagging(m, T_bag, train, numerical=True, key=key, randForest=True,
                         Gsize=Gsize)
    run_bagging(forestInit)
    
    err_forest_train.append(apply_bagging(forestInit, train))
    toc = time.perf_counter()
    print('Time to train and apply random forest with {} trees was {:0.4f} seconds.\n'.format(T_bag, toc-tic))
    tic = time.perf_counter()
    err_forest_test.append(apply_bagging(forestInit, test))
    toc = time.perf_counter()
    print('Time to apply random forest to test set with {} trees was {:0.4f} seconds.\n'.format(T_bag, toc-tic))

# %%
fig, ax = plt.subplots(figsize=(10,5))
colors = ['red', 'blue', 'black']
train_lab = ['|G|=2, Train', '|G|=4, Train', '|G|=6, Train']
test_lab = ['|G|=2, Test', '|G|=4, Test', '|G|=6, Test']
for i in range(len([2, 4, 6])):
    plt.plot(np.arange(T_bag), err_forest_train[i], label=train_lab[i], 
             color=colors[i], linewidth=2)
    plt.plot(np.arange(T_bag), err_forest_test[i], '--', label=test_lab[i], 
             color=colors[i], linewidth=2)
ax.tick_params(labelsize = 16, size = 10, width = 2)
for spine in ax.spines:
    ax.spines[spine].set_linewidth(2)
plt.xlim([0,adaInit.T])
plt.xlabel('Number of Trees', fontsize=18)
plt.ylabel('Error', fontsize=18)
plt.legend(fontsize=16)
plt.grid(True)
if T_bag == 500:
    plt.savefig('P2RandForest500.png', dpi=150, bbox_inches='tight')
else:
    plt.savefig('P2RandForest.png', dpi=150, bbox_inches='tight')