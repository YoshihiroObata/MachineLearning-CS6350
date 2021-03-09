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
def var(data, m):
    s_sq = (1/(len(data.T) - 1))*np.sum((data.T - m)**2)
    return s_sq

# %% Change T here:
T = 50 # for AdaBoost
T_bag = 50 # for Bagging and Random Forest

if T == 50:
    print('Running a smaller number of trees just for example of functionality. Results are created using T=500')

# %%
cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
        'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
        'previous', 'poutcome', 'y']
train = pd.read_csv('bank/train.csv', names=cols)
test = pd.read_csv('bank/test.csv', names=cols)

# %% boosting
tic = time.perf_counter()
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
if T == 500:
    plt.savefig('P2AdaBoost500.png', dpi=150, bbox_inches='tight')
else:
    plt.savefig('P2AdaBoost.png', dpi=150, bbox_inches='tight')
    
fig,ax = plt.subplots(figsize=(10,5))
plt.plot(np.arange(T), stump_err_train, 'k', label='Training', linewidth=2)
plt.plot(np.arange(T), stump_err_test, 'r--', label='Test', linewidth=2)
ax.tick_params(labelsize = 16, size = 10, width = 2)
for spine in ax.spines:
    ax.spines[spine].set_linewidth(2)
plt.xlim([0,adaInit.T])
plt.xlabel('Iterations', fontsize=18)
plt.ylabel('Stump Error', fontsize=18)
plt.legend(fontsize=18, loc='upper right')
plt.grid(True)
if T == 500:
    plt.savefig('P2AdaBoostStump500.png', dpi=150, bbox_inches='tight')
else:
    plt.savefig('P2AdaBoostStump.png', dpi=150, bbox_inches='tight')

 # %% bagging
if T_bag == 50:
    print('Running a smaller number of trees just for example of functionality. Results are created using T=500')

tic = time.perf_counter()
m = 1000
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

# %% random forest
if T_bag == 50:
    print('Running a smaller number of trees just for example of functionality. Results are created using T=500')

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
    
# %% 
bag_h = pd.read_csv('h_bags0.csv')
tree_h = pd.read_csv('h_trees0.csv')
fx = test.iloc[:,-1]
fx = np.vectorize(key.get)(fx)

EXP_bag = np.mean(bag_h, axis=1)
EXP_tree = np.mean(tree_h, axis=1)

bias_bag = np.mean((fx - EXP_bag)**2)
var_bag = np.mean(var(bag_h, EXP_bag))
bias_tree = np.mean((fx - EXP_tree)**2)
var_tree = np.mean(var(tree_h, EXP_tree))

# %%
forest_h = pd.read_csv('h_rand_forest2.csv')
tree2_h = pd.read_csv('h_rand_trees2.csv')

EXP_forest = np.mean(forest_h, axis=1)
EXP_tree2 = np.mean(tree2_h, axis=1)

bias_forest = np.mean((fx - EXP_forest)**2)
var_forest = np.mean(var(forest_h, EXP_forest))
bias_tree2 = np.mean((fx - EXP_tree2)**2)
var_tree2 = np.mean(var(tree2_h, EXP_tree2))

# %% print summary of file
result_str = f"""------------------------------------------------------------------------------
Results of boosting, bagging, and random forest decision trees:
NOTE: T values for each technique are less than 500 only for TA testing.
      Values of T=500 are used in the report but not here to save grader time.
------------------------------------------------------------------------------
Boosting: Plots should be generated and saved in the working directory
    Parameters: T={T}, weak learner: decision stump
    Final training error:\t{err_AdaTrain[-1]}
    Final test error:\t\t{err_AdaTest[-1]}
------------------------------------------------------------------------------
Bagging: Plot should be generated and saved in the working directory
    Parameters: m={m}, T={T_bag}
    Final training error:\t{err_bag_train[-1]}
    Final test error:\t\t{err_bag_test[-1]}

    Bagging Experiment: Values obtained from experiment script
    Parameters: m=1000, m'=1000, T=500, iterations: 100
    Bagged trees bias:\t\t{np.round(bias_bag, 3)} 
    Single tree bias:\t\t{np.round(bias_tree, 3)} 
    Bagged trees variance:\t{np.round(var_bag, 3)}
    Single tree variance:\t{np.round(var_tree, 3)}
------------------------------------------------------------------------------
Random Forest: Plot should be generated and saved in the working directory
    Parameters: m={m}, T={T_bag}
    Final training errors:\t|G|=2: {err_forest_train[0][-1]},\t|G|=4: {err_forest_train[1][-1]},\t|G|=6: {err_forest_train[2][-1]}
    Final test errors:\t\t|G|=2: {err_forest_test[0][-1]},\t|G|=4: {err_forest_test[1][-1]},\t|G|=6: {err_forest_test[2][-1]}
    
    Random Forest Experiment: Values obtained from experiment script
    Parameters: m=1000, m'=1000, |G|=2, T=500, iterations: 100
    Random forest bias:\t\t{np.round(bias_forest, 3)} 
    Single tree bias:\t\t{np.round(bias_tree2, 3)} 
    Random forest variance:\t{np.round(var_forest, 3)}
    Single tree variance:\t{np.round(var_tree2, 3)}
------------------------------------------------------------------------------
"""
print(result_str)
if T==500 and T_bag==500:    
    txt = open('BoostBagRandForestResults500.txt', 'wt')
    n = txt.write(result_str)
    txt.close()
else:
    txt = open('BoostBagRandForestResults.txt', 'wt')
    n = txt.write(result_str)
    txt.close()