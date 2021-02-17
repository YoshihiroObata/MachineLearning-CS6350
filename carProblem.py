# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 22:24:06 2021

@author: Yoshihiro Obata
"""

# %% importing packages
import numpy as np
import pandas as pd
from ID3 import decisionTree
from ID3 import run_ID3
from ID3 import applyTree
from ID3 import apply_ID3
from testingTrees import tester
import matplotlib.pyplot as plt

# %% importing the data and splitting it up
cols = list(pd.read_csv('car/data-desc.txt', skiprows=14))
train = pd.read_csv('car/train.csv', names=cols)
test = pd.read_csv('car/test.csv', names=cols)
    
attrTrain = np.array(train.iloc[:,:-1])
attrTest = np.array(test.iloc[:,:-1])
attrNames = cols[:-1]
labelsTrain = np.array(train.iloc[:,-1])
labelsTest = np.array(test.iloc[:,-1])

# %% training the ID3 algo for testing
carTreeInit = decisionTree(attrTrain, attrNames, labelsTrain)
carTree = run_ID3(carTreeInit)

# %% applying the ID3 algo for testing
errinit = applyTree(carTree, test, labelsTest)
errs, total_err = apply_ID3(errinit)

# %% making trees
methods = ['entropy', 'ME', 'gini']
datTrain = [attrTrain, labelsTrain, train]
datTest = [attrTest, labelsTest, test]
depths = len(attrNames)
# DTs = [[0]*len(attrNames)]*len(methods)
# train_err = np.zeros((len(methods), len(attrNames)))
# test_err = np.zeros((len(methods), len(attrNames)))
# for i, method in enumerate(methods):
#     for j in range(len(attrNames)):
#         # print(method)
#         d = j + 1
#         # print(method, d)
#         carTreeInit2 = decisionTree(attrTrain, attrNames, labelsTrain, depth=d,
#                                     method=method)
#         DTs[i][j] = run_ID3(carTreeInit2)
#         errinit_train = applyTree(DTs[i][j], train, labelsTrain)
#         _, train_err[i,j] = apply_ID3(errinit_train)
        
#         errinit_test = applyTree(DTs[i][j], test, labelsTest)
#         _, test_err[i,j] = apply_ID3(errinit_test)
        
# %% gathering errs
# for i, method in enumerate(methods):
#     for j in range(len(attrNames)):
#         # d = j + 1
#         errinit_train = applyTree(DTs[i][j], train, labelsTrain)
#         _, train_err[i,j] = apply_ID3(errinit_train)
        
#         errinit_test = applyTree(DTs[i][j], test, labelsTest)
#         _, test_err[i,j] = apply_ID3(errinit_test)
#         print(test_err)

errinit = tester(methods, attrNames, datTrain, datTest, depths=depths)
train_err_car, test_err_car = tester.test(errinit)
        
# %% plotting results and calc avgs
# avg_train = np.mean(train_err, axis=1)
# avg_test = np.mean(test_err, axis=1)
avg_train = np.mean(train_err_car, axis=1)
avg_test = np.mean(test_err_car, axis=1)
color = ['r', 'b', 'k']
label = ['Entropy', 'Majority Error', 'Gini index']

depth = np.linspace(1,6,6)
fig, ax = plt.subplots(figsize = (10,7))
for method in range(len(methods)):
    plt.plot(depth, train_err_car[method,:], color=color[method],
             label = label[method]+' (training)', linewidth = 3)
    plt.plot(depth, test_err_car[method,:], linestyle = '--', color=color[method],
             label = label[method]+' (testing)', linewidth = 3)
plt.legend(fontsize = 16, loc = (1.025,0.63))
plt.xlabel('Tree Depth', fontsize = 18)
plt.ylabel('Accuracy', fontsize = 18)
ax.tick_params(labelsize = 16, size = 10, width = 2)
plt.ylim([0.65,1])
plt.xlim([0.5,6])
for spine in ax.spines:
    ax.spines[spine].set_linewidth(2)
# plt.savefig('accuracyCAR.png', dpi = 150, bbox_inches = 'tight')
print('Training errors:\nEntropy={}\nMajority Error={}\nGini Index={}'.format(
    avg_train[0], avg_train[1], avg_train[2]))
print('\nTesting errors:\nEntropy={}\nMajority Error={}\nGini Index={}'.format(
    avg_test[0], avg_test[1], avg_test[2]))