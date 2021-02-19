# -*- coding: utf-8 -*-
"""
HW1 Car problem

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
import time

# %% importing the data and splitting it up
cols = list(pd.read_csv('car/data-desc.txt', skiprows=14))
train0 = pd.read_csv('car/train.csv', names=cols)
test0 = pd.read_csv('car/test.csv', names=cols)
    
attrTrain0 = np.array(train0.iloc[:,:-1])
attrTest0= np.array(test0.iloc[:,:-1])
attrNames0 = cols[:-1]
labelsTrain0 = np.array(train0.iloc[:,-1])
labelsTest0 = np.array(test0.iloc[:,-1])

# %% training the ID3 algo for testing
carTreeInit = decisionTree(attrTrain0, attrNames0, labelsTrain0, method = 'entropy')
carTree = run_ID3(carTreeInit)

# %% applying the ID3 algo for testing
car_errinit = applyTree(carTree, test0)
errs0, total_err0 = apply_ID3(car_errinit)

# %% making trees
tic = time.perf_counter()
methods = ['entropy', 'ME', 'gini']
datTrain0 = [attrTrain0, labelsTrain0, train0]
datTest0 = [attrTest0, labelsTest0, test0]
depths0 = len(attrNames0)

errinit = tester(methods, attrNames0, datTrain0, datTest0, depths=depths0)
train_err_car, test_err_car = tester.test(errinit)
toc = time.perf_counter()
print('Time for car code is {:0.4f} seconds.'.format(toc-tic))
        
# %% plotting results and calc avgs
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
plt.savefig('accuracyCAR.png', dpi = 150, bbox_inches = 'tight')
# print('Training errors:\nEntropy={}\nMajority Error={}\nGini Index={}'.format(
#     avg_train[0], avg_train[1], avg_train[2]))
# print('\nTesting errors:\nEntropy={}\nMajority Error={}\nGini Index={}'.format(
#     avg_test[0], avg_test[1], avg_test[2]))