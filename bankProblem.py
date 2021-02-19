# -*- coding: utf-8 -*-
"""
Hw1, bank problem

@author: Yoshihiro Obata
"""
# %% import packages
import numpy as np
import pandas as pd
from ID3 import decisionTree
from ID3 import run_ID3
from ID3 import applyTree
from ID3 import apply_ID3
from testingTrees import tester
import matplotlib.pyplot as plt

# %% replace unknown
def replace_unk(df):
    cols = df.columns
    for col in cols:
        if isinstance(df[col][0], str):
            common = df[col].value_counts()
            idx = 0
            if common.index[0] == 'unknown':
                idx = 1
            df[col] = df[col].replace('unknown', common.index[idx])
    return df
    
# %% importing the data
cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
        'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
        'previous', 'poutcome', 'y']
train = pd.read_csv('bank/train.csv', names=cols)
test = pd.read_csv('bank/test.csv', names=cols)
    
attrTrain = np.array(train.iloc[:,:-1])
attrTest = np.array(test.iloc[:,:-1])
attrNames = cols[:-1]
labelsTrain = np.array(train.iloc[:,-1])
labelsTest = np.array(test.iloc[:,-1])

train_no_unk = replace_unk(train)
test_no_unk = replace_unk(test)
attrTrain_no_unk = np.array(train_no_unk.iloc[:,:-1])
attrTest_no_unk = np.array(test_no_unk.iloc[:,:-1])
labelsTrain_no_unk = np.array(train_no_unk.iloc[:,-1])
labelsTest_no_unk = np.array(test_no_unk.iloc[:,-1])

# %% training the ID3 algo for testing
bankTreeInit = decisionTree(attrTrain, attrNames, labelsTrain, numerical=True)
bankTree = run_ID3(bankTreeInit)

# %% applying the ID3 algo for testing
errinit = applyTree(bankTree, train, labelsTrain, numerical=True)
errs, total_err = apply_ID3(errinit)

# %% making trees
# takes a long time, might need to move somewhere so it's not storing everything
# maybe combine with getting erros in new testing function
methods = ['entropy', 'ME', 'gini']
datTrain = [attrTrain, labelsTrain, train]
datTest = [attrTest, labelsTest, test]
depths = len(attrNames)

errinit = tester(methods, attrNames, datTrain, datTest, depths=depths, num=True)
train_err_bank, test_err_bank = tester.test(errinit)

# %% testing for replaced unknown values
datTrain2 = [attrTrain_no_unk, labelsTrain_no_unk, train_no_unk]
datTest2 = [attrTest_no_unk, labelsTest_no_unk, test_no_unk]

errinit2 = tester(methods, attrNames, datTrain2, datTest2, depths=depths, num=True)
train_err_bank2, test_err_bank2 = tester.test(errinit2)
# %% plotting results and calc avgs
avg_train = np.mean(train_err_bank, axis=1)
avg_test = np.mean(test_err_bank, axis=1)
color = ['r', 'b', 'k']
label = ['Entropy', 'Majority Error', 'Gini index']

depth = np.linspace(1,16,16)
fig, ax = plt.subplots(figsize = (10,7))
for method in range(len(methods)):
    plt.plot(depth, train_err_bank[method,:], color=color[method],
             label = label[method]+' (training)', linewidth = 3)
    plt.plot(depth, test_err_bank[method,:], linestyle = '--', color=color[method],
             label = label[method]+' (testing)', linewidth = 3)
plt.legend(fontsize = 16, loc = (1.025,0.63))
plt.xlabel('Tree Depth', fontsize = 18)
plt.ylabel('Accuracy', fontsize = 18)
ax.tick_params(labelsize = 16, size = 10, width = 2)
# plt.ylim([0.87,0.885])
plt.xlim([0,16])
for spine in ax.spines:
    ax.spines[spine].set_linewidth(2)
# plt.savefig('accuracyBANK.png', dpi = 150, bbox_inches = 'tight')
print('Training errors:\nEntropy={}\nMajority Error={}\nGini Index={}'.format(
    avg_train[0], avg_train[1], avg_train[2]))
print('\nTesting errors:\nEntropy={}\nMajority Error={}\nGini Index={}'.format(
    avg_test[0], avg_test[1], avg_test[2]))

# %% plotting results for replaced unknowns
avg_train_no_unk = np.mean(train_err_bank2, axis=1)
avg_test_no_unk = np.mean(test_err_bank2, axis=1)  

fig2, ax2 = plt.subplots(figsize = (10,7))
for method in range(len(methods)):
    plt.plot(depth, train_err_bank2[method,:], color=color[method],
             label = label[method]+' (training)', linewidth = 3)
    plt.plot(depth, test_err_bank2[method,:], linestyle = '--', color=color[method],
             label = label[method]+' (testing)', linewidth = 3)
plt.legend(fontsize = 16, loc = (1.025,0.63))
plt.xlabel('Tree Depth', fontsize = 18)
plt.ylabel('Accuracy', fontsize = 18)
ax2.tick_params(labelsize = 16, size = 10, width = 2)
# plt.ylim([0.87,0.885])
plt.xlim([0,16])
for spine in ax.spines:
    ax2.spines[spine].set_linewidth(2)
# plt.savefig('accuracyBANK2.png', dpi = 150, bbox_inches = 'tight')
print('\nTraining errors:\nEntropy={}\nMajority Error={}\nGini Index={}'.format(
    avg_train_no_unk[0], avg_train_no_unk[1], avg_train_no_unk[2]))
print('\nTesting errors:\nEntropy={}\nMajority Error={}\nGini Index={}'.format(
    avg_test_no_unk[0], avg_test_no_unk[1], avg_test_no_unk[2]))