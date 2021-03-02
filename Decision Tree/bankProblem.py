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
import time

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
train_no_unk = replace_unk(train.copy())
test_no_unk = replace_unk(test.copy())

# %% training the ID3 algo for testing
tic = time.perf_counter()
bankTreeInit = decisionTree(train, numerical=True)
bankTree = run_ID3(bankTreeInit)

# % applying the ID3 algo for testing
errinit = applyTree(bankTree, train, bankTreeInit, numerical=True)
errs, total_err = apply_ID3(errinit)
toc = time.perf_counter()
print('Time for bank code is {:0.4f} seconds.'.format(toc-tic))

# %% making trees
tic = time.perf_counter()
methods = ['entropy', 'ME', 'gini']
depths = len(train.columns)-1
dfs = [train, test]

errinit = tester(methods, dfs, depths=depths, 
                 numerical=True)
train_err_bank, test_err_bank = tester.test(errinit)

# % testing for replaced unknown values
dfs2 = [train_no_unk, test_no_unk]
errinit2 = tester(methods, dfs2, depths=depths, 
                  numerical=True)
train_err_bank2, test_err_bank2 = tester.test(errinit2)
toc = time.perf_counter()
print('Time for bank code is {:0.4f} seconds.'.format(toc-tic))
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
plt.ylim([0.82,1])
plt.xlim([0,16])
for spine in ax.spines:
    ax.spines[spine].set_linewidth(2)
plt.savefig('accuracyBANK.png', dpi = 150, bbox_inches = 'tight')
print('Training errors:\nEntropy={}\nMajority Error={}\nGini Index={}'.format(
    avg_train[0].round(3), avg_train[1].round(3), avg_train[2].round(3)))
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
plt.ylim([0.82,1])
plt.xlim([0,16])
for spine in ax.spines:
    ax2.spines[spine].set_linewidth(2)
plt.savefig('accuracyBANK2.png', dpi = 150, bbox_inches = 'tight')
print('\nTraining errors:\nEntropy={}\nMajority Error={}\nGini Index={}'.format(
    avg_train_no_unk[0], avg_train_no_unk[1], avg_train_no_unk[2]))
print('\nTesting errors:\nEntropy={}\nMajority Error={}\nGini Index={}'.format(
    avg_test_no_unk[0].round(3), avg_test_no_unk[1].round(3), avg_test_no_unk[2].round(3)))
