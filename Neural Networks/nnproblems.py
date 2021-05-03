# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 19:58:24 2021

@author: Yoshihiro Obata
"""
# %%
from neuralnet import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% test case
# features = np.array([1,1])
# allw = pd.read_csv('p2weights.csv')
# layer1w = allw[allw.layer == 1].value.values.reshape((3,2))
# layer2w = allw[allw.layer == 2].value.values.reshape((3,2))
# layer3w = allw[allw.layer == 3].value.values
# winit = [layer1w, layer2w, layer3w]

# df = pd.DataFrame(np.array([[1,1,1]]), columns=['x1', 'x2', 'y'])

# nnet = nn(2, df, T=1, winit=winit)
# y = nnet.ff(features) # looks right
# nnet.bp(features, 1) # checks out with hand calcs

# # %% testing a single network
# lr0 = 0.01
# d0 = 0.1
# nnet2 = nn(2, df, T=2, lr0=lr0, d0=d0, winit='Zero')
# nnet_test, traintoy, testtoy = nnet2.train_and_apply(df, df)

# %% Problem 3
# importing data
cols = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
train = pd.read_csv('bank-note/bank-note/train.csv', names=cols)
test = pd.read_csv('bank-note/bank-note/test.csv', names=cols)

# %% tuning the hyperparams
T = 50
w = 25
lr0s = [0.1, 0.05, 0.01, 0.005, 0.001]
d0s = [1, 0.5, 0.1, 0.05, 0.01]
print('Beginning to tune hyperparameters...')
for lr0 in lr0s:
    for d0 in d0s:
        nnet = nn(w, train, T=T, lr0=lr0, d0=d0, winit='Gauss')
        trained_net, train_err, test_err = nnet.train_and_apply(train, test)
        print(f'lr: {lr0}, d: {d0}, tr_err: {train_err[-1]}, te_err: {test_err[-1]}')
        plt.figure()
        Tvals = np.arange(T)
        plt.plot(Tvals, train_err)
        plt.plot(Tvals, test_err)
# many of the plots are very similar, but from what I can see, I like the
# hyperparams of lr = 0.01, d = 0.5
print('Done. Combo to use will be lr = 0.01, d = 0.5\n')

# %% running the fully connected nn for varying widths
T = 50
lr0 = 0.01
d0 = 0.5
widths = [5, 10, 25, 50, 100]
werrs_train = []
werrs_test = []
print('Running neural nets on varying widths...')
for w in widths:
    print(f'Working on w={w}')
    nnet = nn(w, train, T=T, lr0=lr0, d0=d0, winit='Gauss')
    trained_net, train_err, test_err = nnet.train_and_apply(train, test)
    werrs_train.append(train_err)
    werrs_test.append(test_err)
    print('Done.\n')

# %% plotting
fig, ax = plt.subplots(figsize=(8,4))
Tvals = np.arange(T)
c = ['blue', 'red', 'black', 'cyan', 'green']
for i in range(len(werrs_train)):
    plt.plot(Tvals, werrs_train[i], linewidth=3, color=c[i])
    plt.plot(Tvals, werrs_test[i], linewidth=3, linestyle='--', color=c[i])
plt.xlim([0,50])
plt.ylim([0,0.1])
ax.tick_params(labelsize=14, width=2, size=7)
for spine in ax.spines:
    ax.spines[spine].set_linewidth(2)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Error', fontsize=16)
lstr = ['Training: w=5', 'Test: w=5',
        'Training: w=10', 'Test: w=10',
        'Training: w=25', 'Test: w=25',
        'Training: w=50', 'Test: w=50',
        'Training: w=100', 'Test: w=100']
plt.legend(lstr, fontsize=13)
plt.savefig('GaussInit.png', dpi=150, bbox_inches='tight')
# plt.savefig('GaussInitZoom.png', dpi=150, bbox_inches='tight')

# %% running the fully connected nn for varying widths
T = 50
lr0 = 0.01
d0 = 0.5
widths = [5, 10, 25, 50, 100]
werrs_train0 = []
werrs_test0 = []
print('Running neural nets on varying widths with zero initialization...')
for w in widths:
    print(f'Working on w={w}')
    nnet = nn(w, train, T=T, lr0=lr0, d0=d0, winit='Zero')
    trained_net, train_err, test_err = nnet.train_and_apply(train, test)
    werrs_train0.append(train_err)
    werrs_test0.append(test_err)
    print('Done.\n')
    
# %% plotting
fig, ax = plt.subplots(figsize=(8,4))
Tvals = np.arange(T)
c = ['blue', 'red', 'black', 'cyan', 'green']
for i in range(len(werrs_train)):
    plt.plot(Tvals, werrs_train0[i], linewidth=3, color=c[i])
    plt.plot(Tvals, werrs_test0[i], linewidth=3, linestyle='--', color=c[i])
plt.xlim([0,50])
plt.ylim([0,0.6])
ax.tick_params(labelsize=14, width=2, size=7)
for spine in ax.spines:
    ax.spines[spine].set_linewidth(2)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Error', fontsize=16)
lstr = ['Training: w=5', 'Test: w=5',
        'Training: w=10', 'Test: w=10',
        'Training: w=25', 'Test: w=25',
        'Training: w=50', 'Test: w=50',
        'Training: w=100', 'Test: w=100']
plt.legend(lstr, fontsize=13)
plt.savefig('ZerosInit.png', dpi=150, bbox_inches='tight')
# plt.savefig('ZerosInitZoom.png', dpi=150, bbox_inches='tight')

# %% getting final training error df
wtrainG = [method[-1] for method in werrs_train]
wtestG = [method[-1] for method in werrs_test]
wtrain0 = [method[-1] for method in werrs_train0]
wtest0 = [method[-1] for method in werrs_test0]
df = pd.DataFrame({'Width':widths,
                   'Gtrain': wtrainG,
                   'Gtest': wtestG,
                   'Ztrain': wtrain0,
                   'Ztest': wtest0})
df.to_csv('table_errs.csv')