# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 20:46:21 2021

@author: Yoshihiro Obata
"""

# %% importing packages
import pandas as pd
from logistic_regr import log_regr
import matplotlib.pyplot as plt
import numpy as np

# %% importing the data
cols = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
train = pd.read_csv('bank-note/bank-note/train.csv', names=cols)
test = pd.read_csv('bank-note/bank-note/test.csv', names=cols)

# %% testing implementation
T = 50
lr0 = 0.1
d0 = 0.05
var = 1
lregr = log_regr(train, T=T, lr=lr0, d=d0, var=var)
ws, Ls, _ = lregr.train_log_regr()
err = lregr.predict_log_regr(test)
Ts = np.arange(T)
plt.plot(Ts, Ls)
desc = f'lr: {lr0}, d: {d0}, err: {err}'
plt.annotate(desc, (1,1), fontsize=20)

# %% tuning the hyperparams for MAP
lr0s = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
d0s = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
for lr0 in lr0s:
    for d0 in d0s:
        print(f'starting tuning with lr: {lr0}, d: {d0}')
        lregr = log_regr(train, T=T, lr=lr0, d=d0, var=var)
        ws, Ls, _ = lregr.train_log_regr()
        err = lregr.predict_log_regr(test)
        # Ts = np.arange(T)
        # plt.figure()
        # plt.plot(Ts, Ls)
        # desc = f'lr: {lr0}, d: {d0}, err: {err}'
        # plt.annotate(desc, (1,1), fontsize=20)

# from looking at errs and the loss, lr=0.05, d=0.001 was chosen

# %% tuning the hyperparams for MLE
lr0s = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
d0s = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
for lr0 in lr0s:
    for d0 in d0s:
        print(f'starting tuning with lr: {lr0}, d: {d0}')
        lregr = log_regr(train, T=T, lr=lr0, d=d0, var=var, method='MLE')
        ws, Ls, _ = lregr.train_log_regr()
        err = lregr.predict_log_regr(test)
        # Ts = np.arange(T)
        # plt.figure()
        # plt.plot(Ts, Ls)
        # desc = f'lr: {lr0}, d: {d0}, err: {err}'
        # plt.annotate(desc, (0,0), fontsize=20)
# from looking at errs and the loss, lr=0.005 and d=0.05 was chosen

# %% part a
lr0 = 0.05
d0 = 0.001
T = 50
varis = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
MAP_train_errs = []
MAP_test_errs = []
for var in varis:
    print(f'Running logistic regression with var={var}...')
    lregr = log_regr(train, T=T, lr=lr0, d=d0, var=var)
    ws, Ls, errs = lregr.train_log_regr(get_err=True, test=(train, test))
    train_err, test_err = errs
    MAP_train_errs.append(train_err)
    MAP_test_errs.append(test_err)

# %% part b
lr0 = 0.005
d0 = 0.05    
lregr = log_regr(train, T=T, lr=lr0, d=d0, var=var, method='MLE')
ws, Ls, errs = lregr.train_log_regr(get_err=True, test=(train, test))
train_err, test_err = errs
MLE_train_errs = train_err
MLE_test_errs = test_err

# %% plotting the errors
fig, ax = plt.subplots(1,2,figsize=(16,4))
Tvals = np.arange(T)
c1 = ['blue', 'red', 'black', 'cyan', 'green', 'yellow', 'pink', 'purple']
for i in range(4):
    ax[0].plot(Tvals, MAP_train_errs[i], linewidth=2, color=c1[i])
    ax[0].plot(Tvals, MAP_test_errs[i], linewidth=2, linestyle='--', color=c1[i])
for i in range(4,8,1):
    ax[1].plot(Tvals, MAP_train_errs[i], linewidth=2, color=c1[i])
    ax[1].plot(Tvals, MAP_test_errs[i], linewidth=2, linestyle='--', color=c1[i])

for i in range(2):    
    ax[i].tick_params(labelsize=14, width=2, size=7)
    ax[i].set_xlim([0,50])
    ax[i].set_ylim([0,0.5])
    ax[i].set_xlabel('Epoch', fontsize=16)
    ax[i].set_ylabel('Error', fontsize=16)
    for spine in ax[i].spines:
        ax[i].spines[spine].set_linewidth(2)

lstr1 = [r'Training: $\sigma^2$=0.01', 'Test: $\sigma^2$=0.01',
         r'Training: $\sigma^2$=0.1', 'Test: $\sigma^2$=0.1',
         r'Training: $\sigma^2$=0.5', 'Test: $\sigma^2$=0.5',
         r'Training: $\sigma^2$=1', 'Test: $\sigma^2$=1']
lstr2 = [r'Training: $\sigma^2$=3', 'Test: $\sigma^2$=3',
         r'Training: $\sigma^2$=5', 'Test: $\sigma^2$=5',
         r'Training: $\sigma^2$=10', 'Test: $\sigma^2$=10',
         r'Training: $\sigma^2$=100', 'Test: $\sigma^2$=100']
ax[0].legend(lstr1, fontsize=13, loc='upper right')
ax[1].legend(lstr2, fontsize=13, loc='upper right')
# plt.savefig('MAP_errs.png', dpi=150, bbox_inches='tight')
# plt.savefig('MAP_errs_ZOOM.png', dpi=150, bbox_inches='tight')

# %% plotting MLE
Tvals = np.arange(T)
fig, ax = plt.subplots(figsize=(8,4))
plt.plot(Tvals, MLE_train_errs, linewidth=2, label='Training')
plt.plot(Tvals, MLE_test_errs, linewidth=2, label='Test')
ax.tick_params(labelsize=14, width=2, size=7)
ax.set_xlim([0,50])
ax.set_ylim([0,0.1])
ax.set_xlabel('Epoch', fontsize=16)
ax.set_ylabel('Error', fontsize=16)
for spine in ax.spines:
    ax.spines[spine].set_linewidth(2)
plt.legend(fontsize=13)
plt.savefig('MLE_errs.png', dpi=150, bbox_inches='tight')

# %% writting the error values
MAPtrain = [method[-1] for method in MAP_train_errs]
MAPtest = [method[-1] for method in MAP_test_errs]
df = pd.DataFrame({'var':varis,
                   'MAPtrain': MAPtrain,
                   'MAPtest': MAPtest})
df['MLEtrain'] = MLE_train_errs[-1]
df['MLEtest'] = MLE_test_errs[-1]

df.to_csv('table_errs.csv')
