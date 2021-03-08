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

# %%
# lrs = 1/(2**np.arange(10))
lr = 0.014
bgd = LMS(train, 6000, lr)
w_bgd = run_LMS_GD(bgd)

# %%
X_test = np.append(np.ones((len(test),1)), test.iloc[:,:-1], 1)
y_test = test.iloc[:,-1]
J_test_bgd = 0.5*sum((y_test - w_bgd.dot(X_test.T))**2)

# %%
J = bgd.J[bgd.J != 0]
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(np.arange(len(J)), J, 'b', linewidth=2)
ax.tick_params(labelsize = 16, size = 10, width = 2)
for spine in ax.spines:
    ax.spines[spine].set_linewidth(2)
plt.xlim([-100,len(J)])
plt.xlabel('Number of Iterations', fontsize=18)
plt.ylabel('Cost Function, J', fontsize=18)
plt.grid(True)
plt.savefig('P2_bgd.png', dpi=150, bbox_inches='tight')

# %%
# lrs = np.linspace(0.02, 0.01, 10)
t = 100000
lr2 = 0.003
sgd = LMS(train, t, lr2)
w_sgd = run_LMS_SGD(sgd)

# %%
J_test_sgd = 0.5*sum((y_test - w_sgd.dot(X_test.T))**2)

# %%
J_sgd = sgd.J[sgd.J != 0]
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(np.arange(len(J_sgd)), J_sgd, 'b', linewidth=2)
ax.tick_params(labelsize = 16, size = 10, width = 2)
for spine in ax.spines:
    ax.spines[spine].set_linewidth(2)
plt.xlim([-100,len(J_sgd)])
plt.xlabel('Number of Iterations', fontsize=18)
plt.ylabel('Cost Function, J', fontsize=18)
plt.grid(True)
plt.savefig('P2_sgd.png', dpi=150, bbox_inches='tight')
    
# %% optimal weight vector
X = np.append(np.ones((len(train),1)), train.iloc[:,:-1], 1)
Y = np.array(train.iloc[:,-1])
part1 = np.linalg.inv(X.T.dot(X))
part2 = X.T.dot(Y)
w_optimal = part1.dot(part2)

# %% print results
results_reg = f"""------------------------------------------------------------------------------
Results for gradient descent and stochastic gradient descent:
------------------------------------------------------------------------------
Batch Gradient Descent: Cost function plot should be in working directory
    learning rate: {lr}
    (first weight is bias)
    weights: {np.round(w_bgd, 4)}
    test data cost function value: {J_test_bgd}
------------------------------------------------------------------------------
Stochastic Gradient Descent: Cost function plot should be in working directory
    learning rate: {lr2}
    (first weight is bias)
    weights: {np.round(w_sgd, 4)}
    test data cost function value: {J_test_sgd}
------------------------------------------------------------------------------
Optimal weight vector for comparison:
    (first weight is bias)
    optimal w: {np.round(w_optimal,4)}
------------------------------------------------------------------------------
"""
print(results_reg)
txt = open('LinRegResults.txt', 'wt')
n = txt.write(results_reg)
txt.close()
