# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 12:29:32 2021

@author: Yoshihiro Obata
"""
# %% importing packages
import pandas as pd
from svm import SVM
import numpy as np
import matplotlib.pyplot as plt
import time

# %% importing data
cols = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
train = pd.read_csv('bank-note/bank-note/train.csv', names=cols)
test = pd.read_csv('bank-note/bank-note/test.csv', names=cols)

# %% testing primal svm for bugs
C = 100/873
T = 100
lr = 0.01
d = 0.1
args = (T, lr, d, 1)
primal_svm = SVM(train, C, args=args)
trained_svm_primal = primal_svm.train_primal()
err = trained_svm_primal.predict_svm(test)
print(err)

# %% tuning the learning rate on test set (d = 0.1, C = 100/873, T = 100)
tune_lr = False
if tune_lr:
    T_tune = 100
    lrs = 0.1/(2**np.arange(1,8))
    trials = 20
    lr_err = np.zeros((trials, len(lrs)))
    
    for trial in range(trials):
        for idx, lr in enumerate(lrs):
            primal_svm = SVM(train, T_tune, lr, C, d)
            trained_svm_primal = primal_svm.train_primal()
            lr_err[trial, idx] = trained_svm_primal.predict_svm(test)

    mean_err = np.mean(lr_err, axis=0)
    min_err = min(mean_err)
    min_lr = lrs[(mean_err == min_err).argmax()]

# %% tuning d variable on the test set (lr = 0.0125, , C = 100/873, T = 100)
tune_d = False
if tune_d:
    lr = min_lr
    ds = min_lr*np.array([0.25, 0.5, 1, 2, 4])
    d_err = np.zeros((trials, len(ds)))
    for trial in range(trials):
        for idx, d in enumerate(ds):
            primal_svm = SVM(train, T_tune, lr, C, d)
            trained_svm_primal = primal_svm.train_primal()
            d_err[trial, idx] = trained_svm_primal.predict_svm(test)
            
    mean_err = np.mean(d_err, axis=0)
    min_err = min(mean_err)
    min_d = ds[(mean_err == min_err).argmax()]
    
# %% getting errs for C: schedule 1
Cs = [100/873, 500/873, 700/873]
# trials = 50
trials = 1
# obtained from tuning
T = 100
lr = 0.0125
d = 0.003125
method = 1
train_errs = np.zeros((trials, len(Cs)))
test_errs = np.zeros((trials, len(Cs)))
if trials == 1:
    ws = []
for col, C in enumerate(Cs):
    for trial in range(trials):
        args = (T, lr, d, method)
        primal_svm = SVM(train, C, args=args)
        trained_svm_primal = primal_svm.train_primal()
        
        err = trained_svm_primal.predict_svm(train)
        train_errs[trial, col] = err
        err = trained_svm_primal.predict_svm(test)
        test_errs[trial, col] = err
    if trials == 1:
        ws.append(trained_svm_primal.w)

# %% getting errs for C: schedule 2
Cs = [100/873, 500/873, 700/873]
# trials = 100
trials = 1
# obtained from tuning
T = 100
lr = 0.0125
d = 0.003125
method=2
train_errs2 = np.zeros((trials, len(Cs)))
test_errs2 = np.zeros((trials, len(Cs)))
if trials == 1:
    ws2 = []
for col, C in enumerate(Cs):
    for trial in range(trials):
        args = (T, lr, d, method)
        primal_svm = SVM(train, C, args=args)
        trained_svm_primal2 = primal_svm.train_primal()
        
        err = trained_svm_primal2.predict_svm(train)
        train_errs2[trial, col] = err
        err = trained_svm_primal2.predict_svm(test)
        test_errs2[trial, col] = err
    if trials == 1:
        ws2.append(trained_svm_primal2.w)

# %% writing weight results
if trials == 1:
    results = f"""Learned weights from primal SVM
    
Schedule 1:
    w_C1 = {ws[0]}
    w_C2 = {ws[1]}
    w_C3 = {ws[2]}
Schedule 2:
    w_C1 = {ws2[0]}
    w_C2 = {ws2[1]}
    w_C3 = {ws2[2]}
"""
    print(results)
    txt = open('primal_weights.txt', 'wt')
    n = txt.write(results)
    txt.close
        
# %% plotting primal errs
if trials != 1:
    all_errs = [train_errs[:,0], test_errs[:,0],
                train_errs[:,1], test_errs[:,1],
                train_errs[:,2], test_errs[:,2],]
    fig, ax = plt.subplots()
    plt.boxplot(all_errs, showmeans=True, positions=[1,1.75,3,3.75,5,5.75],
                labels=['C=0.115: Train','C=0.115: Test','C=0.573: Train',
                        'C=0.573: Test','C=0.802: Train','C=0.802: Test'])
    plt.ylim([0,0.04])
    plt.xticks(fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    plt.ylabel('Error', fontsize=14)
    plt.savefig('primal_errs_s1.png', dpi=300, bbox_inches='tight')
    
    all_errs = [train_errs2[:,0], test_errs2[:,0],
                train_errs2[:,1], test_errs2[:,1],
                train_errs2[:,2], test_errs2[:,2],]
    fig, ax = plt.subplots()
    plt.boxplot(all_errs, showmeans=True, positions=[1,1.75,3,3.75,5,5.75],
                labels=['C=0.115: Train','C=0.115: Test','C=0.573: Train',
                        'C=0.573: Test','C=0.802: Train','C=0.802: Test'])
    plt.ylim([0,0.11])
    plt.xticks(fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    plt.ylabel('Error', fontsize=14)
    if trials == 100:
        plt.savefig('primal_errs_s2.png', dpi=300, bbox_inches='tight')

# %% dual form test
C = 100/873
kernel = 'None'
gamma = None
args = (kernel, gamma)
dual_svm = SVM(train, C, args=args, form='dual')
a_star = dual_svm.solve_dual()
w, b = dual_svm.recover_wb()
err = dual_svm.predict_dual(test)
print(err)

# %% dual form solve
kernel = 'None'
gamma = None
Cs = [100/873, 500/873, 700/873]
dual_ws = []
dual_bs = []
dual_train_errs = []
dual_test_errs = []
for C in Cs:
    args = (kernel, gamma)
    dual_svm = SVM(train, C, args=args, form='dual')
    a_star = dual_svm.solve_dual()
    w, b = dual_svm.recover_wb()
    
    dual_ws.append(w)
    dual_bs.append(b)
    
    err = dual_svm.predict_dual(train)
    dual_train_errs.append(err)
    err = dual_svm.predict_dual(test)
    dual_test_errs.append(err)
    
# %% writing dual svm results
write = True
if write:
    results = f"""Learned weights from dual SVM
    
weights:
    w_C1 = {dual_ws[0]}
    w_C2 = {dual_ws[1]}
    w_C3 = {dual_ws[2]}

biases:
    b_C1 = {dual_bs[0]}
    b_C2 = {dual_bs[1]}
    b_C3 = {dual_bs[2]}

train errors:
    err_C1 = {dual_train_errs[0]}
    err_C2 = {dual_train_errs[1]}
    err_C3 = {dual_train_errs[2]}
    
test errors:
    err_C1 = {dual_test_errs[0]}
    err_C2 = {dual_test_errs[1]}
    err_C3 = {dual_test_errs[2]}
"""
    print(results)
    txt = open('dual_weights.txt', 'wt')
    n = txt.write(results)
    txt.close
    
# %% dual svm guassian kernel solving for alphas to make good initial guess
C = 500/873
gamma = 1
kernel = 'Gauss'
args = (kernel, gamma)
dual_svmG = SVM(train, C, args=args, form='dual')

print('starting timer for dual svm solver')
tic = time.perf_counter()
a_starG = dual_svmG.solve_dual()
toc = time.perf_counter()
print(f'solved dual svm in {toc-tic} seconds')

wG, bG = dual_svmG.recover_wb()
err_kernel1 = dual_svmG.predict_dual(train)
err_kernel2 = dual_svmG.predict_dual(test)
print('train error: ', err_kernel1)
print('test error: ', err_kernel2)
alpha_guess = a_starG.x

# %% 
Cs = [100/873, 500/873, 700/873]
gammas = [0.1, 0.5, 1, 5, 100]
kernel = 'Gauss'
dual_train_errsG = np.zeros((len(Cs), len(gammas)))
dual_test_errsG = np.zeros((len(Cs), len(gammas)))
models = []
i = 1
for row, C in enumerate(Cs):
    for col, gamma in enumerate(gammas):
        print(f'Starting dual SVM with gamma: {gamma} and C: {C} ({i}/15)')
        args = (kernel, gamma)
        dual_svmG = SVM(train, C, args=args, form='dual')
        a_starG = dual_svmG.solve_dual(guess=alpha_guess)     
        wG, bG = dual_svmG.recover_wb()
        models.append(dual_svmG)
        
        err_kern = dual_svmG.predict_dual(train)
        dual_train_errsG[row, col] = err_kern
        err_kern = dual_svmG.predict_dual(test)
        dual_test_errsG[row, col] = err_kern
        i += 1
        print('This case is complete.')

# %% saving results to text
write = True
if write:
    results = f'''Results from varying the parameters for the kernel SVM
Training Errors:
{dual_train_errsG}
    
Test Errors:
{dual_test_errsG}
'''
    print(results)
    txt = open('gauss_errs.txt', 'wt')
    n = txt.write(results)
    txt.close
    
# %%
num_support = []
for i in range(15):
    model = models[i]
    j_idx1 = model.astar > 1e-6
    j_idx2 = model.astar < model.C - 1e-6
    j_idx = np.logical_and(j_idx1, j_idx2)
    num_support.append(np.sum(j_idx))

sames = []
for i in range(5,9):
    j = i + 1
    alphas1 = models[i].astar
    alphas2 = models[j].astar
    same = np.sum(alphas1 == alphas2)
    sames.append(same)
    