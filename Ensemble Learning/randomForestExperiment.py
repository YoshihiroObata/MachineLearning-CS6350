# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 21:25:18 2021

@author: Yoshihiro Obata
"""
# %% 
import numpy as np
import pandas as pd
import time
from Bagging import Bagging, run_bagging
from ID3 import applyTree, apply_ID3
import glob

# %% functions
def sample_with_replace(m, train):
    idx = np.random.choice(np.arange(len(train)), m, replace=False)
    return train.iloc[list(idx)]

def get_bagged_tree(m, T_bag, train, key, globaldf, Gsize):
    bagInit = Bagging(m, T_bag, train, numerical=True, key=key, 
                      global_override=True, globaldf=globaldf, verbose=False,
                      randForest=True, Gsize=Gsize)
    run_bagging(bagInit)
    return bagInit

def get_predictions(bagInit, test, key):
    h_bag = np.array(bagInit._apply_bagging_loop(test))
    apply_single_tree = applyTree(test, bagInit.treesInit[0], numerical=True)
    apply_ID3(apply_single_tree)
    h_tree = np.array(apply_single_tree.predict)
    
    h_bag = (np.vectorize(key.get)(h_bag)).T
    alpha = np.array(bagInit.alpha)
    alpha_h = alpha*h_bag
    H = np.sum(alpha_h, axis=1) > 0
    H_bag = H*2 - 1
    
    h_tree = np.vectorize(key.get)(h_tree)
    
    return H_bag, h_tree
    
def logger(h_bag, h_tree, inum):
    if inum == 0:
        numcsv = int(len(glob.glob('*.csv'))/2)
        h_bags = pd.DataFrame(h_bag, columns=[inum])
        h_trees = pd.DataFrame(h_tree, columns=[inum])
        h_bags.to_csv('h_rand_forest'+str(numcsv)+'.csv', index=False)
        h_trees.to_csv('h_rand_trees'+str(numcsv)+'.csv', index=False)
    else:
        numcsv = int(len(glob.glob('*.csv'))/2 - 1)
        bagname = 'h_rand_forest'+str(numcsv)+'.csv'
        treename = 'h_rand_trees'+str(numcsv)+'.csv'
        bag_curr = pd.read_csv(bagname)
        tree_curr = pd.read_csv(treename)
        bag_curr[inum] = h_bag
        tree_curr[inum] = h_tree
        bag_curr.to_csv(bagname, index=False)
        tree_curr.to_csv(treename, index=False)
        
def progress_report(i_time, i, itera):
    print('\nProgress----------------------------------------')
    print(f'Number of random forests complete: {i+1}/{itera}')
    print(f'Total elapsed time: {np.round(sum(i_time)/3600,3)} hrs')
    mean_t = np.round(np.mean(i_time),1)
    print(f'Average time per iteration: {mean_t} sec ({np.round(mean_t/60,2)} mins)')
    time_left = (mean_t*(itera-(i+1)))/3600
    print(f'Estimated time remaining: {np.round(time_left,3)} hrs\n')
    
# %%
cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
        'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
        'previous', 'poutcome', 'y']
train = pd.read_csv('bank/train.csv', names=cols)
test = pd.read_csv('bank/test.csv', names=cols)

m = 1000
T_bag = 50
key = {'no':-1, 'yes':1}
itera = 10
Gsize = 2

# %%
if itera < 500:
    print('Less iterations are used to show functionality but save time')
i_time = []
expStart = """Starting new experiment with the following parameters:
    m = {}
    T = {}
    iterations = {}
    |G| = {}
    
This takes a long time. Beware.
""".format(m, T_bag, itera, Gsize)
print(expStart)

for i in range(itera):
    tic = time.perf_counter()
    print('Training new tree...')
    train_new = sample_with_replace(m, train)
    bag = get_bagged_tree(m, T_bag, train_new, key, globaldf=train, 
                          Gsize=Gsize)
    print('Getting and writing predictions...')
    h_bag, h_tree = get_predictions(bag, test, key)
    logger(h_bag, h_tree, i)
    toc = time.perf_counter()
    i_time.append(toc-tic)
    progress_report(i_time, i, itera)
