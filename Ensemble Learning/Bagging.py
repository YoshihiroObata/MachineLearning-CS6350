# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 20:57:48 2021

@author: Yoshihiro Obata
"""
import numpy as np
from ID3 import decisionTree, run_ID3, applyTree, apply_ID3
import random

class Bagging:
    """
    
    """
    def __init__(self, m, T, data, numerical=False, key=None, 
                 randForest=False, Gsize=6):
        self.mp = m
        self.T = T
        self.data = data
        self.errs = np.zeros((T,))
        self.alpha = np.zeros((T,))
        self.treesInit = []
        self.trees = []
        self.numerical = numerical
        self.key = key
        if m < 0.5*len(data):
            self.small_sub = True
            self.globaldf = data.copy()
        else:
            self.small_sub = False
        self.randForest = randForest
        self.Gsize = Gsize
        
    def draw_with_replacement(self):
        idx = np.random.choice(np.arange(len(self.data)), self.mp)
        return self.data.iloc[list(idx)]
    
    def _bag_progress(self, t):
        percent = np.round(100*t/self.T)
        if len(self.trees) != self.T:        
            print(f'{percent}% done. {t} trees created...')
        else:
            print(f'{percent}% done. {t} trees applied...')
    
    def _calc_vote(self, tree, tree_init, t, numerical=False):
        err_init = applyTree(tree, self.data, tree_init, 
                             numerical=numerical)
        h_t, total_acc = apply_ID3(err_init)
        total_err = 1 - total_acc
        self.errs[t] = total_err
        self.alpha[t] = 0.5*np.log((1 - total_err)/total_err)
        
    def _bagging_loop(self):
        for t in range(self.T):
            if (t)%np.round(self.T/10) == 0:
                self._bag_progress(t)
            bootstrap = self.draw_with_replacement()
            if self.small_sub:
                tree_init = decisionTree(bootstrap, numerical=self.numerical,
                                         small_sub=self.small_sub,
                                         globaldf=self.globaldf,
                                         randForest=self.randForest,
                                         Gsize=self.Gsize)
            else:
                tree_init = decisionTree(bootstrap, numerical=self.numerical,
                                         Gsize=self.Gsize)
            self.treesInit.append(tree_init)
            tree = run_ID3(tree_init)
            self.trees.append(tree)
            
            self._calc_vote(tree, tree_init, t, numerical=self.numerical)
        print('100% done.\n')
        return self.trees, self.alpha

    def _map2posneg(self, h, key):
        # print(h)
        h_mapped = [key[i] for i in h]
        return np.array(h_mapped)    

    def _apply_bagging_loop(self, data):
        H_final_raw = 0
        for t in range(self.T):
            if (t)%np.round(self.T/10) == 0:
                self._bag_progress(t)
            applyInit = applyTree(self.trees[t], data, self.treesInit[t],
                                  numerical=self.numerical)
            errs, _ = apply_ID3(applyInit)
            h = self._map2posneg(applyInit.predict, key=self.key)
            H_final_raw += self.alpha[t]*h
        print('100% done.\n')    
        
        return H_final_raw
    
def run_bagging(self):
    trees, votes = self._bagging_loop()
    # return trees, votes

def apply_bagging(self, data):
    H_final_raw = self._apply_bagging_loop(data)
    
    H_final = (H_final_raw > 0)*2-1
    true_lab = data.iloc[:,-1]
    true_lab = np.array([self.key[true_lab[i]] for i in range(len(data))])
    err = 1 - sum(H_final == true_lab)/len(data)
    return err        