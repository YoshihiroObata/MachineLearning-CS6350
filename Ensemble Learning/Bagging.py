# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 20:57:48 2021

@author: Yoshihiro Obata
"""
import numpy as np
from ID3 import decisionTree, run_ID3, applyTree, apply_ID3

class Bagging:
    """
    
    """
    def __init__(self, m, T, data, numerical=False, key=None, 
                 randForest=False, Gsize=6, verbose=True, 
                 global_override=False, globaldf = None):
        self.mp = m
        self.T = T
        self.data = data
        self.errs = np.zeros((T,))
        self.alpha = np.zeros((T,))
        self.treesInit = []
        self.numerical = numerical
        self.key = key
        if m < 0.5*len(data):
            self.small_sub = True
            self.globaldf = data
        else:
            self.small_sub = False
        self.randForest = randForest
        self.Gsize = Gsize
        self.verbose = verbose
        if global_override:
            self.small_sub = True
            self.globaldf = globaldf
        
    def draw_with_replacement(self):
        idx = np.random.choice(np.arange(len(self.data)), self.mp)
        return self.data.iloc[list(idx)]
    
    def _bag_progress(self, t):
        percent = np.round(100*t/self.T)
        if self.verbose:
            if len(self.treesInit) != self.T:        
                print(f'{percent}% done. {t} trees created...')
            else:
                print(f'{percent}% done. {t} trees applied...')
    
    def _calc_vote(self, tree_init, t, numerical=False):
        err_init = applyTree(self.data, tree_init, 
                             numerical=numerical)
        h_t, total_err = apply_ID3(err_init)
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
                                         randForest=self.randForest,
                                         Gsize=self.Gsize)
            self.treesInit.append(tree_init)
            run_ID3(tree_init)
            self._calc_vote(tree_init, t, numerical=self.numerical)
        if self.verbose:
            print('100% done.\n')

    def _map2posneg(self, h, key):
        h_mapped = [key[i] for i in h]
        return np.array(h_mapped)    

    def _apply_bagging_loop(self, data):
        predicts = []
        for t in range(self.T):
            if (t)%np.round(self.T/10) == 0:
                self._bag_progress(t)
            applyInit = applyTree(data, self.treesInit[t],
                                  numerical=self.numerical)
            apply_ID3(applyInit)
            predicts.append(applyInit.predict)
        if self.verbose:
            print('100% done.\n')     
        return predicts  
   
def run_bagging(self):
    self._bagging_loop()
    
    
def apply_bagging(self, data):
    h_t = np.array(self._apply_bagging_loop(data))
    h_t = (np.vectorize(self.key.get)(h_t)).T
    alpha = np.array(self.alpha)
    alpha_h = alpha*h_t
    err = np.zeros((self.T,))
    true_lab = data.iloc[:,-1]
    true_lab = np.array([self.key[true_lab[i]] for i in range(len(data))])
    for t in range(self.T):
        H = np.sum(alpha_h[:,:t+1], axis=1) > 0
        H = H*2 - 1
        err[t] = sum(H != true_lab)/len(true_lab)
    return err     