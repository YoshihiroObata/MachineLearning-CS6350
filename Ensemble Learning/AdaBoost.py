# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 09:09:17 2021

@author: Yoshihiro Obata
"""
import numpy as np
from ID3 import decisionTree, run_ID3, applyTree, apply_ID3

class AdaBoost:
    """
    
    """
    def __init__(self, train, T, depth=1):
        self.data = train
        self.labels = np.array(train.iloc[:,-1])
        self.D_init = np.ones((len(train),))/len(train)
        self.T = T
        self.depth = depth # decision stump
        self.learners = []
        self.alpha = np.zeros((T,))
        self.errs = np.zeros((T,))
    
    
    def _calc_vote(self, stump, stump_init, t, D, numerical=False):
        err_init = applyTree(stump, self.data, stump_init, weights=D, 
                             numerical=numerical)
        print('initialized tree...')
        h_t, total_acc = apply_ID3(err_init)
        print('applied..')
        total_err = 1 - total_acc
        if total_err > 0.5:
            print(f'Total error was {total_err}, which is greater than 50%')
        self.errs[t] = total_err
        self.alpha[t] = 0.5*np.log((1 - total_err)/total_err)
        
        return h_t
    
    def _map2posneg(self, h_t):
        return h_t*2 - 1
    
    def _update_weights(self, D, t, h_t):
        yh = self._map2posneg(h_t)
        D_tp1 = D*np.exp(-self.alpha[t]*yh)
        Z_t = np.sum(D_tp1)
        D_tp1 /= Z_t
        return D_tp1
    
    def _AdaLoop(self, D):
        
        for t in range(self.T):
            stump_init = decisionTree(self.data, numerical=True, 
                                      depth=self.depth, weights=D)
            stump = run_ID3(stump_init)
            self.learners.append(stump)
            print('Made stump...')
            h_t = self._calc_vote(stump, stump_init, t, D, numerical=True)
            # print(h_t)
            Dtp1 = self._update_weights(D, t, h_t)
            print('Updated weights...')
            D = Dtp1
            
        return self.learners, self.alpha
    
def run_adaBoost(self):
    D_init = self.D_init.copy()
    learners, alphas = self._AdaLoop(D_init)
    return learners, alphas
    