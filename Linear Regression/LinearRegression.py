# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 19:05:19 2021

@author: Yoshihiro Obata
"""

import numpy as np

class LMS:
    """
    
    """
    def __init__(self, data, T, lr, verbose=False):
        self.X = np.append(np.ones((len(data),1)), data.iloc[:,:-1], 1)
        self.y = data.iloc[:,-1]
        self.T = T
        self.lr = lr
        self.w_init = np.zeros((len(self.X.T),))
        self.converged = False
        self.J = np.zeros((T,))
        
    def _check_converge_bgd(self, w, w_next, t):
        norm = np.linalg.norm(w - w_next)
        if norm < 1e-6 and t!=0:
            self.converged = True
        return norm
        
    def bgd_loop(self, w):
        for t in range(self.T):
            self.J[t] = 0.5*sum((self.y - w.dot(self.X.T))**2)
            grad_J = (-(self.y - w.dot(self.X.T))).dot(self.X)
            w_next = w - self.lr*grad_J
            norm = self._check_converge_bgd(w, w_next, t)
            if self.converged:
                print(f'Done with lr={self.lr}, t={t}. Converged: {self.converged}...\n')
                return w
            w = w_next
            # self.ws.append(w)
        print(f'Done with lr={self.lr}, t={t}. Converged: {self.converged}...')
        print(f'norm weight difference vector: {norm}...\n')
        return w
    
    def sgd_loop(self, w):
        for t in range(self.T):
            self.J[t] = 0.5*sum((self.y - w.dot(self.X.T))**2)
            # if (self.J[t] - self.J[t-1]) < 1e-6:
                # self.converged = True
            i = np.random.choice(np.arange(len(self.y)))
            w_next = w + self.lr*(self.y[i] - sum(w*self.X[i,:]))*self.X[i,:]
            if self.converged:
                print(f'Done with lr={self.lr}, t={t}. Converged: {self.converged}...\n')
                return w
            w = w_next
        print(f'Done with lr={self.lr}, t={t}. Converged: {self.converged}...')
        # print(f'norm weight difference vector: {norm}...\n')
        return w
    
def run_LMS_GD(self):
    w_init = self.w_init
    w_final = self.bgd_loop(w_init)
    return w_final

def run_LMS_SGD(self):
    w_init = self.w_init
    w_final = self.sgd_loop(w_init)
    return w_final
        
        