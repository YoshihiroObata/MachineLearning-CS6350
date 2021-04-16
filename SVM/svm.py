# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 12:28:54 2021

@author: Yoshihiro Obata
"""

import numpy as np
from scipy import optimize

class SVM:
    def __init__(self, train, C, args, form='primal'):
        ''' Implementation of SVM with options for primal and dual/dual with 
        kernel form
        
        Params
        -----
        :primal args: (T, lr, d, lr_update) order matters
        :dual args: (kernel, gamma) order matters
        '''
        self.train = np.append(np.ones((len(train),1)), train.values[:,:-1], 
                               axis=1)
        self.labels = train.values[:,-1]*2 - 1
        self.C = C
        self.form = form
        self.N = len(train)
        if form == 'primal':
            self.train = np.append(np.ones((len(train),1)), train.values[:,:-1], 
                               axis=1)
            self.w = np.zeros(self.train.shape[1])
            self.T = args[0]
            self.lr0 = args[1]         
            self.d = args[2]
            lr_update = args[3]
            if lr_update == 1:
                self.lr_update = self._schedule_1
            elif lr_update == 2:
                self.lr_update = self._schedule_2
            self.lr = self.lr_update(0)
        
        elif form == 'dual':
            self.train = train.values[:,:-1]
            self.kernel = args[0]
            self.astar = None
            self.bstar = None
            if self.kernel == 'Gauss':
                self.gamma = args[1]
            elif self.kernel == 'None':
                self.wstar = None
            else:
                raise ValueError('Define a valid kernel ("Gauss" or "None")')
        
    ################# primal form SVM functions #################
    
    def _schedule_1(self, t):
        return self.lr0/(1 + (self.lr0*t)/self.d)
    
    def _schedule_2(self, t):
        return self.lr0/(1 + t)
    
    def _shuffle_data(self):
        idx = np.random.choice(np.arange(len(self.train)), len(self.train),
                               replace=False)
        shuff_data = self.train[idx,:]
        shuff_lab = self.labels[idx]
        return shuff_data, shuff_lab
    
    def train_primal(self):
        for t in range(self.T):
            data, labels = self._shuffle_data()
            for ex in range(self.N):
                w0 = self.w[1:]
                is_err = labels[ex]*self.w.dot(data[ex,:].T)
                if is_err <= 1:
                    w_fold = np.append(0, w0)
                    self.w -= self.lr*w_fold - self.lr*self.C*self.N*labels[ex]*data[ex,:]
                else:
                    w0 = (1 - self.lr)*w0
                    self.w = np.append(self.w[0], w0)
            self.lr = self.lr_update(t)
                    
        return self
    
    def predict_svm(self, test):
        test_data = np.append(np.ones((len(test),1)), test.iloc[:,:-1], 1)
        test_lab = np.array(test.iloc[:, -1])
        test_lab = test_lab*2 - 1

        predict = self.w.dot(test_data.T) >= 0
        predict = predict*2 - 1
    
        incorrect = predict != test_lab
        err = sum(incorrect)/len(incorrect)
        
        return err
    
    ################# dual form SVM functions #################
    
    def Gauss_kernel(self, x, z, gamma):
        xnorm = np.sum(x**2, axis=1)
        znorm = np.sum(z**2, axis=1)
        norm_term = xnorm.reshape(-1,1) + znorm.reshape(1,-1) - 2*x.dot(z.T)
        return np.exp(-norm_term/gamma)
    
    def recover_wb(self):
        y = self.labels
        x = self.train
        w_star = (self.astar*y).dot(x)
        j_idx1 = self.astar > 1e-6
        j_idx2 = self.astar < self.C - 1e-6
        j_idx = np.logical_and(j_idx1, j_idx2)
        if self.kernel == 'Gauss':
            # K = self.Gauss_kernel(x[j_idx], x, self.gamma)
            # b_star = y - (self.astar[j_idx]*y[j_idx]).dot(K)
            K = self.Gauss_kernel(x, x[j_idx], self.gamma)
            b_star = y[j_idx] - (self.astar*y).dot(K)
        else:
            b_star = y[j_idx] - w_star.dot(x[j_idx,:].T)
            self.wstar = w_star
        b_star = b_star.mean()
        self.bstar = b_star
        return w_star, b_star
      
    def solve_dual(self, guess=None):
        C = self.C
        y = self.labels
        x = self.train
        yy = y.reshape(-1,1).dot(y.reshape(-1,1).T)
        if self.kernel == 'Gauss':
            K = self.Gauss_kernel(x, x, self.gamma)
            xxyy = K*yy
        else:
            xx = x.dot(x.T)
            xxyy = xx*yy
        
        def _con3(alphas, y):
            con = np.sum(alphas*y)
            return con
        
        def _dual_fun(alphas, xxyy):  
            alphas = alphas.reshape(-1,1)
            aa = alphas.dot(alphas.T)
            return 0.5*np.sum(aa*xxyy) - np.sum(alphas)
        
        cons = {'type':'eq', 'fun':_con3, 'args': [y]}
        conds = (cons)
        
        if guess is None:
            alpha_guess = C*np.ones((self.N, 1))/2
        else:
            alpha_guess = guess
        bounds = optimize.Bounds(0, C)
        
        alpha_star = optimize.minimize(_dual_fun, x0=alpha_guess, args=(xxyy),
                                        method='SLSQP', 
                                        bounds=bounds,
                                        constraints=conds,
                                        options={'maxiter':10000})
        self.astar = alpha_star.x
        
        return alpha_star
    
    def predict_dual(self, test):
        # test_data = np.append(np.ones((len(test),1)), test.iloc[:,:-1], 1)
        test_data = np.array(test.iloc[:,:-1])
        test_lab = np.array(test.iloc[:,-1])
        test_lab = test_lab*2 - 1
        
        if self.kernel == 'Gauss':
            K = self.Gauss_kernel(self.train, test_data, gamma=self.gamma)
            predict = (self.astar*self.labels).dot(K) + self.bstar >= 0
            # predict = (self.astar*self.labels).dot(K) >= 0
        else:
            predict = self.wstar.dot(test_data.T) + self.bstar >= 0
        
        predict = predict*2 - 1
        incorrect = predict != test_lab
        err = sum(incorrect)/len(incorrect)
        return err