# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 20:39:31 2021

@author: Yoshihiro Obata
"""
import numpy as np

class log_regr:
    def __init__(self, data, T=25, lr=0.01, d=0.5, var=1, method='MAP'):
        self.train = data.iloc[:,:-1].values
        self.N = len(self.train)
        self.y = (data.iloc[:,-1].values)*2 - 1
        self.T = T
        self.lr = lr
        self.d = d
        self.var = var
        self.w = np.zeros(self.train.shape[1] + 1)
        self.method = method
    
    def train_log_regr(self, get_err=False, test=None):
        Ls = np.zeros(self.T)
        if get_err:
            train_err = np.zeros(self.T)
            test_err = np.zeros(self.T)
        else:
            train_err = None
            test_err = None
        for t in range(self.T):
            train, lab = shuffle(self.train, self.y)
            lr = schedule(self.lr, self.d, t)
            for ex in range(self.N):
                X = np.append(1, train[ex,:])
                y = lab[ex]
                logL = np.log(1 + np.exp(-y*self.w.dot(X.T)))*self.N
                if self.method == 'MAP':
                    reg = (1/self.var)*self.w.dot(self.w.T)
                    L = logL + reg
                    grad_L = -(1 - sigmoid(y*self.w.dot(X.T)))*y*X*self.N + 2*(1/self.var)*self.w
                else:
                    L = logL
                    grad_L = -(1 - sigmoid(y*self.w.dot(X.T)))*y*X*self.N
                Ls[t] = L
                # grad_L = -(y - self.w.dot(X))*X
                self.w -= lr*grad_L
            if get_err:
                train0, test0 = test
                err = self.predict_log_regr(train0)
                train_err[t] = err
                err = self.predict_log_regr(test0)
                test_err[t] = err
                
        return self.w, Ls, (train_err, test_err)
    
    def predict_log_regr(self, test):
        test_data = np.append(np.ones((len(test), 1)), 
                              test.iloc[:,:-1].values, axis=1)
        test_lab = test.iloc[:, -1].values
        test_lab = test_lab*2 - 1

        predict = (self.w.dot(test_data.T) >= 0)*2 - 1
        incorrect = predict != test_lab
        err = sum(incorrect)/len(incorrect)
        
        return err
        

def shuffle(train, labels):
    idx = np.random.choice(np.arange(len(train)), len(train),
                               replace=False)
    shuff_data = train[idx,:]
    shuff_lab = labels[idx]
    return shuff_data, shuff_lab

def schedule(lr0, d0, t):
    return lr0/(1 + (lr0*t)/d0)

def sigmoid(x):
    return 1/(1 + np.exp(-x))