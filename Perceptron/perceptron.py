# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 13:57:04 2021

@author: Yoshihiro Obata
"""
import numpy as np


class Perceptron:
    def __init__(self, T, lr, data, variant='standard'):
        self.T = T
        self.lr = lr
        # self.train = np.array(data.iloc[:, :-1])
        self.train = np.append(np.ones((len(data),1)), data.iloc[:,:-1], 1)
        self.label = np.array(data.iloc[:, -1])*2 - 1
        self.w = np.zeros((self.train.shape[1],))
        if variant == 'voted':
            self.ws = []
            self.cs = []
        elif variant == 'averaged':
            self.a = np.zeros((self.train.shape[1],))
        self.variant = variant

    def _shuffle_data(self):
        idx = np.random.choice(np.arange(len(self.train)), len(self.train),
                               replace=False)
        # print(len(set(idx)))
        shuff_data = self.train[idx,:]
        shuff_lab = self.label[idx]
        return shuff_data, shuff_lab

    def train_percep(self):
        for t in range(self.T):
            data, labels = self._shuffle_data()
            for ex in range(len(data)):
                is_err = labels[ex]*self.w.dot(data[ex,:].T)
                if is_err <= 0:
                    self.w += self.lr*labels[ex]*data[ex,:]
                else:
                    continue
        return self

    def train_voted(self):
        Cm = 1
        for t in range(self.T):
            data, labels = self._shuffle_data()
            for ex in range(len(data)):
                is_err = labels[ex]*self.w.dot(data[ex,:].T)
                if is_err <= 0:
                    # print('happened')
                    self.w += self.lr*labels[ex]*data[ex,:]
                    self.ws.append(self.w.copy())
                    self.cs.append(Cm)
                    Cm = 1
                else:
                    Cm += 1                   
        return self
    
    def train_averaged(self):
        for t in range(self.T):
            data, labels = self._shuffle_data()
            for ex in range(len(data)):
                is_err = labels[ex]*self.w.dot(data[ex,:].T)
                if is_err <= 0:
                    self.w += self.lr*labels[ex]*data[ex,:]
                else:
                    pass
                self.a += self.w
                    
        return self

    def predict_percep(self, test):
        # test_data = np.array(test.iloc[:, :-1])
        test_data = np.append(np.ones((len(test),1)), test.iloc[:,:-1], 1)
        test_lab = np.array(test.iloc[:, -1]) > 0
        test_lab = test_lab*2 - 1
        if self.variant == 'standard':
            predict = self.w.dot(test_data.T) >= 0
            predict = predict*2 - 1
        elif self.variant == 'voted':
            predict = np.zeros_like(test_lab)
            for m in range(len(self.cs)):
                step_predict = self.ws[m].dot(test_data.T) >= 0
                predict += self.cs[m]*(step_predict*2 - 1)
            predict = (predict >= 0)*2 - 1
        else:
            predict = self.a.dot(test_data.T) >= 0
            predict = predict*2 - 1

        incorrect = predict != test_lab
        err = sum(incorrect)/len(incorrect)
        return err