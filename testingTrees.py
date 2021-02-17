# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 14:47:46 2021

@author: Yoshihiro Obata
"""
import numpy as np
from ID3 import decisionTree
from ID3 import run_ID3
from ID3 import applyTree
from ID3 import apply_ID3

class tester:
    def __init__(self, methods, attrNames, datTrain, datTest, depths, num=False):
        self.methods = methods
        self.attrNames = attrNames
        self.dfTrain = datTrain[2]
        self.attrsTrain = datTrain[0]
        self.labelsTrain = datTrain[1]
        self.dfTest = datTest[2]
        self.attrsTest = datTest[0]
        self.labelsTest = datTest[1]
        self.depths = np.linspace(1,depths,depths)
        self.numerical = num
        self.train_err = np.zeros((len(methods), len(self.depths)))
        self.test_err = np.zeros((len(methods), len(self.depths)))

    def _applyAndError(self, dt, attr, labels, num):
        err = 0
        errinit = applyTree(dt, attr, labels, numerical=num)
        _, err = apply_ID3(errinit)
        return err
    
    
    def test(self):
        for i, method in enumerate(self.methods):
            for j, d in enumerate(self.depths):
                # d = int(d)
                # print(method,d)
                treeInit = None
                dt = None
                treeInit = decisionTree(self.attrsTrain, self.attrNames, 
                                        self.labelsTrain,
                                        depth=d, method=method, 
                                        numerical=self.numerical)
                dt = run_ID3(treeInit)

                self.train_err[i,j] = self._applyAndError(dt, self.dfTrain, 
                                                     self.labelsTrain, 
                                                     num=self.numerical)
                self.test_err[i,j] = self._applyAndError(dt, self.dfTest, 
                                                     self.labelsTest, 
                                                     num=self.numerical)
        return self.train_err, self.test_err