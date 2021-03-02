# -*- coding: utf-8 -*-
"""
contains tester class that has testing funcitonality

@author: Yoshihiro Obata
"""
import numpy as np
from ID3 import decisionTree
from ID3 import run_ID3
from ID3 import applyTree
from ID3 import apply_ID3

class tester:
    """Tester class that gives errors for varying depths and methods 
    
    """
    def __init__(self, methods, dfs, depths, 
                 numerical=False, tie=True):
        self.methods = methods
        self.attrNames = np.array(dfs[0].columns[:-1])
        self.dfTrain = dfs[0]
        self.dfTest = dfs[1]
        self.depths = np.linspace(1,depths,depths)
        self.numerical = numerical
        self.train_err = np.zeros((len(methods), len(self.depths)))
        self.test_err = np.zeros((len(methods), len(self.depths)))
        self.tie = tie

    def _applyAndError(self, dt, test, treeInit, numerical=False):
        """applies the tree and gives you total error

        Parameters
        ----------
        :dt: decisionTree object
        :attr: training attributes
        :labels: training labels
        :num: if numerical or not

        Returns
        -------
        :err: total accuracy
        """
        # apply 
        err = 0
        errinit = applyTree(dt, test, treeInit, numerical=numerical)
        _, err = apply_ID3(errinit)
        return err
    
    
    def test(self):
        """Function that tests all depths and method you want. Give it a tester object

        Returns
        -------
        :self.train_err: numpy array of training errs (row: method, col: depth)
        :self.test_err: numpy array of test errs (row: method, col: depth)
        """
        # loop through methods and depths for each method
        for i, method in enumerate(self.methods):
            for j, d in enumerate(self.depths):
                # initialize and make decision tree with specified depth and method
                treeInit = None
                dt = None
                treeInit = decisionTree(self.dfTrain, depth=d, method=method, 
                                        numerical=self.numerical,
                                        randTieBreak = self.tie)
                print('Creating DT with depth limit: {} and method: {}...'.format(d, method))
                dt = run_ID3(treeInit)
                print('Tree complete')
                # get errors by applying the tree to both train and test sets
                print('Applying the tree to train and test...')
                self.train_err[i,j] = self._applyAndError(dt, self.dfTrain,
                                                          treeInit,
                                                          numerical=self.numerical)
                self.test_err[i,j] = self._applyAndError(dt, self.dfTest,
                                                         treeInit,
                                                         numerical=self.numerical)
                print('Applying complete\n')
        print('Done')
        
        return self.train_err, self.test_err
