# -*- coding: utf-8 -*-
"""
HW 1: Problems 1-3

Decision Trees: checking hand calcs and running a few examples. Uses the ID3
file.

Yoshihiro Obata
"""

# %% Importing Packages
import numpy as np
from ID3 import decisionTree
from ID3 import run_ID3
    
# %% Training Data

# making the training data with columns as x1, x2, x3, x4
attributes = np.array([[0, 0, 0, 1, 0, 1, 0],
                       [0, 1, 0, 0, 1, 1, 1],
                       [1, 0, 1, 0, 1, 0, 0],
                       [0, 0, 1, 1, 0, 0, 1]]).T
# Boolean labels
y = np.array([0, 0, 1, 1, 0, 0, 0]).T
attrNames = [0,1,2,3]

# %% run ID3 on 1a
tree0 = decisionTree(attributes, attrNames, y) 
tree1 = run_ID3(tree0)

# %% tennis data
attributes2 = np.array([['S', 'S', 'O', 'R', 'R', 'R', 'O', 
                         'S', 'S', 'R', 'S', 'O', 'O', 'R'],
                        ['H', 'H', 'H', 'M', 'C', 'C', 'C', 
                         'M', 'C', 'M', 'M', 'M', 'H', 'M'],
                        ['H', 'H', 'H', 'H', 'N', 'N', 'N', 
                         'H', 'N', 'N', 'N', 'H', 'N', 'H'],
                        ['W', 'S', 'W', 'W', 'W', 'S', 'S', 
                         'W', 'W', 'W', 'S', 'S', 'W', 'S']]).T
y2 = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]).T
attrNames2 = ['Outlook', 'Temp', 'Humidity', 'Wind']

# %% run ID3 on 2
tree2 = decisionTree(attributes2, attrNames2, y2, method='ME')
tree3 = run_ID3(tree2)

# %% ID3 on 3a
attributes3 = np.array([['S', 'S', 'O', 'R', 'R', 'R', 'O', 
                         'S', 'S', 'R', 'S', 'O', 'O', 'R', 'S'],
                        ['H', 'H', 'H', 'M', 'C', 'C', 'C', 
                         'M', 'C', 'M', 'M', 'M', 'H', 'M', 'M'],
                        ['H', 'H', 'H', 'H', 'N', 'N', 'N', 
                         'H', 'N', 'N', 'N', 'H', 'N', 'H', 'N'],
                        ['W', 'S', 'W', 'W', 'W', 'S', 'S', 
                         'W', 'W', 'W', 'S', 'S', 'W', 'S', 'W']]).T
y3 = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1]).T
attrNames3 = ['Outlook', 'Temp', 'Humidity', 'Wind']

tree4 = decisionTree(attributes3, attrNames3, y3)
tree5 = run_ID3(tree4)

# %% ID3 on 3b
attributes4 = np.array([['S', 'S', 'O', 'R', 'R', 'R', 'O', 
                         'S', 'S', 'R', 'S', 'O', 'O', 'R', 'O'],
                        ['H', 'H', 'H', 'M', 'C', 'C', 'C', 
                         'M', 'C', 'M', 'M', 'M', 'H', 'M', 'M'],
                        ['H', 'H', 'H', 'H', 'N', 'N', 'N', 
                         'H', 'N', 'N', 'N', 'H', 'N', 'H', 'N'],
                        ['W', 'S', 'W', 'W', 'W', 'S', 'S', 
                         'W', 'W', 'W', 'S', 'S', 'W', 'S', 'W']]).T

tree6 = decisionTree(attributes4, attrNames3, y3, depth=1)
tree7 = run_ID3(tree6)

# %% testing label str
y4 = np.array(['n', 'n', 'y', 'y', 'y', 'n', 'y', 'n', 
               'y', 'y', 'y', 'y', 'y', 'n', 'y']).T
tree8 = decisionTree(attributes4, attrNames3, y4)
tree9 = run_ID3(tree8)