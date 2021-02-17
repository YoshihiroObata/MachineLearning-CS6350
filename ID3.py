# -*- coding: utf-8 -*-
"""
Contains a function to run an ID3 algorithm with options for entropy, majority
error, and gini index information gain. Contains a node and decision tree class
 

@author: Yoshihiro Obata
"""
# Importing Packages
import numpy as np
import pandas as pd
import random
 
# Classes
class Node:
    """
    Node information and children
    """
    
    def __init__(self):
        self.attribute = None
        self.next = None
        self.children = None
        self.valName = None
        self.depth = 0
        self.leaf = False
        
class decisionTree:
    """
    Decision tree information
    """
    def __init__(self, attributes, attrNames, labels, method = 'entropy', 
                 depth = None, randTieBreak = True, numerical = False):
        self.attributes = attributes # np array with columns as attributes
        self.attrClone = None
        self.attrNames = attrNames # attribute names
        self.labels = labels # np array of labels (target)
        self.labelSet = list(set(labels)) # list of unique labels
        # list of number of a certain label
        self.labelCount = [list(labels).count(i) for i in self.labelSet]
        self.node = None
        self.numerical = numerical
        if randTieBreak:
            self.randPick = True
        
        if method == 'entropy':
            self.gainMethod = self._getEntropy
        elif method == 'ME':
            self.gainMethod = self._getME
        else:
            self.gainMethod = self._getGini
        
        if depth is None:
            self.depthLimit = len(self.attrNames)
        elif depth is not None:
            self.depthLimit = depth
        if self.depthLimit > len(self.attrNames) or self.depthLimit < 1:
            raise ValueError('Enter a depth between 1 and {}'.format(
                len(self.attrNames)))
            
    def _getEntropy(self, idx):
        """
        Calculates the entropy of a given list of indices of an attribute
        
        Params
        -----
        Inputs:
        
        idx: a list of indices for a given attribute
        
        Outputs:
        
        entropy: the log2 entropy of the attribute
        """
        labels = [self.labels[i] for i in idx] # remaking the labels
        # print(labels)
        # np array of label count
        labelCount = np.array([labels.count(i) for i in self.labelSet])
        # for each unique label, calculate entropy with log 2
        if 0 in labelCount:
            entropy = 0
        else:
            for count in labelCount:
                ps = labelCount/len(idx)
                entropy = -(np.sum([p*np.log2(p) for p in ps]))
            
        return entropy
    
    def _getME(self, idx):
        """
        Calculates the majority error of a given list of indices of an 
        attribute
        
        Params
        -----
        Inputs:
        
        idx: a list of indices for a given attribute
        
        Outputs:
        
        ME: the majority error of the attribute
        """
        labels = [self.labels[i] for i in idx] # remaking the labels
        # print(labels)
        # np array of label count
        labelCount = np.array([labels.count(i) for i in self.labelSet])
        # for each unique label, calculate majority error
        majority = np.max(labelCount)
        ME = 1 - majority/len(idx)
            
        return ME
    
    def _getGini(self, idx):
        """
        Calculates the Gini index of a given list of indices of an attribute
        
        Params
        -----
        Inputs:
        
        idx: a list of indices for a given attribute
        
        Outputs:
        
        gini: the Gini index of the attribute
        """
        labels = [self.labels[i] for i in idx] # remaking the labels
        # print(labels)
        # np array of label count
        labelCount = np.array([labels.count(i) for i in self.labelSet])
        # for each unique label, calculate entropy with log 2
        for count in labelCount:
            ps = labelCount/len(idx)
            gini = 1 - (np.sum([p**2 for p in ps]))
            
        return gini
    
    def _getInfoGain(self, idx, attrID):
        """
        Calculates the information gain of a single attribute
        -----
        Inputs:
            
        idx: all the indices
        
        attrID: the name of the attribute to calculate the attribute info gain
        
        Outputs:
            
        infoGain: The information gain of using attrID
        """
        infoGain = self.gainMethod(idx) # total entropy
        # print(idx, infoGain)
        # list of indices of an attribute
        attrVals = list(self.attributes[idx,attrID])
        # if isinstance(attrVals[0], (float, int)):
        #     attrVals = attrVals > np.median(attrVals)
        attrSet = list(set(attrVals)) # list of unique vals for attrVals
        # print(attrVals, attrSet)
        # np array of count of vals in each attribute
        attrSetCount = np.array([attrVals.count(i) for i in attrSet])
        # print(attrSetCount)
        infoGainAttr = 0
        for value in range(len(attrSet)):
            attridx = [idx[i] for i in range(len(idx)) if attrVals[i] == attrSet[value]]
            # print(attridx)
            gainSubset = self.gainMethod(attridx)
            infoGainAttr += attrSetCount[value]/len(attrVals)*gainSubset
        infoGain -= infoGainAttr
        # print(infoGain)
        return infoGain
    
    def _getNextAttr(self, idx, attrNames):
        """
        
        """
        attrIDs = [i for i in range(len(self.attrNames)) if self.attrNames[i] in attrNames]
        # print(attrNames, self.attrNames, attrIDs)
        attrInfoGain = [self._getInfoGain(idx, attrID) for attrID in attrIDs]
        # print(attrNames)
        # print(attrInfoGain)
        if self.randPick:
            maxGain = np.array(attrInfoGain) == max(attrInfoGain)
            tieidx = [i for i in range(len(maxGain)) if maxGain[i] == True]
            randidx = random.choice(tieidx)
            # print(randidx)
            
            bestAttr = attrNames[attrInfoGain.index(attrInfoGain[randidx])]
            bestAttridx = attrIDs[attrInfoGain.index(attrInfoGain[randidx])]
        else:
            bestAttr = attrNames[attrInfoGain.index(max(attrInfoGain))]
            bestAttridx = attrIDs[attrInfoGain.index(max(attrInfoGain))]
        
        return bestAttr, bestAttridx
    
    def _num2Bool(self, idx, attrNames):
        attrIDs = [i for i in range(len(self.attrNames)) if self.attrNames[i] in attrNames]
        attrID_type = [self.attributes[0,i] for i in attrIDs]
        numAttrID = [ID for i, ID in enumerate(attrIDs) if isinstance(attrID_type[i], (float,int))]
        for attr in numAttrID:
            median = np.median(self.attributes[idx, attr])
            # print('median {}: {}'.format(self.attrNames[attr], median))
            self.attrClone[:,attr] = self.attributes[:,attr] > median
    
    def _ID3Rec(self, idx, attrNames, node):
        if not node:
            node = Node()
        labelsAttr = [self.labels[i] for i in idx]
        if len(set(labelsAttr)) == 1:
            node.attribute = self.labels[idx[0]]
            node.leaf = True
            return node
        if len(attrNames) == 0:
            node.leaf = True
            return node
        if node.depth == self.depthLimit:
            node.attribute = max(set(labelsAttr), key=labelsAttr.count)
            node.leaf = True
            return node
        
        self.attrClone = self.attributes.copy()
        if self.numerical:
            self._num2Bool(idx, attrNames)
        
        bestAttr, bestAttridx = self._getNextAttr(idx, attrNames)
        node.attribute = bestAttr
        node.children = []
        if isinstance(self.attrNames[0], str):
            chosenAttrSet = set()
            chosenAttrVals = [i for i in self.attrClone[:, bestAttridx] \
                              if i not in chosenAttrSet and \
                                  (chosenAttrSet.add(i) or True)]
        # elif isinstance(self.attributes[0, bestAttridx], (float,int)):
        #     median = np.median(self.attributes[idx, bestAttridx])
        #     chosenAttrVals = 0
        else:
            chosenAttrVals = list(set(self.attrClone[:, bestAttridx]))
        
        for val in chosenAttrVals:
            child = Node()
            child.attribute = val
            node.children.append(child)
            child.depth = node.depth + 1
            if self.numerical and isinstance(self.attributes[0,bestAttridx], (float,int)):
                child.valName = (np.median(self.attributes[idx, bestAttridx]),
                                 val)
            else:
                child.valName = val
            childidx = [i for i in idx if self.attrClone[i, bestAttridx] == val]
            if not childidx:
                # child.next = max(set(labelsAttr), key=labelsAttr.count)
                child.attribute = max(set(labelsAttr), key=labelsAttr.count)
                # print('this happened')
                child.next = self._ID3Rec(childidx, [], child)
            else:
                if attrNames and bestAttr in attrNames:
                    nextAttrs = attrNames.copy()
                    nextAttrs.pop(attrNames.index(bestAttr))
                child.next = self._ID3Rec(childidx, nextAttrs, child)

        return node
  
class applyTree:
    """
    
    """
    def __init__(self, trainedTree, test, labelsTest, numerical=False):
        self.errs = pd.DataFrame(columns=['Test Labels', 'Result', 'Acc'])
        self.root = trainedTree
        self.startTest = test
        self.startLabels = labelsTest
        self.numerical = numerical
        
    def _update_errs(self, subLabel, currNode):
        test_lab = list(subLabel)
        trainLabel = [currNode.attribute]
        train_lab = trainLabel*len(subLabel)
        acc = [trainLabel[0] == i for i in test_lab]
        labdict = {'Test Labels': test_lab,
                   'Result': train_lab,
                   'Acc': acc}
        # print(labdict)
        self.errs = self.errs.append(pd.DataFrame(labdict))
        
    def _applyRec(self, currNode, subset, subLabel):
        if currNode.leaf:
            self._update_errs(subLabel, currNode)
            return
        split_on = currNode.attribute
        for child in currNode.children:
            nextNode = child
            subVal = child.valName
            if not (isinstance(subVal[0], (int, float)) and self.numerical):
                nextSubset = subset[subset[split_on] == subVal]
                nextLabels = subLabel[subset[split_on] == subVal]
                self._applyRec(nextNode, nextSubset, nextLabels)
            else:
                above_median = subVal[1]
                if above_median:
                    nextSubset = subset[subset[split_on] > subVal[0]]
                    nextLabels = subLabel[subset[split_on] > subVal[0]]
                else:
                    nextSubset = subset[subset[split_on] <= subVal[0]]
                    nextLabels = subLabel[subset[split_on] <= subVal[0]]
                self._applyRec(nextNode, nextSubset, nextLabels)
      
# ID3 function
def run_ID3(self):
    """
    
    """
    idx = list(range(len(self.attributes)))
    attrNames = self.attrNames.copy()
    self.node = self._ID3Rec(idx, attrNames, self.node)
    return self.node

# apply ID3
def apply_ID3(self):
    """
    
    """
    currNode = self.root
    allTest = self.startTest
    allLabels = self.startLabels
    self._applyRec(currNode, allTest, allLabels)    
    errdf = self.errs
    total_err = np.sum(errdf['Acc'])/len(errdf)
    
    return errdf, total_err
