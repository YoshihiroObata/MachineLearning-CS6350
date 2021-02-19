# -*- coding: utf-8 -*-
"""
Contains a function to run an ID3 algorithm with options for entropy, majority
error, and gini index information gain. Contains a node and decision tree
class as well as an applying tree class and applying function

@author: Yoshihiro Obata
"""
# Importing Packages
import numpy as np
import pandas as pd
import random
 
# Classes
class Node:
    """Node information and children
    
    Contains all information that I used for each node of the tree. Most
    important ones are attribute, children, valName, depth, and leaf. Child
    is mainly used to define the next node
    """
    
    def __init__(self):
        self.attribute = None # hold the attribute to split on or the leaf val
        self.next = None # next child
        self.children = None # list of children
        self.valName = None # what is the value of the split attribute
        self.depth = 0 # depth of the tree (0 at root)
        self.leaf = False # indicates leaf node
        
class decisionTree:
    """ Decision tree information
    
    Contains all the main calculations needed to perform the ID3 algorithm to
    create a decision tree. Information gain and depth can be changed. Must
    indicate if numerical data is present, otherwise, assumes catergorical.
    The algorithm is set to automatically break ties in information gain with
    a random choice. If randTieBreak set to False, it will always pick the 
    first instance of max info gain.  
    """
    def __init__(self, attributes, attrNames, labels, method = 'entropy', 
                 depth = None, randTieBreak = True, numerical = False):
        self.attributes = attributes # np array with columns as attributes
        self.attrClone = None # clone to handle numerical data without altering orignal attributes
        self.attrNames = attrNames # attribute names
        self.labels = labels # np array of labels (target)
        self.labelSet = list(set(labels)) # list of unique labels
        self.labelCount = [list(labels).count(i) for i in self.labelSet] # list of number of a certain label
        self.node = None # node
        self.numerical = numerical # bool of if numerical data
        if randTieBreak:
            self.randPick = True # pick random info gain attrName if tied 
        else:
            self.randPick = False
        
        # setting the gain method
        if method == 'entropy':
            self.gainMethod = self._getEntropy
        elif method == 'ME':
            self.gainMethod = self._getME
        else:
            self.gainMethod = self._getGini
        
        # setting the depth limit (0 < d <= # of attributes)
        if depth is None:
            self.depthLimit = len(self.attrNames)
        elif depth is not None:
            self.depthLimit = depth
        if self.depthLimit > len(self.attrNames) or self.depthLimit < 1:
            raise ValueError('Enter a depth between 1 and {}'.format(
                len(self.attrNames)))
            
    def _getEntropy(self, idx):
        """Calculates the entropy of a given list of indices of an attribute
         
        Params
        -----
        Inputs:
        
        :idx: a list of indices for a given attribute
        
        Outputs:
        
        :entropy: the log2 entropy of the attribute
        """
        labels = [self.labels[i] for i in idx] # remaking the labels
        # np array of label count
        labelCount = np.array([labels.count(i) for i in self.labelSet])
        # labelCount = np.array([labels.count])
        # for each unique label, calculate entropy with log 2
        if len(labelCount[labelCount != 0]) == 1:
            entropy = 0
        else:
            for count in labelCount:
                # ps = labelCount/len(idx)
                ps = labelCount/sum(labelCount)
                entropy = -(np.sum([p*np.log2(p) for p in ps if p != 0]))

        return entropy
    
    def _getME(self, idx):
        """Calculates the majority error of a given list of indices of an attribute
        
        Params
        -----
        Inputs:
        
        :idx: a list of indices for a given attribute
        
        Outputs:
        
        :ME: the majority error of the attribute
        """
        labels = [self.labels[i] for i in idx] # remaking the labels
        # np array of label count
        labelCount = np.array([labels.count(i) for i in self.labelSet])
        # for each unique label, calculate majority error
        majority = max(labelCount)
        ME = 1 - majority/sum(labelCount)
            
        return ME
    
    def _getGini(self, idx):
        """Calculates the Gini index of a given list of indices of an attribute
               
        Params
        -----
        Inputs:
        
        :idx: a list of indices for a given attribute
        
        Outputs:
        
        :gini: the Gini index of the attribute
        """
        labels = [self.labels[i] for i in idx] # remaking the labels
        # np array of label count
        labelCount = np.array([labels.count(i) for i in self.labelSet])
        # for each unique label, calculate gini
        for count in labelCount:
            ps = labelCount/sum(labelCount)
            gini = 1 - (np.sum([p**2 for p in ps]))
            
        return gini
    
    def _getInfoGain(self, idx, attrID):
        """Calculates the information gain of a single attribute        
        
        Params
        -----
        Inputs:
            
        :idx: all the indices      
        :attrID: the name of the attribute to calculate the attribute info gain
        
        Outputs:
            
        :infoGain: The information gain of using attrID
        """
        infoGain = self.gainMethod(idx) # total entropy
        # list of indices of an attribute
        attrVals = list(self.attributes[idx,attrID])
        attrSet = list(set(attrVals)) # list of unique vals for attrVals
        # np array of count of vals in each attribute
        attrSetCount = np.array([attrVals.count(i) for i in attrSet])
        infoGainAttr = 0
        # uses info gain method to calc info gain for the attr
        for value in range(len(attrSet)):
            attridx = [idx[i] for i in range(len(idx)) if attrVals[i] == attrSet[value]]
            gainSubset = self.gainMethod(attridx)
            infoGainAttr += attrSetCount[value]/len(attrVals)*gainSubset
        infoGain -= infoGainAttr
        return infoGain
    
    def _getNextAttr(self, idx, attrNames):
        """
        Calculates the information gain of a single attribute
        
        Params
        -----
        Inputs:
            
        :idx: all the indices
        :attrNames: the name of the attributes to calculate the info gain
        
        Outputs:
            
        :bestAttr: name of the next attribute    
        :bestAttridx: index (col #) of the best attribute in self.attributes
        """
        # get the attr indices and calculate the info gain of them
        attrIDs = [i for i in range(len(self.attrNames)) if self.attrNames[i] in attrNames]
        attrInfoGain = [self._getInfoGain(idx, attrID) for attrID in attrIDs]
        # print('info gains: ', attrInfoGain)
        # if you want to pick randomly, get the tied indices and do a random choice
        if self.randPick:
            maxGain = np.array(attrInfoGain) == max(attrInfoGain)
            tieidx = [i for i in range(len(maxGain)) if maxGain[i] == True]
            randidx = random.choice(tieidx)
            
            bestAttr = attrNames[attrInfoGain.index(attrInfoGain[randidx])]
            bestAttridx = attrIDs[attrInfoGain.index(attrInfoGain[randidx])]
        # if not random, max picks the first thing
        else:
            bestAttr = attrNames[attrInfoGain.index(max(attrInfoGain))]
            bestAttridx = attrIDs[attrInfoGain.index(max(attrInfoGain))]
        
        return bestAttr, bestAttridx
    
    def _num2Bool(self, idx, attrNames):
        """ Changes the self.attrClone variable to a boolean of > and <= median

        Parameters
        ----------
        :idx: indices of all attributes
        :attrNames: names of attributes
            
        Returns
        -------
        None.
        """
        # get the attr indices, get the first value of these attributes
        attrIDs = [i for i in range(len(self.attrNames)) if self.attrNames[i] in attrNames]
        attrID_type = [self.attributes[0,i] for i in attrIDs]
        # get the attr index if the first val is a float or int
        numAttrID = [ID for i, ID in enumerate(attrIDs) if isinstance(attrID_type[i], (float,int))]
        # find median, split if > (True) or <= (False) median, update self.attrClone
        for attr in numAttrID:
            median = np.median(self.attributes[idx, attr])
            self.attrClone[:,attr] = self.attributes[:,attr] > median
    
    def _ID3Rec(self, idx, attrNames, node, prevMax=None):
        """function for the recursive part of ID3

        Parameters
        ----------
        :idx: all indices of the subset (int of rows to be used)
        :attrNames: names of attributes of subset (str)
        :node: node class object (current root to consider)

        Returns
        -------
        node : node object of root for the root node called in ID3Rec
        """
        # if not Node, make a node
        if not node: 
            node = Node()
        labelsAttr = [self.labels[i] for i in idx] # get labels for all idx
        # stopping conditions:
        # if there's only 1 label, return node with attribute label that is leaf
        if len(set(labelsAttr)) == 1:
            node.attribute = self.labels[idx[0]]
            node.leaf = True
            return node
        # if there's no attrNames, return the node as leaf node
        if len(idx) == 0:
            # print('this happened')
            node.attribute = prevMax
            node.leaf = True
            return node
        # if you hit depth limit, set the node attr to the max in the subset
        if node.depth == self.depthLimit:
            node.attribute = max(set(labelsAttr), key=labelsAttr.count)
            node.leaf = True
            return node
        
        # set attrClone to attrs and if numerical, convert to bool
        self.attrClone = self.attributes.copy()
        if self.numerical:
            self._num2Bool(idx, attrNames)
        
        # get best attr, set this node attr to best one, init child list
        # print(idx, labelsAttr, attrNames)
        bestAttr, bestAttridx = self._getNextAttr(idx, attrNames)
        node.attribute = bestAttr
        node.children = []
        # if its categorical, get vals in order, if not, just get them
        # NOTE: getting vals from attrClone which contains all data, not just subset
        if isinstance(self.attrNames[0], str):
            chosenAttrSet = set()
            chosenAttrVals = [i for i in self.attrClone[:, bestAttridx] \
                              if i not in chosenAttrSet and \
                                  (chosenAttrSet.add(i) or True)]
        else:
            chosenAttrVals = list(set(self.attrClone[:, bestAttridx]))
        
        for val in chosenAttrVals: # for all vals the attr can take:
            # make new child node, append it to child list, update depth
            child = Node()
            # node.children.append(child)
            child.depth = node.depth + 1
            # if the data is numerical, set child val to the median value, else, make it val
            if self.numerical and isinstance(self.attributes[0,bestAttridx], (float,int)):
                child.valName = (np.median(self.attributes[idx, bestAttridx]),
                                 val)
            else:
                child.valName = val
            # get indices of child
            node.children.append(child)
            childidx = [i for i in idx if self.attrClone[i, bestAttridx] == val]
            # if no children with this attr value, give empty list and run id3
            if not childidx:               
                child.next = self._ID3Rec(childidx, [], child, 
                                          prevMax=max(set(labelsAttr), key=labelsAttr.count))
                # remove the attr we split on from attr names and run id3
            else:
                if attrNames and bestAttr in attrNames:
                    nextAttrs = attrNames.copy()
                    nextAttrs.pop(attrNames.index(bestAttr))
                    # print('next attr: ', nextAttrs, 'idx: ', childidx)
                child.next = self._ID3Rec(childidx, nextAttrs, child)
        
        return node # return tree root
  
class applyTree:
    """ Class for applying a trained decision tree made using the decisionTree class
    
    Main purpose here is to take the test data and see how the tree created on
    training data performs on this set. Note, although the apply function is
    recursive, it does not return anything since I didn't specify it to. It
    just updates the self.errs dataframe and we get that dataframe at the end
    """
    def __init__(self, trainedTree, test, labelsTest, numerical=False):
        self.errs = pd.DataFrame(columns=['Test Labels', 'Result', 'Acc']) # make err df
        self.root = trainedTree # root is the whole trained tree
        self.startTest = test # test data
        self.startLabels = labelsTest # test labels
        self.numerical = numerical # if data is numerical or not
        
    def _update_errs(self, subset, currNode):
        """Updates self.errs with the err at the leaf of the tree

        Parameters
        ----------
        :subLabel: labels of the test data subset that took the branches to list node
        :currNode : the leaf node in the tree we are at

        Returns
        -------
        None.
        """
        # labels of test data subset
        test_lab = subset.iloc[:,-1]
        # get value of node, make same len as test_lab, and find if they are the same
        trainLabel = [currNode.attribute]
        train_lab = np.array(trainLabel*len(test_lab))
        acc = train_lab == test_lab
        # make df and append to errs
        labdict = {'Test Labels': test_lab,
                   'Result': train_lab,
                   'Acc': acc}
        self.errs = self.errs.append(pd.DataFrame(labdict))
        
    def _applyRec(self, currNode, subset):
        """Main function for traversing the tree

        Parameters
        ----------
        :currNode: current node in the tree
        :subset: current subset of the test data
        :subLabel: current subset of test labels

        Returns
        -------
        None.

        """
        # if we hit a leaf, update the errs df and return None
        if currNode.leaf:
            self._update_errs(subset, currNode)
            return
        # split on the attribute of the node, loop through node children
        split_on = currNode.attribute
        for child in currNode.children:
            nextNode = child
            subVal = child.valName
            # if not sumerical, get the next attr subset and its labels and do again
            # print(isinstance(subVal[0], (int, float)) and self.numerical)
            # print(subVal, split_on)
            if not (isinstance(subVal[0], (int, float)) and self.numerical):       
                nextSubset = subset[subset[split_on] == subVal]
                # nextLabels = subLabel[subset[split_on] == subVal]
                self._applyRec(nextNode, nextSubset)
            # for numerical data, the median value was stored in the node, split on that
            else:
                above_median = subVal[1]
                if above_median:
                    nextSubset = subset[subset[split_on] > subVal[0]]
                    # nextLabels = subLabel[subset[split_on] > subVal[0]]
                else:
                    nextSubset = subset[subset[split_on] <= subVal[0]]
                    # nextLabels = subLabel[subset[split_on] <= subVal[0]]
                self._applyRec(nextNode, nextSubset)
      
def run_ID3(self):
    """runs the ID3 algo, give it an initialized decisionTree object

    Returns
    -------
    :self.node: the root node of the whole tree

    """
    idx = list(range(len(self.attributes))) # all indices of attributes
    attrNames = self.attrNames.copy() # attribute names
    self.node = self._ID3Rec(idx, attrNames, self.node) # get the root node using id3
    return self.node

def apply_ID3(self):
    """applies the decision tree to test data, give it an initialized applyTree
    
    Returns
    -------
    :errdf: the full dataframe each data point and its values
    :total_err: the accuracy of the tree (correct/total)
    """
    currNode = self.root # root node of tree
    allTest = self.startTest # test data
    # allLabels = self.startLabels # test labels
    self._applyRec(currNode, allTest) # run apply tree    
    errdf = self.errs # updated error df
    total_err = np.sum(errdf['Acc'])/len(errdf) # calculate total error
    
    return errdf, total_err
