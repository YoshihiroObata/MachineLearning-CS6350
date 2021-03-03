# -*- coding: utf-8 -*-
"""
Contains a function to run an ID3 algorithm with options for entropy, majority
error, and gini index information gain. Contains a node and decision tree
class as well as an applying tree class and applying function

@author: Yoshihiro Obata
"""
# Importing Packages
import numpy as np
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
    def __init__(self, df, method='entropy', depth=None, randTieBreak=True, 
                 numerical=False, weights=None, 
                 small_sub=False, globaldf=None):
        self.attributes = np.array(df.iloc[:,:-1]) # np array with columns as attributes
        self.attrNames = np.array(df.columns[:-1]) # attribute names
        self.labels = np.array(df.iloc[:,-1]) # np array of labels (target)
        self.labelSet = list(set(self.labels)) # list of unique labels
        self.node = None # node
        self.numerical = numerical # bool of if numerical data
        self.media = None
        self.numerical_idx = []
        self.randPick = randTieBreak # pick random info gain attrName if tied 
        
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
            
        if weights is None:
            self.weights = np.ones((len(self.attributes),))    
        else:
            self.weights = weights
        
        if small_sub:
            self.small_sub = small_sub
            self.globaldf = np.array(globaldf.iloc[:,:-1])
        else:
            self.small_sub = small_sub
        
    def _getEntropy(self, idx):
        """Calculates the entropy of a given list of indices of an attribute
         
        Params
        -----
        Inputs:
        
        :idx: a list of indices for a given attribute
        
        Outputs:
        
        :entropy: the log2 entropy of the attribute
        """
        labels = self.labels[idx] # remaking the labels
        weights = self.weights[idx]
        # np array of label count
        
        labelCount = np.zeros((len(self.labelSet),))
        for i, lab in enumerate(self.labelSet):
            setidx = np.array(labels) == lab
            labelCount[i] = sum(weights[setidx])
        
        if len(labelCount[labelCount != 0]) == 1:
            entropy = 0
        else:
            ps = labelCount/sum(labelCount)
            ps = ps[ps != 0]
            entropy = -(np.sum(ps*np.log2(ps)))

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
        labels = self.labels[idx] # remaking the labels
        weights = self.weights[idx]
        
        labelCount = np.zeros((len(self.labelSet),))
        for i, lab in enumerate(self.labelSet):
            setidx = np.array(labels) == lab
            labelCount[i] = sum(weights[setidx])
        
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
        labels = self.labels[idx] # remaking the labels
        # np array of label count
        weights = self.weights[idx]   
        # np array of label count
        
        labelCount = np.zeros((len(self.labelSet),))
        for i, lab in enumerate(self.labelSet):
            setidx = np.array(labels) == lab
            labelCount[i] = sum(weights[setidx])
        # for each unique label, calculate gini
        ps = labelCount/sum(labelCount)
        gini = 1 - (np.sum(ps**2))
            
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
        attrVals = self.attributes[idx,attrID]
        w = sum(self.weights[idx])
        attrSet = list(set(attrVals)) # list of unique vals for attrVals
        infoGainAttr = 0
        # uses info gain method to calc info gain for the attr
        for value in range(len(attrSet)):
            idxloc = np.where(attrVals == attrSet[value])[0]
            attridx = list(idx[list(idxloc)])
            attr_w = sum(self.weights[attridx])
            gainSubset = self.gainMethod(attridx)
            infoGainAttr += attr_w/w*gainSubset
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
            if sum(maxGain) != 1:
                tieidx = [i for i in range(len(maxGain)) if maxGain[i] == True]
                randidx = random.choice(tieidx)
                bestAttr = attrNames[attrInfoGain.index(attrInfoGain[randidx])]
                bestAttridx = attrIDs[attrInfoGain.index(attrInfoGain[randidx])]
            else:
                bestAttr = attrNames[attrInfoGain.index(max(attrInfoGain))]
                bestAttridx = attrIDs[attrInfoGain.index(max(attrInfoGain))]
                
        # if not random, max picks the first thing
        else:
            bestAttr = attrNames[attrInfoGain.index(max(attrInfoGain))]
            bestAttridx = attrIDs[attrInfoGain.index(max(attrInfoGain))]
        
        return bestAttr, bestAttridx
    
    def _num2Bool(self, idx, attrNames):
        """ Changes the self.attributes variable to a boolean of > and <= median

        Parameters
        ----------
        :idx: indices of all attributes
        :attrNames: names of attributes
            
        Returns
        -------
        None.
        """
        # get the attr indices, get the first value of these attributes
        attrIDs = list(range(len(self.attrNames)))
        attrID_type = self.attributes[0,attrIDs]
        # get the attr index if the first val is a float or int
        numAttrID = [ID for i, ID in enumerate(attrIDs) if isinstance(attrID_type[i], (float,int))]
        # find median, split if > (True) or <= (False) median, update self.attributes
        self.media = np.zeros((len(attrNames),))
        for attr in numAttrID:
            median = np.median(self.attributes[idx, attr])
            # print(median)
            self.media[attr] = median
            # print(self.media)
            self.numerical_idx.append(attr)
            self.attributes[:,attr] = self.attributes[:,attr] > median
            if self.small_sub:
                self.globaldf[:,attr] = self.globaldf[:,attr] > median
    
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
        labelsAttr = self.labels[idx] # get labels for all idx
        
        # stopping conditions:
        # if there's only 1 label, return node with attribute label that is leaf
        if len(set(labelsAttr)) == 1:
            node.attribute = self.labels[idx[0]]
            node.leaf = True
            return node
        # if there's no attrNames, return the node as leaf node
        elif len(idx) == 0:
            # print('this happened')
            node.attribute = prevMax
            node.leaf = True
            return node
        
        unique, pos = np.unique(labelsAttr, return_inverse=True)
        sub_common = unique[np.argmax(np.bincount(pos))]
        
        # if you hit depth limit, set the node attr to the max in the subset
        if node.depth == self.depthLimit:
            node.attribute = sub_common
            node.leaf = True
            return node

        bestAttr, bestAttridx = self._getNextAttr(idx, attrNames)
        node.attribute = bestAttr
        node.children = []
        if self.small_sub:
            chosenAttrVals = list(set(self.globaldf[:, bestAttridx]))
        else:
            chosenAttrVals = list(set(self.attributes[:, bestAttridx]))
        
        for val in chosenAttrVals: # for all vals the attr can take:
            # make new child node, append it to child list, update depth
            child = Node()
            # node.children.append(child)
            child.depth = node.depth + 1
            # if the data is numerical, set child val to the median value, else, make it val
            child.valName = val
            # get indices of child
            node.children.append(child)
            idxloc = np.where(self.attributes[idx, bestAttridx] == val)[0]
            childidx = idx[list(idxloc)]
            # if no children with this attr value, give empty list and run id3
            if len(childidx)==0:               
                child.next = self._ID3Rec(childidx, [], child, 
                                          prevMax=sub_common)
            # remove the attr we split on from attr names and run id3
            else:
                if len(attrNames) != 0 and bestAttr in attrNames:
                    nextAttrs = attrNames.copy()
                    idx2del = np.where(bestAttr == nextAttrs)[0][0]
                    nextAttrs = np.delete(nextAttrs, idx2del)
                child.next = self._ID3Rec(childidx, nextAttrs, child)
        
        return node # return tree root
  
class applyTree:
    """ Class for applying a trained decision tree made using the decisionTree class
    
    Main purpose here is to take the test data and see how the tree created on
    training data performs on this set. Note, although the apply function is
    recursive, it does not return anything since I didn't specify it to. It
    just updates the self.errs dataframe and we get that dataframe at the end
    """
    def __init__(self, trainedTree, test, treeInit, numerical=False, 
                 weights=None):
        # self.errs = pd.DataFrame(columns=['Test Labels', 'Result', 'Acc']) # make err df
        self.root = trainedTree # root is the whole trained tree
        self.attrNames = np.array(test.columns)
        self.startTest = np.array(test.iloc[:,:-1]) # test DataFrame
        self.startLabels = np.array(test.iloc[:,-1]) # test labels
        self.numerical = numerical # if data is numerical or not
        if self.numerical:
            self.media = treeInit.media
            self.numerical_idx = treeInit.numerical_idx
        if weights is None:
            self.weights = np.ones((len(self.startTest),))    
        else:
            self.weights = weights
        self.predict = []
            
    def _applyLoops(self, currNode, subset, sublab):
        errs = np.zeros((len(sublab),))
        for row in range(len(subset)):
            leaf = False
            node = currNode
            while not leaf:
                split_on = node.attribute
                split_idx = np.where(self.attrNames==split_on)[0]
                nextval = subset[row,split_idx]
                for child in node.children:
                    if child.valName == nextval:
                        node = child
                        break
                if node.leaf == True:
                    leaf = True 
            errs[row] = sublab[row] == node.attribute
            self.predict.append(node.attribute)
        return errs
    
def run_ID3(self):
    """runs the ID3 algo, give it an initialized decisionTree object

    Returns
    -------
    :self.node: the root node of the whole tree
    """
    idx = np.arange(len(self.attributes)) # all indices of attributes
    attrNames = self.attrNames.copy() # attribute names
    if self.numerical:
        self._num2Bool(idx, attrNames)
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
    allLabels = self.startLabels # test labels
    if self.numerical:
        for idx in self.numerical_idx:
            allTest[:,idx] = allTest[:,idx].copy() > self.media[idx]
    errs = self._applyLoops(currNode, allTest, allLabels) # run apply tree
    weighted_errs = errs*self.weights
    total_err = np.sum(weighted_errs)/sum(self.weights)
    
    return errs, total_err
