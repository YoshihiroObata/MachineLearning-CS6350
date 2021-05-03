# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 19:57:14 2021

@author: Yoshihiro Obata
"""
import numpy as np
from utils import shuffle, schedule

class layer:
    def __init__(self, num, fin, fout, layer_type='hidden', winit='Gauss'):
        self.layer_type = layer_type
        self.layer_num = num
        
        self.nodes = np.zeros(fout)
        if self.layer_type == 'hidden':
            self.s = np.zeros(fout)
        else:
            pass
        
        if winit == 'Gauss':
            np.random.seed(15)
            self.ws = np.random.randn(fin + 1, fout)
        elif winit == 'Zero':
            self.ws = np.zeros((fin + 1, fout))            
        else:
            self.ws = winit
            
        self.dnodes = np.zeros_like(self.nodes)
        self.dws = np.zeros_like(self.ws)
                
class nn:
    def __init__(self, w, data, T=5, lr0=0.01, d0=0.005, d=3, winit='Gauss'):
        self.w = w
        self.d = d
        self.T = T
        self.train = data.iloc[:,:-1].values
        self.labels = (data.iloc[:,-1].values)*2 - 1
        self.N = len(self.train)
        self.lr0 = lr0
        self.d0 = d0
        
        # layer creation
        if (winit == 'Gauss') or (winit == 'Zero'):
            inlayer = [layer(1, self.train.shape[1], w, layer_type='hidden', winit=winit)]
            hlayers = [layer(i, w, w, layer_type='hidden', winit=winit) for i in range(2,d,1)]
            outlayer = [layer(d, w, 1, layer_type='out', winit=winit)]
            self.layers = inlayer + hlayers + outlayer
        else:
            hlayers = [layer(i, w, w, layer_type='hidden', winit=winit[i-1]) for i in range(1,d,1)]
            outlayer = [layer(d, w, 1, layer_type='out', winit=winit[-1])]
            self.layers = hlayers + outlayer
        
    def ff(self, features):
        for layer in self.layers:
            n = layer.layer_num
            if len(features.shape) == 1:
                features = features.reshape(1,-1)
                
            # input layer
            # print(b.shape, self.layers[n-2].nodes.shape)
            if n == 1:
                b = np.ones((features.shape[0], 1))
                fin = np.append(b, features, axis=1)
            else:
                b = np.ones((self.layers[n-2].nodes.shape[0], 1))
                fin = np.append(b, self.layers[n-2].nodes, axis=1)
            
            node_vals = fin.dot(layer.ws)
            
            if layer.layer_type == 'out':
                layer.nodes = node_vals
            else:
                layer.s = node_vals
                layer.nodes = sigmoid(node_vals)
                
        return self.layers[2].nodes
    
    def bp(self, features, label):
        reverse_layers = np.flip(self.layers.copy())
        for layer in reverse_layers:
            n = layer.layer_num
            down_layer = self.d - (n - 1) # network depth - layer_num
            if layer.layer_type == 'out':
                down_nodes = np.append(1, reverse_layers[down_layer].nodes)
                
                dL = dLoss(layer.nodes, label)
                layer.dnodes = dL
                layer.dws = dLinear(dL, down_nodes)
            
            elif layer.layer_type == 'hidden':
                if layer.layer_num == 1:
                    down_nodes = np.append(1, features)
                else:
                    down_nodes = np.append(1, reverse_layers[down_layer].nodes)
                     
                up_layer = self.d - (n + 1) # +1 for layer num, +1 for index
                if up_layer == 0:
                    dUp = reverse_layers[up_layer].dnodes
                    wsUp = reverse_layers[up_layer].ws
                    dSig = dSigmoid(layer.s)
                    dN = dUp.reshape(1,-1).dot(wsUp.T)
                else:
                    dUp = reverse_layers[up_layer].dnodes[:,1:] # node values
                    wsUp = reverse_layers[up_layer].ws # upstream w [Nxw]
                    dSig = dSigmoid(layer.s)
                    dSigUp = dSigmoid(reverse_layers[up_layer].s) # [1xw]
                    dN = (dUp*dSigUp).reshape(1,-1).dot(wsUp.T)
                
                layer.dnodes = dN
                dNodes = dN[:,1:]
                layer.dws = down_nodes.reshape(-1,1).dot((dNodes*dSig).reshape(1,-1))
                
    def train_network(self):
        for t in range(self.T):
            lr = schedule(self.lr0, self.d0, t)
            # data, labels = shuffle(self.train, self.labels)
            data = self.train
            labels = self.labels
            # for ex in range(self.N):
            for ex in range(1):
                features = data[ex,:]
                lab = labels[ex]
                print(features, lab)
                self.ff(features)
                self.bp(features, lab)
                for layer in self.layers:
                    if layer.layer_num == self.d:
                        layer.ws -= lr*layer.dws.T
                    else:
                        layer.ws -= lr*layer.dws
                          
        return self
        
    def apply_network(self, test):
        test_data = test.iloc[:,:-1].values
        test_lab = test.iloc[:, -1].values.reshape(-1,1)
        test_lab = test_lab*2 - 1

        predict = (self.ff(test_data) >= 0)*2 - 1
        
        incorrect = predict != test_lab
        err = sum(incorrect)/len(incorrect)
        
        return err
    
    def train_and_apply(self, train, test):
        train_err = np.zeros(self.T)
        test_err = np.zeros(self.T)
        for t in range(self.T):
            lr = schedule(self.lr0, self.d0, t)
            data, labels = shuffle(self.train, self.labels)
            for ex in range(self.N):
                features = data[ex,:]
                lab = labels[ex]
                self.ff(features)
                self.bp(features, lab)
                for layer in self.layers:
                    if layer.layer_type == 'out':
                        layer.ws -= lr*layer.dws.T
                    else:
                        layer.ws -= lr*layer.dws
            
            err = self.apply_network(train)
            train_err[t] = err
            err = self.apply_network(test)
            test_err[t] = err
                          
        return self, train_err, test_err

# forward pass helper functions
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def loss(y, lab):
    return 0.5*(y - lab)**2

# backprop helper functions
def dLoss(y, lab):
    return y - lab

def dLinear(upchain, downchain):
    return upchain*downchain

def dSigmoid(s):
    return sigmoid(s)*(1 - sigmoid(s))      