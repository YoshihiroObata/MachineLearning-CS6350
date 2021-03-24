# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:18:21 2021

@author: Yoshihiro Obata
"""
# %% 
import pandas as pd
from perceptron import Perceptron
import matplotlib.pyplot as plt

# %%
cols = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
train = pd.read_csv('bank-note/bank-note/train.csv', names=cols)
test = pd.read_csv('bank-note/bank-note/test.csv', names=cols)

# %% testing each method
T = 10
lr = 0.01
percep_init = Perceptron(T, lr, train)
trained_percep = Perceptron.train_percep(percep_init)
err_standard = Perceptron.predict_percep(trained_percep, test)

#  voted 
voted_init = Perceptron(T, lr, train, variant='voted')
trained_voted = Perceptron.train_voted(voted_init)
err_voted = Perceptron.predict_percep(trained_voted, test)

#  averaged perceptron
averaged_init = Perceptron(T, lr, train, variant='averaged')
trained_averaged = Perceptron.train_averaged(averaged_init)
err_averaged = Perceptron.predict_percep(trained_averaged, test)

# %%
train1, train2, train3 = [], [], []
err1, err2, err3 = [], [], []

print('Starting to train 100 Perceptrons of each variant...')
for i in range(100):
    percep_init = Perceptron(T, lr, train)
    trained_percep = Perceptron.train_percep(percep_init)
    train1.append(Perceptron.predict_percep(trained_percep, train))
    err1.append(Perceptron.predict_percep(trained_percep, test))
    
    voted_init = Perceptron(T, lr, train, variant='voted')
    trained_voted = Perceptron.train_voted(voted_init)
    train2.append(Perceptron.predict_percep(trained_voted, train))
    err2.append(Perceptron.predict_percep(trained_voted, test))
    
    averaged_init = Perceptron(T, lr, train, variant='averaged')
    trained_averaged = Perceptron.train_averaged(averaged_init)
    train3.append(Perceptron.predict_percep(trained_averaged, train))
    err3.append(Perceptron.predict_percep(trained_averaged, test))
    if i%10 == 0:
        print(f'{i}% done...')
print('Done')

# %% plotting
fig, ax = plt.subplots(figsize=(5,5))
plt.boxplot([train1, err1, train2, err2, train3, err3], showmeans=True, 
            meanline=True, positions=[1,1.75,3,3.75,5,5.75],
            labels=['Standard Train','Standard Test','Voted Train',
                    'Voted Test','Averaged Train','Averaged Test'])
plt.ylim([0,0.06])
plt.xticks(fontsize=10, rotation=45, ha='right')
plt.yticks(fontsize=12)
plt.ylabel('Error', fontsize=14)
plt.savefig('errs.png', dpi=300, bbox_inches='tight')

# %%
results = f"""Results from Perception
------------------------------------------------------------------------------
One Perceptron for 10 epochs:
------------------------------------------------------------------------------
Standard Perceptron:
    Weight vector: {trained_percep.w}
    Error: {err_standard}
    
Voted Perceptron:
    Number of weight vectors: {len(trained_voted.ws)}
    Weight vector 50:\t\t{trained_voted.ws[49]}
    Weight vector 100:\t\t{trained_voted.ws[99]}
    Weight vector 150:\t\t{trained_voted.ws[149]}
    Weight vector 200:\t\t{trained_voted.ws[199]}
    Final weight vector:\t{trained_voted.w}
    Error: {err_voted}
    
Averaged Perceptron:
    Weight vector: {trained_averaged.w}
    Error: {err_averaged}
"""
print(results)
txt = open('PercepResults.txt', 'wt')
n = txt.write(results)
txt.close()