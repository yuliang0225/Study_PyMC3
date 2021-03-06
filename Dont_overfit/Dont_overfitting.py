#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 16:22:21 2018
https://www.kaggle.com/c/overfitting/discussion/593

@author: smuch
"""

#%%
import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano.tensor as TT
import theano
#%%
# Set data path
data_path = '/Users/smuch/Documents/coding/Study_PyMC3/Dont_overfit/'
#%%
data = pd.read_csv(data_path+'overfitting.csv')
#%% Get Traindata and Test data 
training_data = data[data.iloc[:, 1] == 1].iloc[:,5:]
testing_data = data[data.iloc[:, 1] == 0].iloc[:,5:]

training_labels = data[data.iloc[:, 1] == 1].iloc[:,2]
testing_labels = data[data.iloc[:, 1] == 0].iloc[:,2]

print ("training:", training_data.shape, training_labels.shape)
print ("testing: ", testing_data.shape, testing_labels.shape)
#%% 
# He mentions that the X variables are from a Uniform distribution. Let's investigate this:
#figsize(12, 4)
plt.hist(np.array(training_data).flatten())
print (training_data.shape[0] * training_data.shape[1])

training_data = np.array(training_data)
testing_data = np.array(testing_data)

#%%    

def Z(coef, to_include, training_data):
    ym = TT.dot(to_include * training_data, coef)
    return ym - ym.mean()

def T(z=Z):
    return TT.switch(z<0,1.0,0)

with pm.Model() as model:
    
# Select the para?
    to_include = pm.Bernoulli("to_include", 0.5, shape=200)
# To give a diss for 200 vars 
    coef = pm.Uniform("coefs", 0, 1, shape=200)   
    
    Z = pm.Deterministic("Z",Z(coef=coef, to_include=to_include, training_data=theano.shared(training_data)))
    
    T = pm.Deterministic("T",T(Z))
    
    obs = pm.Bernoulli("obs", T, observed=training_labels)
    
    start = pm.find_MAP()
    step = pm.Metropolis()
    trace = pm.sample(100000, step=step, start=start)

#%%
pm.traceplot(trace)
#%%
(np.round((trace["T"][-500:-300, :]).mean())== training_labels).mean()
