#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 14:49:48 2018

@author: smuch
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import pymc3 as pm
import scipy.stats as st
#%% Sampler statistics

model1 = pm.Model()

with model1:
    mu1 = pm.Normal("mu1", mu=0, sd=1, shape=10)
    step = pm.NUTS()
    trace = pm.sample(2000, tune=1000, init=None, step=step, cores=2)
    
#%%    

plt.plot(trace['step_size_bar'])
#%%

sizes1, sizes2 = trace.get_sampler_stats('depth', combine=False)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
ax1.plot(sizes1)
ax2.plot(sizes2)

accept = trace.get_sampler_stats('mean_tree_accept', burn=1000)
sb.distplot(accept, kde=False)
#%%
accept.mean()
#%%
trace['diverging'].nonzero()
#If the overall distribution of energy levels has longer tails, the efficiency of the sampler will deteriorate quickly.
energy = trace['energy']
energy_diff = np.diff(energy)
sb.distplot(energy - energy.mean(), label='energy')
sb.distplot(energy_diff, label='energy diff')
plt.legend()
#%%
#Multiple samplers
#%%

model2 = pm.Model()
with model2:
    mu1 = pm.Bernoulli("mu1", p=0.8)
    mu2 = pm.Normal("mu2", mu=0, sd=1, shape=10)
    step1 = pm.BinaryMetropolis([mu1])
    step2 = pm.Metropolis([mu2])
    trace = pm.sample(10000, init=None, step=[step1, step2], cores=2, tune=1000)

#%% Both samplers export accept, so we get one acceptance probability for each sampler:
print(trace.stat_names)
print(trace.get_sampler_stats('accept'))

#%%%
#Posterior Predictive Checks
#%%
data = np.random.randn(100)

with pm.Model() as model3:
    mu = pm.Normal('mu', mu=0, sd=1, testval=0)
    sd = pm.HalfNormal('sd', sd=1)
    n = pm.Normal('n', mu=mu, sd=sd, observed=data)

    trace = pm.sample(5000)
pm.traceplot(trace);
#%%
ppc = pm.sample_ppc(trace, samples=500, model=model, size=100)
print(np.asarray(ppc['n']).shape)

_, ax = plt.subplots(figsize=(12, 6))
ax.hist([n.mean() for n in ppc['n']], bins=19, alpha=0.5)
ax.axvline(data.mean())
ax.set(title='Posterior predictive of the mean', xlabel='mean(x)', ylabel='Frequency');
#%%
    

