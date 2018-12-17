# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 15:52:54 2018

@author: ych324
"""

import pandas
import os
import pickle
import gzip
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import sklearn.cluster

#load DATA
#x=pandas.read_csv("..\data\Po_multidose.csv")
#X = np.array(x)

sample_indice = 11800
tt = np.arange(0,48,0.1)
t_sparse = np.arange(0,48,1)
t_indice = t_sparse*10
Y=X[sample_indice,t_indice]
Y_variance = Y*(1+np.random.normal(size=Y.size)/50)
plt.scatter(t_sparse,Y);
plt.scatter(t_sparse,Y_variance)
