# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 21:08:34 2018

Multiple dose fitting

@author: hasee
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
x=pandas.read_csv("E:\hasee\OneDrive - University of Kentucky\study\courses\Final Paper\CPSC-532M-project\data\Po_multidose.csv")
X = np.array(x)
plt.scatter(t_sparse,X[450,t_indice]);plt.scatter(t_sparse,X[451,t_indice]);plt.scatter(t_sparse,X[452,t_indice]);plt.scatter(tt,X[2253]);