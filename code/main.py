# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 18:32:09 2018

@author: hasee
"""

import os
import PK_model
import pickle
import gzip
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

t = np.arange(0,100,1)
plt.plot(t,PK_model.PO_multidose_ori(1.2,0.15,t,4))

Y = PK_model.PO_multidose_ori(1.2,0.15,t,4)
Y_variance = Y*(1+np.random.normal(size=t.shape[0])/20)
Input = np.vstack((t,Y_variance))

t = np.arange(0,48,1)
plt.plot(t,PK_model.IV_onedose(2,t))

Y =PK_model.IV_onedose(2,t)
Y_variance = Y*(1+np.random.normal(size=t.shape[0])/20)
Input = np.vstack((t,Y_variance))
