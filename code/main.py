# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 18:32:09 2018

@author: hasee
"""

import os
import pickle
import gzip
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

t = np.arange(0,12,1)
plt.plot(t,PO_multidose_ori(1.2,0.15,t,4))

Y = PO_multidose_ori(1.2,0.15,t,4)
Y_variance = Y*(1+np.random.normal(size=t.shape[0])/50)
Input = np.vstack((t,Y_variance))

