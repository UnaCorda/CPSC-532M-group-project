# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 15:38:28 2018

@author: hasee
"""

import PK_model
from scipy import optimize
import numpy as np

def objfun(x):
    return PO_multidose(0.3,0.5,[x],10)[0]

test_point = np.arange(1,48,0.5)



optimize.minimize(objfun,31,method='BFGS').x

min_points=np.array(list(map(lambda c: optimize.minimize(objfun,c,method='BFGS').x,test_point)))
zero_point.reshape(-1,1)
plt.scatter(np.arange(zero_point.size),zero_point)