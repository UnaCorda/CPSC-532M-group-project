# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 17:27:20 2018

@author: hasee
"""

import PK_model.py
import numpy as np
#Generate the Time serials 

t=np.arange(0,48,0.1)

ka1=np.arange(0.012,0.152,0.01)
ka2=np.arange(0.152,5,0.1)
ka=np.append(ka1,ka2)

k1=np.arange(0.010,0.15,0.01)
k2=np.arange(0.10,5,0.1)
k=np.append(k1,k2)
#Generate the combination
K=np.repeat(ka,ka.size,axis=0)
L=np.matrix.flatten(np.repeat([k],k.size,axis=0))
combined_array=np.array([K]+[L])
combined_array_T=combined_array.T

out=np.array(list(map(lambda x: PO_onedose(x[0],x[1],t),combined_array_T)))
#Ka serials
