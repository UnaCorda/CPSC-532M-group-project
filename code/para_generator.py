# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 17:27:20 2018

@author: hasee
"""

import PK_model
import numpy as np
import multiprocessing
from multiprocessing import Pool
#Generate the Time serials 

t=np.arange(0,48,0.1)

ka1=np.arange(0.012,0.152,0.01)
ka2=np.arange(0.152,5.152,0.1)
ka=np.append(ka1,ka2)

k1=np.arange(0.010,0.15,0.01)
k2=np.arange(0.10,5.1,0.1)
k=np.append(k1,k2)
#Generate the combination
K=np.repeat(ka,ka.size,axis=0)
L=np.matrix.flatten(np.repeat([k],k.size,axis=0))
combined_array=np.array([K]+[L])
combined_array_T=combined_array.T

#out=np.array(list(map(lambda x: PO_onedose(x[0],x[1],t),combined_array_T)))
#Ka serials


#for PO multidose
def fun(x):
    return PO_multidose(x[0],x[1],t,x[2])

tau=np.arange(1,24,0.5)

ka1=np.arange(0.012,0.152,0.01)
ka2=np.arange(0.152,5.152,0.1)
ka=np.append(ka1,ka2)

k1=np.arange(0.010,0.15,0.01)
k2=np.arange(0.10,5.1,0.1)
k=np.append(k1,k2)
#Generate the combination
K=np.repeat(ka,ka.size,axis=0)
L=np.matrix.flatten(np.repeat([k],k.size,axis=0))
combined_array=np.array([K]+[L])
combined_array_T=combined_array.T

N=np.matrix.flatten(np.repeat([tau],combined_array.shape[1],axis=-1))

A=np.repeat(combined_array,tau.size,axis=-1)

multi_combanation=np.append(A,[N],axis=0)
multi_combanation_T=multi_combanation.T[1:10]

p=multiprocessing.Pool(8)
#out=np.array(list(p.map(lambda x: PO_multidose(x[0],x[1],t,x[2]),multi_combanation_T)))
out=np.array(list(p.map(fun,multi_combanation_T)))




numpy.savetxt("foo.csv", out, delimiter=",")

