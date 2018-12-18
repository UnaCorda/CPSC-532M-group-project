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

def fu(x):
    return PK_model.PO_onedose(x[0],x[1])

tt=np.arange(0,48,0.1)
t_sparse = np.arange(0,48,4)
t_indice = np.arange(0,480,40)

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
p=multiprocessing.Pool(8)
out=np.array(list(p.map(fu,combined_array_T)))
#out=np.array(list(map(lambda x: PO_onedose(x[0],x[1],tt),combined_array_T)))
np.savetxt("..\data\PO_onedose.csv", out, delimiter=",")
#IV_onedose 
def IV(x):
    return PK_model.IV_onecom(x,tt)
k1=np.arange(0.010,0.15,0.01)
k2=np.arange(0.10,5.1,0.1)
k=np.append(k1,k2)
out=np.array(list(map(IV,k)))
np.savetxt("..\data\IV_onedose.csv", out, delimiter=",")
np.savetxt("..\data\IV_onedose_para.csv", k.reshape(-1,1), delimiter=",")

#for PO multidose
def fun(x):
    return PK_model.PO_multidose(x[0],x[1],t,x[2])

tau=np.arange(1,24,0.5)

ka1=np.arange(0.012,0.152,0.01)
ka2=np.arange(0.152,2.152,0.1)
ka=np.append(ka1,ka2)

k1=np.arange(0.010,0.15,0.01)
k2=np.arange(0.10,2.1,0.1)
k=np.append(k1,k2)
#Generate the combination
K=np.repeat(ka,ka.size,axis=0)
L=np.matrix.flatten(np.repeat([k],k.size,axis=0))
combined_array=np.array([K]+[L])
combined_array_T=combined_array.T

N=np.matrix.flatten(np.repeat([tau],combined_array.shape[1],axis=-1))

A=np.repeat(combined_array,tau.size,axis=-1)

multi_combanation=np.append(A,[N],axis=0)
multi_combanation_T=multi_combanation.T

modeltype = 4
label = np.ones(multi_combanation_T.shape[0])*modeltype
np.hstack((Y,label.reshape(-1,1)))

#p=multiprocessing.Pool(8)
#out=np.array(list(p.map(lambda x: PO_multidose(x[0],x[1],t,x[2]),multi_combanation_T)))
"""Using the map function to do tbe paralization"""
#out=np.array(list(p.map(fun,multi_combanation_T[1:500])))


