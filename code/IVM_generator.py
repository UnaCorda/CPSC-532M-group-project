import PK_model
import numpy as np
import multiprocessing
from multiprocessing import Pool
from numpy import random


def fun(x):
    t = np.arange(0,48,0.1)
    return PK_model.IV_multidose(x[0],t,x[1],x[0])

#tt=np.arange(0,48,0.1)
#t_sparse = np.arange(0,48,4)
#t_indice = np.arange(0,480,40)
#
#t = t_indice

tau=np.arange(3,24,0.5)

k1=np.arange(0.010,0.15,0.01)
k2=np.arange(0.10,2.1,0.1)
k=np.append(k1,k2)

#Generate the combination
L=np.repeat(k,tau.size)
N=np.matrix.flatten(np.repeat([tau],k.size,axis=0))
combined_array=np.array([L]+[N])
combined_T=combined_array.T

print(combined_T)
p = Pool(4)
#out=np.array(list(p.map(lambda x: IV_multidose(x[0],x[1],t,x[2]),multi_combanation_T)))
"""Using the map function to do tbe paralization"""
out = np.array(list(p.map(fun,combined_T)))


np.savetxt("IVM.csv", out, delimiter=",")
np.savetxt("IVM_para.csv",combined_T,delimiter=",")


test = np.random.rand(10,2)
test[:,1]  = test[:,1]*24
out = np.array(list(p.map(fun,combined_T)))

np.savetxt("IVM_t.csv", out, delimiter=",")
np.savetxt("IVM_para_t.csv",combined_T,delimiter=",")