# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:42:10 2018

@author: hasee
"""

import numpy as np


def IV_onecom(k,t,c0=1):
    C =c0*np.exp(-k*t)
    return C
def PO_onecom(ka,k,t):
    C =ka/(ka-k)*(np.exp(-k*t)-np.exp(-ka*t))
    return C
def Infusion(k,t,c0=0):
    C =1-np.exp(-k*t)+c0*np.exp(-k*t)
    return C


class IV_onecom_class:
    def __init__(self,k,t,start_point,c0):
        self.k = k
        self.t1 = t-start_point
        self.start_point = start_point
        self.c0 = c0
    def predict(self):
        if self.t1.any() < 0:
            return 0
        else:
            return IV_onecom(self.k,self.t1,self.c0)

class PO_onecom_class:
    def __init__(self,ka,k,t,start_point):
        self.ka = ka
        self.k = k
        self.t1 = t-start_point
        self.start_point = start_point

    def predict(self):
        if self.t1[0] < 0:
            return 0
        else:
            return  PO_onecom(self.ka,self.k,self.t1)        

    def predict(self):
        if self.t1[0] < 0:
            return 0
        else:
            return  PO_onecom(self.ka,self.k,self.t1)          



def IV_multidose(k,t,tau,c1,c0=1):
    
    def single_predict(t0):
        model=[]
        for i in range(int(np.floor(t0/tau)+1)):
            if i ==0:
                model.append(IV_onecom_class(k,t0,0,c1))
            else:
                model.append(IV_onecom_class(k,t0,tau*i,c0))
                
        result_list = list(map(lambda x : x.predict(),model))
        return np.sum(result_list)
        
    out=list(map(single_predict,t))
    #result_list = list(map(lambda x : x.predict(),model))
    return out


def PO_multidose(ka,k,t,tau):
    def single_predict(t0):
        model=[]
        for i in range(int(np.floor(t0/tau)+1)):
            if i ==0:
                model.append(PO_onecom_class(ka,k,t0,0))
            else:
                model.append(PO_onecom_class(ka,k,t0,tau*i))
                
        result_list = list(map(lambda x : x.predict(),model))
        return np.sum(result_list)
        
    out=list(map(single_predict,t))
    #result_list = list(map(lambda x : x.predict(),model))
    return out

def PO_multidose(ka,k,t,tau):
    def single_predict(t0):
        model=[]
        for i in range(int(np.floor(t0/tau)+1)):
            if i ==0:
                model.append(PO_onecom_class(ka,k,t0,0))
            else:
                model.append(PO_onecom_class(ka,k,t0,tau*i))
                
        result_list = list(map(lambda x : x.predict(),model))
        return np.sum(result_list)
        
    out=list(map(single_predict,t))
    #result_list = list(map(lambda x : x.predict(),model))
    return out

def PO_onedose(ka,k,t):
    def single_predict(t0):
        result=PO_onecom_class(ka,k,t0,0)
        return result.predict()
        
    out=list(map(single_predict,t))
    #result_list = list(map(lambda x : x.predict(),model))
    return out

def PO_onedose_fitting(ka,k,t,t0):
    def single_predict(tx):
        result=PO_onecom_class(ka,k,tx,t0)
        return result.predict()
        
    out=list(map(single_predict,t))
    #result_list = list(map(lambda x : x.predict(),model))
    return out

#optimize.minimize(fun,([1,2,1]))
"""def fun(A):
    a=A[0]
    b=A[1]
    c=A[2]
    return np.square(np.array(PO_multidose(a,b,X,c))-np.array(Y)).sum()"""