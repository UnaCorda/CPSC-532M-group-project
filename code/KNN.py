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
"""
#load DATA
x=pandas.read_csv("..\data\Po_multidose.csv")
X = np.array(x)



sample_indice = 11800
tt = np.arange(0,48,0.1)
t_sparse = np.arange(0,48,1)
t_indice = t_sparse*10
Y=X[sample_indice,t_indice]
Y_variance = Y*(1+np.random.normal(size=Y.size)/50)
plt.scatter(t_sparse,Y);
plt.scatter(t_sparse,Y_variance)
"""
class KNN:
    def __init__(self,n_class):
        self.model = sklearn.neighbors.KNeighborsClassifier(n_class)
        self.model_dic={}
        self.DataList = ["Po_multidose.csv"]
        self.ParaList = ["PO_multidose_para.csv"]
    def fit(self):
        """Read in the Data"""
        X_all = np.array([])
        Y_all = np.array([])
        modeltype=0
        for i in self.DataList:
            
            x=pandas.read_csv(os.path.join('..','data',i))
            X=np.array(x)
            if modeltype == 1:
                X_all = np.array(x)
            else:
                X_all=np.vstack((X_all,X))
            self.Y_all = np.append(Y_all,np.ones(X.shape[0])*modeltype)
            
            modeltype+=1
            """
        for j in self.ParaList:
            y = pandas.read_csv(os.path.join('..','data',j))
            np.append(Y_all,y)
            """
        Y_label = np.arange(X_all.shape[0])
        
        self.model.fit(X_all,Y_label)
        
    def predict(self,x):
        result_list = self.model.predict(x)
        out1 = self.Y_all[result_list]
        y = pandas.read_csv(os.path.join('..','data',self.ParaList[out1]))
        Y = np.array(y)
        size0 = (pandas.read_csv(os.path.join('..','data',self.ParaList[0]))).shape[0]
        size1 = (pandas.read_csv(os.path.join('..','data',self.ParaList[1]))).shape[0]
        size2 = (pandas.read_csv(os.path.join('..','data',self.ParaList[2]))).shape[0]
        size3 = (pandas.read_csv(os.path.join('..','data',self.ParaList[3]))).shape[0]
        if out1 == 0:
            
            return 0
        elif out1 == 1:
            return 0
        elif out1 ==2:
            return 0
        elif out1 ==3:
            return 0
        return result_list,out1
    
    def return_para():
        
        