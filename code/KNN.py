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
import RBF_rewrite
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
"""
#load DATA
x=pandas.read_csv("..\data\IV_onedose.csv")
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
        self.DataList = np.array(["Po_multidose.csv","IVM.csv","Po_onedose.csv","IV_onedose.csv"])
        #self.DataList = np.array(["IV_onedose.csv","Po_multidose.csv"])
        self.ParaList = np.array(["Po_multidose_para.csv","IVM_para.csv","PO_onedose_para.csv","IV_onedose_para.csv"])
        #self.ParaList = np.array(["IV_onedose_para.csv","PO_multidose_para.csv"])
        self.kernal = RBF(length_scale= 1, length_scale_bounds=(1e-1, 1e+2))
        self.gp = GaussianProcessRegressor(kernel=self.kernal ,n_restarts_optimizer=12)

    def RBF_interpolation(self,Input_points):
        Input_points = Input_points.T
        test_points = np.arange(0,48,0.1)
        t_input=Input_points[:,0][:,np.newaxis]
        y_input=Input_points[:,1]
        self.gp.fit(t_input,y_input)
        out = self.gp.predict(test_points[:,np.newaxis])
        return np.array([out])

    def fit(self):
        """Read in the Data"""
        X_all = np.array([])
        Y_all = np.array([])
        modeltype=0
        for i in self.DataList:
            x=pandas.read_csv(os.path.join('..','data',i))
            X=np.array(x)
            if modeltype == 0:
                X_all = X
                Y_all=np.ones(X.shape[0])*modeltype
            else:
                X_all=np.vstack((X_all,X))
                Y_all = np.append(Y_all,np.ones(X.shape[0])*modeltype)

            self.X_all = X_all
            self.Y_all = np.append(Y_all,np.ones(X.shape[0])*modeltype)
            modeltype=modeltype+1
            self.modeltype = modeltype
            """
        for j in self.ParaList:
            y = pandas.read_csv(os.path.join('..','data',j))
            np.append(Y_all,y)
            """
        Y_label = np.arange(X_all.shape[0])

        self.model.fit(X_all,Y_label)

    def predict(self,Input):
        x = self.RBF_interpolation(Input)

        result_list = self.model.predict(x)
        self.result_list = result_list
        out1 = self.Y_all[result_list]
        indice = np.int(out1[0])
        y = pandas.read_csv(os.path.join('..','data',self.ParaList[indice]))
        self.Y = np.array(y)
        size0 = np.array(pandas.read_csv(os.path.join('..','data',self.ParaList[0]))).shape[0]
        size1 = np.array(pandas.read_csv(os.path.join('..','data',self.ParaList[1]))).shape[0]
        size2 = np.array(pandas.read_csv(os.path.join('..','data',self.ParaList[2]))).shape[0]
        size3 = np.array(pandas.read_csv(os.path.join('..','data',self.ParaList[3]))).shape[0]
        self.size0 = size0
        self.size1 = size1
        self.size2 = size2
        self.size3 = size3
        if out1 == 0:
            para = self.Y[result_list[0]]
        elif out1 == 1:
            para = self.Y[result_list[0]-size0]
        elif out1 ==2:
            para = self.Y[result_list[0]-size0-size1]
        elif out1 ==3:
            para = self.Y[result_list[0]-size0-size1-size2]

        return result_list,out1,para

    def return_para():
        return NotImplemented
