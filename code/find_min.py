# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 15:38:28 2018

@author: hasee
"""

import PK_model
import RBF_rewrite
from scipy import optimize
import numpy as np
import sklearn.cluster

def objfun_PK(x):
    return PO_multidose(0.3,0.5,[x],10)[0]

def objfun_RBF(x,model):
    return  model.predict(x[:,np.newaxis])[0]

k = RBF(length_scale= 1, length_scale_bounds=(1e-1, 1e+2))
#Input points are nX2 long matrix
def return_intervals(Input_points,kernal=k,seeds_int=0.3,eps=0.001,min_samples=20,objfun=objfun_RBF,):
    cluster_model = sklearn.cluster.DBSCAN(eps=0.001,min_samples=10)
    test_point = np.arange(1,48,seeds_int)
    gp = GaussianProcessRegressor(kernel=kernal ,n_restarts_optimizer=12)
    t_input=Input_points[:,0][:,np.newaxis]
    y_input=Input_points[:,1]
    gp.fit(t_input,y_input)
    def fun(x):
        return objfun(x,model=gp)

    min_points=np.array(list(map(lambda c: optimize.minimize(fun,c,method='BFGS').x,test_point)))
    
    cluster_model = sklearn.cluster.DBSCAN(eps=0.001,min_samples=10)
    cluster_model.fit(min_points)
    
    matrix = np.vstack((cluster_model.labels_,min_points[:,0])).T
    #The Label-min_points without the outlier
    matrix_clean=matrix[matrix[:,0]!=-1]
    group = np.unique(matrix_clean[:,0])
    #Calculate the mean of the minpoints
    mean_minpoints=np.array(list(map(lambda x : matrix_clean[matrix_clean[:,0]==x][:,1].mean(),group)))
    mean_minpoints_include_zero = np.append([0],mean_minpoints)
    mean_minpoints.sort()
    plt.plot(Input_points[:,0],gp.predict(Input_points[:,0][:,np.newaxis]))
    plt.scatter(mean_minpoints_include_zero,gp.predict(mean_minpoints_include_zero[:,np.newaxis]),c='red')
    
    return mean_minpoints_include_zero

Input=np.vstack((t_sparse,X[9200,t_indice])).T

#The function used for fitting the multiple dose
def split_fit(Input_points,PK_model=PK_model.PO_onedose):
    return NotImplementedError
    

