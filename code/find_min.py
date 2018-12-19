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

#k = RBF(length_scale= 1, length_scale_bounds=(1e-1, 1e+2))
#Input points are nX2 long matrix
def return_intervals(Input_points,kernal=None,seeds_int=0.1,eps=0.5,min_samples=30,objfun=objfun_RBF,):
    cluster_model = sklearn.cluster.DBSCAN(eps=eps,min_samples=min_samples)
    kernal = RBF(length_scale= 1, length_scale_bounds=(1e+6, 1e+2))
    test_point = np.arange(1,Input_points.shape[0],seeds_int)
    gp = GaussianProcessRegressor(kernel=kernal ,n_restarts_optimizer=12)
    t_input=Input_points[:,0][:,np.newaxis]
    y_input=Input_points[:,1]
    gp.fit(t_input,y_input)
    def print_cluster(model):
        plt.title('The DBSCAN based inverval identification', fontdict=None, loc='center')
        plt.xlabel('The nth point')
        plt.ylabel('Time(h)')
        plt.scatter(np.arange(model.components_.size),model.components_)
    
    def fun(x):
        return objfun(x,model=gp)

    min_points=np.array(list(map(lambda c: optimize.minimize(fun,c,method='BFGS').x,test_point)))
    
    cluster_model = sklearn.cluster.DBSCAN(eps=eps,min_samples=min_samples)
    cluster_model.fit(min_points)
    
    print_cluster(cluster_model)
    
    matrix = np.vstack((cluster_model.labels_,min_points[:,0])).T
    #The Label-min_points without the outlier
    matrix_clean=matrix[matrix[:,0]!=-1]
    group = np.unique(matrix_clean[:,0])
    #Calculate the mean of the minpoints
    mean_minpoints=np.array(list(map(lambda x : matrix_clean[matrix_clean[:,0]==x][:,1].mean(),group)))
    mean_minpoints.sort()
    mean_minpoints_include_zero = np.unique(np.append([0],mean_minpoints))
    
    fig = plt.figure(figsize=(12, 2.5))
    ax = fig.add_subplot(111)
    
    ax.plot(np.arange(0,Input_points.shape[0],0.1),gp.predict(np.arange(0,Input_points.shape[0],0.1)[:,np.newaxis]),c='black',label='RBF Interpolation')
    ax.scatter(Input_points[:,0],Input_points[:,1],c='blue',label='Input Data Point')
    ax.scatter(mean_minpoints_include_zero,gp.predict(mean_minpoints_include_zero[:,np.newaxis]),c='red',label='Detected Zero')
    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')
    plt.title('Po_multiDose-Oxycodo', fontdict=None, loc='center')
    plt.xlabel('Time(h)')
    plt.ylabel('Concentration(unit)')
    for i in mean_minpoints:
        plt.axvline(x=i, ymin=0, ymax=1,c='red')
# Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('C0')

    plt.show()
    return mean_minpoints_include_zero[mean_minpoints_include_zero>=0]

#Input=np.vstack((t_sparse,X[12400,t_indice])).T

#The function used for fitting the multiple dose
def split_fit(Input_points,PK_model=PK_model.PO_onedose):
    mean_minpoints_include_zero = return_intervals(Input_points)
    
    
    return NotImplementedError
    

