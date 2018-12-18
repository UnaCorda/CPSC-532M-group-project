# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 20:47:13 2018

@author: ych324
"""

import PK_model

from scipy import optimize
import numpy as np
import sklearn.cluster
import find_min

def fitting_OLS(Input_points,model_array):
    t_input=Input_points[:,0][:,np.newaxis]
    y_input=Input_points[:,1]
    
    def object_fun(input_array):
        ka = input_array[0]
        k = input_array[1]
        t = t_input
        obs = y_input
        #print(obs.size)
        """Weight Function"""
        max_value = np.ones(obs.size)
        #max_value[-6:-3] = 10
        #max_value[0:4] = 100       
        #est=PK_model.PO_onedose(ka=ka,k=k,t=t)+PK_model.PO_onedose(ka=ka,k=k,t=t+4)
        result_list = list(map(lambda x : x.predict(),model_array))
        #est = np.array(result_list)
        #print(np.sum(result_list,axis=0))
        est=np.sum(result_list,axis=0)+PK_model.PO_onedose(ka=ka,k=k,t=t-t_input[0])
        #print(t_input[0])
        #est=PK_model.PO_onedose(ka=ka,k=k,t=t)
        OLS=np.linalg.norm((obs-est)*max_value)
        #print(OLS)
        return OLS
        
    return optimize.minimize(object_fun,[0.5,1])
    

def fitting_OLS_splitpoints(Input):

    min_points = return_intervals(Input)
    model=[]
    ka_initial = 0.1
    k_inital = 1
    parameterlist = []
    for i in range(min_points.size-1):
    #for i in range(5):
        lower_bound = min_points[i]
        upper_bound = min_points[i+1]
        Partial_indices = np.intersect1d(np.where(Input[:,0]>=lower_bound)[0],np.where(Input[:,0]<=upper_bound)[0])
        Partial_points = Input[Partial_indices]
        t = Partial_points[:,0]
        model=[]
        for j in range(i):
        #for j in range(i):
            if i == 0:
                model.append(PK_model.PO_onecom_class(ka=ka_initial,k=k_inital,t=t,start_point=min_points[j]))
            else:
                model.append(PK_model.PO_onecom_class(ka=parameterlist[j][0],k=parameterlist[j][1],t=t,start_point=min_points[j]))
                
        parameter = fitting_OLS(Partial_points,model_array=model).x
        parameterlist.append(parameter)

    fig, ax = plt.subplots()
    ka = np.array(parameterlist)[:,0]
    k = np.array(parameterlist)[:,1]
    ax.plot(min_points[0:-1],ka,label="ka")
    ax.plot(min_points[0:-1],k,label="k")
    legend = ax.legend(loc='center right', shadow=True, fontsize='x-large')
    plt.title('Po_multiDose-Oxycodo-PK Coefficient', fontdict=None)
    plt.xlabel('Time(h)')
    plt.ylabel('h-1')
    return parameterlist
        


"""
Y=X[9200,45:90]
X_test = np.arange((X[9200,45:90]).size)

Y=Input[4:8,1]
X_test = Input[4:8,0]

Input_points = np.vstack((np.arange(X[14000].size)/10,X[14000])).T

X[19100,t_indice])
    """