import PK_model

#from scipy import optimize
import numpy as np
#import sklearn.cluster
import find_min
import KNN

t= np.arange(0,48,1)
Y = np.genfromtxt("../data/IVM_t.csv", delimiter=',')
Y_variance = Y*(1+np.random.normal(size=t.shape[0])/50)
Input = np.vstack((t,Y_variance))

model= KNN.KNN(n_class = 3)
model.fit()
result_list,out1,para = model.predict(Input)
print(result_list)
print(out1)
print(para)
