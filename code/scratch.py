# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 17:50:32 2018

@author: ych324
"""

#scratch


test_point = np.arange(1,48,0.3)
#x=pandas.read_csv("..\data\Po_multidose.csv")
X = np.array(x)
po_multi_dose = X[9200]
k = RBF(length_scale= 1, length_scale_bounds=(1e-1, 1e+2))
#k = Matern(length_scale= 1, length_scale_bounds=(1e-2, 1e+3), nu=5)
gp = GaussianProcessRegressor(kernel=k ,n_restarts_optimizer=12)
tt = np.arange(0,48,0.1)
tt_processed = tt[:,np.newaxis]
t_sparse = np.arange(0,48,1)
t_sparse_processed = t_sparse[:,np.newaxis]
t_indice=t_sparse*10

gp.fit(t_sparse_processed,po_multi_dose[t_indice])
Y=gp.predict(tt_processed)


#optimize.minimize(objfun,31,method='BFGS').x

min_points=np.array(list(map(lambda c: optimize.minimize(objfun_RBF,c,method='BFGS').x,test_point)))
zero_point = min_points
zero_point.reshape(-1,1)
plt.scatter(np.arange(zero_point.size),zero_point)

model = sklearn.cluster.DBSCAN(eps=0.001,min_samples=10)
model.fit(zero_point)
