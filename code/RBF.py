# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 22:11:55 2018

@author: hasee
"""

import numpy as np

#######RBF scratch####################

kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
gp = GaussianProcessRegressor(kernel=kernel,
                              alpha=0.0)

kernel = 1.0 * RBF()
gp = GaussianProcessRegressor(kernel=kernel,
                              alpha=0.0).fit(X, y)

kernel=4.4**2 * RBF(length_scale=41.8)+ 3.27**2 * RBF(length_scale=180)


kernel = 1.0 * RBF(length_scale=48, length_scale_bounds=(1e-1, 1e3))

#######################################

#Generating the sparse point
tt = np.arange(0,48,0.1)

tt_processed = tt[:,np.newaxis]

t_sparse = np.arange(0,48,6)
t_sparse_processed = t_sparse[:,np.newaxis]

gp = GaussianProcessRegressor(kernel=kernel)
gp.fit(t_sparse_processed,X1[451,t_sparse_processed*10])
Y=gp.predict(tt_processed)

plt.plot(tt,Y);plt.plot(t_sparse,X1[451,t_sparse*10]);plt.plot(tt,X1[451])
plt.plot(tt,Y);plt.scatter(t_sparse,X1[451,t_sparse*10]);plt.plot(tt,X1[451])


