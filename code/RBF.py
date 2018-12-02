# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 22:11:55 2018

@author: hasee
"""

kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
gp = GaussianProcessRegressor(kernel=kernel,
                              alpha=0.0).fit(X, y)

kernel = 1.0 * RBF()
gp = GaussianProcessRegressor(kernel=kernel,
                              alpha=0.0).fit(X, y)