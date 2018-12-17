from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
import pandas
import os
import pickle
import gzip
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import numpy as np

def sparsedata(X,y):
    # k = RBF(length_scale= 1, length_scale_bounds=(1e-6, 1e+1))
    k = Matern(length_scale= 1, length_scale_bounds=(1e-2, 1e+3), nu=10)
    gp = GaussianProcessRegressor(kernel=k ,n_restarts_optimizer=12)
    gp.fit(X,y)

    #tt_processed = tt[:,np.newaxis]
    t_sparse = np.arange(0,48,6)    
    t_sparse_processed = t_sparse[:,np.newaxis]

    y_ = gp.perdict(t_sparse_processed)
    return y_

#load DATA


