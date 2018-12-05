from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF

import numpy as np

def sparsedata(X,y):
    # k = RBF(length_scale= 1, length_scale_bounds=(1e-6, 1e+1))
    k = Matern(length_scale= 1, length_scale_bounds=(1e-2, 1e+3), nu=2.5)
    gp = GaussianProcessRegressor(kernel=k ,n_restarts_optimizer=12)
    gp.fit(X,y)

    #tt_processed = tt[:,np.newaxis]
    t_sparse = np.arange(0,48,6)
    t_sparse_processed = t_sparse[:,np.newaxis]

    y_ = gp.perdict(t_sparse_processed)
    return y_
