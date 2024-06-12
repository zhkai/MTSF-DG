from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import TVGL as tvgl
import numpy as np
import torch
import pandas as pd
import pickle
import argparse

if __name__ == "__main__":
    df = pd.read_hdf('./data/metr-la.h5')
    df = df.values

    lamb = 3
    beta = 15
    lengthOfSlice = 400
    thetaSet = tvgl.TVGL(df, lengthOfSlice, lamb, beta, eps=2e-3, indexOfPenalty=1, verbose=False)
    
    with open('adjlist.pkl', 'wb') as f:
        pickle.dump(thetaSet, f, protocol=2)
    '''
    for i in thetaSet:
        s = (i != 0).sum(-1)
        print('s', s.sum())
        adj_mx = i[...]
        #adj_mx = np.linalg.inv(i)
        xx = np.diag_indices_from(adj_mx)
        dia = adj_mx[xx]

        ##id = adj_mx > 0
        ##adj_mx[id] = 0
        adj_mx = np.abs(adj_mx)
        d = np.array(adj_mx.sum(1))
        d_inv = np.power(d, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = np.diag(d_inv)
        random_walk_mx = d_mat_inv.dot(adj_mx)
        s = (random_walk_mx != 0).sum(-1)
        print('r', s.sum())
    '''
