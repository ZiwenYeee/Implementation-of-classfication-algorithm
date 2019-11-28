#regression
from sklearn.datasets import load_breast_cancer
from joblib import Parallel, delayed
import gc
import numpy as np
import pandas as pd

import time
from contextlib import contextmanager
from numba import jit



class svm_CD(object):
    def __init__(self, C = 1, opt = 'L1'):
        self.C = C
        self.opt = opt
        self.w = None
    def data_transform(self, X, y = None):
        if y is None:
            X = np.append(X, np.ones((X.shape[0],1)), axis = 1)
            return X
        else:
            y = 2 * y - 1
            X = np.append(X, np.ones((X.shape[0],1)), axis = 1)
            return X, y
    def fit(self, X, y, max_iters = 100):
        X, y = self.data_transform(X, y)
        self.w = dual_coordinate_descent(X = X, y = y, C = self.C, iters = max_iters, opt = self.opt)
    def predict(self, X):
        X = self.data_transform(X)
        pred = np.sum(self.w * X, axis = 1)
        pred = np.where(pred > 0, 1, 0)
        return pred


@jit(nopython = True)
def soft_helper(G, U, PG, alpha):
    if alpha == 0:
        PG = min(G, 0)
    elif alpha == U:
        PG = max(G, 0)
    elif (0 < alpha < U):
        PG = G
    else:
        raise ValueError('alpha value errors.')
    return PG

@jit(nopython = True)
def svm_loss(X, y, w, C, opt):
    max_term = 0
    if opt == 'L1':
        for i in range(X.shape[0]):
            max_term += C * max(1 - sum(y[i] * w * x[i, :]), 0)
    elif opt == 'L2':
        for i in range(X.shape[0]):
            max_term += C * max(1 - sum(y[i] * w * x[i, :]), 0) ** 2
    else:
        raise Exception("only support L1 and L2.")
    loss = 1/2 * sum(w**2) + max_term
    return loss

@jit(nopython = True)
def dual_coordinate_descent(X, y, C, iters, opt = 'L1'):
    n = X.shape[0]
    m = X.shape[1]  
    if opt == 'L1':
        U = C
        D = np.zeros((n, ))
    elif opt == 'L2':
        U = np.inf
        D = np.ones((n, )) * 1/(2 * C)
    else:
        raise Exception("only support L1 and L2.")
    alpha = np.zeros((n, ))
    w = np.zeros((m, ))
    PG = None
    tmp = None
    tol = 1
    for i in range(n):
        w += y[i] * alpha[i] * X[i, :]
    for iteration in range(iters):
        while tol > 10e-5:
            w_old = w
            for i in range(n):
                G = y[i] * np.sum(X[i, :] * w) - 1 + D[i] * alpha[i]
                PG = soft_helper(G, U, PG, alpha[i])
                if abs(PG) != 0:
                    alpha_old = alpha[i]
                    Q_ii = np.sum(X[i, :] **2) + D[i]
                    tmp = max(alpha[i] - G/Q_ii, 0)
                    alpha[i] = min(tmp, U)
                    w = w + (alpha[i] - alpha_old) * y[i] * X[i, :]
            tol = np.sum(np.abs(w - w_old))
    return w
