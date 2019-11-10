
from joblib import Parallel, delayed
import gc
import numpy as np
import pandas as pd


class Lasso_CR(object):
    def __init__(self, **kwargs):
        self.beta = 0
        self.intercept = 0
        self.lambda_ = 0
        self.max_iter = 500
        self.lambda_ = kwargs.get('lambda_')
        
    def standardlize(self, data):
        return (data - data.mean(axis = 0))/data.std(axis = 0)
    
    def soft_thresholding(self, beta_old, P, Q, threshold):
        if P * beta_old + Q > threshold:
            beta_new = beta_old + (Q - threshold)/P
        elif P * beta_old + Q < - threshold:
            beta_new = beta_old + (Q + threshold)/P
        else:
            beta_new = 0
        return beta_new
    def coordinate_descent(self, x, y, beta, intercept, lambda_):
        def process_helper(x):
            x = np.where(x > 1 - 10**-5, 1 - 10**-5, x)
            x = np.where(x < 10**-5, 10**-5, x)
            return x
        for j in range(len(beta)):
            linear = intercept + np.sum(beta * x, axis = 1)
            prob = 1/(1 + np.exp(-linear)) 
            prob = process_helper(prob)
            w = prob * (1 - prob)
            P = np.mean(w * (x[:, j] ** 2))
            Q = np.mean((y - prob) * x[:, j])
            beta[j] = self.soft_thresholding(beta[j], P, Q, lambda_)
    
        linear = intercept + np.sum(beta * x, axis = 1)
        prob = process_helper(1/(1 + np.exp(-linear)))
        w_ = prob * (1 - prob)
        dir_ = y - prob
        intercept_new = intercept + np.mean(dir_)/np.mean(w)
        return beta, intercept_new

    def fit(self, X, y):
        X_std = self.standardlize(X)
        max_iter = 500
        self.beta = np.ones(X.shape[1])/X_std.shape[1]
        self.intercept = 0
        for iteration in range(max_iter):
            self.beta, self.intercept = self.coordinate_descent(X_std, y, self.beta, self.intercept, self.lambda_)

