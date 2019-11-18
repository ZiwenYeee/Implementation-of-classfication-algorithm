from sklearn.datasets import load_breast_cancer
from joblib import Parallel, delayed
import gc
import numpy as np
import pandas as pd


data = load_breast_cancer().data
target = load_breast_cancer().target

class Logistic_Regression(object):
    def __init__(self, eta = 0.005, tol = 10e-4):
        self.eta = eta
        self.beta = None
        self.intercept = 0
        self.tol = tol
    def process_helper(self, x):
        x = np.where(x > 1 - 10**-4, 1 - 10**-4, x)
        x = np.where(x < 10**-4, 10**-4, x)
        return x
    def standard_function(self, data):
        return (data - data.mean(axis = 0))/data.std(axis = 0)
    def gradient_descent(self, X, y, beta, intercept, parallel = False):
        linear = intercept + np.sum(beta * X, axis = 1)
        prob = 1/(1 + np.exp(-linear))
        new_beta = np.zeros(beta.shape)
        if parallel:
            def cal_grad(y, prob, x, beta, eta):
                return beta - eta * np.sum(-(y - prob) * x)
            new_beta = Parallel(n_jobs = -1, verbose = 0)\
            (delayed(cal_grad)(target, prob, X[:,j], beta[j], eta) for j in range(data_std.shape[1]))
            new_beta = np.array(new_beta)
        else:
            for j in range(X.shape[1]):
                grad = -np.sum((y - prob) * X[:, j])
                new_beta[j] = beta[j] - self.eta * grad
        new_intercept = intercept - self.eta * np.sum(-(y - prob))
        return new_beta, new_intercept
    def newton_descent(self, X, y, beta, intercept):
        linear = intercept + np.sum(beta * X, axis = 1)
        prob = 1/(1 + np.exp(-linear))
        new_beta = np.zeros(beta.shape)
        prob = self.process_helper(prob)
        for j in range(X.shape[1]):
            grad = -np.sum((y - prob) * X[:, j])
            hess = np.sum(prob * (1 - prob) * X[:, j] ** 2)
            new_beta[j] = beta[j] - self.eta * grad/hess
        grad_intercept = -np.sum(y - prob)
        hess_intercept = np.sum(prob * (1 - prob))
        new_intercept = intercept - self.eta * grad_intercept/hess_intercept
        return new_beta, new_intercept
    def IRLS(self, X, y, beta, intercept):
        new_beta = np.zeros(beta.shape)
        linear = intercept + np.sum(beta * X, axis = 1)
        prob = 1/(1 + np.exp(-linear))
        prob = self.process_helper(prob)
        w = prob * (1 - prob)
        for j in range(X.shape[1]):
            h = np.sum(w * X[:, j] ** 2) 
            z = (y - prob)/w + beta[j] * X[:, j]
            beta[j] = np.sum(X[:, j] * w * z)/h
        new_intercept = np.sum(w * (intercept + (y - prob)/w))/np.sum(w)
        return beta, new_intercept
    def fit(self, X, y, method = 'newton'):
        self.beta = np.zeros(X.shape[1])
        for i in range(1000):
            converge = 1
            beta_p, intercept_p = self.beta, self.intercept
            if method == 'newton':
                self.beta, self.intercept = self.newton_descent(data_std, target, self.beta, self.intercept)
            elif method == 'gradient':
                self.beta, self.intercept = self.gradient_descent(data_std, target, self.beta, self.intercept)
            elif method == 'IRLS':
                self.beta, self.intercept = self.IRLS(data_std, target, self.beta, self.intercept)
            else:
                raise Exception("only support ['gradient', 'newton', 'IRLS']")
            converge = np.sum((beta - beta_p) ** 2) + (intercept - intercept_p) ** 2
            if converge < self.tol:
                break
    def predict(self, X):
        linear = self.intercept + np.sum(self.beta * X, axis = 1)
        prob = 1/(1 + np.exp(-linear))
        return np.where(prob > 0.5, 1, 0)        