import numpy as np
import pandas as pd

class Linear_Regression():
    def __init__(self):
        self.w = None
        self.b = None
    def soft_thresholding(self, beta, L1):
        if beta > L1:
            return beta - L1
        elif -L1 <= beta <= L1:
            return 0
        else:
            return beta + L1
    def Proximal_Gradient_Descent(self, X, y, lambda_l1):
        beta = np.zeros(X.shape[1] + 1)
        X = np.c_[np.ones(X.shape[0]), X]
        t = 0.0001
        vfunc = np.vectorize(self.soft_thresholding)
        Likelihood = np.mean((y - np.dot(X, beta)) ** 2)
        iteration = 0
        delta = 10
        while iteration < 500 and abs(delta) > 0.001:
            L_tmp = Likelihood
            grad = -np.dot(X.T, y - np.dot(X, beta))
            beta = beta - t * grad #step 1
            beta = vfunc(beta, lambda_l1 * t) # step 2
            
            Likelihood = np.mean((y - np.dot(X, beta)) ** 2)
            delta = Likelihood - L_tmp
            iteration += 1
        return beta
    
    def fit(self, X, y, L1 = 1, L2 = 1, method = 'OLS'):
        if method == 'OLS':
            self.w = np.linalg.multi_dot([np.linalg.inv(np.dot(X.T, X)),X.T, y])
            self.b = np.mean(y - np.dot(X, self.w))
        elif method == 'Ridge':
            I = np.identity(X.shape[1])
            self.w = np.linalg.multi_dot([np.linalg.inv(np.dot(X.T, X) + + L2 * I),X.T, y])
            self.b = np.mean(y - np.dot(X, self.w))
        elif method =='Lasso':
            w = self.Proximal_Gradient_Descent(X, y, L1)
            self.b = w[0]
            self.w = w[1:]
        else:
            raise ValueError("only support ['OLS, 'Ridge', 'Lasso'].")
    def predict(self, X):
        return np.dot(self.w, X) + self.b
    