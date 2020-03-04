import numpy as np
import pandas as pd

class Linear_Regression():
    def __init__(self):
        self.w = None
        self.b = None
    def fit(self, X, y, L2 = 1, method = 'OLS'):
        if method == 'OLS':
            self.w = np.linalg.multi_dot([np.linalg.inv(np.dot(X.T, X)),X.T, y])
            self.b = np.mean(y - np.dot(X, self.w))
        elif method == 'Ridge':
            I = np.identity(X.shape[1])
            self.w = np.linalg.multi_dot([np.linalg.inv(np.dot(X.T, X) + + L2 * I),X.T, y])
            self.w = np.mean(y - np.dot(X, self.w))
        else:
            raise ValueError("only support ['OLS, 'Ridge'].")
    def predict(self, X):
        return np.dot(self.w, X) + self.b
