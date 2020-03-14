import gc
import numpy as np
import pandas as pd

class KMeans():
    def __init__(self, max_round = 200, seed = 2):
        self.u = None
        self.K = None
        self.seed = seed
        self.epi = None
        self.max_epi = 1e-6
        self.max_round = max_round
    def fit(self, X, K):
        np.random.seed(self.seed)
        self.K = K
        n = X.shape[0]
        m = X.shape[1]
        iteration = 0
        self.u = np.random.normal(size = X.shape[1] * 2).reshape(K, m)
        z = np.zeros((n, K))
        label = np.zeros((n))
        epi = 10
        while iteration < self.max_round:
#             old_u = self.u
            for i in range(n):
                for j in range(K):
                    z[i, j] = np.sum((X[i, :] - self.u[j])**2)
            label = np.argmax(z, axis = 1)
        
            for j in range(K):
                self.u[j, :] = np.mean(X[np.where(label == j)[0],:], axis = 0)
            iteration += 1
#             epi = np.mean(abs(self.u - old_u))
        self.epi = epi
    def predict(self, X):
        n = X.shape[0]
        z = np.zeros((n, K))
        for i in range(n):
            for j in range(K):
                z[i, j] = np.sum((X[i, :] - self.u[j])**2)
        return np.argmax(z, axis = 1)


class Gaussian_Mixture_Model():
    def __init__(self, K):
        self.tol = 1e-3
        self.max_round = 100
        self.u = None
        self.sigma = None
        self.K = K
        self.delta_list = []
    def Gaussian_init(self, X):
        K = self.K
        n = X.shape[0]
        m = X.shape[1]
        u = np.zeros((K, m))
        sigma = np.zeros((K, m, m))
        idx = np.array_split(np.arange(n), K)
        for i in range(K):
            sampled = idx[i]
            u[i, :] = np.mean(X[sampled], axis = 0)
            sigma[i, :] = np.cov(X[sampled].T) 
        return u, sigma
    def Gaussian_NLL(self, alpha, pred):
        return np.sum(np.log(np.sum(alpha * pred)))
    def Gaussian_Model(self, x, u, sigma):
        D = x.shape[0]
        mat_det = np.linalg.det(sigma)
        mat_inv = np.linalg.inv(sigma)
        scale = np.sqrt((2 * np.pi) ** D * mat_det)
        dist = np.exp(-1/2 * np.linalg.multi_dot([(x - u).T, mat_inv, (x - u)]))
        pred = dist/scale
        return pred
    def E_Step(self, X, u, sigma):
        K = self.K
        n = X.shape[0]
        pred = np.zeros((n, K),dtype=np.float128)
        for i in range(n):
            for k in range(K):
                pred[i, k] = self.Gaussian_Model(X[i, :], u[k, :], sigma[k, :])
        label = np.argmax(pred, axis = 1)
        alpha = np.zeros((K))
        num = np.zeros((K))
        for i in range(K):
            alpha[i] = np.sum(label == i)/n
            num[i] = np.sum(label == i)
        w = np.zeros((n, K))
        for i in range(K):
            w[:, i] = pred[:, i] * alpha[i]/(np.sum(pred * alpha, axis = 1) + 1e-6)
        return w, num, pred
    def M_Step(self, X, w, num):
        K = self.K
        n = X.shape[0]
        m = X.shape[1]
        u = np.zeros((K, m))
        sigma = np.zeros((K, m, m))
        for i in range(K):
            u[i, :] = np.sum(np.repeat(w[:, i]/num[i], m).reshape(n, m) * X, axis = 0)
            sigma[i, :] = self.cov_cal(X, u[i, :], w[:, i], num[i])
        return u, sigma
    def cov_cal(self, X, u, w, num):
        n = X.shape[0]
        m = X.shape[1]
        sigma = np.zeros((m, m),dtype=np.float128)
        for i in range(n):
            sigma += np.dot((X[i, :] - u).reshape(1,30).T, (X[i, :] - u).reshape(1,30)) * w[i]/num
        return sigma
    def fit(self, X):
        n = X.shape[0]
        m = X.shape[1]
        u, sigma = self.Gaussian_init(X)
        w, num, pred = self.E_Step(X, u, sigma)
        u, sigma = self.M_Step(X, w, num)
        delta = 10
        iteration = 0
        NLL_old = self.Gaussian_NLL(w/n, pred)
        while delta > self.tol and iteration < self.max_round:
            w, num, pred = self.E_Step(X, u, sigma)
            u, sigma = self.M_Step(X, w, num)
            NLL = self.Gaussian_NLL(w/n, pred)
            delta = NLL - NLL_old
            NLL_old = NLL
            iteration += 1
        self.u = u
        self.sigma = sigma
        self.alpha = num/n
    def predict(self, X):
        n = X.shape[0]
        alpha = self.alpha
        K = self.K
        pred = np.zeros((n, K))
        for i in range(n):
            for k in range(K):
                pred[i, k] = self.Gaussian_Model(X[i, :], self.u[k, :], self.sigma[k, :])
                
        w = np.zeros((n, K))
        for i in range(K):
            w[:, i] = pred[:, i] * alpha[i]/(np.sum(pred * alpha, axis = 1) + 1e-6)
        return np.argmax(w, axis = 1)

class Principle_Compenent_Analysis(object):
    def __init__(self):
        self.var_ratio = 0.8
        self.w = None
        self.v = None
    def standard_function(self, data):
        return (data - np.mean(data, axis = 0))/np.std(data, axis = 0)
    def fit(self, X):
        X_std = self.standard_function(X)
        w, v = np.linalg.eig(np.cov(X_std.T))
        w_acc = 0
        w_sum = np.sum(w)
        for i in range(len(w)):
            w_acc += w[i]/w_sum
            if w_acc > self.var_ratio:
                break
        cols = [col for col in range(i)]
        self.w = w[cols]
        self.v = v[cols,:]
    def transform(self, X):
        X_transform = np.zeros((X.shape[0], self.v.shape[0]))
        X_std = self.standard_function(X)
        for i in range(len(self.w)):
            X_transform[:, i] = np.dot(X_std, self.v[i, :])
        return X_transform
    
