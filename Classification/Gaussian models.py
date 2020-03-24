import numpy as np
import pandas as pd

def Quadratic_Discrminant_Analysis(X, y, prior = False):
    dim = X.shape[1]
    class_num = len(np.unique(y))
    mean = np.zeros((class_num, dim))
    sigma = np.zeros((class_num, dim, dim))
    sigma_inv = np.zeros((class_num, dim, dim))
    det = np.zeros((class_num, ))
    for y_class in range(class_num):
        y_index = np.where(y == y_class)[0]
        X_mat = X[y_index, :]
        mean[y_class, :] = np.mean(X_mat, axis = 0)
        sigma[y_class, :] = np.cov(X_mat.T)
        sigma_inv[y_class, :] = np.linalg.inv(sigma[y_class, :])
        det[y_class, ] = np.linalg.det(sigma[y_class, :])
        
    n = X.shape[0]
    pred = np.zeros((n, class_num))
            
    for row in range(n):
        for i in range(class_num):
            numerator = np.exp(-1/2 * np.dot(np.dot((X[row, :] - mean[i, :]), 
                                                    sigma_inv[i, :]), 
                                                    (X[row, :] - mean[i, :]).T))
            denominator = np.sqrt((2*np.pi) ** D * det[i, ])
            pred[row, i] = numerator/denominator
    
    if prior:
        prior_mat = np.zeros((class_num, ))
        for i in range(class_num):
            prior_mat[i, ] = np.mean(np.where(y == i, 1, 0))
        pred = pred * prior_mat
        
    dist_sum = pred.sum(axis = 1)

    for i in range(class_num):
        pred[:, i] = pred[:, i]/dist_sum
    return pred

def Linear_Discrminant_Analysis(X, y):
    dim = X.shape[1]
    class_num = len(np.unique(y))
    mean = np.zeros((class_num, dim))
    sigma = np.cov(X.T)
    sigma_inv = np.linalg.inv(sigma)
    prior = np.zeros((class_num, ), dtype=np.float128)
    gamma = np.zeros((class_num, ), dtype=np.float128)
    beta = np.zeros((class_num, dim), dtype=np.float128)
    n = X.shape[0]
    for y_class in range(class_num):
        y_index = np.where(y == y_class)[0]
        X_mat = X[y_index, :]
        mean[y_class, :] = np.mean(X_mat, axis = 0)
        gamma[y_class] = -1/2 * np.dot(np.dot(mean[y_class, :], sigma_inv),mean[y_class, :].T)
        beta[y_class, :] = np.dot(sigma_inv, mean[y_class,:].T)
    pred = np.zeros((n, class_num),dtype=np.float128)
    for row in range(n):
        for i in range(class_num):
            pred[row, i] = np.exp(np.dot(beta[i,:], X[row,:]) + gamma[i])
    tmp = np.sum(pred, axis = 1)
    for i in range(class_num):
        pred[:, i] = pred[:, i]/tmp
    return pred
