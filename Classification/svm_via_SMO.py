import numpy as np
import pandas as pd

def standard_function(data):
    return (data - np.mean(data, axis = 0))/np.std(data, axis = 0)

def svm_f(X, y, alpha, b, idx):
    f = 0
    for i in range(alpha.shape[0]):
        f += alpha[i] * y[i] * np.dot(X[i,:], X[idx,:])
    f += b
    return f


def SMO_iter(X, y,alpha, b, C, tol):
    n = X.shape[0]
    num_changed = 0
    for i in range(n):
        E_i = svm_f(X, y, alpha, b, i) - y[i]
        if (y[i] * E_i < -tol and alpha[i] < C) or (y[i] * E_i > tol and alpha[i] > 0):
            j = np.random.choice([num for num in range(n) if num != i])
            E_j = svm_f(X, y, alpha, b, j) - y[j]
            a_i = alpha[i]
            a_j = alpha[j]
            if y[i] != y[j]:
                L = np.max([0, a_j - a_i])
                H = np.min([C, C + a_j - a_i])
            else:
                L = np.max([0, a_i + a_j - C])
                H = np.min([C, a_i + a_j])
            if L == H:
                continue

            eta = 2 * np.dot(X[i,:], X[j, :]) - np.dot(X[i, :], X[i, :]) - np.dot(X[j, :], X[j, :])
            if eta >= 0:
                continue
            alpha[j] = alpha[j] - y[j] * (E_i - E_j)/eta
            if alpha[j] > H:
                alpha[j] = H
            elif alpha[j] >= L:
                alpha[j] = alpha[j]
            else:
                alpha[j] = L
                
            if np.abs(alpha[j] - a_j) < 1e-5:
                continue
            alpha[i] = alpha[i] + y[i] * y[j] * (a_j - alpha[j])
            b1 =\
            b - E_i - \
            y[i] * (alpha[i] - a_i) * np.dot(X[i, :], X[i,:]) - y[j] * (alpha[j] - a_j) * np.dot(X[i, :], X[j, :])
            b2 =\
            b - E_j - \
            y[i] * (alpha[i] - a_i) * np.dot(X[i, :], X[j, :]) - y[j] * (alpha[j] - a_j) * np.dot(X[j, :], X[j, :])
            if 0 < alpha[i] < C:
                b = b1
            elif 0 < alpha[j] < C:
                b = b2
            else:
                b = (b1 + b2)/2
            num_changed += 1
    return alpha, b, num_changed


def svm(X, y):
    C = 1
    tol = 1e-4
    max_pass = 3
    passes = 0
    while passes < max_pass:
        alpha, b, num_changed = SMO_iter(X, y, alpha, b, C, tol)
        if num_changed == 0:
            passes += 1
        else:
            passes = 0
    return alpha, b