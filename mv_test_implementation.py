from joblib import Parallel, delayed
import gc
import numpy as np
import pandas as pd
from numba import jit


@jit(nopython = True)
def mv_cal_sorted(x:np.array, y:np.array):
    x_idx = np.argsort(x)
    y = y[x_idx]
    y_unique = np.unique(y)
    n = y.shape[0]
    mv_test = np.zeros((len(y_unique), ))
    y_array = np.zeros((len(y_unique), ))
    prior = np.zeros((len(y_unique), ))
    for c in y_unique:
        prior[c] = np.sum(y == c) #O(p*n)
    y_num = 0
    for row in range(n): #O(n)
        y_array[y[row]] += 1
        y_num += 1
        F_r = y_array/prior
        F = y_num/n
        mv_test += prior/n * (F_r - F) ** 2
    mv_test = np.sum(mv_test)
    return mv_test

def mv_test_sorted(data:np.array, target:np.array):
    m = data.shape[1]
    res = Parallel(n_jobs = -1, verbose=0) \
             (delayed(mv_cal_sorted)(data[:, col], target) 
              for col in range(m))
    return res

def mv_feature_selection(data: pd.DataFrame, target:str, features:list, alpha = 0.001, feat_imp = False):
    try:
        from scipy import stats
    except:
        Exception("please pip install scipy.")
    fe_used = np.array(data[features])
    target = np.array(data[target])
    feat_val = mv_test_sorted(fe_used, target)
    R = len(np.unique(target))
    chi_value = stats.chi2.ppf(1 - alpha, R - 1)
    asymptotic = np.sum([chi_value/(np.pi ** 2 * j**2) for j in range(1, data.shape[0] + 1)])
    pval = np.where(feat_val >= asymptotic, 1, 0)
    feat_new = []
    for col, indicator in zip(features, pval):
        if indicator == 1:
            feat_new.append(col)
    if feat_imp:
        imp = pd.DataFrame([], columns=['name', 'value'])
        c = 0
        for name, val in zip(features, feat_val):
            imp.loc[c, 'name'] = name
            imp.loc[c, 'value'] = val
            c += 1
        return imp
    return feat_new

