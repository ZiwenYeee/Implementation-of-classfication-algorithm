#regression
from sklearn.datasets import load_breast_cancer
from joblib import Parallel, delayed
import gc
import numpy as np
import pandas as pd

import time
from contextlib import contextmanager
from numba import jit

class boosting_numba(object):
    def __init__(self, params):
        self.learning_rate = 0.1
        self.loss_function = likelihood_loss
        self.eval_function = logloss_eval
        self.boosters = []
        self.tree_params = {
                'lambda_T': 0.0,
                'gamma' : 0,
                'min_split_gain': 0.0,
                'max_depth' : 6,
                'feature_fraction':0.8,
                'subsample':0.7,
                'seed':1}
        self.tree_params.update(params)
        if 'loss_function' in params:
            self.loss_function = params['loss_function']
        if 'eval_function' in params:
            self.eval_function = params['eval_function']
        if 'learning_rate' in params:
            self.learning_rate = params['learning_rate']
    def calc_data_score(self, data, boosters, learning_rate):
        if len(boosters) == 0:
            return np.zeros((data.shape[0], ))
        else:
            res = Parallel(n_jobs=-1, verbose=0) \
                    (delayed(booster.predict)(data) for booster in boosters)
            res = np.array(res).sum(axis = 0) * learning_rate
            return res
    @jit
    def train(self, X_train, y_train, X_valid = None, y_valid = None, num_boost_round = 50,
          early_stopping_round = 10, eval_rounds = 1):
        boosters = []
        best_iteration = None
        best_validation_loss = np.inf
        print("Training until validation scores don't improve for {} rounds.".format(early_stopping_round))
        for iteration in range(num_boost_round):
            scores = self.calc_data_score(X_train, boosters, self.learning_rate)
            grad, hess = self.loss_function(y_train, scores)
            Tree = GBM_Tree_numba(self.tree_params)
            Tree.fit(X_train, grad, hess)
            boosters.append(Tree)
            train_score = self.calc_data_score(X_train, boosters, self.learning_rate)
            train_loss = self.eval_function(y_train, train_score)
            valid_loss_str = '-'
            if X_valid != None:
                valid_score = self.calc_data_score(X_valid, boosters, self.learning_rate)
                valid_loss = self.calc_data_score(y_valid, valid_score)
                valid_loss_str = '{:.6f}'.format(valid_loss)
            if iteration % eval_rounds == 0:
                print("[{}]    Train's loss: {:.6f}, Valid's loss: {}"
                      .format(iteration, train_loss, valid_loss_str))
        self.boosters = boosters
    @jit
    def predict(self, X):
        pred = self.calc_data_score(X, self.boosters, self.learning_rate)
        return pred


@jit(nopython = True)
def likelihood_loss(labels, preds):
    preds = 1./(1. + np.exp(-preds) )
    grad = -(preds - labels)
    hess = preds * (1 - preds)
    return grad, hess

@jit(nopython = True)
def logloss_eval(labels, preds):
    preds = 1./(1. + np.exp(-preds) )
    logloss = np.mean(- labels * np.log(preds) - (1 - labels) * np.log(1-preds))
    return logloss


class GBM_Node(object):
    def __init__(self, **kwargs):
        self.children_left = kwargs.get('children_left')
        self.children_right = kwargs.get('children_right')
        self.children_default = kwargs.get('children_default')
        self.feature = kwargs.get('feature')
        self.feature_index = kwargs.get('feature_index')
        self.threshold = kwargs.get('threshold')
        self.score = kwargs.get("score")
        self.node_sample_weight = kwargs.get("node_sample_weight")
        self.weighted_n_node_samples = kwargs.get("weighted_n_node_samples")
        self.gain = kwargs.get("gain")


@jit(nopython = True)
def calc_l2_split_gain(G, H, G_l, H_l, G_r, H_r, gamma, lambda_):
    def cal_term(g, h, lambda_):
        return np.power(g, 2)/ (h + lambda_)
    return 1/2 * (cal_term(G_l, H_l, lambda_) + cal_term(G_r, H_r, lambda_) - cal_term(G, H, lambda_) ) - gamma

@jit(nopython = True)
def calc_l2_leaf_score(grad, hess, lambda_):
    return np.sum(grad)/(np.sum(hess) + lambda_)

@jit(nopython = True)
def find_feature_split_point(X, X_sort, grad, hess, gamma, lambda_, col):
    grad = grad[X_sort]
    hess = hess[X_sort]
    X = X[X_sort]
    G_l = 0
    H_l = 0
    G = np.sum(grad)
    H = np.sum(hess)
    idx = -1
    gain = -np.inf
    val = -1
    for i in range(1, X.shape[0]):
        G_l += grad[i]
        H_l += hess[i]
        G_r = G - G_l
        H_r = H - H_l
        split_gain = calc_l2_split_gain(G, H, G_l, H_l, G_r, H_r, gamma, lambda_)
        if split_gain > gain:
            gain = split_gain
            val = X[i]
    return gain, val, col

@jit(nopython = True)
def pre_sorted(X_train):
    X_sort = np.zeros(X_train.shape)
    for i in range(X_train.shape[1]):
        X_sort[:, i] = np.argsort(X_train[:, i])
    return X_sort


def find_best_split(X, X_sort, grad_sort, hess_sort, gamma = 0, lambda_ = 0):
    res = Parallel(n_jobs = 12, verbose=0) \
            (delayed(find_feature_split_point)(X[:, col], X_sort[:, col], 
                                                    grad_sort, hess_sort, gamma, lambda_, col)
                                                     for col in range(X_sort.shape[1]) )
    value_list = [res[i][0] for i in range(len(res))]
    best_split = value_list.index(np.max(value_list))
    gc.collect()#gain, val, col
    return res[best_split]

@jit
def column_sampling(X, feature_fraction):
    col_nums = X.shape[1]
    sub_nums = int(col_nums * feature_fraction)
    sub_cols = np.random.choice(range(col_nums), sub_nums,)
    return X[:, sub_cols], sub_cols

@jit
def row_sampling(X, grad, hess, subsample):
    row_nums = X.shape[0]
    sub_nums = int(row_nums * subsample)
    sub_rows = np.random.choice(range(row_nums), sub_nums)
    return X[sub_rows], grad[sub_rows], hess[sub_rows]

class GBM_Tree_numba(object):
    def __init__(self, kwargs):
        params = {
                'lambda_': 0.1,
                'gamma' : 0,
                'n_jobs' : 12,
                'min_split_gain': 0.01,
                'max_depth' : 6,
                'feature_fraction':1,
                'subsample':1,
                'seed':1,
                'min_data_sample':2
            }
        params.update(kwargs)
        self.root = None
        self.min_split_gain = params['min_split_gain']
        self.lambda_ = params['lambda_']
        self.gamma = params['gamma']
        self.n_jobs = params['n_jobs']
        self.max_depth = params['max_depth']
        self.feature_fraction = params['feature_fraction']
        self.subsample = params['subsample']
        self.seed = params['seed']
        self.min_data_sample = params['min_data_sample']
        self.sub_cols = []
    @jit
    def build_tree(self, X, grad, hess, depth, col_list, gamma, lambda_):
        if (depth > self.max_depth or X.shape[0] < self.min_data_sample):
            score = calc_l2_leaf_score(grad, hess, lambda_)
            return GBM_Node(score = score)
        else:
            X_sort = pre_sorted(X).astype(int)
            gain, value, feat_idx = find_best_split(X, X_sort, grad, hess, gamma, lambda_)
            score = calc_l2_leaf_score(grad, hess, lambda_)
            if (gain <= self.min_split_gain):
                return GBM_Node(score = score, gain = gain)
            split_left = np.where(X[:, feat_idx] < value)
            split_right = np.where(X[:, feat_idx] >= value)
            left_branch = self.build_tree(X[split_left], grad[split_left], hess[split_left],
                                              depth+1, col_list, gamma, lambda_)
            right_branch = self.build_tree(X[split_right], grad[split_right], hess[split_right],
                                              depth+1, col_list, gamma, lambda_)
            return GBM_Node(children_left = left_branch, children_right = right_branch, gain = gain,
                            feature_index = col_list[feat_idx], threshold = value, score = score)
    def fit(self, X, grad, hess):
        X, sub_cols = column_sampling(X, self.feature_fraction)
        X, grad, hess = row_sampling(X, grad, hess, self.subsample)
        self.sub_cols = sub_cols
        self.root = self.build_tree(X, grad, hess, 1, sub_cols, self.gamma, self.lambda_)
    def _predict(self, x):
        node = self.root
        while (node.children_left != None) or (node.children_right != None):
            if x[node.feature_index] < node.threshold:
                node = node.children_left
            else:
                node = node.children_right
        return node.score
    def predict(self, X):
        res = np.zeros((X.shape[0], ))
        for i in range(X.shape[0]):
            res[i] = self._predict(X[i, :])
        return np.array(res) 


class Random_Forest_numba(object):
    def __init__(self, max_jobs = 8, max_tree = 100, max_depth = 6, 
                 min_gain = 0.0, subsample = 0.6, subcol = 0.8, seed = 0):
        self.tree_list = []
        self.max_depth = max_depth
        self.min_gain = min_gain
        self.subsample = subsample
        self.subcol = subcol
        self.seed = seed
        self.max_tree = max_tree
        self.max_jobs = max_jobs
    def column_sampling(self, X):
        col_nums = X.shape[1]
        sub_nums = int(col_nums * self.subcol)
        sub_cols = np.random.choice(range(col_nums), sub_nums)
        return X[:, sub_cols], sub_cols
    def row_sampling(self, X, y):
        row_nums = X.shape[0]
        sub_nums = int(row_nums * self.subsample)
        sub_rows = np.random.choice(range(row_nums), sub_nums)
        return X[sub_rows, :], y[sub_rows,]
    def tree_fit(self, X, y, seed):
        np.random.seed(self.seed + seed)
        X, sub_cols = self.column_sampling(X)
        X, y = self.row_sampling(X, y)
        Tree_ = Decision_Tree_numba(max_depth = self.max_depth, gain_tol = self.min_gain)
        Tree_.fit(X, y, col_list = sub_cols)
        return Tree_
    def fit(self , X, y):
        from tqdm import tqdm
        res = Parallel(n_jobs = self.max_jobs, verbose=0) \
                (delayed(self.tree_fit)(X, y, i) for i in tqdm(range(self.max_tree)))
        self.tree_list = res
    def predict(self, X):
        res = Parallel(n_jobs = self.max_jobs, verbose=0) \
                (delayed(tree.predict)(X) for tree in self.tree_list) 
        res = np.mean(res, axis = 0)
        res = np.where(res > 0.5, 1, 0)
        return res
    def predict_proba(self, X):
        res = Parallel(n_jobs = self.max_jobs, verbose=0) \
                (delayed(tree.predict)(X) for tree in self.tree_list) 
        res = np.mean(res, axis = 0)
        return res        

from numba import jit
@jit(nopython = True)
def find_split(X, y):
    col = -1
    gain = -np.inf
    threshold = -np.inf
    label = 1 if np.mean(y) > 0.5 else 0
    for i in range(X.shape[1]):
        y_sort = y[np.argsort(X[:, i])]
        gini_tmp, idx_tmp = gini_index(y_sort)
        if gini_tmp > gain:
            col = i
            gain = gini_tmp
            label = 1 if np.mean(y_sort) > 0.5 else 0
            threshold = X[:, i][np.argsort(X[:, i])][idx_tmp]
    return gain, col, label, threshold

@jit(nopython = True)
def gini_index(y_sort):
    def cal_term(y):
        p_1 = np.mean(y)
        p_0 = 1 - np.mean(y)
        gini = 1 - p_1 ** 2 - p_0 ** 2
        return gini
    delta = -np.inf
    pos = -1
    if len(y_sort) == 0:
        return delta, pos
    branch = cal_term(y_sort)
    N = len(y_sort)
    for i in range(1, N):
        left = cal_term(y_sort[:i]) * i/N
        right = cal_term(y_sort[i:]) * (N - i)/N
        moving_delta = branch - (left + right)
        if moving_delta > delta:
            delta = moving_delta
            pos = i
    return delta, pos

# from numba import jitclass
from numba import jit
@jit(nopython = True)
def find_split(X, y):
    col = -1
    gain = -np.inf
    threshold = -np.inf
    label = 1 if np.mean(y) > 0.5 else 0
    for i in range(X.shape[1]):
        y_sort = y[np.argsort(X[:, i])]
        gini_tmp, idx_tmp = gini_index(y_sort)
        if gini_tmp > gain:
            col = i
            gain = gini_tmp
            label = 1 if np.mean(y_sort) > 0.5 else 0
            threshold = X[:, i][np.argsort(X[:, i])][idx_tmp]
    return gain, col, label, threshold

@jit(nopython = True)
def gini_index(y_sort):
    def cal_term(y):
        p_1 = np.mean(y)
        p_0 = 1 - np.mean(y)
        gini = 1 - p_1 ** 2 - p_0 ** 2
        return gini
    delta = -np.inf
    pos = -1
    if len(y_sort) == 0:
        return delta, pos
    branch = cal_term(y_sort)
    N = len(y_sort)
    for i in range(1, N):
        left = cal_term(y_sort[:i]) * i/N
        right = cal_term(y_sort[i:]) * (N - i)/N
        moving_delta = branch - (left + right)
        if moving_delta > delta:
            delta = moving_delta
            pos = i
    return delta, pos

class Decision_TreeNode(object):
    def __init__(self, **kwargs):
        self.children_left = kwargs.get('children_left')
        self.children_right = kwargs.get('children_right')
        self.feature = kwargs.get('feature')
        self.feature_index = kwargs.get('feature_index')
        self.gini = kwargs.get('gini')
        self.label = kwargs.get('label')
        self.threshold = kwargs.get('threshold')
           
class Decision_Tree_numba(object):
    def __init__(self, max_depth = 6, gain_tol = 0.0):
        self.root = None
        self.max_depth = max_depth
        self.gain_tol = gain_tol
        self.min_data_sample = 5
    def build_tree(self, X, y, depth, col_list):
        best_label = 1 if np.mean(y) > 0.5 else 0
        if ((depth > self.max_depth) or (X.shape[0] < self.min_data_sample)):
#             return 
            return Decision_TreeNode(label = best_label)
        else:
            best_gain, best_col, best_label, threshold = find_split(X, y)
            if ((best_gain <= self.gain_tol)):
                return Decision_TreeNode(label = best_label)
                
            left = np.where(X[:, best_col] < threshold)
            right = np.where(X[:, best_col] >= threshold)
            left_branch = self.build_tree(X[left], y[left], depth + 1, col_list)
            right_branch = self.build_tree(X[right], y[right], depth + 1, col_list)
        return Decision_TreeNode(children_left = left_branch, children_right = right_branch,
                                feature_index = col_list[best_col], gini = best_gain,
                                  threshold = threshold, label = best_label)
    def fit(self, X, y, col_list = []):
        if col_list == []:
            col_list = list(range(X.shape[1]))
        self.root = self.build_tree(X, y, 1, col_list)
    def _predict(self, x):
        node = self.root
        while (node.children_left != None) or (node.children_right != None):
            if x[node.feature_index] < node.threshold:
                node = node.children_left
            else:
                node = node.children_right
        return node.label
    def predict(self, X):
        res = np.zeros((X.shape[0], ))
        for i in range(X.shape[0]):
            res[i] = self._predict(X[i, :])
        return np.array(res)    