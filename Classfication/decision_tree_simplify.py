from sklearn.datasets import load_breast_cancer
from joblib import Parallel, delayed
import gc
import numpy as np
import pandas as pd

class Decision_TreeNode(object):
    def __init__(self, **kwargs):
        self.children_left = kwargs.get('children_left')
        self.children_right = kwargs.get('children_right')
        self.feature = kwargs.get('feature')
        self.feature_index = kwargs.get('feature_index')
        self.gini = kwargs.get('gini')
        self.label = kwargs.get('label')
        self.threshold = kwargs.get('threshold')
        
class Decision_Tree(object):
    def __init__(self, max_depth = 6, gain_tol = 0.0):
        self.root = None
        self.max_depth = max_depth
        self.gain_tol = gain_tol
    def gini_index(self, y_sort):
        def cal_term(y):
            p_1 = sum(y)/len(y)
            p_0 = (len(y) - sum(y))/(len(y))
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
    def find_split(self, X, y):
        col = -1
        gain = -np.inf
        threshold = -np.inf
        label = 1 if np.mean(y) > 0.5 else 0
        for i in range(X.shape[1]):
            y_sort = y[np.argsort(X[:, i])]
            gini_tmp, idx_tmp = self.gini_index(y_sort)
            if gini_tmp > gain:
                col = i
                gain = gini_tmp
                label = 1 if np.mean(y_sort) > 0.5 else 0
                threshold = X[:, i][np.argsort(X[:, i])][idx_tmp]
        return gain, col, label, threshold
    def build_tree(self, X, y, depth):
        best_gain, best_col, best_label, threshold = self.find_split(X, y)
        if (depth > max_depth) or (best_col == - 1) or (best_gain <= 0):
            return Decision_TreeNode(gini = best_gain, label = best_label)
        else:
            left = np.where(X[:, best_col] <= threshold)
            right = np.where(X[:, best_col] > threshold)
            left_branch = self.build_tree(X[left], y[left], depth + 1)
            right_branch = self.build_tree(X[right], y[right], depth + 1)
        return Decision_TreeNode(children_left = left_branch, children_right = right_branch,
                                feature_index = best_col, gini = best_gain,threshold = threshold, label = best_label)
    def fit(self, X, y):
        self.root = self.build_tree(X, y, 1)
    def _predict(self, x):
        node = self.root
        while (node.children_left != None) or (node.children_right != None):
            if x[node.feature_index] < node.threshold:
                node = node.children_left
            else:
                node = node.children_right
        return node.label

    def predict(self, X):
        res = Parallel(n_jobs = 8, verbose=0) \
                (delayed(self._predict)(x) for x in X)
        return np.array(res)
