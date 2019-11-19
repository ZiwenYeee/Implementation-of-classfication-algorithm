from sklearn.datasets import load_breast_cancer
from joblib import Parallel, delayed
import gc
import numpy as np
import pandas as pd

class Decision_TreeNode(object):
    def __init__(self, **kwargs):
        self.children_left = kwargs.get('children_left')
        self.children_right = kwargs.get('children_right')
        self.children_default = kwargs.get('children_default')
        self.feature = kwargs.get('feature')
        self.feature_index = kwargs.get('feature_index')
        self.node_sample_weight = kwargs.get("node_sample_weight")
        self.gini = kwargs.get('gini')
        self.feature_value = kwargs.get('feature_value')
        
        
class Decision_Tree(object):
    def __init__(self):
        self.root = None
        self.max_depth = 7
    def build_tree(self, X_sort, y_array, depth):
        best_col, best_gain, best_index = self.find_split(y_array)
        if best_gain <= 0:
            return Decision_TreeNode(gini = best_gain)
        else:
            if (depth > self.max_depth) or (best_index == - 1):
                return Decision_TreeNode(gini = best_gain)
            else:
                left_branch = self.build_tree(X_sort[:best_index + 1, :], y_array[:best_index + 1, :], depth + 1)
                right_branch = self.build_tree(X_sort[best_index + 1:, :], y_array[best_index + 1:, :], depth + 1)
                return Decision_TreeNode(children_left = left_branch, children_right = right_branch,
                                        feature_index = best_col, gini = best_gain, 
                                         feature_value = X_sort[best_index, best_col])
    def fit(self, X, y):
        X_index = np.zeros(X.shape)
        y_array = np.zeros(X.shape)
        X_sort = np.zeros(X.shape)
        for i in range(X.shape[1]):
            X_index[:, i] = np.argsort(X[:, i])
        X_index = X_index.astype(int)
        for i in range(X.shape[1]):
            X_sort[:, i] = X[:, i][X_index[:, i]]
            y_array[:, i] = y[X_index[:, i]]
        self.root = self.build_tree(X_sort, y_array, 1)
        
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
    
    def find_split(self, y_array):
        best_col = -1
        best_gain = -np.inf
        best_index = -1
        for i in range(y_array.shape[1]):
            gini_tmp, idx_tmp = self.gini_index(y_array[:, i])
            if gini_tmp > best_gain:
                best_col = i
                best_index = idx_tmp
                best_gain = gini_tmp
        return best_col, best_gain, best_index
    