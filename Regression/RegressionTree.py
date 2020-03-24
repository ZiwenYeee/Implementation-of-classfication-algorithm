import pandas as pd
import numpy as np

class RegressionTree(object):
    def __init__(self):
        self.Tree = None
    def rmse_loss(self, y):
        val = np.mean(y)
        loss = np.mean((y - val) ** 2) 
        return loss
    def split_cal(self, y_sort):
        n = len(y_sort)
        max_delta = 0
        bst_idx = None
        if n == 1:
            return (max_delta, bst_idx)
        for i in range(1, n):
            loss_l = self.rmse_loss(y_sort[:i])
            loss_r = self.rmse_loss(y_sort[i:])
            delta = loss - (i/n * loss_l + (n - i)/n * loss_r)
            if delta > max_delta:
                max_delta = delta
                bst_idx = i
        return (max_delta, bst_idx)
    def split(self, X, y):
        bst_rec = []
        for i in range(X.shape[1]):
            idx = np.argsort(X[:, i])
            y_sort = y[idx]
            rec = self.split_cal(y_sort)
            bst_rec.append(rec)
    
        bst_delta = 0
        bst_idx = None
        bst_col = None
        bst_val = None
        for i in range(X.shape[1]):
            if bst_rec[i][0] > bst_delta:
                bst_delta = bst_rec[i][0]
                bst_idx = bst_rec[i][1]
                bst_col = i
                bst_val = X[bst_idx, i]
        
        return (bst_val, bst_col)
    
    def buildTree(self, X, y, depth):
        if depth > 8 or len(y) < 5:
            return None
        split_node = split(X, y)
        node = TreeNode()
        node.threshold = split_node[0]
        node.col = split_node[1]
        node.num = len(y)
        node.val = np.mean(y)
        idx_l = np.where(X[:, node.col] <= node.threshold)[0]
        idx_r = np.where(X[:, node.col] > node.threshold)[0]
    
        node.left = self.buildTree(X[idx_l, :], y[idx_l], depth + 1)
        node.right = self.buildTree(X[idx_r, :], y[idx_r], depth + 1)
        return node
    def fit(self, X, y):
        self.Tree = self.buildTree(X, y, 0)
    def predict_(self, node, x):
        while (node.left != None) or (node.right != None):
            if x[node.col] < node.threshold:
                if node.left:
                    node = node.left
                else:
                    return node.val
            else:
                if node.right:
                    node = node.right
                else:
                    return node.val
        return node.val

    def predict(self, X):
        res = np.zeros((X.shape[0],))
        for i in range(X.shape[0]):
            res[i] = self.predict_(self.Tree, X[i, :])
        return res
