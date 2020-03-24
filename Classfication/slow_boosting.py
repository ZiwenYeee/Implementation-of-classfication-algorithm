from joblib import Parallel, delayed
import gc
import numpy as np
import pandas as pd


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


class GBM_Tree(object):
    def __init__(self, kwargs):
        self.kwargs = kwargs
        params = {
                'lambda_T': 1,
                'gamma' : 0,
                'n_jobs' : 8,
                'min_split_gain': 0.01,
                'max_depth' : 3,
                'feature_fraction':1,
                'subsample':1,
                'seed':1
            }
        params.update(self.kwargs)
        self.root = None
        self.lambda_T = params['lambda_T']
        self.gamma = params['gamma']
        self.n_jobs = params['n_jobs']
        self.min_split_gain = params['min_split_gain']
        self.max_depth = params['max_depth']
        self.feature_fraction = params['feature_fraction']
        self.subsample = params['subsample']
        self.seed = params['seed']
    def calc_l2_split_gain(self, G, H, G_l, H_l, G_r, H_r):
        def cal_term(g, h):
            return g ** 2/ (h + self.lambda_T)
        return 1/2 * (cal_term(G_l, H_l) + cal_term(G_r, H_r) - cal_term(G, H) ) - self.gamma

    def calc_l2_leaf_score(self, grad, hess):
        return sum(grad)/ (sum(hess) + self.lambda_T)

    def find_feature_split_point(self, X_array_sort, grad_sort, hess_sort, col):
        assert len(grad_sort) !=0, "no more split."
        assert X_array_sort.shape == grad_sort.shape == hess_sort.shape,"wrong range."
        G_l = 0.
        H_l = 0.
        grad = grad_sort[:, col]
        hess = hess_sort[:, col]
        G = sum(grad)
        H = sum(hess)
        split_candidate = []
        for i in range(X_array_sort.shape[0]):
            G_l += grad[i]
            H_l += hess[i]
            G_r = G - G_l
            H_r = H - H_l
            split_gain = self.calc_l2_split_gain(G, H, G_l, H_l, G_r, H_r)
            split_candidate.append(split_gain)
        split_rank = split_candidate.index(max(split_candidate) )
        split_gain = max(split_candidate)
        split_value = X_array_sort[:, col][split_rank]
        split_left = range(split_rank + 1)
        split_right = range(split_rank + 1, X_array_sort.shape[0])
        return (split_value, split_gain, split_left, split_right, col)

    def find_best_split(self, X_array_sort, grad_sort, hess_sort):
        res = Parallel(n_jobs=self.n_jobs, verbose=0) \
            (delayed(self.find_feature_split_point)(X_array_sort, grad_sort, hess_sort, col)
             for col in range(X_array_sort.shape[1]) )
    #         value_list = []
        value_list = [res[i][1] for i in range(len(res))]
        best_split = value_list.index(max(value_list))
        gc.collect()
        return res[best_split]

    def build_tree(self, X_array_sort, grad_sort, hess_sort, depth, col_list):
        split_value, split_gain, split_left, split_right, feature_index = self.find_best_split(X_array_sort, grad_sort, hess_sort)
        score = self.calc_l2_leaf_score(grad_sort[:, feature_index] , hess_sort[:, feature_index])
        if split_gain <= self.min_split_gain:
            return GBM_Node(feature_index = col_list[feature_index], threshold = split_value,
                        score = score, weighted_n_node_samples = X_array_sort.shape[0])
        else:
            if depth > self.max_depth:
                return GBM_Node(feature_index = col_list[feature_index], threshold = split_value,
                        score = score, weighted_n_node_samples = X_array_sort.shape[0])
            else:
                left_branch = self.build_tree(X_array_sort[split_left, :], grad_sort[split_left, :], hess_sort[split_left, :],
                                              depth+1, col_list)
                right_branch = self.build_tree(X_array_sort[split_right, :], grad_sort[split_right, :], hess_sort[split_right, :],
                                              depth+1, col_list)
                return GBM_Node(children_left = left_branch, children_right = right_branch,
                            feature_index = col_list[feature_index], threshold = split_value, score = score,
                            weighted_n_node_samples = X_array_sort.shape[0])
    def fit(self, X_array_sort, grad_sort, hess_sort):
        X_array_sort, grad_sort, hess_sort, sub_cols = self.column_sampling(X_array_sort, grad_sort, hess_sort)
        X_array_sort, grad_sort, hess_sort = self.row_sampling(X_array_sort, grad_sort, hess_sort)
        self.sub_cols = sub_cols
        self.root = self.build_tree(X_array_sort, grad_sort, hess_sort, 1, sub_cols)
    def _predict(self, x):
            #  x = x[self.sub_cols]
        node = self.root
        while (node.children_left != None) or (node.children_right != None):
            if x[node.feature_index] < node.threshold:
                node = node.children_left
            else:
                node = node.children_right
        return node.score

    def predict(self, X):
        res = Parallel(n_jobs=self.n_jobs, verbose=0) \
            (delayed(self._predict)(x) for x in X)
        return np.array(res)
    def column_sampling(self, X_array_sort, grad_sort, hess_sort):
        np.random.seed(self.seed)
        col_nums = X_array_sort.shape[1]
        sub_nums = int(col_nums * self.feature_fraction)
        sub_cols = sorted(np.random.choice(range(col_nums), sub_nums,))
        return X_array_sort[:, sub_cols],grad_sort[:, sub_cols],hess_sort[:, sub_cols], sub_cols
    def row_sampling(self, X_array_sort, grad_sort, hess_sort):
        np.random.seed(self.seed)
        row_nums = X_array_sort.shape[0]
        sub_nums = int(row_nums * self.subsample)
        sub_rows = sorted(np.random.choice(range(row_nums), sub_nums))
        return X_array_sort[sub_rows, :], grad_sort[sub_rows, :], hess_sort[sub_rows, :]

                # loss_function = likelihood_loss
                # eval_function = logloss_eval

class Slow_boosting(object):
    def __init__(self, params):
        self.learning_rate = 0.1
        self.loss_function = None
        self.eval_function = None
        self.tree_params = {
                'lambda_T': 1,
                'gamma' : 0,
                'n_jobs' : 16,
                'min_split_gain': 0.0,
                'max_depth' : 5,
                'feature_fraction':0.7,
                'subsample':0.7,
                'seed':1}
    def calc_data_score(self, data, boosters):
        if len(boosters) == 0:
            return np.zeros((data.shape[0], ))
        else:
            res = Parallel(n_jobs=-1, verbose=0) \
                    (delayed(booster.predict)(data) for booster in boosters)
            ans = np.array(res).sum(axis = 0) * self.learning_rate
            return ans
        
    def sort_gradient(self, X_sort, X_array, grad, hess):
        X_array_sort = np.zeros(X_sort.shape)
        grad_sort = np.zeros(X_sort.shape)
        hess_sort = np.zeros(X_sort.shape)
        for i in range(X_sort.shape[1]):
            X_array_sort[:, i] = X_array[:,i][X_sort[:, i]]
            grad_sort[:, i] = grad[X_sort[:, i]]
            hess_sort[:, i] = hess[X_sort[:, i]]
        return X_array_sort, grad_sort, hess_sort
    
    def likelihood_loss(self, labels, preds):
        preds = 1./(1. + np.exp(-preds) )
        grad = -(preds - labels)
        hess = preds * (1 - preds)
        return grad, hess
    
    def logloss_eval(self, labels, preds):
        preds = 1./(1. + np.exp(-preds) )
        eps = 1e-15
        p = np.clip(preds, eps, 1-eps)
        logloss = np.mean(- labels * np.log(p) - (1 - labels) * np.log(1-p))
        return logloss
    
    def pre_sorted(self, X_train):
        return np.argsort(X_train)
    
    def train(self, X_train, y_train, X_valid = None, y_valid = None, num_boost_round = 50,
          early_stopping_round = 10, eval_rounds = 1):
        X_index = self.pre_sorted(X_train)
        boosters = []
        best_iteration = None
        best_validation_loss = np.inf
        print("Training until validation scores don't improve for {} rounds."
              .format(early_stopping_round))

        for iteration in range(num_boost_round):
            scores = self.calc_data_score(X_train, boosters)
            grad, hess = self.loss_function(y_train, scores)
            X_array_sort, grad_sort, hess_sort = self.sort_gradient(X_index, X_train, grad, hess)
            Tree = GBM_Tree(self.tree_params)
            Tree.fit(X_array_sort, grad_sort, hess_sort)
            boosters.append(Tree)
            train_score = self.calc_data_score(X_train, boosters)
            train_loss = self.eval_function(y_train, train_score)
            valid_loss_str = '-'
            if X_valid != None:
                valid_score = self.calc_data_score(X_valid, boosters)
                valid_loss = self.calc_data_score(y_valid, valid_score)
                valid_loss_str = '{:.6f}'.format(valid_loss)
            if iteration % eval_rounds == 0:
                print("[{}]    Train's loss: {:.6f}, Valid's loss: {}"
                      .format(iteration, train_loss, valid_loss_str))
        return boosters
