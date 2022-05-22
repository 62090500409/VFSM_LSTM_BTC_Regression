import numpy as np
import pandas as pd

from collections.abc import Sequence

class vikor_selector():
    
    def _init_decision_matrix(self):
        D = np.zeros((self.n_features, self.n_tagets), dtype=np.float32)
        X = self.X.T
        L = self.L.T
        for i in range(self.n_features):
            for j in range(self.n_tagets):
                D[i, j] = np.dot(X[i], L[j])/(np.linalg.norm(X[i])*np.linalg.norm(L[j]))
        return D
    
    def _normalize_decision_matrix(self):
        if getattr(self, 'decision_matrix', None) is None:
            self.decision_matrix = self._init_decision_matrix() 
        sum_sq = np.sum(np.sqrt(np.power(self.decision_matrix, 2)), axis=0)
        return self.decision_matrix / sum_sq
    
    def __init__(self, n_selected: int):
        assert n_selected > 0, f"number of selected must more than 0"
        self.n_selected = n_selected
        
    def set_n_selected(self, n_selected: int):
        assert n_selected > 0, f"number of selected must more than 0"
        self.n_selected = n_selected
        
    def fit(self, x: pd.DataFrame, y: pd.DataFrame, weight: Sequence=None):
        assert x.ndim == 2, f"dimension of x expected 2, got { x.ndim }"
        assert y.ndim == 2, f"dimension of x expected 2, got { y.ndim }"
        assert x.shape[0] == y.shape[0], f"number of x and y must equal, got { x.shape[0] } and { y.shape[0] }"
        assert x.shape[0] > 0, f"got an empty x"
        assert y.shape[0] > 0, f"got an empty y"
        assert self.n_selected <= x.shape[0], f"number of feature must more than number of selected, expect >= {self.n_features} got {x.shape[0]}"
        
        self.x_df = x
        self.y_df = y
        self.n_features = len(x.columns)
        self.n_tagets = len(y.columns)
        self.n_data = len(x)
        self.X = np.asarray([ x.T[i] for i in range(len(x)) ])
        self.L = np.asarray([ y.T[i] for i in range(len(y)) ])
        
        self.features_indices = { name: i for i, name in enumerate(x.columns) }
        self.targets_indices = { name: i for i, name in enumerate(y.columns) }
        
        self.rank_features = { f: 0 for _, f in enumerate(self.features_indices) }
        self.sorted_features = []
        
        if weight is not None:
            assert len(weight) == self.n_tagets, f"length weight must equal to number of targets, expect { self.n_tagets } got { len(weight) }"
            self.weight = weight
        else:
            self.weight = np.full(self.n_tagets, (1/self.n_tagets), dtype=np.float32)
        
        self.decision_matrix = self._init_decision_matrix()
        self.norm_decision_matrix = self._normalize_decision_matrix()
        
    def transform(self, max_utility_group = 0.5):
        assert max_utility_group <= 1 and max_utility_group >= 0, "maximum utility group must in [0, 1]"
        
        self.best_norm = np.max(self.norm_decision_matrix, axis=0)
        self.worst_norm = np.min(self.norm_decision_matrix, axis=0)
        
        self.S = self.weight * (self.best_norm - self.norm_decision_matrix) / (self.best_norm - self.worst_norm)
        
        self.utility = np.sum(self.S, axis=1)
        self.regret = np.max(self.S, axis=1)
        
        self.max_utility, self.min_utility = max(self.utility), min(self.utility)
        self.max_regret, self.min_regret = max(self.regret), min(self.regret)
        self.max_utility_group = max_utility_group
        
        self.vikor_index = \
            (self.max_utility_group)*(self.utility - self.min_utility)/(self.max_utility - self.min_utility) \
                + (1 - self.max_utility_group)*(self.regret - self.max_regret)/(self.max_regret - self.min_regret)
                
        for i, f in enumerate(self.rank_features):
            self.rank_features[f] = self.vikor_index[i]
            
        self.sorted_features = sorted(self.rank_features.items(), key= lambda x: x[1])
        return self.sorted_features[:self.n_selected]
    
    def fit_tranform(self, x: pd.DataFrame, y: pd.DataFrame, weight: Sequence=None, max_utility_group = 0.5):
        self.fit(x, y, weight)
        return self.transform(max_utility_group)