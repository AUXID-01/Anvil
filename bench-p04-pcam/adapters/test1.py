import numpy as np
from adapter import Adapter

class Engine(Adapter):
    def __init__(self, stored_patterns, model_params):
        self.X = stored_patterns
        self.N = stored_patterns.shape[1]
        self.model_params = model_params
        
        # precompute once — used by variance approach
        self.var = np.var(self.X, axis=0)
        
        # store R for hessian computation
        self.R = model_params['R']
        self.beta = model_params['beta']
        self.eta = model_params['eta']

    def _hessian(self, pattern):
        # compute hessian at a given pattern
        z = self.beta * (self.X @ pattern)
        z = z - z.max()
        e = np.exp(z)
        s = e / e.sum()
        D = np.diag(s) - np.outer(s, s)
        H = self.R - self.eta * self.beta * (self.X.T @ (D @ self.X))
        return H

    def predict_precision(self, corrupted_query):
        
        # --- Approach 1: magnitude ---
        signal = np.abs(corrupted_query)
        
        # --- Approach 2: variance ---
        var_weight = self.var / (self.var.mean() + 1e-8)
        
        # --- Approach 3: hessian ---
        cosines = self.X @ corrupted_query
        best_pattern = self.X[np.argmax(cosines)]
        H = self._hessian(best_pattern)
        H_diag = np.diag(H)
        hessian_precision = 1.0 / (np.abs(H_diag) + 1e-6)

        # --- Approach 4: hybrid (combine all) ---
        precision = signal * var_weight * hessian_precision

        # normalize and clip
        precision = precision / (precision.mean() + 1e-8)
        return np.clip(precision, 0.1, 10.0)

