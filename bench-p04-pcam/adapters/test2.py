from __future__ import annotations

from typing import Any

import numpy as np

from adapter import Adapter


class Engine(Adapter):
    def __init__(self,
                 stored_patterns: np.ndarray,
                 model_params: dict[str, Any]) -> None:
        self.X = stored_patterns
        self.K, self.N = stored_patterns.shape
        self.var_X = np.var(stored_patterns, axis=0) + 1e-8
        self.beta = model_params.get('beta', 8.0)
        self.R_inv = np.linalg.inv(model_params['R'])
        self.eta = model_params.get('eta', 0.5)

        self.precomputed_patterns = []
        from pcam_model import PCAMModel
        self.model = PCAMModel(
            stored_patterns,
            R=model_params['R'],
            eta=model_params['eta'],
            beta=model_params['beta'],
            dt=model_params['dt'],
            T_max=model_params['T_max'],
            tol=model_params['tol'],
            T_in=model_params['T_in'],
            pi_min=model_params['pi_min'],
            pi_max=model_params['pi_max'],
        )

        for k in range(self.K):
            a_star = self.eta * (self.R_inv @ stored_patterns[k])
            H = self.model.hessian(a_star)
            H = 0.5 * (H + H.T)
            eigvals, eigvecs = np.linalg.eigh(H)
            self.precomputed_patterns.append({
                'h_diag': np.diag(H),
                'eigvals': eigvals,
                'eigvecs': eigvecs,
            })

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q = corrupted_query
        nq = np.linalg.norm(q)
        q_hat = q / nq if nq > 1e-12 else q

        sims = self.X @ q_hat
        k1 = int(np.argmax(sims))
        sims_copy = sims.copy()
        sims_copy[k1] = -np.inf
        k2 = int(np.argmax(sims_copy))

        logits = self.beta * sims
        logits -= logits.max()
        exp_logits = np.exp(logits)
        softmax = exp_logits / exp_logits.sum()
        confidence = float(softmax[k1])
        ambiguity = 1.0 - confidence

        diff_sq = (self.X[k1] - self.X[k2]) ** 2
        pi_fisher = diff_sq / self.var_X
        pi_fisher = pi_fisher ** 1.5
        pi_fisher = pi_fisher / (np.mean(pi_fisher) + 1e-10)
        pi_fisher = 0.7 + 0.3 * pi_fisher

        alpha = min(1.0, ambiguity * 2.0)
        pi = (1 - alpha) * np.ones(self.N) + alpha * pi_fisher

        pi = pi / np.mean(pi)
        return pi

