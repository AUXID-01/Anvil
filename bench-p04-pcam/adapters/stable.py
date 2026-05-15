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

        # Decoupled stabilization modes
        self.geometry_mode = "global_slow_spectral"

        self.precomputed_patterns = []
        
        # Final sparsity refinement parameters
        self.k_sparse = 8
        self.suppression_floor = 0.20
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

        H_sum = np.zeros((self.N, self.N))
        for k in range(self.K):
            a_star = self.eta * (self.R_inv @ stored_patterns[k])
            H = self.model.hessian(a_star)
            H = 0.5 * (H + H.T)
            H_sum += H
            # eigh returns eigenvalues in ascending order
            eigvals, eigvecs = np.linalg.eigh(H)
            
            # Spectral inverse-diagonal approximation: diag(H^-1)
            eps_eig = 1e-6
            spectral_geom = np.sum((eigvecs ** 2) / (eigvals + eps_eig), axis=1)
            
            # SLOW-MODE SPECTRAL PRECONDITIONING
            # Focus only on the bottom K slowest modes to steer trajectories
            bottom_k = 8
            p_slow = 1.0
            slow_geom = np.sum((eigvecs[:, :bottom_k] ** 2) / (eigvals[:bottom_k] + eps_eig)**p_slow, axis=1)
            
            # Stabilization for spectral signals
            spectral_geom = np.maximum(spectral_geom, 1e-8)
            spectral_geom /= (spectral_geom.mean() + 1e-10)
            spectral_geom = spectral_geom ** 0.5
            
            slow_geom = np.maximum(slow_geom, 1e-8)
            slow_geom /= (slow_geom.mean() + 1e-10)
            slow_geom = slow_geom ** 0.5  # mild compression
            
            self.precomputed_patterns.append({
                'h_diag': np.diag(H),
                'eigvals': eigvals,
                'eigvecs': eigvecs,
                'spectral_geom': spectral_geom,
                'slow_geom': slow_geom,
            })
            
        # Global Slow Spectral Template
        H_mean = H_sum / self.K
        eigvals_g, eigvecs_g = np.linalg.eigh(H_mean)
        
        bottom_k_g = 8
        global_geom = np.sum((eigvecs_g[:, :bottom_k_g] ** 2) / (eigvals_g[:bottom_k_g] + 1e-6), axis=1)
        
        global_geom = np.maximum(global_geom, 1e-8)
        global_geom /= (global_geom.mean() + 1e-10)
        self.global_geom = global_geom ** 0.5

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        # 1. Retrieval Branch: Probabilistic Attractor Uncertainty
        q = corrupted_query
        nq = np.linalg.norm(q)
        q_hat = q / nq if nq > 1e-12 else q

        # 1. Retrieval Branch: Localized Contrastive Fisher
        sims = self.X @ q_hat
        top_indices = np.argsort(sims)[::-1]
        k1 = int(top_indices[0])
        
        # Local competitor selection (top M excluding k1)
        top_m = 4
        competitors = top_indices[1 : top_m + 1]
        selected_indices = top_indices[: top_m + 1]
        
        # Localized contrastive weights (softmax over neighborhood)
        local_logits = self.beta * sims[selected_indices]
        local_logits -= local_logits.max()
        local_exp = np.exp(local_logits)
        local_weights = local_exp / local_exp.sum()
        
        confidence = float(local_weights[0])
        ambiguity = 1.0 - confidence
        
        # Normalize competitor weights only (excluding k1) to distribute discrimination
        comp_weights = local_weights[1:]
        comp_weights /= (comp_weights.sum() + 1e-10)
        
        # Localized contrastive Fisher: Weighted sum of squared differences from top competitor
        diff_sq = np.zeros(self.N)
        for i, kj in enumerate(competitors):
            diff_sq += comp_weights[i] * (self.X[k1] - self.X[kj])**2
            
        # Convert into precision using the Fisher construction
        pi_fisher = diff_sq / self.var_X
        
        # Ambiguity-adaptive Fisher sharpening
        # Confident queries benefit from aggressive sharpening (power ~2.0)
        # Ambiguous queries use smoother precision (power ~1.0)
        gamma_sharp = 1.0
        fisher_power = 1.0 + gamma_sharp * confidence
        pi_fisher = pi_fisher ** fisher_power
        
        pi_fisher /= (np.mean(pi_fisher) + 1e-10)
        
        # Top-K Fisher Sparsification: concentrate steering on decisive coordinates
        topk_indices = np.argsort(pi_fisher)[-self.k_sparse:]
        mask = np.zeros(self.N)
        mask[topk_indices] = 1.0
        
        pi_sparse = self.suppression_floor + (1.0 - self.suppression_floor) * mask * pi_fisher
        pi_sparse /= (pi_sparse.mean() + 1e-10)
        pi_fisher = pi_sparse
        
        pi_fisher = 0.7 + 0.3 * pi_fisher
        
        # Ambiguity-aware modulation
        alpha = min(1.0, ambiguity * 2.0)
        pi_base = (1 - alpha) * np.ones(self.N) + alpha * pi_fisher

        # 2. Geometry Branch: Global Slow Spectral steering
        if self.geometry_mode == "global_slow_spectral":
            geom = self.global_geom
        elif self.geometry_mode == "slow_spectral":
            geom = self.precomputed_patterns[k1]["slow_geom"]
        else:
            geom = np.ones(self.N)

        # 3. Apply geometry multiplicatively with Critical Damping (Global)
        geom = geom / (np.mean(geom) + 1e-10)
        epsilon_geom = 0.02
        geom = 1.0 + epsilon_geom * (geom - 1.0)
        
        pi = pi_base * geom
        
        # Final normalization as required by benchmark
        pi = pi / (np.mean(pi) + 1e-10)
        
        return pi
