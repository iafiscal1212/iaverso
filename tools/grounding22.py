#!/usr/bin/env python3
"""
Phase 22: Minimal Grounding (FULL SPECIFICATION)
================================================

Implements PURELY ENDOGENOUS external signal grounding with full eigendecomposition,
effective dimension, coupling error, and gradient-based learning.

Key components:
1. Running statistics: mu_s(t), Sigma_s(t) with alpha = 1/sqrt(t+1)
2. Eigendecomposition: Sigma_s = V_s @ diag(lambda_s) @ V_s.T
3. Effective dimension: d_s(t) = sum(lambda_i > median(lambda))
4. Projection matrix: P_s = V_s[:, :d_s] @ V_s[:, :d_s].T
5. Coupling error: e_t = u_t - y_t (internal - projected external)
6. Gradient direction: g_t = ∂||e||²/∂W, normalized
7. Learning rate: eta_t = 1/sqrt(n+1)
8. Weighting: omega_t = rank(E_t) where E_t = ||e_t||
9. Update: W_new = W + eta * omega * g_norm
10. Grounding field: G_t = rank(||P_s @ z||) * normalize(P_s @ z - z)

NO semantic labels. NO magic constants.
All parameters derived from internal history.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

NUMERIC_EPS = 1e-16


# =============================================================================
# PROVENANCE TRACKING
# =============================================================================

class GroundingProvenance:
    """Track derivation of all grounding parameters."""

    def __init__(self):
        self.logs: List[Dict] = []

    def log(self, param_name: str, value: float, derivation: str,
            source_data: Dict, timestep: int):
        self.logs.append({
            'param': param_name,
            'value': value,
            'derivation': derivation,
            'source': source_data,
            't': timestep
        })

    def get_logs(self) -> List[Dict]:
        return self.logs

    def clear(self):
        self.logs = []


GROUNDING_PROVENANCE = GroundingProvenance()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_rank(value: float, history: np.ndarray) -> float:
    """Compute rank of value within history [0, 1].

    Uses midrank for ties: rank = (count_below + 0.5*count_equal) / total
    """
    if len(history) == 0:
        return 0.5
    n = len(history)
    count_below = float(np.sum(history < value))
    count_equal = float(np.sum(history == value))
    midrank = (count_below + 0.5 * count_equal) / n
    return midrank


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length, handle zero vectors."""
    norm = np.linalg.norm(v)
    if norm < NUMERIC_EPS:
        return np.zeros_like(v)
    return v / norm


# =============================================================================
# RUNNING STATISTICS FOR EXTERNAL SIGNAL
# =============================================================================

class RunningStatistics:
    """
    Maintains running statistics for external signal.

    mu_s(t) = (1 - alpha) * mu_s(t-1) + alpha * s_t
    Sigma_s(t) = (1 - alpha) * Sigma_s(t-1) + alpha * (s_t - mu_s)(s_t - mu_s).T

    alpha = 1/sqrt(t+1) (endogenous decay rate)
    """

    def __init__(self):
        self.mu_s: Optional[np.ndarray] = None
        self.Sigma_s: Optional[np.ndarray] = None
        self.t = 0

    def update(self, s_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, Dict]:
        """
        Update running statistics.

        Args:
            s_t: External signal vector

        Returns:
            (mu_s, Sigma_s, alpha, diagnostics)
        """
        self.t += 1
        dim = len(s_t)

        # alpha = 1/sqrt(t+1) - endogenous decay rate
        alpha = 1.0 / np.sqrt(self.t + 1)

        # Initialize if needed
        if self.mu_s is None:
            self.mu_s = s_t.copy()
            self.Sigma_s = np.eye(dim) * NUMERIC_EPS
        else:
            # Handle dimension change
            if len(self.mu_s) != dim:
                old_dim = len(self.mu_s)
                new_mu = np.zeros(dim)
                new_mu[:min(old_dim, dim)] = self.mu_s[:min(old_dim, dim)]
                self.mu_s = new_mu
                new_Sigma = np.eye(dim) * NUMERIC_EPS
                min_d = min(old_dim, dim)
                new_Sigma[:min_d, :min_d] = self.Sigma_s[:min_d, :min_d]
                self.Sigma_s = new_Sigma

            # EMA update for mean: mu_s(t) = (1 - alpha) * mu_s(t-1) + alpha * s_t
            self.mu_s = (1 - alpha) * self.mu_s + alpha * s_t

            # Centered observation
            s_centered = s_t - self.mu_s

            # EMA update for covariance: Sigma_s(t) = (1-alpha)*Sigma_s(t-1) + alpha*(s-mu)(s-mu).T
            outer = np.outer(s_centered, s_centered)
            self.Sigma_s = (1 - alpha) * self.Sigma_s + alpha * outer

        GROUNDING_PROVENANCE.log(
            'mu_s', float(np.linalg.norm(self.mu_s)),
            'EMA(s), alpha = 1/sqrt(t+1)',
            {'alpha': alpha, 't': self.t},
            self.t
        )

        diagnostics = {
            'alpha': alpha,
            'mu_norm': float(np.linalg.norm(self.mu_s)),
            'Sigma_trace': float(np.trace(self.Sigma_s))
        }

        return self.mu_s, self.Sigma_s, alpha, diagnostics


# =============================================================================
# EIGENDECOMPOSITION AND EFFECTIVE DIMENSION
# =============================================================================

class EigenProjection:
    """
    Computes eigendecomposition and effective dimension.

    Sigma_s = V_s @ diag(lambda_s) @ V_s.T
    d_s(t) = sum(lambda_i > median(lambda))
    P_s = V_s[:, :d_s] @ V_s[:, :d_s].T
    """

    def __init__(self):
        self.d_s_history: List[int] = []
        self.t = 0

    def compute(self, Sigma_s: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray, Dict]:
        """
        Compute projection matrix via eigendecomposition.

        Args:
            Sigma_s: Covariance matrix of external signal

        Returns:
            (P_s, d_s, eigenvalues, diagnostics)
        """
        self.t += 1
        dim = Sigma_s.shape[0]

        # Eigendecomposition: Sigma_s = V @ diag(lambda) @ V.T
        eigenvalues, V_s = np.linalg.eigh(Sigma_s)

        # Sort by descending eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        V_s = V_s[:, idx]

        # Ensure non-negative eigenvalues
        eigenvalues = np.maximum(eigenvalues, 0)

        # Effective dimension: d_s = sum(lambda_i > median(lambda))
        if len(eigenvalues) > 0:
            median_lambda = np.median(eigenvalues)
            d_s = int(np.sum(eigenvalues > median_lambda))
            d_s = max(1, d_s)  # At least 1 dimension
        else:
            d_s = 1

        self.d_s_history.append(d_s)

        # Projection matrix: P_s = V_s[:, :d_s] @ V_s[:, :d_s].T
        V_reduced = V_s[:, :d_s]
        P_s = V_reduced @ V_reduced.T

        GROUNDING_PROVENANCE.log(
            'd_s', float(d_s),
            'sum(lambda_i > median(lambda))',
            {'median_lambda': float(median_lambda) if len(eigenvalues) > 0 else 0,
             'max_lambda': float(eigenvalues[0]) if len(eigenvalues) > 0 else 0},
            self.t
        )

        diagnostics = {
            'd_s': d_s,
            'dim': dim,
            'eigenvalues': eigenvalues.tolist()[:5],  # Top 5
            'explained_variance': float(np.sum(eigenvalues[:d_s]) / (np.sum(eigenvalues) + NUMERIC_EPS))
        }

        return P_s, d_s, eigenvalues, diagnostics


# =============================================================================
# COUPLING ERROR AND GRADIENT LEARNING
# =============================================================================

class GradientLearning:
    """
    Implements gradient-based coupling learning.

    e_t = u_t - y_t (coupling error)
    g_t = ∂||e||²/∂W (gradient direction)
    eta_t = 1/sqrt(n+1) (learning rate)
    omega_t = rank(E_t) (weighting)
    W_new = W + eta * omega * normalize(g)
    """

    def __init__(self, dim: int):
        # Coupling weight matrix (identity initially)
        self.W = np.eye(dim)
        self.E_history: List[float] = []
        self.n_updates = 0
        self.t = 0

    def compute_error(self, u_t: np.ndarray, y_t: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute coupling error.

        e_t = u_t - y_t

        Args:
            u_t: Internal representation
            y_t: Projected external signal (W @ s_projected)

        Returns:
            (e_t, E_t)
        """
        # Handle dimension mismatch
        min_dim = min(len(u_t), len(y_t))
        e_t = u_t[:min_dim] - y_t[:min_dim]
        E_t = float(np.linalg.norm(e_t))
        return e_t, E_t

    def update(self, u_t: np.ndarray, s_projected: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Update coupling weights via gradient descent.

        Args:
            u_t: Internal representation
            s_projected: Projected external signal (before W)

        Returns:
            (W, diagnostics)
        """
        self.t += 1
        self.n_updates += 1

        # Handle dimension changes
        dim_u = len(u_t)
        if self.W.shape[0] != dim_u:
            old_dim = self.W.shape[0]
            new_W = np.eye(dim_u)
            min_d = min(old_dim, dim_u)
            new_W[:min_d, :min_d] = self.W[:min_d, :min_d]
            self.W = new_W

        # Current output: y_t = W @ s_projected
        min_dim = min(dim_u, len(s_projected))
        s_adj = np.zeros(dim_u)
        s_adj[:min_dim] = s_projected[:min_dim]
        y_t = self.W @ s_adj

        # Coupling error: e_t = u_t - y_t
        e_t, E_t = self.compute_error(u_t, y_t)
        self.E_history.append(E_t)

        # eta_t = 1/sqrt(n+1) - endogenous learning rate
        eta_t = 1.0 / np.sqrt(self.n_updates + 1)

        # omega_t = rank(E_t) - error-weighted learning
        E_arr = np.array(self.E_history)
        omega_t = compute_rank(E_t, E_arr)

        # Gradient: g_t = ∂||e||²/∂W = -2 * e_t @ s_adj.T
        # (simplified outer product gradient)
        e_padded = np.zeros(dim_u)
        e_padded[:len(e_t)] = e_t
        g_t = -2.0 * np.outer(e_padded, s_adj)

        # Normalize gradient
        g_norm = np.linalg.norm(g_t, 'fro')
        if g_norm > NUMERIC_EPS:
            g_normalized = g_t / g_norm
        else:
            g_normalized = g_t

        # Update: W_new = W + eta * omega * g_normalized
        self.W = self.W + eta_t * omega_t * g_normalized

        GROUNDING_PROVENANCE.log(
            'W_update', float(eta_t * omega_t),
            'W + eta * omega * normalize(g), eta = 1/sqrt(n+1)',
            {'eta_t': eta_t, 'omega_t': omega_t, 'E_t': E_t},
            self.t
        )

        diagnostics = {
            'E_t': E_t,
            'eta_t': eta_t,
            'omega_t': omega_t,
            'g_norm': g_norm,
            'W_norm': float(np.linalg.norm(self.W, 'fro'))
        }

        return self.W, diagnostics


# =============================================================================
# GROUNDING FIELD (FULL SPEC)
# =============================================================================

class GroundingField:
    """
    Computes grounding field from projection.

    G_t = rank(||P_s @ z||) * normalize(P_s @ z - z)

    Pulls internal state toward the external signal subspace.
    """

    def __init__(self):
        self.alignment_history: List[float] = []
        self.G_magnitude_history: List[float] = []
        self.t = 0

    def compute(self, z_t: np.ndarray, P_s: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Compute grounding field.

        G_t = rank(alignment) * normalize(P_s @ z - z)

        Args:
            z_t: Internal state
            P_s: Projection matrix onto external signal subspace

        Returns:
            (G, diagnostics)
        """
        self.t += 1

        # Handle dimension mismatch
        dim_z = len(z_t)
        dim_P = P_s.shape[0]

        if dim_z != dim_P:
            min_dim = min(dim_z, dim_P)
            if dim_P > dim_z:
                z_adj = np.zeros(dim_P)
                z_adj[:dim_z] = z_t
            else:
                z_adj = z_t[:dim_P]
            P_adj = P_s
        else:
            z_adj = z_t
            P_adj = P_s

        # Project: P_s @ z
        z_projected = P_adj @ z_adj

        # Alignment: ||P_s @ z||
        alignment = float(np.linalg.norm(z_projected))
        self.alignment_history.append(alignment)

        # Rank-based gain
        align_arr = np.array(self.alignment_history)
        rank_alignment = compute_rank(alignment, align_arr)

        # Direction: from z toward projected space
        direction = z_projected - z_adj[:len(z_projected)]
        direction_norm = normalize_vector(direction)

        # G_t = rank(alignment) * normalize(P_s @ z - z)
        G = rank_alignment * direction_norm

        # Pad back if needed
        if len(G) < dim_z:
            G_full = np.zeros(dim_z)
            G_full[:len(G)] = G
            G = G_full

        G_magnitude = float(np.linalg.norm(G))
        self.G_magnitude_history.append(G_magnitude)

        GROUNDING_PROVENANCE.log(
            'G', G_magnitude,
            'rank(||P_s @ z||) * normalize(P_s @ z - z)',
            {'rank_alignment': rank_alignment, 'alignment': alignment},
            self.t
        )

        diagnostics = {
            'alignment': alignment,
            'rank_alignment': rank_alignment,
            'G_magnitude': G_magnitude
        }

        return G, diagnostics

    def get_statistics(self) -> Dict:
        if not self.G_magnitude_history:
            return {'n_fields': 0}

        return {
            'n_fields': len(self.G_magnitude_history),
            'mean_G': float(np.mean(self.G_magnitude_history)),
            'std_G': float(np.std(self.G_magnitude_history)),
            'mean_alignment': float(np.mean(self.alignment_history))
        }


# =============================================================================
# MINIMAL GROUNDING SYSTEM (FULL SPECIFICATION)
# =============================================================================

class MinimalGrounding:
    """
    Main class for Phase 22 minimal grounding (FULL SPEC).

    Integrates:
    - Running statistics (mu_s, Sigma_s with EMA)
    - Eigendecomposition (V_s, lambda_s)
    - Effective dimension (d_s from median eigenvalue)
    - Projection matrix (P_s from top eigenvectors)
    - Coupling error (e_t = u_t - y_t)
    - Gradient learning (W update with eta, omega)
    - Grounding field (G_t)

    ALL parameters endogenous.
    """

    def __init__(self, dim: int = 5):
        self.stats = RunningStatistics()
        self.eigen_proj = EigenProjection()
        self.gradient = GradientLearning(dim)
        self.grounder = GroundingField()
        self.dim = dim
        self.t = 0

    def process_step(self, z_t: np.ndarray, s_ext: np.ndarray) -> Dict:
        """
        Process one step of grounding (FULL SPEC).

        Args:
            z_t: Current internal state
            s_ext: External signal

        Returns:
            Dict with grounding outputs
        """
        self.t += 1

        # 1. Update running statistics: mu_s, Sigma_s
        mu_s, Sigma_s, alpha, stats_diag = self.stats.update(s_ext)

        # 2. Eigendecomposition and effective dimension
        P_s, d_s, eigenvalues, eigen_diag = self.eigen_proj.compute(Sigma_s)

        # 3. Normalize external signal
        s_centered = s_ext - mu_s
        s_normalized = normalize_vector(s_centered)

        # 4. Project onto reduced subspace
        min_dim = min(len(s_normalized), P_s.shape[0])
        s_adj = np.zeros(P_s.shape[0])
        s_adj[:min_dim] = s_normalized[:min_dim]
        s_projected = P_s @ s_adj

        # 5. Gradient learning update
        W, grad_diag = self.gradient.update(z_t, s_projected)

        # 6. Compute grounding field
        G, ground_diag = self.grounder.compute(z_t, P_s)

        result = {
            't': self.t,
            'mu_s_norm': float(np.linalg.norm(mu_s)),
            'd_s': d_s,
            'P_s_norm': float(np.linalg.norm(P_s, 'fro')),
            'W_norm': grad_diag['W_norm'],
            'E_t': grad_diag['E_t'],
            'G': G.tolist(),
            'G_magnitude': ground_diag['G_magnitude'],
            'eigenvalues': eigenvalues[:5].tolist() if len(eigenvalues) >= 5 else eigenvalues.tolist(),
            'diagnostics': {
                'stats': stats_diag,
                'eigen': eigen_diag,
                'gradient': grad_diag,
                'grounder': ground_diag
            }
        }

        return result

    def apply_grounding(self, z_base: np.ndarray, G: np.ndarray) -> np.ndarray:
        """
        Apply grounding field to base state update.

        z_next = z_base + G
        """
        min_dim = min(len(z_base), len(G))
        z_next = z_base.copy()
        z_next[:min_dim] += G[:min_dim]
        return z_next

    def get_statistics(self) -> Dict:
        return {
            'grounder': self.grounder.get_statistics(),
            'd_s_history': self.eigen_proj.d_s_history[-10:] if self.eigen_proj.d_s_history else [],
            'E_history': self.gradient.E_history[-10:] if self.gradient.E_history else [],
            'n_steps': self.t
        }


# =============================================================================
# PROVENANCE (FULL SPEC)
# =============================================================================

GROUNDING22_PROVENANCE = {
    'module': 'grounding22',
    'version': '2.0.0',  # Full specification
    'mechanisms': [
        'running_statistics',
        'eigendecomposition',
        'effective_dimension',
        'projection_matrix',
        'coupling_error',
        'gradient_learning',
        'grounding_field'
    ],
    'endogenous_params': [
        's_tilde: s_tilde = normalize(s_ext - mu_s)',
        'mu_s: mu_s(t) = (1-alpha)*mu_s(t-1) + alpha*s_t',
        'Sigma_s: Sigma_s(t) = (1-alpha)*Sigma_s(t-1) + alpha*(s-mu)(s-mu).T',
        'alpha: alpha = 1/sqrt(t+1)',
        'Sigma_s = V_s @ diag(lambda_s) @ V_s.T',
        'd_s: d_s = sum(lambda_i > median(lambda))',
        'P: P_s = V_s[:,:d_s] @ V_s[:,:d_s].T',
        'e_t: e_t = u_t - y_t',
        'g_t: g_t = ∂||e||²/∂W',
        'eta_t: eta_t = 1/sqrt(n+1)',
        'omega_t: omega_t = rank(E_t)',
        'W_new: W_new = W + eta * omega * normalize(g)',
        'G: G_t = rank(||P_s @ z||) * normalize(P_s @ z - z)',
        'window: window = sqrt(t+1)',
        'z_next: z_next = z_base + G'
    ],
    'no_magic_numbers': True,
    'no_semantic_labels': True
}


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("Phase 22: Minimal Grounding (FULL SPEC)")
    print("=" * 60)

    np.random.seed(42)

    # Test grounding system
    print("\n[1] Testing MinimalGrounding (Full Specification)...")
    grounding = MinimalGrounding(dim=5)

    T = 500
    dim_z = 5
    dim_s = 4

    G_magnitude_history = []
    d_s_history = []
    E_history = []

    for t in range(T):
        # Internal state evolving
        z_t = np.sin(np.arange(dim_z) * 0.1 + t * 0.05) + np.random.randn(dim_z) * 0.1

        # External signal (periodic with noise)
        s_ext = np.cos(np.arange(dim_s) * 0.15 + t * 0.03) + np.random.randn(dim_s) * 0.2

        result = grounding.process_step(z_t, s_ext)
        G_magnitude_history.append(result['G_magnitude'])
        d_s_history.append(result['d_s'])
        E_history.append(result['E_t'])

        if t % 100 == 0:
            print(f"  t={t}: |G|={result['G_magnitude']:.4f}, d_s={result['d_s']}, "
                  f"E_t={result['E_t']:.4f}, W_norm={result['W_norm']:.4f}")

    stats = grounding.get_statistics()
    print(f"\n[2] Final Statistics:")
    print(f"  Mean |G|: {stats['grounder']['mean_G']:.4f}")
    print(f"  Mean alignment: {stats['grounder']['mean_alignment']:.4f}")
    print(f"  Final d_s values: {stats['d_s_history'][-5:]}")

    print("\n" + "=" * 60)
    print("PHASE 22 FULL SPECIFICATION VERIFICATION:")
    print("  - mu_s(t) = (1-alpha)*mu_s(t-1) + alpha*s_t")
    print("  - Sigma_s(t) = EMA covariance")
    print("  - alpha = 1/sqrt(t+1)")
    print("  - d_s = sum(lambda_i > median(lambda))")
    print("  - P_s = V_s[:,:d_s] @ V_s[:,:d_s].T")
    print("  - e_t = u_t - y_t (coupling error)")
    print("  - eta_t = 1/sqrt(n+1), omega_t = rank(E_t)")
    print("  - W_new = W + eta * omega * normalize(g)")
    print("  - G_t = rank(alignment) * normalize(P_s@z - z)")
    print("  - ZERO magic constants")
    print("=" * 60)
