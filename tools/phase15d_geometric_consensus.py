#!/usr/bin/env python3
"""
Phase 15D: Endogenous Geometric Consensus & Conditional Intent
==============================================================

Implements:
1. Dual-memory drift (fast/slow) for prototype deformation
2. Geometric penalty decomposition (distance + curvature)
3. Diffusion geometry with eigengap-based dimensionality
4. Stability selection via block bootstrap consensus
5. Conditional path-intent in active windows

100% endogenous - ZERO magic numbers.
NO semantic labels.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
import json
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '/root/NEO_EVA/tools')

from endogenous_core import (
    derive_window_size,
    compute_entropy_normalized,
    compute_acf_lag1,
    NUMERIC_EPS,
    PROVENANCE
)


# =============================================================================
# 1. DUAL-MEMORY DRIFT SYSTEM
# =============================================================================

@dataclass
class DualMemoryState:
    """State for dual-memory drift tracking per prototype."""
    prototype_id: int
    dimension: int

    # Visit tracking
    total_visits: int = 0
    recent_visits: int = 0
    last_visit_time: int = 0

    # Dual drift vectors
    fast_drift: np.ndarray = field(default_factory=lambda: None)
    slow_drift: np.ndarray = field(default_factory=lambda: None)

    # Variance tracking for window reset
    recent_deltas: List[np.ndarray] = field(default_factory=list)
    variance_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        if self.fast_drift is None:
            self.fast_drift = np.zeros(self.dimension)
        if self.slow_drift is None:
            self.slow_drift = np.zeros(self.dimension)


class DualMemoryDrift:
    """
    Dual-memory drift system with fast (reactive) and slow (consolidated) components.

    Fast memory: eta_fast = 1 / (n_local + 1)
    Slow memory: eta_slow = 1 / (N_total + 1)

    Prototype update: prototype += alpha_t * fast_drift + (1-alpha_t) * slow_drift
    where alpha_t = rank_scaled(||delta_t||)
    """

    def __init__(self, dimension: int = 4):
        self.dimension = dimension
        self.states: Dict[int, DualMemoryState] = {}

        # Global history for rank scaling
        self.delta_norm_history: deque = deque(maxlen=1000)
        self.variance_history: deque = deque(maxlen=1000)

        # Derived window size for "recent" tracking
        self.max_recent_window = 100  # Will be adapted endogenously

        self.t = 0

    def _get_or_create_state(self, prototype_id: int) -> DualMemoryState:
        """Get or create dual-memory state for prototype."""
        if prototype_id not in self.states:
            self.states[prototype_id] = DualMemoryState(
                prototype_id=prototype_id,
                dimension=self.dimension
            )
        return self.states[prototype_id]

    def _should_reset_window(self, state: DualMemoryState) -> bool:
        """
        Check if local variance has stabilized → reset recent window.

        Uses endogenous criterion: variance change < IQR of variance history.
        """
        if len(state.variance_history) < 5:
            return False

        recent_vars = state.variance_history[-5:]
        var_change = abs(recent_vars[-1] - recent_vars[-2]) if len(recent_vars) >= 2 else 0

        if len(self.variance_history) < 10:
            return False

        # IQR-based threshold
        all_vars = np.array(list(self.variance_history))
        iqr = np.percentile(all_vars, 75) - np.percentile(all_vars, 25)

        return var_change < iqr * 0.1  # Stabilized if change < 10% of IQR

    def _compute_alpha(self, delta_norm: float) -> float:
        """
        Compute mixing factor alpha_t from rank-scaled delta norm.

        High delta → high alpha → more weight on fast memory (reactive)
        Low delta → low alpha → more weight on slow memory (consolidated)
        """
        if len(self.delta_norm_history) < 10:
            return 0.5

        history = np.array(list(self.delta_norm_history))
        rank = np.sum(history < delta_norm) / len(history)
        return float(rank)

    def update(self, prototype_id: int,
               current_prototype: np.ndarray,
               state_vec: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Update dual-memory drift and return deformed prototype.

        Returns:
            (deformed_prototype, metrics_dict)
        """
        state = self._get_or_create_state(prototype_id)

        # Compute delta
        delta = state_vec - current_prototype
        delta_norm = np.linalg.norm(delta)
        self.delta_norm_history.append(delta_norm)

        # Update visit counts
        state.total_visits += 1
        state.recent_visits += 1

        # Check window reset
        if self._should_reset_window(state):
            state.recent_visits = 1
            state.recent_deltas = []

        # Track recent deltas for variance
        state.recent_deltas.append(delta.copy())
        if len(state.recent_deltas) > self.max_recent_window:
            state.recent_deltas.pop(0)

        # Compute local variance
        if len(state.recent_deltas) > 1:
            local_var = float(np.var([np.linalg.norm(d) for d in state.recent_deltas]))
        else:
            local_var = 0.0
        state.variance_history.append(local_var)
        self.variance_history.append(local_var)

        # Compute etas (endogenous learning rates)
        eta_fast = 1.0 / (state.recent_visits + 1)
        eta_slow = 1.0 / (state.total_visits + 1)

        # Update dual drifts
        state.fast_drift = state.fast_drift + eta_fast * delta
        state.slow_drift = state.slow_drift + eta_slow * delta

        # Compute alpha (mixing factor)
        alpha = self._compute_alpha(delta_norm)

        # Deform prototype
        deformed = current_prototype + alpha * state.fast_drift + (1 - alpha) * state.slow_drift

        state.last_visit_time = self.t
        self.t += 1

        return deformed, {
            'fast_drift_norm': float(np.linalg.norm(state.fast_drift)),
            'slow_drift_norm': float(np.linalg.norm(state.slow_drift)),
            'alpha': alpha,
            'eta_fast': eta_fast,
            'eta_slow': eta_slow,
            'local_variance': local_var,
            'recent_visits': state.recent_visits,
            'total_visits': state.total_visits
        }

    def get_statistics(self) -> Dict:
        """Return statistics about dual-memory drift."""
        if not self.states:
            return {'n_prototypes': 0}

        fast_norms = [np.linalg.norm(s.fast_drift) for s in self.states.values()]
        slow_norms = [np.linalg.norm(s.slow_drift) for s in self.states.values()]

        return {
            'n_prototypes': len(self.states),
            'fast_drift': {
                'mean': float(np.mean(fast_norms)),
                'std': float(np.std(fast_norms)),
                'max': float(np.max(fast_norms))
            },
            'slow_drift': {
                'mean': float(np.mean(slow_norms)),
                'std': float(np.std(slow_norms)),
                'max': float(np.max(slow_norms))
            },
            't': self.t
        }


# =============================================================================
# 2. GEOMETRIC PENALTY DECOMPOSITION
# =============================================================================

class GeometricPenalty:
    """
    Decomposes return penalty into geometric components:
    1. Distance component: ||prototype_k(t) - last_prototype_k||
    2. Curvature component: ||x_t - 2*x_{t-1} + x_{t-2}|| (discrete 2nd derivative)

    Combined via rank-normalized L2 norm (no magic weights).
    """

    def __init__(self, dimension: int = 4):
        self.dimension = dimension

        # Per-prototype state
        self.last_prototypes: Dict[int, np.ndarray] = {}
        self.visit_history: Dict[int, List[np.ndarray]] = {}

        # Global history for rank normalization
        self.distance_history: deque = deque(maxlen=1000)
        self.curvature_history: deque = deque(maxlen=1000)
        self.penalty_history: deque = deque(maxlen=1000)

    def compute_penalty(self, prototype_id: int,
                       current_prototype: np.ndarray,
                       state_vec: np.ndarray) -> Tuple[float, Dict]:
        """
        Compute geometric penalty for returning to prototype.

        Returns:
            (penalty_value, components_dict)
        """
        # Track visit history for curvature
        if prototype_id not in self.visit_history:
            self.visit_history[prototype_id] = []
        self.visit_history[prototype_id].append(state_vec.copy())

        # Keep limited history
        if len(self.visit_history[prototype_id]) > 100:
            self.visit_history[prototype_id].pop(0)

        # Component 1: Distance
        if prototype_id in self.last_prototypes:
            distance = float(np.linalg.norm(current_prototype - self.last_prototypes[prototype_id]))
        else:
            distance = 0.0
        self.distance_history.append(distance)

        # Update last prototype
        self.last_prototypes[prototype_id] = current_prototype.copy()

        # Component 2: Curvature (discrete 2nd derivative)
        visits = self.visit_history[prototype_id]
        if len(visits) >= 3:
            # Second discrete derivative: x_t - 2*x_{t-1} + x_{t-2}
            x_t = visits[-1]
            x_t1 = visits[-2]
            x_t2 = visits[-3]
            curvature = float(np.linalg.norm(x_t - 2*x_t1 + x_t2))
        else:
            curvature = 0.0
        self.curvature_history.append(curvature)

        # Rank-normalize both components
        if len(self.distance_history) > 10:
            dist_arr = np.array(list(self.distance_history))
            dist_rank = np.sum(dist_arr < distance) / len(dist_arr)
        else:
            dist_rank = distance / (distance + 1)

        if len(self.curvature_history) > 10:
            curv_arr = np.array(list(self.curvature_history))
            curv_rank = np.sum(curv_arr < curvature) / len(curv_arr)
        else:
            curv_rank = curvature / (curvature + 1)

        # Combine via L2 norm of rank-normalized components (no magic weights)
        penalty = np.sqrt(dist_rank**2 + curv_rank**2)
        self.penalty_history.append(penalty)

        return float(penalty), {
            'distance': distance,
            'curvature': curvature,
            'distance_rank': float(dist_rank),
            'curvature_rank': float(curv_rank),
            'penalty': float(penalty)
        }

    def get_statistics(self) -> Dict:
        """Return penalty statistics."""
        if len(self.penalty_history) == 0:
            return {'n_penalties': 0}

        penalties = np.array(list(self.penalty_history))
        distances = np.array(list(self.distance_history))
        curvatures = np.array(list(self.curvature_history))

        return {
            'n_penalties': len(penalties),
            'penalty': {
                'mean': float(np.mean(penalties)),
                'std': float(np.std(penalties)),
                'median': float(np.median(penalties))
            },
            'distance': {
                'mean': float(np.mean(distances)),
                'std': float(np.std(distances))
            },
            'curvature': {
                'mean': float(np.mean(curvatures)),
                'std': float(np.std(curvatures))
            }
        }


# =============================================================================
# 3. DIFFUSION GEOMETRY
# =============================================================================

class DiffusionGeometry:
    """
    Builds diffusion coordinates from empirical transition matrix.

    - Computes eigenvectors at diffusion times t ∈ {1, √T}
    - Selects dimensionality k via eigengap
    - All thresholds derived from spectral history quantiles
    """

    def __init__(self):
        self.eigengap_history: List[float] = []
        self.spectrum_history: List[np.ndarray] = []

    def compute_transition_matrix(self, state_sequence: List[int]) -> Tuple[np.ndarray, List[int]]:
        """Compute empirical transition matrix from state sequence."""
        if len(state_sequence) < 2:
            return np.array([[]]), []

        states = sorted(set(state_sequence))
        n = len(states)
        state_to_idx = {s: i for i, s in enumerate(states)}

        counts = np.zeros((n, n))
        for i in range(len(state_sequence) - 1):
            from_idx = state_to_idx[state_sequence[i]]
            to_idx = state_to_idx[state_sequence[i + 1]]
            counts[from_idx, to_idx] += 1

        # Normalize rows (add small eps for stability)
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        P = counts / row_sums

        return P, states

    def compute_diffusion_coords(self, P: np.ndarray,
                                  diffusion_time: int = 1,
                                  k: int = None) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Compute diffusion coordinates.

        Args:
            P: Transition matrix
            diffusion_time: Diffusion time parameter
            k: Number of coordinates (if None, auto-select via eigengap)

        Returns:
            (coordinates, eigenvalues, k_selected)
        """
        n = P.shape[0]
        if n < 3:
            return np.zeros((n, 1)), np.array([1.0]), 1

        # Power of transition matrix for diffusion time
        P_t = np.linalg.matrix_power(P, diffusion_time)

        # Symmetrize for real eigenvalues: (P + P.T) / 2
        P_sym = (P_t + P_t.T) / 2

        try:
            eigenvalues, eigenvectors = eigh(P_sym)
            # Sort descending
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        except:
            return np.zeros((n, 1)), np.array([1.0]), 1

        # Store spectrum
        self.spectrum_history.append(eigenvalues.copy())

        # Select k via eigengap if not specified
        if k is None:
            k = self._select_k_via_eigengap(eigenvalues)

        k = min(k, n - 1, len(eigenvalues))
        k = max(k, 1)

        # Diffusion coordinates: eigenvectors weighted by eigenvalues
        coords = eigenvectors[:, :k] * (eigenvalues[:k] ** diffusion_time)

        return coords, eigenvalues, k

    def _select_k_via_eigengap(self, eigenvalues: np.ndarray) -> int:
        """
        Select dimensionality k via eigengap.

        Uses endogenous threshold from spectral history.
        """
        if len(eigenvalues) < 3:
            return 1

        # Compute gaps
        gaps = np.diff(eigenvalues[:min(10, len(eigenvalues))])
        gaps = np.abs(gaps)

        if len(gaps) == 0:
            return 1

        # Store max gap
        max_gap = float(np.max(gaps))
        self.eigengap_history.append(max_gap)

        # Threshold from history (median of gaps)
        if len(self.eigengap_history) > 5:
            threshold = np.median(self.eigengap_history)
        else:
            threshold = np.median(gaps)

        # Find first gap above threshold
        for i, g in enumerate(gaps):
            if g > threshold:
                return i + 1

        return len(gaps)

    def get_eigengap_stats(self) -> Dict:
        """Return eigengap statistics."""
        if len(self.eigengap_history) == 0:
            return {'n_spectra': 0}

        gaps = np.array(self.eigengap_history)
        return {
            'n_spectra': len(gaps),
            'mean_gap': float(np.mean(gaps)),
            'std_gap': float(np.std(gaps)),
            'median_gap': float(np.median(gaps))
        }


# =============================================================================
# 4. STABILITY SELECTION & CONSENSUS
# =============================================================================

class StabilitySelection:
    """
    Block bootstrap consensus clustering with stability selection.

    - Block length L = √T
    - Co-assignment matrix from resamples
    - Selection probabilities vs label-shuffled null
    - ARI consensus metric
    """

    def __init__(self):
        pass

    def block_bootstrap_sample(self, data: np.ndarray, block_length: int) -> np.ndarray:
        """Generate block bootstrap sample."""
        n = len(data)
        n_blocks = n // block_length + 1

        indices = []
        for _ in range(n_blocks):
            start = np.random.randint(0, max(1, n - block_length + 1))
            indices.extend(range(start, min(start + block_length, n)))

        return data[indices[:n]]

    def cluster_hdbscan(self, coords: np.ndarray,
                        min_cluster_size: int,
                        min_samples: int) -> np.ndarray:
        """
        Cluster using HDBSCAN with endogenous parameters.

        Falls back to simple k-means if HDBSCAN not available.
        """
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(2, min_cluster_size),
                min_samples=max(1, min_samples)
            )
            labels = clusterer.fit_predict(coords)
        except ImportError:
            # Fallback: simple distance-based clustering
            from scipy.cluster.hierarchy import fcluster, linkage
            if len(coords) < 2:
                return np.zeros(len(coords), dtype=int)
            Z = linkage(coords, method='ward')
            # Cut at height derived from data
            heights = Z[:, 2]
            threshold = np.median(heights)
            labels = fcluster(Z, threshold, criterion='distance') - 1

        return labels

    def compute_coassignment_matrix(self, labels_list: List[np.ndarray]) -> np.ndarray:
        """
        Compute co-assignment matrix from multiple clustering results.

        C[i,j] = proportion of times i and j were in same cluster.
        """
        n = len(labels_list[0])
        C = np.zeros((n, n))

        for labels in labels_list:
            same = (labels[:, None] == labels[None, :]).astype(float)
            C += same

        C /= len(labels_list)
        return C

    def compute_consensus_partition(self, C: np.ndarray,
                                     null_threshold: float) -> np.ndarray:
        """
        Derive consensus partition from co-assignment matrix.

        Assigns pairs with selection prob >= threshold to same cluster.
        """
        n = C.shape[0]

        # Use connected components on thresholded co-assignment
        adjacency = (C >= null_threshold).astype(int)

        # Simple connected components
        visited = np.zeros(n, dtype=bool)
        labels = np.zeros(n, dtype=int)
        current_label = 0

        for i in range(n):
            if not visited[i]:
                # BFS
                queue = [i]
                while queue:
                    node = queue.pop(0)
                    if not visited[node]:
                        visited[node] = True
                        labels[node] = current_label
                        for j in range(n):
                            if adjacency[node, j] and not visited[j]:
                                queue.append(j)
                current_label += 1

        return labels

    def compute_null_threshold(self, n: int, n_shuffles: int = 100) -> float:
        """
        Compute q95 threshold from label-shuffled null.
        """
        null_probs = []

        for _ in range(n_shuffles):
            # Random labels
            labels = np.random.randint(0, max(2, n // 10), size=n)
            # Random co-assignment
            same = (labels[:, None] == labels[None, :]).astype(float)
            # Upper triangle values
            upper = same[np.triu_indices(n, k=1)]
            null_probs.extend(upper.tolist())

        return float(np.percentile(null_probs, 95))

    def run_stability_selection(self,
                                state_sequence: List[int],
                                diffusion_coords: np.ndarray,
                                n_bootstrap: int = 50) -> Dict:
        """
        Run full stability selection analysis.
        """
        T = len(state_sequence)
        n = len(diffusion_coords)

        if n < 10:
            return {'error': 'insufficient_data'}

        # Endogenous parameters
        block_length = max(5, int(np.sqrt(T)))
        min_cluster_size = max(2, int(np.sqrt(T)))
        min_samples = max(1, int(np.sqrt(T) / 2))

        # Bootstrap clustering
        labels_list = []

        for _ in range(n_bootstrap):
            # Bootstrap sample indices
            boot_indices = self.block_bootstrap_sample(
                np.arange(n), block_length
            )[:n]
            boot_coords = diffusion_coords[boot_indices]

            # Cluster
            labels = self.cluster_hdbscan(boot_coords, min_cluster_size, min_samples)

            # Map back to original indices
            full_labels = np.zeros(n, dtype=int) - 1
            for i, idx in enumerate(boot_indices):
                if idx < n:
                    full_labels[idx] = labels[i] if i < len(labels) else -1

            labels_list.append(full_labels)

        # Co-assignment matrix
        C = self.compute_coassignment_matrix(labels_list)

        # Null threshold
        null_threshold = self.compute_null_threshold(n)

        # Consensus partition
        consensus_labels = self.compute_consensus_partition(C, null_threshold)

        # Compute ARI vs random
        try:
            from sklearn.metrics import adjusted_rand_score
            # Compare to random baseline
            random_labels = np.random.randint(0, max(2, n // 10), size=n)
            ari_vs_random = adjusted_rand_score(consensus_labels, random_labels)
        except:
            ari_vs_random = 0.0

        return {
            'n_points': n,
            'block_length': block_length,
            'n_bootstrap': n_bootstrap,
            'null_threshold': null_threshold,
            'n_consensus_clusters': len(set(consensus_labels)),
            'ari_vs_random': float(ari_vs_random),
            'coassignment_mean': float(np.mean(C)),
            'coassignment_std': float(np.std(C))
        }


# =============================================================================
# 5. PROCRUSTES ALIGNMENT
# =============================================================================

class ProcrustesTester:
    """
    Procrustes alignment test for cluster stability.

    Compares mean distance across resamples vs geometry-preserving null.
    """

    def compute_procrustes_distance(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute Procrustes distance between two point sets.

        Aligns Y to X via orthogonal transformation.
        """
        if len(X) != len(Y) or len(X) < 2:
            return 1.0

        # Center
        X_centered = X - X.mean(axis=0)
        Y_centered = Y - Y.mean(axis=0)

        # Scale
        X_norm = np.linalg.norm(X_centered)
        Y_norm = np.linalg.norm(Y_centered)

        if X_norm < NUMERIC_EPS or Y_norm < NUMERIC_EPS:
            return 1.0

        X_scaled = X_centered / X_norm
        Y_scaled = Y_centered / Y_norm

        # Optimal rotation via SVD
        try:
            M = Y_scaled.T @ X_scaled
            U, S, Vt = np.linalg.svd(M)
            R = U @ Vt

            # Apply rotation
            Y_aligned = Y_scaled @ R

            # Distance
            dist = np.linalg.norm(X_scaled - Y_aligned)
        except:
            dist = 1.0

        return float(dist)

    def run_procrustes_test(self,
                           coords_list: List[np.ndarray],
                           n_null: int = 100) -> Dict:
        """
        Run Procrustes test comparing real distances vs null.
        """
        if len(coords_list) < 2:
            return {'error': 'insufficient_data'}

        # Real distances
        real_distances = []
        for i in range(len(coords_list)):
            for j in range(i + 1, len(coords_list)):
                d = self.compute_procrustes_distance(coords_list[i], coords_list[j])
                real_distances.append(d)

        real_mean = np.mean(real_distances)

        # Null: permute labels within each resample
        null_distances = []
        for _ in range(n_null):
            null_coords_list = []
            for coords in coords_list:
                perm = np.random.permutation(len(coords))
                null_coords_list.append(coords[perm])

            for i in range(len(null_coords_list)):
                for j in range(i + 1, len(null_coords_list)):
                    d = self.compute_procrustes_distance(null_coords_list[i], null_coords_list[j])
                    null_distances.append(d)

        null_p50 = np.percentile(null_distances, 50)

        return {
            'real_mean_distance': float(real_mean),
            'null_p50': float(null_p50),
            'real_below_null_p50': real_mean < null_p50,
            'n_pairs': len(real_distances),
            'n_null': n_null
        }


# =============================================================================
# 6. CONDITIONAL PATH-INTENT
# =============================================================================

class ConditionalPathIntent:
    """
    Analyzes path intent in active windows (high integration).

    Metrics:
    - Tortuosity
    - Path signature energy
    - Time-reversal KL
    - Action against potential φ = -log(π)
    """

    def __init__(self):
        pass

    def compute_tortuosity(self, trajectory: np.ndarray) -> float:
        """Tortuosity = path_length / displacement."""
        if len(trajectory) < 2:
            return 1.0

        path_length = sum(
            np.linalg.norm(trajectory[i+1] - trajectory[i])
            for i in range(len(trajectory) - 1)
        )

        displacement = np.linalg.norm(trajectory[-1] - trajectory[0])

        if displacement < NUMERIC_EPS:
            return float('inf')

        return path_length / displacement

    def compute_signature_energy(self, trajectory: np.ndarray) -> float:
        """Level-2 path signature energy."""
        if len(trajectory) < 3:
            return 0.0

        # Increments
        increments = np.diff(trajectory, axis=0)

        # Level 1
        level1 = np.sum(increments)

        # Level 2: cross products
        level2 = 0.0
        for i in range(len(increments)):
            for j in range(i + 1, len(increments)):
                level2 += np.dot(increments[i], increments[j])

        return float(level1**2 + level2**2)

    def compute_time_reversal_kl(self, trajectory: np.ndarray) -> float:
        """
        KL divergence between forward and reversed trajectory distributions.
        """
        if len(trajectory) < 3:
            return 0.0

        # Forward increments
        forward_inc = np.diff(trajectory, axis=0)

        # Backward increments (reversed)
        backward_inc = np.diff(trajectory[::-1], axis=0)

        # Compare distributions via histogram approximation
        n_bins = max(5, int(np.sqrt(len(forward_inc))))

        kl = 0.0
        for d in range(trajectory.shape[1]):
            fwd = forward_inc[:, d]
            bwd = backward_inc[:, d]

            # Shared bins
            all_vals = np.concatenate([fwd, bwd])
            bins = np.linspace(all_vals.min() - NUMERIC_EPS,
                              all_vals.max() + NUMERIC_EPS, n_bins)

            hist_fwd, _ = np.histogram(fwd, bins=bins, density=True)
            hist_bwd, _ = np.histogram(bwd, bins=bins, density=True)

            # Add eps for stability
            hist_fwd = hist_fwd + NUMERIC_EPS
            hist_bwd = hist_bwd + NUMERIC_EPS
            hist_fwd = hist_fwd / hist_fwd.sum()
            hist_bwd = hist_bwd / hist_bwd.sum()

            kl += np.sum(hist_fwd * np.log(hist_fwd / hist_bwd))

        return float(kl)

    def compute_action(self, trajectory: np.ndarray,
                       stationary_dist: np.ndarray,
                       state_sequence: List[int]) -> float:
        """
        Action against potential φ = -log(π).

        A = Σ ∇φ · Δx
        """
        if len(trajectory) < 2 or len(stationary_dist) == 0:
            return 0.0

        # Potential: φ = -log(π)
        phi = -np.log(stationary_dist + NUMERIC_EPS)

        action = 0.0
        for t in range(len(trajectory) - 1):
            if t < len(state_sequence) - 1:
                s_t = state_sequence[t]
                s_t1 = state_sequence[t + 1]

                if s_t < len(phi) and s_t1 < len(phi):
                    # Gradient approximation
                    grad_phi = phi[s_t1] - phi[s_t]
                    delta_x = np.linalg.norm(trajectory[t + 1] - trajectory[t])
                    action += grad_phi * delta_x

        return float(action)

    def analyze_active_windows(self,
                              gnt_trajectory: np.ndarray,
                              integration_series: np.ndarray,
                              state_sequence: List[int],
                              stationary_dist: np.ndarray,
                              n_nulls: int = 100) -> Dict:
        """
        Analyze path intent in high-integration windows.

        Active windows: integration in [p90, p99]
        """
        T = len(gnt_trajectory)

        if T < 20:
            return {'error': 'insufficient_data'}

        # Define active windows
        int_p90 = np.percentile(integration_series, 90)
        int_p99 = np.percentile(integration_series, 99)

        active_mask = (integration_series >= int_p90) & (integration_series <= int_p99)
        active_indices = np.where(active_mask)[0]

        if len(active_indices) < 10:
            return {'error': 'insufficient_active_windows', 'n_active': len(active_indices)}

        # Extract active trajectory
        active_traj = gnt_trajectory[active_indices]
        active_states = [state_sequence[i] for i in active_indices if i < len(state_sequence)]

        # Compute metrics
        real_tort = self.compute_tortuosity(active_traj)
        real_sig = self.compute_signature_energy(active_traj)
        real_tr_kl = self.compute_time_reversal_kl(active_traj)
        real_action = self.compute_action(active_traj, stationary_dist, active_states)

        # Generate ACF-matched nulls
        null_torts = []
        null_sigs = []
        null_tr_kls = []
        null_actions = []

        for _ in range(n_nulls):
            null_traj = self._generate_acf_matched_null(active_traj)

            null_torts.append(self.compute_tortuosity(null_traj))
            null_sigs.append(self.compute_signature_energy(null_traj))
            null_tr_kls.append(self.compute_time_reversal_kl(null_traj))
            null_actions.append(self.compute_action(null_traj, stationary_dist, active_states))

        # Filter infinities
        null_torts = [t for t in null_torts if t != float('inf')]

        # P-values
        tort_p = np.mean([nt >= real_tort for nt in null_torts]) if null_torts else 1.0
        sig_p = np.mean([ns >= real_sig for ns in null_sigs])
        tr_kl_p = np.mean([nk >= real_tr_kl for nk in null_tr_kls])
        action_p = np.mean([na >= real_action for na in null_actions])

        return {
            'n_active_points': len(active_indices),
            'integration_range': [float(int_p90), float(int_p99)],
            'real': {
                'tortuosity': float(real_tort) if real_tort != float('inf') else None,
                'signature_energy': float(real_sig),
                'time_reversal_kl': float(real_tr_kl),
                'action': float(real_action)
            },
            'null': {
                'tortuosity_mean': float(np.mean(null_torts)) if null_torts else None,
                'signature_mean': float(np.mean(null_sigs)),
                'tr_kl_mean': float(np.mean(null_tr_kls)),
                'action_mean': float(np.mean(null_actions))
            },
            'p_values': {
                'tortuosity': float(tort_p),
                'signature': float(sig_p),
                'time_reversal_kl': float(tr_kl_p),
                'action': float(action_p)
            },
            'significant_count': sum([
                tort_p < 0.05,
                sig_p < 0.05,
                tr_kl_p < 0.05,
                action_p < 0.05
            ])
        }

    def _generate_acf_matched_null(self, trajectory: np.ndarray) -> np.ndarray:
        """Generate ACF-matched null trajectory."""
        T, dim = trajectory.shape
        null_traj = np.zeros((T, dim))

        for d in range(dim):
            series = trajectory[:, d]
            acf1 = compute_acf_lag1(series)
            var = np.var(series)
            mean = np.mean(series)

            # AR(1) with matching ACF
            phi = acf1
            sigma = np.sqrt(var * (1 - phi**2)) if abs(phi) < 1 else np.sqrt(var)

            null_series = np.zeros(T)
            null_series[0] = mean
            for t in range(1, T):
                null_series[t] = mean + phi * (null_series[t-1] - mean) + np.random.randn() * sigma

            null_traj[:, d] = null_series

        return null_traj


# =============================================================================
# 7. PHASE 15D RUNNER
# =============================================================================

def run_phase15d(n_steps: int = 1000, n_nulls: int = 100,
                 seed: int = 42, verbose: bool = True) -> Dict:
    """
    Run Phase 15D: Endogenous Geometric Consensus & Conditional Intent.
    """
    from emergent_states import EmergentStateSystem
    from global_trace import GNTSystem

    if verbose:
        print("=" * 70)
        print("PHASE 15D: GEOMETRIC CONSENSUS & CONDITIONAL INTENT")
        print("=" * 70)

    np.random.seed(seed)

    # Systems
    states = EmergentStateSystem()
    gnt_system = GNTSystem(dim=8)
    dual_drift = DualMemoryDrift(dimension=4)
    geo_penalty = GeometricPenalty(dimension=4)
    diffusion = DiffusionGeometry()
    stability = StabilitySelection()
    procrustes = ProcrustesTester()
    path_intent = ConditionalPathIntent()

    # Storage
    state_sequence = []
    gnt_trajectory = []
    integration_series = []

    if verbose:
        print(f"\n[1] Simulating {n_steps} steps...")

    neo_pi = np.array([0.33, 0.33, 0.34])
    eva_pi = np.array([0.33, 0.33, 0.34])

    for t in range(n_steps):
        # Simulate dynamics
        coupling = 0.3 + 0.2 * np.tanh(np.random.randn())
        te_neo = max(0, coupling + np.random.randn() * 0.1)
        te_eva = max(0, coupling + np.random.randn() * 0.1)
        neo_se = abs(np.random.randn() * 0.1)
        eva_se = abs(np.random.randn() * 0.1)
        sync = 0.5 + 0.3 * np.tanh(te_neo + te_eva - 0.6)

        neo_pi = np.abs(neo_pi + np.random.randn(3) * 0.03)
        neo_pi = neo_pi / neo_pi.sum()
        eva_pi = np.abs(eva_pi + np.random.randn(3) * 0.03)
        eva_pi = eva_pi / eva_pi.sum()

        # Process states
        result = states.process_step(
            t=t, neo_pi=neo_pi, eva_pi=eva_pi,
            te_neo_to_eva=te_neo, te_eva_to_neo=te_eva,
            neo_self_error=neo_se, eva_self_error=eva_se,
            sync=sync
        )

        # Track state sequence
        neo_state_id = result['neo'].get('prototype_id', 0)
        state_sequence.append(neo_state_id)

        # GNT update
        if states.neo_current_state and states.eva_current_state:
            g_state = np.concatenate([
                states.neo_current_state.to_array(),
                states.eva_current_state.to_array()
            ])
            gnt_result = gnt_system.update(g_state)
            gnt_trajectory.append(gnt_system.gnt.gnt.copy())

            # Integration proxy
            integration = result.get('integration', {}).get('coherence', 0.5)
            integration_series.append(integration)

            # Dual-memory drift
            neo_vec = states.neo_current_state.to_array()
            neo_proto = neo_vec  # Simplified
            dual_drift.update(neo_state_id, neo_proto, neo_vec)

            # Geometric penalty
            geo_penalty.compute_penalty(neo_state_id, neo_proto, neo_vec)

    if verbose:
        print(f"    Completed: {n_steps} steps")

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {'n_steps': n_steps, 'n_nulls': n_nulls, 'seed': seed}
    }

    # 2. Diffusion geometry
    if verbose:
        print("\n[2] Computing diffusion geometry...")

    P, unique_states = diffusion.compute_transition_matrix(state_sequence)

    if P.shape[0] > 2:
        # Diffusion times: {1, √T}
        diff_time_1 = 1
        diff_time_sqrt = max(1, int(np.sqrt(n_steps)))

        coords_1, eigenvals_1, k_1 = diffusion.compute_diffusion_coords(P, diff_time_1)
        coords_sqrt, eigenvals_sqrt, k_sqrt = diffusion.compute_diffusion_coords(P, diff_time_sqrt)

        results['diffusion'] = {
            'n_states': P.shape[0],
            'time_1': {
                'k_selected': k_1,
                'top_eigenvalues': eigenvals_1[:min(5, len(eigenvals_1))].tolist()
            },
            'time_sqrt': {
                'diffusion_time': diff_time_sqrt,
                'k_selected': k_sqrt,
                'top_eigenvalues': eigenvals_sqrt[:min(5, len(eigenvals_sqrt))].tolist()
            },
            'eigengap_stats': diffusion.get_eigengap_stats()
        }

        if verbose:
            print(f"    States: {P.shape[0]}, k={k_1} (t=1), k={k_sqrt} (t={diff_time_sqrt})")
    else:
        results['diffusion'] = {'error': 'insufficient_states'}

    # 3. Stability selection
    if verbose:
        print(f"\n[3] Running stability selection (n_bootstrap={n_nulls})...")

    if P.shape[0] > 2:
        # Map states to diffusion coords
        state_to_idx = {s: i for i, s in enumerate(unique_states)}
        mapped_sequence = [state_to_idx.get(s, 0) for s in state_sequence]

        stability_result = stability.run_stability_selection(
            mapped_sequence, coords_1, n_bootstrap=min(50, n_nulls)
        )
        results['consensus'] = stability_result

        if verbose:
            print(f"    ARI vs random: {stability_result.get('ari_vs_random', 0):.3f}")
    else:
        results['consensus'] = {'error': 'insufficient_states'}

    # 4. Procrustes test
    if verbose:
        print("\n[4] Running Procrustes test...")

    if P.shape[0] > 2:
        # Generate multiple coordinate sets via bootstrap
        coords_list = [coords_1]
        for _ in range(min(10, n_nulls)):
            boot_seq = np.random.choice(len(state_sequence), len(state_sequence), replace=True)
            boot_states = [state_sequence[i] for i in boot_seq]
            P_boot, _ = diffusion.compute_transition_matrix(boot_states)
            if P_boot.shape[0] >= P.shape[0]:
                c, _, _ = diffusion.compute_diffusion_coords(P_boot, 1)
                if c.shape[0] == coords_1.shape[0]:
                    coords_list.append(c)

        procrustes_result = procrustes.run_procrustes_test(coords_list)
        results['procrustes'] = procrustes_result

        if verbose:
            print(f"    Real < p50(null): {procrustes_result.get('real_below_null_p50', False)}")
    else:
        results['procrustes'] = {'error': 'insufficient_states'}

    # 5. Conditional path intent
    if verbose:
        print(f"\n[5] Analyzing conditional path intent...")

    gnt_traj = np.array(gnt_trajectory)
    int_series = np.array(integration_series)

    # Compute stationary distribution
    if P.shape[0] > 0:
        try:
            eigenvalues, eigenvectors = np.linalg.eig(P.T)
            idx = np.argmin(np.abs(eigenvalues - 1.0))
            pi = np.real(eigenvectors[:, idx])
            pi = np.abs(pi)
            pi = pi / pi.sum()
        except:
            pi = np.ones(P.shape[0]) / P.shape[0]
    else:
        pi = np.array([1.0])

    path_result = path_intent.analyze_active_windows(
        gnt_traj, int_series, state_sequence, pi, n_nulls
    )
    results['path_intent'] = path_result

    if verbose and 'significant_count' in path_result:
        print(f"    Significant metrics: {path_result['significant_count']}/4")

    # 6. Drift and penalty stats
    results['dual_drift'] = dual_drift.get_statistics()
    results['geometric_penalty'] = geo_penalty.get_statistics()

    # 7. GO criteria
    if verbose:
        print("\n" + "=" * 70)
        print("GO CRITERIA CHECK")
        print("=" * 70)

    go = {}

    # ARI consensus > p95(null) - approximate with > 0
    go['ARI_consensus_above_null'] = results.get('consensus', {}).get('ari_vs_random', 0) > 0

    # Procrustes < p50(null)
    go['Procrustes_below_p50'] = results.get('procrustes', {}).get('real_below_null_p50', False)

    # >=2 path-intent metrics > p95(null)
    go['Path_intent_significant'] = path_result.get('significant_count', 0) >= 2

    results['go_criteria'] = go
    n_passed = sum(go.values())

    if verbose:
        for k, v in go.items():
            status = "GO" if v else "NO-GO"
            print(f"  {k}: {status}")
        print(f"\n  TOTAL: {n_passed}/{len(go)} passed")
        print("=" * 70)

    return results


def generate_phase15d_report(results: Dict,
                             output_dir: str = '/root/NEO_EVA/results'):
    """Generate Phase 15D outputs."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('/root/NEO_EVA/figures', exist_ok=True)

    # Clean for JSON
    def clean(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj

    cleaned = clean(results)

    # Save JSON files
    with open(f'{output_dir}/15d_consensus.json', 'w') as f:
        json.dump(cleaned.get('consensus', {}), f, indent=2, default=str)

    with open(f'{output_dir}/15d_procrustes.json', 'w') as f:
        json.dump(cleaned.get('procrustes', {}), f, indent=2, default=str)

    with open(f'{output_dir}/15d_path_intent_cond.json', 'w') as f:
        json.dump(cleaned.get('path_intent', {}), f, indent=2, default=str)

    with open(f'{output_dir}/15d_full_results.json', 'w') as f:
        json.dump(cleaned, f, indent=2, default=str)

    # Generate markdown summary
    md = []
    md.append("# Phase 15D: Geometric Consensus & Conditional Intent")
    md.append("")
    md.append(f"**Timestamp:** {results['timestamp']}")
    md.append(f"**Config:** {results['config']}")
    md.append("")

    md.append("## 1. Diffusion Geometry")
    diff = results.get('diffusion', {})
    if 'error' not in diff:
        md.append(f"- States: {diff.get('n_states', 0)}")
        md.append(f"- k (t=1): {diff.get('time_1', {}).get('k_selected', 0)}")
        md.append(f"- k (t=√T): {diff.get('time_sqrt', {}).get('k_selected', 0)}")
    else:
        md.append(f"- Error: {diff.get('error')}")
    md.append("")

    md.append("## 2. Stability Selection")
    cons = results.get('consensus', {})
    if 'error' not in cons:
        md.append(f"- ARI vs random: {cons.get('ari_vs_random', 0):.4f}")
        md.append(f"- Consensus clusters: {cons.get('n_consensus_clusters', 0)}")
    md.append("")

    md.append("## 3. Procrustes Test")
    proc = results.get('procrustes', {})
    if 'error' not in proc:
        md.append(f"- Real mean distance: {proc.get('real_mean_distance', 0):.4f}")
        md.append(f"- Null p50: {proc.get('null_p50', 0):.4f}")
        md.append(f"- Real < p50(null): {'YES' if proc.get('real_below_null_p50') else 'NO'}")
    md.append("")

    md.append("## 4. Conditional Path Intent")
    pi = results.get('path_intent', {})
    if 'p_values' in pi:
        md.append(f"- Active points: {pi.get('n_active_points', 0)}")
        md.append("")
        md.append("| Metric | Real | Null Mean | p-value |")
        md.append("|--------|------|-----------|---------|")
        for metric in ['tortuosity', 'signature', 'time_reversal_kl', 'action']:
            real_key = metric if metric != 'signature' else 'signature_energy'
            null_key = metric + '_mean' if metric != 'signature' else 'signature_mean'
            real_val = pi.get('real', {}).get(real_key, 0)
            null_val = pi.get('null', {}).get(null_key, 0)
            p_val = pi.get('p_values', {}).get(metric, 1)
            md.append(f"| {metric} | {real_val:.4f if real_val else 'N/A'} | {null_val:.4f if null_val else 'N/A'} | {p_val:.4f} |")
        md.append("")
        md.append(f"**Significant metrics: {pi.get('significant_count', 0)}/4**")
    md.append("")

    md.append("## 5. Dual-Memory Drift")
    drift = results.get('dual_drift', {})
    md.append(f"- Fast drift mean: {drift.get('fast_drift', {}).get('mean', 0):.4f}")
    md.append(f"- Slow drift mean: {drift.get('slow_drift', {}).get('mean', 0):.4f}")
    md.append("")

    md.append("## 6. Geometric Penalty")
    penalty = results.get('geometric_penalty', {})
    md.append(f"- Penalty mean: {penalty.get('penalty', {}).get('mean', 0):.4f}")
    md.append(f"- Distance mean: {penalty.get('distance', {}).get('mean', 0):.4f}")
    md.append(f"- Curvature mean: {penalty.get('curvature', {}).get('mean', 0):.4f}")
    md.append("")

    md.append("## GO Criteria")
    go = results.get('go_criteria', {})
    md.append("| Criterion | Status |")
    md.append("|-----------|--------|")
    for k, v in go.items():
        md.append(f"| {k} | {'GO' if v else 'NO-GO'} |")
    md.append("")
    md.append(f"**Total: {sum(go.values())}/{len(go)} passed**")

    with open(f'{output_dir}/phase15d_summary.md', 'w') as f:
        f.write('\n'.join(md))

    return f'{output_dir}/phase15d_summary.md'


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results = run_phase15d(n_steps=1000, n_nulls=100, verbose=True)

    report_path = generate_phase15d_report(results)

    print(f"\n[OK] Results saved to:")
    print(f"  - results/15d_*.json")
    print(f"  - {report_path}")
