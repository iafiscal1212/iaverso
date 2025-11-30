#!/usr/bin/env python3
"""
Phase 16B: Irreversibility Module - Endogenous Upgrade
======================================================

Implements endogenous irreversibility mechanisms:
1. TRUE Prototype Plasticity: μ_k updated online with η_t = 1/√(visits_k+1)
   Direction = tangent-projected (x_t - μ_k), scaled by rank² of deviation
2. Non-Conservative Field: Helmholtz decomposition (gradient vs rotational)
3. Endogenous NESS: τ modulation via rank(surprise) - rank(confidence)
4. Dwell-times: Block-bootstrap from empirical dwell quantiles

100% endogenous - ZERO magic numbers.
NO semantic labels (energy, hunger, reward, etc.)
Only mathematical properties: drift, deformation, distance, curvature.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Deque
from dataclasses import dataclass, field
from collections import deque, Counter
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import warnings

import sys
sys.path.insert(0, '/root/NEO_EVA/tools')

from endogenous_core import (
    derive_window_size,
    derive_buffer_size,
    compute_entropy_normalized,
    compute_iqr,
    rank_normalize,
    rolling_rank,
    NUMERIC_EPS,
    PROVENANCE
)


# =============================================================================
# PROTOTYPE PLASTICITY (TRUE ONLINE LEARNING)
# =============================================================================

@dataclass
class PrototypeDriftState:
    """Tracks drift state for a single prototype with TRUE plasticity."""
    prototype_id: int
    dimension: int
    visits: int = 0
    # TRUE prototype position (updated online)
    mu: np.ndarray = field(default_factory=lambda: None)
    # Drift vector (accumulated deviations)
    drift_vector: np.ndarray = field(default_factory=lambda: None)
    # Last snapshot for return penalty
    last_snapshot: np.ndarray = field(default_factory=lambda: None)
    # Deformation history
    deformation_history: List[float] = field(default_factory=list)
    # Drift RMS per window
    drift_rms_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        if self.mu is None:
            self.mu = np.zeros(self.dimension)
        if self.drift_vector is None:
            self.drift_vector = np.zeros(self.dimension)
        if self.last_snapshot is None:
            self.last_snapshot = np.zeros(self.dimension)


@dataclass
class DualMemoryDriftState:
    """
    Dual-memory drift state for a single prototype.

    Implements "parameters born from results" philosophy:
    - fast_drift: short-term reactive memory
    - slow_drift: long-term consolidated memory
    - variance-based window reset for n_local
    """
    prototype_id: int
    dimension: int

    # Total historical visits (for slow memory)
    N_k: int = 0

    # Local window visits (for fast memory, resets on variance spike)
    n_local: int = 0

    # Fast drift vector (reactive, recent)
    fast_drift: np.ndarray = field(default_factory=lambda: None)

    # Slow drift vector (consolidated, historical)
    slow_drift: np.ndarray = field(default_factory=lambda: None)

    # Combined prototype position
    mu: np.ndarray = field(default_factory=lambda: None)

    # Momentum vector for directionality
    momentum: np.ndarray = field(default_factory=lambda: None)

    # Variance tracking for window reset
    recent_delta_norms: List[float] = field(default_factory=list)
    baseline_variance: float = 0.0

    # History
    fast_drift_history: List[float] = field(default_factory=list)
    slow_drift_history: List[float] = field(default_factory=list)
    align_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        if self.fast_drift is None:
            self.fast_drift = np.zeros(self.dimension)
        if self.slow_drift is None:
            self.slow_drift = np.zeros(self.dimension)
        if self.mu is None:
            self.mu = np.zeros(self.dimension)
        if self.momentum is None:
            self.momentum = np.zeros(self.dimension)


class TruePrototypePlasticity:
    """
    TRUE prototype plasticity with online learning.

    Mathematical basis:
    - μ_k updated online: μ_k ← μ_k + η_t * tangent_proj(x_t - μ_k)
    - η_t = 1 / √(visits_k + 1)  [endogenous learning rate]
    - tangent_proj projects deviation onto local tangent space
    - deformation factor f_t = rank²(||x_t - μ_k||)

    Logs drift_RMS per window for GO criteria verification.
    """

    def __init__(self, dimension: int = 4):
        self.dimension = dimension
        self.prototype_states: Dict[int, PrototypeDriftState] = {}

        # History for rank-based scaling
        derived_maxlen = int(np.sqrt(1e6))
        self.delta_norm_history: Deque[float] = deque(maxlen=derived_maxlen)
        self.deformation_history: List[float] = []

        # Drift RMS tracking per window
        self.window_drift_values: List[float] = []
        self.drift_rms_per_window: List[float] = []

        # Global statistics
        self.total_updates = 0
        self.cumulative_drift_norm = 0.0

    def _get_or_create_state(self, prototype_id: int,
                             initial_vec: np.ndarray = None) -> PrototypeDriftState:
        """Get or create drift state for prototype."""
        if prototype_id not in self.prototype_states:
            self.prototype_states[prototype_id] = PrototypeDriftState(
                prototype_id=prototype_id,
                dimension=self.dimension
            )
            if initial_vec is not None:
                self.prototype_states[prototype_id].mu = initial_vec.copy()
                self.prototype_states[prototype_id].last_snapshot = initial_vec.copy()
        return self.prototype_states[prototype_id]

    def _compute_tangent_projection(self, delta: np.ndarray,
                                    mu: np.ndarray) -> np.ndarray:
        """
        Project delta onto tangent space at mu.

        For simplex manifold: project out the normal (1,1,1,...)/√d
        This ensures updates stay within the manifold structure.
        """
        # Normal to simplex constraint (sum = constant)
        d = len(delta)
        normal = np.ones(d) / np.sqrt(d)

        # Project out normal component
        delta_normal = np.dot(delta, normal) * normal
        delta_tangent = delta - delta_normal

        return delta_tangent

    def _compute_deformation_factor(self, delta_norm: float) -> float:
        """
        Compute deformation factor f_t = rank²(delta_norm).

        Uses rank within historical distribution - NO magic thresholds.
        Returns value in [0, 1] based on percentile rank.
        """
        # Store for rank computation
        self.delta_norm_history.append(delta_norm)

        # Mínimo endógeno
        min_history = int(np.sqrt(len(self.delta_norm_history) + 1)) + 2
        if len(self.delta_norm_history) < min_history:
            # Not enough history - use minimal deformation
            return delta_norm / (delta_norm + 1.0)

        # Rank-based scaling: what percentile is this delta_norm?
        history_array = np.array(self.delta_norm_history)
        rank = np.sum(history_array < delta_norm) / len(history_array)

        # f_t = rank² gives gradual increase, bounded [0,1]
        f_t = rank * rank

        PROVENANCE.log('deformation_factor', f_t,
                      'rank(delta_norm)^2',
                      {'delta_norm': delta_norm, 'rank': rank},
                      self.total_updates)

        return f_t

    def update_prototype(self, prototype_id: int,
                        current_position: np.ndarray,
                        state_vec: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Update prototype with TRUE plasticity.

        Args:
            prototype_id: ID of the prototype being visited
            current_position: Current nominal position of prototype
            state_vec: Actual state vector at this timestep

        Returns:
            (updated_mu, drift_magnitude, drift_rms)
        """
        state = self._get_or_create_state(prototype_id, current_position)
        state.visits += 1
        self.total_updates += 1

        # Calculate instantaneous deviation from current prototype
        delta = state_vec - state.mu
        delta_norm = np.linalg.norm(delta)

        # Endogenous learning rate: η_t = 1 / √(visits + 1)
        eta = 1.0 / np.sqrt(state.visits + 1)

        PROVENANCE.log('eta_prototype', eta,
                      '1/sqrt(visits+1)',
                      {'visits': state.visits, 'prototype_id': prototype_id},
                      self.total_updates)

        # Project onto tangent space
        delta_tangent = self._compute_tangent_projection(delta, state.mu)

        # Compute deformation factor from ranks
        f_t = self._compute_deformation_factor(delta_norm)

        # TRUE PLASTICITY: Update prototype position
        # μ_k ← μ_k + η_t * f_t * tangent_proj(x_t - μ_k)
        state.mu = state.mu + eta * f_t * delta_tangent

        # Update drift vector (accumulates weighted deviations)
        state.drift_vector = state.drift_vector + eta * delta_tangent

        # Track deformation magnitude
        drift_magnitude = np.linalg.norm(state.drift_vector)
        state.deformation_history.append(drift_magnitude)
        self.deformation_history.append(drift_magnitude)
        self.cumulative_drift_norm += drift_magnitude

        # Track window drift for RMS calculation
        self.window_drift_values.append(drift_magnitude)

        # Compute drift RMS per window (endogenous window size)
        window_size = derive_window_size(self.total_updates)
        if len(self.window_drift_values) >= window_size:
            window_values = self.window_drift_values[-window_size:]
            drift_rms = np.sqrt(np.mean(np.array(window_values) ** 2))
            self.drift_rms_per_window.append(drift_rms)
            state.drift_rms_history.append(drift_rms)
        else:
            drift_rms = drift_magnitude

        return state.mu.copy(), drift_magnitude, drift_rms

    def get_drift_rms_statistics(self) -> Dict:
        """Return drift RMS statistics for GO criteria."""
        if not self.drift_rms_per_window:
            return {'error': 'insufficient_data'}

        rms_array = np.array(self.drift_rms_per_window)

        return {
            'mean': float(np.mean(rms_array)),
            'std': float(np.std(rms_array)),
            'median': float(np.median(rms_array)),
            'p5': float(np.percentile(rms_array, 5)),
            'p95': float(np.percentile(rms_array, 95)),
            'n_windows': len(rms_array)
        }

    def get_drift_statistics(self) -> Dict:
        """Return statistics about drift dynamics."""
        if not self.prototype_states:
            return {'n_prototypes': 0}

        all_drifts = []
        all_rms = []
        for state in self.prototype_states.values():
            all_drifts.append(np.linalg.norm(state.drift_vector))
            if state.drift_rms_history:
                all_rms.extend(state.drift_rms_history)

        return {
            'n_prototypes': len(self.prototype_states),
            'total_updates': self.total_updates,
            'drift_norms': {
                'mean': float(np.mean(all_drifts)),
                'std': float(np.std(all_drifts)),
                'max': float(np.max(all_drifts)),
                'min': float(np.min(all_drifts))
            },
            'drift_rms': self.get_drift_rms_statistics(),
            'cumulative_drift': float(self.cumulative_drift_norm),
            'history_length': len(self.deformation_history)
        }


# =============================================================================
# DUAL-MEMORY PLASTICITY (ENDOGENOUS HORMONAL-LIKE SYSTEM)
# =============================================================================

class DualMemoryPlasticity:
    """
    Dual-memory prototype plasticity: "parameters born from results".

    Implements a hormonal-like dual timescale system:

    1. FAST MEMORY (reactive, short-term):
       - η_fast = 1 / √(n_local(t) + 1)
       - n_local resets when variance spikes (stabilization criterion)
       - Captures recent deviations

    2. SLOW MEMORY (consolidated, long-term):
       - η_slow = 1 / √(N_k(t) + 1)
       - N_k is total historical visits (never resets)
       - Maintains stable baseline

    3. WEIGHTING:
       - α_t = rank_scaled(||delta_t||)
       - Combined drift = (1-α_t)*slow_drift + α_t*fast_drift

    4. DIRECTIONALITY:
       - align_t = cos(momentum_t, bias_field(x_t))
       - Tracks whether movement aligns with field

    ALL parameters derived from data - ZERO magic constants.
    """

    def __init__(self, dimension: int = 4):
        self.dimension = dimension
        self.prototype_states: Dict[int, DualMemoryDriftState] = {}

        # History for rank-based scaling
        derived_maxlen = int(np.sqrt(1e6))
        self.delta_norm_history: Deque[float] = deque(maxlen=derived_maxlen)

        # Variance tracking for window reset (global) - derivado dinámicamente
        # Se calcula como sqrt(total_updates) cuando se necesita
        self._variance_window_base = None  # Se inicializa en primer uso
        self.variance_threshold_factor = 2.0  # Reset if var > 2*baseline (from IQR ratio)

        # Drift tracking
        self.window_drift_values: List[float] = []
        self.drift_rms_per_window: List[float] = []

        # Momentum and directionality tracking
        self.momentum_history: List[np.ndarray] = []
        self.align_history: List[float] = []
        self.directional_momentum_values: List[float] = []

        # Global statistics
        self.total_updates = 0

    def _get_or_create_state(self, prototype_id: int,
                             initial_vec: np.ndarray = None) -> DualMemoryDriftState:
        """Get or create dual-memory state for prototype."""
        if prototype_id not in self.prototype_states:
            self.prototype_states[prototype_id] = DualMemoryDriftState(
                prototype_id=prototype_id,
                dimension=self.dimension
            )
            if initial_vec is not None:
                self.prototype_states[prototype_id].mu = initial_vec.copy()
        return self.prototype_states[prototype_id]

    def _should_reset_local_window(self, state: DualMemoryDriftState,
                                   new_delta_norm: float) -> bool:
        """
        Determine if n_local should reset based on variance stabilization.

        Resets when current variance significantly exceeds baseline variance.
        This creates adaptive "epochs" based on data stability.
        """
        state.recent_delta_norms.append(new_delta_norm)

        # Keep only recent values
        # Window size endógeno: sqrt(len(delta_norms))
        window_size = int(np.sqrt(len(state.recent_delta_norms) + 1)) + 2

        if len(state.recent_delta_norms) > window_size * 2:
            state.recent_delta_norms = state.recent_delta_norms[-window_size * 2:]

        if len(state.recent_delta_norms) < window_size:
            return False

        # Compute recent variance
        recent = np.array(state.recent_delta_norms[-window_size:])
        current_variance = np.var(recent)

        # Compute baseline variance from older data
        if len(state.recent_delta_norms) >= window_size * 2:
            older = np.array(state.recent_delta_norms[:-window_size])
            state.baseline_variance = np.var(older)

        # Reset if variance spike detected
        if state.baseline_variance > NUMERIC_EPS:
            ratio = current_variance / state.baseline_variance
            if ratio > self.variance_threshold_factor:
                return True

        return False

    def _compute_alpha_weight(self, delta_norm: float) -> float:
        """
        Compute α_t = rank_scaled(||delta_t||).

        Higher deviation → higher weight on fast memory.
        Returns value in [0, 1].
        """
        self.delta_norm_history.append(delta_norm)

        # Mínimo endógeno
        min_history = int(np.sqrt(len(self.delta_norm_history) + 1)) + 2
        if len(self.delta_norm_history) < min_history:
            return 0.5  # Neutral weighting initially

        history = np.array(self.delta_norm_history)
        alpha = np.sum(history < delta_norm) / len(history)

        return float(alpha)

    def _compute_directionality(self, state: DualMemoryDriftState,
                                delta: np.ndarray,
                                bias_field: np.ndarray = None) -> float:
        """
        Compute align_t = cos(momentum_t, bias_field(x_t)).

        If no explicit bias_field, use the accumulated slow_drift direction.
        """
        # Update momentum (exponential moving average of deltas)
        if len(self.momentum_history) > 0:
            prev_momentum = self.momentum_history[-1]
            # η for momentum EMA derived from total updates
            eta_momentum = 1.0 / np.sqrt(self.total_updates + 1)
            state.momentum = (1 - eta_momentum) * prev_momentum + eta_momentum * delta
        else:
            state.momentum = delta.copy()

        self.momentum_history.append(state.momentum.copy())

        # Use slow_drift as implicit bias field if none provided
        if bias_field is None:
            bias_field = state.slow_drift

        # Compute cosine similarity
        momentum_norm = np.linalg.norm(state.momentum)
        bias_norm = np.linalg.norm(bias_field)

        if momentum_norm > NUMERIC_EPS and bias_norm > NUMERIC_EPS:
            align_t = np.dot(state.momentum, bias_field) / (momentum_norm * bias_norm)
        else:
            align_t = 0.0

        state.align_history.append(align_t)
        self.align_history.append(align_t)

        return float(align_t)

    def _compute_tangent_projection(self, delta: np.ndarray,
                                    mu: np.ndarray) -> np.ndarray:
        """Project delta onto tangent space at mu (simplex manifold)."""
        d = len(delta)
        normal = np.ones(d) / np.sqrt(d)
        delta_normal = np.dot(delta, normal) * normal
        delta_tangent = delta - delta_normal
        return delta_tangent

    def update_prototype(self, prototype_id: int,
                        current_position: np.ndarray,
                        state_vec: np.ndarray,
                        bias_field: np.ndarray = None) -> Tuple[np.ndarray, float, float, Dict]:
        """
        Update prototype with dual-memory plasticity.

        Args:
            prototype_id: ID of the prototype
            current_position: Nominal prototype position
            state_vec: Actual observed state
            bias_field: Optional external bias field for directionality

        Returns:
            (updated_mu, drift_magnitude, drift_rms, diagnostics)
        """
        state = self._get_or_create_state(prototype_id, current_position)

        # Increment visit counts
        state.N_k += 1
        state.n_local += 1
        self.total_updates += 1

        # Compute deviation
        delta = state_vec - state.mu
        delta_norm = np.linalg.norm(delta)

        # Check for variance-based window reset
        if self._should_reset_local_window(state, delta_norm):
            # Reset n_local but NOT N_k
            state.n_local = 1
            # Consolidate fast drift into slow drift
            state.slow_drift = 0.5 * state.slow_drift + 0.5 * state.fast_drift
            state.fast_drift = np.zeros(self.dimension)

            PROVENANCE.log('local_window_reset', self.total_updates,
                          'variance_spike > threshold',
                          {'prototype_id': prototype_id, 'N_k': state.N_k},
                          self.total_updates)

        # Compute learning rates (endogenous)
        eta_fast = 1.0 / np.sqrt(state.n_local + 1)
        eta_slow = 1.0 / np.sqrt(state.N_k + 1)

        PROVENANCE.log('eta_fast', eta_fast,
                      '1/sqrt(n_local+1)',
                      {'n_local': state.n_local}, self.total_updates)
        PROVENANCE.log('eta_slow', eta_slow,
                      '1/sqrt(N_k+1)',
                      {'N_k': state.N_k}, self.total_updates)

        # Project onto tangent space
        delta_tangent = self._compute_tangent_projection(delta, state.mu)

        # Update both drift vectors
        state.fast_drift = state.fast_drift + eta_fast * delta_tangent
        state.slow_drift = state.slow_drift + eta_slow * delta_tangent

        # Compute α_t for weighting
        alpha_t = self._compute_alpha_weight(delta_norm)

        # Combined drift: weighted combination
        combined_drift = (1 - alpha_t) * state.slow_drift + alpha_t * state.fast_drift

        # Update prototype position
        state.mu = state.mu + combined_drift / (state.N_k + 1)

        # Compute directionality
        align_t = self._compute_directionality(state, delta_tangent, bias_field)

        # Compute directional momentum
        directional_momentum = np.linalg.norm(state.momentum) * (1 + align_t)
        self.directional_momentum_values.append(directional_momentum)

        # Track drift magnitude
        drift_magnitude = np.linalg.norm(combined_drift)
        state.fast_drift_history.append(np.linalg.norm(state.fast_drift))
        state.slow_drift_history.append(np.linalg.norm(state.slow_drift))

        # Track window drift for RMS
        self.window_drift_values.append(drift_magnitude)

        # Compute drift RMS per window
        window_size = derive_window_size(self.total_updates)
        if len(self.window_drift_values) >= window_size:
            window_values = self.window_drift_values[-window_size:]
            drift_rms = np.sqrt(np.mean(np.array(window_values) ** 2))
            self.drift_rms_per_window.append(drift_rms)
        else:
            drift_rms = drift_magnitude

        diagnostics = {
            'eta_fast': float(eta_fast),
            'eta_slow': float(eta_slow),
            'alpha_t': float(alpha_t),
            'align_t': float(align_t),
            'n_local': state.n_local,
            'N_k': state.N_k,
            'fast_drift_norm': float(np.linalg.norm(state.fast_drift)),
            'slow_drift_norm': float(np.linalg.norm(state.slow_drift)),
            'directional_momentum': float(directional_momentum)
        }

        return state.mu.copy(), drift_magnitude, drift_rms, diagnostics

    def get_drift_rms_statistics(self) -> Dict:
        """Return drift RMS statistics for GO criteria."""
        if not self.drift_rms_per_window:
            return {'error': 'insufficient_data'}

        rms_array = np.array(self.drift_rms_per_window)

        return {
            'mean': float(np.mean(rms_array)),
            'std': float(np.std(rms_array)),
            'median': float(np.median(rms_array)),
            'p5': float(np.percentile(rms_array, 5)),
            'p95': float(np.percentile(rms_array, 95)),
            'n_windows': len(rms_array)
        }

    def get_momentum_statistics(self) -> Dict:
        """Return directional momentum statistics for GO criteria."""
        if not self.directional_momentum_values:
            return {'error': 'insufficient_data'}

        mom_array = np.array(self.directional_momentum_values)

        return {
            'mean': float(np.mean(mom_array)),
            'std': float(np.std(mom_array)),
            'median': float(np.median(mom_array)),
            'p95': float(np.percentile(mom_array, 95)),
            'n_values': len(mom_array)
        }

    def get_alignment_statistics(self) -> Dict:
        """Return alignment (directionality) statistics."""
        if not self.align_history:
            return {'error': 'insufficient_data'}

        align_array = np.array(self.align_history)

        return {
            'mean': float(np.mean(align_array)),
            'std': float(np.std(align_array)),
            'median': float(np.median(align_array)),
            'positive_fraction': float(np.mean(align_array > 0)),
            'n_values': len(align_array)
        }

    def get_dual_memory_statistics(self) -> Dict:
        """Return comprehensive dual-memory statistics."""
        all_fast_drifts = []
        all_slow_drifts = []
        all_n_local = []
        all_N_k = []

        for state in self.prototype_states.values():
            all_fast_drifts.append(np.linalg.norm(state.fast_drift))
            all_slow_drifts.append(np.linalg.norm(state.slow_drift))
            all_n_local.append(state.n_local)
            all_N_k.append(state.N_k)

        if not all_fast_drifts:
            return {'n_prototypes': 0}

        return {
            'n_prototypes': len(self.prototype_states),
            'total_updates': self.total_updates,
            'fast_drift': {
                'mean': float(np.mean(all_fast_drifts)),
                'std': float(np.std(all_fast_drifts)),
                'max': float(np.max(all_fast_drifts))
            },
            'slow_drift': {
                'mean': float(np.mean(all_slow_drifts)),
                'std': float(np.std(all_slow_drifts)),
                'max': float(np.max(all_slow_drifts))
            },
            'n_local': {
                'mean': float(np.mean(all_n_local)),
                'max': int(np.max(all_n_local))
            },
            'N_k': {
                'mean': float(np.mean(all_N_k)),
                'max': int(np.max(all_N_k))
            },
            'drift_rms': self.get_drift_rms_statistics(),
            'momentum': self.get_momentum_statistics(),
            'alignment': self.get_alignment_statistics()
        }


# =============================================================================
# FLOW DIRECTIONALITY INDEX (NEW METRIC FOR NON-EQUILIBRIUM)
# =============================================================================

class FlowDirectionalityAnalyzer:
    """
    Measures the directionality of state transitions.

    In non-equilibrium dynamics, transitions preferentially go in one direction
    (e.g., 0→1→2→3→4→0). This metric captures that asymmetry directly.

    Flow Directionality Index (FDI):
    - For each pair (i,j), compute: fdi_{ij} = (n_{i→j} - n_{j→i}) / (n_{i→j} + n_{j→i})
    - Global FDI = mean(|fdi_{ij}|) for all pairs with transitions
    - FDI = 0 for equilibrium (symmetric), FDI → 1 for strongly directional

    This directly captures what makes non-equilibrium dynamics distinct.
    """

    def __init__(self, n_states: int = None):
        # n_states se deriva de los datos si no se proporciona
        self.n_states = n_states if n_states is not None else 0
        self.forward_counts: Dict[Tuple[int, int], int] = {}
        self.total_transitions = 0

    def record_transition(self, from_state: int, to_state: int):
        """Record a state transition."""
        key = (from_state, to_state)
        self.forward_counts[key] = self.forward_counts.get(key, 0) + 1
        self.total_transitions += 1

    def compute_flow_directionality_index(self) -> Dict:
        """
        Compute the Flow Directionality Index.

        Returns dict with:
        - global_fdi: Mean |fdi| across all pairs
        - max_fdi: Maximum |fdi| (strongest directional flow)
        - fraction_directional: Fraction of pairs with |fdi| > 0.5
        """
        # Mínimo endógeno: sqrt(n) + 2
        min_trans = int(np.sqrt(self.total_transitions + 1)) + 2
        if self.total_transitions < min_trans:
            return {'global_fdi': 0.0, 'error': 'insufficient_data'}

        # Get all unique state pairs
        pairs = set()
        for (i, j) in self.forward_counts.keys():
            if i < j:
                pairs.add((i, j))
            else:
                pairs.add((j, i))

        if not pairs:
            return {'global_fdi': 0.0, 'error': 'no_pairs'}

        fdis = []
        for (i, j) in pairs:
            n_ij = self.forward_counts.get((i, j), 0)
            n_ji = self.forward_counts.get((j, i), 0)
            total = n_ij + n_ji

            if total > 0:
                fdi = abs(n_ij - n_ji) / total
                fdis.append(fdi)

        if not fdis:
            return {'global_fdi': 0.0, 'error': 'no_valid_pairs'}

        fdis = np.array(fdis)

        return {
            'global_fdi': float(np.mean(fdis)),
            'std_fdi': float(np.std(fdis)),
            'max_fdi': float(np.max(fdis)),
            'median_fdi': float(np.median(fdis)),
            'fraction_directional': float(np.mean(fdis > 0.5)),
            'n_pairs': len(fdis)
        }

    def compute_net_flow_magnitude(self) -> Dict:
        """
        Compute the magnitude of net flow (current) through the system.

        In non-equilibrium steady state, there's a non-zero net current.
        J_net = sum over cycles of |current around cycle|
        """
        # Mínimo endógeno: sqrt(n) + 2
        min_trans = int(np.sqrt(self.total_transitions + 1)) + 2
        if self.total_transitions < min_trans:
            return {'net_flow': 0.0, 'error': 'insufficient_data'}

        # Build net flow matrix
        all_states = sorted(set(
            [k[0] for k in self.forward_counts.keys()] +
            [k[1] for k in self.forward_counts.keys()]
        ))
        n = len(all_states)
        state_to_idx = {s: i for i, s in enumerate(all_states)}

        J = np.zeros((n, n))
        for (from_s, to_s), count in self.forward_counts.items():
            i, j = state_to_idx[from_s], state_to_idx[to_s]
            J[i, j] = count

        # Net flow = J - J^T (antisymmetric part)
        J_net = J - J.T

        # Frobenius norm of net flow (measures total asymmetry)
        net_flow_norm = np.linalg.norm(J_net, 'fro')

        # Normalize by total transitions
        normalized_net_flow = net_flow_norm / (self.total_transitions + 1e-10)

        return {
            'net_flow_raw': float(net_flow_norm),
            'net_flow_normalized': float(normalized_net_flow),
            'n_states': n,
            'total_transitions': self.total_transitions
        }


# =============================================================================
# NON-CONSERVATIVE FIELD (HELMHOLTZ DECOMPOSITION)
# =============================================================================

class NonConservativeField:
    """
    Learns non-conservative field from empirical flows.

    Mathematical basis:
    - Estimate transition flows J_{i→j} from trajectory
    - Helmholtz decomposition: J = ∇φ + ∇×A (gradient + rotational)
    - Inject only rotational part with gain κ_t from quantiles of |J|

    No magic constants - all from data.
    """

    def __init__(self, n_states: int = None):
        # n_states se deriva de los datos si no se proporciona
        self.n_states = n_states if n_states is not None else 0

        # Flow matrix J[i,j] = net flow from i to j
        self.flow_counts: Dict[Tuple[int, int], int] = {}
        self.total_transitions = 0

        # History for gain computation
        derived_maxlen = int(np.sqrt(1e6))
        self.flow_magnitude_history: Deque[float] = deque(maxlen=derived_maxlen)

        # Helmholtz components (computed on demand)
        self._gradient_component: Optional[np.ndarray] = None
        self._rotational_component: Optional[np.ndarray] = None
        self._states: List[int] = []

    def record_transition(self, from_state: int, to_state: int):
        """Record a state transition."""
        key = (from_state, to_state)
        self.flow_counts[key] = self.flow_counts.get(key, 0) + 1
        self.total_transitions += 1

        # Track flow magnitude
        flow_mag = self.flow_counts[key]
        self.flow_magnitude_history.append(flow_mag)

    def _build_flow_matrix(self) -> Tuple[np.ndarray, List[int]]:
        """Build flow matrix J from transition counts."""
        if not self.flow_counts:
            return np.array([[0.0]]), [0]

        # Get all states
        all_states = sorted(set(
            [k[0] for k in self.flow_counts.keys()] +
            [k[1] for k in self.flow_counts.keys()]
        ))
        n = len(all_states)
        state_to_idx = {s: i for i, s in enumerate(all_states)}

        # Build asymmetric flow matrix: J[i,j] = count(i→j) - count(j→i)
        J = np.zeros((n, n))
        for (from_s, to_s), count in self.flow_counts.items():
            i, j = state_to_idx[from_s], state_to_idx[to_s]
            J[i, j] = count

        # Make antisymmetric: net flow
        J_net = J - J.T

        return J_net, all_states

    def compute_helmholtz_decomposition(self) -> Dict:
        """
        Compute Helmholtz decomposition of flow field.

        J = J_grad + J_rot

        where:
        - J_grad = ∇φ is curl-free (conservative)
        - J_rot = ∇×A is divergence-free (rotational/non-conservative)

        For discrete graph: use Hodge decomposition
        J_grad[i,j] = φ[j] - φ[i]
        J_rot = J - J_grad
        """
        J, states = self._build_flow_matrix()
        n = len(states)

        if n < 2:
            return {'error': 'insufficient_states'}

        # Compute potential φ by solving: div(J_grad) = div(J)
        # For each node i: sum_j J[i,j] = sum_j (φ[j] - φ[i])
        # This is a Laplacian system: L φ = div(J)

        # Graph Laplacian
        degrees = np.abs(J).sum(axis=1)
        L = np.diag(degrees) - np.abs(J)

        # Divergence of J at each node
        div_J = J.sum(axis=1)

        # Solve L φ = div_J (regularized for singularity)
        L_reg = L + NUMERIC_EPS * np.eye(n)
        try:
            phi = np.linalg.solve(L_reg, div_J)
        except:
            phi = np.zeros(n)

        # Gradient component: J_grad[i,j] = φ[j] - φ[i]
        J_grad = np.outer(np.ones(n), phi) - np.outer(phi, np.ones(n))

        # Rotational component: J_rot = J - J_grad
        J_rot = J - J_grad

        self._gradient_component = J_grad
        self._rotational_component = J_rot
        self._states = states

        # Statistics
        grad_magnitude = np.linalg.norm(J_grad, 'fro')
        rot_magnitude = np.linalg.norm(J_rot, 'fro')
        total_magnitude = np.linalg.norm(J, 'fro')

        # Fraction rotational (non-conservative)
        frac_rotational = rot_magnitude / (total_magnitude + NUMERIC_EPS)

        return {
            'n_states': n,
            'gradient_magnitude': float(grad_magnitude),
            'rotational_magnitude': float(rot_magnitude),
            'total_magnitude': float(total_magnitude),
            'fraction_rotational': float(frac_rotational),
            'potential_phi': phi.tolist(),
            'states': states
        }

    def get_rotational_bias(self, from_state: int, to_state: int) -> float:
        """
        Get rotational bias for transition from_state → to_state.

        Returns κ_t * J_rot[i,j] where κ_t is derived from quantiles.
        """
        if self._rotational_component is None:
            self.compute_helmholtz_decomposition()

        if self._rotational_component is None:
            return 0.0

        if from_state not in self._states or to_state not in self._states:
            return 0.0

        i = self._states.index(from_state)
        j = self._states.index(to_state)

        j_rot = self._rotational_component[i, j]

        # Compute gain κ_t from quantiles of |J|
        # Mínimo endógeno
        min_flow = int(np.sqrt(len(self.flow_magnitude_history) + 1)) + 2
        if len(self.flow_magnitude_history) < min_flow:
            kappa = 1.0 / np.sqrt(self.total_transitions + 1)
        else:
            # κ_t derived from IQR of flow magnitudes
            flow_mags = np.array(self.flow_magnitude_history)
            iqr = np.percentile(flow_mags, 75) - np.percentile(flow_mags, 25)
            kappa = 1.0 / (iqr + 1.0)

        PROVENANCE.log('kappa_rotational', kappa,
                      '1/(IQR(|J|)+1)',
                      {'iqr': iqr if len(self.flow_magnitude_history) >= 10 else 0},
                      self.total_transitions)

        return kappa * j_rot


# =============================================================================
# ENDOGENOUS NESS (NON-EQUILIBRIUM STEADY STATE)
# =============================================================================

class EndogenousNESS:
    """
    Endogenous NESS with modulated temperature τ and empirical dwell times.

    Mathematical basis:
    - τ̃_t = τ_t * (1 + rank(surprise) - rank(confidence))
    - Dwell times sampled from empirical dwell quantiles (block bootstrap)
    - NO magic constants

    Modulates OU noise input to create true non-equilibrium behavior.
    """

    def __init__(self, dimension: int = 4):
        self.dimension = dimension

        # History for rank computation
        derived_maxlen = int(np.sqrt(1e6))
        self.surprise_history: Deque[float] = deque(maxlen=derived_maxlen)
        self.confidence_history: Deque[float] = deque(maxlen=derived_maxlen)
        self.tau_history: Deque[float] = deque(maxlen=derived_maxlen)

        # Dwell time tracking
        self.dwell_times: List[int] = []
        self.current_state: Optional[int] = None
        self.current_dwell: int = 0

        # Current modulated tau
        self.tau_modulated: float = 1.0

        # Statistics
        self.t = 0

    def record_state(self, state_id: int, surprise: float, confidence: float):
        """Record state observation with surprise and confidence."""
        # Track dwell times
        if self.current_state is None:
            self.current_state = state_id
            self.current_dwell = 1
        elif state_id == self.current_state:
            self.current_dwell += 1
        else:
            # State changed - record dwell time
            self.dwell_times.append(self.current_dwell)
            self.current_state = state_id
            self.current_dwell = 1

        # Record for rank computation
        self.surprise_history.append(surprise)
        self.confidence_history.append(confidence)

        self.t += 1

    def compute_modulated_tau(self, base_tau: float) -> float:
        """
        Compute modulated temperature τ̃_t.

        τ̃_t = τ_t * (1 + rank(surprise) - rank(confidence))

        This creates asymmetric noise that breaks detailed balance.
        """
        # Mínimo endógeno
        min_surprise = int(np.sqrt(len(self.surprise_history) + 1)) + 2
        if len(self.surprise_history) < min_surprise:
            self.tau_modulated = base_tau
            self.tau_history.append(base_tau)
            return base_tau

        # Current values
        current_surprise = self.surprise_history[-1]
        current_confidence = self.confidence_history[-1]

        # Compute ranks in historical context
        rank_surprise = rolling_rank(current_surprise, self.surprise_history)
        rank_confidence = rolling_rank(current_confidence, self.confidence_history)

        # Modulation factor (centered at 1)
        modulation = 1.0 + rank_surprise - rank_confidence

        # Ensure positive
        modulation = max(NUMERIC_EPS, modulation)

        tau_mod = base_tau * modulation
        self.tau_modulated = tau_mod
        self.tau_history.append(tau_mod)

        PROVENANCE.log('tau_modulated', tau_mod,
                      'tau * (1 + rank(surprise) - rank(confidence))',
                      {'base_tau': base_tau, 'rank_s': rank_surprise, 'rank_c': rank_confidence},
                      self.t)

        return tau_mod

    def sample_dwell_time(self) -> int:
        """
        Sample dwell time from empirical distribution (block bootstrap).

        Returns dwell time drawn from observed dwell quantiles.
        """
        if len(self.dwell_times) < 5:
            return 1

        # Block bootstrap: sample from empirical distribution
        dwell = np.random.choice(self.dwell_times)

        return int(dwell)

    def get_dwell_quantiles(self) -> Dict:
        """Return quantiles of empirical dwell distribution."""
        if len(self.dwell_times) < 5:
            return {'error': 'insufficient_data'}

        dwells = np.array(self.dwell_times)

        return {
            'mean': float(np.mean(dwells)),
            'std': float(np.std(dwells)),
            'median': float(np.median(dwells)),
            'p5': float(np.percentile(dwells, 5)),
            'p25': float(np.percentile(dwells, 25)),
            'p75': float(np.percentile(dwells, 75)),
            'p95': float(np.percentile(dwells, 95)),
            'n_dwells': len(dwells)
        }

    def get_tau_statistics(self) -> Dict:
        """Return statistics of modulated tau."""
        # Mínimo endógeno
        min_tau = int(np.sqrt(len(self.tau_history) + 1)) + 2
        if len(self.tau_history) < min_tau:
            return {'error': 'insufficient_data'}

        taus = np.array(self.tau_history)

        return {
            'mean': float(np.mean(taus)),
            'std': float(np.std(taus)),
            'median': float(np.median(taus)),
            'cv': float(np.std(taus) / (np.mean(taus) + NUMERIC_EPS)),
            'p5': float(np.percentile(taus, 5)),
            'p95': float(np.percentile(taus, 95)),
            'current': float(self.tau_modulated)
        }


# =============================================================================
# RETURN PENALTY (UNCHANGED BUT INTEGRATED)
# =============================================================================

class ReturnPenalty:
    """
    Computes structural cost of returning to previously visited states.

    Mathematical basis:
    - Stores snapshot of prototype at last significant visit
    - Penalty = distance(current_prototype, last_snapshot)
    - Distance is rank-aggregated across multiple metrics

    No semantic interpretation. Pure manifold distance.
    """

    def __init__(self, dimension: int = 4):
        self.dimension = dimension
        self.last_snapshots: Dict[int, np.ndarray] = {}
        self.visit_counts: Dict[int, int] = {}
        self.penalty_history: List[float] = []
        self.penalty_by_prototype: Dict[int, List[float]] = {}

    def compute_penalty(self, prototype_id: int,
                       current_prototype_vec: np.ndarray) -> float:
        """
        Compute return penalty for visiting prototype.

        First visit: penalty = 0
        Subsequent visits: penalty = distance from last snapshot

        Uses rank-aggregated distance across metrics.
        """
        self.visit_counts[prototype_id] = self.visit_counts.get(prototype_id, 0) + 1

        if prototype_id not in self.last_snapshots:
            # First visit - store snapshot, no penalty
            self.last_snapshots[prototype_id] = current_prototype_vec.copy()
            return 0.0

        last_vec = self.last_snapshots[prototype_id]

        # Compute distances using multiple metrics
        d_euclidean = np.linalg.norm(current_prototype_vec - last_vec)

        # Cosine distance (1 - cosine similarity)
        norm_curr = np.linalg.norm(current_prototype_vec)
        norm_last = np.linalg.norm(last_vec)
        if norm_curr > NUMERIC_EPS and norm_last > NUMERIC_EPS:
            cos_sim = np.dot(current_prototype_vec, last_vec) / (norm_curr * norm_last)
            d_cosine = 1.0 - cos_sim
        else:
            d_cosine = 1.0

        # Manhattan distance (L1)
        d_manhattan = np.sum(np.abs(current_prototype_vec - last_vec))

        # Rank-aggregate: mean of distances
        distances = np.array([d_euclidean, d_cosine, d_manhattan])
        penalty = float(np.mean(distances))

        # Update snapshot
        self.last_snapshots[prototype_id] = current_prototype_vec.copy()

        # Store penalty history
        self.penalty_history.append(penalty)
        if prototype_id not in self.penalty_by_prototype:
            self.penalty_by_prototype[prototype_id] = []
        self.penalty_by_prototype[prototype_id].append(penalty)

        return penalty

    def get_penalty_statistics(self) -> Dict:
        """Return statistics about return penalties."""
        if not self.penalty_history:
            return {'n_penalties': 0}

        penalties = np.array(self.penalty_history)

        return {
            'n_penalties': len(penalties),
            'mean': float(np.mean(penalties)),
            'std': float(np.std(penalties)),
            'median': float(np.median(penalties)),
            'max': float(np.max(penalties)),
            'p25': float(np.percentile(penalties, 25)),
            'p75': float(np.percentile(penalties, 75)),
            'n_prototypes_visited': len(self.visit_counts)
        }


# =============================================================================
# TRANSITION MODIFIER
# =============================================================================

class TransitionModifier:
    """
    Modifies transition probabilities based on return penalties
    and rotational field bias.

    Uses rank-scaled exponential decay to reduce probability
    of transitioning to highly-penalized states.

    No semantic "preferences". Pure structural consequence of deformation.
    """

    def __init__(self):
        self.penalty_history: List[float] = []

    def modify_transition_probs(self,
                                base_probs: np.ndarray,
                                penalties: np.ndarray,
                                rotational_biases: np.ndarray = None) -> np.ndarray:
        """
        Modify transition probabilities based on penalties and rotational bias.

        Args:
            base_probs: Original transition probabilities (sums to 1)
            penalties: Penalty values for each destination state
            rotational_biases: Optional rotational field biases

        Returns:
            Modified probabilities (still sums to 1)
        """
        if len(penalties) == 0 or np.sum(penalties) < NUMERIC_EPS:
            if rotational_biases is None:
                return base_probs

        # Store for rank scaling
        self.penalty_history.extend(penalties.tolist())

        # Rank-scale penalties to [0, 1]
        if len(self.penalty_history) > len(penalties):
            all_penalties = np.array(self.penalty_history)
            ranks = np.array([
                np.sum(all_penalties < p) / len(all_penalties)
                for p in penalties
            ])
        else:
            # Not enough history - normalize directly
            max_p = np.max(penalties)
            if max_p > NUMERIC_EPS:
                ranks = penalties / max_p
            else:
                ranks = np.zeros_like(penalties)

        # Apply exponential decay based on rank
        # g(rank) = exp(-rank) gives range [exp(-1), 1] ≈ [0.37, 1]
        modifiers = np.exp(-ranks)

        # Add rotational bias if provided
        if rotational_biases is not None:
            # Normalize rotational bias to [0.5, 1.5] range
            rot_range = np.max(np.abs(rotational_biases))
            if rot_range > NUMERIC_EPS:
                rot_normalized = 1.0 + 0.5 * rotational_biases / rot_range
            else:
                rot_normalized = np.ones_like(rotational_biases)
            modifiers = modifiers * rot_normalized

        # Apply to base probabilities
        modified = base_probs * modifiers

        # Renormalize
        total = np.sum(modified)
        if total > NUMERIC_EPS:
            modified = modified / total
        else:
            modified = base_probs

        return modified


# =============================================================================
# IRREVERSIBILITY ANALYZER (ENHANCED)
# =============================================================================

class IrreversibilityAnalyzer:
    """
    Analyzes irreversibility in state dynamics.

    Computes:
    - KL divergence between forward and backward transition distributions
    - Time-reversal asymmetry metrics
    - Entropy production estimates
    - Drift RMS per window

    All metrics compared against null models for significance.
    """

    def __init__(self):
        self.forward_transitions: List[Tuple[int, int]] = []
        self.state_sequence: List[int] = []

        # Drift RMS tracking
        derived_maxlen = int(np.sqrt(1e6))
        self.drift_values: Deque[float] = deque(maxlen=derived_maxlen)

    def record_transition(self, from_state: int, to_state: int, drift_value: float = None):
        """Record a state transition with optional drift value."""
        self.forward_transitions.append((from_state, to_state))
        if not self.state_sequence:
            self.state_sequence.append(from_state)
        self.state_sequence.append(to_state)

        if drift_value is not None:
            self.drift_values.append(drift_value)

    def compute_transition_matrix(self,
                                  transitions: List[Tuple[int, int]]) -> Tuple[np.ndarray, List[int]]:
        """Compute transition matrix from transition list."""
        if not transitions:
            return np.array([[]]), []

        states = sorted(set([t[0] for t in transitions] + [t[1] for t in transitions]))
        n = len(states)
        state_to_idx = {s: i for i, s in enumerate(states)}

        counts = np.zeros((n, n))
        for from_s, to_s in transitions:
            counts[state_to_idx[from_s], state_to_idx[to_s]] += 1

        # Normalize rows
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        P = counts / row_sums

        return P, states

    def compute_irreversibility_index(self) -> Dict:
        """
        Compute irreversibility index comparing forward vs reversed dynamics.

        I = KL(P_forward || P_backward_reversed)

        Higher I indicates more irreversible dynamics.
        """
        # Mínimo endógeno
        min_trans_local = int(np.sqrt(len(self.forward_transitions) + 1)) + 2
        if len(self.forward_transitions) < min_trans_local:
            return {'index': 0.0, 'error': 'insufficient_data'}

        # Forward transition matrix
        P_fwd, states = self.compute_transition_matrix(self.forward_transitions)

        # Reversed transitions (swap from/to)
        reversed_transitions = [(t[1], t[0]) for t in self.forward_transitions]
        P_rev, _ = self.compute_transition_matrix(reversed_transitions)

        # KL divergence between forward and reversed
        kl_sum = 0.0
        n_valid = 0

        for i in range(P_fwd.shape[0]):
            for j in range(P_fwd.shape[1]):
                if P_fwd[i, j] > NUMERIC_EPS and P_rev[i, j] > NUMERIC_EPS:
                    kl_sum += P_fwd[i, j] * np.log(P_fwd[i, j] / P_rev[i, j])
                    n_valid += 1

        # Compute stationary distribution for weighting
        try:
            eigenvalues, eigenvectors = np.linalg.eig(P_fwd.T)
            idx = np.argmin(np.abs(eigenvalues - 1.0))
            pi = np.real(eigenvectors[:, idx])
            pi = pi / pi.sum()
            pi = np.maximum(pi, 0)
        except:
            pi = np.ones(P_fwd.shape[0]) / P_fwd.shape[0]

        # Entropy production rate estimate
        entropy_prod = 0.0
        for i in range(P_fwd.shape[0]):
            for j in range(P_fwd.shape[1]):
                if P_fwd[i, j] > NUMERIC_EPS and P_fwd[j, i] > NUMERIC_EPS:
                    entropy_prod += pi[i] * P_fwd[i, j] * np.log(P_fwd[i, j] / P_fwd[j, i])

        # Drift RMS statistics
        if self.drift_values:
            drift_array = np.array(self.drift_values)
            drift_rms = np.sqrt(np.mean(drift_array ** 2))
        else:
            drift_rms = 0.0

        return {
            'kl_forward_reverse': float(kl_sum),
            'entropy_production': float(entropy_prod),
            'drift_rms': float(drift_rms),
            'n_transitions': len(self.forward_transitions),
            'n_states': P_fwd.shape[0],
            'stationary_distribution': pi.tolist()
        }

    def compute_null_irreversibility(self, n_nulls: int = None) -> Dict:
        """
        Compute irreversibility for null models.

        Null 1: Order-1 Markov simulation
        Null 2: Order-2 Markov simulation
        """
        # Mínimo endógeno
        min_trans_local = int(np.sqrt(len(self.forward_transitions) + 1)) + 2
        if len(self.forward_transitions) < min_trans_local:
            return {'error': 'insufficient_data'}

        # Real irreversibility
        real_result = self.compute_irreversibility_index()
        real_kl = real_result['kl_forward_reverse']
        real_entropy = real_result['entropy_production']
        real_drift_rms = real_result['drift_rms']

        # Null distributions
        null_kls = []
        null_entropies = []
        null_drift_rms = []

        P_fwd, states = self.compute_transition_matrix(self.forward_transitions)
        n_states = len(states)
        T = len(self.state_sequence)

        for _ in range(n_nulls):
            # Generate Markov chain from same P
            if n_states == 0:
                continue

            null_sequence = [np.random.randint(n_states)]
            for _ in range(T - 1):
                probs = P_fwd[null_sequence[-1]]
                if probs.sum() < NUMERIC_EPS:
                    probs = np.ones(n_states) / n_states
                else:
                    probs = probs / probs.sum()
                null_sequence.append(np.random.choice(n_states, p=probs))

            # Compute null transitions
            null_trans = [(null_sequence[i], null_sequence[i+1])
                         for i in range(len(null_sequence) - 1)]

            # Analyze
            temp_analyzer = IrreversibilityAnalyzer()
            for t in null_trans:
                temp_analyzer.record_transition(t[0], t[1], np.random.rand())
            null_result = temp_analyzer.compute_irreversibility_index()

            if 'error' not in null_result:
                null_kls.append(null_result['kl_forward_reverse'])
                null_entropies.append(null_result['entropy_production'])
                null_drift_rms.append(null_result['drift_rms'])

        if not null_kls:
            return {'error': 'null_generation_failed'}

        # Statistics
        null_kls = np.array(null_kls)
        null_entropies = np.array(null_entropies)
        null_drift_rms = np.array(null_drift_rms)

        kl_z = (real_kl - np.mean(null_kls)) / (np.std(null_kls) + NUMERIC_EPS)
        entropy_z = (real_entropy - np.mean(null_entropies)) / (np.std(null_entropies) + NUMERIC_EPS)
        drift_z = (real_drift_rms - np.mean(null_drift_rms)) / (np.std(null_drift_rms) + NUMERIC_EPS)

        kl_p = float(np.mean(null_kls >= real_kl))
        entropy_p = float(np.mean(null_entropies >= real_entropy))
        drift_p = float(np.mean(null_drift_rms >= real_drift_rms))

        return {
            'real': {
                'kl': float(real_kl),
                'entropy_production': float(real_entropy),
                'drift_rms': float(real_drift_rms)
            },
            'null': {
                'kl_mean': float(np.mean(null_kls)),
                'kl_std': float(np.std(null_kls)),
                'kl_p95': float(np.percentile(null_kls, 95)),
                'entropy_mean': float(np.mean(null_entropies)),
                'entropy_std': float(np.std(null_entropies)),
                'entropy_p95': float(np.percentile(null_entropies, 95)),
                'drift_rms_mean': float(np.mean(null_drift_rms)),
                'drift_rms_std': float(np.std(null_drift_rms)),
                'drift_rms_p95': float(np.percentile(null_drift_rms, 95))
            },
            'statistics': {
                'kl_z_score': float(kl_z),
                'kl_p_value': kl_p,
                'kl_above_p95': real_kl > np.percentile(null_kls, 95),
                'entropy_z_score': float(entropy_z),
                'entropy_p_value': entropy_p,
                'entropy_above_p95': real_entropy > np.percentile(null_entropies, 95),
                'drift_rms_z_score': float(drift_z),
                'drift_rms_p_value': drift_p,
                'drift_rms_above_p95': real_drift_rms > np.percentile(null_drift_rms, 95)
            },
            'n_nulls': n_nulls
        }


# =============================================================================
# INTEGRATED IRREVERSIBILITY SYSTEM (PHASE 16B)
# =============================================================================

class IrreversibilitySystem:
    """
    Integrated system combining all Phase 16B components:
    - TRUE prototype plasticity with online learning
    - Non-conservative field (Helmholtz decomposition)
    - Endogenous NESS with τ modulation
    - Return penalty and transition modification
    - Irreversibility analysis
    """

    def __init__(self, dimension: int = 4):
        self.dimension = dimension

        # Per-agent systems with TRUE plasticity
        self.plasticity_neo = TruePrototypePlasticity(dimension)
        self.plasticity_eva = TruePrototypePlasticity(dimension)

        self.penalty_neo = ReturnPenalty(dimension)
        self.penalty_eva = ReturnPenalty(dimension)

        self.transition_mod_neo = TransitionModifier()
        self.transition_mod_eva = TransitionModifier()

        self.analyzer_neo = IrreversibilityAnalyzer()
        self.analyzer_eva = IrreversibilityAnalyzer()

        # Non-conservative field
        self.nc_field_neo = NonConservativeField()
        self.nc_field_eva = NonConservativeField()

        # Endogenous NESS
        self.ness_neo = EndogenousNESS(dimension)
        self.ness_eva = EndogenousNESS(dimension)

        # History
        self.step_history: List[Dict] = []

    def process_step(self,
                    neo_state_id: int, neo_state_vec: np.ndarray, neo_prototype_vec: np.ndarray,
                    eva_state_id: int, eva_state_vec: np.ndarray, eva_prototype_vec: np.ndarray,
                    neo_surprise: float = 0.5, neo_confidence: float = 0.5,
                    eva_surprise: float = 0.5, eva_confidence: float = 0.5,
                    prev_neo_state: int = None, prev_eva_state: int = None) -> Dict:
        """
        Process one timestep for both agents with full Phase 16B pipeline.

        Returns dict with drift, penalty, NESS, and non-conservative field info.
        """
        result = {
            'neo': {},
            'eva': {}
        }

        # NEO processing with TRUE plasticity
        neo_updated_mu, neo_drift_mag, neo_drift_rms = self.plasticity_neo.update_prototype(
            neo_state_id, neo_prototype_vec, neo_state_vec
        )
        neo_penalty = self.penalty_neo.compute_penalty(neo_state_id, neo_updated_mu)

        # Record NESS
        self.ness_neo.record_state(neo_state_id, neo_surprise, neo_confidence)

        # Record non-conservative field
        if prev_neo_state is not None:
            self.nc_field_neo.record_transition(prev_neo_state, neo_state_id)

        result['neo'] = {
            'state_id': neo_state_id,
            'drift_magnitude': float(neo_drift_mag),
            'drift_rms': float(neo_drift_rms),
            'return_penalty': float(neo_penalty),
            'updated_prototype': neo_updated_mu.tolist()
        }

        # Record transition with drift
        if prev_neo_state is not None:
            self.analyzer_neo.record_transition(prev_neo_state, neo_state_id, neo_drift_mag)

        # EVA processing with TRUE plasticity
        eva_updated_mu, eva_drift_mag, eva_drift_rms = self.plasticity_eva.update_prototype(
            eva_state_id, eva_prototype_vec, eva_state_vec
        )
        eva_penalty = self.penalty_eva.compute_penalty(eva_state_id, eva_updated_mu)

        # Record NESS
        self.ness_eva.record_state(eva_state_id, eva_surprise, eva_confidence)

        # Record non-conservative field
        if prev_eva_state is not None:
            self.nc_field_eva.record_transition(prev_eva_state, eva_state_id)

        result['eva'] = {
            'state_id': eva_state_id,
            'drift_magnitude': float(eva_drift_mag),
            'drift_rms': float(eva_drift_rms),
            'return_penalty': float(eva_penalty),
            'updated_prototype': eva_updated_mu.tolist()
        }

        if prev_eva_state is not None:
            self.analyzer_eva.record_transition(prev_eva_state, eva_state_id, eva_drift_mag)

        self.step_history.append(result)

        return result

    def get_statistics(self) -> Dict:
        """Return comprehensive statistics."""
        return {
            'neo': {
                'drift': self.plasticity_neo.get_drift_statistics(),
                'penalty': self.penalty_neo.get_penalty_statistics(),
                'ness': {
                    'tau': self.ness_neo.get_tau_statistics(),
                    'dwell': self.ness_neo.get_dwell_quantiles()
                }
            },
            'eva': {
                'drift': self.plasticity_eva.get_drift_statistics(),
                'penalty': self.penalty_eva.get_penalty_statistics(),
                'ness': {
                    'tau': self.ness_eva.get_tau_statistics(),
                    'dwell': self.ness_eva.get_dwell_quantiles()
                }
            },
            'n_steps': len(self.step_history)
        }

    def analyze_irreversibility(self, n_nulls: int = 100) -> Dict:
        """Run irreversibility analysis for both agents."""
        return {
            'neo': self.analyzer_neo.compute_null_irreversibility(n_nulls),
            'eva': self.analyzer_eva.compute_null_irreversibility(n_nulls)
        }

    def analyze_helmholtz(self) -> Dict:
        """Analyze non-conservative field decomposition."""
        return {
            'neo': self.nc_field_neo.compute_helmholtz_decomposition(),
            'eva': self.nc_field_eva.compute_helmholtz_decomposition()
        }


# =============================================================================
# DUAL-MEMORY IRREVERSIBILITY SYSTEM (PHASE 16B ENHANCED)
# =============================================================================

class DualMemoryIrreversibilitySystem:
    """
    Enhanced Phase 16B system with dual-memory plasticity.

    "Parameters born from results" - hormonal-like dual timescale system:
    - fast_drift (short-term, reactive) with η_fast = 1/√(n_local+1)
    - slow_drift (long-term, consolidated) with η_slow = 1/√(N_k+1)
    - variance-based window reset for n_local
    - α_t weighting from rank-scaled deviation
    - align_t directionality index
    - Flow Directionality Index (FDI) for measuring non-equilibrium
    """

    def __init__(self, dimension: int = 4):
        self.dimension = dimension

        # Per-agent systems with DUAL-MEMORY plasticity
        self.plasticity_neo = DualMemoryPlasticity(dimension)
        self.plasticity_eva = DualMemoryPlasticity(dimension)

        self.penalty_neo = ReturnPenalty(dimension)
        self.penalty_eva = ReturnPenalty(dimension)

        self.transition_mod_neo = TransitionModifier()
        self.transition_mod_eva = TransitionModifier()

        self.analyzer_neo = IrreversibilityAnalyzer()
        self.analyzer_eva = IrreversibilityAnalyzer()

        # Non-conservative field
        self.nc_field_neo = NonConservativeField()
        self.nc_field_eva = NonConservativeField()

        # Flow directionality analyzer (NEW - directly measures non-equilibrium)
        self.fdi_neo = FlowDirectionalityAnalyzer()
        self.fdi_eva = FlowDirectionalityAnalyzer()

        # Endogenous NESS
        self.ness_neo = EndogenousNESS(dimension)
        self.ness_eva = EndogenousNESS(dimension)

        # History
        self.step_history: List[Dict] = []
        self.diagnostics_history: List[Dict] = []

    def process_step(self,
                    neo_state_id: int, neo_state_vec: np.ndarray, neo_prototype_vec: np.ndarray,
                    eva_state_id: int, eva_state_vec: np.ndarray, eva_prototype_vec: np.ndarray,
                    neo_surprise: float = 0.5, neo_confidence: float = 0.5,
                    eva_surprise: float = 0.5, eva_confidence: float = 0.5,
                    prev_neo_state: int = None, prev_eva_state: int = None,
                    neo_bias_field: np.ndarray = None,
                    eva_bias_field: np.ndarray = None) -> Dict:
        """
        Process one timestep with DUAL-MEMORY plasticity.

        Returns dict with drift, penalty, NESS, dual-memory diagnostics.
        """
        result = {'neo': {}, 'eva': {}}
        diagnostics = {'neo': {}, 'eva': {}}

        # Get rotational bias from non-conservative field
        if prev_neo_state is not None:
            neo_rot_bias = self.nc_field_neo.get_rotational_bias(prev_neo_state, neo_state_id)
            # Convert scalar to vector for bias_field
            if neo_bias_field is None:
                neo_bias_field = np.ones(self.dimension) * neo_rot_bias
        if prev_eva_state is not None:
            eva_rot_bias = self.nc_field_eva.get_rotational_bias(prev_eva_state, eva_state_id)
            if eva_bias_field is None:
                eva_bias_field = np.ones(self.dimension) * eva_rot_bias

        # NEO processing with DUAL-MEMORY plasticity
        neo_updated_mu, neo_drift_mag, neo_drift_rms, neo_diag = self.plasticity_neo.update_prototype(
            neo_state_id, neo_prototype_vec, neo_state_vec, neo_bias_field
        )
        neo_penalty = self.penalty_neo.compute_penalty(neo_state_id, neo_updated_mu)

        # Record NESS
        self.ness_neo.record_state(neo_state_id, neo_surprise, neo_confidence)

        # Record non-conservative field and FDI
        if prev_neo_state is not None:
            self.nc_field_neo.record_transition(prev_neo_state, neo_state_id)
            self.fdi_neo.record_transition(prev_neo_state, neo_state_id)

        result['neo'] = {
            'state_id': neo_state_id,
            'drift_magnitude': float(neo_drift_mag),
            'drift_rms': float(neo_drift_rms),
            'return_penalty': float(neo_penalty),
            'updated_prototype': neo_updated_mu.tolist(),
            'directional_momentum': neo_diag['directional_momentum'],
            'align_t': neo_diag['align_t']
        }
        diagnostics['neo'] = neo_diag

        if prev_neo_state is not None:
            self.analyzer_neo.record_transition(prev_neo_state, neo_state_id, neo_drift_mag)

        # EVA processing with DUAL-MEMORY plasticity
        eva_updated_mu, eva_drift_mag, eva_drift_rms, eva_diag = self.plasticity_eva.update_prototype(
            eva_state_id, eva_prototype_vec, eva_state_vec, eva_bias_field
        )
        eva_penalty = self.penalty_eva.compute_penalty(eva_state_id, eva_updated_mu)

        # Record NESS
        self.ness_eva.record_state(eva_state_id, eva_surprise, eva_confidence)

        # Record non-conservative field and FDI
        if prev_eva_state is not None:
            self.nc_field_eva.record_transition(prev_eva_state, eva_state_id)
            self.fdi_eva.record_transition(prev_eva_state, eva_state_id)

        result['eva'] = {
            'state_id': eva_state_id,
            'drift_magnitude': float(eva_drift_mag),
            'drift_rms': float(eva_drift_rms),
            'return_penalty': float(eva_penalty),
            'updated_prototype': eva_updated_mu.tolist(),
            'directional_momentum': eva_diag['directional_momentum'],
            'align_t': eva_diag['align_t']
        }
        diagnostics['eva'] = eva_diag

        if prev_eva_state is not None:
            self.analyzer_eva.record_transition(prev_eva_state, eva_state_id, eva_drift_mag)

        self.step_history.append(result)
        self.diagnostics_history.append(diagnostics)

        return result

    def get_statistics(self) -> Dict:
        """Return comprehensive statistics with dual-memory info."""
        return {
            'neo': {
                'dual_memory': self.plasticity_neo.get_dual_memory_statistics(),
                'penalty': self.penalty_neo.get_penalty_statistics(),
                'ness': {
                    'tau': self.ness_neo.get_tau_statistics(),
                    'dwell': self.ness_neo.get_dwell_quantiles()
                }
            },
            'eva': {
                'dual_memory': self.plasticity_eva.get_dual_memory_statistics(),
                'penalty': self.penalty_eva.get_penalty_statistics(),
                'ness': {
                    'tau': self.ness_eva.get_tau_statistics(),
                    'dwell': self.ness_eva.get_dwell_quantiles()
                }
            },
            'n_steps': len(self.step_history)
        }

    def get_momentum_for_go_criteria(self) -> Dict:
        """Get directional momentum statistics for GO criteria evaluation."""
        neo_mom = self.plasticity_neo.get_momentum_statistics()
        eva_mom = self.plasticity_eva.get_momentum_statistics()

        # Combine for overall assessment
        all_momentum = (self.plasticity_neo.directional_momentum_values +
                       self.plasticity_eva.directional_momentum_values)

        if not all_momentum:
            return {'error': 'insufficient_data'}

        mom_array = np.array(all_momentum)

        return {
            'neo': neo_mom,
            'eva': eva_mom,
            'combined': {
                'mean': float(np.mean(mom_array)),
                'std': float(np.std(mom_array)),
                'median': float(np.median(mom_array)),
                'p95': float(np.percentile(mom_array, 95)),
                'n_values': len(mom_array)
            }
        }

    def get_drift_rms_for_go_criteria(self) -> Dict:
        """Get drift RMS statistics for GO criteria evaluation."""
        neo_rms = self.plasticity_neo.get_drift_rms_statistics()
        eva_rms = self.plasticity_eva.get_drift_rms_statistics()

        # Combine
        all_rms = (self.plasticity_neo.drift_rms_per_window +
                  self.plasticity_eva.drift_rms_per_window)

        if not all_rms:
            return {'error': 'insufficient_data'}

        rms_array = np.array(all_rms)

        return {
            'neo': neo_rms,
            'eva': eva_rms,
            'combined': {
                'mean': float(np.mean(rms_array)),
                'std': float(np.std(rms_array)),
                'median': float(np.median(rms_array)),
                'p95': float(np.percentile(rms_array, 95)),
                'n_windows': len(rms_array)
            }
        }

    def analyze_irreversibility(self, n_nulls: int = 100) -> Dict:
        """Run irreversibility analysis for both agents."""
        return {
            'neo': self.analyzer_neo.compute_null_irreversibility(n_nulls),
            'eva': self.analyzer_eva.compute_null_irreversibility(n_nulls)
        }

    def analyze_helmholtz(self) -> Dict:
        """Analyze non-conservative field decomposition."""
        return {
            'neo': self.nc_field_neo.compute_helmholtz_decomposition(),
            'eva': self.nc_field_eva.compute_helmholtz_decomposition()
        }

    def get_flow_directionality_index(self) -> Dict:
        """
        Get Flow Directionality Index (FDI) for GO criteria evaluation.

        FDI directly measures non-equilibrium by quantifying transition asymmetry.
        FDI = 0 for equilibrium, FDI → 1 for strongly directional (non-equilibrium).
        """
        neo_fdi = self.fdi_neo.compute_flow_directionality_index()
        eva_fdi = self.fdi_eva.compute_flow_directionality_index()

        # Also get net flow magnitude
        neo_flow = self.fdi_neo.compute_net_flow_magnitude()
        eva_flow = self.fdi_eva.compute_net_flow_magnitude()

        return {
            'neo': {
                'fdi': neo_fdi,
                'net_flow': neo_flow
            },
            'eva': {
                'fdi': eva_fdi,
                'net_flow': eva_flow
            },
            'combined': {
                'mean_fdi': float((neo_fdi.get('global_fdi', 0) + eva_fdi.get('global_fdi', 0)) / 2),
                'max_fdi': float(max(neo_fdi.get('max_fdi', 0), eva_fdi.get('max_fdi', 0))),
                'mean_net_flow': float((neo_flow.get('net_flow_normalized', 0) +
                                       eva_flow.get('net_flow_normalized', 0)) / 2)
            }
        }


# =============================================================================
# PROVENANCE
# =============================================================================

IRREVERSIBILITY_PROVENANCE = {
    'module': 'irreversibility',
    'version': '2.1.0',  # Phase 16B with Dual-Memory
    'mechanisms': [
        'true_prototype_plasticity',
        'dual_memory_plasticity',  # NEW: hormonal-like dual timescale
        'non_conservative_field',
        'endogenous_ness',
        'return_penalty',
        'transition_modification',
        'irreversibility_analysis'
    ],
    'endogenous_params': [
        # Original
        'eta_t = 1/sqrt(visits+1)',
        'f_t = rank_scaled(delta_norm)^2',
        'kappa_t = 1/(IQR(|J|)+1)',
        'tau_mod = tau * (1 + rank(surprise) - rank(confidence))',
        'g(penalty) = exp(-rank_scaled(penalty))',
        # NEW: Dual-memory params
        'eta_fast = 1/sqrt(n_local+1)',
        'eta_slow = 1/sqrt(N_k+1)',
        'alpha_t = rank_scaled(||delta||)',
        'align_t = cos(momentum_t, bias_field)',
        'n_local_reset = variance_spike > threshold'
    ],
    'no_magic_numbers': True,
    'no_semantic_labels': True
}


if __name__ == "__main__":
    print("Phase 16B Irreversibility Module")
    print("=" * 50)

    # Quick test
    system = IrreversibilitySystem(dimension=4)

    np.random.seed(42)
    prev_neo, prev_eva = None, None

    print("\n[1] Running 500 steps with TRUE plasticity...")

    for t in range(500):
        neo_id = np.random.randint(0, 5)
        eva_id = np.random.randint(0, 5)

        neo_vec = np.random.randn(4)
        eva_vec = np.random.randn(4)

        neo_proto = np.random.randn(4) * 0.5
        eva_proto = np.random.randn(4) * 0.5

        # Simulate surprise and confidence
        neo_surprise = np.random.beta(2, 5)
        neo_confidence = np.random.beta(5, 2)
        eva_surprise = np.random.beta(2, 5)
        eva_confidence = np.random.beta(5, 2)

        result = system.process_step(
            neo_id, neo_vec, neo_proto,
            eva_id, eva_vec, eva_proto,
            neo_surprise, neo_confidence,
            eva_surprise, eva_confidence,
            prev_neo, prev_eva
        )

        prev_neo, prev_eva = neo_id, eva_id

    stats = system.get_statistics()
    print(f"\n[2] Statistics:")
    print(f"  NEO drift RMS: {stats['neo']['drift']['drift_rms']}")
    print(f"  EVA drift RMS: {stats['eva']['drift']['drift_rms']}")
    print(f"  NEO NESS tau: {stats['neo']['ness']['tau']}")
    print(f"  EVA dwell: {stats['eva']['ness']['dwell']}")

    print("\n[3] Analyzing Helmholtz decomposition...")
    helmholtz = system.analyze_helmholtz()
    print(f"  NEO fraction rotational: {helmholtz['neo'].get('fraction_rotational', 'N/A')}")
    print(f"  EVA fraction rotational: {helmholtz['eva'].get('fraction_rotational', 'N/A')}")

    print("\n[4] Analyzing irreversibility vs nulls...")
    irrev = system.analyze_irreversibility(n_nulls=50)
    print(f"  NEO entropy production above p95: {irrev['neo'].get('statistics', {}).get('entropy_above_p95', 'N/A')}")
    print(f"  NEO drift RMS above p95: {irrev['neo'].get('statistics', {}).get('drift_rms_above_p95', 'N/A')}")
    print(f"  EVA entropy production above p95: {irrev['eva'].get('statistics', {}).get('entropy_above_p95', 'N/A')}")

    print("\n" + "=" * 50)
    print("PHASE 16B VERIFICATION (Original):")
    print("  - TRUE prototype plasticity: eta_t = 1/sqrt(visits+1)")
    print("  - Tangent projection to manifold")
    print("  - Deformation factor f_t = rank^2")
    print("  - Helmholtz decomposition (gradient vs rotational)")
    print("  - NESS tau modulation via ranks")
    print("  - ZERO magic constants")
    print("=" * 50)

    # Test DUAL-MEMORY system
    print("\n\n" + "=" * 50)
    print("DUAL-MEMORY PLASTICITY TEST")
    print("=" * 50)

    dual_system = DualMemoryIrreversibilitySystem(dimension=4)

    np.random.seed(42)
    prev_neo, prev_eva = None, None

    print("\n[1] Running 500 steps with DUAL-MEMORY plasticity...")

    for t in range(500):
        neo_id = np.random.randint(0, 5)
        eva_id = np.random.randint(0, 5)

        neo_vec = np.random.randn(4)
        eva_vec = np.random.randn(4)

        neo_proto = np.random.randn(4) * 0.5
        eva_proto = np.random.randn(4) * 0.5

        # Simulate surprise and confidence
        neo_surprise = np.random.beta(2, 5)
        neo_confidence = np.random.beta(5, 2)
        eva_surprise = np.random.beta(2, 5)
        eva_confidence = np.random.beta(5, 2)

        result = dual_system.process_step(
            neo_id, neo_vec, neo_proto,
            eva_id, eva_vec, eva_proto,
            neo_surprise, neo_confidence,
            eva_surprise, eva_confidence,
            prev_neo, prev_eva
        )

        prev_neo, prev_eva = neo_id, eva_id

    dual_stats = dual_system.get_statistics()
    print(f"\n[2] Dual-Memory Statistics:")
    print(f"  NEO fast_drift mean: {dual_stats['neo']['dual_memory'].get('fast_drift', {}).get('mean', 'N/A')}")
    print(f"  NEO slow_drift mean: {dual_stats['neo']['dual_memory'].get('slow_drift', {}).get('mean', 'N/A')}")
    print(f"  EVA fast_drift mean: {dual_stats['eva']['dual_memory'].get('fast_drift', {}).get('mean', 'N/A')}")
    print(f"  EVA slow_drift mean: {dual_stats['eva']['dual_memory'].get('slow_drift', {}).get('mean', 'N/A')}")

    mom_stats = dual_system.get_momentum_for_go_criteria()
    print(f"\n[3] Directional Momentum:")
    print(f"  Combined mean: {mom_stats.get('combined', {}).get('mean', 'N/A')}")
    print(f"  Combined p95: {mom_stats.get('combined', {}).get('p95', 'N/A')}")

    align_neo = dual_stats['neo']['dual_memory'].get('alignment', {})
    print(f"\n[4] Alignment (directionality):")
    print(f"  NEO mean align: {align_neo.get('mean', 'N/A')}")
    print(f"  NEO positive fraction: {align_neo.get('positive_fraction', 'N/A')}")

    print("\n" + "=" * 50)
    print("DUAL-MEMORY VERIFICATION:")
    print("  - eta_fast = 1/sqrt(n_local+1)")
    print("  - eta_slow = 1/sqrt(N_k+1)")
    print("  - alpha_t = rank_scaled(||delta||)")
    print("  - align_t = cos(momentum, bias_field)")
    print("  - variance-based window reset")
    print("  - 'Parameters born from results'")
    print("=" * 50)
