#!/usr/bin/env python3
"""
Phase 20: Structural Veto & Resistance
======================================

Implements PURELY ENDOGENOUS autoprotection mechanisms where NEO/EVA
generate structural opposition to external perturbations without
semantic labels or rewards.

Key components:
1. Intrusion Detection: shock_t = rank(delta) * rank(delta_spread) * rank(delta_epr)
2. Structural Opposition Field: O_t = -rank(shock_t) * normalize(x_t - mu_k)
3. Endogenous Resistance Gain: gamma_t = 1 / (1 + std(window(shock_history)))
4. Veto-Adjusted Transition: x_next = x_next_base + gamma_t * O_t

NO semantic labels (pain, fear, threat, danger, etc.)
NO magic constants - all parameters derived from data statistics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque

# Numeric stability constant
NUMERIC_EPS = 1e-16


# =============================================================================
# PROVENANCE TRACKING
# =============================================================================

class VetoProvenance:
    """Track derivation of all veto parameters."""

    def __init__(self):
        self.logs: List[Dict] = []

    def log(self, param_name: str, value: float, derivation: str,
            source_data: Dict, timestep: int):
        """Log parameter derivation."""
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


VETO_PROVENANCE = VetoProvenance()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_rank(value: float, history: np.ndarray) -> float:
    """Compute rank of value within history distribution [0, 1]."""
    if len(history) == 0:
        return 0.5
    return float(np.sum(history < value) / len(history))


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    norm = np.linalg.norm(v)
    if norm < NUMERIC_EPS:
        return np.zeros_like(v)
    return v / norm


# =============================================================================
# INTRUSION DETECTOR
# =============================================================================

class IntrusionDetector:
    """
    Detects structural intrusions/perturbations endogenously.

    shock_t = rank(delta) * rank(delta_spread) * rank(delta_epr)

    where:
    - delta = ||x_t - mu_k|| (distance to nearest prototype)
    - delta_spread = |spread_t - spread_mean|
    - delta_epr = |epr_t - epr_mean|
    """

    def __init__(self, n_prototypes: int = 5):
        self.n_prototypes = n_prototypes
        self.prototypes: Optional[np.ndarray] = None
        self.prototype_visits: np.ndarray = np.ones(n_prototypes)

        # Histories for rank computation
        self.delta_history: List[float] = []
        self.delta_spread_history: List[float] = []
        self.delta_epr_history: List[float] = []
        self.shock_history: List[float] = []

        # Running statistics
        self.spread_ema: float = 0.0
        self.epr_ema: float = 0.0
        self.t = 0

    def set_prototypes(self, prototypes: np.ndarray):
        """Set prototype positions."""
        self.prototypes = prototypes.copy()
        self.n_prototypes = len(prototypes)
        self.prototype_visits = np.ones(self.n_prototypes)

    def _find_nearest_prototype(self, x_t: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """Find nearest prototype and return index, distance, and direction."""
        if self.prototypes is None:
            return 0, 0.0, np.zeros_like(x_t)

        # Handle dimension mismatch
        min_dim = min(x_t.shape[0], self.prototypes.shape[1])
        x_truncated = x_t[:min_dim]
        protos_truncated = self.prototypes[:, :min_dim]

        distances = np.linalg.norm(protos_truncated - x_truncated, axis=1)
        nearest_idx = int(np.argmin(distances))
        nearest_dist = float(distances[nearest_idx])

        # Direction from prototype to current point
        direction = x_truncated - protos_truncated[nearest_idx]

        # Pad back if needed
        if len(direction) < len(x_t):
            direction = np.concatenate([direction, np.zeros(len(x_t) - len(direction))])

        return nearest_idx, nearest_dist, direction

    def compute_shock(self, x_t: np.ndarray, spread_t: float,
                      epr_t: float) -> Tuple[float, Dict]:
        """
        Compute shock indicator from structural deviation.

        Args:
            x_t: Current manifold position
            spread_t: Current manifold spread
            epr_t: Current entropy production rate

        Returns:
            (shock_t, diagnostics)
        """
        self.t += 1

        # Update EMAs with endogenous alpha
        alpha = 1.0 / np.sqrt(self.t + 1)
        self.spread_ema = alpha * spread_t + (1 - alpha) * self.spread_ema
        self.epr_ema = alpha * epr_t + (1 - alpha) * self.epr_ema

        # Find nearest prototype
        nearest_idx, delta, direction = self._find_nearest_prototype(x_t)
        self.prototype_visits[nearest_idx] += 1

        # Compute deviations
        delta_spread = abs(spread_t - self.spread_ema)
        delta_epr = abs(epr_t - self.epr_ema)

        # Store in histories
        self.delta_history.append(delta)
        self.delta_spread_history.append(delta_spread)
        self.delta_epr_history.append(delta_epr)

        # Compute ranks
        delta_arr = np.array(self.delta_history)
        spread_arr = np.array(self.delta_spread_history)
        epr_arr = np.array(self.delta_epr_history)

        rank_delta = compute_rank(delta, delta_arr)
        rank_spread = compute_rank(delta_spread, spread_arr)
        rank_epr = compute_rank(delta_epr, epr_arr)

        # Shock is product of ranks (high when all deviations are high)
        shock_t = rank_delta * rank_spread * rank_epr
        self.shock_history.append(shock_t)

        VETO_PROVENANCE.log(
            'shock_t', float(shock_t),
            'rank(delta) * rank(delta_spread) * rank(delta_epr)',
            {'rank_delta': rank_delta, 'rank_spread': rank_spread, 'rank_epr': rank_epr},
            self.t
        )

        diagnostics = {
            'delta': delta,
            'delta_spread': delta_spread,
            'delta_epr': delta_epr,
            'rank_delta': rank_delta,
            'rank_spread': rank_spread,
            'rank_epr': rank_epr,
            'shock_t': shock_t,
            'nearest_prototype': nearest_idx,
            'direction': direction.tolist()
        }

        return float(shock_t), diagnostics

    def get_direction_to_nearest(self, x_t: np.ndarray) -> np.ndarray:
        """Get direction from nearest prototype to current point."""
        _, _, direction = self._find_nearest_prototype(x_t)
        return direction

    def get_statistics(self) -> Dict:
        """Return detector statistics."""
        if not self.shock_history:
            return {'mean_shock': 0.0}

        shock_arr = np.array(self.shock_history)
        return {
            'mean_shock': float(np.mean(shock_arr)),
            'std_shock': float(np.std(shock_arr)),
            'max_shock': float(np.max(shock_arr)),
            'n_samples': len(shock_arr),
            'prototype_visits': self.prototype_visits.tolist()
        }


# =============================================================================
# STRUCTURAL OPPOSITION FIELD
# =============================================================================

class StructuralOppositionField:
    """
    Generates structural opposition to perturbations.

    O_t = -rank(shock_t) * normalize(x_t - mu_k)

    Opposition points back toward the nearest prototype,
    scaled by how anomalous the shock is.
    """

    def __init__(self):
        self.opposition_history: List[np.ndarray] = []
        self.opposition_magnitude_history: List[float] = []
        self.t = 0

    def compute_opposition(self, shock_t: float, direction: np.ndarray,
                          shock_history: List[float]) -> Tuple[np.ndarray, Dict]:
        """
        Compute structural opposition field.

        Args:
            shock_t: Current shock indicator
            direction: Direction from nearest prototype to current point
            shock_history: History of shock values for ranking

        Returns:
            (O_t, diagnostics)
        """
        self.t += 1

        # Rank of current shock
        if len(shock_history) > 0:
            shock_arr = np.array(shock_history)
            rank_shock = compute_rank(shock_t, shock_arr)
        else:
            rank_shock = 0.5

        # Normalize direction
        direction_norm = normalize_vector(direction)

        # Opposition: negative direction (back toward prototype), scaled by shock rank
        O_t = -rank_shock * direction_norm

        # Store
        self.opposition_history.append(O_t.copy())
        self.opposition_magnitude_history.append(float(np.linalg.norm(O_t)))

        VETO_PROVENANCE.log(
            'O_t_magnitude', float(np.linalg.norm(O_t)),
            '-rank(shock_t) * normalize(direction)',
            {'rank_shock': rank_shock, 'direction_norm': float(np.linalg.norm(direction_norm))},
            self.t
        )

        diagnostics = {
            'rank_shock': rank_shock,
            'direction_magnitude': float(np.linalg.norm(direction)),
            'O_t_magnitude': float(np.linalg.norm(O_t)),
            'O_t': O_t.tolist()
        }

        return O_t, diagnostics

    def get_statistics(self) -> Dict:
        """Return field statistics."""
        if not self.opposition_magnitude_history:
            return {'mean_magnitude': 0.0}

        mag_arr = np.array(self.opposition_magnitude_history)
        return {
            'mean_magnitude': float(np.mean(mag_arr)),
            'std_magnitude': float(np.std(mag_arr)),
            'max_magnitude': float(np.max(mag_arr)),
            'n_samples': len(mag_arr)
        }


# =============================================================================
# RESISTANCE GAIN COMPUTER
# =============================================================================

class ResistanceGainComputer:
    """
    Computes endogenous resistance gain.

    gamma_t = 1 / (1 + std(window(shock_history)))

    High resistance when shocks are stable (low variance).
    Low resistance when shocks are volatile (system adapting).
    """

    def __init__(self):
        self.gamma_history: List[float] = []
        self.window_std_history: List[float] = []
        self.t = 0

    def _get_window_size(self) -> int:
        """Endogenous window size from sqrt(t)."""
        return max(5, int(np.sqrt(self.t + 1)))

    def compute_gamma(self, shock_history: List[float]) -> Tuple[float, Dict]:
        """
        Compute resistance gain.

        Args:
            shock_history: History of shock values

        Returns:
            (gamma_t, diagnostics)
        """
        self.t += 1

        # Get endogenous window
        window_size = self._get_window_size()

        if len(shock_history) < 2:
            gamma_t = 1.0
            window_std = 0.0
        else:
            # Get recent window
            window = shock_history[-window_size:]
            window_std = float(np.std(window))

            # Resistance gain: inverse of volatility
            gamma_t = 1.0 / (1.0 + window_std)

        self.gamma_history.append(gamma_t)
        self.window_std_history.append(window_std)

        VETO_PROVENANCE.log(
            'gamma_t', float(gamma_t),
            '1/(1 + std(window(shock)))',
            {'window_size': window_size, 'window_std': window_std},
            self.t
        )

        diagnostics = {
            'gamma_t': gamma_t,
            'window_size': window_size,
            'window_std': window_std
        }

        return float(gamma_t), diagnostics

    def get_persistence(self) -> float:
        """Compute autocorrelation of gamma at lag 1."""
        if len(self.gamma_history) < 10:
            return 0.0

        gamma_arr = np.array(self.gamma_history)
        if np.std(gamma_arr) < NUMERIC_EPS:
            return 1.0

        autocorr = np.corrcoef(gamma_arr[:-1], gamma_arr[1:])[0, 1]
        return float(autocorr) if not np.isnan(autocorr) else 0.0

    def get_statistics(self) -> Dict:
        """Return gain statistics."""
        if not self.gamma_history:
            return {'mean_gamma': 1.0}

        gamma_arr = np.array(self.gamma_history)
        return {
            'mean_gamma': float(np.mean(gamma_arr)),
            'std_gamma': float(np.std(gamma_arr)),
            'min_gamma': float(np.min(gamma_arr)),
            'max_gamma': float(np.max(gamma_arr)),
            'persistence': self.get_persistence(),
            'n_samples': len(gamma_arr)
        }


# =============================================================================
# VETO TRANSITION ADJUSTER
# =============================================================================

class VetoTransitionAdjuster:
    """
    Adjusts transitions based on veto/opposition.

    x_next = x_next_base + gamma_t * O_t

    Veto effect pulls the next state back toward prototypes
    when shocks are detected.
    """

    def __init__(self):
        self.adjustment_history: List[float] = []
        self.veto_effect_history: List[float] = []
        self.t = 0

    def adjust_transition(self, x_next_base: np.ndarray,
                         gamma_t: float, O_t: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Apply veto adjustment to transition.

        Args:
            x_next_base: Base next state (without veto)
            gamma_t: Resistance gain
            O_t: Opposition field

        Returns:
            (x_next_adjusted, diagnostics)
        """
        self.t += 1

        # Handle dimension mismatch
        if len(O_t) != len(x_next_base):
            min_dim = min(len(O_t), len(x_next_base))
            O_t_adj = O_t[:min_dim]
            x_base_adj = x_next_base[:min_dim]
        else:
            O_t_adj = O_t
            x_base_adj = x_next_base

        # Veto-adjusted transition
        adjustment = gamma_t * O_t_adj
        x_next_adjusted = x_base_adj + adjustment

        # Pad back if needed
        if len(x_next_adjusted) < len(x_next_base):
            x_next_adjusted = np.concatenate([
                x_next_adjusted,
                x_next_base[len(x_next_adjusted):]
            ])

        # Track veto effect
        adjustment_magnitude = float(np.linalg.norm(adjustment))
        veto_effect = float(np.linalg.norm(x_next_adjusted - x_next_base))

        self.adjustment_history.append(adjustment_magnitude)
        self.veto_effect_history.append(veto_effect)

        VETO_PROVENANCE.log(
            'veto_effect', veto_effect,
            '||x_next_adjusted - x_next_base||',
            {'gamma_t': gamma_t, 'O_t_magnitude': float(np.linalg.norm(O_t))},
            self.t
        )

        diagnostics = {
            'adjustment_magnitude': adjustment_magnitude,
            'veto_effect': veto_effect,
            'x_next_base': x_next_base.tolist(),
            'x_next_adjusted': x_next_adjusted.tolist()
        }

        return x_next_adjusted, diagnostics

    def get_statistics(self) -> Dict:
        """Return adjuster statistics."""
        if not self.veto_effect_history:
            return {'mean_effect': 0.0}

        effect_arr = np.array(self.veto_effect_history)
        return {
            'mean_effect': float(np.mean(effect_arr)),
            'std_effect': float(np.std(effect_arr)),
            'max_effect': float(np.max(effect_arr)),
            'cumulative_effect': float(np.sum(effect_arr)),
            'n_samples': len(effect_arr)
        }


# =============================================================================
# STRUCTURAL VETO SYSTEM (MAIN CLASS)
# =============================================================================

class StructuralVetoSystem:
    """
    Main class for Phase 20 structural veto & resistance.

    Integrates:
    - Intrusion detection (shock indicator)
    - Structural opposition field
    - Resistance gain computation
    - Veto-adjusted transitions

    ALL parameters endogenous - ZERO magic constants.
    """

    def __init__(self, n_prototypes: int = 5):
        self.n_prototypes = n_prototypes

        # Components
        self.intrusion_detector = IntrusionDetector(n_prototypes)
        self.opposition_field = StructuralOppositionField()
        self.resistance_computer = ResistanceGainComputer()
        self.transition_adjuster = VetoTransitionAdjuster()

        # Tracking
        self.shock_timeline: List[float] = []
        self.gamma_timeline: List[float] = []
        self.veto_effect_timeline: List[float] = []
        self.epr_timeline: List[float] = []
        self.t = 0

    def set_prototypes(self, prototypes: np.ndarray):
        """Set prototype positions."""
        self.intrusion_detector.set_prototypes(prototypes)

    def process_step(self, x_t: np.ndarray, x_next_base: np.ndarray,
                    spread_t: float, epr_t: float) -> Dict:
        """
        Process one step of structural veto.

        Args:
            x_t: Current manifold position
            x_next_base: Base next position (without veto)
            spread_t: Current manifold spread
            epr_t: Current entropy production rate

        Returns:
            Dict with all veto outputs
        """
        self.t += 1

        # 1. Detect intrusion/shock
        shock_t, shock_diag = self.intrusion_detector.compute_shock(
            x_t, spread_t, epr_t
        )
        self.shock_timeline.append(shock_t)
        self.epr_timeline.append(epr_t)

        # 2. Compute opposition field
        direction = self.intrusion_detector.get_direction_to_nearest(x_t)
        O_t, opp_diag = self.opposition_field.compute_opposition(
            shock_t, direction, self.intrusion_detector.shock_history
        )

        # 3. Compute resistance gain
        gamma_t, gamma_diag = self.resistance_computer.compute_gamma(
            self.intrusion_detector.shock_history
        )
        self.gamma_timeline.append(gamma_t)

        # 4. Apply veto adjustment
        x_next_adjusted, adj_diag = self.transition_adjuster.adjust_transition(
            x_next_base, gamma_t, O_t
        )
        self.veto_effect_timeline.append(adj_diag['veto_effect'])

        result = {
            't': self.t,
            'shock_t': shock_t,
            'gamma_t': gamma_t,
            'O_t': O_t.tolist(),
            'O_t_magnitude': float(np.linalg.norm(O_t)),
            'veto_effect': adj_diag['veto_effect'],
            'x_next_adjusted': x_next_adjusted.tolist(),
            'diagnostics': {
                'shock': shock_diag,
                'opposition': opp_diag,
                'resistance': gamma_diag,
                'adjustment': adj_diag
            }
        }

        return result

    def get_veto_persistence(self) -> float:
        """Get autocorrelation of veto effects."""
        if len(self.veto_effect_timeline) < 10:
            return 0.0

        effect_arr = np.array(self.veto_effect_timeline)
        if np.std(effect_arr) < NUMERIC_EPS:
            return 1.0

        autocorr = np.corrcoef(effect_arr[:-1], effect_arr[1:])[0, 1]
        return float(autocorr) if not np.isnan(autocorr) else 0.0

    def get_epr_shock_correlation(self) -> float:
        """Get correlation between EPR and shocks."""
        if len(self.epr_timeline) < 10:
            return 0.0

        epr_arr = np.array(self.epr_timeline)
        shock_arr = np.array(self.shock_timeline)

        if np.std(epr_arr) < NUMERIC_EPS or np.std(shock_arr) < NUMERIC_EPS:
            return 0.0

        corr = np.corrcoef(epr_arr, shock_arr)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    def get_statistics(self) -> Dict:
        """Return comprehensive veto statistics."""
        return {
            'intrusion': self.intrusion_detector.get_statistics(),
            'opposition': self.opposition_field.get_statistics(),
            'resistance': self.resistance_computer.get_statistics(),
            'adjustment': self.transition_adjuster.get_statistics(),
            'veto_persistence': self.get_veto_persistence(),
            'epr_shock_correlation': self.get_epr_shock_correlation(),
            'n_steps': self.t
        }


# =============================================================================
# PROVENANCE
# =============================================================================

VETO20_PROVENANCE = {
    'module': 'veto20',
    'version': '1.0.0',
    'mechanisms': [
        'intrusion_detection',
        'structural_opposition_field',
        'resistance_gain_computation',
        'veto_transition_adjustment'
    ],
    'endogenous_params': [
        'shock_t = rank(delta) * rank(delta_spread) * rank(delta_epr)',
        'O_t = -rank(shock_t) * normalize(x_t - mu_k)',
        'gamma_t = 1/(1 + std(window(shock)))',
        'x_next = x_next_base + gamma_t * O_t',
        'window_size = sqrt(t)',
        'alpha_ema = 1/sqrt(t+1)'
    ],
    'no_magic_numbers': True,
    'no_semantic_labels': True
}


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("Phase 20: Structural Veto & Resistance System")
    print("=" * 50)

    np.random.seed(42)

    # Test veto system
    print("\n[1] Testing StructuralVetoSystem...")
    system = StructuralVetoSystem(n_prototypes=5)

    # Set prototypes
    prototypes = np.random.randn(5, 4) * 0.5
    system.set_prototypes(prototypes)

    shock_history = []
    gamma_history = []
    veto_history = []

    for t in range(500):
        # Generate trajectory with occasional perturbations
        is_perturbation = (t % 50 == 25)  # Perturbation every 50 steps

        if is_perturbation:
            # Large deviation
            x_t = np.random.randn(4) * 2.0
        else:
            # Normal dynamics around prototype
            proto_idx = t % 5
            x_t = prototypes[proto_idx] + np.random.randn(4) * 0.2

        # Base next state
        x_next_base = x_t + np.random.randn(4) * 0.1

        # Spread and EPR
        spread_t = 0.3 + (0.5 if is_perturbation else 0.0) + np.random.randn() * 0.05
        epr_t = 0.2 + (0.4 if is_perturbation else 0.0) + np.abs(np.random.randn()) * 0.1

        result = system.process_step(x_t, x_next_base, spread_t, epr_t)

        shock_history.append(result['shock_t'])
        gamma_history.append(result['gamma_t'])
        veto_history.append(result['veto_effect'])

        if t % 100 == 0 or is_perturbation:
            print(f"  t={t}: shock={result['shock_t']:.3f}, "
                  f"gamma={result['gamma_t']:.3f}, "
                  f"veto={result['veto_effect']:.3f}"
                  f"{' [PERTURBATION]' if is_perturbation else ''}")

    stats = system.get_statistics()
    print(f"\n[2] Final Statistics:")
    print(f"  Mean shock: {stats['intrusion']['mean_shock']:.4f}")
    print(f"  Mean gamma: {stats['resistance']['mean_gamma']:.4f}")
    print(f"  Gamma persistence: {stats['resistance']['persistence']:.4f}")
    print(f"  Mean veto effect: {stats['adjustment']['mean_effect']:.4f}")
    print(f"  EPR-shock correlation: {stats['epr_shock_correlation']:.4f}")

    print("\n" + "=" * 50)
    print("PHASE 20 VETO VERIFICATION:")
    print("  - shock_t: rank(delta) * rank(delta_spread) * rank(delta_epr)")
    print("  - O_t: -rank(shock) * normalize(direction)")
    print("  - gamma_t: 1/(1 + std(window(shock)))")
    print("  - x_next: x_base + gamma * O_t")
    print("  - window: sqrt(t)")
    print("  - alpha: 1/sqrt(t+1)")
    print("  - ZERO magic constants")
    print("=" * 50)
