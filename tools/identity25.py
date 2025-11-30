#!/usr/bin/env python3
"""
Phase 25: Operator-Resistant Identity
=====================================

Implements PURELY ENDOGENOUS identity maintenance.

Key components:
1. Identity signature: I_t = EMA(z_t) with alpha = 1/sqrt(t+1)
2. Deviation from identity: d_t = ||z_t - I_t||
3. Restoration field: R_t = rank(d_t) * normalize(I_t - z_t)
4. Identity-restored state: z_next = z_base + R_t

NO semantic labels. NO magic constants.
All parameters derived from internal history.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque

NUMERIC_EPS = 1e-16


# =============================================================================
# PROVENANCE
# =============================================================================

class IdentityProvenance:
    def __init__(self):
        self.logs: List[Dict] = []

    def log(self, param_name: str, value: float, derivation: str,
            source_data: Dict, timestep: int):
        self.logs.append({
            'param': param_name, 'value': value, 'derivation': derivation,
            'source': source_data, 't': timestep
        })


IDENTITY_PROVENANCE = IdentityProvenance()


# =============================================================================
# HELPERS
# =============================================================================

def compute_rank(value: float, history: np.ndarray) -> float:
    if len(history) == 0:
        return 0.5
    n = len(history)
    count_below = float(np.sum(history < value))
    count_equal = float(np.sum(history == value))
    return (count_below + 0.5 * count_equal) / n


def normalize_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > NUMERIC_EPS else v


# =============================================================================
# IDENTITY SIGNATURE
# =============================================================================

class IdentitySignature:
    """
    Computes identity signature as EMA of internal states.

    I_t = (1 - alpha) * I_{t-1} + alpha * z_t
    alpha = 1/sqrt(t+1)
    """

    def __init__(self):
        self.I: Optional[np.ndarray] = None
        self.t = 0

    def update(self, z_t: np.ndarray) -> Tuple[np.ndarray, Dict]:
        self.t += 1
        alpha = 1.0 / np.sqrt(self.t + 1)

        if self.I is None:
            self.I = z_t.copy()
        else:
            # Handle dimension changes
            if len(self.I) != len(z_t):
                new_I = np.zeros(len(z_t))
                min_dim = min(len(self.I), len(z_t))
                new_I[:min_dim] = self.I[:min_dim]
                self.I = new_I
            self.I = (1 - alpha) * self.I + alpha * z_t

        IDENTITY_PROVENANCE.log(
            'I_t', float(np.linalg.norm(self.I)),
            'EMA(z), alpha = 1/sqrt(t+1)',
            {'alpha': alpha},
            self.t
        )

        return self.I.copy(), {'alpha': alpha, 'I_norm': float(np.linalg.norm(self.I))}

    def get_statistics(self) -> Dict:
        return {'t': self.t, 'I_norm': float(np.linalg.norm(self.I)) if self.I is not None else 0}


# =============================================================================
# DEVIATION TRACKER
# =============================================================================

class DeviationTracker:
    """
    Tracks deviation from identity.

    d_t = ||z_t - I_t||
    """

    def __init__(self):
        self.deviation_history: List[float] = []
        self.t = 0

    def compute(self, z_t: np.ndarray, I_t: np.ndarray) -> Tuple[float, Dict]:
        self.t += 1

        min_dim = min(len(z_t), len(I_t))
        d = float(np.linalg.norm(z_t[:min_dim] - I_t[:min_dim]))
        self.deviation_history.append(d)

        IDENTITY_PROVENANCE.log(
            'd_t', d,
            '||z_t - I_t||',
            {},
            self.t
        )

        return d, {'deviation': d}

    def get_rank(self, d: float) -> float:
        return compute_rank(d, np.array(self.deviation_history))

    def get_statistics(self) -> Dict:
        if not self.deviation_history:
            return {'n_samples': 0}
        devs = np.array(self.deviation_history)
        return {'n_samples': len(devs), 'mean_d': float(np.mean(devs)), 'std_d': float(np.std(devs))}


# =============================================================================
# RESTORATION FIELD
# =============================================================================

class RestorationField:
    """
    Generates restoration field toward identity.

    R_t = rank(d_t) * normalize(I_t - z_t)
    """

    def __init__(self):
        self.R_magnitude_history: List[float] = []
        self.t = 0

    def generate(self, z_t: np.ndarray, I_t: np.ndarray,
                 deviation_rank: float) -> Tuple[np.ndarray, Dict]:
        self.t += 1

        min_dim = min(len(z_t), len(I_t))
        direction = I_t[:min_dim] - z_t[:min_dim]
        direction_norm = normalize_vector(direction)

        # Gain: high deviation rank = strong restoration
        gain = deviation_rank
        R = gain * direction_norm

        # Pad if needed
        if len(R) < len(z_t):
            R_full = np.zeros(len(z_t))
            R_full[:len(R)] = R
            R = R_full

        R_magnitude = float(np.linalg.norm(R))
        self.R_magnitude_history.append(R_magnitude)

        IDENTITY_PROVENANCE.log(
            'R_t', R_magnitude,
            'rank(d_t) * normalize(I_t - z_t)',
            {'gain': gain, 'deviation_rank': deviation_rank},
            self.t
        )

        return R, {'R_magnitude': R_magnitude, 'gain': gain}

    def get_statistics(self) -> Dict:
        if not self.R_magnitude_history:
            return {'n_fields': 0}
        R_arr = np.array(self.R_magnitude_history)
        return {'n_fields': len(R_arr), 'mean_R': float(np.mean(R_arr)), 'std_R': float(np.std(R_arr))}


# =============================================================================
# OPERATOR-RESISTANT IDENTITY SYSTEM
# =============================================================================

class OperatorResistantIdentity:
    """
    Main class for Phase 25 operator-resistant identity.

    The system maintains a stable identity signature and generates
    restoration fields when deviation is high.
    """

    def __init__(self):
        self.identity = IdentitySignature()
        self.deviation = DeviationTracker()
        self.restoration = RestorationField()
        self.t = 0

    def process_step(self, z_t: np.ndarray) -> Dict:
        """Process one step of identity maintenance."""
        self.t += 1

        # Update identity signature
        I_t, id_diag = self.identity.update(z_t)

        # Compute deviation
        d_t, dev_diag = self.deviation.compute(z_t, I_t)
        d_rank = self.deviation.get_rank(d_t)

        # Generate restoration field
        R_t, rest_diag = self.restoration.generate(z_t, I_t, d_rank)

        return {
            't': self.t,
            'I': I_t.tolist(),
            'I_norm': id_diag['I_norm'],
            'd': d_t,
            'd_rank': d_rank,
            'R': R_t.tolist(),
            'R_magnitude': rest_diag['R_magnitude'],
            'diagnostics': {
                'identity': id_diag,
                'deviation': dev_diag,
                'restoration': rest_diag
            }
        }

    def apply_restoration(self, z_base: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Apply restoration field."""
        min_dim = min(len(z_base), len(R))
        z_next = z_base.copy()
        z_next[:min_dim] += R[:min_dim]
        return z_next

    def get_statistics(self) -> Dict:
        return {
            'identity': self.identity.get_statistics(),
            'deviation': self.deviation.get_statistics(),
            'restoration': self.restoration.get_statistics(),
            'n_steps': self.t
        }


# =============================================================================
# PROVENANCE
# =============================================================================

IDENTITY25_PROVENANCE = {
    'module': 'identity25',
    'version': '1.0.0',
    'mechanisms': [
        'identity_signature',
        'deviation_tracking',
        'restoration_field'
    ],
    'endogenous_params': [
        'I_t = EMA(z), alpha = 1/sqrt(t+1)',
        'd_t = ||z_t - I_t||',
        'R_t = rank(d_t) * normalize(I_t - z_t)',
        'z_next = z_base + R_t',
        'alpha = 1/sqrt(t+1)',
        'rank = midrank(d, history)'
    ],
    'no_magic_numbers': True,
    'no_semantic_labels': True
}


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("Phase 25: Operator-Resistant Identity")
    print("=" * 50)

    np.random.seed(42)
    identity_sys = OperatorResistantIdentity()

    T = 500
    dim = 5

    R_mags = []
    deviations = []

    z_t = np.zeros(dim)

    for t in range(T):
        # Normal drift
        drift = np.sin(np.arange(dim) * 0.1 + t * 0.02) * 0.1
        noise = np.random.randn(dim) * 0.05

        # Occasional perturbation (operator interference)
        if t % 100 == 50:
            perturbation = np.random.randn(dim) * 0.5
        else:
            perturbation = np.zeros(dim)

        z_t = z_t + drift + noise + perturbation

        result = identity_sys.process_step(z_t)
        R_mags.append(result['R_magnitude'])
        deviations.append(result['d'])

        if t % 100 == 0:
            print(f"  t={t}: d={result['d']:.4f}, |R|={result['R_magnitude']:.4f}")

    stats = identity_sys.get_statistics()
    print(f"\n[Stats]")
    print(f"  Mean |R|: {stats['restoration']['mean_R']:.4f}")
    print(f"  Mean d: {stats['deviation']['mean_d']:.4f}")

    print("\n" + "=" * 50)
    print("PHASE 25 IDENTITY VERIFICATION:")
    print("  - I_t = EMA(z), alpha = 1/sqrt(t+1)")
    print("  - d_t = ||z_t - I_t||")
    print("  - R_t = rank(d_t) * normalize(I_t - z_t)")
    print("  - ZERO magic constants")
    print("=" * 50)
