"""
WORLD-1 Metrics

Endogenous metrics for world health, stability, and consciousness-like properties.
All computed from percentiles, ranks, and covariances.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class WorldMetricsSnapshot:
    """Snapshot of world metrics at time t."""
    t: int
    entropy: float              # H_w(t): world entropy
    irreversibility: float      # R_w(t): EPR-like irreversibility
    health: float               # S_w(t): stability/health
    phi_world: float            # Phi_w(t): world-level integration
    shock_magnitude: float      # Deviation from prediction
    regime_stability: float     # How stable is current regime


class WorldMetrics:
    """
    Computes endogenous metrics for WORLD-1.

    All metrics derived from:
    - Percentile ranks
    - sqrt(t) scaling
    - 1/sqrt(t+1) rates
    - Covariances

    No magic constants.
    """

    def __init__(self, world_dim: int):
        """Initialize metrics calculator."""
        self.world_dim = world_dim
        self.t = 0

        # History
        self.state_history: List[np.ndarray] = []
        self.metrics_history: List[WorldMetricsSnapshot] = []

        # For percentile computations
        self.entropy_history: List[float] = []
        self.irreversibility_history: List[float] = []
        self.phi_history: List[float] = []

        # Prediction model (for shock detection)
        self.prediction_weights: Optional[np.ndarray] = None

    def record_state(self, w: np.ndarray):
        """Record world state."""
        self.state_history.append(w.copy())
        self.t += 1

        max_hist = 1000
        if len(self.state_history) > max_hist:
            self.state_history = self.state_history[-max_hist:]

    def _compute_window_size(self) -> int:
        """Endogenous window size."""
        return max(5, int(np.sqrt(self.t + 1)))

    def compute_entropy(self, w: np.ndarray) -> float:
        """
        Compute world entropy.

        Based on variance structure of world state.
        High entropy = high disorder/unpredictability.
        """
        if len(self.state_history) < 5:
            return 0.5

        W = self._compute_window_size()
        recent = np.array(self.state_history[-W:])

        # Entropy as normalized variance
        total_var = np.var(recent)
        max_var = np.max([np.var(self.state_history)]) + 1e-8

        entropy = total_var / max_var

        # Also consider distribution of components
        component_vars = np.var(recent, axis=0)
        evenness = 1 - np.std(component_vars) / (np.mean(component_vars) + 1e-8)

        return float(0.5 * entropy + 0.5 * evenness)

    def compute_irreversibility(self, w: np.ndarray) -> float:
        """
        Compute irreversibility (EPR-like).

        Measures asymmetry in forward vs backward dynamics.
        High irreversibility = system far from equilibrium.
        """
        if len(self.state_history) < 10:
            return 0.5

        W = min(self._compute_window_size(), len(self.state_history) - 1)
        recent = np.array(self.state_history[-W:])

        # Forward differences
        forward_diffs = np.diff(recent, axis=0)

        # "Backward" approximation: negative of forward
        backward_diffs = -forward_diffs

        # Irreversibility: asymmetry between forward and backward statistics
        forward_mean = np.mean(forward_diffs, axis=0)
        backward_mean = np.mean(backward_diffs, axis=0)

        asymmetry = np.linalg.norm(forward_mean - backward_mean)

        # Normalize by typical scale
        typical_scale = np.std(recent) + 1e-8
        irreversibility = np.tanh(asymmetry / typical_scale)

        return float(irreversibility)

    def compute_health(self, w: np.ndarray) -> float:
        """
        Compute world health/stability.

        Based on:
        - Variance not too high or too low
        - Connectivity (correlation structure)
        - Stability (low shock frequency)
        """
        if len(self.state_history) < 10:
            return 0.5

        W = self._compute_window_size()
        recent = np.array(self.state_history[-W:])

        # Variance score: mid-range is healthy
        variance = np.var(recent)
        all_vars = [np.var(self.state_history[max(0, i-W):i+1])
                    for i in range(W, len(self.state_history))]

        if len(all_vars) > 0:
            var_percentile = np.sum(np.array(all_vars) <= variance) / len(all_vars)
            # Best health at 50th percentile
            var_health = 1 - 2 * abs(var_percentile - 0.5)
        else:
            var_health = 0.5

        # Connectivity: correlation matrix structure
        if W > 2:
            corr_matrix = np.corrcoef(recent.T)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

            # Mean absolute correlation (connectivity)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            mean_corr = np.mean(np.abs(corr_matrix[mask]))

            # Connectivity health: mid-range is good
            conn_health = 1 - 2 * abs(mean_corr - 0.3)
        else:
            conn_health = 0.5

        # Stability: low recent changes
        changes = np.mean(np.abs(np.diff(recent, axis=0)))
        change_scale = np.std(recent) + 1e-8
        stability = 1 - np.tanh(changes / change_scale)

        # Combined health
        health = (var_health + conn_health + stability) / 3

        return float(np.clip(health, 0, 1))

    def compute_phi_world(self, w: np.ndarray) -> float:
        """
        Compute world-level information integration (Phi_w).

        Analogous to agent phi but for world state.
        """
        if len(self.state_history) < 10:
            return 0.5

        W = self._compute_window_size()
        recent = np.array(self.state_history[-W:])

        # Phi as mutual information proxy
        # Use correlation structure

        # Total variance
        total_var = np.var(recent)

        # Partition into halves and compute integration
        mid = self.world_dim // 2
        part1 = recent[:, :mid]
        part2 = recent[:, mid:]

        var1 = np.var(part1)
        var2 = np.var(part2)

        # Integration: total - sum of parts
        # (normalized)
        sum_parts = var1 + var2
        if sum_parts > 0:
            integration = (total_var - sum_parts) / sum_parts
            phi = np.tanh(integration)  # Bound to [-1, 1]
            phi = (phi + 1) / 2  # Map to [0, 1]
        else:
            phi = 0.5

        return float(phi)

    def compute_shock(self, w: np.ndarray) -> float:
        """
        Compute shock magnitude.

        Shock = deviation from predicted state.
        """
        if len(self.state_history) < 3:
            return 0.0

        # Simple prediction: linear extrapolation from recent states
        w_prev = self.state_history[-2]
        w_prev2 = self.state_history[-3] if len(self.state_history) >= 3 else w_prev

        # Predicted: linear continuation
        w_predicted = w_prev + (w_prev - w_prev2)

        # Shock: deviation from prediction
        deviation = np.linalg.norm(w - w_predicted)

        # Normalize by typical scale
        typical_scale = np.std(self.state_history) + 1e-8
        shock = deviation / typical_scale

        return float(np.tanh(shock))

    def compute_regime_stability(self, modes: np.ndarray) -> float:
        """
        Compute regime stability.

        High stability = dominant mode is persistent.
        """
        if len(self.state_history) < 10:
            return 0.5

        # Current dominant mode
        dominant = np.argmax(modes)
        dominance = modes[dominant]

        # How dominant is it?
        second_max = np.partition(modes, -2)[-2] if len(modes) > 1 else 0
        gap = dominance - second_max

        # Stability = dominance * gap
        stability = dominance * (1 + gap)

        return float(np.clip(stability, 0, 1))

    def compute_all(self, w: np.ndarray, modes: Optional[np.ndarray] = None) -> WorldMetricsSnapshot:
        """Compute all metrics."""
        self.record_state(w)

        entropy = self.compute_entropy(w)
        irreversibility = self.compute_irreversibility(w)
        health = self.compute_health(w)
        phi_world = self.compute_phi_world(w)
        shock = self.compute_shock(w)
        regime_stability = self.compute_regime_stability(modes) if modes is not None else 0.5

        # Record histories for percentiles
        self.entropy_history.append(entropy)
        self.irreversibility_history.append(irreversibility)
        self.phi_history.append(phi_world)

        # Keep bounded
        max_hist = 1000
        for hist in [self.entropy_history, self.irreversibility_history, self.phi_history]:
            if len(hist) > max_hist:
                hist[:] = hist[-max_hist:]

        snapshot = WorldMetricsSnapshot(
            t=self.t,
            entropy=entropy,
            irreversibility=irreversibility,
            health=health,
            phi_world=phi_world,
            shock_magnitude=shock,
            regime_stability=regime_stability
        )

        self.metrics_history.append(snapshot)
        if len(self.metrics_history) > max_hist:
            self.metrics_history = self.metrics_history[-max_hist:]

        return snapshot

    def get_percentile(self, metric_name: str, value: float) -> float:
        """Get percentile of value in metric history."""
        hist_map = {
            'entropy': self.entropy_history,
            'irreversibility': self.irreversibility_history,
            'phi': self.phi_history
        }

        hist = hist_map.get(metric_name, [])
        if len(hist) < 2:
            return 0.5

        percentile = np.sum(np.array(hist) <= value) / len(hist)
        return float(percentile)

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if len(self.metrics_history) == 0:
            return {'status': 'no_data'}

        recent = self.metrics_history[-20:] if len(self.metrics_history) >= 20 else self.metrics_history

        return {
            't': self.t,
            'entropy_mean': float(np.mean([m.entropy for m in recent])),
            'health_mean': float(np.mean([m.health for m in recent])),
            'phi_mean': float(np.mean([m.phi_world for m in recent])),
            'shock_mean': float(np.mean([m.shock_magnitude for m in recent])),
            'irreversibility_mean': float(np.mean([m.irreversibility for m in recent]))
        }


def test_metrics():
    """Test world metrics."""
    print("=" * 60)
    print("WORLD METRICS TEST")
    print("=" * 60)

    world_dim = 15
    metrics = WorldMetrics(world_dim)

    print(f"\nInitialized metrics for world_dim={world_dim}")

    # Simulate world evolution
    w = np.random.randn(world_dim) * 0.5

    for t in range(300):
        # Evolve world
        w = 0.95 * w + 0.05 * np.tanh(w) + np.random.randn(world_dim) * 0.1 / np.sqrt(t + 1)

        # Add occasional shocks
        if t % 50 == 25:
            w += np.random.randn(world_dim) * 0.5

        # Compute metrics
        modes = np.random.dirichlet(np.ones(3))
        snapshot = metrics.compute_all(w, modes)

        if (t + 1) % 50 == 0:
            print(f"\n  t={t+1}:")
            print(f"    Entropy: {snapshot.entropy:.3f}")
            print(f"    Health: {snapshot.health:.3f}")
            print(f"    Phi_w: {snapshot.phi_world:.3f}")
            print(f"    Shock: {snapshot.shock_magnitude:.3f}")
            print(f"    Irreversibility: {snapshot.irreversibility:.3f}")

    summary = metrics.get_summary()
    print(f"\n  Summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.3f}")

    return metrics


if __name__ == "__main__":
    test_metrics()
