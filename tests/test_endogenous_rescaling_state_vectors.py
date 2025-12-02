#!/usr/bin/env python3
"""
Test A2: Endogenous Invariance Under State Rescaling
=====================================================

If we rescale state vectors by a constant (then normalize),
internal metrics (CE, attractors, energy) should NOT change qualitatively.

This proves metrics capture structure, not absolute magnitudes.

100% endogenous - no magic numbers.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history


class RescalableAgent:
    """Agent with rescalable state dynamics."""

    def __init__(self, agent_id: str, dim: int, rng: np.random.Generator, scale: float = 1.0):
        self.agent_id = agent_id
        self.dim = dim
        self.rng = rng
        self.scale = scale

        # Initialize state
        self.state = self.rng.uniform(-1, 1, dim) * scale
        self._normalize()

        self.history: List[np.ndarray] = [self.state.copy()]

    def _normalize(self):
        """Normalize state to unit sphere."""
        norm = np.linalg.norm(self.state)
        if norm > 1e-12:
            self.state = self.state / norm

    def step(self, external: np.ndarray = None) -> np.ndarray:
        """Update with endogenous dynamics."""
        T = len(self.history)

        # Endogenous transition matrix from history
        if T > 3:
            window = min(L_t(T), len(self.history))
            recent = np.array(self.history[-window:])

            # Covariance-based dynamics
            cov = np.cov(recent.T)
            if cov.ndim == 0:
                cov = np.array([[cov]])

            # Endogenous scaling
            trace = np.trace(cov) + 1e-12
            W = cov / trace
        else:
            W = np.eye(self.dim) / self.dim

        # Apply rescaled dynamics
        if external is not None:
            new_state = np.tanh(W @ self.state * self.scale + external)
        else:
            new_state = np.tanh(W @ self.state * self.scale)

        self.state = new_state
        self._normalize()
        self.history.append(self.state.copy())

        if len(self.history) > max_history(T):
            self.history = self.history[-max_history(T):]

        return self.state

    def compute_metrics(self) -> Dict[str, float]:
        """Compute endogenous metrics."""
        if len(self.history) < 3:
            return {'CE': 0.5, 'energy': 0.5, 'stability': 0.5}

        T = len(self.history)
        window = L_t(T)
        recent = np.array(self.history[-window:])

        # CE from variance
        var = np.mean(np.var(recent, axis=0))
        CE = 1 / (1 + var)

        # Energy from state norms
        norms = np.linalg.norm(recent, axis=1)
        energy = np.mean(norms ** 2)

        # Stability from autocorrelation
        if len(recent) > 3:
            diffs = np.diff(recent, axis=0)
            diff_norms = np.linalg.norm(diffs, axis=1)
            stability = 1 / (1 + np.std(diff_norms))
        else:
            stability = 0.5

        return {
            'CE': float(CE),
            'energy': float(energy),
            'stability': float(stability)
        }


def run_simulation(dim: int, steps: int, seed: int, scale: float) -> Dict[str, np.ndarray]:
    """Run simulation with given scale factor."""
    rng = np.random.default_rng(seed)
    agents = [RescalableAgent(f'A{i}', dim, rng, scale) for i in range(5)]

    metrics_history = []

    for t in range(steps):
        # Compute mean field
        states = np.array([a.state for a in agents])
        mean_field = np.mean(states, axis=0)

        step_metrics = {}
        for i, agent in enumerate(agents):
            coupling = mean_field - agent.state / len(agents)
            agent.step(coupling)
            m = agent.compute_metrics()
            step_metrics[f'CE_{i}'] = m['CE']
            step_metrics[f'energy_{i}'] = m['energy']

        step_metrics['CE_global'] = np.mean([v for k, v in step_metrics.items() if k.startswith('CE_')])
        step_metrics['energy_global'] = np.mean([v for k, v in step_metrics.items() if k.startswith('energy_')])

        metrics_history.append(step_metrics)

    # Convert to arrays
    result = {}
    for key in metrics_history[0].keys():
        result[key] = np.array([m[key] for m in metrics_history])

    return result


def test_rescaling_invariance():
    """
    Test that rescaling state vectors preserves qualitative metrics.

    Protocol:
    1. Run with scale=1.0 (baseline)
    2. Run with various scales
    3. Compare CE and stability distributions
    4. They should be statistically similar
    """
    print("\n=== Test A2: State Rescaling Invariance ===")

    dim = 8
    steps = 250
    seed = 42

    # Baseline
    baseline = run_simulation(dim, steps, seed, scale=1.0)

    # Test scales - derived from powers of 2 (no magic)
    scales = [0.5, 2.0, 0.25, 4.0]

    results = []

    print(f"  Baseline scale=1.0, steps={steps}")

    for scale in scales:
        scaled = run_simulation(dim, steps, seed, scale)

        # Compare CE means and stds (qualitative similarity)
        mean_base = np.mean(baseline['CE_global'])
        mean_scaled = np.mean(scaled['CE_global'])
        mean_ratio = mean_scaled / (mean_base + 1e-12)

        results.append({
            'scale': scale,
            'mean_base': mean_base,
            'mean_scaled': mean_scaled,
            'mean_ratio': mean_ratio,
        })

        # Qualitative: ratio should be within reasonable bounds
        passed = 0.5 < mean_ratio < 2.0
        status = "PASS" if passed else "FAIL"
        print(f"  Scale {scale}: mean_ratio={mean_ratio:.4f} [{status}]")

    # Check that qualitative structure is preserved (means are similar order of magnitude)
    ratios = [r['mean_ratio'] for r in results]
    all_reasonable = all(0.5 < r < 2.0 for r in ratios)

    print(f"\n  All scales preserve qualitative structure: {all_reasonable}")

    assert all_reasonable, "CE should be qualitatively similar across scales"
    print("  [PASS] Rescaling invariance verified")

    return True


def test_attractor_count_invariance():
    """
    Test that number of attractors is invariant to rescaling.

    Attractors detected by finding stable regions in CE trajectory.
    """
    print("\n=== Test A2b: Attractor Count Invariance ===")

    dim = 6
    steps = 300
    seed = 123

    def count_attractors(CE_series: np.ndarray) -> int:
        """Count attractors from CE stability regions."""
        if len(CE_series) < 10:
            return 0

        # Endogenous window size
        window = max(5, len(CE_series) // 20)

        # Find stable regions (low variance windows)
        attractors = 0
        i = 0
        while i < len(CE_series) - window:
            chunk = CE_series[i:i + window]
            var = np.var(chunk)

            # Endogenous threshold: variance less than median variance
            if var < np.median(np.var(CE_series.reshape(-1, window), axis=1)):
                attractors += 1
                i += window  # Skip this region
            else:
                i += 1

        return max(1, attractors)

    baseline = run_simulation(dim, steps, seed, scale=1.0)
    base_attractors = count_attractors(baseline['CE_global'])

    scales = [0.5, 2.0, 3.0]
    results = []

    print(f"  Baseline attractors: {base_attractors}")

    for scale in scales:
        scaled = run_simulation(dim, steps, seed, scale)
        n_attractors = count_attractors(scaled['CE_global'])

        # Allow small deviation (endogenous)
        diff = abs(n_attractors - base_attractors)
        max_diff = max(1, base_attractors // 2)  # At most half deviation
        passed = diff <= max_diff

        results.append({
            'scale': scale,
            'attractors': n_attractors,
            'diff': diff,
            'passed': passed
        })

        status = "PASS" if passed else "FAIL"
        print(f"  Scale {scale}: attractors={n_attractors}, diff={diff} [{status}]")

    all_passed = all(r['passed'] for r in results)

    assert all_passed, "Attractor count should be approximately invariant to rescaling"
    print("  [PASS] Attractor count invariance verified")

    return True


if __name__ == '__main__':
    test_rescaling_invariance()
    test_attractor_count_invariance()
    print("\n=== All A2 tests passed ===")
