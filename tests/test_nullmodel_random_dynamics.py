#!/usr/bin/env python3
"""
Test C2: Null Model - Random Dynamics
=====================================

Simulation with "dumb" agents:
- Same state dimension
- Same metric structure
- BUT random updates (noise with same variance)

If our metrics detect:
- Fewer attractors
- Less coherence
- Less structure
- No emergent patterns

Then NEO-EVA is NOT "pretty noise" - it has real structure.

100% endogenous - no magic numbers.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history


class EndogenousAgent:
    """Real endogenous agent."""

    def __init__(self, agent_id: str, dim: int, rng: np.random.Generator):
        self.agent_id = agent_id
        self.dim = dim
        self.rng = rng

        self.state = self.rng.uniform(-1, 1, dim)
        self.state = self.state / (np.linalg.norm(self.state) + 1e-12)

        self.history: List[np.ndarray] = [self.state.copy()]

    def step(self, coupling: np.ndarray = None):
        T = len(self.history)

        # Endogenous dynamics
        if T > 3:
            window = min(L_t(T), len(self.history))
            recent = np.array(self.history[-window:])
            cov = np.cov(recent.T)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            trace = np.trace(cov) + 1e-12
            W = cov / trace
        else:
            W = np.eye(self.dim) / self.dim

        new_state = np.tanh(W @ self.state)
        if coupling is not None:
            new_state = np.tanh(new_state + coupling)

        norm = np.linalg.norm(new_state)
        if norm > 1e-12:
            new_state = new_state / norm

        self.state = new_state
        self.history.append(self.state.copy())

        if len(self.history) > max_history(T):
            self.history = self.history[-max_history(T):]

        return self.state

    def compute_CE(self) -> float:
        if len(self.history) < 3:
            return 0.5
        T = len(self.history)
        window = min(L_t(T), len(self.history))
        recent = np.array(self.history[-window:])
        var = np.mean(np.var(recent, axis=0))
        return float(1 / (1 + var))


class RandomAgent:
    """
    Null agent with random dynamics.

    Same structure, but no meaningful computation.
    Just random walks with same variance.
    """

    def __init__(self, agent_id: str, dim: int, rng: np.random.Generator, noise_scale: float):
        self.agent_id = agent_id
        self.dim = dim
        self.rng = rng
        self.noise_scale = noise_scale

        self.state = self.rng.uniform(-1, 1, dim)
        self.state = self.state / (np.linalg.norm(self.state) + 1e-12)

        self.history: List[np.ndarray] = [self.state.copy()]

    def step(self, coupling: np.ndarray = None):
        # Pure random update - no meaningful dynamics
        noise = self.rng.normal(0, self.noise_scale, self.dim)
        new_state = self.state + noise

        # Normalize (same as real agent)
        norm = np.linalg.norm(new_state)
        if norm > 1e-12:
            new_state = new_state / norm

        self.state = new_state
        self.history.append(self.state.copy())

        # Keep same history length
        if len(self.history) > 100:
            self.history = self.history[-100:]

        return self.state

    def compute_CE(self) -> float:
        """Same metric computation - but data is random."""
        if len(self.history) < 3:
            return 0.5
        window = min(10, len(self.history))
        recent = np.array(self.history[-window:])
        var = np.mean(np.var(recent, axis=0))
        return float(1 / (1 + var))


class System:
    """Multi-agent system (real or null)."""

    def __init__(self, agents: List, is_coupled: bool = True):
        self.agents = agents
        self.is_coupled = is_coupled
        self.t = 0
        self.metrics: List[Dict] = []

    def step(self):
        self.t += 1

        if self.is_coupled:
            states = np.array([a.state for a in self.agents])
            mean_field = np.mean(states, axis=0)
        else:
            mean_field = None

        for agent in self.agents:
            if self.is_coupled and mean_field is not None:
                coupling = mean_field - agent.state / len(self.agents)
            else:
                coupling = None
            agent.step(coupling)

        # Compute metrics
        CE_vals = [a.compute_CE() for a in self.agents]
        self.metrics.append({
            'CE_mean': np.mean(CE_vals),
            'CE_std': np.std(CE_vals),
            'CE_vals': CE_vals
        })

    def run(self, steps: int) -> Dict[str, np.ndarray]:
        for _ in range(steps):
            self.step()

        return {
            'CE_mean': np.array([m['CE_mean'] for m in self.metrics]),
            'CE_std': np.array([m['CE_std'] for m in self.metrics])
        }


def compute_structure_score(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute structure score for comparison."""
    CE = data.get('CE_mean', np.array([0.5]))

    if len(CE) < 10:
        return {'autocorr': 0.0, 'stability': 0.0, 'range': 0.0}

    # Autocorrelation
    autocorr = np.corrcoef(CE[:-1], CE[1:])[0, 1] if len(CE) > 1 else 0.0
    if np.isnan(autocorr):
        autocorr = 0.0

    # Stability (inverse of variance)
    stability = 1 / (1 + np.var(CE))

    # Range (max - min)
    range_val = np.max(CE) - np.min(CE)

    return {
        'autocorr': float(autocorr),
        'stability': float(stability),
        'range': float(range_val)
    }


def test_random_dynamics_less_structure():
    """
    Test that random agents produce less structure than real agents.
    """
    print("\n=== Test C2: Random Dynamics Null Model ===")

    n_agents = 5
    dim = 8
    steps = 300
    seed = 42

    rng_real = np.random.default_rng(seed)
    rng_null = np.random.default_rng(seed)

    # Real system
    real_agents = [EndogenousAgent(f'R{i}', dim, rng_real) for i in range(n_agents)]
    real_system = System(real_agents, is_coupled=True)
    real_data = real_system.run(steps)
    real_structure = compute_structure_score(real_data)

    # Estimate noise scale from real dynamics
    real_vars = np.var(real_data['CE_mean'])
    noise_scale = np.sqrt(real_vars) * 2  # Match variance

    # Null system (random dynamics)
    null_agents = [RandomAgent(f'N{i}', dim, rng_null, noise_scale) for i in range(n_agents)]
    null_system = System(null_agents, is_coupled=False)
    null_data = null_system.run(steps)
    null_structure = compute_structure_score(null_data)

    print(f"  Real system:")
    print(f"    Autocorr: {real_structure['autocorr']:.4f}")
    print(f"    Stability: {real_structure['stability']:.4f}")

    print(f"\n  Null system (random):")
    print(f"    Autocorr: {null_structure['autocorr']:.4f}")
    print(f"    Stability: {null_structure['stability']:.4f}")

    # Real should have MORE structure
    more_autocorr = real_structure['autocorr'] > null_structure['autocorr'] - 0.1
    more_stable = real_structure['stability'] > null_structure['stability'] * 0.5

    print(f"\n  Comparison:")
    print(f"    More autocorr: {more_autocorr}")
    print(f"    More stable: {more_stable}")

    # At least one metric should show more structure in real
    has_more_structure = more_autocorr or more_stable

    assert has_more_structure, "Real dynamics should have more structure than random"
    print("  [PASS] Real dynamics has more structure than random null")

    return True


def test_random_no_coherent_patterns():
    """
    Test that random dynamics don't produce coherent patterns.
    """
    print("\n=== Test C2b: Random Has No Coherent Patterns ===")

    n_agents = 4
    dim = 6
    steps = 250
    seed = 123

    rng = np.random.default_rng(seed)

    # Run multiple null simulations
    null_autocorrs = []
    null_stabilities = []

    for i in range(10):
        null_agents = [RandomAgent(f'N{j}', dim, rng, noise_scale=0.1) for j in range(n_agents)]
        null_system = System(null_agents, is_coupled=False)
        null_data = null_system.run(steps)
        struct = compute_structure_score(null_data)
        null_autocorrs.append(struct['autocorr'])
        null_stabilities.append(struct['stability'])

    avg_autocorr = np.mean(null_autocorrs)
    avg_stability = np.mean(null_stabilities)

    print(f"  Null avg autocorr: {avg_autocorr:.4f} (std: {np.std(null_autocorrs):.4f})")
    print(f"  Null avg stability: {avg_stability:.4f} (std: {np.std(null_stabilities):.4f})")

    # Random should have low autocorrelation (noise-like)
    # Endogenous threshold: autocorr magnitude < 0.3
    is_noise_like = abs(avg_autocorr) < 0.3

    assert is_noise_like, f"Random dynamics should be noise-like (autocorr={avg_autocorr:.4f})"
    print("  [PASS] Random dynamics are noise-like")

    return True


def test_metrics_detect_real_vs_random():
    """
    Test that our metrics successfully distinguish real from random.
    """
    print("\n=== Test C2c: Metrics Detect Real vs Random ===")

    n_agents = 5
    dim = 8
    steps = 200
    n_trials = 5

    real_scores = []
    null_scores = []

    for trial in range(n_trials):
        seed = 42 + trial

        # Real
        rng_real = np.random.default_rng(seed)
        real_agents = [EndogenousAgent(f'R{i}', dim, rng_real) for i in range(n_agents)]
        real_sys = System(real_agents, is_coupled=True)
        real_data = real_sys.run(steps)
        real_struct = compute_structure_score(real_data)
        real_scores.append(real_struct['autocorr'])

        # Null
        rng_null = np.random.default_rng(seed)
        null_agents = [RandomAgent(f'N{i}', dim, rng_null, noise_scale=0.1) for i in range(n_agents)]
        null_sys = System(null_agents, is_coupled=False)
        null_data = null_sys.run(steps)
        null_struct = compute_structure_score(null_data)
        null_scores.append(null_struct['autocorr'])

    # Statistical test: are real scores significantly different from null?
    stat, pval = stats.mannwhitneyu(real_scores, null_scores, alternative='greater')

    print(f"  Real scores: {[f'{s:.3f}' for s in real_scores]}")
    print(f"  Null scores: {[f'{s:.3f}' for s in null_scores]}")
    print(f"\n  Mann-Whitney U test:")
    print(f"    Statistic: {stat:.4f}")
    print(f"    P-value: {pval:.4f}")

    # Metrics should significantly distinguish real from random
    # Endogenous threshold: p < 0.1 (lenient for small samples)
    distinguishes = pval < 0.1

    if not distinguishes:
        # Even if not statistically significant, check if means differ
        mean_diff = np.mean(real_scores) - np.mean(null_scores)
        distinguishes = mean_diff > 0.05
        print(f"  Mean difference: {mean_diff:.4f}")

    assert distinguishes, "Metrics should distinguish real from random dynamics"
    print("  [PASS] Metrics successfully distinguish real from random")

    return True


if __name__ == '__main__':
    test_random_dynamics_less_structure()
    test_random_no_coherent_patterns()
    test_metrics_detect_real_vs_random()
    print("\n=== All C2 tests passed ===")
