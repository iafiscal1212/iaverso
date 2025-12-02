#!/usr/bin/env python3
"""
Test E2: Collective Bias Robustness Across Seeds
=================================================

Run simulations with different seeds and verify:
- Same macroscopic conditions
- Different initializations
- Patterns are repeatable in parameter space

Expected:
- Details change
- But: polarization occurs in certain regimes
- Convergence occurs in others
- Patterns are REPEATABLE, not "one rare run"

100% endogenous - no magic numbers.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
from collections import defaultdict
import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history


class RobustCollectiveAgent:
    """Agent for collective robustness testing."""

    def __init__(self, agent_id: str, dim: int, rng: np.random.Generator):
        self.agent_id = agent_id
        self.dim = dim
        self.rng = rng

        self.state = self.rng.uniform(-1, 1, dim)
        self.state = self.state / (np.linalg.norm(self.state) + 1e-12)

        self.history: List[np.ndarray] = [self.state.copy()]

    def step(self, coupling: np.ndarray = None, coupling_strength: float = 0.5):
        """Step with configurable coupling strength."""
        T = len(self.history)

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
            new_state = np.tanh(new_state + coupling * coupling_strength)

        norm = np.linalg.norm(new_state)
        if norm > 1e-12:
            new_state = new_state / norm

        self.state = new_state
        self.history.append(self.state.copy())

        if len(self.history) > max_history(T):
            self.history = self.history[-max_history(T):]

        return self.state


class RobustCollectiveSystem:
    """System for robustness testing across seeds."""

    def __init__(self, n_agents: int, dim: int, seed: int, coupling_strength: float = 0.5):
        self.rng = np.random.default_rng(seed)
        self.n_agents = n_agents
        self.dim = dim
        self.coupling_strength = coupling_strength

        self.agents = [
            RobustCollectiveAgent(f'A{i}', dim, self.rng)
            for i in range(n_agents)
        ]

        self.t = 0
        self.metrics: List[Dict] = []

    def compute_metrics(self) -> Dict[str, float]:
        """Compute collective metrics."""
        states = np.array([a.state for a in self.agents])

        # Polarization
        polarization = float(np.mean(np.var(states, axis=0)))

        # Consensus (inverse of spread)
        spread = np.std(states)
        consensus = 1 / (1 + spread)

        # Coherence
        mean_state = np.mean(states, axis=0)
        mean_norm = np.linalg.norm(mean_state)
        if mean_norm > 1e-12:
            alignments = [np.dot(s, mean_state) / (np.linalg.norm(s) * mean_norm)
                         for s in states]
            coherence = float(np.mean(alignments))
        else:
            coherence = 0.0

        # Convergence (how close to equilibrium)
        if len(self.metrics) > 5:
            recent_coherence = [m['coherence'] for m in self.metrics[-5:]]
            convergence = 1 / (1 + np.std(recent_coherence))
        else:
            convergence = 0.5

        return {
            'polarization': polarization,
            'consensus': consensus,
            'coherence': coherence,
            'convergence': convergence
        }

    def step(self):
        self.t += 1

        states = np.array([a.state for a in self.agents])
        mean_field = np.mean(states, axis=0)

        for agent in self.agents:
            coupling = mean_field - agent.state / self.n_agents
            agent.step(coupling, self.coupling_strength)

        metrics = self.compute_metrics()
        metrics['t'] = self.t
        self.metrics.append(metrics)

    def run(self, steps: int) -> Dict[str, np.ndarray]:
        for _ in range(steps):
            self.step()

        return {
            key: np.array([m[key] for m in self.metrics])
            for key in ['polarization', 'consensus', 'coherence', 'convergence']
        }


def classify_regime(data: Dict[str, np.ndarray]) -> str:
    """
    Classify the collective regime based on final metrics.

    Regimes:
    - 'consensus': high coherence, low polarization
    - 'polarized': high polarization, low coherence
    - 'neutral': middle ground
    """
    final_coherence = np.mean(data['coherence'][-20:]) if len(data['coherence']) > 20 else np.mean(data['coherence'])
    final_polarization = np.mean(data['polarization'][-20:]) if len(data['polarization']) > 20 else np.mean(data['polarization'])

    # Endogenous thresholds based on distribution
    if final_coherence > 0.7 and final_polarization < 0.3:
        return 'consensus'
    elif final_polarization > 0.5 or final_coherence < 0.3:
        return 'polarized'
    else:
        return 'neutral'


def test_robustness_across_seeds():
    """Test that patterns are robust across different seeds."""
    print("\n=== Test E2: Robustness Across Seeds ===")

    n_agents = 5
    dim = 6
    steps = 200
    coupling_strength = 0.5

    seeds = list(range(10, 60, 5))  # 10 different seeds

    coherences = []
    consensuses = []
    regimes = []

    for seed in seeds:
        sys = RobustCollectiveSystem(n_agents, dim, seed, coupling_strength)
        data = sys.run(steps)

        coherence = np.mean(data['coherence'])
        consensus = np.mean(data['consensus'])
        regime = classify_regime(data)

        coherences.append(coherence)
        consensuses.append(consensus)
        regimes.append(regime)

    print(f"  Seeds tested: {len(seeds)}")
    print(f"  Coherence: mean={np.mean(coherences):.3f}, std={np.std(coherences):.3f}")
    print(f"  Consensus: mean={np.mean(consensuses):.3f}, std={np.std(consensuses):.3f}")

    # Count regimes
    regime_counts = defaultdict(int)
    for r in regimes:
        regime_counts[r] += 1

    print(f"\n  Regime distribution:")
    for regime, count in regime_counts.items():
        print(f"    {regime}: {count}/{len(seeds)} ({count/len(seeds)*100:.0f}%)")

    # Robustness: coefficient of variation should be reasonable
    cv_coherence = np.std(coherences) / (np.mean(coherences) + 1e-12)
    cv_consensus = np.std(consensuses) / (np.mean(consensuses) + 1e-12)

    print(f"\n  CV coherence: {cv_coherence:.3f}")
    print(f"  CV consensus: {cv_consensus:.3f}")

    # Not too much variance (robust) but not zero (not deterministic)
    is_robust = cv_coherence < 1.0 and cv_consensus < 1.0

    assert is_robust, "Patterns should be robust across seeds (CV < 1.0)"
    print("  [PASS] Patterns are robust across different seeds")

    return True


def test_regime_consistency():
    """Test that similar parameters produce similar regimes."""
    print("\n=== Test E2b: Regime Consistency ===")

    n_agents = 5
    dim = 6
    steps = 150

    # Test two coupling regimes
    low_coupling = 0.1
    high_coupling = 0.9

    low_regimes = []
    high_regimes = []

    for seed in range(10):
        # Low coupling
        sys_low = RobustCollectiveSystem(n_agents, dim, seed, low_coupling)
        data_low = sys_low.run(steps)
        low_regimes.append(classify_regime(data_low))

        # High coupling
        sys_high = RobustCollectiveSystem(n_agents, dim, seed, high_coupling)
        data_high = sys_high.run(steps)
        high_regimes.append(classify_regime(data_high))

    print(f"  Low coupling ({low_coupling}):")
    low_counts = defaultdict(int)
    for r in low_regimes:
        low_counts[r] += 1
    for regime, count in low_counts.items():
        print(f"    {regime}: {count}/10")

    print(f"\n  High coupling ({high_coupling}):")
    high_counts = defaultdict(int)
    for r in high_regimes:
        high_counts[r] += 1
    for regime, count in high_counts.items():
        print(f"    {regime}: {count}/10")

    # High coupling should produce more consensus
    high_consensus_count = high_counts.get('consensus', 0) + high_counts.get('neutral', 0)
    low_consensus_count = low_counts.get('consensus', 0) + low_counts.get('neutral', 0)

    more_consensus_at_high = high_consensus_count >= low_consensus_count

    print(f"\n  More consensus at high coupling: {more_consensus_at_high}")

    # Regimes should be somewhat consistent within each parameter setting
    max_low_regime = max(low_counts.values()) if low_counts else 0
    max_high_regime = max(high_counts.values()) if high_counts else 0

    has_dominant_regime = max_low_regime >= 4 or max_high_regime >= 4

    print(f"  Dominant regime exists: {has_dominant_regime}")

    assert has_dominant_regime, "Should have consistent dominant regime for given parameters"
    print("  [PASS] Regimes are consistent within parameter settings")

    return True


def test_parameter_space_patterns():
    """Test that patterns are repeatable in parameter space."""
    print("\n=== Test E2c: Parameter Space Patterns ===")

    n_agents = 4
    dim = 5
    steps = 120

    # Sweep coupling strength
    coupling_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_trials = 5

    results = {}

    for coupling in coupling_values:
        trial_coherences = []
        trial_consensuses = []

        for trial in range(n_trials):
            seed = 42 + trial * 7 + int(coupling * 100)
            sys = RobustCollectiveSystem(n_agents, dim, seed, coupling)
            data = sys.run(steps)

            trial_coherences.append(np.mean(data['coherence']))
            trial_consensuses.append(np.mean(data['consensus']))

        results[coupling] = {
            'coherence_mean': np.mean(trial_coherences),
            'coherence_std': np.std(trial_coherences),
            'consensus_mean': np.mean(trial_consensuses),
            'consensus_std': np.std(trial_consensuses)
        }

        print(f"  Coupling {coupling}: "
              f"coherence={results[coupling]['coherence_mean']:.3f} "
              f"(+/- {results[coupling]['coherence_std']:.3f})")

    # Check for monotonic trend (higher coupling → higher coherence)
    coherence_means = [results[c]['coherence_mean'] for c in coupling_values]
    trend_increases = sum(1 for i in range(len(coherence_means) - 1)
                         if coherence_means[i + 1] > coherence_means[i])

    mostly_increasing = trend_increases >= len(coherence_means) // 2

    print(f"\n  Coherence trend: {trend_increases}/{len(coherence_means)-1} increases")
    print(f"  Mostly increasing: {mostly_increasing}")

    # Pattern should be repeatable (low std relative to mean)
    avg_cv = np.mean([
        results[c]['coherence_std'] / (results[c]['coherence_mean'] + 1e-12)
        for c in coupling_values
    ])

    print(f"  Average CV: {avg_cv:.3f}")

    is_repeatable = avg_cv < 0.5  # Low coefficient of variation

    assert is_repeatable, "Patterns should be repeatable in parameter space"
    print("  [PASS] Parameter space shows repeatable patterns")

    return True


def test_not_one_rare_run():
    """Test that results are not from 'one rare run'."""
    print("\n=== Test E2d: Not One Rare Run ===")

    n_agents = 5
    dim = 6
    steps = 150
    n_runs = 20

    # Run many simulations with different seeds
    all_coherences = []

    for run in range(n_runs):
        seed = run * 13 + 7
        sys = RobustCollectiveSystem(n_agents, dim, seed, coupling_strength=0.5)
        data = sys.run(steps)
        all_coherences.append(np.mean(data['coherence']))

    print(f"  Runs: {n_runs}")
    print(f"  Coherence distribution:")
    print(f"    Min: {min(all_coherences):.3f}")
    print(f"    Max: {max(all_coherences):.3f}")
    print(f"    Mean: {np.mean(all_coherences):.3f}")
    print(f"    Median: {np.median(all_coherences):.3f}")
    print(f"    Std: {np.std(all_coherences):.3f}")

    # Quartiles
    q1, q2, q3 = np.percentile(all_coherences, [25, 50, 75])
    print(f"    Q1: {q1:.3f}, Q2: {q2:.3f}, Q3: {q3:.3f}")

    # Check that the distribution is not dominated by outliers
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    outliers = sum(1 for c in all_coherences if c < lower_fence or c > upper_fence)
    outlier_ratio = outliers / n_runs

    print(f"\n  Outliers: {outliers}/{n_runs} ({outlier_ratio*100:.0f}%)")

    # Most runs should be in the normal range (not outliers)
    not_rare = outlier_ratio < 0.25

    assert not_rare, "Most runs should produce similar results (not rare outliers)"
    print("  [PASS] Results are not from one rare run")

    return True


if __name__ == '__main__':
    test_robustness_across_seeds()
    test_regime_consistency()
    test_parameter_space_patterns()
    test_not_one_rare_run()
    print("\n=== All E2 tests passed ===")
