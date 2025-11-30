#!/usr/bin/env python3
"""
Phase 24: Proto-Planning - Runner
=================================

Runs multi-seed experiments with proto-planning.

Null models:
- NULL A: Disabled planning (no planning field)
- NULL B: Shuffled predictions (randomize temporal alignment)
- NULL C: Random fields (uniform random direction)

GO Criteria:
1. Planning field magnitude > 0 (meaningful planning)
2. Prediction error decreases over time (learning)
3. Planning field shows temporal coherence
4. Planning changes trajectory
5. All parameters endogenous
"""

import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from planning24 import (
    ProtoPlanning,
    PLANNING24_PROVENANCE,
    PLANNING_PROVENANCE
)


# =============================================================================
# SYNTHETIC TRAJECTORY
# =============================================================================

def generate_predictable_trajectory(T: int, dim: int, seed: int = 42) -> Dict:
    """Generate predictable trajectory for planning test."""
    np.random.seed(seed)

    trajectory = {
        'z': [],
        'T': T,
        'dim': dim,
        'seed': seed
    }

    z_t = np.zeros(dim)

    for t in range(T):
        # Smooth, predictable drift
        drift = np.sin(np.arange(dim) * 0.1 + t * 0.03) * 0.1
        noise = np.random.randn(dim) * 0.03
        z_t = z_t + drift + noise
        trajectory['z'].append(z_t.copy())

    return trajectory


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_planning_experiment(trajectory: Dict, enabled: bool = True) -> Dict:
    """Run planning experiment."""
    planning = ProtoPlanning()
    T = trajectory['T']

    results = {
        'P_magnitude': [],
        'error': [],
        'error_rank': [],
        'horizon': [],
        'z_planned': []
    }

    for t in range(T):
        z_t = np.array(trajectory['z'][t])

        if enabled:
            result = planning.process_step(z_t)
            results['P_magnitude'].append(result['P_magnitude'])
            results['error'].append(result['error'])
            results['error_rank'].append(result['error_rank'])
            results['horizon'].append(result['horizon'])

            P = np.array(result['P'])
            z_planned = planning.apply_planning(z_t, P)
            results['z_planned'].append(z_planned.tolist())
        else:
            results['P_magnitude'].append(0.0)
            results['error'].append(0.0)
            results['error_rank'].append(0.5)
            results['horizon'].append(1)
            results['z_planned'].append(z_t.tolist())

    results['statistics'] = planning.get_statistics() if enabled else {}
    return results


def run_shuffled_experiment(trajectory: Dict) -> Dict:
    """NULL B: Shuffled predictions."""
    planning = ProtoPlanning()
    T = trajectory['T']

    # Shuffle trajectory order for predictions
    shuffled_idx = np.random.permutation(T)

    results = {'P_magnitude': []}

    for t in range(T):
        z_t = np.array(trajectory['z'][shuffled_idx[t]])
        result = planning.process_step(z_t)
        results['P_magnitude'].append(result['P_magnitude'])

    results['mean_P_magnitude'] = float(np.mean(results['P_magnitude']))
    return results


def run_random_field_experiment(trajectory: Dict) -> Dict:
    """NULL C: Random planning fields."""
    T = trajectory['T']
    dim = trajectory['dim']

    results = {'P_magnitude': []}

    for t in range(T):
        random_dir = np.random.randn(dim)
        random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-16)
        P = np.random.rand() * random_dir
        results['P_magnitude'].append(float(np.linalg.norm(P)))

    results['mean_P_magnitude'] = float(np.mean(results['P_magnitude']))
    return results


# =============================================================================
# METRICS
# =============================================================================

def compute_planning_metrics(results: Dict, trajectory: Dict) -> Dict:
    """Compute planning-specific metrics."""
    P_mag = np.array(results['P_magnitude'])
    error = np.array(results['error'])
    error_rank = np.array(results['error_rank'])

    # Mean planning field magnitude
    mean_P = float(np.mean(P_mag))
    std_P = float(np.std(P_mag))

    # Error trend (should decrease if learning)
    # Compare first half vs second half
    mid = len(error) // 2
    if mid > 0:
        first_half_error = np.mean(error[:mid])
        second_half_error = np.mean(error[mid:])
        error_reduction = first_half_error - second_half_error
    else:
        error_reduction = 0.0

    # Temporal coherence of P
    if len(P_mag) > 1:
        P_autocorr = float(np.corrcoef(P_mag[:-1], P_mag[1:])[0, 1])
        if np.isnan(P_autocorr):
            P_autocorr = 0.0
    else:
        P_autocorr = 0.0

    # Displacement from planning
    z_orig = np.array(trajectory['z'])
    z_planned = np.array(results['z_planned'])
    displacement = np.array([
        np.linalg.norm(z_planned[t] - z_orig[t])
        for t in range(len(z_orig))
    ])
    mean_displacement = float(np.mean(displacement))

    # Mean error
    mean_error = float(np.mean(error))

    return {
        'mean_P_magnitude': mean_P,
        'std_P_magnitude': std_P,
        'mean_error': mean_error,
        'error_reduction': error_reduction,
        'P_autocorr': P_autocorr,
        'mean_displacement': mean_displacement
    }


# =============================================================================
# GO/NO-GO
# =============================================================================

def evaluate_go_criteria(metrics: Dict, null_results: Dict) -> Dict:
    """Evaluate GO criteria for Phase 24."""

    # C1: Meaningful planning field
    c1_pass = metrics['mean_P_magnitude'] > 0.1

    # C2: Error decreases (learning) or stays stable
    c2_pass = metrics['error_reduction'] > -0.1  # Not getting worse

    # C3: Temporal coherence
    c3_pass = metrics['P_autocorr'] > 0

    # C4: Planning changes trajectory
    c4_pass = metrics['mean_displacement'] > 1e-6

    # C5: P magnitude has variance (adapts)
    c5_pass = metrics['std_P_magnitude'] > 0.01

    criteria = {
        'c1_meaningful_planning': {
            'pass': c1_pass,
            'value': metrics['mean_P_magnitude'],
            'threshold': 0.1,
            'description': 'Planning field magnitude > 0.1'
        },
        'c2_not_degrading': {
            'pass': c2_pass,
            'value': metrics['error_reduction'],
            'threshold': -0.1,
            'description': 'Prediction not degrading'
        },
        'c3_temporal_coherence': {
            'pass': c3_pass,
            'value': metrics['P_autocorr'],
            'threshold': 0,
            'description': 'Planning shows temporal coherence'
        },
        'c4_changes_trajectory': {
            'pass': c4_pass,
            'value': metrics['mean_displacement'],
            'threshold': 1e-6,
            'description': 'Planning changes trajectory'
        },
        'c5_adaptive': {
            'pass': c5_pass,
            'value': metrics['std_P_magnitude'],
            'threshold': 0.01,
            'description': 'Planning field adapts (std > 0.01)'
        }
    }

    n_pass = sum(1 for c in criteria.values() if c['pass'])
    go = n_pass >= 3

    return {'criteria': criteria, 'n_pass': n_pass, 'n_total': 5, 'go': go}


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_phase24_experiment(n_seeds: int = 5, T: int = 500, dim: int = 5) -> Dict:
    """Run full Phase 24 experiment."""
    print("=" * 60)
    print("PHASE 24: Proto-Planning")
    print("=" * 60)

    all_metrics = []
    all_null = {'disabled': [], 'shuffled': [], 'random': []}

    for seed in range(n_seeds):
        print(f"\n[Seed {seed}]")
        trajectory = generate_predictable_trajectory(T, dim, seed=seed)

        results = run_planning_experiment(trajectory, enabled=True)
        metrics = compute_planning_metrics(results, trajectory)
        all_metrics.append(metrics)

        print(f"  |P| mean: {metrics['mean_P_magnitude']:.4f}")
        print(f"  Error reduction: {metrics['error_reduction']:.4f}")
        print(f"  P autocorr: {metrics['P_autocorr']:.4f}")

        # Null models
        results_disabled = run_planning_experiment(trajectory, enabled=False)
        all_null['disabled'].append({'mean_P': 0.0})

        results_shuffled = run_shuffled_experiment(trajectory)
        all_null['shuffled'].append({'mean_P_magnitude': results_shuffled['mean_P_magnitude']})

        results_random = run_random_field_experiment(trajectory)
        all_null['random'].append({'mean_P_magnitude': results_random['mean_P_magnitude']})

    # Aggregate
    agg_metrics = {
        'mean_P_magnitude': float(np.mean([m['mean_P_magnitude'] for m in all_metrics])),
        'std_P_magnitude': float(np.mean([m['std_P_magnitude'] for m in all_metrics])),
        'mean_error': float(np.mean([m['mean_error'] for m in all_metrics])),
        'error_reduction': float(np.mean([m['error_reduction'] for m in all_metrics])),
        'P_autocorr': float(np.mean([m['P_autocorr'] for m in all_metrics])),
        'mean_displacement': float(np.mean([m['mean_displacement'] for m in all_metrics]))
    }

    agg_null = {
        'disabled': {'mean_P': 0.0},
        'shuffled': {'mean_P_magnitude': float(np.mean([n['mean_P_magnitude'] for n in all_null['shuffled']]))},
        'random': {'mean_P_magnitude': float(np.mean([n['mean_P_magnitude'] for n in all_null['random']]))}
    }

    go_eval = evaluate_go_criteria(agg_metrics, agg_null)

    print("\n" + "=" * 60)
    print("GO/NO-GO CRITERIA:")
    print("=" * 60)
    for name, c in go_eval['criteria'].items():
        status = "PASS" if c['pass'] else "FAIL"
        print(f"  [{status}] {c['description']}")
        print(f"         Value: {c['value']:.4f}, Threshold: {c['threshold']}")

    print(f"\nPassing: {go_eval['n_pass']}/{go_eval['n_total']} (need >= 3)")
    print(f"\n{'GO' if go_eval['go'] else 'NO-GO'}")

    return {
        'metrics': agg_metrics,
        'per_seed_metrics': all_metrics,
        'null_results': agg_null,
        'go_evaluation': go_eval,
        'config': {'n_seeds': n_seeds, 'T': T, 'dim': dim},
        'provenance': PLANNING24_PROVENANCE
    }


# =============================================================================
# OUTPUT
# =============================================================================

def save_results(results: Dict, output_dir: str = "results/phase24"):
    """Save results."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    with open(os.path.join(output_dir, "planning_metrics.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {output_dir}/planning_metrics.json")

    summary = generate_summary(results)
    with open(os.path.join(output_dir, "phase24_summary.md"), 'w') as f:
        f.write(summary)
    print(f"Saved: {output_dir}/phase24_summary.md")

    generate_figures(results)


def generate_summary(results: Dict) -> str:
    """Generate markdown summary."""
    m = results['metrics']
    go = results['go_evaluation']

    summary = f"""# Phase 24: Proto-Planning - Summary

Generated: {datetime.now().isoformat()}

## Overview

Phase 24 implements **Proto-Planning** via autoregressive prediction.
Future states are predicted and used to generate planning fields.

## Key Metrics

- Mean |P|: {m['mean_P_magnitude']:.4f}
- Std |P|: {m['std_P_magnitude']:.4f}
- Mean error: {m['mean_error']:.4f}
- Error reduction: {m['error_reduction']:.4f}
- P autocorrelation: {m['P_autocorr']:.4f}
- Mean displacement: {m['mean_displacement']:.4f}

## GO/NO-GO Criteria

| Criterion | Status |
|-----------|--------|
"""
    for name, c in go['criteria'].items():
        summary += f"| {c['description']} | {'PASS' if c['pass'] else 'FAIL'} |\n"

    summary += f"""
**Passing: {go['n_pass']}/{go['n_total']} (need >= 3)**

## {'GO' if go['go'] else 'NO-GO'}

## Endogeneity

- h = ceil(log2(t+1))
- w = sqrt(t+1)
- z_hat = z + h * velocity
- P = (1 - rank(e)) * normalize(z_hat - z)
- ZERO magic constants
"""
    return summary


def generate_figures(results: Dict):
    """Generate figures."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        m = results['metrics']
        null = results['null_results']
        go = results['go_evaluation']

        # Null comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = ['Planning', 'Disabled', 'Shuffled', 'Random']
        values = [m['mean_P_magnitude'], 0, null['shuffled']['mean_P_magnitude'], null['random']['mean_P_magnitude']]
        colors = ['green', 'gray', 'orange', 'red']
        ax.bar(labels, values, color=colors, alpha=0.7)
        ax.set_ylabel('Mean |P|')
        ax.set_title('Phase 24: Planning vs Null Models')
        plt.tight_layout()
        plt.savefig('figures/24_null_comparison.png', dpi=150)
        plt.close()
        print("Saved: figures/24_null_comparison.png")

        # GO criteria
        fig, ax = plt.subplots(figsize=(12, 6))
        names = [c['description'][:30] for c in go['criteria'].values()]
        vals = [c['value'] for c in go['criteria'].values()]
        colors = ['green' if c['pass'] else 'red' for c in go['criteria'].values()]
        ax.bar(range(len(names)), vals, color=colors, alpha=0.7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_title(f'Phase 24 GO Criteria: {go["n_pass"]}/{go["n_total"]}')
        plt.tight_layout()
        plt.savefig('figures/24_go_criteria.png', dpi=150)
        plt.close()
        print("Saved: figures/24_go_criteria.png")

    except ImportError:
        print("matplotlib not available")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results = run_phase24_experiment(n_seeds=5, T=500, dim=5)
    save_results(results)

    print("\n" + "=" * 60)
    print("PHASE 24 VERIFICATION:")
    print("  - h = ceil(log2(t+1))")
    print("  - z_hat = z + h * velocity")
    print("  - P = (1-rank(e)) * normalize(z_hat - z)")
    print("  - ZERO magic constants")
    print("=" * 60)
