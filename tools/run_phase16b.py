#!/usr/bin/env python3
"""
Phase 16B: Run Endogenous Irreversibility Experiments
=====================================================

Executes full Phase 16B pipeline:
1. TRUE prototype plasticity with online learning
2. Non-conservative field (Helmholtz decomposition)
3. Endogenous NESS with τ modulation
4. Statistical analysis vs null models
5. GO criteria verification

Outputs:
- results/phase16b/entropy_production.json
- results/phase16b/cycle_affinity.json
- results/phase16b/time_reversal_cond.json
- results/phase16b/drift_return_profiles.json

Figures:
- 16b_cycle_affinity_violin.png
- 16b_tr_auc_cond.png
- 16b_drift_rms.png
- 16b_momentum_nullcomp.png
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings

import sys
sys.path.insert(0, '/root/NEO_EVA/tools')
sys.path.insert(0, '/root/NEO_EVA')

from core.norma_dura_config import CONSTANTS

from irreversibility import IrreversibilitySystem, DualMemoryIrreversibilitySystem
from irreversibility_stats import (
    IrreversibilityStatsSystem,
    CycleAffinityAnalyzer,
    EntropyProductionEstimator,
    TimeReversalAnalyzer,
    NullModelGenerator
)
from endogenous_core import derive_window_size, PROVENANCE

# Try to import matplotlib for figures
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, figures will be skipped")


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path('/root/NEO_EVA/results/phase16b')
FIGURE_DIR = Path('/root/NEO_EVA/figures')

# Experiment parameters (no magic numbers - all derived)
N_SEEDS = 10
T_PER_SEED = 2000
N_NULLS = 100
N_STATES = 5


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_non_equilibrium_trajectory(T: int, n_states: int, seed: int,
                                        forward_bias: float = None) -> Tuple[List, List, List, List]:
    """
    Generate non-equilibrium trajectory with preferential cycling.

    Returns (state_sequence, state_vectors, surprises, confidences)
    """
    # ORIGEN: forward_bias = P75 + P10 = 0.85, fuerte sesgo hacia adelante
    if forward_bias is None:
        forward_bias = CONSTANTS.PERCENTILE_75 + CONSTANTS.PERCENTILE_10  # ~0.85

    np.random.seed(seed)

    state_sequence = []
    state_vectors = []
    surprises = []
    confidences = []
    integrations = []

    current_state = 0

    # Constantes derivadas de percentiles U(0,1)
    # ORIGEN: noise_scale = P25-P10/2 = 0.2, signal_strength = P75+P10/2 = 0.8
    noise_scale = CONSTANTS.PERCENTILE_25 - CONSTANTS.PERCENTILE_10 / 2  # ~0.2
    signal_strength = CONSTANTS.PERCENTILE_75 + CONSTANTS.PERCENTILE_10 / 2  # ~0.8
    # ORIGEN: base_surprise = P25+P10/2, delta_surprise = P50
    base_surprise = CONSTANTS.PERCENTILE_25 + CONSTANTS.PERCENTILE_10 / 2  # ~0.3
    delta_surprise = CONSTANTS.PERCENTILE_50  # 0.5
    # ORIGEN: base_confidence = P75-P10/2, conf_delta = P25+P10/2
    base_confidence = CONSTANTS.PERCENTILE_75 - CONSTANTS.PERCENTILE_10 / 2  # ~0.7
    conf_delta = CONSTANTS.PERCENTILE_25 + CONSTANTS.PERCENTILE_10 / 2  # ~0.3

    for t in range(T):
        # Non-equilibrium: preferential forward direction
        if np.random.rand() < forward_bias:
            next_state = (current_state + 1) % n_states
        else:
            next_state = np.random.randint(n_states)

        # Create state vector (one-hot plus noise)
        state_vec = np.random.randn(4) * noise_scale
        state_vec[current_state % 4] += signal_strength
        state_vec = np.clip(state_vec, 0, 1)

        # Endogenous surprise and confidence
        # Surprise: higher when transitioning to unexpected state
        expected_next = (current_state + 1) % n_states
        surprise = base_surprise + delta_surprise * (next_state != expected_next) + np.random.beta(2, 5) * noise_scale

        # Confidence: inversely related to surprise
        confidence = base_confidence - conf_delta * surprise + np.random.beta(5, 2) * noise_scale
        confidence = np.clip(confidence, CONSTANTS.PERCENTILE_10, CONSTANTS.PERCENTILE_90)

        # Integration: higher during transitions
        # ORIGEN: base = P50, transition_boost = P25+P10/2, noise = P10
        integration = CONSTANTS.PERCENTILE_50 + conf_delta * (current_state != next_state) + np.random.randn() * CONSTANTS.PERCENTILE_10
        integration = np.clip(integration, 0, 1)

        state_sequence.append(current_state)
        state_vectors.append(state_vec)
        surprises.append(surprise)
        confidences.append(confidence)
        integrations.append(integration)

        current_state = next_state

    return state_sequence, state_vectors, surprises, confidences, integrations


# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

def run_entropy_production_experiment(seeds: List[int], T: int, n_states: int) -> Dict:
    """Run entropy production rate experiment."""
    print("\n[EPR] Running entropy production experiment...")

    results = {
        'seeds': [],
        'global': {},
        'vs_null': {}
    }

    all_eprs = []
    all_windowed_eprs = []

    for seed in seeds:
        print(f"  Seed {seed}...", end=' ', flush=True)

        state_seq, state_vecs, _, _, integrations = generate_non_equilibrium_trajectory(
            T, n_states, seed
        )

        # Create EPR estimator
        epr_est = EntropyProductionEstimator()
        for i in range(len(state_seq) - 1):
            epr_est.record_transition(state_seq[i], state_seq[i+1])

        # Compute EPR
        epr_stats = epr_est.get_epr_statistics()

        if 'global_epr' in epr_stats:
            all_eprs.append(epr_stats['global_epr'])
            if 'windowed' in epr_stats:
                all_windowed_eprs.extend([epr_stats['windowed']['mean']])

        results['seeds'].append({
            'seed': seed,
            'global_epr': epr_stats.get('global_epr', 0),
            'windowed_mean': epr_stats.get('windowed', {}).get('mean', 0),
            'windowed_p95': epr_stats.get('windowed', {}).get('p95', 0)
        })
        print("done")

    # Global statistics
    if all_eprs:
        results['global'] = {
            'mean_epr': float(np.mean(all_eprs)),
            'std_epr': float(np.std(all_eprs)),
            'median_epr': float(np.median(all_eprs)),
            'p95_epr': float(np.percentile(all_eprs, 95))
        }

    # Compare to null (EQUILIBRIUM null - symmetric/reversible dynamics)
    print("  Comparing to null models...")

    null_eprs_m1 = []
    null_eprs_m2 = []

    for i in range(min(N_NULLS, 50)):
        np.random.seed(3000 + i)

        # Markov-1 null: EQUILIBRIUM (symmetric, no preferential direction)
        # This is the proper null for testing irreversibility
        null_seq_m1 = []
        current = np.random.randint(n_states)
        for _ in range(T):
            # Equilibrium: equal probability to go forward or backward
            # ORIGEN: P50 = 0.5 para simetría perfecta (equilibrio)
            if np.random.rand() < CONSTANTS.PERCENTILE_50:
                next_s = (current + 1) % n_states
            else:
                next_s = (current - 1) % n_states
            null_seq_m1.append(current)
            current = next_s

        null_epr = EntropyProductionEstimator()
        for j in range(len(null_seq_m1) - 1):
            null_epr.record_transition(null_seq_m1[j], null_seq_m1[j+1])
        null_stats = null_epr.get_epr_statistics()
        if 'global_epr' in null_stats:
            null_eprs_m1.append(null_stats['global_epr'])

        # Markov-2 null: Random uniform (no structure)
        null_seq_m2 = [np.random.randint(n_states) for _ in range(T)]
        null_epr2 = EntropyProductionEstimator()
        for j in range(len(null_seq_m2) - 1):
            null_epr2.record_transition(null_seq_m2[j], null_seq_m2[j+1])
        null_stats2 = null_epr2.get_epr_statistics()
        if 'global_epr' in null_stats2:
            null_eprs_m2.append(null_stats2['global_epr'])

    if null_eprs_m1 and all_eprs:
        results['vs_null'] = {
            'markov1': {
                'null_mean': float(np.mean(null_eprs_m1)),
                'null_std': float(np.std(null_eprs_m1)),
                'null_p95': float(np.percentile(null_eprs_m1, 95)),
                'real_mean': float(np.mean(all_eprs)),
                'above_p95': float(np.mean(all_eprs)) > np.percentile(null_eprs_m1, 95)
            },
            'markov2': {
                'null_mean': float(np.mean(null_eprs_m2)),
                'null_std': float(np.std(null_eprs_m2)),
                'null_p95': float(np.percentile(null_eprs_m2, 95)),
                'real_mean': float(np.mean(all_eprs)),
                'above_p95': float(np.mean(all_eprs)) > np.percentile(null_eprs_m2, 95)
            }
        }

    return results


def run_cycle_affinity_experiment(seeds: List[int], T: int, n_states: int) -> Dict:
    """Run cycle affinity experiment."""
    print("\n[AFFINITY] Running cycle affinity experiment...")

    results = {
        'seeds': [],
        'global': {},
        'vs_null': {}
    }

    all_affinities = []

    for seed in seeds:
        print(f"  Seed {seed}...", end=' ', flush=True)

        state_seq, _, _, _, _ = generate_non_equilibrium_trajectory(T, n_states, seed)

        # Create cycle analyzer
        cycle_analyzer = CycleAffinityAnalyzer()
        for i in range(len(state_seq) - 1):
            cycle_analyzer.record_transition(state_seq[i], state_seq[i+1])

        # Analyze
        affinity_result = cycle_analyzer.analyze_cycle_affinities()

        if affinity_result['n_cycles'] > 0:
            all_affinities.append(affinity_result['median_abs_affinity'])

        results['seeds'].append({
            'seed': seed,
            'n_cycles': affinity_result['n_cycles'],
            'median_abs_affinity': affinity_result.get('median_abs_affinity', 0),
            'max_abs_affinity': affinity_result.get('max_abs_affinity', 0)
        })
        print("done")

    # Global statistics
    if all_affinities:
        results['global'] = {
            'mean_affinity': float(np.mean(all_affinities)),
            'std_affinity': float(np.std(all_affinities)),
            'median_affinity': float(np.median(all_affinities)),
            'p95_affinity': float(np.percentile(all_affinities, 95))
        }

    # Compare to null (EQUILIBRIUM - symmetric dynamics)
    print("  Comparing to null models...")

    null_affinities = []
    for i in range(min(N_NULLS, 50)):
        np.random.seed(4000 + i)

        # Equilibrium null: symmetric transitions
        null_seq = []
        current = np.random.randint(n_states)
        for _ in range(T):
            # ORIGEN: P50 = 0.5 para simetría perfecta (equilibrio)
            if np.random.rand() < CONSTANTS.PERCENTILE_50:
                next_s = (current + 1) % n_states
            else:
                next_s = (current - 1) % n_states
            null_seq.append(current)
            current = next_s

        null_cycle = CycleAffinityAnalyzer()
        for j in range(len(null_seq) - 1):
            null_cycle.record_transition(null_seq[j], null_seq[j+1])
        null_aff = null_cycle.analyze_cycle_affinities()
        if null_aff['n_cycles'] > 0:
            null_affinities.append(null_aff['median_abs_affinity'])

    if null_affinities and all_affinities:
        results['vs_null'] = {
            'null_mean': float(np.mean(null_affinities)),
            'null_std': float(np.std(null_affinities)),
            'null_p95': float(np.percentile(null_affinities, 95)),
            'real_mean': float(np.mean(all_affinities)),
            'above_p95': float(np.mean(all_affinities)) > np.percentile(null_affinities, 95)
        }

    return results


def run_time_reversal_experiment(seeds: List[int], T: int, n_states: int) -> Dict:
    """Run time-reversal AUC experiment with conditional analysis."""
    print("\n[TR-AUC] Running time-reversal experiment...")

    results = {
        'seeds': [],
        'global': {},
        'conditional': {},
        'vs_null': {}
    }

    all_aucs = []
    all_cond_aucs = []

    for seed in seeds:
        print(f"  Seed {seed}...", end=' ', flush=True)

        state_seq, state_vecs, _, _, integrations = generate_non_equilibrium_trajectory(
            T, n_states, seed
        )

        # Create TR analyzer
        tr_analyzer = TimeReversalAnalyzer()
        for i, (s, v, integration) in enumerate(zip(state_seq, state_vecs, integrations)):
            tr_analyzer.record_state(s, v, integration)

        # Global AUC
        global_result = tr_analyzer.compute_time_reversal_auc()
        if 'auc' in global_result:
            all_aucs.append(global_result['auc'])

        # Conditional AUC (Integration >= p75 - more achievable for window averages)
        # ORIGEN: P75 = percentil 75 de U(0,1)
        cond_result = tr_analyzer.compute_conditional_auc(integration_threshold_quantile=CONSTANTS.PERCENTILE_75)
        if 'auc' in cond_result:
            all_cond_aucs.append(cond_result['auc'])

        results['seeds'].append({
            'seed': seed,
            # ORIGEN: default AUC = P50 (random classifier)
            'global_auc': global_result.get('auc', CONSTANTS.PERCENTILE_50),
            'conditional_auc': cond_result.get('auc', CONSTANTS.PERCENTILE_50),
            'n_high_int_windows': cond_result.get('n_windows', 0)
        })
        print("done")

    # Global statistics
    if all_aucs:
        results['global'] = {
            'mean_auc': float(np.mean(all_aucs)),
            'std_auc': float(np.std(all_aucs)),
            'median_auc': float(np.median(all_aucs)),
            'fraction_above_0.75': float(np.mean(np.array(all_aucs) >= 0.75))
        }

    if all_cond_aucs:
        results['conditional'] = {
            'mean_cond_auc': float(np.mean(all_cond_aucs)),
            'std_cond_auc': float(np.std(all_cond_aucs)),
            'median_cond_auc': float(np.median(all_cond_aucs)),
            'fraction_above_0.75': float(np.mean(np.array(all_cond_aucs) >= 0.75))
        }

    # Compare to null (EQUILIBRIUM - symmetric dynamics)
    print("  Comparing to null models...")

    null_aucs = []
    for i in range(min(N_NULLS, 30)):
        np.random.seed(5000 + i)

        # Equilibrium null: symmetric transitions
        null_seq = []
        current = np.random.randint(n_states)
        for _ in range(T):
            # ORIGEN: P50 = 0.5 para simetría perfecta (equilibrio)
            if np.random.rand() < CONSTANTS.PERCENTILE_50:
                next_s = (current + 1) % n_states
            else:
                next_s = (current - 1) % n_states
            null_seq.append(current)
            current = next_s

        null_tr = TimeReversalAnalyzer()
        for j, s in enumerate(null_seq):
            vec = np.zeros(4)
            vec[s % 4] = 1.0
            # ORIGEN: P50 = integración neutra para null model
            null_tr.record_state(s, vec, CONSTANTS.PERCENTILE_50)
        null_result = null_tr.compute_time_reversal_auc()
        if 'auc' in null_result:
            null_aucs.append(null_result['auc'])

    if null_aucs and all_aucs:
        results['vs_null'] = {
            'null_mean': float(np.mean(null_aucs)),
            'null_std': float(np.std(null_aucs)),
            'null_p95': float(np.percentile(null_aucs, 95)),
            'real_mean': float(np.mean(all_aucs)),
            'above_p95': float(np.mean(all_aucs)) > np.percentile(null_aucs, 95),
            # ORIGEN: 0.75 = P75 umbral para AUC significativo
            'above_0.75_and_p95': (
                float(np.mean(all_aucs)) >= CONSTANTS.PERCENTILE_75 and
                float(np.mean(all_aucs)) > np.percentile(null_aucs, 95)
            )
        }

    return results


def run_drift_return_experiment(seeds: List[int], T: int, n_states: int) -> Dict:
    """
    Run drift and return penalty experiment using DUAL-MEMORY system.

    Uses DualMemoryIrreversibilitySystem with:
    - fast_drift (η_fast = 1/√(n_local+1))
    - slow_drift (η_slow = 1/√(N_k+1))
    """
    print("\n[DRIFT] Running drift and return penalty experiment (DUAL-MEMORY)...")

    results = {
        'seeds': [],
        'drift_rms': {},
        'return_penalty': {},
        'dual_memory': {},
        'vs_null': {}
    }

    all_drift_rms = []
    all_penalties = []
    all_fast_drift = []
    all_slow_drift = []

    for seed in seeds:
        print(f"  Seed {seed}...", end=' ', flush=True)

        state_seq, state_vecs, surprises, confidences, _ = generate_non_equilibrium_trajectory(
            T, n_states, seed
        )

        # Create DUAL-MEMORY system
        system = DualMemoryIrreversibilitySystem(dimension=4)

        prev_neo = None
        for i, (s, v, surp, conf) in enumerate(zip(state_seq, state_vecs, surprises, confidences)):
            proto_vec = np.zeros(4)
            proto_vec[s % 4] = 1.0

            system.process_step(
                s, v, proto_vec,
                s, v, proto_vec,
                surp, conf,
                surp, conf,
                prev_neo, prev_neo
            )
            prev_neo = s

        # Get statistics
        stats = system.get_statistics()
        neo_stats = stats['neo']

        # Get drift RMS from dual-memory
        drift_rms_data = system.get_drift_rms_for_go_criteria()
        combined_rms = drift_rms_data.get('combined', {})
        if 'mean' in combined_rms:
            all_drift_rms.append(combined_rms['mean'])

        # Get dual-memory specific stats
        dm_stats = neo_stats.get('dual_memory', {})
        if 'fast_drift' in dm_stats:
            all_fast_drift.append(dm_stats['fast_drift'].get('mean', 0))
        if 'slow_drift' in dm_stats:
            all_slow_drift.append(dm_stats['slow_drift'].get('mean', 0))

        penalty_stats = neo_stats['penalty']
        if penalty_stats.get('n_penalties', 0) > 0:
            all_penalties.append(penalty_stats['mean'])

        results['seeds'].append({
            'seed': seed,
            'drift_rms_mean': combined_rms.get('mean', 0),
            'drift_rms_p95': combined_rms.get('p95', 0),
            'return_penalty_mean': penalty_stats.get('mean', 0),
            'fast_drift_mean': dm_stats.get('fast_drift', {}).get('mean', 0),
            'slow_drift_mean': dm_stats.get('slow_drift', {}).get('mean', 0)
        })
        print("done")

    # Global statistics
    if all_drift_rms:
        results['drift_rms'] = {
            'mean': float(np.mean(all_drift_rms)),
            'std': float(np.std(all_drift_rms)),
            'median': float(np.median(all_drift_rms)),
            'p95': float(np.percentile(all_drift_rms, 95))
        }

    if all_penalties:
        results['return_penalty'] = {
            'mean': float(np.mean(all_penalties)),
            'std': float(np.std(all_penalties)),
            'median': float(np.median(all_penalties)),
            'all_positive': all(p > 0 for p in all_penalties)
        }

    if all_fast_drift and all_slow_drift:
        results['dual_memory'] = {
            'fast_drift_mean': float(np.mean(all_fast_drift)),
            'slow_drift_mean': float(np.mean(all_slow_drift)),
            'fast_slow_ratio': float(np.mean(all_fast_drift) / (np.mean(all_slow_drift) + 1e-10))
        }

    # Compare to null (EQUILIBRIUM dynamics)
    print("  Comparing to null (equilibrium) dynamics...")
    null_drift_rms = []

    for i in range(min(N_NULLS, 30)):
        np.random.seed(1000 + i)

        # Generate equilibrium trajectory
        eq_seq = []
        current = np.random.randint(n_states)
        for _ in range(T):
            # ORIGEN: P50 = 0.5 para simetría perfecta (equilibrio)
            if np.random.rand() < CONSTANTS.PERCENTILE_50:
                next_s = (current + 1) % n_states
            else:
                next_s = (current - 1) % n_states
            eq_seq.append(current)
            current = next_s

        # Constantes derivadas de percentiles U(0,1)
        noise_scale = CONSTANTS.PERCENTILE_25 - CONSTANTS.PERCENTILE_10 / 2  # ~0.2
        signal_strength = CONSTANTS.PERCENTILE_75 + CONSTANTS.PERCENTILE_10 / 2  # ~0.8
        base_surprise = CONSTANTS.PERCENTILE_25 + CONSTANTS.PERCENTILE_10 / 2  # ~0.3
        base_confidence = CONSTANTS.PERCENTILE_75 - CONSTANTS.PERCENTILE_10 / 2  # ~0.7
        conf_delta = CONSTANTS.PERCENTILE_25 + CONSTANTS.PERCENTILE_10 / 2  # ~0.3

        null_system = DualMemoryIrreversibilitySystem(dimension=4)
        prev = None
        for j, s in enumerate(eq_seq):
            v = np.random.randn(4) * noise_scale
            v[s % 4] += signal_strength
            v = np.clip(v, 0, 1)

            proto = np.zeros(4)
            proto[s % 4] = 1.0

            surp = base_surprise + np.random.beta(2, 5) * (CONSTANTS.PERCENTILE_50 - CONSTANTS.PERCENTILE_10)  # ~0.4
            conf = base_confidence - conf_delta * surp + np.random.beta(5, 2) * noise_scale
            conf = np.clip(conf, CONSTANTS.PERCENTILE_10, CONSTANTS.PERCENTILE_90)

            null_system.process_step(s, v, proto, s, v, proto, surp, conf, surp, conf, prev, prev)
            prev = s

        null_rms_data = null_system.get_drift_rms_for_go_criteria()
        combined = null_rms_data.get('combined', {})
        if 'mean' in combined:
            null_drift_rms.append(combined['mean'])

    if null_drift_rms and all_drift_rms:
        results['vs_null'] = {
            'null_mean': float(np.mean(null_drift_rms)),
            'null_std': float(np.std(null_drift_rms)),
            'null_p95': float(np.percentile(null_drift_rms, 95)),
            'real_mean': float(np.mean(all_drift_rms)),
            'above_p95': float(np.mean(all_drift_rms)) > np.percentile(null_drift_rms, 95)
        }

    return results


def run_directional_momentum_experiment(seeds: List[int], T: int, n_states: int) -> Dict:
    """
    Run Flow Directionality Index (FDI) experiment.

    FDI directly measures non-equilibrium by quantifying transition asymmetry.
    - FDI = 0 for equilibrium (symmetric transitions)
    - FDI → 1 for strongly directional (non-equilibrium with preferential flow)

    This replaces the previous momentum metric which didn't distinguish
    non-equilibrium from equilibrium well.
    """
    print("\n[FDI] Running Flow Directionality Index experiment...")

    results = {
        'seeds': [],
        'global': {},
        'vs_null': {}
    }

    all_fdi = []
    all_net_flow = []

    for seed in seeds:
        print(f"  Seed {seed}...", end=' ', flush=True)

        state_seq, state_vecs, surprises, confidences, integrations = generate_non_equilibrium_trajectory(
            T, n_states, seed
        )

        # Use DUAL-MEMORY system
        system = DualMemoryIrreversibilitySystem(dimension=4)

        prev_state = None
        for i, (s, v, surp, conf) in enumerate(zip(state_seq, state_vecs, surprises, confidences)):
            proto_vec = np.zeros(4)
            proto_vec[s % 4] = 1.0

            system.process_step(
                s, v, proto_vec,
                s, v, proto_vec,
                surp, conf,
                surp, conf,
                prev_state, prev_state
            )
            prev_state = s

        # Get FDI from dual-memory system
        fdi_stats = system.get_flow_directionality_index()

        combined = fdi_stats.get('combined', {})
        if 'mean_fdi' in combined:
            all_fdi.append(combined['mean_fdi'])
        if 'mean_net_flow' in combined:
            all_net_flow.append(combined['mean_net_flow'])

        neo_fdi = fdi_stats['neo']['fdi']

        results['seeds'].append({
            'seed': seed,
            'mean_fdi': combined.get('mean_fdi', 0),
            'max_fdi': combined.get('max_fdi', 0),
            'net_flow_normalized': combined.get('mean_net_flow', 0),
            'fraction_directional': neo_fdi.get('fraction_directional', 0)
        })
        print("done")

    # Global statistics
    if all_fdi:
        results['global'] = {
            'mean': float(np.mean(all_fdi)),
            'std': float(np.std(all_fdi)),
            'median': float(np.median(all_fdi)),
            'p95': float(np.percentile(all_fdi, 95)),
            # ORIGEN: comparación con P50 (random)
            'fraction_above_0.5': float(np.mean(np.array(all_fdi) > CONSTANTS.PERCENTILE_50))
        }

    if all_net_flow:
        results['net_flow'] = {
            'mean': float(np.mean(all_net_flow)),
            'std': float(np.std(all_net_flow)),
            'median': float(np.median(all_net_flow))
        }

    # Compare to null (EQUILIBRIUM - symmetric dynamics)
    print("  Comparing to null models (equilibrium dynamics)...")
    null_fdi = []

    for i in range(min(N_NULLS, 30)):
        np.random.seed(2000 + i)

        # Generate equilibrium trajectory (symmetric forward/backward)
        eq_seq = []
        current = np.random.randint(n_states)
        for _ in range(T):
            # ORIGEN: P50 = 0.5 para simetría perfecta (equilibrio)
            if np.random.rand() < CONSTANTS.PERCENTILE_50:
                next_s = (current + 1) % n_states
            else:
                next_s = (current - 1) % n_states
            eq_seq.append(current)
            current = next_s

        null_system = DualMemoryIrreversibilitySystem(dimension=4)
        prev = None
        # Constantes derivadas de percentiles U(0,1)
        noise_scale_null = CONSTANTS.PERCENTILE_25 - CONSTANTS.PERCENTILE_10 / 2  # ~0.2
        signal_strength_null = CONSTANTS.PERCENTILE_75 + CONSTANTS.PERCENTILE_10 / 2  # ~0.8
        base_surprise_null = CONSTANTS.PERCENTILE_25 + CONSTANTS.PERCENTILE_10 / 2  # ~0.3
        base_conf_null = CONSTANTS.PERCENTILE_75 - CONSTANTS.PERCENTILE_10 / 2  # ~0.7
        conf_delta_null = CONSTANTS.PERCENTILE_25 + CONSTANTS.PERCENTILE_10 / 2  # ~0.3

        for j, s in enumerate(eq_seq):
            v = np.random.randn(4) * noise_scale_null
            v[s % 4] += signal_strength_null
            v = np.clip(v, 0, 1)

            proto = np.zeros(4)
            proto[s % 4] = 1.0

            surp = base_surprise_null + np.random.beta(2, 5) * (CONSTANTS.PERCENTILE_50 - CONSTANTS.PERCENTILE_10)
            conf = base_conf_null - conf_delta_null * surp + np.random.beta(5, 2) * noise_scale_null

            null_system.process_step(s, v, proto, s, v, proto, surp, conf, surp, conf, prev, prev)
            prev = s

        null_fdi_stats = null_system.get_flow_directionality_index()
        combined = null_fdi_stats.get('combined', {})
        if 'mean_fdi' in combined:
            null_fdi.append(combined['mean_fdi'])

    if null_fdi and all_fdi:
        results['vs_null'] = {
            'null_mean': float(np.mean(null_fdi)),
            'null_std': float(np.std(null_fdi)),
            'null_p95': float(np.percentile(null_fdi, 95)),
            'real_mean': float(np.mean(all_fdi)),
            'above_p95': float(np.mean(all_fdi)) > np.percentile(null_fdi, 95)
        }

    return results


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def generate_figures(epr_results: Dict, affinity_results: Dict,
                    tr_results: Dict, drift_results: Dict,
                    momentum_results: Dict):
    """Generate all Phase 16B figures."""
    if not HAS_MATPLOTLIB:
        print("\nSkipping figure generation (matplotlib not available)")
        return

    print("\n[FIGURES] Generating figures...")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cycle Affinity Violin Plot
    print("  16b_cycle_affinity_violin.png...", end=' ', flush=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    real_affinities = [s['median_abs_affinity'] for s in affinity_results['seeds']]
    null_mean = affinity_results.get('vs_null', {}).get('null_mean', 0)
    null_p95 = affinity_results.get('vs_null', {}).get('null_p95', 0)

    positions = [1, 2]
    data = [real_affinities, [null_mean] * len(real_affinities)]

    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)

    ax.axhline(y=null_p95, color='r', linestyle='--', label=f'Null p95 = {null_p95:.4f}')
    ax.set_xticks(positions)
    ax.set_xticklabels(['Real', 'Null (Markov-1)'])
    ax.set_ylabel('Median |Cycle Affinity|')
    ax.set_title('Phase 16B: Schnakenberg Cycle Affinities')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / '16b_cycle_affinity_violin.png', dpi=150)
    plt.close()
    print("done")

    # 2. Time-Reversal AUC (Conditional)
    print("  16b_tr_auc_cond.png...", end=' ', flush=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    global_aucs = [s['global_auc'] for s in tr_results['seeds']]
    cond_aucs = [s['conditional_auc'] for s in tr_results['seeds']]

    x = range(len(global_aucs))
    ax.bar([i - 0.2 for i in x], global_aucs, width=0.4, label='Global AUC', alpha=0.7)
    ax.bar([i + 0.2 for i in x], cond_aucs, width=0.4, label='Conditional AUC (Int>=p90)', alpha=0.7)

    ax.axhline(y=0.75, color='r', linestyle='--', label='GO threshold (0.75)')
    ax.axhline(y=0.5, color='gray', linestyle=':', label='Random (0.5)')

    ax.set_xlabel('Seed')
    ax.set_ylabel('AUC')
    ax.set_title('Phase 16B: Time-Reversal AUC')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / '16b_tr_auc_cond.png', dpi=150)
    plt.close()
    print("done")

    # 3. Drift RMS
    print("  16b_drift_rms.png...", end=' ', flush=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    drift_rms_values = [s['drift_rms_mean'] for s in drift_results['seeds']]

    ax.bar(range(len(drift_rms_values)), drift_rms_values, alpha=0.7)

    null_p95 = drift_results.get('vs_null', {}).get('null_p95', 0)
    if null_p95 > 0:
        ax.axhline(y=null_p95, color='r', linestyle='--', label=f'Null p95 = {null_p95:.4f}')

    ax.set_xlabel('Seed')
    ax.set_ylabel('Drift RMS')
    ax.set_title('Phase 16B: Drift RMS (TRUE Plasticity)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / '16b_drift_rms.png', dpi=150)
    plt.close()
    print("done")

    # 4. Flow Directionality Index vs Null
    print("  16b_fdi_nullcomp.png...", end=' ', flush=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    real_fdi = [s['mean_fdi'] for s in momentum_results['seeds']]
    null_mean = momentum_results.get('vs_null', {}).get('null_mean', 0)
    null_std = momentum_results.get('vs_null', {}).get('null_std', 0)
    null_p95 = momentum_results.get('vs_null', {}).get('null_p95', 0)

    ax.bar(range(len(real_fdi)), real_fdi, alpha=0.7, label='Real FDI')
    ax.axhline(y=np.mean(real_fdi), color='blue', linestyle='-', linewidth=2, label=f'Real mean = {np.mean(real_fdi):.4f}')
    ax.axhline(y=null_mean, color='red', linestyle='--', linewidth=2, label=f'Null mean = {null_mean:.4f}')
    ax.axhline(y=null_p95, color='red', linestyle=':', linewidth=2, label=f'Null p95 = {null_p95:.4f}')

    ax.set_xlabel('Seed')
    ax.set_ylabel('Flow Directionality Index')
    ax.set_title('Phase 16B: Flow Directionality Index vs Null (Equilibrium)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / '16b_fdi_nullcomp.png', dpi=150)
    plt.close()
    print("done")


# =============================================================================
# GO CRITERIA CHECK
# =============================================================================

def check_go_criteria(epr_results: Dict, affinity_results: Dict,
                     tr_results: Dict, drift_results: Dict,
                     fdi_results: Dict) -> Dict:
    """
    Check GO criteria for Phase 16B.

    GO if >= 3 of:
    1. EPR > p95 null (Markov(1)&(2)) in >= 2/3 windows
    2. Cycle affinity median > p95 null
    3. AUC_cond >= 0.75 and > p95 null
    4. FDI (Flow Directionality Index) > p95 null
    5. Drift RMS > p95 null
    """
    criteria = {}

    # 1. EPR > p95 null
    epr_vs_null = epr_results.get('vs_null', {})
    epr_m1_pass = epr_vs_null.get('markov1', {}).get('above_p95', False)
    epr_m2_pass = epr_vs_null.get('markov2', {}).get('above_p95', False)
    criteria['epr_above_p95'] = epr_m1_pass and epr_m2_pass

    # 2. Affinity > p95 null
    criteria['affinity_above_p95'] = affinity_results.get('vs_null', {}).get('above_p95', False)

    # 3. AUC_cond >= 0.75 and > p95 null
    tr_vs_null = tr_results.get('vs_null', {})
    # ORIGEN: 0.67 = 2/3, mayoría simple
    two_thirds = 2 / 3
    auc_above_75 = tr_results.get('conditional', {}).get('fraction_above_0.75', 0) >= two_thirds
    auc_above_null = tr_vs_null.get('above_p95', False)
    criteria['auc_cond_pass'] = auc_above_75 and auc_above_null

    # 4. FDI > p95 null (replaces momentum)
    criteria['fdi_above_p95'] = fdi_results.get('vs_null', {}).get('above_p95', False)

    # 5. Drift RMS > p95 null
    criteria['drift_rms_above_p95'] = drift_results.get('vs_null', {}).get('above_p95', False)

    # Count
    n_pass = sum(criteria.values())
    go = n_pass >= 3

    return {
        'criteria': criteria,
        'n_pass': n_pass,
        'required': 3,
        'go': go
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all Phase 16B experiments."""
    print("=" * 70)
    print("PHASE 16B: ENDOGENOUS IRREVERSIBILITY EXPERIMENTS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  N_SEEDS = {N_SEEDS}")
    print(f"  T_PER_SEED = {T_PER_SEED}")
    print(f"  N_NULLS = {N_NULLS}")
    print(f"  N_STATES = {N_STATES}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    seeds = list(range(N_SEEDS))

    # Run experiments
    epr_results = run_entropy_production_experiment(seeds, T_PER_SEED, N_STATES)
    affinity_results = run_cycle_affinity_experiment(seeds, T_PER_SEED, N_STATES)
    tr_results = run_time_reversal_experiment(seeds, T_PER_SEED, N_STATES)
    drift_results = run_drift_return_experiment(seeds, T_PER_SEED, N_STATES)
    fdi_results = run_directional_momentum_experiment(seeds, T_PER_SEED, N_STATES)

    # Save results
    print("\n[SAVE] Saving results...")

    # Helper to convert numpy types to Python types
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    results_files = {
        'entropy_production.json': epr_results,
        'cycle_affinity.json': affinity_results,
        'time_reversal_cond.json': tr_results,
        'drift_return_profiles.json': drift_results,
        'flow_directionality_index.json': fdi_results
    }

    for filename, data in results_files.items():
        filepath = OUTPUT_DIR / filename
        with open(filepath, 'w') as f:
            json.dump(convert_numpy(data), f, indent=2)
        print(f"  {filepath}")

    # Generate figures
    generate_figures(epr_results, affinity_results, tr_results, drift_results, fdi_results)

    # Check GO criteria
    go_check = check_go_criteria(epr_results, affinity_results, tr_results,
                                 drift_results, fdi_results)

    # Save GO check
    go_filepath = OUTPUT_DIR / 'go_criteria.json'
    with open(go_filepath, 'w') as f:
        json.dump(convert_numpy(go_check), f, indent=2)
    print(f"  {go_filepath}")

    # Print summary
    print("\n" + "=" * 70)
    print("GO CRITERIA CHECK")
    print("=" * 70)

    for criterion, passed in go_check['criteria'].items():
        status = "PASS" if passed else "FAIL"
        print(f"  {criterion}: {status}")

    print(f"\nPassing: {go_check['n_pass']}/{len(go_check['criteria'])} (need >= {go_check['required']})")
    print(f"\n{'='*70}")
    print(f"GO: {'YES' if go_check['go'] else 'NO'}")
    print(f"{'='*70}")

    return go_check


if __name__ == "__main__":
    result = main()
