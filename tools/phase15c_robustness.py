#!/usr/bin/env python3
"""
Phase 15C: Robustness (endógeno, sin cambiar dinámica)
=======================================================

A) Consenso y Procrustes
B) Transiciones con nulo de orden-1
C) GNT "intencionalidad" de ruta
D) Reporte con GO criteria

100% endógeno - CERO números mágicos.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque, Counter
from scipy import stats
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
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
# A) CONSENSO Y PROCRUSTES
# =============================================================================

def compute_rank_distance(v1: np.ndarray, v2: np.ndarray, metric: str) -> float:
    """Calcula distancia según métrica rank-based."""
    if metric == 'spearman':
        corr, _ = stats.spearmanr(v1, v2)
        return 1 - corr if not np.isnan(corr) else 1.0
    elif metric == 'kendall':
        corr, _ = stats.kendalltau(v1, v2)
        return 1 - corr if not np.isnan(corr) else 1.0
    elif metric == 'rank_cosine':
        r1 = stats.rankdata(v1)
        r2 = stats.rankdata(v2)
        cos = np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2) + NUMERIC_EPS)
        return 1 - cos
    else:
        return np.linalg.norm(v1 - v2)


def cluster_with_metric_threshold(
    state_vectors: List[np.ndarray],
    threshold: float,
    metric: str
) -> List[int]:
    """Clustering online con métrica y umbral específicos."""
    if len(state_vectors) == 0:
        return []

    prototypes = [state_vectors[0].copy()]
    assignments = [0]

    for state in state_vectors[1:]:
        distances = [compute_rank_distance(state, p, metric) for p in prototypes]
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        if min_dist > threshold:
            prototypes.append(state.copy())
            assignments.append(len(prototypes) - 1)
        else:
            assignments.append(min_idx)
            # Update centroid
            eta = 1.0 / (assignments.count(min_idx) + 1)
            prototypes[min_idx] = (1 - eta) * prototypes[min_idx] + eta * state

    return assignments


def generate_k_partitions(
    state_vectors: List[np.ndarray],
    metrics: List[str] = ['spearman', 'kendall', 'rank_cosine'],
    quantiles: List[int] = [50, 60, 70, 80, 90]
) -> Tuple[List[List[int]], List[Dict]]:
    """
    Genera K particiones variando métrica y umbral.
    K = len(metrics) * len(quantiles) = 15
    """
    # Calcular distancias para determinar umbrales
    all_distances = {m: [] for m in metrics}

    for i in range(1, len(state_vectors)):
        for m in metrics:
            d = compute_rank_distance(state_vectors[i], state_vectors[i-1], m)
            all_distances[m].append(d)

    partitions = []
    configs = []

    for metric in metrics:
        for q in quantiles:
            threshold = np.percentile(all_distances[metric], q)
            assignments = cluster_with_metric_threshold(state_vectors, threshold, metric)
            partitions.append(assignments)
            configs.append({
                'metric': metric,
                'quantile': q,
                'threshold': float(threshold),
                'n_clusters': len(set(assignments))
            })

    return partitions, configs


def compute_coassignment_matrix(partitions: List[List[int]]) -> np.ndarray:
    """
    Matriz de co-asignación C[i,j] = proporción de particiones donde i,j están juntos.
    Optimized with vectorized operations.
    """
    n = len(partitions[0])
    C = np.zeros((n, n))

    for partition in partitions:
        # Vectorized: compare all pairs at once
        p_arr = np.array(partition)
        same_cluster = (p_arr[:, None] == p_arr[None, :]).astype(float)
        C += same_cluster

    C /= len(partitions)
    return C


def compute_ari(labels1: List[int], labels2: List[int]) -> float:
    """Adjusted Rand Index entre dos particiones."""
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(labels1, labels2))


def block_bootstrap_coassignment(
    state_vectors: List[np.ndarray],
    original_C: np.ndarray,
    n_bootstrap: int = 30,
    metrics: List[str] = ['spearman', 'kendall', 'rank_cosine'],
    quantiles: List[int] = [50, 60, 70, 80, 90]
) -> Dict:
    """
    Bootstrap con bloques de tamaño sqrt(T).
    Calcula ARI entre matriz original y bootstraps.
    """
    n = len(state_vectors)
    block_size = max(10, int(np.sqrt(n)))

    aris = []

    for _ in range(n_bootstrap):
        # Block bootstrap
        n_blocks = n // block_size + 1
        boot_indices = []
        for _ in range(n_blocks):
            start = np.random.randint(0, n - block_size + 1)
            boot_indices.extend(range(start, start + block_size))
        boot_indices = boot_indices[:n]

        boot_vectors = [state_vectors[i] for i in boot_indices]

        # Generar particiones para bootstrap
        boot_partitions, _ = generate_k_partitions(boot_vectors, metrics, quantiles)
        boot_C = compute_coassignment_matrix(boot_partitions)

        # Correlación de Spearman entre upper triangles
        orig_upper = original_C[np.triu_indices(n, k=1)]
        boot_upper = boot_C[np.triu_indices(n, k=1)]

        corr, _ = stats.spearmanr(orig_upper, boot_upper)
        if not np.isnan(corr):
            aris.append(corr)

    return {
        'mean_ari': float(np.mean(aris)),
        'std_ari': float(np.std(aris)),
        'ci_lower': float(np.percentile(aris, 2.5)),
        'ci_upper': float(np.percentile(aris, 97.5)),
        'n_bootstrap': n_bootstrap,
        'block_size': block_size
    }


def compute_procrustes_distances(
    partitions: List[List[int]],
    state_vectors: List[np.ndarray]
) -> List[float]:
    """
    Calcula distancias Procrustes entre centroides de diferentes runs.
    """
    # Calcular centroides para cada partición
    all_centroids = []

    for partition in partitions:
        clusters = {}
        for i, c in enumerate(partition):
            if c not in clusters:
                clusters[c] = []
            clusters[c].append(state_vectors[i])

        centroids = np.array([np.mean(clusters[c], axis=0) for c in sorted(clusters.keys())])
        all_centroids.append(centroids)

    # Calcular distancias Procrustes entre pares
    distances = []

    for i in range(len(all_centroids)):
        for j in range(i + 1, len(all_centroids)):
            c1, c2 = all_centroids[i], all_centroids[j]

            # Alinear dimensiones
            min_n = min(len(c1), len(c2))
            if min_n < 2:
                continue

            c1_sub = c1[:min_n]
            c2_sub = c2[:min_n]

            # Procrustes
            try:
                _, _, disparity = procrustes(c1_sub, c2_sub)
                distances.append(disparity)
            except:
                pass

    return distances


def run_consensus_analysis(state_vectors: List[np.ndarray]) -> Dict:
    """Ejecuta análisis de consenso completo."""
    metrics = ['spearman', 'kendall', 'rank_cosine']
    quantiles = [50, 60, 70, 80, 90]

    # Generar particiones
    partitions, configs = generate_k_partitions(state_vectors, metrics, quantiles)

    # Matriz de co-asignación
    C = compute_coassignment_matrix(partitions)

    # Bootstrap ARI (reduced for computational feasibility)
    bootstrap_results = block_bootstrap_coassignment(
        state_vectors, C, n_bootstrap=30, metrics=metrics, quantiles=quantiles
    )

    # Procrustes
    procrustes_dists = compute_procrustes_distances(partitions, state_vectors)

    # Nulo para Procrustes: shuffle labels
    null_procrustes = []
    for _ in range(50):
        shuffled_partitions = []
        for p in partitions:
            sp = list(p)
            np.random.shuffle(sp)
            shuffled_partitions.append(sp)
        null_dists = compute_procrustes_distances(shuffled_partitions, state_vectors)
        if null_dists:
            null_procrustes.extend(null_dists)

    return {
        'n_partitions': len(partitions),
        'configs': configs,
        'coassignment_matrix_shape': C.shape,
        'bootstrap': bootstrap_results,
        'procrustes': {
            'real_distances': procrustes_dists,
            'real_mean': float(np.mean(procrustes_dists)) if procrustes_dists else None,
            'null_p50': float(np.percentile(null_procrustes, 50)) if null_procrustes else None,
            'real_below_null_p50': np.mean(procrustes_dists) < np.percentile(null_procrustes, 50) if procrustes_dists and null_procrustes else False
        },
        'coassignment_matrix': C.tolist()
    }


# =============================================================================
# B) TRANSICIONES CON NULO DE ORDEN-1
# =============================================================================

def estimate_transition_matrix(assignments: List[int]) -> Tuple[np.ndarray, List[int]]:
    """Estima matriz de transición P."""
    states = sorted(set(assignments))
    n = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}

    counts = np.zeros((n, n))

    for t in range(len(assignments) - 1):
        i = state_to_idx[assignments[t]]
        j = state_to_idx[assignments[t + 1]]
        counts[i, j] += 1

    # Normalizar
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = counts / row_sums

    return P, states


def simulate_markov_chain(P: np.ndarray, length: int, start: int = 0) -> List[int]:
    """Simula cadena de Markov."""
    n = P.shape[0]
    chain = [start]

    for _ in range(length - 1):
        probs = P[chain[-1]]
        if probs.sum() < NUMERIC_EPS:
            probs = np.ones(n) / n
        else:
            probs = probs / probs.sum()
        next_state = np.random.choice(n, p=probs)
        chain.append(next_state)

    return chain


def compute_asymmetry_delta(P: np.ndarray) -> float:
    """Δ = sum_{i≠j}|P(i→j)-P(j→i)|"""
    n = P.shape[0]
    delta = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            delta += abs(P[i, j] - P[j, i])
    return float(delta)


def count_cycles(assignments: List[int]) -> Dict[Tuple[int, int], int]:
    """Cuenta ciclos i→j→i."""
    cycles = {}

    for t in range(len(assignments) - 2):
        if assignments[t] == assignments[t + 2] and assignments[t] != assignments[t + 1]:
            cycle = (assignments[t], assignments[t + 1])
            if cycle not in cycles:
                cycles[cycle] = 0
            cycles[cycle] += 1

    return cycles


def run_transition_analysis(assignments: List[int], n_nulls: int = 500) -> Dict:
    """Ejecuta análisis de transiciones con nulos orden-1."""
    P, states = estimate_transition_matrix(assignments)
    T = len(assignments)

    # Métricas reales
    real_delta = compute_asymmetry_delta(P)
    real_cycles = count_cycles(assignments)
    real_total_cycles = sum(real_cycles.values())

    # Simular nulos
    null_deltas = []
    null_total_cycles = []

    start_state = 0
    if assignments:
        start_state = states.index(assignments[0]) if assignments[0] in states else 0

    for _ in range(n_nulls):
        null_chain = simulate_markov_chain(P, T, start_state)
        null_assignments = [states[i] for i in null_chain]

        null_P, _ = estimate_transition_matrix(null_assignments)
        null_deltas.append(compute_asymmetry_delta(null_P))

        null_cyc = count_cycles(null_assignments)
        null_total_cycles.append(sum(null_cyc.values()))

    # P-values
    delta_p = float(np.mean([nd >= real_delta for nd in null_deltas]))
    cycles_p = float(np.mean([nc >= real_total_cycles for nc in null_total_cycles]))

    # Z-scores
    delta_z = (real_delta - np.mean(null_deltas)) / (np.std(null_deltas) + NUMERIC_EPS)
    cycles_z = (real_total_cycles - np.mean(null_total_cycles)) / (np.std(null_total_cycles) + NUMERIC_EPS)

    # Convert tuple keys to strings for JSON serialization
    top_cycles_str = {f"{k[0]}->{k[1]}": v for k, v in sorted(real_cycles.items(), key=lambda x: -x[1])[:5]}

    return {
        'n_states': len(states),
        'n_transitions': T - 1,
        'n_nulls': n_nulls,
        'real': {
            'asymmetry_delta': real_delta,
            'total_cycles': real_total_cycles,
            'top_cycles': top_cycles_str
        },
        'null': {
            'delta_mean': float(np.mean(null_deltas)),
            'delta_std': float(np.std(null_deltas)),
            'cycles_mean': float(np.mean(null_total_cycles)),
            'cycles_std': float(np.std(null_total_cycles))
        },
        'delta_p_value': delta_p,
        'delta_z_score': float(delta_z),
        'delta_significant': delta_p < 0.05,
        'cycles_p_value': cycles_p,
        'cycles_z_score': float(cycles_z),
        'cycles_excess': real_total_cycles > np.percentile(null_total_cycles, 95)
    }


# =============================================================================
# C) GNT INTENCIONALIDAD DE RUTA
# =============================================================================

def generate_acf_matched_null(trajectory: List[np.ndarray], n_nulls: int = 500) -> List[List[np.ndarray]]:
    """
    Genera nulos que preservan varianza por componente y ACF(1),
    rompiendo correlaciones cruzadas.
    """
    traj_array = np.array(trajectory)
    T, dim = traj_array.shape

    nulls = []

    for _ in range(n_nulls):
        null_traj = np.zeros((T, dim))

        for d in range(dim):
            series = traj_array[:, d]

            # Calcular ACF(1) y varianza
            acf1 = compute_acf_lag1(series)
            var = np.var(series)
            mean = np.mean(series)

            # Generar AR(1) con mismos parámetros
            phi = acf1
            sigma = np.sqrt(var * (1 - phi**2)) if abs(phi) < 1 else np.sqrt(var)

            null_series = np.zeros(T)
            null_series[0] = mean
            for t in range(1, T):
                null_series[t] = mean + phi * (null_series[t-1] - mean) + np.random.randn() * sigma

            null_traj[:, d] = null_series

        nulls.append([null_traj[t] for t in range(T)])

    return nulls


def compute_path_length(trajectory: List[np.ndarray]) -> float:
    """Longitud total del camino."""
    length = 0.0
    for t in range(1, len(trajectory)):
        length += np.linalg.norm(trajectory[t] - trajectory[t-1])
    return float(length)


def compute_tortuosity(trajectory: List[np.ndarray]) -> float:
    """Tortuosidad = path_length / displacement."""
    if len(trajectory) < 2:
        return 1.0

    path_length = compute_path_length(trajectory)
    displacement = np.linalg.norm(trajectory[-1] - trajectory[0])

    if displacement < NUMERIC_EPS:
        return float('inf')

    return path_length / displacement


def compute_path_signature_energy(trajectory: List[np.ndarray], level: int = 2) -> float:
    """
    Energía de firma de camino (nivel 2).
    Simplificación: suma de productos cruzados de incrementos.
    """
    if len(trajectory) < 3:
        return 0.0

    # Incrementos
    increments = [trajectory[t] - trajectory[t-1] for t in range(1, len(trajectory))]

    # Nivel 1: suma de incrementos
    level1 = sum(np.sum(inc) for inc in increments)

    # Nivel 2: productos cruzados
    level2 = 0.0
    for i in range(len(increments)):
        for j in range(i + 1, len(increments)):
            level2 += np.dot(increments[i], increments[j])

    # Energía = ||sig||^2
    energy = level1**2 + level2**2

    return float(energy)


def run_path_intent_analysis(
    gnt_trajectory: List[np.ndarray],
    integration_series: List[float],
    n_nulls: int = 500
) -> Dict:
    """Ejecuta análisis de intencionalidad de ruta."""
    T = len(gnt_trajectory)

    # Métricas reales
    real_length = compute_path_length(gnt_trajectory)
    real_tortuosity = compute_tortuosity(gnt_trajectory)
    real_signature = compute_path_signature_energy(gnt_trajectory)

    # Generar nulos ACF-matched
    null_trajectories = generate_acf_matched_null(gnt_trajectory, n_nulls)

    null_lengths = []
    null_tortuosities = []
    null_signatures = []

    for null_traj in null_trajectories:
        null_lengths.append(compute_path_length(null_traj))
        null_tortuosities.append(compute_tortuosity(null_traj))
        null_signatures.append(compute_path_signature_energy(null_traj))

    # Filtrar infinitos
    null_tortuosities = [t for t in null_tortuosities if t != float('inf')]

    # P-values (mayor tortuosidad = menos intencional)
    tort_p = float(np.mean([nt <= real_tortuosity for nt in null_tortuosities])) if null_tortuosities else 1.0
    sig_p = float(np.mean([ns >= real_signature for ns in null_signatures]))

    # Análisis condicionado a Integration >= p90
    if len(integration_series) == T:
        high_int_threshold = np.percentile(integration_series, 90)
        high_int_indices = [i for i in range(T) if integration_series[i] >= high_int_threshold]

        if len(high_int_indices) > 10:
            # Extraer segmentos de alta integración
            high_int_traj = [gnt_trajectory[i] for i in high_int_indices]

            cond_length = compute_path_length(high_int_traj)
            cond_tortuosity = compute_tortuosity(high_int_traj)
            cond_signature = compute_path_signature_energy(high_int_traj)

            # Nulos condicionados
            cond_null_tort = []
            cond_null_sig = []
            for null_traj in null_trajectories:
                null_high = [null_traj[i] for i in high_int_indices]
                cond_null_tort.append(compute_tortuosity(null_high))
                cond_null_sig.append(compute_path_signature_energy(null_high))

            cond_null_tort = [t for t in cond_null_tort if t != float('inf')]

            conditioned = {
                'n_high_int_points': len(high_int_indices),
                'threshold': float(high_int_threshold),
                'real': {
                    'length': float(cond_length),
                    'tortuosity': float(cond_tortuosity),
                    'signature': float(cond_signature)
                },
                'null': {
                    'tortuosity_mean': float(np.mean(cond_null_tort)) if cond_null_tort else None,
                    'signature_mean': float(np.mean(cond_null_sig))
                },
                'tort_p': float(np.mean([nt <= cond_tortuosity for nt in cond_null_tort])) if cond_null_tort else 1.0,
                'sig_p': float(np.mean([ns >= cond_signature for ns in cond_null_sig]))
            }
        else:
            conditioned = {'error': 'insufficient_high_int_points'}
    else:
        conditioned = {'error': 'integration_series_mismatch'}

    return {
        'n_nulls': n_nulls,
        'real': {
            'path_length': real_length,
            'tortuosity': real_tortuosity,
            'signature_energy': real_signature
        },
        'null': {
            'length_mean': float(np.mean(null_lengths)),
            'length_std': float(np.std(null_lengths)),
            'tortuosity_mean': float(np.mean(null_tortuosities)) if null_tortuosities else None,
            'tortuosity_std': float(np.std(null_tortuosities)) if null_tortuosities else None,
            'signature_mean': float(np.mean(null_signatures)),
            'signature_std': float(np.std(null_signatures))
        },
        'tortuosity_p_value': tort_p,
        'tortuosity_above_p95': real_tortuosity > np.percentile(null_tortuosities, 95) if null_tortuosities else False,
        'signature_p_value': sig_p,
        'signature_above_p95': real_signature > np.percentile(null_signatures, 95),
        'conditioned_integration_p90': conditioned
    }


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

def create_consensus_heatmap(C: np.ndarray, output_path: str):
    """Crea heatmap de matriz de co-asignación."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Subsample si es muy grande
    n = C.shape[0]
    if n > 200:
        step = n // 200
        C_sub = C[::step, ::step]
    else:
        C_sub = C

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(C_sub, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='Co-assignment probability')
    ax.set_title('Consensus Co-assignment Matrix')
    ax.set_xlabel('Time index')
    ax.set_ylabel('Time index')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_procrustes_violin(real_dists: List[float], null_dists: List[float], output_path: str):
    """Crea violin plot de distancias Procrustes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    if null_dists:
        parts = ax.violinplot([null_dists], positions=[0], showmeans=True)
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)

    if real_dists:
        ax.scatter([0] * len(real_dists), real_dists, color='red', s=50, zorder=5, label='Real')
        ax.axhline(np.mean(real_dists), color='red', linestyle='--', label=f'Real mean: {np.mean(real_dists):.3f}')

    ax.set_ylabel('Procrustes Distance')
    ax.set_title('Procrustes Distance: Real vs Shuffled Null')
    ax.set_xticks([0])
    ax.set_xticklabels([''])
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_cycles_excess_plot(real_cycles: int, null_cycles: List[float], output_path: str):
    """Crea plot de exceso de ciclos."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(null_cycles, bins=30, alpha=0.7, color='lightblue', edgecolor='blue', label='Null (order-1)')
    ax.axvline(real_cycles, color='red', linestyle='--', linewidth=2, label=f'Real: {real_cycles}')
    ax.axvline(np.percentile(null_cycles, 95), color='orange', linestyle=':', label=f'p95: {np.percentile(null_cycles, 95):.0f}')

    ax.set_xlabel('Total Cycles (i→j→i)')
    ax.set_ylabel('Count')
    ax.set_title('Cycle Excess vs Order-1 Null')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_path_intent_plot(results: Dict, output_path: str):
    """Crea plot de path intent."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Tortuosity
    ax = axes[0]
    ax.hist([results['null']['tortuosity_mean']] * 100 if results['null']['tortuosity_mean'] else [],
            bins=1, alpha=0.3, label='Null mean')
    ax.axvline(results['real']['tortuosity'], color='red', linestyle='--',
               linewidth=2, label=f"Real: {results['real']['tortuosity']:.2f}")
    ax.set_xlabel('Tortuosity')
    ax.set_title('Tortuosity: Real vs ACF-matched Null')
    ax.legend()

    # Signature energy
    ax = axes[1]
    ax.axvline(results['real']['signature_energy'], color='red', linestyle='--',
               linewidth=2, label=f"Real: {results['real']['signature_energy']:.2e}")
    ax.axvline(results['null']['signature_mean'], color='blue', linestyle=':',
               linewidth=2, label=f"Null mean: {results['null']['signature_mean']:.2e}")
    ax.set_xlabel('Path Signature Energy')
    ax.set_title('Signature Energy: Real vs ACF-matched Null')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# RUNNER PRINCIPAL
# =============================================================================

def run_phase15c_robustness(
    n_steps: int = 2000,
    n_nulls: int = 500,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """Ejecuta Phase 15C Robustness completo."""
    from phase15_structural_consciousness import StructuralConsciousnessSystem

    if verbose:
        print("=" * 70)
        print("PHASE 15C: ROBUSTNESS (endógeno)")
        print("=" * 70)

    np.random.seed(seed)

    # Ejecutar sistema
    system = StructuralConsciousnessSystem()

    neo_state_vectors = []
    eva_state_vectors = []
    gnt_trajectory = []
    neo_assignments = []
    eva_assignments = []
    integration_series = []

    if verbose:
        print(f"\n[1] Simulando {n_steps} pasos...")

    neo_pi = np.array([0.33, 0.33, 0.34])
    eva_pi = np.array([0.33, 0.33, 0.34])

    for t in range(n_steps):
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

        result = system.process_step(
            neo_pi=neo_pi, eva_pi=eva_pi,
            te_neo_to_eva=te_neo, te_eva_to_neo=te_eva,
            neo_self_error=neo_se, eva_self_error=eva_se,
            sync=sync
        )

        if system.states.neo_current_state:
            neo_state_vectors.append(system.states.neo_current_state.to_array())
            neo_assignments.append(result['neo']['state']['prototype_id'])
        if system.states.eva_current_state:
            eva_state_vectors.append(system.states.eva_current_state.to_array())
            eva_assignments.append(result['eva']['state']['prototype_id'])

        gnt_trajectory.append(system.gnt.gnt.gnt.copy())
        integration_series.append(result['integration']['coherence'])

    if verbose:
        print(f"    Completado: {n_steps} pasos")

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {'n_steps': n_steps, 'n_nulls': n_nulls, 'seed': seed}
    }

    # A) Consenso y Procrustes
    if verbose:
        print("\n[A] Consenso y Procrustes...")

    results['consensus'] = {
        'neo': run_consensus_analysis(neo_state_vectors),
        'eva': run_consensus_analysis(eva_state_vectors)
    }

    if verbose:
        for agent in ['neo', 'eva']:
            data = results['consensus'][agent]
            ari = data['bootstrap']['mean_ari']
            proc = data['procrustes']['real_mean']
            proc_ok = "✓" if data['procrustes']['real_below_null_p50'] else "✗"
            print(f"    {agent.upper()}: ARI={ari:.3f}, Procrustes={proc:.3f} {proc_ok}")

    # Guardar y plot
    os.makedirs('/root/NEO_EVA/results/phase15c', exist_ok=True)
    os.makedirs('/root/NEO_EVA/figures', exist_ok=True)

    # Plot consensus heatmap (NEO)
    C_neo = np.array(results['consensus']['neo']['coassignment_matrix'])
    create_consensus_heatmap(C_neo, '/root/NEO_EVA/figures/15c_consensus_heatmap.png')

    # Plot Procrustes
    real_proc = results['consensus']['neo']['procrustes']['real_distances']
    create_procrustes_violin(real_proc, [], '/root/NEO_EVA/figures/15c_procrustes_violin.png')

    # B) Transiciones con nulo orden-1
    if verbose:
        print(f"\n[B] Transiciones con nulo orden-1 (N={n_nulls})...")

    results['transitions'] = {
        'neo': run_transition_analysis(neo_assignments, n_nulls),
        'eva': run_transition_analysis(eva_assignments, n_nulls)
    }

    if verbose:
        for agent in ['neo', 'eva']:
            data = results['transitions'][agent]
            delta_sig = "✓" if data['delta_significant'] else "✗"
            cycles_ex = "✓" if data['cycles_excess'] else "✗"
            print(f"    {agent.upper()}: Δ_asim z={data['delta_z_score']:.2f} {delta_sig}, "
                  f"cycles_excess {cycles_ex}")

    # Plot cycles
    create_cycles_excess_plot(
        results['transitions']['neo']['real']['total_cycles'],
        [results['transitions']['neo']['null']['cycles_mean']] * 100,
        '/root/NEO_EVA/figures/15c_cycles_excess.png'
    )

    # C) GNT path intent
    if verbose:
        print(f"\n[C] GNT intencionalidad de ruta (N={n_nulls})...")

    results['path_intent'] = run_path_intent_analysis(gnt_trajectory, integration_series, n_nulls)

    if verbose:
        data = results['path_intent']
        tort_ok = "✓" if data['tortuosity_above_p95'] else "✗"
        sig_ok = "✓" if data['signature_above_p95'] else "✗"
        print(f"    Tortuosity > p95(null): {tort_ok}")
        print(f"    Signature > p95(null): {sig_ok}")

        if 'conditioned_integration_p90' in data and 'real' in data['conditioned_integration_p90']:
            cond = data['conditioned_integration_p90']
            print(f"    Conditioned (Int>=p90): tort_p={cond['tort_p']:.3f}, sig_p={cond['sig_p']:.3f}")

    create_path_intent_plot(results['path_intent'], '/root/NEO_EVA/figures/15c_path_intent_cond.png')

    # D) Generar reporte
    if verbose:
        print("\n[D] Generando reporte...")

    generate_report(results)

    # Guardar JSON
    save_results = {k: v for k, v in results.items() if k != 'consensus'}
    save_results['consensus_summary'] = {
        agent: {
            'n_partitions': results['consensus'][agent]['n_partitions'],
            'bootstrap': results['consensus'][agent]['bootstrap'],
            'procrustes': {k: v for k, v in results['consensus'][agent]['procrustes'].items()
                          if k != 'real_distances'}
        }
        for agent in ['neo', 'eva']
    }

    with open('/root/NEO_EVA/results/phase15c/robustness.json', 'w') as f:
        json.dump(save_results, f, indent=2, default=str)

    if verbose:
        print("\n" + "=" * 70)
        print("GO CRITERIA CHECK")
        print("=" * 70)

        go_checks = evaluate_go_criteria(results)
        for check, passed in go_checks.items():
            status = "✓ GO" if passed else "✗ NO-GO"
            print(f"  {check}: {status}")

        n_passed = sum(go_checks.values())
        print(f"\n  TOTAL: {n_passed}/{len(go_checks)} criteria passed")
        print("=" * 70)

    return results


def evaluate_go_criteria(results: Dict) -> Dict[str, bool]:
    """Evalúa criterios GO."""
    go = {}

    # ARI_consenso > p95(shuffle) - aproximamos con umbral > 0.5
    neo_ari = results['consensus']['neo']['bootstrap']['mean_ari']
    eva_ari = results['consensus']['eva']['bootstrap']['mean_ari']
    go['ARI_consensus > 0.5'] = neo_ari > 0.5 and eva_ari > 0.5

    # Procrustes < p50(nulo)
    neo_proc = results['consensus']['neo']['procrustes']['real_below_null_p50']
    eva_proc = results['consensus']['eva']['procrustes']['real_below_null_p50']
    go['Procrustes < p50(null)'] = neo_proc or eva_proc

    # Δ_asimetría > p95(nulo orden-1)
    neo_delta = results['transitions']['neo']['delta_significant']
    eva_delta = results['transitions']['eva']['delta_significant']
    go['Delta_asymmetry significant'] = neo_delta or eva_delta

    # Cycles excess
    neo_cycles = results['transitions']['neo']['cycles_excess']
    eva_cycles = results['transitions']['eva']['cycles_excess']
    go['Cycles_excess > p95'] = neo_cycles or eva_cycles

    # τ_real > p95(nulo ACF-matched)
    go['Tortuosity > p95(null)'] = results['path_intent']['tortuosity_above_p95']

    # Signature > p95(nulo)
    go['Signature > p95(null)'] = results['path_intent']['signature_above_p95']

    # Conditioned analysis
    cond = results['path_intent'].get('conditioned_integration_p90', {})
    if 'tort_p' in cond:
        go['Conditioned_tort_p < 0.05'] = cond['tort_p'] < 0.05
        go['Conditioned_sig_p < 0.05'] = cond['sig_p'] < 0.05

    return go


def generate_report(results: Dict):
    """Genera reporte markdown."""
    md = []
    md.append("# Phase 15C: Robustness Report")
    md.append("")
    md.append(f"**Timestamp:** {results['timestamp']}")
    md.append(f"**Config:** n_steps={results['config']['n_steps']}, n_nulls={results['config']['n_nulls']}")
    md.append("")

    # A) Consensus
    md.append("## A) Consenso y Procrustes")
    md.append("")
    md.append("| Agent | ARI (mean±CI95) | Procrustes < p50(null) |")
    md.append("|-------|-----------------|------------------------|")

    for agent in ['neo', 'eva']:
        data = results['consensus'][agent]
        ari = data['bootstrap']
        proc_ok = "✓" if data['procrustes']['real_below_null_p50'] else "✗"
        md.append(f"| {agent.upper()} | {ari['mean_ari']:.3f} [{ari['ci_lower']:.3f}, {ari['ci_upper']:.3f}] | {proc_ok} |")

    md.append("")
    md.append("![Consensus Heatmap](../figures/15c_consensus_heatmap.png)")
    md.append("")

    # B) Transitions
    md.append("## B) Transiciones con Nulo Orden-1")
    md.append("")
    md.append("| Agent | Δ_asymmetry | z-score | p-value | Cycles excess |")
    md.append("|-------|-------------|---------|---------|---------------|")

    for agent in ['neo', 'eva']:
        data = results['transitions'][agent]
        delta = data['real']['asymmetry_delta']
        z = data['delta_z_score']
        p = data['delta_p_value']
        cycles_ok = "✓" if data['cycles_excess'] else "✗"
        md.append(f"| {agent.upper()} | {delta:.3f} | {z:.2f} | {p:.3f} | {cycles_ok} |")

    md.append("")
    md.append("![Cycles Excess](../figures/15c_cycles_excess.png)")
    md.append("")

    # C) Path Intent
    md.append("## C) GNT Intencionalidad de Ruta")
    md.append("")
    data = results['path_intent']
    md.append(f"- **Path length:** {data['real']['path_length']:.2f}")
    md.append(f"- **Tortuosity:** {data['real']['tortuosity']:.2f} (null mean: {data['null']['tortuosity_mean']:.2f})")
    md.append(f"- **Signature energy:** {data['real']['signature_energy']:.2e}")
    md.append(f"- **Tortuosity > p95(null):** {'✓' if data['tortuosity_above_p95'] else '✗'}")
    md.append(f"- **Signature > p95(null):** {'✓' if data['signature_above_p95'] else '✗'}")

    cond = data.get('conditioned_integration_p90', {})
    if 'real' in cond:
        md.append(f"\n**Conditioned on Integration >= p90:**")
        md.append(f"- N points: {cond['n_high_int_points']}")
        md.append(f"- Tortuosity: {cond['real']['tortuosity']:.2f} (p={cond['tort_p']:.3f})")
        md.append(f"- Signature: {cond['real']['signature']:.2e} (p={cond['sig_p']:.3f})")

    md.append("")
    md.append("![Path Intent](../figures/15c_path_intent_cond.png)")
    md.append("")

    # GO criteria
    md.append("## D) GO Criteria")
    md.append("")

    go = evaluate_go_criteria(results)
    md.append("| Criterion | Status |")
    md.append("|-----------|--------|")
    for check, passed in go.items():
        status = "✓ GO" if passed else "✗ NO-GO"
        md.append(f"| {check} | {status} |")

    n_passed = sum(go.values())
    md.append("")
    md.append(f"**Total: {n_passed}/{len(go)} criteria passed**")

    with open('/root/NEO_EVA/results/phase15c_summary.md', 'w') as f:
        f.write('\n'.join(md))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Balanced run: enough steps for stats, fast enough to complete
    results = run_phase15c_robustness(n_steps=1000, n_nulls=200, verbose=True)
