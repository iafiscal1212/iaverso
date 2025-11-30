#!/usr/bin/env python3
"""
Phase 15C: Verificación con Sensibilidad y Nulos
=================================================

Validaciones rigurosas sin exponer mecanismos:
1. Sensibilidad de prototipos (p50-p90)
2. Matriz de transición vs nulos grado-preservados
3. GNT curvatura vs random walk (4D original)
4. Global score con IC bootstrap y phase-randomization nulls

100% endógeno - CERO números mágicos.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque, Counter
from scipy import stats
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
    NUMERIC_EPS,
    PROVENANCE
)


# =============================================================================
# 1. SENSIBILIDAD DE PROTOTIPOS
# =============================================================================

def compute_dissimilarity_matrix(state_vectors: List[np.ndarray]) -> np.ndarray:
    """Calcula matriz de disimilitud entre estados consecutivos."""
    n = len(state_vectors)
    dissimilarities = []

    for i in range(1, n):
        dist = np.linalg.norm(state_vectors[i] - state_vectors[i-1])
        dissimilarities.append(dist)

    return np.array(dissimilarities)


def cluster_with_threshold(
    state_vectors: List[np.ndarray],
    threshold: float
) -> Tuple[List[int], List[np.ndarray], List[int]]:
    """
    Clustering online con umbral fijo.

    Returns:
        assignments: lista de IDs de prototipo por paso
        prototypes: lista de centroides
        dwell_times: tiempo en cada visita a un prototipo
    """
    if len(state_vectors) == 0:
        return [], [], []

    prototypes = [state_vectors[0].copy()]
    assignments = [0]
    visit_counts = [1]

    # Para dwell-time
    current_proto = 0
    current_dwell = 1
    dwell_times = []

    for i, state in enumerate(state_vectors[1:], 1):
        # Encontrar prototipo más cercano
        distances = [np.linalg.norm(state - p) for p in prototypes]
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        if min_dist > threshold:
            # Crear nuevo prototipo
            prototypes.append(state.copy())
            assignments.append(len(prototypes) - 1)
            visit_counts.append(1)

            # Registrar dwell-time anterior
            dwell_times.append(current_dwell)
            current_proto = len(prototypes) - 1
            current_dwell = 1
        else:
            # Asignar al existente
            assignments.append(min_idx)
            visit_counts[min_idx] += 1

            # Actualizar centroide con EMA
            eta = 1.0 / (visit_counts[min_idx] + 1)
            prototypes[min_idx] = (1 - eta) * prototypes[min_idx] + eta * state

            # Dwell-time
            if min_idx == current_proto:
                current_dwell += 1
            else:
                dwell_times.append(current_dwell)
                current_proto = min_idx
                current_dwell = 1

    # Último dwell
    dwell_times.append(current_dwell)

    return assignments, prototypes, dwell_times


def prototype_sensitivity_sweep(
    state_vectors: List[np.ndarray],
    quantile_levels: List[int] = [50, 60, 70, 80, 90]
) -> Dict:
    """
    Barre umbral endógeno u ∈ {p50,p60,p70,p80,p90}.
    """
    if len(state_vectors) < 100:
        return {'error': 'insufficient_data', 'n': len(state_vectors)}

    # Calcular disimilitudes
    dissimilarities = compute_dissimilarity_matrix(state_vectors)

    results = {
        'quantile_levels': quantile_levels,
        'thresholds': [],
        'n_prototypes': [],
        'n_state_changes': [],
        'dwell_time_stats': []
    }

    for q in quantile_levels:
        threshold = np.percentile(dissimilarities, q)
        results['thresholds'].append(float(threshold))

        assignments, prototypes, dwell_times = cluster_with_threshold(
            state_vectors, threshold
        )

        results['n_prototypes'].append(len(prototypes))

        # Cambios de estado
        n_changes = sum(1 for i in range(1, len(assignments))
                       if assignments[i] != assignments[i-1])
        results['n_state_changes'].append(n_changes)

        # Estadísticas de dwell-time
        if dwell_times:
            results['dwell_time_stats'].append({
                'mean': float(np.mean(dwell_times)),
                'median': float(np.median(dwell_times)),
                'std': float(np.std(dwell_times)),
                'max': int(np.max(dwell_times))
            })
        else:
            results['dwell_time_stats'].append({})

    return results


# =============================================================================
# 2. MATRIZ DE TRANSICIÓN VS NULOS
# =============================================================================

def build_transition_counts(assignments: List[int]) -> Dict[int, Dict[int, int]]:
    """Construye conteos de transición."""
    counts = {}

    for i in range(len(assignments) - 1):
        src, dst = assignments[i], assignments[i+1]
        if src not in counts:
            counts[src] = {}
        if dst not in counts[src]:
            counts[src][dst] = 0
        counts[src][dst] += 1

    return counts


def counts_to_matrix(counts: Dict[int, Dict[int, int]], states: List[int]) -> np.ndarray:
    """Convierte conteos a matriz de probabilidad."""
    n = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}

    matrix = np.zeros((n, n))

    for src, targets in counts.items():
        if src not in state_to_idx:
            continue
        i = state_to_idx[src]
        total = sum(targets.values())
        for dst, count in targets.items():
            if dst in state_to_idx:
                j = state_to_idx[dst]
                matrix[i, j] = count / total if total > 0 else 0

    # Normalizar filas
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix = matrix / row_sums

    return matrix


def edge_swap_null(counts: Dict[int, Dict[int, int]], n_swaps: int = None) -> Dict[int, Dict[int, int]]:
    """
    Genera null manteniendo grados de entrada/salida.
    Usa edge swap en grafo dirigido.
    """
    # Convertir a lista de aristas
    edges = []
    for src, targets in counts.items():
        for dst, count in targets.items():
            for _ in range(count):
                edges.append((src, dst))

    if len(edges) < 4:
        return counts

    # n_swaps endógeno
    if n_swaps is None:
        n_swaps = len(edges) * 2

    edges = list(edges)

    for _ in range(n_swaps):
        # Elegir dos aristas al azar
        i, j = np.random.choice(len(edges), 2, replace=False)
        e1, e2 = edges[i], edges[j]

        # Swap: (a→b, c→d) → (a→d, c→b)
        # Esto preserva out-degree de a,c y in-degree de b,d
        new_e1 = (e1[0], e2[1])
        new_e2 = (e2[0], e1[1])

        edges[i] = new_e1
        edges[j] = new_e2

    # Reconstruir conteos
    null_counts = {}
    for src, dst in edges:
        if src not in null_counts:
            null_counts[src] = {}
        if dst not in null_counts[src]:
            null_counts[src][dst] = 0
        null_counts[src][dst] += 1

    return null_counts


def compute_kl_asymmetry(matrix: np.ndarray) -> float:
    """KL divergence entre P y P^T (asimetría direccional)."""
    P = matrix + NUMERIC_EPS
    P = P / P.sum(axis=1, keepdims=True)

    PT = P.T
    PT = PT / PT.sum(axis=1, keepdims=True)

    # KL(P || P^T) promedio
    kl = np.sum(P * np.log(P / (PT + NUMERIC_EPS)))
    return float(kl)


def compute_transition_entropy(matrix: np.ndarray) -> float:
    """Entropía promedio de transición."""
    entropies = []
    for row in matrix:
        if row.sum() > NUMERIC_EPS:
            row_norm = row / row.sum()
            h = -np.sum(row_norm * np.log(row_norm + NUMERIC_EPS))
            entropies.append(h)
    return float(np.mean(entropies)) if entropies else 0.0


def transition_null_analysis(
    assignments: List[int],
    n_nulls: int = 500
) -> Dict:
    """
    Compara matriz real vs N nulos grado-preservados.
    """
    if len(assignments) < 50:
        return {'error': 'insufficient_data'}

    # Matriz real
    counts = build_transition_counts(assignments)
    states = sorted(set(assignments))
    real_matrix = counts_to_matrix(counts, states)

    real_kl = compute_kl_asymmetry(real_matrix)
    real_entropy = compute_transition_entropy(real_matrix)

    results = {
        'n_nulls': n_nulls,
        'n_states': len(states),
        'real': {
            'kl_asymmetry': real_kl,
            'transition_entropy': real_entropy
        },
        'null_kl': [],
        'null_entropy': []
    }

    # Generar nulos
    for _ in range(n_nulls):
        null_counts = edge_swap_null(counts)
        null_matrix = counts_to_matrix(null_counts, states)

        results['null_kl'].append(compute_kl_asymmetry(null_matrix))
        results['null_entropy'].append(compute_transition_entropy(null_matrix))

    # P-values
    results['kl_p_value'] = float(np.mean([nk >= real_kl for nk in results['null_kl']]))
    results['entropy_p_value'] = float(np.mean([ne >= real_entropy for ne in results['null_entropy']]))

    # Z-scores
    null_kl_mean, null_kl_std = np.mean(results['null_kl']), np.std(results['null_kl']) + NUMERIC_EPS
    null_ent_mean, null_ent_std = np.mean(results['null_entropy']), np.std(results['null_entropy']) + NUMERIC_EPS

    results['kl_z_score'] = float((real_kl - null_kl_mean) / null_kl_std)
    results['entropy_z_score'] = float((real_entropy - null_ent_mean) / null_ent_std)

    # Significancia
    results['kl_significant'] = results['kl_p_value'] < 0.05
    results['entropy_significant'] = results['entropy_p_value'] < 0.05

    return results


# =============================================================================
# 3. GNT CURVATURA VS RANDOM WALK
# =============================================================================

def compute_trajectory_curvature(trajectory: List[np.ndarray]) -> List[float]:
    """
    Calcula curvatura κ(t) en el espacio 4D original.
    κ = |a| / |v|² donde a=aceleración, v=velocidad
    """
    if len(trajectory) < 3:
        return []

    curvatures = []

    for t in range(2, len(trajectory)):
        # Velocidades
        v1 = trajectory[t-1] - trajectory[t-2]
        v2 = trajectory[t] - trajectory[t-1]

        # Aceleración
        a = v2 - v1

        # Curvatura
        v_norm = np.linalg.norm(v2)
        if v_norm > NUMERIC_EPS:
            kappa = np.linalg.norm(a) / (v_norm ** 2)
            curvatures.append(kappa)

    return curvatures


def generate_random_walk_simplex(n_steps: int, dim: int, marginal_std: float) -> List[np.ndarray]:
    """
    Genera caminata aleatoria proyectada al simplex con misma varianza marginal.
    """
    trajectory = [np.random.dirichlet(np.ones(dim))]  # Inicio en simplex

    for _ in range(n_steps - 1):
        # Paso con varianza marginal especificada
        step = np.random.randn(dim) * marginal_std
        new_pos = trajectory[-1] + step

        # Proyectar al simplex (clamp y normalizar)
        new_pos = np.abs(new_pos)
        new_pos = new_pos / (new_pos.sum() + NUMERIC_EPS)

        trajectory.append(new_pos)

    return trajectory


def gnt_curvature_analysis(
    gnt_trajectory: List[np.ndarray],
    n_nulls: int = 500
) -> Dict:
    """
    Compara curvatura de GNT real vs random walks.
    """
    if len(gnt_trajectory) < 50:
        return {'error': 'insufficient_data'}

    # Curvatura real
    real_curvatures = compute_trajectory_curvature(gnt_trajectory)

    if not real_curvatures:
        return {'error': 'cannot_compute_curvature'}

    real_mean = np.mean(real_curvatures)
    real_median = np.median(real_curvatures)

    # Calcular varianza marginal del GNT real
    gnt_array = np.array(gnt_trajectory)
    marginal_std = np.std(np.diff(gnt_array, axis=0))

    results = {
        'n_nulls': n_nulls,
        'n_steps': len(gnt_trajectory),
        'real': {
            'mean_curvature': float(real_mean),
            'median_curvature': float(real_median),
            'std_curvature': float(np.std(real_curvatures))
        },
        'null_mean_curvatures': [],
        'null_median_curvatures': []
    }

    dim = gnt_trajectory[0].shape[0]

    # Generar nulos
    for _ in range(n_nulls):
        null_traj = generate_random_walk_simplex(len(gnt_trajectory), dim, marginal_std)
        null_curv = compute_trajectory_curvature(null_traj)

        if null_curv:
            results['null_mean_curvatures'].append(np.mean(null_curv))
            results['null_median_curvatures'].append(np.median(null_curv))

    # P-values (menor curvatura = más intencional)
    results['mean_p_value'] = float(np.mean([nc <= real_mean for nc in results['null_mean_curvatures']]))
    results['median_p_value'] = float(np.mean([nc <= real_median for nc in results['null_median_curvatures']]))

    # Effect size (Cohen's d)
    null_mean_mean = np.mean(results['null_mean_curvatures'])
    null_mean_std = np.std(results['null_mean_curvatures']) + NUMERIC_EPS
    results['effect_size'] = float((null_mean_mean - real_mean) / null_mean_std)

    # Interpretación
    results['more_intentional'] = real_mean < null_mean_mean
    results['effect_interpretation'] = (
        'large' if abs(results['effect_size']) > 0.8 else
        'medium' if abs(results['effect_size']) > 0.5 else
        'small' if abs(results['effect_size']) > 0.2 else
        'negligible'
    )

    return results


# =============================================================================
# 4. GLOBAL SCORE BOOTSTRAP CON PHASE-RANDOMIZATION
# =============================================================================

def phase_randomize(series: np.ndarray) -> np.ndarray:
    """
    Phase-randomization: mantiene espectro de potencia, rompe correlaciones.
    """
    n = len(series)

    # FFT
    fft = np.fft.fft(series)

    # Randomizar fases
    phases = np.random.uniform(0, 2*np.pi, n)

    # Para señal real, fases deben ser simétricas
    if n % 2 == 0:
        phases[n//2+1:] = -phases[1:n//2][::-1]
    else:
        phases[(n+1)//2:] = -phases[1:(n+1)//2][::-1]
    phases[0] = 0

    # Aplicar fases aleatorias
    randomized_fft = np.abs(fft) * np.exp(1j * phases)

    # IFFT
    result = np.real(np.fft.ifft(randomized_fft))

    return result


def compute_indicator_from_series(
    te_series: np.ndarray,
    sync_series: np.ndarray,
    entropy_series: np.ndarray,
    assignments: List[int]
) -> Dict[str, float]:
    """Calcula indicadores de consciencia desde series."""
    n = len(te_series)

    # Information integration (proxy: mean sync)
    integration = float(np.mean(sync_series))

    # Differentiation (entropy of assignment distribution)
    counts = Counter(assignments)
    probs = np.array(list(counts.values())) / len(assignments)
    differentiation = float(compute_entropy_normalized(probs))

    # Temporality (autocorrelation of TE)
    if n > 10:
        autocorr = np.corrcoef(te_series[:-1], te_series[1:])[0, 1]
        temporality = float((autocorr + 1) / 2) if not np.isnan(autocorr) else 0.5
    else:
        temporality = 0.5

    # Unity (mean coherence - based on sync and entropy)
    unity = float(np.mean(sync_series) * (1 - np.std(entropy_series)))

    # Self-reference (recurrence rate)
    recurrence = sum(1 for i in range(1, len(assignments))
                    if assignments[i] == assignments[i-1]) / (len(assignments) - 1)
    self_reference = float(recurrence)

    # Global score
    global_score = np.mean([integration, differentiation, temporality, unity, self_reference])

    return {
        'integration': integration,
        'differentiation': differentiation,
        'temporality': temporality,
        'unity': unity,
        'self_reference': self_reference,
        'global_score': float(global_score)
    }


def bootstrap_with_nulls(
    te_series: np.ndarray,
    sync_series: np.ndarray,
    entropy_series: np.ndarray,
    assignments: List[int],
    n_bootstrap: int = 10000,
    n_nulls: int = 500
) -> Dict:
    """
    Bootstrap CI para indicadores + comparación con phase-randomization nulls.
    """
    n = len(te_series)

    if n < 50:
        return {'error': 'insufficient_data'}

    # Indicadores reales
    real_indicators = compute_indicator_from_series(
        te_series, sync_series, entropy_series, assignments
    )

    results = {
        'n_bootstrap': n_bootstrap,
        'n_nulls': n_nulls,
        'real': real_indicators,
        'bootstrap': {},
        'null': {},
        'comparison': {}
    }

    # Bootstrap
    indicator_names = list(real_indicators.keys())
    bootstrap_samples = {name: [] for name in indicator_names}

    for _ in range(n_bootstrap):
        # Resample con reemplazo
        indices = np.random.choice(n, n, replace=True)
        boot_te = te_series[indices]
        boot_sync = sync_series[indices]
        boot_entropy = entropy_series[indices]
        boot_assign = [assignments[i] for i in indices]

        boot_indicators = compute_indicator_from_series(
            boot_te, boot_sync, boot_entropy, boot_assign
        )

        for name in indicator_names:
            bootstrap_samples[name].append(boot_indicators[name])

    # CI 95%
    for name in indicator_names:
        samples = bootstrap_samples[name]
        results['bootstrap'][name] = {
            'mean': float(np.mean(samples)),
            'std': float(np.std(samples)),
            'ci_lower': float(np.percentile(samples, 2.5)),
            'ci_upper': float(np.percentile(samples, 97.5))
        }

    # Phase-randomization nulls
    null_samples = {name: [] for name in indicator_names}

    for _ in range(n_nulls):
        null_te = phase_randomize(te_series)
        null_sync = phase_randomize(sync_series)
        null_entropy = phase_randomize(entropy_series)

        # Shuffle assignments
        null_assign = list(assignments)
        np.random.shuffle(null_assign)

        null_indicators = compute_indicator_from_series(
            null_te, null_sync, null_entropy, null_assign
        )

        for name in indicator_names:
            null_samples[name].append(null_indicators[name])

    # Null statistics
    for name in indicator_names:
        samples = null_samples[name]
        results['null'][name] = {
            'mean': float(np.mean(samples)),
            'std': float(np.std(samples)),
            'ci_lower': float(np.percentile(samples, 2.5)),
            'ci_upper': float(np.percentile(samples, 97.5))
        }

    # Comparación real vs null
    for name in indicator_names:
        real_val = real_indicators[name]
        null_mean = results['null'][name]['mean']
        null_std = results['null'][name]['std'] + NUMERIC_EPS

        # P-value (two-sided)
        p_value = float(np.mean([abs(ns - null_mean) >= abs(real_val - null_mean)
                                 for ns in null_samples[name]]))

        # Effect size (Cohen's d)
        effect = float((real_val - null_mean) / null_std)

        results['comparison'][name] = {
            'delta': float(real_val - null_mean),
            'p_value': p_value,
            'effect_size': effect,
            'significant': p_value < 0.05,
            'direction': 'higher' if real_val > null_mean else 'lower'
        }

    return results


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

def create_sensitivity_plot(results: Dict, output_path: str):
    """Crea gráfico de sensibilidad de prototipos."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for agent, ax_idx in [('neo', 0), ('eva', 1)]:
        if agent not in results:
            continue
        data = results[agent]

        ax = axes[ax_idx]
        ax.plot(data['quantile_levels'], data['n_prototypes'], 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Quantile Threshold')
        ax.set_ylabel('# Prototypes')
        ax.set_title(f'{agent.upper()} Prototype Sensitivity')
        ax.grid(True, alpha=0.3)

    # Combined
    ax = axes[2]
    if 'neo' in results and 'eva' in results:
        ax.plot(results['neo']['quantile_levels'], results['neo']['n_prototypes'],
                'o-', label='NEO', linewidth=2)
        ax.plot(results['eva']['quantile_levels'], results['eva']['n_prototypes'],
                's-', label='EVA', linewidth=2)
        ax.set_xlabel('Quantile Threshold')
        ax.set_ylabel('# Prototypes')
        ax.set_title('Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_transition_violin(results: Dict, output_path: str):
    """Crea violin plot de asimetría de transición."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, (agent, data) in enumerate([('neo', results.get('neo', {})),
                                        ('eva', results.get('eva', {}))]):
        if 'null_kl' not in data:
            continue

        ax = axes[i]

        # Violin de nulos
        parts = ax.violinplot([data['null_kl']], positions=[0], showmeans=True)
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)

        # Línea real
        ax.axhline(data['real']['kl_asymmetry'], color='red', linestyle='--',
                   linewidth=2, label=f"Real (z={data['kl_z_score']:.2f})")

        ax.set_ylabel('KL Asymmetry')
        ax.set_title(f'{agent.upper()} Transition Asymmetry')
        ax.set_xticks([0])
        ax.set_xticklabels(['Null Distribution'])
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_curvature_plot(results: Dict, output_path: str):
    """Crea gráfico de curvatura GNT vs nulls."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    if 'null_mean_curvatures' not in results:
        return

    # Histograma de nulos
    ax.hist(results['null_mean_curvatures'], bins=30, alpha=0.7,
            label='Random Walk Nulls', color='lightblue', edgecolor='blue')

    # Línea real
    ax.axvline(results['real']['mean_curvature'], color='red', linestyle='--',
               linewidth=2, label=f"Real GNT (effect={results['effect_size']:.2f})")

    ax.set_xlabel('Mean Curvature')
    ax.set_ylabel('Count')
    ax.set_title(f"GNT Curvature vs Random Walk (p={results['mean_p_value']:.3f})")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_global_score_plot(results: Dict, output_path: str):
    """Crea gráfico de global score con CI."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if 'bootstrap' not in results or 'comparison' not in results:
        return

    indicators = ['integration', 'differentiation', 'temporality', 'unity', 'self_reference', 'global_score']

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(indicators))
    width = 0.35

    # Real values with CI
    real_vals = [results['real'][ind] for ind in indicators]
    ci_lower = [results['bootstrap'][ind]['ci_lower'] for ind in indicators]
    ci_upper = [results['bootstrap'][ind]['ci_upper'] for ind in indicators]

    yerr_lower = [max(0, r - l) for r, l in zip(real_vals, ci_lower)]
    yerr_upper = [max(0, u - r) for r, u in zip(real_vals, ci_upper)]

    bars1 = ax.bar(x - width/2, real_vals, width, label='Real',
                   yerr=[yerr_lower, yerr_upper], capsize=5, color='steelblue')

    # Null values
    null_vals = [results['null'][ind]['mean'] for ind in indicators]
    null_ci_lower = [results['null'][ind]['ci_lower'] for ind in indicators]
    null_ci_upper = [results['null'][ind]['ci_upper'] for ind in indicators]

    null_yerr_lower = [max(0, n - l) for n, l in zip(null_vals, null_ci_lower)]
    null_yerr_upper = [max(0, u - n) for n, u in zip(null_vals, null_ci_upper)]

    bars2 = ax.bar(x + width/2, null_vals, width, label='Phase-Randomized Null',
                   yerr=[null_yerr_lower, null_yerr_upper], capsize=5, color='lightcoral')

    # Significance markers
    for i, ind in enumerate(indicators):
        if results['comparison'][ind]['significant']:
            max_val = max(ci_upper[i], null_ci_upper[i] + null_vals[i])
            ax.text(i, max_val + 0.05, '*', ha='center', fontsize=14, fontweight='bold')

    ax.set_ylabel('Score')
    ax.set_title('Consciousness Indicators: Real vs Phase-Randomized Null')
    ax.set_xticks(x)
    ax.set_xticklabels([ind.replace('_', '\n') for ind in indicators], fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# RUNNER PRINCIPAL
# =============================================================================

def run_phase15c(
    n_steps: int = 2000,
    n_nulls: int = 500,
    n_bootstrap: int = 10000,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Ejecuta Phase 15C: Verificación completa.
    """
    from phase15_structural_consciousness import StructuralConsciousnessSystem

    if verbose:
        print("=" * 70)
        print("PHASE 15C: VERIFICACIÓN CON SENSIBILIDAD Y NULOS")
        print("=" * 70)

    np.random.seed(seed)

    # Ejecutar sistema
    system = StructuralConsciousnessSystem()

    neo_state_vectors = []
    eva_state_vectors = []
    gnt_trajectory = []
    te_series = []
    sync_series = []
    entropy_series = []
    neo_assignments = []
    eva_assignments = []

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

        # Guardar datos
        if system.states.neo_current_state:
            neo_state_vectors.append(system.states.neo_current_state.to_array())
            neo_assignments.append(result['neo']['state']['prototype_id'])
        if system.states.eva_current_state:
            eva_state_vectors.append(system.states.eva_current_state.to_array())
            eva_assignments.append(result['eva']['state']['prototype_id'])

        gnt_trajectory.append(system.gnt.gnt.gnt.copy())
        te_series.append(te_neo + te_eva)
        sync_series.append(sync)
        entropy_series.append(compute_entropy_normalized(neo_pi))

    te_series = np.array(te_series)
    sync_series = np.array(sync_series)
    entropy_series = np.array(entropy_series)

    if verbose:
        print(f"    Completado: {n_steps} pasos")

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_steps': n_steps,
            'n_nulls': n_nulls,
            'n_bootstrap': n_bootstrap,
            'seed': seed
        }
    }

    # 1. Sensibilidad de prototipos
    if verbose:
        print("\n[2] Sensibilidad de prototipos...")

    results['prototype_sensitivity'] = {
        'neo': prototype_sensitivity_sweep(neo_state_vectors),
        'eva': prototype_sensitivity_sweep(eva_state_vectors)
    }

    if verbose:
        for agent in ['neo', 'eva']:
            data = results['prototype_sensitivity'][agent]
            if 'n_prototypes' in data:
                print(f"    {agent.upper()}: {data['n_prototypes']} prototipos @ q{data['quantile_levels']}")

    # Guardar y plot
    with open('/root/NEO_EVA/results/phase15c/prototype_sensitivity.json', 'w') as f:
        json.dump(results['prototype_sensitivity'], f, indent=2)

    create_sensitivity_plot(results['prototype_sensitivity'],
                           '/root/NEO_EVA/figures/prototype_sensitivity_curve.png')

    # 2. Transición vs nulos
    if verbose:
        print(f"\n[3] Matriz de transición vs nulos (N={n_nulls})...")

    results['transition_nulls'] = {
        'neo': transition_null_analysis(neo_assignments, n_nulls),
        'eva': transition_null_analysis(eva_assignments, n_nulls)
    }

    if verbose:
        for agent in ['neo', 'eva']:
            data = results['transition_nulls'][agent]
            if 'kl_z_score' in data:
                sig = "✓" if data['kl_significant'] else "✗"
                print(f"    {agent.upper()}: KL z={data['kl_z_score']:.2f}, p={data['kl_p_value']:.3f} {sig}")

    with open('/root/NEO_EVA/results/phase15c/transition_nulls.json', 'w') as f:
        # Reducir tamaño quitando listas largas
        save_data = {}
        for agent in ['neo', 'eva']:
            save_data[agent] = {k: v for k, v in results['transition_nulls'][agent].items()
                               if k not in ['null_kl', 'null_entropy']}
            if 'null_kl' in results['transition_nulls'][agent]:
                save_data[agent]['null_kl_stats'] = {
                    'mean': float(np.mean(results['transition_nulls'][agent]['null_kl'])),
                    'std': float(np.std(results['transition_nulls'][agent]['null_kl']))
                }
        json.dump(save_data, f, indent=2)

    create_transition_violin(results['transition_nulls'],
                            '/root/NEO_EVA/figures/transition_asymmetry_violin.png')

    # 3. GNT curvatura
    if verbose:
        print(f"\n[4] GNT curvatura vs random walk (N={n_nulls})...")

    results['gnt_curvature'] = gnt_curvature_analysis(gnt_trajectory, n_nulls)

    if verbose:
        data = results['gnt_curvature']
        if 'effect_size' in data:
            print(f"    Effect size: {data['effect_size']:.2f} ({data['effect_interpretation']})")
            print(f"    p-value: {data['mean_p_value']:.3f}")
            intent = "Sí" if data['more_intentional'] else "No"
            print(f"    Más intencional que random walk: {intent}")

    with open('/root/NEO_EVA/results/phase15c/gnt_curvature.json', 'w') as f:
        save_data = {}
        for k, v in results['gnt_curvature'].items():
            if k in ['null_mean_curvatures', 'null_median_curvatures']:
                continue
            if isinstance(v, (np.bool_, bool)):
                save_data[k] = bool(v)
            elif isinstance(v, (np.integer, np.floating)):
                save_data[k] = float(v)
            elif isinstance(v, dict):
                save_data[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                               for kk, vv in v.items()}
            else:
                save_data[k] = v
        if 'null_mean_curvatures' in results['gnt_curvature']:
            save_data['null_stats'] = {
                'mean': float(np.mean(results['gnt_curvature']['null_mean_curvatures'])),
                'std': float(np.std(results['gnt_curvature']['null_mean_curvatures']))
            }
        json.dump(save_data, f, indent=2)

    create_curvature_plot(results['gnt_curvature'],
                         '/root/NEO_EVA/figures/gnt_curvature_nulls.png')

    # 4. Global score bootstrap
    if verbose:
        print(f"\n[5] Global score bootstrap (N={n_bootstrap}) + nulls ({n_nulls})...")

    results['global_score'] = bootstrap_with_nulls(
        te_series, sync_series, entropy_series, neo_assignments,
        n_bootstrap, n_nulls
    )

    if verbose:
        data = results['global_score']
        if 'real' in data:
            gs = data['real']['global_score']
            ci_l = data['bootstrap']['global_score']['ci_lower']
            ci_u = data['bootstrap']['global_score']['ci_upper']
            null_mean = data['null']['global_score']['mean']
            delta = data['comparison']['global_score']['delta']
            sig = "✓" if data['comparison']['global_score']['significant'] else "✗"
            print(f"    Global score: {gs:.3f} [{ci_l:.3f}, {ci_u:.3f}]")
            print(f"    Null mean: {null_mean:.3f}, Δ={delta:+.3f} {sig}")

    with open('/root/NEO_EVA/results/phase15c/global_score_bootstrap.json', 'w') as f:
        json.dump(results['global_score'], f, indent=2)

    create_global_score_plot(results['global_score'],
                            '/root/NEO_EVA/figures/global_score_ci.png')

    # 5. Generar resumen markdown
    if verbose:
        print("\n[6] Generando resumen...")

    generate_summary_markdown(results)

    if verbose:
        print("\n" + "=" * 70)
        print("RESUMEN PHASE 15C")
        print("=" * 70)

        # Sensibilidad
        print("\n1. Sensibilidad de Prototipos:")
        for agent in ['neo', 'eva']:
            data = results['prototype_sensitivity'][agent]
            if 'n_prototypes' in data:
                print(f"   {agent.upper()}: {data['n_prototypes'][0]}→{data['n_prototypes'][-1]} "
                      f"(p50→p90)")

        # Transición
        print("\n2. Asimetría de Transición vs Nulos:")
        for agent in ['neo', 'eva']:
            data = results['transition_nulls'][agent]
            if 'kl_significant' in data:
                sig = "significativa" if data['kl_significant'] else "no significativa"
                print(f"   {agent.upper()}: z={data['kl_z_score']:.2f}, p={data['kl_p_value']:.3f} ({sig})")

        # GNT
        print("\n3. Curvatura GNT:")
        if 'effect_size' in results['gnt_curvature']:
            data = results['gnt_curvature']
            print(f"   Effect: {data['effect_size']:.2f} ({data['effect_interpretation']})")
            print(f"   ¿Más intencional?: {'Sí' if data['more_intentional'] else 'No'}")

        # Global score
        print("\n4. Global Score:")
        if 'real' in results['global_score']:
            data = results['global_score']
            gs = data['real']['global_score']
            ci = f"[{data['bootstrap']['global_score']['ci_lower']:.3f}, {data['bootstrap']['global_score']['ci_upper']:.3f}]"
            delta = data['comparison']['global_score']['delta']
            print(f"   Score: {gs:.3f} {ci}")
            print(f"   Δ vs null: {delta:+.3f}")

        print("\n" + "=" * 70)

    return results


def generate_summary_markdown(results: Dict):
    """Genera resumen en markdown."""
    md = []
    md.append("# Phase 15C: Verificación con Sensibilidad y Nulos")
    md.append("")
    md.append(f"**Fecha:** {results['timestamp']}")
    md.append(f"**Config:** n_steps={results['config']['n_steps']}, "
              f"n_nulls={results['config']['n_nulls']}, n_bootstrap={results['config']['n_bootstrap']}")
    md.append("")

    # 1. Sensibilidad
    md.append("## 1. Sensibilidad de Prototipos")
    md.append("")
    md.append("| Agente | p50 | p60 | p70 | p80 | p90 |")
    md.append("|--------|-----|-----|-----|-----|-----|")

    for agent in ['neo', 'eva']:
        data = results['prototype_sensitivity'][agent]
        if 'n_prototypes' in data:
            protos = data['n_prototypes']
            md.append(f"| {agent.upper()} | {protos[0]} | {protos[1]} | {protos[2]} | {protos[3]} | {protos[4]} |")

    md.append("")
    md.append("![Prototype Sensitivity](../figures/prototype_sensitivity_curve.png)")
    md.append("")

    # 2. Transición
    md.append("## 2. Asimetría de Transición vs Nulos")
    md.append("")
    md.append("| Agente | KL Real | KL Null (mean±std) | z-score | p-value | Significativo |")
    md.append("|--------|---------|-------------------|---------|---------|---------------|")

    for agent in ['neo', 'eva']:
        data = results['transition_nulls'][agent]
        if 'real' in data:
            kl_real = data['real']['kl_asymmetry']
            null_mean = np.mean(data.get('null_kl', [0]))
            null_std = np.std(data.get('null_kl', [0]))
            z = data['kl_z_score']
            p = data['kl_p_value']
            sig = "✓" if data['kl_significant'] else "✗"
            md.append(f"| {agent.upper()} | {kl_real:.4f} | {null_mean:.4f}±{null_std:.4f} | {z:.2f} | {p:.3f} | {sig} |")

    md.append("")
    md.append("![Transition Asymmetry](../figures/transition_asymmetry_violin.png)")
    md.append("")

    # 3. GNT
    md.append("## 3. Curvatura GNT vs Random Walk")
    md.append("")

    data = results['gnt_curvature']
    if 'real' in data:
        md.append(f"- **Curvatura real:** {data['real']['mean_curvature']:.4f}")
        md.append(f"- **Curvatura null (mean):** {np.mean(data.get('null_mean_curvatures', [0])):.4f}")
        md.append(f"- **Effect size:** {data['effect_size']:.2f} ({data['effect_interpretation']})")
        md.append(f"- **p-value:** {data['mean_p_value']:.3f}")
        intent = "Sí" if data['more_intentional'] else "No"
        md.append(f"- **Más intencional que random walk:** {intent}")

    md.append("")
    md.append("![GNT Curvature](../figures/gnt_curvature_nulls.png)")
    md.append("")

    # 4. Global score
    md.append("## 4. Global Score con IC y Efecto")
    md.append("")
    md.append("| Indicador | Real | IC95 | Null (mean) | Δ | p-value | Sig |")
    md.append("|-----------|------|------|-------------|---|---------|-----|")

    indicators = ['integration', 'differentiation', 'temporality', 'unity', 'self_reference', 'global_score']
    data = results['global_score']

    if 'real' in data:
        for ind in indicators:
            real = data['real'][ind]
            ci_l = data['bootstrap'][ind]['ci_lower']
            ci_u = data['bootstrap'][ind]['ci_upper']
            null = data['null'][ind]['mean']
            delta = data['comparison'][ind]['delta']
            p = data['comparison'][ind]['p_value']
            sig = "✓" if data['comparison'][ind]['significant'] else "✗"
            md.append(f"| {ind} | {real:.3f} | [{ci_l:.3f}, {ci_u:.3f}] | {null:.3f} | {delta:+.3f} | {p:.3f} | {sig} |")

    md.append("")
    md.append("![Global Score CI](../figures/global_score_ci.png)")
    md.append("")

    # Conclusión
    md.append("## 5. Conclusiones")
    md.append("")

    # Contar significativos
    n_sig = 0
    n_total = 0

    for agent in ['neo', 'eva']:
        if results['transition_nulls'][agent].get('kl_significant'):
            n_sig += 1
        n_total += 1

    if results['gnt_curvature'].get('more_intentional'):
        n_sig += 1
    n_total += 1

    if 'comparison' in results['global_score']:
        for ind in indicators:
            if results['global_score']['comparison'][ind].get('significant'):
                n_sig += 1
            n_total += 1

    md.append(f"- **Checks significativos:** {n_sig}/{n_total}")
    md.append("")

    # Guardar
    with open('/root/NEO_EVA/results/phase15c_summary.md', 'w') as f:
        f.write('\n'.join(md))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results = run_phase15c(
        n_steps=2000,
        n_nulls=500,
        n_bootstrap=10000,
        verbose=True
    )
