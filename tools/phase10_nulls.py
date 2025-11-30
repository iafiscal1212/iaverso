#!/usr/bin/env python3
"""
Phase 10: Nulos Agresivos
=========================
Nulos más estrictos para falsación real:
1. Full shuffle (rompe toda autocorrelación)
2. Inter-agent independence (permuta un agente, no el otro)
3. Time-reversed (invierte temporalmente)
"""

import sys
import os
import json
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/root/NEO_EVA/tools')


@dataclass
class AggressiveNull:
    """Resultado de un test de nulo agresivo."""
    name: str
    null_type: str
    observed: float
    null_mean: float
    null_std: float
    null_q025: float
    null_q975: float
    p_value: float
    n_bootstrap: int
    significant: bool


def full_shuffle(x: np.ndarray) -> np.ndarray:
    """Shuffle completo - rompe toda estructura temporal."""
    shuffled = x.copy()
    np.random.shuffle(shuffled)
    return shuffled


def time_reverse(x: np.ndarray) -> np.ndarray:
    """Invierte temporalmente la serie."""
    return x[::-1].copy()


def inter_agent_shuffle(x_neo: np.ndarray, x_eva: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Permuta un agente pero no el otro - rompe dependencia inter-agente."""
    # Shuffle solo EVA, mantener NEO intacto
    eva_shuffled = full_shuffle(x_eva)
    return x_neo.copy(), eva_shuffled


def compute_bilateral_auc(pi_series: np.ndarray, bilateral_ts: set) -> float:
    """Calcula AUC de π prediciendo eventos bilaterales."""
    labels = np.array([1 if t in bilateral_ts else 0 for t in range(len(pi_series))])

    if labels.sum() < 5 or labels.sum() == len(labels):
        return 0.5

    return roc_auc_score(labels, pi_series)


def compute_correlation_during_coupling(pi_neo: np.ndarray, pi_eva: np.ndarray,
                                         bilateral_ts: set, window: int = 5) -> float:
    """Correlación de π entre agentes durante ventanas de coupling."""
    if not bilateral_ts:
        return 0.0

    # Extraer valores en ventanas alrededor de eventos
    neo_vals = []
    eva_vals = []

    for t in bilateral_ts:
        for dt in range(-window, window + 1):
            idx = t + dt
            if 0 <= idx < len(pi_neo):
                neo_vals.append(pi_neo[idx])
                eva_vals.append(pi_eva[idx])

    if len(neo_vals) < 10:
        return 0.0

    return np.corrcoef(neo_vals, eva_vals)[0, 1]


def compute_entropy_stability(health_series: List[float]) -> float:
    """Estabilidad de la entropía (inverso de varianza)."""
    if len(health_series) < 10:
        return 0.0

    var = np.var(health_series)
    return 1 / (1 + var)


def run_aggressive_nulls(data_dir: str = '/root/NEO_EVA/results/phase10',
                         n_bootstrap: int = 200) -> Dict[str, AggressiveNull]:
    """
    Ejecuta tests de nulos agresivos.
    """
    print("=" * 70)
    print("PHASE 10: NULOS AGRESIVOS")
    print("=" * 70)

    # Cargar datos
    print("\n[1] Cargando datos...")

    with open(f"{data_dir}/pi_log_neo.json") as f:
        pi_neo_log = json.load(f)
    with open(f"{data_dir}/pi_log_eva.json") as f:
        pi_eva_log = json.load(f)
    with open(f"{data_dir}/bilateral_events.json") as f:
        bilateral_events = json.load(f)
    with open(f"{data_dir}/health_log.json") as f:
        health_log = json.load(f)

    pi_neo = np.array([p['pi'] for p in pi_neo_log])
    pi_eva = np.array([p['pi'] for p in pi_eva_log])
    bilateral_ts = set(e['t'] for e in bilateral_events)

    print(f"    Ciclos: {len(pi_neo)}")
    print(f"    Eventos bilaterales: {len(bilateral_ts)}")
    print(f"    Bootstrap iterations: {n_bootstrap}")

    results = {}
    np.random.seed(42)

    # =================================================================
    # TEST 1: AUC con full shuffle
    # =================================================================
    print("\n[2] Test AUC vs Full Shuffle...")

    observed_auc = compute_bilateral_auc(pi_neo, bilateral_ts)
    null_aucs = []

    for i in range(n_bootstrap):
        # Shuffle completo de π
        pi_shuffled = full_shuffle(pi_neo)
        null_auc = compute_bilateral_auc(pi_shuffled, bilateral_ts)
        null_aucs.append(null_auc)

    null_aucs = np.array(null_aucs)
    p_value = np.mean(null_aucs >= observed_auc)

    results['auc_full_shuffle'] = AggressiveNull(
        name='AUC',
        null_type='full_shuffle',
        observed=float(observed_auc),
        null_mean=float(np.mean(null_aucs)),
        null_std=float(np.std(null_aucs)),
        null_q025=float(np.percentile(null_aucs, 2.5)),
        null_q975=float(np.percentile(null_aucs, 97.5)),
        p_value=float(p_value),
        n_bootstrap=n_bootstrap,
        significant=p_value < 0.05
    )

    print(f"    Observed AUC: {observed_auc:.4f}")
    print(f"    Null mean: {np.mean(null_aucs):.4f} +/- {np.std(null_aucs):.4f}")
    print(f"    p-value: {p_value:.4f} {'*' if p_value < 0.05 else ''}")

    # =================================================================
    # TEST 2: Correlación inter-agente con inter-agent shuffle
    # =================================================================
    print("\n[3] Test Correlación vs Inter-Agent Shuffle...")

    observed_corr = compute_correlation_during_coupling(pi_neo, pi_eva, bilateral_ts)
    null_corrs = []

    for i in range(n_bootstrap):
        # Shuffle solo un agente
        neo_intact, eva_shuffled = inter_agent_shuffle(pi_neo, pi_eva)
        null_corr = compute_correlation_during_coupling(neo_intact, eva_shuffled, bilateral_ts)
        null_corrs.append(null_corr)

    null_corrs = np.array(null_corrs)
    p_value = np.mean(np.abs(null_corrs) >= np.abs(observed_corr))

    results['corr_inter_agent'] = AggressiveNull(
        name='Inter-agent correlation',
        null_type='inter_agent_shuffle',
        observed=float(observed_corr),
        null_mean=float(np.mean(null_corrs)),
        null_std=float(np.std(null_corrs)),
        null_q025=float(np.percentile(null_corrs, 2.5)),
        null_q975=float(np.percentile(null_corrs, 97.5)),
        p_value=float(p_value),
        n_bootstrap=n_bootstrap,
        significant=p_value < 0.05
    )

    print(f"    Observed corr: {observed_corr:.4f}")
    print(f"    Null mean: {np.mean(null_corrs):.4f} +/- {np.std(null_corrs):.4f}")
    print(f"    p-value: {p_value:.4f} {'*' if p_value < 0.05 else ''}")

    # =================================================================
    # TEST 3: AUC con time-reversal
    # =================================================================
    print("\n[4] Test AUC vs Time Reversal...")

    # Invertir tiempo
    pi_reversed = time_reverse(pi_neo)
    bilateral_reversed = set(len(pi_neo) - 1 - t for t in bilateral_ts)

    null_aucs_tr = []
    for i in range(n_bootstrap):
        # Permutar los tiempos de eventos en la serie invertida
        n_events = len(bilateral_ts)
        random_ts = set(np.random.choice(len(pi_neo), n_events, replace=False))
        null_auc = compute_bilateral_auc(pi_reversed, random_ts)
        null_aucs_tr.append(null_auc)

    null_aucs_tr = np.array(null_aucs_tr)

    # El observado es el AUC en la serie invertida con eventos invertidos
    observed_auc_tr = compute_bilateral_auc(pi_reversed, bilateral_reversed)
    p_value = np.mean(null_aucs_tr >= observed_auc_tr)

    results['auc_time_reversal'] = AggressiveNull(
        name='AUC time-reversed',
        null_type='time_reversal',
        observed=float(observed_auc_tr),
        null_mean=float(np.mean(null_aucs_tr)),
        null_std=float(np.std(null_aucs_tr)),
        null_q025=float(np.percentile(null_aucs_tr, 2.5)),
        null_q975=float(np.percentile(null_aucs_tr, 97.5)),
        p_value=float(p_value),
        n_bootstrap=n_bootstrap,
        significant=p_value < 0.05
    )

    print(f"    Observed AUC (reversed): {observed_auc_tr:.4f}")
    print(f"    Null mean: {np.mean(null_aucs_tr):.4f} +/- {np.std(null_aucs_tr):.4f}")
    print(f"    p-value: {p_value:.4f} {'*' if p_value < 0.05 else ''}")

    # =================================================================
    # TEST 4: Intensidad de coupling vs shuffle de tiempos
    # =================================================================
    print("\n[5] Test Intensidad vs Event Time Shuffle...")

    intensities = [e['intensity'] for e in bilateral_events]
    observed_mean_intensity = np.mean(intensities) if intensities else 0

    null_intensities = []
    for i in range(n_bootstrap):
        # Shuffle tiempos de eventos
        n_events = len(bilateral_events)
        random_times = np.random.choice(len(pi_neo), n_events, replace=False)

        # Intensidad sería min(π) en esos tiempos
        fake_intensities = [min(pi_neo[t], pi_eva[t]) for t in random_times if t < len(pi_eva)]
        null_intensities.append(np.mean(fake_intensities) if fake_intensities else 0)

    null_intensities = np.array(null_intensities)
    p_value = np.mean(null_intensities >= observed_mean_intensity)

    results['intensity_shuffle'] = AggressiveNull(
        name='Mean coupling intensity',
        null_type='event_time_shuffle',
        observed=float(observed_mean_intensity),
        null_mean=float(np.mean(null_intensities)),
        null_std=float(np.std(null_intensities)),
        null_q025=float(np.percentile(null_intensities, 2.5)),
        null_q975=float(np.percentile(null_intensities, 97.5)),
        p_value=float(p_value),
        n_bootstrap=n_bootstrap,
        significant=p_value < 0.05
    )

    print(f"    Observed intensity: {observed_mean_intensity:.4f}")
    print(f"    Null mean: {np.mean(null_intensities):.4f} +/- {np.std(null_intensities):.4f}")
    print(f"    p-value: {p_value:.4f} {'*' if p_value < 0.05 else ''}")

    # =================================================================
    # TEST 5: Entropía vs random walk
    # =================================================================
    print("\n[6] Test Entropía vs Random Walk...")

    health_neo = [h['neo'] for h in health_log]
    observed_stability = compute_entropy_stability(health_neo)

    null_stabilities = []
    for i in range(n_bootstrap):
        # Random walk con misma varianza
        steps = np.random.randn(len(health_neo)) * np.std(np.diff(health_neo) if len(health_neo) > 1 else [0.1])
        random_walk = np.cumsum(steps)
        random_walk = (random_walk - random_walk.min()) / (random_walk.max() - random_walk.min() + 1e-10)
        null_stabilities.append(compute_entropy_stability(random_walk.tolist()))

    null_stabilities = np.array(null_stabilities)
    p_value = np.mean(null_stabilities >= observed_stability)

    results['entropy_stability'] = AggressiveNull(
        name='Entropy stability',
        null_type='random_walk',
        observed=float(observed_stability),
        null_mean=float(np.mean(null_stabilities)),
        null_std=float(np.std(null_stabilities)),
        null_q025=float(np.percentile(null_stabilities, 2.5)),
        null_q975=float(np.percentile(null_stabilities, 97.5)),
        p_value=float(p_value),
        n_bootstrap=n_bootstrap,
        significant=p_value < 0.05
    )

    print(f"    Observed stability: {observed_stability:.4f}")
    print(f"    Null mean: {np.mean(null_stabilities):.4f} +/- {np.std(null_stabilities):.4f}")
    print(f"    p-value: {p_value:.4f} {'*' if p_value < 0.05 else ''}")

    # =================================================================
    # RESUMEN
    # =================================================================
    print("\n" + "=" * 70)
    print("RESUMEN DE NULOS AGRESIVOS")
    print("=" * 70)

    print(f"\n{'Test':<30} {'Obs':>10} {'Null':>10} {'p-value':>10} {'Sig':>5}")
    print("-" * 70)

    n_significant = 0
    for name, result in results.items():
        sig_mark = "*" if result.significant else ""
        print(f"{result.name:<30} {result.observed:>10.4f} {result.null_mean:>10.4f} "
              f"{result.p_value:>10.4f} {sig_mark:>5}")
        if result.significant:
            n_significant += 1

    print("-" * 70)
    print(f"Significativos: {n_significant}/{len(results)}")

    # Guardar (convertir numpy bool)
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    output_file = f"{data_dir}/nulls_aggressive.json"
    with open(output_file, 'w') as f:
        json.dump({k: convert_types(asdict(v)) for k, v in results.items()}, f, indent=2)

    print(f"\n[OK] Guardado en {output_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/root/NEO_EVA/results/phase10')
    parser.add_argument('--n-bootstrap', type=int, default=200)
    args = parser.parse_args()

    run_aggressive_nulls(args.data_dir, args.n_bootstrap)
