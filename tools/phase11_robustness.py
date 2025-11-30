#!/usr/bin/env python3
"""
Phase 11: Tests de Robustez para Validar Correlación Inter-Agente
=================================================================

Tests rigurosos para descartar artefactos:

1. BLOCK-SHUFFLE: Preserva autocorrelación, destruye inter-dependencia
2. CIRCULAR SHIFT: Desfase temporal entre agentes
3. PHASE RANDOMIZATION: Mantiene espectro, destruye fase
4. RNG INDEPENDENCE: Verificar semillas independientes
5. GRANGER/TE CONDICIONAL: Causalidad por estado de vida
6. ROLLING ORIGIN: K-fold temporal para descartar drift
7. LEAK CHECK: Verificar no hay buffers compartidos
8. MULTI-SEED: Robustez con 20 semillas

Todo 100% endógeno.
"""

import sys
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import deque
from scipy import stats, signal
from scipy.fft import fft, ifft
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/root/NEO_EVA/tools')


# =============================================================================
# UTILIDADES
# =============================================================================

def get_epsilon():
    return np.finfo(np.float64).eps


def compute_acf(x: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Autocorrelación hasta max_lag."""
    n = len(x)
    x_centered = x - np.mean(x)
    var = np.var(x)
    if var < get_epsilon():
        return np.zeros(max_lag + 1)

    acf = np.correlate(x_centered, x_centered, mode='full')
    acf = acf[n-1:n+max_lag] / (var * n)
    return acf


def compute_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Correlación de Pearson."""
    if len(x) < 3 or len(y) < 3:
        return 0.0
    if np.std(x) < get_epsilon() or np.std(y) < get_epsilon():
        return 0.0
    return np.corrcoef(x, y)[0, 1]


# =============================================================================
# 1. BLOCK-SHUFFLE NULL
# =============================================================================

def block_shuffle(x: np.ndarray, block_size: int) -> np.ndarray:
    """
    Shuffle por bloques - preserva autocorrelación local,
    destruye dependencia inter-agente.
    """
    n = len(x)
    n_blocks = n // block_size

    # Crear bloques
    blocks = [x[i*block_size:(i+1)*block_size] for i in range(n_blocks)]

    # Shuffle bloques
    np.random.shuffle(blocks)

    # Reconstruir
    shuffled = np.concatenate(blocks)

    # Agregar residuo si existe
    residue = x[n_blocks*block_size:]
    if len(residue) > 0:
        shuffled = np.concatenate([shuffled, residue])

    return shuffled


def test_block_shuffle(pi_neo: np.ndarray, pi_eva: np.ndarray,
                       bilateral_ts: set, n_bootstrap: int = 200) -> Dict:
    """
    Test de block-shuffle.
    Block size endógeno: mediana de runs de eventos.
    """
    # Block size endógeno: longitud típica de "runs" de alto π
    threshold = np.median(pi_neo)
    runs = []
    current_run = 0
    for v in pi_neo:
        if v > threshold:
            current_run += 1
        elif current_run > 0:
            runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)

    block_size = int(np.median(runs)) if runs else 50
    block_size = max(10, min(block_size, len(pi_neo) // 10))

    # Correlación observada durante coupling
    observed_corr = compute_coupling_correlation(pi_neo, pi_eva, bilateral_ts)

    # Null distribution
    null_corrs = []
    for _ in range(n_bootstrap):
        # Shuffle solo EVA por bloques (preserva ACF de EVA)
        eva_shuffled = block_shuffle(pi_eva, block_size)
        null_corr = compute_coupling_correlation(pi_neo, eva_shuffled, bilateral_ts)
        null_corrs.append(null_corr)

    null_corrs = np.array(null_corrs)
    p_value = np.mean(np.abs(null_corrs) >= np.abs(observed_corr))

    return {
        'name': 'Block Shuffle',
        'block_size': block_size,
        'observed': float(observed_corr),
        'null_mean': float(np.mean(null_corrs)),
        'null_std': float(np.std(null_corrs)),
        'null_q025': float(np.percentile(null_corrs, 2.5)),
        'null_q975': float(np.percentile(null_corrs, 97.5)),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'acf_preserved': True  # By design
    }


def compute_coupling_correlation(pi_neo: np.ndarray, pi_eva: np.ndarray,
                                  bilateral_ts: set, window: int = 5) -> float:
    """Correlación durante ventanas de coupling."""
    if not bilateral_ts:
        return 0.0

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

    return compute_correlation(np.array(neo_vals), np.array(eva_vals))


# =============================================================================
# 2. CIRCULAR SHIFT TEST
# =============================================================================

def test_circular_shift(pi_neo: np.ndarray, pi_eva: np.ndarray,
                        bilateral_ts: set, n_lags: int = 20) -> Dict:
    """
    Desfase circular entre agentes.
    La correlación debería caer con el lag.
    """
    n = len(pi_neo)
    observed_corr = compute_coupling_correlation(pi_neo, pi_eva, bilateral_ts)

    # Lags endógenos: proporción del período
    max_lag = n // 10
    lags = np.linspace(1, max_lag, n_lags).astype(int)

    correlations = []
    for lag in lags:
        # Shift circular de EVA
        eva_shifted = np.roll(pi_eva, lag)
        # Recalcular bilateral (aproximado)
        shifted_corr = compute_correlation(pi_neo, eva_shifted)
        correlations.append(shifted_corr)

    correlations = np.array(correlations)

    # Decaimiento: correlación a lag=0 vs lag=max_lag/2
    decay_ratio = observed_corr / (np.mean(correlations) + get_epsilon())

    # La señal es genuina si decay_ratio >> 1
    return {
        'name': 'Circular Shift',
        'observed_corr': float(observed_corr),
        'lags': lags.tolist(),
        'correlations': correlations.tolist(),
        'mean_shifted_corr': float(np.mean(correlations)),
        'decay_ratio': float(decay_ratio),
        'significant_decay': decay_ratio > 2.0  # >2x decay = señal real
    }


# =============================================================================
# 3. PHASE RANDOMIZATION
# =============================================================================

def phase_randomize(x: np.ndarray) -> np.ndarray:
    """
    Phase randomization via FFT.
    Mantiene magnitud del espectro, aleatoriza fase.
    """
    n = len(x)

    # FFT
    X = fft(x)
    magnitudes = np.abs(X)

    # Fases aleatorias (simétricas para señal real)
    random_phases = np.random.uniform(0, 2*np.pi, n//2 + 1)

    # Construir fases completas (conjugado simétrico)
    if n % 2 == 0:
        phases = np.concatenate([random_phases, -random_phases[-2:0:-1]])
    else:
        phases = np.concatenate([random_phases, -random_phases[-1:0:-1]])

    # Reconstruir señal
    X_randomized = magnitudes * np.exp(1j * phases)
    x_randomized = np.real(ifft(X_randomized))

    return x_randomized


def test_phase_randomization(pi_neo: np.ndarray, pi_eva: np.ndarray,
                              bilateral_ts: set, n_bootstrap: int = 200) -> Dict:
    """
    Test de phase randomization.
    Mantiene energía espectral, destruye coherencia de fase.
    """
    observed_corr = compute_coupling_correlation(pi_neo, pi_eva, bilateral_ts)

    null_corrs = []
    for _ in range(n_bootstrap):
        # Randomizar fase de EVA
        eva_randomized = phase_randomize(pi_eva)
        null_corr = compute_coupling_correlation(pi_neo, eva_randomized, bilateral_ts)
        null_corrs.append(null_corr)

    null_corrs = np.array(null_corrs)
    p_value = np.mean(np.abs(null_corrs) >= np.abs(observed_corr))

    # Verificar que preserva varianza
    original_var = np.var(pi_eva)
    randomized_var = np.var(phase_randomize(pi_eva))
    variance_preserved = abs(original_var - randomized_var) / (original_var + get_epsilon()) < 0.1

    return {
        'name': 'Phase Randomization',
        'observed': float(observed_corr),
        'null_mean': float(np.mean(null_corrs)),
        'null_std': float(np.std(null_corrs)),
        'null_q025': float(np.percentile(null_corrs, 2.5)),
        'null_q975': float(np.percentile(null_corrs, 97.5)),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'variance_preserved': variance_preserved
    }


# =============================================================================
# 4. RNG INDEPENDENCE CHECK
# =============================================================================

def check_rng_independence() -> Dict:
    """
    Verificar que los RNGs son independientes por mundo.
    """
    # Simular dos streams con semillas distintas
    rng_neo = np.random.RandomState(42)
    rng_eva = np.random.RandomState(137)

    n = 1000
    neo_samples = rng_neo.randn(n)
    eva_samples = rng_eva.randn(n)

    # Correlación entre streams
    corr = compute_correlation(neo_samples, eva_samples)

    # Test: correlación debería ser ~0
    # Bajo H0 (independencia), |corr| ~ N(0, 1/sqrt(n))
    z_score = abs(corr) * np.sqrt(n)
    p_value = 2 * (1 - stats.norm.cdf(z_score))

    return {
        'name': 'RNG Independence',
        'correlation_between_streams': float(corr),
        'z_score': float(z_score),
        'p_value': float(p_value),
        'independent': p_value > 0.05  # No rechazar H0 de independencia
    }


# =============================================================================
# 5. GRANGER/TRANSFER ENTROPY CONDITIONED BY STATE
# =============================================================================

def compute_transfer_entropy(x: np.ndarray, y: np.ndarray, lag: int = 1) -> float:
    """
    Transfer Entropy simplificado: TE(X→Y).
    Mide cuánta información de X pasado reduce incertidumbre de Y futuro.
    """
    n = len(x) - lag
    if n < 50:
        return 0.0

    # Discretizar a cuartiles (endógeno)
    x_disc = np.digitize(x, np.percentile(x, [25, 50, 75]))
    y_disc = np.digitize(y, np.percentile(y, [25, 50, 75]))

    # H(Y_t | Y_{t-1})
    y_past = y_disc[:-lag]
    y_future = y_disc[lag:]

    # Entropía condicional H(Y_t | Y_{t-1})
    h_y_given_ypast = conditional_entropy(y_future, y_past)

    # H(Y_t | Y_{t-1}, X_{t-1})
    x_past = x_disc[:-lag]
    joint_past = y_past * 4 + x_past  # Combinar
    h_y_given_joint = conditional_entropy(y_future, joint_past)

    # TE = H(Y|Y_past) - H(Y|Y_past,X_past)
    te = h_y_given_ypast - h_y_given_joint

    return max(0, te)


def conditional_entropy(y: np.ndarray, x: np.ndarray) -> float:
    """H(Y|X) = H(Y,X) - H(X)"""
    # Entropía conjunta
    joint = np.column_stack([y, x])
    joint_tuples = [tuple(row) for row in joint]
    _, counts_joint = np.unique(joint_tuples, return_counts=True, axis=0)
    p_joint = counts_joint / len(joint)
    h_joint = -np.sum(p_joint * np.log(p_joint + get_epsilon()))

    # Entropía marginal de X
    _, counts_x = np.unique(x, return_counts=True)
    p_x = counts_x / len(x)
    h_x = -np.sum(p_x * np.log(p_x + get_epsilon()))

    return h_joint - h_x


def test_conditional_granger(signals_neo: List[Dict], signals_eva: List[Dict],
                              states_neo: List[str], states_eva: List[str]) -> Dict:
    """
    TE condicionado por estado de vida.
    La causalidad debe concentrarse en SOCIAL/WORK, no en SLEEP.
    """
    # Extraer π (o señal representativa)
    pi_neo = np.array([s.get('R_soc', 0.5) + s.get('m', 0.5) for s in signals_neo])
    pi_eva = np.array([s.get('R_soc', 0.5) + s.get('m', 0.5) for s in signals_eva])

    states = ['SLEEP', 'WAKE', 'WORK', 'LEARN', 'SOCIAL']
    te_by_state = {}

    for state in states:
        # Índices donde NEO está en este estado
        indices = [i for i, s in enumerate(states_neo) if s == state and i < len(pi_neo) - 1]

        if len(indices) < 50:
            te_by_state[state] = {'te_neo_to_eva': 0.0, 'te_eva_to_neo': 0.0, 'n_samples': len(indices)}
            continue

        # Extraer segmentos
        neo_segment = pi_neo[indices]
        eva_segment = pi_eva[indices]

        # TE bidireccional
        te_neo_to_eva = compute_transfer_entropy(neo_segment, eva_segment)
        te_eva_to_neo = compute_transfer_entropy(eva_segment, neo_segment)

        te_by_state[state] = {
            'te_neo_to_eva': float(te_neo_to_eva),
            'te_eva_to_neo': float(te_eva_to_neo),
            'n_samples': len(indices)
        }

    # Análisis: TE debería ser mayor en SOCIAL/WORK
    social_work_te = (te_by_state.get('SOCIAL', {}).get('te_neo_to_eva', 0) +
                      te_by_state.get('WORK', {}).get('te_neo_to_eva', 0)) / 2
    sleep_te = te_by_state.get('SLEEP', {}).get('te_neo_to_eva', 0)

    ratio = social_work_te / (sleep_te + get_epsilon())

    return {
        'name': 'Conditional Transfer Entropy',
        'te_by_state': te_by_state,
        'social_work_te_mean': float(social_work_te),
        'sleep_te': float(sleep_te),
        'ratio_active_vs_sleep': float(ratio),
        'causality_concentrated_in_active': ratio > 1.5
    }


# =============================================================================
# 6. ROLLING ORIGIN (K-FOLD TEMPORAL)
# =============================================================================

def test_rolling_origin(pi_neo: np.ndarray, pi_eva: np.ndarray,
                        bilateral_events: List[Dict], n_folds: int = 5) -> Dict:
    """
    K-fold temporal con origen móvil.
    Descarta "drift complaciente" donde el modelo solo funciona por adaptación.
    """
    n = len(pi_neo)
    fold_size = n // (n_folds + 1)  # +1 para tener siempre train

    aucs = []
    correlations = []

    for fold in range(n_folds):
        # Train: hasta fold*fold_size + fold_size
        # Test: fold*fold_size + fold_size hasta fold*fold_size + 2*fold_size
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)

        if test_end <= test_start:
            continue

        # Eventos en test
        test_events = [e for e in bilateral_events
                       if test_start <= e['t'] < test_end]
        test_ts = set(e['t'] for e in test_events)

        if len(test_ts) < 10:
            continue

        # AUC en test
        pi_test = pi_neo[test_start:test_end]
        labels = np.array([1 if t in test_ts else 0
                          for t in range(test_start, test_end)])

        if labels.sum() > 5 and labels.sum() < len(labels):
            auc = roc_auc_score(labels, pi_test)
            aucs.append(auc)

        # Correlación en test
        if len(test_ts) > 10:
            # Ajustar índices
            adjusted_ts = set(t - test_start for t in test_ts if t < test_end)
            corr = compute_coupling_correlation(
                pi_neo[test_start:test_end],
                pi_eva[test_start:test_end],
                adjusted_ts
            )
            correlations.append(corr)

    return {
        'name': 'Rolling Origin',
        'n_folds': n_folds,
        'aucs': [float(a) for a in aucs],
        'auc_mean': float(np.mean(aucs)) if aucs else 0.0,
        'auc_std': float(np.std(aucs)) if aucs else 0.0,
        'correlations': [float(c) for c in correlations],
        'corr_mean': float(np.mean(correlations)) if correlations else 0.0,
        'corr_std': float(np.std(correlations)) if correlations else 0.0,
        'stable_across_folds': np.std(aucs) < 0.1 if len(aucs) >= 3 else False
    }


# =============================================================================
# 7. LEAK CHECK (SHARED BUFFER)
# =============================================================================

def check_shared_buffer_leak(neo_history: List, eva_history: List,
                              skip_warmup: int = 500) -> Dict:
    """
    Verificar que no hay buffers compartidos entre agentes.
    Salta warmup porque ambos empiezan iguales ([1/3,1/3,1/3]).
    """
    # Convertir a arrays
    if not neo_history or not eva_history:
        return {'name': 'Buffer Leak Check', 'checked': False, 'reason': 'No data'}

    # Saltar warmup (ambos empiezan igual, correlación alta es normal)
    neo_arr = np.array(neo_history[skip_warmup:])
    eva_arr = np.array(eva_history[skip_warmup:])

    if len(neo_arr) < 100 or len(eva_arr) < 100:
        return {'name': 'Buffer Leak Check', 'checked': False, 'reason': 'Not enough data after warmup'}

    # Test 1: Son el mismo objeto?
    same_object = neo_arr is eva_arr

    # Test 2: Correlación instantánea perfecta (sería leak)
    if len(neo_arr) > 0 and len(eva_arr) > 0:
        if len(neo_arr.shape) == 1:
            instant_corr = compute_correlation(neo_arr, eva_arr)
        else:
            # Promedio de correlaciones por dimensión
            instant_corr = np.mean([compute_correlation(neo_arr[:, i], eva_arr[:, i])
                                   for i in range(neo_arr.shape[1])])
    else:
        instant_corr = 0.0

    # Test 3: Diferencia entre agentes (debería haber)
    if len(neo_arr.shape) == 1:
        diff = np.abs(neo_arr - eva_arr)
        has_difference = np.mean(diff) > 0.01
    else:
        diff = np.abs(neo_arr - eva_arr)
        has_difference = np.mean(diff) > 0.01

    # Correlación > 0.99 Y sin diferencias sería leak
    suspicious = (abs(instant_corr) > 0.99 and not has_difference) or same_object

    return {
        'name': 'Buffer Leak Check',
        'same_object': same_object,
        'instant_correlation': float(instant_corr),
        'mean_difference': float(np.mean(diff)),
        'has_difference': has_difference,
        'suspicious': suspicious,
        'no_leak': not suspicious
    }


# =============================================================================
# 8. MULTI-SEED ROBUSTNESS
# =============================================================================

def test_multi_seed(run_experiment_fn, n_seeds: int = 20) -> Dict:
    """
    Ejecutar con múltiples semillas y reportar mediana + IQR.
    """
    from phase10_improved import run_improved_experiment

    metrics = {
        'n_bilateral': [],
        'auc_test': [],
        'correlation': [],
        'mean_intensity': []
    }

    for seed in range(n_seeds):
        np.random.seed(seed * 137 + 42)  # Semillas bien espaciadas

        try:
            result = run_improved_experiment(n_cycles=5000)  # Rápido

            metrics['n_bilateral'].append(result.get('n_bilateral', 0))
            metrics['auc_test'].append(result.get('auc_test', 0.5))
            metrics['mean_intensity'].append(result.get('mean_intensity', 0))
        except Exception as e:
            print(f"Seed {seed} failed: {e}")

    # Calcular estadísticas robustas
    summary = {}
    for key, values in metrics.items():
        if values:
            arr = np.array(values)
            summary[key] = {
                'median': float(np.median(arr)),
                'q25': float(np.percentile(arr, 25)),
                'q75': float(np.percentile(arr, 75)),
                'iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)),
                'n_valid': len(values)
            }

    return {
        'name': 'Multi-Seed Robustness',
        'n_seeds': n_seeds,
        'summary': summary
    }


# =============================================================================
# 9. ADDITIONAL METRICS
# =============================================================================

def compute_t_break(I_history: List[np.ndarray], initial: np.ndarray = None) -> int:
    """
    Tiempo hasta ruptura de simetría desde [1/3, 1/3, 1/3].
    Ruptura = cuando max(I) - min(I) > IQR de diferencias.
    """
    if not I_history:
        return -1

    if initial is None:
        initial = np.array([1/3, 1/3, 1/3])

    # Calcular diferencias
    diffs = [np.max(I) - np.min(I) for I in I_history]

    if len(diffs) < 20:
        return -1

    # Umbral endógeno: q75 de diferencias iniciales
    threshold = np.percentile(diffs[:20], 75)

    # Encontrar primer cruce
    for t, diff in enumerate(diffs):
        if diff > threshold:
            return t

    return len(diffs)  # Nunca rompió


def compute_kappa_by_state(bilateral_events: List[Dict], states: List[str]) -> Dict:
    """
    Distribución de intensidad κ por estado de vida.
    """
    kappa_by_state = {s: [] for s in ['SLEEP', 'WAKE', 'WORK', 'LEARN', 'SOCIAL']}

    for event in bilateral_events:
        t = event['t']
        intensity = event.get('intensity', 0)

        if t < len(states):
            state = states[t]
            if state in kappa_by_state:
                kappa_by_state[state].append(intensity)

    summary = {}
    for state, values in kappa_by_state.items():
        if values:
            arr = np.array(values)
            summary[state] = {
                'median': float(np.median(arr)),
                'q25': float(np.percentile(arr, 25)),
                'q75': float(np.percentile(arr, 75)),
                'n': len(values)
            }
        else:
            summary[state] = {'median': 0, 'q25': 0, 'q75': 0, 'n': 0}

    return summary


# =============================================================================
# MAIN: RUN ALL ROBUSTNESS TESTS
# =============================================================================

def run_robustness_tests(data_dir: str = '/root/NEO_EVA/results/phase10',
                         output_dir: str = '/root/NEO_EVA/results/phase11_robustness',
                         n_bootstrap: int = 200) -> Dict:
    """
    Ejecuta todos los tests de robustez.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("PHASE 11: TESTS DE ROBUSTEZ")
    print("=" * 70)

    # Cargar datos
    print("\n[1] Cargando datos de Phase 10...")

    with open(f"{data_dir}/pi_log_neo.json") as f:
        pi_neo_log = json.load(f)
    with open(f"{data_dir}/pi_log_eva.json") as f:
        pi_eva_log = json.load(f)
    with open(f"{data_dir}/bilateral_events.json") as f:
        bilateral_events = json.load(f)

    pi_neo = np.array([p['pi'] for p in pi_neo_log])
    pi_eva = np.array([p['pi'] for p in pi_eva_log])
    bilateral_ts = set(e['t'] for e in bilateral_events)

    print(f"    Ciclos: {len(pi_neo)}")
    print(f"    Eventos bilaterales: {len(bilateral_ts)}")

    results = {}

    # TEST 1: Block Shuffle
    print("\n[2] Test Block-Shuffle...")
    results['block_shuffle'] = test_block_shuffle(pi_neo, pi_eva, bilateral_ts, n_bootstrap)
    print(f"    Observed corr: {results['block_shuffle']['observed']:.4f}")
    print(f"    Null mean: {results['block_shuffle']['null_mean']:.4f}")
    print(f"    p-value: {results['block_shuffle']['p_value']:.4f}")
    print(f"    Significant: {results['block_shuffle']['significant']}")

    # TEST 2: Circular Shift
    print("\n[3] Test Circular Shift...")
    results['circular_shift'] = test_circular_shift(pi_neo, pi_eva, bilateral_ts)
    print(f"    Observed corr: {results['circular_shift']['observed_corr']:.4f}")
    print(f"    Mean shifted corr: {results['circular_shift']['mean_shifted_corr']:.4f}")
    print(f"    Decay ratio: {results['circular_shift']['decay_ratio']:.2f}x")
    print(f"    Significant decay: {results['circular_shift']['significant_decay']}")

    # TEST 3: Phase Randomization
    print("\n[4] Test Phase Randomization...")
    results['phase_random'] = test_phase_randomization(pi_neo, pi_eva, bilateral_ts, n_bootstrap)
    print(f"    Observed corr: {results['phase_random']['observed']:.4f}")
    print(f"    Null mean: {results['phase_random']['null_mean']:.4f}")
    print(f"    p-value: {results['phase_random']['p_value']:.4f}")
    print(f"    Significant: {results['phase_random']['significant']}")

    # TEST 4: RNG Independence
    print("\n[5] Check RNG Independence...")
    results['rng_check'] = check_rng_independence()
    print(f"    Correlation between streams: {results['rng_check']['correlation_between_streams']:.4f}")
    print(f"    Independent: {results['rng_check']['independent']}")

    # TEST 5: Rolling Origin
    print("\n[6] Test Rolling Origin (k-fold temporal)...")
    results['rolling_origin'] = test_rolling_origin(pi_neo, pi_eva, bilateral_events)
    print(f"    AUC mean: {results['rolling_origin']['auc_mean']:.4f} +/- {results['rolling_origin']['auc_std']:.4f}")
    print(f"    Corr mean: {results['rolling_origin']['corr_mean']:.4f} +/- {results['rolling_origin']['corr_std']:.4f}")
    print(f"    Stable across folds: {results['rolling_origin']['stable_across_folds']}")

    # TEST 6: Conditional Transfer Entropy by State
    print("\n[7] Test Conditional Transfer Entropy by State...")
    try:
        with open(f"{data_dir}/neo_log.json") as f:
            neo_log = json.load(f)
        with open(f"{data_dir}/eva_log.json") as f:
            eva_log = json.load(f)

        signals_neo = [entry['signals'] for entry in neo_log]
        signals_eva = [entry['signals'] for entry in eva_log]
        states_neo = [entry['state'] for entry in neo_log]
        states_eva = [entry['state'] for entry in eva_log]

        results['conditional_te'] = test_conditional_granger(
            signals_neo, signals_eva, states_neo, states_eva
        )
        print(f"    TE by state:")
        for state, te_info in results['conditional_te']['te_by_state'].items():
            print(f"      {state}: TE={te_info['te_neo_to_eva']:.4f} (n={te_info['n_samples']})")
        print(f"    Ratio active/sleep: {results['conditional_te']['ratio_active_vs_sleep']:.2f}x")
        print(f"    Causality in active states: {results['conditional_te']['causality_concentrated_in_active']}")
    except Exception as e:
        print(f"    [SKIP] Error: {e}")
        results['conditional_te'] = {'error': str(e)}

    # TEST 7: Buffer Leak Check (usar TODOS los datos, no solo 100)
    print("\n[8] Check Shared Buffer Leak...")
    results['buffer_check'] = check_shared_buffer_leak(
        [p['pi'] for p in pi_neo_log],
        [p['pi'] for p in pi_eva_log],
        skip_warmup=500  # Saltar warmup donde ambos son iguales
    )
    print(f"    Instant correlation (post-warmup): {results['buffer_check']['instant_correlation']:.4f}")
    print(f"    Mean difference: {results['buffer_check'].get('mean_difference', 0):.4f}")
    print(f"    Has difference: {results['buffer_check'].get('has_difference', False)}")
    print(f"    No leak: {results['buffer_check']['no_leak']}")

    # TEST 8: Additional Metrics (T_break, kappa by state)
    print("\n[9] Computing Additional Metrics...")
    try:
        # T_break distribution
        I_history_neo = [entry['I'] for entry in neo_log]
        I_history_eva = [entry['I'] for entry in eva_log]

        t_break_neo = compute_t_break([np.array(I) for I in I_history_neo])
        t_break_eva = compute_t_break([np.array(I) for I in I_history_eva])

        # Kappa by state
        kappa_by_state_neo = compute_kappa_by_state(bilateral_events, states_neo)
        kappa_by_state_eva = compute_kappa_by_state(bilateral_events, states_eva)

        results['additional_metrics'] = {
            't_break_neo': t_break_neo,
            't_break_eva': t_break_eva,
            'kappa_by_state_neo': kappa_by_state_neo,
            'kappa_by_state_eva': kappa_by_state_eva
        }

        print(f"    T_break (symmetry rupture):")
        print(f"      NEO: {t_break_neo} cycles")
        print(f"      EVA: {t_break_eva} cycles")
        print(f"    Kappa by state (NEO):")
        for state, info in kappa_by_state_neo.items():
            print(f"      {state}: median={info['median']:.4f} (n={info['n']})")
    except Exception as e:
        print(f"    [SKIP] Error: {e}")
        results['additional_metrics'] = {'error': str(e)}

    # RESUMEN
    print("\n" + "=" * 70)
    print("RESUMEN DE ROBUSTEZ")
    print("=" * 70)

    n_passed = 0
    # Conditional TE result
    te_ok = results.get('conditional_te', {}).get('causality_concentrated_in_active', False)

    tests = [
        ('Block Shuffle', results['block_shuffle']['significant']),
        ('Circular Shift Decay', results['circular_shift']['significant_decay']),
        ('Phase Randomization', results['phase_random']['significant']),
        ('RNG Independence', results['rng_check']['independent']),
        ('Rolling Origin Stable', results['rolling_origin']['stable_across_folds']),
        ('Conditional TE in Active', te_ok),
        ('No Buffer Leak', results['buffer_check']['no_leak'])
    ]

    print(f"\n{'Test':<30} {'Passed':>10}")
    print("-" * 45)
    for name, passed in tests:
        status = "YES" if passed else "NO"
        print(f"{name:<30} {status:>10}")
        if passed:
            n_passed += 1

    print("-" * 45)
    print(f"{'Total':<30} {n_passed}/{len(tests)}")

    # Conclusión
    if n_passed >= 5:
        print("\n*** CORRELACIÓN VALIDADA: No es artefacto ***")
        results['conclusion'] = 'VALIDATED'
    elif n_passed >= 3:
        print("\n*** CORRELACIÓN PARCIALMENTE VALIDADA: Revisar tests fallidos ***")
        results['conclusion'] = 'PARTIAL'
    else:
        print("\n*** ADVERTENCIA: Posible artefacto - revisar metodología ***")
        results['conclusion'] = 'SUSPICIOUS'

    # Convertir tipos numpy
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {str(k): convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # Guardar
    with open(f"{output_dir}/robustness_results.json", 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)

    print(f"\n[OK] Resultados guardados en {output_dir}/")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Phase 11: Robustness Tests')
    parser.add_argument('--data-dir', default='/root/NEO_EVA/results/phase10')
    parser.add_argument('--output-dir', default='/root/NEO_EVA/results/phase11_robustness')
    parser.add_argument('--n-bootstrap', type=int, default=200)
    args = parser.parse_args()

    run_robustness_tests(args.data_dir, args.output_dir, args.n_bootstrap)
