#!/usr/bin/env python3
"""
Análisis de Direccionalidad con Transfer Entropy (TE)

Calcula TE(NEO→EVA) y TE(EVA→NEO) para múltiples lags,
con tests de significación por phase randomization.
"""

import json
import numpy as np
from scipy.stats import pearsonr
from typing import Dict, List, Tuple

def entropy(x: np.ndarray, bins: int = 20) -> float:
    """Entropía de Shannon."""
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist + 1e-12) * (1.0 / bins))

def conditional_entropy(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """H(X|Y) - entropía condicional."""
    hist_xy, _, _ = np.histogram2d(x, y, bins=bins, density=True)
    hist_y, _ = np.histogram(y, bins=bins, density=True)

    # H(X,Y) - H(Y)
    hist_xy = hist_xy[hist_xy > 0]
    hist_y = hist_y[hist_y > 0]

    h_xy = -np.sum(hist_xy * np.log(hist_xy + 1e-12) * (1.0 / bins**2))
    h_y = -np.sum(hist_y * np.log(hist_y + 1e-12) * (1.0 / bins))

    return h_xy - h_y

def transfer_entropy(source: np.ndarray, target: np.ndarray, lag: int = 1, bins: int = 20) -> float:
    """
    Transfer Entropy: TE(source → target)
    TE = H(target_future | target_past) - H(target_future | target_past, source_past)
    """
    n = len(target)
    if n <= lag + 1:
        return 0.0

    # Variables
    target_future = target[lag:]
    target_past = target[:-lag]
    source_past = source[:-lag]

    # Ajustar longitudes
    min_len = min(len(target_future), len(target_past), len(source_past))
    target_future = target_future[:min_len]
    target_past = target_past[:min_len]
    source_past = source_past[:min_len]

    # H(target_future | target_past)
    h_cond_1 = conditional_entropy(target_future, target_past, bins)

    # H(target_future | target_past, source_past) - aproximación
    joint_past = target_past + source_past  # Simple sum como proxy
    h_cond_2 = conditional_entropy(target_future, joint_past, bins)

    return max(0, h_cond_1 - h_cond_2)

def phase_randomize(x: np.ndarray) -> np.ndarray:
    """Phase randomization para generar nulo."""
    fft = np.fft.fft(x)
    phases = np.angle(fft)
    random_phases = np.random.uniform(-np.pi, np.pi, len(phases))
    # Mantener simetría hermitiana
    random_phases[0] = 0
    if len(x) % 2 == 0:
        random_phases[len(x)//2] = 0
    random_phases[len(x)//2+1:] = -random_phases[1:len(x)//2][::-1]

    new_fft = np.abs(fft) * np.exp(1j * random_phases)
    return np.real(np.fft.ifft(new_fft))

def first_zero_acf(x: np.ndarray, max_lag: int = 50) -> int:
    """Primer lag donde ACF cruza cero."""
    n = len(x)
    x = x - np.mean(x)
    acf = np.correlate(x, x, mode='full')[n-1:] / (np.var(x) * n)

    for lag in range(1, min(max_lag, len(acf))):
        if acf[lag] <= 0:
            return lag
    return max_lag

def analyze_directionality(neo_series: List[Dict], eva_series: List[Dict],
                           n_surrogates: int = 100) -> Dict:
    """Análisis completo de direccionalidad."""

    # Extraer componente S (principal)
    neo_S = np.array([s['S_new'] for s in neo_series])
    eva_S = np.array([s['S_new'] for s in eva_series])

    # Determinar L (primer cero de ACF)
    L_neo = first_zero_acf(neo_S)
    L_eva = first_zero_acf(eva_S)
    L = max(L_neo, L_eva, 5)

    print(f"L (primer cero ACF): NEO={L_neo}, EVA={L_eva}, usando L={L}")

    results = {
        'L_neo': L_neo,
        'L_eva': L_eva,
        'L_used': L,
        'lags': list(range(1, L + 1)),
        'te_neo_to_eva': [],
        'te_eva_to_neo': [],
        'p_neo_to_eva': [],
        'p_eva_to_neo': [],
        'te_null_neo_to_eva': [],
        'te_null_eva_to_neo': [],
    }

    print("\nCalculando TE por lag...")
    print(f"{'Lag':<5} {'TE(N→E)':<12} {'TE(E→N)':<12} {'p(N→E)':<10} {'p(E→N)':<10}")
    print("-" * 50)

    for lag in range(1, L + 1):
        # TE observada
        te_n2e = transfer_entropy(neo_S, eva_S, lag=lag)
        te_e2n = transfer_entropy(eva_S, neo_S, lag=lag)

        # Nulos por phase randomization
        te_null_n2e = []
        te_null_e2n = []

        for _ in range(n_surrogates):
            neo_rand = phase_randomize(neo_S)
            eva_rand = phase_randomize(eva_S)
            te_null_n2e.append(transfer_entropy(neo_rand, eva_S, lag=lag))
            te_null_e2n.append(transfer_entropy(eva_rand, neo_S, lag=lag))

        # p-values
        p_n2e = np.mean([t >= te_n2e for t in te_null_n2e])
        p_e2n = np.mean([t >= te_e2n for t in te_null_e2n])

        results['te_neo_to_eva'].append(te_n2e)
        results['te_eva_to_neo'].append(te_e2n)
        results['p_neo_to_eva'].append(p_n2e)
        results['p_eva_to_neo'].append(p_e2n)
        results['te_null_neo_to_eva'].append(np.mean(te_null_n2e))
        results['te_null_eva_to_neo'].append(np.mean(te_null_e2n))

        sig_n2e = '*' if p_n2e < 0.05 else ''
        sig_e2n = '*' if p_e2n < 0.05 else ''
        print(f"{lag:<5} {te_n2e:<12.6f} {te_e2n:<12.6f} {p_n2e:<10.3f}{sig_n2e} {p_e2n:<10.3f}{sig_e2n}")

    # Lag óptimo (máxima TE significativa)
    te_n2e_arr = np.array(results['te_neo_to_eva'])
    te_e2n_arr = np.array(results['te_eva_to_neo'])
    p_n2e_arr = np.array(results['p_neo_to_eva'])
    p_e2n_arr = np.array(results['p_eva_to_neo'])

    # Enmascarar no significativos
    te_n2e_sig = np.where(p_n2e_arr < 0.05, te_n2e_arr, 0)
    te_e2n_sig = np.where(p_e2n_arr < 0.05, te_e2n_arr, 0)

    lag_opt_n2e = np.argmax(te_n2e_sig) + 1 if np.any(te_n2e_sig > 0) else -1
    lag_opt_e2n = np.argmax(te_e2n_sig) + 1 if np.any(te_e2n_sig > 0) else -1

    results['lag_optimal_neo_to_eva'] = lag_opt_n2e
    results['lag_optimal_eva_to_neo'] = lag_opt_e2n
    results['te_max_neo_to_eva'] = float(np.max(te_n2e_arr))
    results['te_max_eva_to_neo'] = float(np.max(te_e2n_arr))
    results['n_significant_neo_to_eva'] = int(np.sum(p_n2e_arr < 0.05))
    results['n_significant_eva_to_neo'] = int(np.sum(p_e2n_arr < 0.05))

    print("\n" + "=" * 50)
    print("Resumen:")
    print(f"  TE(NEO→EVA) max: {results['te_max_neo_to_eva']:.6f} en lag={lag_opt_n2e}")
    print(f"  TE(EVA→NEO) max: {results['te_max_eva_to_neo']:.6f} en lag={lag_opt_e2n}")
    print(f"  Lags significativos N→E: {results['n_significant_neo_to_eva']}/{L}")
    print(f"  Lags significativos E→N: {results['n_significant_eva_to_neo']}/{L}")

    # Dirección dominante
    if results['te_max_neo_to_eva'] > results['te_max_eva_to_neo'] * 1.5:
        direction = "NEO → EVA"
    elif results['te_max_eva_to_neo'] > results['te_max_neo_to_eva'] * 1.5:
        direction = "EVA → NEO"
    else:
        direction = "Bidireccional"

    results['dominant_direction'] = direction
    print(f"  Dirección dominante: {direction}")

    return results


def main():
    """Ejecutar análisis de direccionalidad."""

    print("=" * 70)
    print("Análisis de Direccionalidad - Transfer Entropy")
    print("=" * 70)

    # Cargar datos de corrida reciente
    try:
        with open('/root/NEO_EVA/results/phase6_v2_neo.json') as f:
            neo_data = json.load(f)
        with open('/root/NEO_EVA/results/phase6_v2_eva.json') as f:
            eva_data = json.load(f)
    except FileNotFoundError:
        print("Archivos no encontrados. Ejecutando corrida nueva...")
        import sys
        sys.path.insert(0, '/root/NEO_EVA/tools')
        from phase6_coupled_system_v2 import CoupledSystemRunner

        runner = CoupledSystemRunner(enable_coupling=True)
        runner.run(cycles=2000, verbose=False)

        neo_data = {'series': runner.neo.series}
        eva_data = {'series': runner.eva.series}

    neo_series = neo_data['series']
    eva_series = eva_data['series']

    print(f"\nDatos cargados: {len(neo_series)} ciclos")

    # Análisis
    results = analyze_directionality(neo_series, eva_series, n_surrogates=100)

    # Guardar resultados
    output_path = '/root/NEO_EVA/repro/directionality_te.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Resultados guardados: {output_path}")

    return results


if __name__ == "__main__":
    main()
