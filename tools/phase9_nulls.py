#!/usr/bin/env python3
"""
Phase 9: Nulos Endogenos (Falsacion)
====================================
Genera distribuciones nulas para cada indice alpha mediante:
1. Shuffle por bloques (preservando ACF local)
2. Surrogates de fase (preservando espectro)
3. Coupling-shuffle (permutando marcas ON/OFF)

Bootstrap por cuantiles, sin constantes fijas.
"""

import sys
import os
import json
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
from scipy.fft import fft, ifft
from dataclasses import dataclass, asdict

sys.path.insert(0, '/root/NEO_EVA/tools')

from phase9_plasticity import (
    compute_acf_lag, rolling_window_indices, rank_in_window,
    compute_alpha_affect, compute_alpha_hyst, compute_alpha_switch,
    compute_alpha_consent_elast, compute_alpha_cross_sus,
    compute_alpha_weights, compute_alpha_manifold,
    WorldSignals, extract_signals, load_data
)


# =============================================================================
# METODOS DE GENERACION DE NULOS
# =============================================================================

def block_shuffle(x: np.ndarray, block_size: int) -> np.ndarray:
    """
    Shuffle por bloques preservando estructura local (ACF).
    """
    T = len(x)
    n_blocks = T // block_size

    if n_blocks < 2:
        return x.copy()

    # Dividir en bloques
    blocks = [x[i*block_size:(i+1)*block_size] for i in range(n_blocks)]

    # Resto
    remainder = x[n_blocks*block_size:]

    # Permutar bloques
    np.random.shuffle(blocks)

    # Reconstruir
    result = np.concatenate(blocks + [remainder] if len(remainder) > 0 else blocks)

    return result


def phase_surrogate(x: np.ndarray) -> np.ndarray:
    """
    Surrogate de fase: preserva espectro de potencia pero randomiza fases.
    """
    T = len(x)

    # FFT
    X = fft(x)

    # Randomizar fases manteniendo simetria conjugada
    phases = np.random.uniform(0, 2*np.pi, T//2)

    # Construir nuevas fases
    new_phases = np.zeros(T)
    new_phases[1:T//2] = phases[1:]
    new_phases[T//2+1:] = -phases[1:][::-1]

    # Aplicar nuevas fases manteniendo amplitudes
    X_surrogate = np.abs(X) * np.exp(1j * new_phases)

    # IFFT
    result = np.real(ifft(X_surrogate))

    return result


def coupling_shuffle(bilateral_events: List[dict], T: int) -> List[dict]:
    """
    Shuffle de marcas ON/OFF conservando tasa.
    Permuta los tiempos de eventos bilaterales.
    """
    n_events = len(bilateral_events)

    if n_events == 0:
        return []

    # Generar nuevos tiempos aleatorios
    new_times = sorted(np.random.choice(T, n_events, replace=False))

    # Crear nuevos eventos con tiempos permutados
    shuffled_events = []
    for i, event in enumerate(bilateral_events):
        new_event = event.copy()
        new_event['t'] = int(new_times[i])
        shuffled_events.append(new_event)

    return shuffled_events


# =============================================================================
# BOOTSTRAP DE NULOS
# =============================================================================

@dataclass
class NullDistribution:
    """Distribucion nula de un indice."""
    name: str
    observed: float
    null_mean: float
    null_std: float
    null_q025: float
    null_q500: float
    null_q975: float
    p_value: float  # Proporcion de nulos >= observado
    n_bootstrap: int


def bootstrap_null_intra(signals: WorldSignals,
                         alpha_name: str,
                         n_bootstrap: int = 100) -> NullDistribution:
    """
    Bootstrap para indices intramundo usando block shuffle y phase surrogates.
    """
    w = signals.w
    block_size = max(10, w // 2)

    # Calcular valor observado
    if alpha_name == 'alpha_affect':
        observed, _ = compute_alpha_affect(signals.V, signals.A, signals.D, w)
    elif alpha_name == 'alpha_hyst':
        observed, _ = compute_alpha_hyst(signals.V, signals.A, w)
    elif alpha_name == 'alpha_switch':
        observed, _ = compute_alpha_switch(signals.states, w)
    else:
        return None

    # Generar nulos
    null_values = []

    for i in range(n_bootstrap):
        # Alternar entre block shuffle y phase surrogate
        if i % 2 == 0:
            V_null = block_shuffle(signals.V, block_size)
            A_null = block_shuffle(signals.A, block_size)
            D_null = block_shuffle(signals.D, block_size)
            states_null = block_shuffle(signals.states, block_size)
        else:
            V_null = phase_surrogate(signals.V)
            A_null = phase_surrogate(signals.A)
            D_null = phase_surrogate(signals.D)
            # Para estados discretos, usar block shuffle
            states_null = block_shuffle(signals.states, block_size)

        if alpha_name == 'alpha_affect':
            val, _ = compute_alpha_affect(V_null, A_null, D_null, w)
        elif alpha_name == 'alpha_hyst':
            val, _ = compute_alpha_hyst(V_null, A_null, w)
        elif alpha_name == 'alpha_switch':
            val, _ = compute_alpha_switch(states_null.astype(np.int32), w)

        null_values.append(val)

    null_values = np.array(null_values)

    # Calcular p-value (proporcion de nulos >= observado)
    p_value = np.mean(null_values >= observed)

    return NullDistribution(
        name=alpha_name,
        observed=float(observed),
        null_mean=float(np.mean(null_values)),
        null_std=float(np.std(null_values)),
        null_q025=float(np.percentile(null_values, 2.5)),
        null_q500=float(np.percentile(null_values, 50)),
        null_q975=float(np.percentile(null_values, 97.5)),
        p_value=float(p_value),
        n_bootstrap=n_bootstrap
    )


def bootstrap_null_inter(neo_signals: WorldSignals,
                         eva_signals: WorldSignals,
                         bilateral_events: List[dict],
                         alpha_name: str,
                         n_bootstrap: int = 100) -> NullDistribution:
    """
    Bootstrap para indices intermundos usando coupling shuffle.
    """
    T = min(neo_signals.T, eva_signals.T)

    # Calcular valor observado
    if alpha_name == 'alpha_consent_elast':
        observed = compute_alpha_consent_elast(neo_signals, eva_signals, bilateral_events)
    elif alpha_name == 'alpha_cross_sus':
        observed = compute_alpha_cross_sus(neo_signals, eva_signals, bilateral_events)
    else:
        return None

    # Generar nulos
    null_values = []

    for i in range(n_bootstrap):
        # Coupling shuffle: permutar tiempos de eventos
        shuffled_events = coupling_shuffle(bilateral_events, T)

        if alpha_name == 'alpha_consent_elast':
            val = compute_alpha_consent_elast(neo_signals, eva_signals, shuffled_events)
        elif alpha_name == 'alpha_cross_sus':
            val = compute_alpha_cross_sus(neo_signals, eva_signals, shuffled_events)

        null_values.append(val)

    null_values = np.array(null_values)
    p_value = np.mean(null_values >= observed)

    return NullDistribution(
        name=alpha_name,
        observed=float(observed),
        null_mean=float(np.mean(null_values)),
        null_std=float(np.std(null_values)),
        null_q025=float(np.percentile(null_values, 2.5)),
        null_q500=float(np.percentile(null_values, 50)),
        null_q975=float(np.percentile(null_values, 97.5)),
        p_value=float(p_value),
        n_bootstrap=n_bootstrap
    )


def bootstrap_null_struct(neo_signals: WorldSignals,
                          eva_signals: WorldSignals,
                          alpha_name: str,
                          n_bootstrap: int = 100) -> NullDistribution:
    """
    Bootstrap para indices estructurales usando block shuffle.
    """
    w = min(neo_signals.w, eva_signals.w)
    block_size = max(10, w // 2)

    # Calcular valor observado
    if alpha_name == 'alpha_weights':
        observed, _ = compute_alpha_weights(neo_signals, eva_signals)
    elif alpha_name == 'alpha_manifold':
        observed, _ = compute_alpha_manifold(neo_signals, eva_signals)
    else:
        return None

    # Generar nulos
    null_values = []

    for i in range(n_bootstrap):
        # Crear copias con datos shuffled
        neo_copy = WorldSignals(
            name=neo_signals.name, T=neo_signals.T, w=neo_signals.w,
            r=neo_signals.r, s=neo_signals.s, m=neo_signals.m, c=neo_signals.c,
            R_soc=neo_signals.R_soc, e=neo_signals.e, q=neo_signals.q, h=neo_signals.h,
            V=block_shuffle(neo_signals.V, block_size),
            A=block_shuffle(neo_signals.A, block_size),
            D=block_shuffle(neo_signals.D, block_size),
            states=neo_signals.states, pi=neo_signals.pi, modes=neo_signals.modes,
            weights=neo_signals.weights
        )

        eva_copy = WorldSignals(
            name=eva_signals.name, T=eva_signals.T, w=eva_signals.w,
            r=eva_signals.r, s=eva_signals.s, m=eva_signals.m, c=eva_signals.c,
            R_soc=eva_signals.R_soc, e=eva_signals.e, q=eva_signals.q, h=eva_signals.h,
            V=block_shuffle(eva_signals.V, block_size),
            A=block_shuffle(eva_signals.A, block_size),
            D=block_shuffle(eva_signals.D, block_size),
            states=eva_signals.states, pi=eva_signals.pi, modes=eva_signals.modes,
            weights=eva_signals.weights
        )

        if alpha_name == 'alpha_weights':
            val, _ = compute_alpha_weights(neo_copy, eva_copy)
        elif alpha_name == 'alpha_manifold':
            val, _ = compute_alpha_manifold(neo_copy, eva_copy)

        null_values.append(val)

    null_values = np.array(null_values)
    p_value = np.mean(null_values >= observed)

    return NullDistribution(
        name=alpha_name,
        observed=float(observed),
        null_mean=float(np.mean(null_values)),
        null_std=float(np.std(null_values)),
        null_q025=float(np.percentile(null_values, 2.5)),
        null_q500=float(np.percentile(null_values, 50)),
        null_q975=float(np.percentile(null_values, 97.5)),
        p_value=float(p_value),
        n_bootstrap=n_bootstrap
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Phase 9: Nulos Endogenos')
    parser.add_argument('--data-dir', default='/root/NEO_EVA/results/phase8_long',
                        help='Directorio de datos de entrada')
    parser.add_argument('--output-dir', default='/root/NEO_EVA/results/phase9',
                        help='Directorio de salida')
    parser.add_argument('--n-bootstrap', type=int, default=100,
                        help='Numero de iteraciones bootstrap')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("PHASE 9: NULOS ENDOGENOS (FALSACION)")
    print("=" * 70)

    # Cargar datos
    print("\n[1] Cargando datos...")
    (affect_neo, affect_eva, consent_neo, consent_eva,
     voluntary_neo, voluntary_eva, bilateral) = load_data(args.data_dir)

    # Extraer senales
    print("\n[2] Extrayendo senales...")
    neo_signals = extract_signals(affect_neo, consent_neo, voluntary_neo, 'NEO')
    eva_signals = extract_signals(affect_eva, consent_eva, voluntary_eva, 'EVA')

    print(f"    Bootstrap iterations: {args.n_bootstrap}")

    # Calcular nulos intramundo
    print("\n[3] Calculando nulos intramundo NEO...")
    np.random.seed(42)

    nulls = {}

    for alpha_name in ['alpha_affect', 'alpha_hyst', 'alpha_switch']:
        print(f"    - {alpha_name}...")
        null_dist = bootstrap_null_intra(neo_signals, alpha_name, args.n_bootstrap)
        if null_dist:
            nulls[f'neo_{alpha_name}'] = asdict(null_dist)
            print(f"      observed={null_dist.observed:.6f}, "
                  f"null_mean={null_dist.null_mean:.6f}, "
                  f"p={null_dist.p_value:.3f}")

    print("\n[4] Calculando nulos intramundo EVA...")
    for alpha_name in ['alpha_affect', 'alpha_hyst', 'alpha_switch']:
        print(f"    - {alpha_name}...")
        null_dist = bootstrap_null_intra(eva_signals, alpha_name, args.n_bootstrap)
        if null_dist:
            nulls[f'eva_{alpha_name}'] = asdict(null_dist)
            print(f"      observed={null_dist.observed:.6f}, "
                  f"null_mean={null_dist.null_mean:.6f}, "
                  f"p={null_dist.p_value:.3f}")

    # Calcular nulos intermundos
    print("\n[5] Calculando nulos intermundos...")
    for alpha_name in ['alpha_consent_elast', 'alpha_cross_sus']:
        print(f"    - {alpha_name}...")
        null_dist = bootstrap_null_inter(neo_signals, eva_signals, bilateral,
                                         alpha_name, args.n_bootstrap)
        if null_dist:
            nulls[f'inter_{alpha_name}'] = asdict(null_dist)
            print(f"      observed={null_dist.observed:.6f}, "
                  f"null_mean={null_dist.null_mean:.6f}, "
                  f"p={null_dist.p_value:.3f}")

    # Calcular nulos estructurales
    print("\n[6] Calculando nulos estructurales...")
    for alpha_name in ['alpha_weights', 'alpha_manifold']:
        print(f"    - {alpha_name}...")
        null_dist = bootstrap_null_struct(neo_signals, eva_signals,
                                          alpha_name, args.n_bootstrap)
        if null_dist:
            nulls[f'struct_{alpha_name}'] = asdict(null_dist)
            print(f"      observed={null_dist.observed:.6f}, "
                  f"null_mean={null_dist.null_mean:.6f}, "
                  f"p={null_dist.p_value:.3f}")

    # Guardar resultados
    print("\n[7] Guardando nulos...")
    with open(f"{args.output_dir}/nulls_bootstrap.json", 'w') as f:
        json.dump(nulls, f, indent=2)

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE NULOS")
    print("=" * 70)
    print(f"\n{'Indice':<30} {'Observado':>12} {'Nulo_med':>12} {'p-value':>10}")
    print("-" * 70)

    for name, data in nulls.items():
        print(f"{name:<30} {data['observed']:>12.6f} {data['null_mean']:>12.6f} {data['p_value']:>10.3f}")

    print(f"\n[OK] Nulos guardados en {args.output_dir}/nulls_bootstrap.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
