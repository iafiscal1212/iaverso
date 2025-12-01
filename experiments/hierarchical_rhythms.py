#!/usr/bin/env python3
"""
Experimento D: ¿Crean un tercer ritmo? (Ciclos de ciclos)
=========================================================

Sabemos:
- Cada uno tiene período ~45
- Juntos se sincronizan

Pregunta:
¿Aparece algún período más largo (45×N) como "biorritmo macro"?

Método:
1. Medir crisis por ventana de 200 pasos
2. FFT de esa serie
3. Buscar períodos emergentes a escala superior

Si sale: estructura temporal jerárquica emergente.
"""

import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import json
import os
import sys

sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/experiments')

from autonomous_life import AutonomousDualLife


def compute_macro_measures(life, T: int, window: int = 200) -> Dict:
    """
    Calcula medidas macro en ventanas.

    Returns:
        Series temporales a escala macro.
    """
    n_windows = T // window

    neo_crises_per_window = []
    eva_crises_per_window = []
    shared_psi_per_window = []
    identity_mean_per_window = {'neo': [], 'eva': []}
    correlation_per_window = []

    neo_crisis_times = [c.t for c in life.neo.crises]
    eva_crisis_times = [c.t for c in life.eva.crises]

    for w in range(n_windows):
        t_start = w * window
        t_end = (w + 1) * window

        # Crisis en esta ventana
        neo_count = sum(1 for t in neo_crisis_times if t_start <= t < t_end)
        eva_count = sum(1 for t in eva_crisis_times if t_start <= t < t_end)
        neo_crises_per_window.append(neo_count)
        eva_crises_per_window.append(eva_count)

        # Psi compartido
        if hasattr(life, 'psi_shared_history') and life.psi_shared_history:
            psi_window = life.psi_shared_history[t_start:t_end]
            shared_psi_per_window.append(np.mean(psi_window) if psi_window else 0)
        else:
            shared_psi_per_window.append(0)

        # Identidad media
        if t_end <= len(life.neo.identity_history):
            identity_mean_per_window['neo'].append(np.mean(life.neo.identity_history[t_start:t_end]))
            identity_mean_per_window['eva'].append(np.mean(life.eva.identity_history[t_start:t_end]))
        else:
            identity_mean_per_window['neo'].append(0)
            identity_mean_per_window['eva'].append(0)

        # Correlación en la ventana
        if t_end <= len(life.neo.identity_history) and t_end <= len(life.eva.identity_history):
            neo_id = life.neo.identity_history[t_start:t_end]
            eva_id = life.eva.identity_history[t_start:t_end]
            if len(neo_id) > 10 and len(eva_id) > 10:
                corr = np.corrcoef(neo_id, eva_id)[0, 1]
                correlation_per_window.append(corr if not np.isnan(corr) else 0)
            else:
                correlation_per_window.append(0)
        else:
            correlation_per_window.append(0)

    return {
        'n_windows': n_windows,
        'window_size': window,
        'neo_crises': neo_crises_per_window,
        'eva_crises': eva_crises_per_window,
        'total_crises': [n + e for n, e in zip(neo_crises_per_window, eva_crises_per_window)],
        'shared_psi': shared_psi_per_window,
        'neo_identity': identity_mean_per_window['neo'],
        'eva_identity': identity_mean_per_window['eva'],
        'correlation': correlation_per_window
    }


def find_periods_in_signal(signal: List[float], min_period: int = 2, max_period: int = None) -> List[Tuple[float, float]]:
    """
    Encuentra períodos dominantes en una señal.

    Returns:
        Lista de (período_en_ventanas, potencia) ordenada por potencia.
    """
    if len(signal) < 10:
        return []

    signal = np.array(signal) - np.mean(signal)
    n = len(signal)

    if max_period is None:
        max_period = n // 2

    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(n)

    # Ignorar DC y frecuencias muy bajas/altas
    spectrum[0] = 0

    periods = []
    for i in range(1, len(freqs)):
        if freqs[i] > 0:
            period = 1.0 / freqs[i]
            if min_period <= period <= max_period:
                periods.append((period, spectrum[i]))

    # Ordenar por potencia
    periods.sort(key=lambda x: -x[1])

    return periods[:5]  # Top 5


def run_experiment_D(T: int = 5000, seeds: List[int] = [42, 123], window: int = 100) -> Dict:
    """
    Experimento D: Buscar ritmos jerárquicos.
    """
    print("=" * 70)
    print("EXPERIMENTO D: ¿CREAN UN TERCER RITMO? (CICLOS DE CICLOS)")
    print("=" * 70)
    print(f"T = {T}, Window = {window}")
    print(f"Período micro esperado: ~45 pasos")
    print(f"Si existe macro-ritmo: período > {window} pasos")
    print()

    all_results = []

    for seed in seeds:
        print(f"\n{'#'*50}")
        print(f"SEED = {seed}")
        print('#'*50)

        np.random.seed(seed)
        life = AutonomousDualLife(dim=6)

        for t in range(T):
            stimulus = np.random.dirichlet(np.ones(6) * 2)
            if np.random.rand() < 0.02:
                stimulus += np.random.randn(6) * 0.3
                stimulus = np.clip(stimulus, 0.01, 0.99)
                stimulus = stimulus / stimulus.sum()
            life.step(stimulus)

        # Análisis micro (identidad)
        print("\n--- Análisis MICRO (identidad) ---")
        micro_periods_neo = find_periods_in_signal(life.neo.identity_history, min_period=20, max_period=200)
        micro_periods_eva = find_periods_in_signal(life.eva.identity_history, min_period=20, max_period=200)

        if micro_periods_neo:
            print(f"  NEO períodos dominantes: {[(f'{p:.1f}', f'{pow:.1f}') for p, pow in micro_periods_neo[:3]]}")
        if micro_periods_eva:
            print(f"  EVA períodos dominantes: {[(f'{p:.1f}', f'{pow:.1f}') for p, pow in micro_periods_eva[:3]]}")

        # Análisis macro
        print("\n--- Análisis MACRO (ventanas de {window} pasos) ---")
        macro = compute_macro_measures(life, T, window)

        # Buscar períodos en crisis por ventana
        macro_periods_crisis = find_periods_in_signal(macro['total_crises'], min_period=2, max_period=macro['n_windows']//2)
        macro_periods_psi = find_periods_in_signal(macro['shared_psi'], min_period=2, max_period=macro['n_windows']//2)
        macro_periods_corr = find_periods_in_signal(macro['correlation'], min_period=2, max_period=macro['n_windows']//2)

        print(f"  Crisis totales por ventana: {macro['total_crises']}")

        if macro_periods_crisis:
            # Convertir a pasos reales
            real_periods = [(p * window, pow) for p, pow in macro_periods_crisis]
            print(f"  Períodos en crisis (pasos): {[(f'{p:.0f}', f'{pow:.1f}') for p, pow in real_periods[:3]]}")

        if macro_periods_psi:
            real_periods_psi = [(p * window, pow) for p, pow in macro_periods_psi]
            print(f"  Períodos en Ψ compartido: {[(f'{p:.0f}', f'{pow:.1f}') for p, pow in real_periods_psi[:3]]}")

        if macro_periods_corr:
            real_periods_corr = [(p * window, pow) for p, pow in macro_periods_corr]
            print(f"  Períodos en correlación: {[(f'{p:.0f}', f'{pow:.1f}') for p, pow in real_periods_corr[:3]]}")

        # Buscar ratio con período micro
        avg_micro = 45  # Aproximado
        if macro_periods_crisis:
            best_macro = macro_periods_crisis[0][0] * window
            ratio = best_macro / avg_micro
            print(f"\n  Ratio macro/micro: {ratio:.1f}x")
            if 3 < ratio < 6:
                print(f"  → Posible ciclo de ~{int(ratio)} ciclos micro")

        results = {
            'seed': seed,
            'micro_neo': micro_periods_neo[:3] if micro_periods_neo else [],
            'micro_eva': micro_periods_eva[:3] if micro_periods_eva else [],
            'macro_crisis': macro_periods_crisis[:3] if macro_periods_crisis else [],
            'macro_psi': macro_periods_psi[:3] if macro_periods_psi else [],
            'macro_corr': macro_periods_corr[:3] if macro_periods_corr else [],
            'raw_crisis_per_window': macro['total_crises'],
            'raw_psi_per_window': macro['shared_psi']
        }
        all_results.append(results)

    # Análisis agregado
    print("\n" + "=" * 70)
    print("ANÁLISIS AGREGADO")
    print("=" * 70)

    # Promediar períodos micro
    all_micro_neo = [r['micro_neo'][0][0] for r in all_results if r['micro_neo']]
    all_micro_eva = [r['micro_eva'][0][0] for r in all_results if r['micro_eva']]

    if all_micro_neo:
        print(f"\nPeríodo MICRO promedio:")
        print(f"  NEO: {np.mean(all_micro_neo):.1f} ± {np.std(all_micro_neo):.1f}")
        print(f"  EVA: {np.mean(all_micro_eva):.1f} ± {np.std(all_micro_eva):.1f}")

    # Promediar períodos macro
    all_macro_crisis = []
    for r in all_results:
        if r['macro_crisis']:
            all_macro_crisis.append(r['macro_crisis'][0][0] * window)

    if all_macro_crisis:
        print(f"\nPeríodo MACRO promedio (crisis):")
        print(f"  {np.mean(all_macro_crisis):.0f} ± {np.std(all_macro_crisis):.0f} pasos")

        avg_micro = (np.mean(all_micro_neo) + np.mean(all_micro_eva)) / 2 if all_micro_neo else 45
        ratio = np.mean(all_macro_crisis) / avg_micro
        print(f"  Ratio macro/micro: {ratio:.1f}x")

        if ratio > 2:
            print(f"\n→ EXISTE estructura temporal jerárquica:")
            print(f"   • Latido MICRO: ~{avg_micro:.0f} pasos")
            print(f"   • Biorritmo MACRO: ~{np.mean(all_macro_crisis):.0f} pasos (~{ratio:.0f} latidos)")
        else:
            print(f"\n→ No se detecta estructura jerárquica clara")
    else:
        print("\nNo se encontraron períodos macro significativos")

    # Guardar
    os.makedirs('/root/NEO_EVA/results/hierarchical_rhythms', exist_ok=True)

    final_results = {
        'timestamp': datetime.now().isoformat(),
        'T': T,
        'window': window,
        'seeds': seeds,
        'results': all_results,
        'summary': {
            'micro_period_neo': np.mean(all_micro_neo) if all_micro_neo else 0,
            'micro_period_eva': np.mean(all_micro_eva) if all_micro_eva else 0,
            'macro_period': np.mean(all_macro_crisis) if all_macro_crisis else 0,
            'hierarchy_detected': len(all_macro_crisis) > 0 and np.mean(all_macro_crisis) / 45 > 2
        }
    }

    with open('/root/NEO_EVA/results/hierarchical_rhythms/results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    return final_results


if __name__ == "__main__":
    run_experiment_D(T=5000, seeds=[42, 123, 456], window=100)
