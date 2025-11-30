#!/usr/bin/env python3
"""
Phase 8 Ablaciones
==================
Tres ablaciones mínimas:
1. sin-reciprocidad: R_soc_ema = 0.5 fijo
2. sin-temperatura: γ = 1.0 fijo (mediana histórica aproximada)
3. sin-refractario: damping = 1.0 siempre
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, '/root/NEO_EVA/tools')

# Importar el módulo principal
from phase8_social_potentiated import (
    PotentiatedWorld, run_experiment, LifeState
)

def run_ablation(name: str, cycles: int = 5000):
    """Ejecuta una ablación específica."""
    output_dir = f"/root/NEO_EVA/results/ablation_{name}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"ABLACIÓN: {name.upper()}")
    print(f"{'='*70}")

    # Crear mundos normales
    neo = PotentiatedWorld("NEO", np.array([1.0, 0.0, 0.0]), specialization="mdl")
    eva = PotentiatedWorld("EVA", np.array([1/3, 1/3, 1/3]), specialization="mi_te")

    # Aplicar ablación
    if name == "sin_reciprocidad":
        # Fijar R_soc_ema a 0.5
        class FixedRSoc:
            R_soc_ema = 0.5
            R_soc_history = []
            in_coupling = False
            def start_coupling_window(self): pass
            def add_metric(self, *args): pass
            def end_coupling_window(self, w): return 0.5
            def get_R_soc_ema_rank(self): return 0.5
        neo.voluntary.social_reward = FixedRSoc()
        eva.voluntary.social_reward = FixedRSoc()

    elif name == "sin_temperatura":
        # Fijar γ a 1.0
        original_compute_gamma_neo = neo.voluntary.compute_gamma
        original_compute_gamma_eva = eva.voluntary.compute_gamma
        neo.voluntary.compute_gamma = lambda: 1.0
        eva.voluntary.compute_gamma = lambda: 1.0

    elif name == "sin_refractario":
        # Fijar damping a 1.0
        class NoRefractory:
            def trigger(self, *args): pass
            def get_damping(self, t): return 1.0
        neo.voluntary.refractory = NoRefractory()
        eva.voluntary.refractory = NoRefractory()

    # Ejecutar simulación
    bilateral_events = []

    for t in range(1, cycles + 1):
        neo_signals = neo._compute_signals()
        eva_signals = eva._compute_signals()

        neo_willing_now, neo_gate_now = neo._pre_step_willingness(eva_signals)
        eva_willing_now, eva_gate_now = eva._pre_step_willingness(neo_signals)

        bilateral = neo_willing_now and eva_willing_now and neo_gate_now and eva_gate_now

        neo.step(other_signals=eva_signals, bilateral_consent=bilateral,
                 precomputed_willing=neo_willing_now, precomputed_gate=neo_gate_now)
        eva.step(other_signals=neo_signals, bilateral_consent=bilateral,
                 precomputed_willing=eva_willing_now, precomputed_gate=eva_gate_now)

        if bilateral:
            bilateral_events.append({
                't': t,
                'neo_mode': neo.current_mode,
                'eva_mode': eva.current_mode,
            })

    # Calcular métricas
    from sklearn.metrics import roc_auc_score

    bilateral_ts = set(e['t'] for e in bilateral_events)

    # AUC
    data = [(r['t'], r['pi']) for r in neo.consent_log
            if not r.get('warmup') and r['pi'] is not None]
    if len(data) > 100:
        ts, pis = zip(*data)
        labels = [1 if t in bilateral_ts else 0 for t in ts]
        if sum(labels) > 5:
            auc = roc_auc_score(labels, pis)
        else:
            auc = None
    else:
        auc = None

    # Rho extremos
    rhos = [r['rho'] for r in neo.consent_log if r['rho'] is not None]
    rho_p95 = np.percentile(rhos, 95) if rhos else None

    # Var extremos
    vars_I = [r['var_I'] for r in neo.consent_log if r['var_I'] is not None]
    var_p25 = np.percentile(vars_I, 25) if vars_I else None

    results = {
        'ablation': name,
        'cycles': cycles,
        'bilateral_events': len(bilateral_events),
        'auc': auc,
        'rho_p95': rho_p95,
        'var_p25': var_p25,
        'neo_coupling_count': neo.coupling_count,
        'eva_coupling_count': eva.coupling_count,
    }

    print(f"\nResultados {name}:")
    print(f"  Bilateral events: {len(bilateral_events)}")
    print(f"  AUC: {auc:.4f}" if auc else "  AUC: N/A")
    print(f"  ρ_P95: {rho_p95:.4f}" if rho_p95 else "  ρ_P95: N/A")
    print(f"  Var_P25: {var_p25:.6f}" if var_p25 else "  Var_P25: N/A")

    # Guardar
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    with open(f"{output_dir}/bilateral_events.json", 'w') as f:
        json.dump(bilateral_events, f)

    return results


def run_baseline(cycles: int = 5000):
    """Ejecuta baseline (sin ablaciones)."""
    output_dir = "/root/NEO_EVA/results/ablation_baseline"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("BASELINE (sin ablaciones)")
    print(f"{'='*70}")

    neo = PotentiatedWorld("NEO", np.array([1.0, 0.0, 0.0]), specialization="mdl")
    eva = PotentiatedWorld("EVA", np.array([1/3, 1/3, 1/3]), specialization="mi_te")

    bilateral_events = []

    for t in range(1, cycles + 1):
        neo_signals = neo._compute_signals()
        eva_signals = eva._compute_signals()

        neo_willing_now, neo_gate_now = neo._pre_step_willingness(eva_signals)
        eva_willing_now, eva_gate_now = eva._pre_step_willingness(neo_signals)

        bilateral = neo_willing_now and eva_willing_now and neo_gate_now and eva_gate_now

        neo.step(other_signals=eva_signals, bilateral_consent=bilateral,
                 precomputed_willing=neo_willing_now, precomputed_gate=neo_gate_now)
        eva.step(other_signals=neo_signals, bilateral_consent=bilateral,
                 precomputed_willing=eva_willing_now, precomputed_gate=eva_gate_now)

        if bilateral:
            bilateral_events.append({'t': t})

    from sklearn.metrics import roc_auc_score
    bilateral_ts = set(e['t'] for e in bilateral_events)

    data = [(r['t'], r['pi']) for r in neo.consent_log
            if not r.get('warmup') and r['pi'] is not None]
    if len(data) > 100:
        ts, pis = zip(*data)
        labels = [1 if t in bilateral_ts else 0 for t in ts]
        auc = roc_auc_score(labels, pis) if sum(labels) > 5 else None
    else:
        auc = None

    rhos = [r['rho'] for r in neo.consent_log if r['rho'] is not None]
    vars_I = [r['var_I'] for r in neo.consent_log if r['var_I'] is not None]

    results = {
        'ablation': 'baseline',
        'cycles': cycles,
        'bilateral_events': len(bilateral_events),
        'auc': auc,
        'rho_p95': np.percentile(rhos, 95) if rhos else None,
        'var_p25': np.percentile(vars_I, 25) if vars_I else None,
    }

    print(f"\nResultados baseline:")
    print(f"  Bilateral events: {len(bilateral_events)}")
    print(f"  AUC: {auc:.4f}" if auc else "  AUC: N/A")

    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=5000)
    args = parser.parse_args()

    # Ejecutar todas las ablaciones
    results = {}

    results['baseline'] = run_baseline(args.cycles)
    results['sin_reciprocidad'] = run_ablation('sin_reciprocidad', args.cycles)
    results['sin_temperatura'] = run_ablation('sin_temperatura', args.cycles)
    results['sin_refractario'] = run_ablation('sin_refractario', args.cycles)

    # Comparación
    print("\n" + "="*70)
    print("COMPARACIÓN DE ABLACIONES")
    print("="*70)
    print(f"\n{'Condición':<20} {'Eventos':>10} {'AUC':>10} {'ρ_P95':>10} {'Var_P25':>12}")
    print("-" * 65)

    for name, r in results.items():
        auc_str = f"{r['auc']:.4f}" if r['auc'] else "N/A"
        rho_str = f"{r['rho_p95']:.4f}" if r['rho_p95'] else "N/A"
        var_str = f"{r['var_p25']:.6f}" if r['var_p25'] else "N/A"
        print(f"{name:<20} {r['bilateral_events']:>10} {auc_str:>10} {rho_str:>10} {var_str:>12}")

    # Guardar resumen
    with open("/root/NEO_EVA/results/ablation_summary.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Resumen guardado en ablation_summary.json")
