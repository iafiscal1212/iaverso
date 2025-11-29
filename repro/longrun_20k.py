#!/usr/bin/env python3
"""
Corrida larga de 20k ciclos con snapshots cada 1k.
Genera longrun_20k.json y longrun_20k.csv
"""

import os
import sys
import json
import csv
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, '/root/NEO_EVA/tools')
from phase6_coupled_system_v2 import CoupledSystemRunner

def run_longrun(cycles=20000, snapshot_interval=1000):
    """Ejecuta corrida larga con snapshots."""

    print("=" * 70)
    print(f"Corrida Larga: {cycles} ciclos, snapshots cada {snapshot_interval}")
    print("=" * 70)

    runner = CoupledSystemRunner(enable_coupling=True)

    snapshots = []
    start_time = time.time()

    for i in range(cycles):
        runner.neo.step(enable_coupling=True)
        runner.eva.step(enable_coupling=True)

        # Snapshot cada interval
        if (i + 1) % snapshot_interval == 0:
            neo_I = runner.neo.I.copy()
            eva_I = runner.eva.I.copy()

            # Calcular correlaciones hasta este punto
            if len(runner.neo.series) > 10:
                neo_arr = np.array([[s['S_new'], s['N_new'], s['C_new']] for s in runner.neo.series])
                eva_arr = np.array([[s['S_new'], s['N_new'], s['C_new']] for s in runner.eva.series])

                corrs = []
                for j in range(3):
                    r = np.corrcoef(neo_arr[:, j], eva_arr[:, j])[0, 1]
                    corrs.append(float(r) if not np.isnan(r) else 0.0)
                mean_corr = np.mean(corrs)
            else:
                corrs = [0.0, 0.0, 0.0]
                mean_corr = 0.0

            snapshot = {
                'cycle': i + 1,
                'timestamp': datetime.now().isoformat(),
                'neo_I': neo_I.tolist(),
                'eva_I': eva_I.tolist(),
                'neo_coupling_acts': runner.neo.coupling_activations,
                'eva_coupling_acts': runner.eva.coupling_activations,
                'corr_S': corrs[0],
                'corr_N': corrs[1],
                'corr_C': corrs[2],
                'mean_corr': mean_corr,
                'tau_last': runner.neo.diagnostics['tau'][-1] if runner.neo.diagnostics['tau'] else 0,
            }
            snapshots.append(snapshot)

            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  t={i+1:5d}: r̄={mean_corr:+.4f}, NEO_acts={runner.neo.coupling_activations}, "
                  f"EVA_acts={runner.eva.coupling_activations}, τ={snapshot['tau_last']:.6f} "
                  f"[{rate:.1f} ciclos/s]")

    elapsed = time.time() - start_time

    # Resultados finales
    final_results = {
        'config': {
            'cycles': cycles,
            'snapshot_interval': snapshot_interval,
            'coupling_enabled': True,
        },
        'timing': {
            'elapsed_seconds': elapsed,
            'cycles_per_second': cycles / elapsed,
        },
        'final_state': {
            'neo_I': runner.neo.I.tolist(),
            'eva_I': runner.eva.I.tolist(),
            'neo_coupling_activations': runner.neo.coupling_activations,
            'eva_coupling_activations': runner.eva.coupling_activations,
            'neo_gate_activations': runner.neo.gate_activations,
            'eva_gate_activations': runner.eva.gate_activations,
        },
        'snapshots': snapshots,
        'quantiles_neo': runner.neo.get_quantile_report(),
        'quantiles_eva': runner.eva.get_quantile_report(),
    }

    # Guardar JSON
    json_path = '/root/NEO_EVA/repro/longrun_20k.json'
    with open(json_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\n[OK] JSON guardado: {json_path}")

    # Guardar CSV
    csv_path = '/root/NEO_EVA/repro/longrun_20k.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=snapshots[0].keys())
        writer.writeheader()
        writer.writerows(snapshots)
    print(f"[OK] CSV guardado: {csv_path}")

    print("\n" + "=" * 70)
    print("Resumen Final:")
    print(f"  Ciclos totales: {cycles}")
    print(f"  Tiempo: {elapsed:.2f}s ({cycles/elapsed:.1f} ciclos/s)")
    print(f"  NEO coupling activations: {runner.neo.coupling_activations}/{cycles} "
          f"({100*runner.neo.coupling_activations/cycles:.1f}%)")
    print(f"  EVA coupling activations: {runner.eva.coupling_activations}/{cycles} "
          f"({100*runner.eva.coupling_activations/cycles:.1f}%)")
    print(f"  Correlación final media: {snapshots[-1]['mean_corr']:.4f}")
    print("=" * 70)

    return final_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=20000)
    parser.add_argument("--interval", type=int, default=1000)
    args = parser.parse_args()

    run_longrun(cycles=args.cycles, snapshot_interval=args.interval)
