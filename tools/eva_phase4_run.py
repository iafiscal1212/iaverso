#!/usr/bin/env python3
"""
EVA Phase 4 Series Generator
============================

Generates EVA intention series with Phase 4 variability for IWVI analysis.
EVA starts from uniform prior [1/3, 1/3, 1/3] and evolves with similar dynamics.
"""

import sys
import os
import json
import math
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/root/NEO_EVA/tools')
from phase4_variability import Phase4Controller


def strong_mirror_descent(I_current: np.ndarray, delta: np.ndarray, eta: float) -> np.ndarray:
    """
    Mirror descent that handles simplex corners properly.
    """
    floor = 0.001
    I_safe = np.maximum(I_current, floor)
    I_safe = I_safe / np.sum(I_safe)

    log_I = np.log(I_safe)
    effective_eta = max(eta, 0.1)
    log_I_new = log_I + effective_eta * delta

    log_I_new = log_I_new - np.max(log_I_new)
    exp_I = np.exp(log_I_new)
    I_new = exp_I / np.sum(exp_I)

    return I_new


def generate_eva_series(cycles: int = 500, perturbation_scale: float = 0.05):
    """
    Generate EVA intention series with Phase 4 variability.

    EVA starts from uniform prior [1/3, 1/3, 1/3] and evolves with
    similar dynamics but independent noise.
    """

    # Initialize controller
    controller = Phase4Controller()

    # Set up Jacobian (EVA uses similar stability)
    J = np.array([
        [0.80, -0.15, -0.35],
        [-0.20, 0.75, -0.25],
        [-0.20, -0.35, 0.75]
    ])
    rho = 0.92
    controller.update_jacobian(J, rho)

    # Tangent basis
    u_c = controller.u_c if controller.u_c is not None else np.array([1, -1, 0]) / np.sqrt(2)
    u_perp = controller.u_perp if controller.u_perp is not None else np.array([1, 1, -2]) / np.sqrt(6)

    # EVA starts from uniform prior
    I = np.array([1/3, 1/3, 1/3])
    T_history = 200  # EVA has shorter history

    series = []
    print(f"Generating {cycles} EVA cycles...")
    print(f"Initial I: [{I[0]:.4f}, {I[1]:.4f}, {I[2]:.4f}]")

    for t in range(cycles):
        # Random noise in tangent plane
        z1 = np.random.randn() * perturbation_scale
        z2 = np.random.randn() * perturbation_scale
        delta = z1 * u_c + z2 * u_perp

        # Strong mirror descent
        I_new = strong_mirror_descent(I, delta, eta=perturbation_scale * 10)

        # Record
        series.append({
            't': T_history + t,
            'S': float(I[0]),
            'N': float(I[1]),
            'C': float(I[2]),
            'S_new': float(I_new[0]),
            'N_new': float(I_new[1]),
            'C_new': float(I_new[2]),
            'delta': float(np.linalg.norm(I_new - I))
        })

        I = I_new.copy()

        if t % 50 == 0:
            print(f"  t={T_history + t}: I=[{I[0]:.4f}, {I[1]:.4f}, {I[2]:.4f}] δ={series[-1]['delta']:.4f}")

    print(f"\nFinal I: [{I[0]:.4f}, {I[1]:.4f}, {I[2]:.4f}]")
    return series


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate EVA Phase 4 series")
    parser.add_argument("--cycles", type=int, default=500, help="Number of cycles")
    parser.add_argument("--scale", type=float, default=0.05, help="Perturbation scale")
    args = parser.parse_args()

    print("=" * 70)
    print("EVA Phase 4 Series Generator")
    print("=" * 70)

    series = generate_eva_series(cycles=args.cycles, perturbation_scale=args.scale)

    # Compute variance
    S_vals = [s['S_new'] for s in series]
    N_vals = [s['N_new'] for s in series]
    C_vals = [s['C_new'] for s in series]

    var_S = np.var(S_vals)
    var_N = np.var(N_vals)
    var_C = np.var(C_vals)

    print(f"\n--- Variance Analysis ---")
    print(f"Var(S): {var_S:.6e}")
    print(f"Var(N): {var_N:.6e}")
    print(f"Var(C): {var_C:.6e}")
    print(f"Total: {var_S + var_N + var_C:.6e}")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "cycles": args.cycles,
        "perturbation_scale": args.scale,
        "initial_I": [1/3, 1/3, 1/3],
        "final_I": [S_vals[-1], N_vals[-1], C_vals[-1]],
        "series": series,
        "variance": {
            "S": float(var_S),
            "N": float(var_N),
            "C": float(var_C),
            "total": float(var_S + var_N + var_C)
        }
    }

    out_path = "/root/NEO_EVA/results/phase4_eva_series.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved: {out_path}")

    # CSV
    csv_path = "/root/NEO_EVA/results/phase4_eva_series.csv"
    with open(csv_path, 'w') as f:
        f.write("t,S,N,C,delta\n")
        for s in series:
            f.write(f"{s['t']},{s['S_new']:.8f},{s['N_new']:.8f},{s['C_new']:.8f},{s['delta']:.8f}\n")
    print(f"[OK] CSV: {csv_path}")


if __name__ == "__main__":
    main()
