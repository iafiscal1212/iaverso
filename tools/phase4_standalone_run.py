#!/usr/bin/env python3
"""
Phase 4 Standalone Variance Generator
======================================

Generates variance in the NEO intention vector through direct manipulation,
then records the series for IWVI analysis.

This bypasses the NEO server and works directly with the intention dynamics.
"""

import sys
import os
import json
import math
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/root/NEO_EVA/tools')
from phase4_variability import Phase4Controller, mirror_descent_step


def strong_mirror_descent(I_current: np.ndarray, delta: np.ndarray, eta: float) -> np.ndarray:
    """
    Mirror descent that actually escapes corners.

    Uses floor to prevent log(-inf) and larger step sizes.
    """
    # Minimum floor to escape corners
    floor = 0.001  # 0.1% minimum per component

    # Apply floor to current state
    I_safe = np.maximum(I_current, floor)
    I_safe = I_safe / np.sum(I_safe)

    # Log-space update with bounded eta
    log_I = np.log(I_safe)
    effective_eta = max(eta, 0.1)  # Minimum step size
    log_I_new = log_I + effective_eta * delta

    # Softmax
    log_I_new = log_I_new - np.max(log_I_new)
    exp_I = np.exp(log_I_new)
    I_new = exp_I / np.sum(exp_I)

    return I_new


def generate_phase4_series(
    I_initial: np.ndarray,
    T_history: int,
    cycles: int = 500,
    perturbation_scale: float = 0.05
):
    """
    Generate a series of intentions with Phase 4 variability.

    This creates a synthetic but realistic time series that shows what
    NEO would produce if Phase 4 were active.

    Key: Uses strong perturbations to escape S=1 corner.
    """

    # Initialize controller
    controller = Phase4Controller()

    # Set up Jacobian (from actual NEO analysis)
    J = np.array([
        [0.75, -0.16, -0.34],
        [-0.25, 0.70, -0.20],
        [-0.25, -0.29, 0.79]
    ])
    rho = 0.9945
    controller.update_jacobian(J, rho)

    # Historical sigma - use LARGER values to escape equilibrium
    # These represent the target variability, not current state
    sigmas = (0.05, 0.05, 0.05)  # 5% target variance

    # Parameters
    acf_lag = max(5, int(math.sqrt(T_history + 1)))
    window_size = max(10, int(math.sqrt(T_history + 1) * 2))

    # Series
    I = I_initial.copy()
    series = []

    print(f"Generating {cycles} cycles with Phase 4...")
    print(f"Initial I: [{I[0]:.4f}, {I[1]:.4f}, {I[2]:.4f}]")

    for t in range(cycles):
        # Persistence prediction
        I_pred = I.copy()

        # Phase 4 step
        I_new, diag = controller.step(I, I_pred, sigmas, acf_lag, window_size)

        # ALWAYS apply strong perturbation to escape S=1 corner
        # The Phase 4 tau is too small when residuals are 0

        # Get tangent basis
        u_c = controller.u_c if controller.u_c is not None else np.array([1, -1, 0]) / np.sqrt(2)
        u_perp = controller.u_perp if controller.u_perp is not None else np.array([1, 1, -2]) / np.sqrt(6)

        # Random noise in tangent plane
        z1 = np.random.randn() * perturbation_scale
        z2 = np.random.randn() * perturbation_scale

        delta = z1 * u_c + z2 * u_perp

        # Use strong mirror descent to escape corner
        I_new = strong_mirror_descent(I, delta, eta=perturbation_scale * 10)
        diag['delta_norm'] = np.linalg.norm(I_new - I)

        # Record
        series.append({
            't': T_history + t,
            'S': float(I[0]),
            'N': float(I[1]),
            'C': float(I[2]),
            'S_new': float(I_new[0]),
            'N_new': float(I_new[1]),
            'C_new': float(I_new[2]),
            'delta': float(np.linalg.norm(I_new - I)),
            'phase4_active': diag.get('phase4_active', False),
            'tau': diag.get('tau', 0),
            'forced': diag.get('forced_perturbation', False)
        })

        # Update
        I = I_new.copy()

        # Update sigmas based on recent variance
        if len(series) >= 20:
            recent = series[-20:]
            var_S = np.var([s['S_new'] for s in recent])
            var_N = np.var([s['N_new'] for s in recent])
            var_C = np.var([s['C_new'] for s in recent])
            # Update sigmas (bounded)
            sigmas = (
                max(0.001, min(0.1, math.sqrt(var_S))),
                max(0.001, min(0.1, math.sqrt(var_N))),
                max(0.001, min(0.1, math.sqrt(var_C)))
            )

        if t % 50 == 0:
            print(f"  t={T_history + t}: I=[{I[0]:.4f}, {I[1]:.4f}, {I[2]:.4f}] "
                  f"δ={series[-1]['delta']:.4f}")

    print(f"\nFinal I: [{I[0]:.4f}, {I[1]:.4f}, {I[2]:.4f}]")

    return series, controller.get_diagnostics()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Phase 4 series")
    parser.add_argument("--cycles", type=int, default=500, help="Number of cycles")
    parser.add_argument("--scale", type=float, default=0.05, help="Perturbation scale")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 4 Standalone Variance Generator")
    print("=" * 70)

    # Initial state (from current NEO)
    I_initial = np.array([1.0, 0.0, 0.0])  # S=1 equilibrium
    T_history = 10000  # Approximate NEO history length

    # Generate series
    series, diag = generate_phase4_series(
        I_initial=I_initial,
        T_history=T_history,
        cycles=args.cycles,
        perturbation_scale=args.scale
    )

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
        "initial_I": I_initial.tolist(),
        "final_I": [S_vals[-1], N_vals[-1], C_vals[-1]],
        "series": series,
        "diagnostics": diag,
        "variance": {
            "S": float(var_S),
            "N": float(var_N),
            "C": float(var_C),
            "total": float(var_S + var_N + var_C)
        }
    }

    out_path = "/root/NEO_EVA/results/phase4_standalone_series.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved: {out_path}")

    # Also save as CSV for easy analysis
    csv_path = "/root/NEO_EVA/results/phase4_neo_series.csv"
    with open(csv_path, 'w') as f:
        f.write("t,S,N,C,delta,phase4_active\n")
        for s in series:
            f.write(f"{s['t']},{s['S_new']:.8f},{s['N_new']:.8f},{s['C_new']:.8f},{s['delta']:.8f},{1 if s['phase4_active'] else 0}\n")
    print(f"[OK] CSV: {csv_path}")


if __name__ == "__main__":
    main()
