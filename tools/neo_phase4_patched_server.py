#!/usr/bin/env python3
"""
NEO Server with Phase 4 In-Memory Integration
=============================================

This script implements Phase 4 variability directly into the intention update,
using mirror descent to avoid sticky vertices.

Key features:
1. Mirror descent: I_{t+1} = softmax(log I_t + η_t * Δ_t)
2. Endogenous thermostat from IQR of residuals
3. Tangent-plane OU noise
4. Critical gate based on historical variance

Usage:
    python3 neo_phase4_patched_server.py [--cycles 500] [--output path.json]
"""

import sys
import os
import json
import numpy as np
import time
from datetime import datetime
from typing import Dict, Tuple

# Add paths
sys.path.insert(0, '/root/NEOSYNT')
sys.path.insert(0, '/root/NEO_EVA/tools')

# Import original server components
from engine.intention_dynamics import update_intention_damped
from engine.meta_intention import compute_meta_intention


class MirrorDescentPhase4:
    """
    Simplified Phase 4 with direct mirror descent implementation.

    This bypasses the Jacobian-dependent Phase4Controller and implements
    the core variability directly using:
    - Mirror descent for simplex-safe updates
    - Endogenous thermostat from residual IQR
    - Tangent-plane OU noise
    """

    def __init__(self, floor: float = 1e-6):
        self.floor = floor

        # History
        self.residuals = []
        self.I_history = []

        # OU state (2D tangent plane)
        self.ou_Z = np.array([0.0, 0.0])
        self.ou_theta = 0.1  # Will be computed endogenously

        # Tangent basis (fixed for 3-simplex)
        self.u_c = np.array([1, 1, 1]) / np.sqrt(3)  # Center direction
        self.u_1 = np.array([1, -1, 0]) / np.sqrt(2)  # Perpendicular 1
        self.u_2 = np.array([1, 1, -2]) / np.sqrt(6)  # Perpendicular 2

        self.t = 0
        self.gate_activations = 0
        self.escape_boost = 2.0  # Boost factor when stuck at corner

    def _compute_thermostat(self) -> float:
        """Compute τ from residual IQR."""
        if len(self.residuals) < 10:
            return 0.1  # Warmup value

        res = np.array(self.residuals[-50:])  # Recent window
        iqr = np.percentile(res, 75) - np.percentile(res, 25)

        T = len(self.I_history) + 1
        tau_base = iqr / np.sqrt(T)

        # Scale by historical variance
        if len(self.I_history) > 20:
            I_arr = np.array(self.I_history[-20:])
            sigma_hist = np.median(np.std(I_arr, axis=0))
            tau = tau_base * max(sigma_hist, 0.01)
        else:
            tau = tau_base * 0.05

        # Bounds
        tau = max(0.01, min(0.5, tau))
        return tau

    def _compute_gate(self) -> Tuple[bool, Dict]:
        """Check if gate should open."""
        if len(self.I_history) < 20:
            return True, {'reason': 'warmup', 'open': True}

        # Check variance in recent window
        I_arr = np.array(self.I_history[-20:])
        recent_var = np.var(I_arr, axis=0).sum()

        # Check variance in full history
        I_full = np.array(self.I_history)
        full_var = np.var(I_full, axis=0).sum()

        # Gate opens if recent variance is relatively high OR if at corner
        at_corner = I_arr[-1, 0] > 0.95 or I_arr[-1, 1] > 0.95 or I_arr[-1, 2] > 0.95

        # Open if variance is low (need perturbation) or at corner (stuck)
        open_gate = recent_var < 0.01 or at_corner

        return open_gate, {
            'recent_var': float(recent_var),
            'full_var': float(full_var),
            'at_corner': at_corner,
            'open': open_gate,
        }

    def _ou_step(self, tau: float) -> np.ndarray:
        """Take OU step in tangent plane."""
        # Compute theta from ACF if enough data
        if len(self.residuals) > 20:
            # First-order autocorrelation
            r = np.array(self.residuals[-20:])
            r_corr = np.corrcoef(r[:-1], r[1:])[0, 1]
            if not np.isnan(r_corr) and abs(r_corr) < 0.99:
                self.ou_theta = min(0.5, max(0.01, -1 / np.log(abs(r_corr) + 1e-10)))
            else:
                self.ou_theta = 0.1

        # OU dynamics: dZ = -θZ dt + σ√τ dW
        # Use larger sigma to actually move the system
        dt = 1.0
        sigma_base = 1.0  # Larger base sigma
        sigma = sigma_base * np.sqrt(max(tau, 0.05))

        drift = -self.ou_theta * self.ou_Z * dt
        diffusion = sigma * np.sqrt(dt) * np.random.randn(2)

        self.ou_Z = self.ou_Z + drift + diffusion

        # Bound OU state
        self.ou_Z = np.clip(self.ou_Z, -5, 5)

        return self.ou_Z

    def _ou_to_delta(self, Z: np.ndarray) -> np.ndarray:
        """Convert OU state to simplex perturbation."""
        # Project onto tangent plane perpendicular to center
        delta = Z[0] * self.u_1 + Z[1] * self.u_2

        # Ensure zero mean (stays on tangent plane)
        delta = delta - delta.mean()

        return delta

    def _mirror_descent(self, I: np.ndarray, delta: np.ndarray, eta: float) -> np.ndarray:
        """Apply mirror descent step: softmax(log I + η * Δ)."""
        # Floor to avoid log(0)
        I_safe = np.maximum(I, self.floor)
        I_safe = I_safe / I_safe.sum()

        # Mirror descent in log space
        log_I = np.log(I_safe)
        log_I_new = log_I + eta * delta

        # Softmax back to simplex
        exp_log = np.exp(log_I_new - np.max(log_I_new))
        I_new = exp_log / exp_log.sum()

        # Ensure floor
        I_new = np.maximum(I_new, self.floor)
        I_new = I_new / I_new.sum()

        return I_new

    def step(self, I_candidate: np.ndarray, I_predicted: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
        """
        Apply Phase 4 to candidate intention.

        Args:
            I_candidate: Candidate from update_intention_damped
            I_predicted: Optional prediction for residual

        Returns:
            I_final: Phase4-modified intention
            info: Diagnostics
        """
        self.t += 1

        # Record history
        self.I_history.append(I_candidate.copy())
        if len(self.I_history) > 500:
            self.I_history = self.I_history[-300:]

        # Compute residual
        if I_predicted is not None:
            residual = np.linalg.norm(I_candidate - I_predicted)
        else:
            residual = 0.01

        self.residuals.append(residual)
        if len(self.residuals) > 500:
            self.residuals = self.residuals[-300:]

        # Check gate
        gate_open, gate_info = self._compute_gate()

        if not gate_open:
            return I_candidate.copy(), {
                't': self.t,
                'phase4_active': False,
                'gate': gate_info,
                'delta': [0, 0, 0],
                'tau': 0,
            }

        self.gate_activations += 1

        # Compute thermostat
        tau = self._compute_thermostat()

        # OU step
        Z = self._ou_step(tau)

        # Convert to delta
        delta = self._ou_to_delta(Z)

        # Mirror descent with η = τ
        # Use larger eta when stuck at corner to escape
        at_corner = np.max(I_candidate) > 0.90
        if at_corner:
            eta = max(tau, 1.0) * self.escape_boost  # Large step to escape corner
        else:
            eta = max(tau, 0.2)

        I_final = self._mirror_descent(I_candidate, delta, eta)

        I_change = np.abs(I_final - I_candidate).sum()

        return I_final, {
            't': self.t,
            'phase4_active': True,
            'gate': gate_info,
            'delta': delta.tolist(),
            'delta_norm': float(np.linalg.norm(delta)),
            'tau': float(tau),
            'eta': float(eta),
            'ou_Z': Z.tolist(),
            'I_change': float(I_change),
        }


class NeoPhase4PatchedServer:
    """
    NEO server simulation with Phase 4 integrated.
    """

    def __init__(self, state_path: str = '/root/NEOSYNT/state/neo_state.json'):
        self.state_path = state_path
        self.state = self._load_state()

        # Phase 4 engine
        self.phase4 = MirrorDescentPhase4(floor=0.001)

        # History
        self.I_history = [np.array(self.state.get('I', [1.0, 0.0, 0.0]))]
        self.series = []

        self.t = self.state.get('t', 0)

    def _load_state(self) -> Dict:
        """Load state from file."""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path) as f:
                    return json.load(f)
            except:
                pass
        return {'I': [1.0, 0.0, 0.0], 't': 0}

    def step(self) -> Dict:
        """Execute one step with Phase 4."""
        self.t += 1
        I_prev = np.array(self.state['I'])

        # --- Simulate meta-intention computation ---
        if len(self.I_history) > 10:
            recent = np.array(self.I_history[-10:])
            world_stability = max(0.1, 1.0 - np.std(recent[:, 0]))
            intention_stability = max(0.1, 1.0 - np.var(recent).sum())
        else:
            world_stability = 0.8
            intention_stability = 0.8

        FE = 0.1 * np.random.rand()
        dz = 0.01 * np.random.randn()
        improvement = 0.5 + 0.1 * np.random.randn()

        meta_int = compute_meta_intention(
            world_stability, intention_stability, FE, dz, improvement
        )

        I_base = I_prev / (I_prev.sum() + 1e-10)

        # Original update (line 387 in server)
        I_candidate = update_intention_damped(I_prev, I_base, meta_int)

        # --- PHASE 4 INJECTION ---
        I_final, phase4_info = self.phase4.step(I_candidate, I_base)

        # --- Update state (line 419 in server) ---
        self.state['I'] = I_final.tolist()
        self.state['t'] = self.t
        self.I_history.append(I_final.copy())

        # Record series
        self.series.append({
            't': self.t,
            'S_prev': float(I_prev[0]),
            'N_prev': float(I_prev[1]),
            'C_prev': float(I_prev[2]),
            'S_new': float(I_final[0]),
            'N_new': float(I_final[1]),
            'C_new': float(I_final[2]),
            'phase4_active': phase4_info.get('phase4_active', False),
            'delta_norm': phase4_info.get('delta_norm', 0),
            'tau': phase4_info.get('tau', 0),
            'I_change': phase4_info.get('I_change', 0),
        })

        # Bound history
        if len(self.I_history) > 1000:
            self.I_history = self.I_history[-500:]

        return {
            't': self.t,
            'I_candidate': I_candidate.tolist(),
            'I_final': I_final.tolist(),
            'I_change': phase4_info.get('I_change', 0),
            'phase4': phase4_info,
        }

    def run(self, cycles: int = 500, verbose: bool = True) -> Dict:
        """Run for specified cycles."""
        print("=" * 70)
        print("NEO Phase 4 Patched Server - Mirror Descent")
        print("=" * 70)
        print(f"Initial I: {self.state['I']}")
        print(f"Running {cycles} cycles...")
        print()

        start_time = time.time()
        phase4_active_count = 0
        total_change = 0

        for i in range(cycles):
            result = self.step()

            if result['phase4'].get('phase4_active', False):
                phase4_active_count += 1
            total_change += result.get('I_change', 0)

            if verbose and (i + 1) % 100 == 0:
                I = self.state['I']
                print(f"  t={self.t:4d}: I=[{I[0]:.4f}, {I[1]:.4f}, {I[2]:.4f}] "
                      f"Δ={result['I_change']:.6f} "
                      f"P4={result['phase4'].get('phase4_active', False)}")

        elapsed = time.time() - start_time

        # Compute variance
        I_arr = np.array([[s['S_new'], s['N_new'], s['C_new']] for s in self.series])
        variance = {
            'S': float(np.var(I_arr[:, 0])),
            'N': float(np.var(I_arr[:, 1])),
            'C': float(np.var(I_arr[:, 2])),
            'total': float(np.var(I_arr[:, 0]) + np.var(I_arr[:, 1]) + np.var(I_arr[:, 2])),
        }

        print()
        print("=" * 70)
        print("Results:")
        print(f"  Final I: {self.state['I']}")
        print(f"  Cycles: {cycles}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Phase 4 active: {phase4_active_count}/{cycles} ({100*phase4_active_count/cycles:.1f}%)")
        print(f"  Mean Δ: {total_change/cycles:.6f}")
        print(f"  Variance: S={variance['S']:.6e}, N={variance['N']:.6e}, C={variance['C']:.6e}")
        print(f"  Total variance: {variance['total']:.6e}")
        print("=" * 70)

        return {
            'cycles': cycles,
            'elapsed': elapsed,
            'initial_I': self.I_history[0].tolist() if self.I_history else [1, 0, 0],
            'final_I': self.state['I'],
            'variance': variance,
            'phase4_active_rate': phase4_active_count / cycles,
            'mean_delta': total_change / cycles,
            'series': self.series,
        }

    def save_series(self, path: str):
        """Save series for IWVI analysis."""
        I_arr = np.array([[s['S_new'], s['N_new'], s['C_new']] for s in self.series])
        data = {
            'timestamp': datetime.now().isoformat(),
            'cycles': len(self.series),
            'initial_I': self.I_history[0].tolist() if self.I_history else [1, 0, 0],
            'final_I': self.state['I'],
            'series': self.series,
            'variance': {
                'S': float(np.var(I_arr[:, 0])),
                'N': float(np.var(I_arr[:, 1])),
                'C': float(np.var(I_arr[:, 2])),
                'total': float(np.var(I_arr).sum() * 3),
            },
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[OK] Saved series: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NEO Phase 4 Patched Server")
    parser.add_argument("--cycles", type=int, default=500, help="Number of cycles")
    parser.add_argument("--output", type=str, default="/root/NEO_EVA/results/phase4_patched_neo_series.json",
                        help="Output path")
    args = parser.parse_args()

    os.makedirs("/root/NEO_EVA/results", exist_ok=True)

    server = NeoPhase4PatchedServer()
    result = server.run(cycles=args.cycles)
    server.save_series(args.output)

    # Sanity check
    print("\n[Sanity Check] Delta test:")
    deltas = [s.get('I_change', 0) for s in server.series if s.get('phase4_active', False)]
    if deltas:
        mean_delta = np.mean(deltas)
        print(f"  Mean ||ΔI||₁ when P4 active: {mean_delta:.6f}")
        if 1e-4 < mean_delta < 1e-1:
            print(f"  ✓ Delta in expected range [1e-4, 1e-1]")
        else:
            print(f"  ⚠ Delta outside expected range")
    else:
        print("  ⚠ No Phase 4 active steps")


if __name__ == "__main__":
    main()
