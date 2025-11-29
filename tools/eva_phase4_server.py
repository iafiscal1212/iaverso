#!/usr/bin/env python3
"""
EVA Phase 4 Server - World B
============================

EVA (Evaluative Variant of Autonomous) is the second world in the dual system.
It starts from a uniform prior [1/3, 1/3, 1/3] and evolves independently.

Uses the same mirror descent Phase 4 as NEO.
"""

import sys
import os
import json
import numpy as np
import time
from datetime import datetime
from typing import Dict

sys.path.insert(0, '/root/NEO_EVA/tools')

from neo_phase4_patched_server import MirrorDescentPhase4


class EvaPhase4Server:
    """
    EVA server with Phase 4 - starts from uniform prior.
    """

    def __init__(self, initial_I: np.ndarray = None):
        if initial_I is None:
            initial_I = np.array([1/3, 1/3, 1/3])  # Uniform prior

        self.state = {'I': initial_I.tolist(), 't': 0}
        self.phase4 = MirrorDescentPhase4(floor=1e-6)
        self.I_history = [initial_I.copy()]
        self.series = []
        self.t = 0

    def step(self) -> Dict:
        """Execute one step."""
        self.t += 1
        I_prev = np.array(self.state['I'])

        # EVA has its own dynamics - simulated as random walk + drift
        # toward higher novelty (N dimension)
        drift = np.array([-0.001, 0.002, -0.001])  # Slight N-bias
        noise = np.random.randn(3) * 0.01

        I_candidate = I_prev + drift + noise
        I_candidate = np.maximum(I_candidate, 1e-6)
        I_candidate = I_candidate / I_candidate.sum()

        # Apply Phase 4
        I_final, phase4_info = self.phase4.step(I_candidate, I_prev)

        # Update state
        self.state['I'] = I_final.tolist()
        self.state['t'] = self.t
        self.I_history.append(I_final.copy())

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

        if len(self.I_history) > 1000:
            self.I_history = self.I_history[-500:]

        return {
            't': self.t,
            'I_final': I_final.tolist(),
            'phase4': phase4_info,
        }

    def run(self, cycles: int = 2000, verbose: bool = True) -> Dict:
        """Run for specified cycles."""
        print("=" * 70)
        print("EVA Phase 4 Server - World B")
        print("=" * 70)
        print(f"Initial I: {self.state['I']}")
        print(f"Running {cycles} cycles...")
        print()

        start_time = time.time()
        phase4_active_count = 0

        for i in range(cycles):
            result = self.step()

            if result['phase4'].get('phase4_active', False):
                phase4_active_count += 1

            if verbose and (i + 1) % 200 == 0:
                I = self.state['I']
                print(f"  t={self.t:4d}: I=[{I[0]:.4f}, {I[1]:.4f}, {I[2]:.4f}]")

        elapsed = time.time() - start_time

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
        print(f"  Phase 4 active: {phase4_active_count}/{cycles}")
        print(f"  Variance: S={variance['S']:.6e}, N={variance['N']:.6e}, C={variance['C']:.6e}")
        print(f"  Total variance: {variance['total']:.6e}")
        print("=" * 70)

        return {
            'cycles': cycles,
            'initial_I': [1/3, 1/3, 1/3],
            'final_I': self.state['I'],
            'variance': variance,
        }

    def save_series(self, path: str):
        """Save series for IWVI."""
        I_arr = np.array([[s['S_new'], s['N_new'], s['C_new']] for s in self.series])
        data = {
            'timestamp': datetime.now().isoformat(),
            'cycles': len(self.series),
            'initial_I': [1/3, 1/3, 1/3],
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
        print(f"[OK] Saved: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="EVA Phase 4 Server")
    parser.add_argument("--cycles", type=int, default=2000, help="Number of cycles")
    parser.add_argument("--output", type=str, default="/root/NEO_EVA/results/phase5_eva_2000_series.json")
    args = parser.parse_args()

    os.makedirs("/root/NEO_EVA/results", exist_ok=True)

    server = EvaPhase4Server()
    server.run(cycles=args.cycles)
    server.save_series(args.output)


if __name__ == "__main__":
    main()
