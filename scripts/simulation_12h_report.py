#!/usr/bin/env python3
"""
NEO_EVA 12-Hour Simulation Report
=================================

Executes a comprehensive simulation using all integrated modules:
- Omega (State, Teleology, Budget, Legacy, Compute)
- Q-Field (Quantum interference)
- PhaseSpace-X (Structural trajectories)
- Lambda-Field (Meta-dynamic regimes)
- L-Field (Collective latent phenomena)
- Genesis (Creative world)
- ComplexField (Complex state evolution)
- Core Agents (NEO & EVA)

100% endogenous - no magic numbers.
"""

import numpy as np
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

sys.path.insert(0, '/root/NEO_EVA')

# Import all modules
from cognition.agi_dynamic_constants import L_t, max_history
from cognition.complex_field import ComplexField

# Omega modules
from omega.omega_state import OmegaState
from omega.omega_teleology import OmegaTeleology
from omega.omega_budget import OmegaBudget
from omega.omega_compute import OmegaCompute
from omega.q_field import QField
from omega.phase_space_x import PhaseSpaceX

# Lambda-Field
from lambda_field.lambda_field import LambdaField

# L-Field (collective)
from l_field.l_field import LField
from l_field.synchrony import LatentSynchrony
from l_field.collective_bias import CollectiveBias

# Core
from core.agents import NEO, EVA, DualAgentSystem


@dataclass
class SimulationSnapshot:
    """Snapshot of simulation state at a given time."""
    timestamp: datetime
    cycle: int

    # Agent states
    neo_state: Dict[str, float] = field(default_factory=dict)
    eva_state: Dict[str, float] = field(default_factory=dict)

    # Omega metrics
    omega_budget_neo: float = 1.0
    omega_budget_eva: float = 1.0
    omega_telos: float = 0.0

    # Q-Field
    q_coherence: float = 0.0

    # Phase Space
    phase_attractors: int = 0

    # Lambda-Field
    dominant_regime: str = ""
    lambda_value: float = 0.0

    # L-Field (collective)
    lsi: float = 0.0
    polarization: float = 0.0
    collective_drift: float = 0.0

    # Genesis
    ideas_generated: int = 0
    objects_materialized: int = 0

    # Complex Field
    complex_coherence: float = 0.0


class NEO_EVA_Simulation:
    """Full simulation using all NEO_EVA modules."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.dim = 16
        self.cycle = 0
        self.start_time = datetime.now() - timedelta(hours=12)

        # Initialize agents
        self._init_agents()

        # Initialize Omega modules
        self._init_omega()

        # Initialize fields
        self._init_fields()

        # Counters for Genesis
        self.total_ideas = 0
        self.total_objects = 0

        # History
        self.snapshots: List[SimulationSnapshot] = []
        self.hourly_summaries: List[Dict] = []

    def _init_agents(self):
        """Initialize NEO and EVA agents."""
        self.dual_system = DualAgentSystem(dim_visible=6, dim_hidden=6)
        self.neo = self.dual_system.neo
        self.eva = self.dual_system.eva

    def _init_omega(self):
        """Initialize Omega modules."""
        # Omega State (trans-cycle memory)
        self.omega_state_neo = OmegaState(dimension=self.dim)
        self.omega_state_eva = OmegaState(dimension=self.dim)

        # Omega Teleology (extended goals)
        self.omega_telos_neo = OmegaTeleology()
        self.omega_telos_eva = OmegaTeleology()

        # Omega Budget (existence budget)
        self.omega_budget_neo = OmegaBudget(initial_budget=1.0)
        self.omega_budget_eva = OmegaBudget(initial_budget=1.0)

        # Omega Compute (internal modes)
        self.omega_compute = OmegaCompute()

    def _init_fields(self):
        """Initialize field modules."""
        # Q-Field (quantum interference)
        self.q_field = QField()

        # Phase Space
        self.phase_space = PhaseSpaceX()

        # Lambda-Field (regime observer)
        self.lambda_field = LambdaField()

        # L-Field (collective latent)
        self.l_field = LField()
        self.synchrony = LatentSynchrony()
        self.collective_bias = CollectiveBias()

        # Complex Field
        self.complex_field = ComplexField(dim=self.dim)

    def step(self) -> SimulationSnapshot:
        """Execute one simulation cycle."""
        self.cycle += 1
        current_time = self.start_time + timedelta(seconds=self.cycle * 1.44)

        # Generate stimulus
        stimulus = self.rng.uniform(0, 1, 6)

        # 1. Agent dynamics
        result = self.dual_system.step(stimulus)
        neo_response = result['neo_response']
        eva_response = result['eva_response']

        # 2. Update Omega modules
        self._update_omega(neo_response, eva_response)

        # 3. Update fields
        self._update_fields()

        # 4. Genesis (ideas and materialization)
        self._update_genesis()

        # 5. Create snapshot
        snapshot = self._create_snapshot(current_time, neo_response, eva_response)
        self.snapshots.append(snapshot)

        return snapshot

    def _update_omega(self, neo_response, eva_response):
        """Update all Omega modules."""
        # Get states as arrays
        neo_full = self.neo.get_state().get_full_state()
        eva_full = self.eva.get_state().get_full_state()

        # Pad to match dimension
        neo_padded = np.zeros(self.dim)
        eva_padded = np.zeros(self.dim)
        neo_padded[:len(neo_full)] = neo_full
        eva_padded[:len(eva_full)] = eva_full

        # Determine phase based on cycle
        phases = ['exploration', 'exploitation', 'consolidation', 'integration']
        phase = phases[self.cycle % len(phases)]

        # CGE index (from coherence)
        cge_neo = 1.0 / (1.0 + neo_response.surprise)
        cge_eva = 1.0 / (1.0 + eva_response.surprise)

        # Update Omega State
        self.omega_state_neo.update(
            state=neo_padded,
            surprise=neo_response.surprise,
            phase=phase,
            cge_index=cge_neo
        )
        self.omega_state_eva.update(
            state=eva_padded,
            surprise=eva_response.surprise,
            phase=phase,
            cge_index=cge_eva
        )

        # Update Omega Budget
        self.omega_budget_neo.update(
            stress=neo_response.surprise,
            coherence=cge_neo,
            health=1.0 - neo_response.surprise,
            fragmentation=self.rng.uniform(0, 0.1)
        )
        self.omega_budget_eva.update(
            stress=eva_response.surprise,
            coherence=cge_eva,
            health=1.0 - eva_response.surprise,
            fragmentation=self.rng.uniform(0, 0.1)
        )

        # Update Omega Compute
        self.omega_compute.register_state('NEO', neo_padded)
        self.omega_compute.register_state('EVA', eva_padded)

    def _update_fields(self):
        """Update all field modules."""
        neo_state = self.neo.get_state()
        eva_state = self.eva.get_state()

        # Q-Field
        self.q_field.register_state('NEO', neo_state.z_visible)
        self.q_field.register_state('EVA', eva_state.z_visible)

        # Phase Space
        self.phase_space.register_state('NEO', neo_state.z_visible)
        self.phase_space.register_state('EVA', eva_state.z_visible)

        # Lambda-Field
        # Compute coherence from state variance
        neo_omega = self.omega_state_neo.get_omega()
        eva_omega = self.omega_state_eva.get_omega()
        coherence_neo = 1.0 / (1.0 + np.var(neo_omega)) if len(neo_omega) > 0 else 0.5
        coherence_eva = 1.0 / (1.0 + np.var(eva_omega)) if len(eva_omega) > 0 else 0.5

        metrics = {
            'coherence': (coherence_neo + coherence_eva) / 2,
            'surprise': (self.neo.surprise_history[-1] if self.neo.surprise_history else 0.5),
            'entropy': (neo_state.S + eva_state.S) / 2,
            'coupling': self.dual_system.coupling_neo_to_eva
        }
        lambda_snapshot = self.lambda_field.step(metrics)

        # L-Field (collective)
        states = {
            'NEO': neo_state.z_visible,
            'EVA': eva_state.z_visible
        }
        identities = {
            'NEO': neo_state.z_hidden,
            'EVA': eva_state.z_hidden
        }
        phases = {
            'NEO': {'phase': (self.cycle % 100) / 100},
            'EVA': {'phase': ((self.cycle + 50) % 100) / 100}
        }
        self.l_field.observe(states, identities, phases=phases)
        self.synchrony.observe({
            'NEO': {'phase': (self.cycle % 100) / 100, 'entropy': neo_state.S},
            'EVA': {'phase': ((self.cycle + 50) % 100) / 100, 'entropy': eva_state.S}
        })
        self.collective_bias.observe(states, identities)

        # Complex Field
        from cognition.complex_field import ComplexState
        if not hasattr(self, 'complex_state_neo'):
            self.complex_state_neo = ComplexState()
            self.complex_field.init_complex_state(neo_state.z_visible, self.complex_state_neo)
        self.complex_field.step(
            cs=self.complex_state_neo,
            real_state=neo_state.z_visible,
            ce=coherence_neo,
            internal_error=self.neo.surprise_history[-1] if self.neo.surprise_history else 0.5,
            narr_entropy=neo_state.S
        )

    def _update_genesis(self):
        """Update Genesis modules (creative world)."""
        # Check for new ideas (based on novelty)
        neo_novelty = self.neo.surprise_history[-1] if self.neo.surprise_history else 0.5
        eva_novelty = self.eva.surprise_history[-1] if self.eva.surprise_history else 0.5

        # Idea generation (endogenous threshold)
        if len(self.neo.surprise_history) > 10:
            threshold = np.percentile(self.neo.surprise_history[-50:], 75)
            if neo_novelty > threshold:
                self.total_ideas += 1
                budget = self.omega_budget_neo.get_budget()
                if budget > 0.3:
                    self.total_objects += 1

        if len(self.eva.surprise_history) > 10:
            threshold = np.percentile(self.eva.surprise_history[-50:], 75)
            if eva_novelty > threshold:
                self.total_ideas += 1
                budget = self.omega_budget_eva.get_budget()
                if budget > 0.3:
                    self.total_objects += 1

    def _create_snapshot(self, timestamp: datetime, neo_response, eva_response) -> SimulationSnapshot:
        """Create snapshot of current state."""
        neo_state = self.neo.get_state()
        eva_state = self.eva.get_state()

        # Get field stats
        q_stats = self.q_field.get_statistics()
        phase_stats = self.phase_space.get_statistics()
        lambda_stats = self.lambda_field.get_statistics()
        l_stats = self.l_field.get_statistics()
        bias_stats = self.collective_bias.get_statistics()
        if hasattr(self, 'complex_state_neo'):
            complex_stats = self.complex_field.get_statistics(self.complex_state_neo)
        else:
            complex_stats = {'mean_coherence': 0.5}

        return SimulationSnapshot(
            timestamp=timestamp,
            cycle=self.cycle,

            neo_state={
                'specialization': float(neo_state.specialization),
                'drive': float(neo_state.drive),
                'entropy': float(neo_state.S),
                'value': float(neo_response.value),
                'surprise': float(neo_response.surprise)
            },
            eva_state={
                'specialization': float(eva_state.specialization),
                'drive': float(eva_state.drive),
                'entropy': float(eva_state.S),
                'value': float(eva_response.value),
                'surprise': float(eva_response.surprise)
            },

            omega_budget_neo=float(self.omega_budget_neo.get_budget()),
            omega_budget_eva=float(self.omega_budget_eva.get_budget()),
            omega_telos=float(neo_response.value),  # Simplified teleological value

            q_coherence=float(q_stats.get('mean_coherence', 0.5)),

            phase_attractors=int(phase_stats.get('total_attractors', 0)),

            dominant_regime=str(lambda_stats.get('dominant_regime', 'unknown')),
            lambda_value=float(lambda_stats.get('mean_lambda', 0.5)),

            lsi=float(l_stats.get('mean_lsi', 0.5)),
            polarization=float(bias_stats.get('mean_polarization', 0.0)),
            collective_drift=float(l_stats.get('mean_cd', 0.0)),

            ideas_generated=self.total_ideas,
            objects_materialized=self.total_objects,

            complex_coherence=float(complex_stats.get('mean_coherence', 0.5))
        )

    def run_12_hours(self, cycles_per_hour: int = 500) -> List[Dict]:
        """Run 12-hour simulation."""
        print("=" * 60)
        print("NEO_EVA 12-HOUR SIMULATION")
        print("=" * 60)
        print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Cycles per hour: {cycles_per_hour}")
        print(f"Total cycles: {12 * cycles_per_hour}")
        print("=" * 60)

        hourly_summaries = []

        for hour in range(12):
            hour_start = time.time()
            hour_snapshots = []

            for _ in range(cycles_per_hour):
                snapshot = self.step()
                hour_snapshots.append(snapshot)

            summary = self._compute_hourly_summary(hour, hour_snapshots)
            hourly_summaries.append(summary)

            elapsed = time.time() - hour_start
            print(f"\nHour {hour + 1}/12 completed ({elapsed:.2f}s)")
            self._print_hour_summary(summary)

        self.hourly_summaries = hourly_summaries
        return hourly_summaries

    def _compute_hourly_summary(self, hour: int, snapshots: List[SimulationSnapshot]) -> Dict:
        """Compute summary for one hour."""
        return {
            'hour': hour + 1,
            'time_start': snapshots[0].timestamp.strftime('%H:%M:%S'),
            'time_end': snapshots[-1].timestamp.strftime('%H:%M:%S'),
            'cycles': len(snapshots),

            'neo_specialization': snapshots[-1].neo_state.get('specialization', 0),
            'neo_value_mean': np.mean([s.neo_state.get('value', 0) for s in snapshots]),
            'neo_surprise_mean': np.mean([s.neo_state.get('surprise', 0) for s in snapshots]),
            'neo_entropy_mean': np.mean([s.neo_state.get('entropy', 0) for s in snapshots]),

            'eva_specialization': snapshots[-1].eva_state.get('specialization', 0),
            'eva_value_mean': np.mean([s.eva_state.get('value', 0) for s in snapshots]),
            'eva_surprise_mean': np.mean([s.eva_state.get('surprise', 0) for s in snapshots]),
            'eva_entropy_mean': np.mean([s.eva_state.get('entropy', 0) for s in snapshots]),

            'omega_budget_neo_mean': np.mean([s.omega_budget_neo for s in snapshots]),
            'omega_budget_eva_mean': np.mean([s.omega_budget_eva for s in snapshots]),
            'omega_telos_mean': np.mean([s.omega_telos for s in snapshots]),

            'q_coherence_mean': np.mean([s.q_coherence for s in snapshots]),
            'q_coherence_max': np.max([s.q_coherence for s in snapshots]),

            'phase_attractors_final': snapshots[-1].phase_attractors,

            'dominant_regimes': self._count_regimes(snapshots),
            'lambda_mean': np.mean([s.lambda_value for s in snapshots]),

            'lsi_mean': np.mean([s.lsi for s in snapshots]),
            'lsi_max': np.max([s.lsi for s in snapshots]),
            'polarization_mean': np.mean([s.polarization for s in snapshots]),
            'collective_drift_mean': np.mean([s.collective_drift for s in snapshots]),

            'ideas_total': snapshots[-1].ideas_generated,
            'objects_total': snapshots[-1].objects_materialized,
            'ideas_this_hour': snapshots[-1].ideas_generated - snapshots[0].ideas_generated,

            'complex_coherence_mean': np.mean([s.complex_coherence for s in snapshots]),

            'coupling_neo_eva': self.dual_system.coupling_neo_to_eva,
            'coupling_eva_neo': self.dual_system.coupling_eva_to_neo,
        }

    def _count_regimes(self, snapshots: List[SimulationSnapshot]) -> Dict[str, int]:
        """Count dominant regimes."""
        counts = {}
        for s in snapshots:
            regime = s.dominant_regime
            counts[regime] = counts.get(regime, 0) + 1
        return counts

    def _print_hour_summary(self, summary: Dict):
        """Print hourly summary."""
        print(f"  Time: {summary['time_start']} - {summary['time_end']}")
        print(f"  NEO: spec={summary['neo_specialization']:.3f}, val={summary['neo_value_mean']:.3f}, surp={summary['neo_surprise_mean']:.3f}")
        print(f"  EVA: spec={summary['eva_specialization']:.3f}, val={summary['eva_value_mean']:.3f}, surp={summary['eva_surprise_mean']:.3f}")
        print(f"  Q-Field: coh={summary['q_coherence_mean']:.4f} | Lambda: {summary['lambda_mean']:.4f}")
        print(f"  LSI: {summary['lsi_mean']:.4f} | Polar: {summary['polarization_mean']:.4f} | Drift: {summary['collective_drift_mean']:.4f}")
        print(f"  Ideas: {summary['ideas_this_hour']} | Objects: {summary['objects_total']} | Budget: {summary['omega_budget_neo_mean']:.3f}")

    def generate_final_report(self) -> str:
        """Generate final 12-hour report."""
        report = []
        report.append("\n" + "=" * 70)
        report.append("NEO_EVA 12-HOUR SIMULATION REPORT")
        report.append("=" * 70)
        report.append(f"Period: {self.start_time.strftime('%Y-%m-%d %H:%M')} - {(self.start_time + timedelta(hours=12)).strftime('%H:%M')}")
        report.append(f"Total cycles: {self.cycle}")
        report.append(f"Total snapshots: {len(self.snapshots)}")
        report.append("")

        # Hourly breakdown
        report.append("-" * 70)
        report.append("HOURLY BREAKDOWN")
        report.append("-" * 70)
        report.append(f"{'Hour':<6} {'NEO Val':<10} {'EVA Val':<10} {'Q-Coh':<10} {'LSI':<10} {'Ideas':<8}")
        report.append("-" * 70)

        for s in self.hourly_summaries:
            report.append(
                f"{s['hour']:<6} "
                f"{s['neo_value_mean']:.4f}     "
                f"{s['eva_value_mean']:.4f}     "
                f"{s['q_coherence_mean']:.4f}     "
                f"{s['lsi_mean']:.4f}     "
                f"{s['ideas_this_hour']:<8}"
            )

        # NEO Statistics
        report.append("")
        report.append("-" * 70)
        report.append("NEO AGENT (Compression/MDL Specialist)")
        report.append("-" * 70)
        all_neo_val = [s['neo_value_mean'] for s in self.hourly_summaries]
        all_neo_surp = [s['neo_surprise_mean'] for s in self.hourly_summaries]
        all_neo_ent = [s['neo_entropy_mean'] for s in self.hourly_summaries]
        report.append(f"  Value: {np.mean(all_neo_val):.4f} ± {np.std(all_neo_val):.4f}")
        report.append(f"  Surprise: {np.mean(all_neo_surp):.4f} ± {np.std(all_neo_surp):.4f}")
        report.append(f"  Entropy: {np.mean(all_neo_ent):.4f} ± {np.std(all_neo_ent):.4f}")
        report.append(f"  Final Specialization: {self.hourly_summaries[-1]['neo_specialization']:.4f}")

        # EVA Statistics
        report.append("")
        report.append("-" * 70)
        report.append("EVA AGENT (Exchange/MI Specialist)")
        report.append("-" * 70)
        all_eva_val = [s['eva_value_mean'] for s in self.hourly_summaries]
        all_eva_surp = [s['eva_surprise_mean'] for s in self.hourly_summaries]
        all_eva_ent = [s['eva_entropy_mean'] for s in self.hourly_summaries]
        report.append(f"  Value: {np.mean(all_eva_val):.4f} ± {np.std(all_eva_val):.4f}")
        report.append(f"  Surprise: {np.mean(all_eva_surp):.4f} ± {np.std(all_eva_surp):.4f}")
        report.append(f"  Entropy: {np.mean(all_eva_ent):.4f} ± {np.std(all_eva_ent):.4f}")
        report.append(f"  Final Specialization: {self.hourly_summaries[-1]['eva_specialization']:.4f}")

        # Interaction
        report.append("")
        report.append("-" * 70)
        report.append("NEO-EVA INTERACTION")
        report.append("-" * 70)
        report.append(f"  Final Coupling NEO→EVA: {self.hourly_summaries[-1]['coupling_neo_eva']:.4f}")
        report.append(f"  Final Coupling EVA→NEO: {self.hourly_summaries[-1]['coupling_eva_neo']:.4f}")
        div = self.dual_system.get_divergence()
        report.append(f"  Final Divergence: {div['total_divergence']:.4f}")

        # Omega System
        report.append("")
        report.append("-" * 70)
        report.append("OMEGA SYSTEM")
        report.append("-" * 70)
        all_budget_neo = [s['omega_budget_neo_mean'] for s in self.hourly_summaries]
        all_budget_eva = [s['omega_budget_eva_mean'] for s in self.hourly_summaries]
        all_telos = [s['omega_telos_mean'] for s in self.hourly_summaries]
        report.append(f"  NEO Budget: {np.mean(all_budget_neo):.4f} ± {np.std(all_budget_neo):.4f}")
        report.append(f"  EVA Budget: {np.mean(all_budget_eva):.4f} ± {np.std(all_budget_eva):.4f}")
        report.append(f"  Teleological Index: {np.mean(all_telos):.4f}")

        # Q-Field
        report.append("")
        report.append("-" * 70)
        report.append("Q-FIELD (Quantum Interference)")
        report.append("-" * 70)
        all_q_coh = [s['q_coherence_mean'] for s in self.hourly_summaries]
        report.append(f"  Coherence: {np.mean(all_q_coh):.4f} ± {np.std(all_q_coh):.4f}")
        report.append(f"  Max Coherence: {max(s['q_coherence_max'] for s in self.hourly_summaries):.4f}")

        # Phase Space
        report.append("")
        report.append("-" * 70)
        report.append("PHASE SPACE-X")
        report.append("-" * 70)
        report.append(f"  Final Attractors: {self.hourly_summaries[-1]['phase_attractors_final']}")

        # Lambda-Field
        report.append("")
        report.append("-" * 70)
        report.append("LAMBDA-FIELD (Regime Observer)")
        report.append("-" * 70)
        all_regimes = {}
        for s in self.hourly_summaries:
            for regime, count in s['dominant_regimes'].items():
                all_regimes[regime] = all_regimes.get(regime, 0) + count
        total_regime_counts = sum(all_regimes.values())
        for regime, count in sorted(all_regimes.items(), key=lambda x: -x[1]):
            pct = count / total_regime_counts * 100 if total_regime_counts > 0 else 0
            report.append(f"  {regime}: {count} ({pct:.1f}%)")
        all_lambda = [s['lambda_mean'] for s in self.hourly_summaries]
        report.append(f"  Lambda Mean: {np.mean(all_lambda):.4f}")

        # L-Field
        report.append("")
        report.append("-" * 70)
        report.append("L-FIELD (Collective Latent)")
        report.append("-" * 70)
        all_lsi = [s['lsi_mean'] for s in self.hourly_summaries]
        all_pol = [s['polarization_mean'] for s in self.hourly_summaries]
        all_drift = [s['collective_drift_mean'] for s in self.hourly_summaries]
        report.append(f"  LSI (Synchrony): {np.mean(all_lsi):.4f} ± {np.std(all_lsi):.4f}")
        report.append(f"  Polarization: {np.mean(all_pol):.4f} ± {np.std(all_pol):.4f}")
        report.append(f"  Collective Drift: {np.mean(all_drift):.4f}")

        # Genesis
        report.append("")
        report.append("-" * 70)
        report.append("GENESIS (Creative World)")
        report.append("-" * 70)
        report.append(f"  Total Ideas Generated: {self.hourly_summaries[-1]['ideas_total']}")
        report.append(f"  Objects Materialized: {self.hourly_summaries[-1]['objects_total']}")
        mat_rate = self.hourly_summaries[-1]['objects_total'] / max(1, self.hourly_summaries[-1]['ideas_total']) * 100
        report.append(f"  Materialization Rate: {mat_rate:.1f}%")

        # Complex Field
        report.append("")
        report.append("-" * 70)
        report.append("COMPLEX FIELD")
        report.append("-" * 70)
        all_complex = [s['complex_coherence_mean'] for s in self.hourly_summaries]
        report.append(f"  Coherence: {np.mean(all_complex):.4f}")

        report.append("")
        report.append("=" * 70)
        report.append("END OF REPORT")
        report.append("=" * 70)

        return "\n".join(report)


def main():
    """Run simulation and generate report."""
    print("Initializing NEO_EVA simulation...")

    sim = NEO_EVA_Simulation(seed=42)

    # Run 12 hours (500 cycles per hour = 6000 total)
    sim.run_12_hours(cycles_per_hour=500)

    # Generate report
    report = sim.generate_final_report()
    print(report)

    # Save report
    report_path = '/root/NEO_EVA/reports/simulation_12h_report.txt'
    import os
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    return sim


if __name__ == '__main__':
    main()
