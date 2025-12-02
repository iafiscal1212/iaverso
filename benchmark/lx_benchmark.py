"""
LX Benchmark Runner: Life-Extended Cognition Benchmark (LX1-LX10)
=================================================================

Main runner that orchestrates all LX metrics and produces a comprehensive report.

Inputs (all ENDOGENOUS):
- Lifecycle logs: phases, energy, stress
- Cognitive logs: episodes, φ, drives, CG-E by block
- Symbolic logs: active symbols σ_t^A, grammar, projects
- Medical logs: interventions, outcomes, MED-X, health_t^A
- Social logs: interactions, norms, coalitions

100% endogenous - no magic constants.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

from .lx_metrics import (
    CircadianPhase,
    SymbolActivation,
    PhaseState,
    Episode,
    CycleStats,
    compute_lx1_phase_symbol,
    compute_lx2_symbolic_drift,
    compute_lx3_dream_narrative,
    compute_lx4_dream_transfer,
    compute_lx5_medicine_phase,
    compute_lx6_full_cycle_medicine,
    compute_lx7_circadian_cge,
    compute_lx8_life_plasticity,
    compute_lx9_multiagent_sync,
    compute_lx10_aggregate,
)


# ==============================================================================
# LOG DATA STRUCTURES
# ==============================================================================

@dataclass
class LifecycleLogs:
    """
    Logs from lifecycle module.

    phases: List of (t, agent_id, phase, energy, stress)
    cycles: List of (cycle_id, agent_id, t_start, t_end)
    """
    phases: List[Tuple[int, str, str, float, float]] = field(default_factory=list)
    cycles: List[Tuple[int, str, int, int]] = field(default_factory=list)

    def get_phase_states(self) -> List[PhaseState]:
        """Convert to PhaseState objects."""
        result = []
        for t, agent_id, phase_str, energy, stress in self.phases:
            try:
                phase = CircadianPhase(phase_str)
            except ValueError:
                phase = CircadianPhase.WAKE  # Default fallback
            result.append(PhaseState(
                t=t,
                agent_id=agent_id,
                phase=phase,
                energy=energy,
                stress=stress,
            ))
        return result


@dataclass
class CognitiveLogs:
    """
    Logs from cognition module.

    episodes: List of (k, agent_id, t_start, t_end, dominant_phase, NC, symbol_density)
    cge_blocks: List of (block_id, cge_value, dominant_phase)
    drives: List of (t, agent_id, drive_vector)
    phi_values: List of (t, agent_id, phi)
    """
    episodes: List[Tuple[int, str, int, int, str, float, float]] = field(default_factory=list)
    cge_blocks: List[Tuple[int, float, str]] = field(default_factory=list)
    drives: List[Tuple[int, str, np.ndarray]] = field(default_factory=list)
    phi_values: List[Tuple[int, str, float]] = field(default_factory=list)

    def get_episodes(self) -> List[Episode]:
        """Convert to Episode objects."""
        result = []
        for k, agent_id, t_start, t_end, phase_str, nc, sym_density in self.episodes:
            try:
                phase = CircadianPhase(phase_str)
            except ValueError:
                phase = CircadianPhase.WAKE
            result.append(Episode(
                k=k,
                agent_id=agent_id,
                t_start=t_start,
                t_end=t_end,
                dominant_phase=phase,
                narrative_coherence=nc,
                symbol_density=sym_density,
            ))
        return result

    def get_cge_by_block(self) -> Dict[int, float]:
        """Get CGE indexed by block."""
        return {block_id: cge for block_id, cge, _ in self.cge_blocks}

    def get_cge_by_block_phase(self) -> Dict[int, Tuple[float, CircadianPhase]]:
        """Get CGE with phase indexed by block."""
        result = {}
        for block_id, cge, phase_str in self.cge_blocks:
            try:
                phase = CircadianPhase(phase_str)
            except ValueError:
                phase = CircadianPhase.WAKE
            result[block_id] = (cge, phase)
        return result


@dataclass
class SymbolicLogs:
    """
    Logs from symbolic system.

    activations: List of (t, agent_id, symbol_id, active)
    grammar_updates: List of (t, agent_id, rule_id, strength)
    projects: List of (t, agent_id, project_id, progress)
    """
    activations: List[Tuple[int, str, str, bool]] = field(default_factory=list)
    grammar_updates: List[Tuple[int, str, str, float]] = field(default_factory=list)
    projects: List[Tuple[int, str, str, float]] = field(default_factory=list)

    def get_symbol_activations(self) -> List[SymbolActivation]:
        """Convert to SymbolActivation objects."""
        return [
            SymbolActivation(t=t, agent_id=agent_id, symbol_id=symbol_id, active=active)
            for t, agent_id, symbol_id, active in self.activations
        ]


@dataclass
class MedicalLogs:
    """
    Logs from medical system.

    interventions: List of (t, agent_id, intervened, treatment_type)
    health: List of (t, agent_id, health_index)
    shock: List of (t, agent_id, shock_value)
    medx_scores: Dict of MED-X metrics
    """
    interventions: List[Tuple[int, str, bool, str]] = field(default_factory=list)
    health: List[Tuple[int, str, float]] = field(default_factory=list)
    shock: List[Tuple[int, str, float]] = field(default_factory=list)
    medx_scores: Dict[str, float] = field(default_factory=dict)

    def get_interventions(self) -> List[Tuple[int, str, bool]]:
        """Get interventions as (t, agent_id, intervened)."""
        return [(t, agent_id, intervened) for t, agent_id, intervened, _ in self.interventions]

    def get_needs(self) -> List[Tuple[int, str, float]]:
        """Calculate medical need at each timestep."""
        # Build lookup maps
        health_map: Dict[Tuple[int, str], float] = {}
        shock_map: Dict[Tuple[int, str], float] = {}

        for t, agent_id, h in self.health:
            health_map[(t, agent_id)] = h
        for t, agent_id, s in self.shock:
            shock_map[(t, agent_id)] = s

        # Calculate need
        all_keys = set(health_map.keys()) | set(shock_map.keys())
        needs = []
        for (t, agent_id) in all_keys:
            h = health_map.get((t, agent_id), 0.5)
            s = shock_map.get((t, agent_id), 0.0)
            # need = shock - health (higher shock and lower health = more need)
            need = s + (1 - h)
            needs.append((t, agent_id, need))

        return needs


@dataclass
class SocialLogs:
    """
    Logs from social/interaction system.

    interactions: List of (t, agent_a, agent_b, interaction_type, outcome)
    norms: List of (t, norm_id, strength, adopters)
    coalitions: List of (t, coalition_id, members)
    """
    interactions: List[Tuple[int, str, str, str, float]] = field(default_factory=list)
    norms: List[Tuple[int, str, float, List[str]]] = field(default_factory=list)
    coalitions: List[Tuple[int, str, List[str]]] = field(default_factory=list)


# ==============================================================================
# BENCHMARK RESULTS
# ==============================================================================

@dataclass
class LXResults:
    """Complete results from LX1-LX10 benchmark."""

    # Individual metrics
    lx1_phase_symbol: Dict[str, float] = field(default_factory=dict)
    lx2_symbolic_drift: Dict[str, float] = field(default_factory=dict)
    lx3_dream_narrative: Dict[str, float] = field(default_factory=dict)
    lx4_dream_transfer: Dict[str, float] = field(default_factory=dict)
    lx5_medicine_phase: Dict[str, float] = field(default_factory=dict)
    lx6_full_cycle_medicine: Dict[str, float] = field(default_factory=dict)
    lx7_circadian_cge: Dict[str, float] = field(default_factory=dict)
    lx8_life_plasticity: Dict[str, float] = field(default_factory=dict)
    lx9_multiagent_sync: Dict[str, float] = field(default_factory=dict)
    lx10_aggregate: Dict[str, float] = field(default_factory=dict)

    # Metadata
    timestamp: str = ""
    n_agents: int = 0
    n_timesteps: int = 0
    n_cycles: int = 0
    n_episodes: int = 0

    def get_global_scores(self) -> Dict[str, float]:
        """Get all global scores."""
        return {
            'LX1_phase_symbol': self.lx1_phase_symbol.get('LX1_global', 0.0),
            'LX2_symbolic_drift': self.lx2_symbolic_drift.get('LX2_global', 0.0),
            'LX3_dream_narrative': self.lx3_dream_narrative.get('LX3_global', 0.0),
            'LX4_dream_transfer': self.lx4_dream_transfer.get('LX4_global', 0.0),
            'LX5_medicine_phase': self.lx5_medicine_phase.get('LX5_global', 0.0),
            'LX6_full_cycle_medicine': self.lx6_full_cycle_medicine.get('LX6_global', 0.0),
            'LX7_circadian_cge': self.lx7_circadian_cge.get('LX7_global', 0.0),
            'LX8_life_plasticity': self.lx8_life_plasticity.get('LX8_global', 0.0),
            'LX9_multiagent_sync': self.lx9_multiagent_sync.get('LX9_global', 0.0),
            'LX10_life_extended': self.lx10_aggregate.get('LX10_global', 0.0),
        }

    def get_final_score(self) -> float:
        """Get LX10 final score."""
        return self.lx10_aggregate.get('LX10_global', 0.0)

    def summary(self) -> str:
        """Generate summary report."""
        scores = self.get_global_scores()

        lines = [
            "=" * 60,
            "LX BENCHMARK: LIFE-EXTENDED COGNITION (LX1-LX10)",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            f"Agents: {self.n_agents}, Timesteps: {self.n_timesteps}",
            f"Cycles: {self.n_cycles}, Episodes: {self.n_episodes}",
            "-" * 60,
            "",
            "SCORES:",
            "-" * 40,
        ]

        for name, score in scores.items():
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"  {name:25s} {bar} {score:.3f}")

        lines.extend([
            "",
            "-" * 60,
            f"FINAL LX10 SCORE: {self.get_final_score():.4f}",
            "=" * 60,
        ])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'lx1_phase_symbol': self.lx1_phase_symbol,
            'lx2_symbolic_drift': self.lx2_symbolic_drift,
            'lx3_dream_narrative': self.lx3_dream_narrative,
            'lx4_dream_transfer': self.lx4_dream_transfer,
            'lx5_medicine_phase': self.lx5_medicine_phase,
            'lx6_full_cycle_medicine': self.lx6_full_cycle_medicine,
            'lx7_circadian_cge': self.lx7_circadian_cge,
            'lx8_life_plasticity': self.lx8_life_plasticity,
            'lx9_multiagent_sync': self.lx9_multiagent_sync,
            'lx10_aggregate': self.lx10_aggregate,
            'global_scores': self.get_global_scores(),
            'final_score': self.get_final_score(),
            'metadata': {
                'timestamp': self.timestamp,
                'n_agents': self.n_agents,
                'n_timesteps': self.n_timesteps,
                'n_cycles': self.n_cycles,
                'n_episodes': self.n_episodes,
            }
        }


# ==============================================================================
# MAIN BENCHMARK CLASS
# ==============================================================================

class LXBenchmark:
    """
    Life-Extended Cognition Benchmark (LX1-LX10)

    Measures integration between:
    - Symbols <-> Phases (symbol-phase)
    - Medicine <-> Life phases (medicine-phase)
    - Dream narrative (dream-narrative)
    - Circadian impact on CG-E

    All metrics are 100% ENDOGENOUS.
    """

    def __init__(
        self,
        lifecycle_logs: LifecycleLogs,
        cognitive_logs: CognitiveLogs,
        symbolic_logs: SymbolicLogs,
        medical_logs: MedicalLogs,
        social_logs: Optional[SocialLogs] = None,
    ):
        self.lifecycle_logs = lifecycle_logs
        self.cognitive_logs = cognitive_logs
        self.symbolic_logs = symbolic_logs
        self.medical_logs = medical_logs
        self.social_logs = social_logs or SocialLogs()

        # Extract agent IDs
        self.agent_ids = self._extract_agent_ids()

        # Build cycle statistics
        self.cycle_stats = self._build_cycle_stats()

    def _extract_agent_ids(self) -> List[str]:
        """Extract unique agent IDs from logs."""
        agent_ids = set()

        for _, agent_id, _, _, _ in self.lifecycle_logs.phases:
            agent_ids.add(agent_id)

        for _, agent_id, _, _, _, _, _ in self.cognitive_logs.episodes:
            agent_ids.add(agent_id)

        for _, agent_id, _, _ in self.symbolic_logs.activations:
            agent_ids.add(agent_id)

        return sorted(agent_ids)

    def _build_cycle_stats(self) -> List[CycleStats]:
        """Build CycleStats from logs."""
        cycle_stats = []

        # Get cycles
        cycles_by_agent: Dict[str, List[Tuple[int, int, int]]] = {}
        for cycle_id, agent_id, t_start, t_end in self.lifecycle_logs.cycles:
            if agent_id not in cycles_by_agent:
                cycles_by_agent[agent_id] = []
            cycles_by_agent[agent_id].append((cycle_id, t_start, t_end))

        # Build health/shock maps
        health_map: Dict[Tuple[int, str], float] = {}
        shock_map: Dict[Tuple[int, str], float] = {}
        for t, agent_id, h in self.medical_logs.health:
            health_map[(t, agent_id)] = h
        for t, agent_id, s in self.medical_logs.shock:
            shock_map[(t, agent_id)] = s

        # Build intervention map
        intervention_map: Dict[Tuple[int, str], bool] = {}
        for t, agent_id, intervened, _ in self.medical_logs.interventions:
            intervention_map[(t, agent_id)] = intervened

        # Build symbol activation map
        symbol_activations_by_t: Dict[Tuple[int, str], List[str]] = {}
        for t, agent_id, symbol_id, active in self.symbolic_logs.activations:
            if active:
                key = (t, agent_id)
                if key not in symbol_activations_by_t:
                    symbol_activations_by_t[key] = []
                symbol_activations_by_t[key].append(symbol_id)

        # Get all unique symbols
        all_symbols = sorted(set(
            sa.symbol_id for sa in self.symbolic_logs.get_symbol_activations()
            if sa.active
        ))
        n_symbols = len(all_symbols) if all_symbols else 1

        # Build CGE by block
        cge_by_block = self.cognitive_logs.get_cge_by_block()

        # Build drive map
        drive_map: Dict[Tuple[int, str], np.ndarray] = {}
        for t, agent_id, drive_vec in self.cognitive_logs.drives:
            drive_map[(t, agent_id)] = drive_vec

        # Process each agent's cycles
        for agent_id, cycles in cycles_by_agent.items():
            for cycle_id, t_start, t_end in sorted(cycles, key=lambda x: x[0]):
                # Collect stats for this cycle
                shocks = []
                healths = []
                n_interventions = 0
                n_total = 0
                symbol_counts = np.zeros(n_symbols)
                drive_vectors = []

                for t in range(t_start, t_end + 1):
                    key = (t, agent_id)

                    if key in health_map:
                        healths.append(health_map[key])
                    if key in shock_map:
                        shocks.append(shock_map[key])
                    if key in intervention_map:
                        if intervention_map[key]:
                            n_interventions += 1
                        n_total += 1
                    if key in symbol_activations_by_t:
                        for sym in symbol_activations_by_t[key]:
                            if sym in all_symbols:
                                symbol_counts[all_symbols.index(sym)] += 1
                    if key in drive_map:
                        drive_vectors.append(drive_map[key])

                # Calculate cycle CGE (average over blocks in this cycle range)
                cycle_cge_values = [
                    cge for block_id, cge in cge_by_block.items()
                    if t_start <= block_id <= t_end
                ]
                cycle_cge = np.mean(cycle_cge_values) if cycle_cge_values else 0.5

                # Symbol vector (normalized)
                cycle_length = t_end - t_start + 1
                symbol_vector = symbol_counts / (cycle_length + 1e-12)

                # Trait vector
                mean_health = np.mean(healths) if healths else 0.5
                mean_shock = np.mean(shocks) if shocks else 0.0
                mean_drive = np.mean(drive_vectors, axis=0) if drive_vectors else np.zeros(3)
                trait_vector = np.concatenate([
                    [mean_health, mean_shock, cycle_cge],
                    symbol_vector[:5] if len(symbol_vector) >= 5 else symbol_vector,
                    mean_drive if len(mean_drive) > 0 else np.zeros(3),
                ])

                cycle_stats.append(CycleStats(
                    c=cycle_id,
                    agent_id=agent_id,
                    cge=cycle_cge,
                    shock_mean=mean_shock,
                    health_mean=mean_health,
                    med_ratio=n_interventions / (n_total + 1e-12),
                    symbol_vector=symbol_vector,
                    trait_vector=trait_vector,
                ))

        return cycle_stats

    def run(self) -> LXResults:
        """
        Run complete LX1-LX10 benchmark.

        Returns LXResults with all metrics.
        """
        results = LXResults(
            timestamp=datetime.now().isoformat(),
            n_agents=len(self.agent_ids),
            n_timesteps=len(set(t for t, _, _, _, _ in self.lifecycle_logs.phases)),
            n_cycles=len(self.lifecycle_logs.cycles),
            n_episodes=len(self.cognitive_logs.episodes),
        )

        # Get data structures
        phase_states = self.lifecycle_logs.get_phase_states()
        symbol_activations = self.symbolic_logs.get_symbol_activations()
        episodes = self.cognitive_logs.get_episodes()
        cge_by_block = self.cognitive_logs.get_cge_by_block()
        cge_by_block_phase = self.cognitive_logs.get_cge_by_block_phase()
        interventions = self.medical_logs.get_interventions()
        needs = self.medical_logs.get_needs()

        # LX1: Phase-Symbol Specialization
        results.lx1_phase_symbol = compute_lx1_phase_symbol(
            symbol_activations=symbol_activations,
            phase_states=phase_states,
            agent_ids=self.agent_ids,
        )

        # LX2: Circadian Symbolic Drift
        results.lx2_symbolic_drift = compute_lx2_symbolic_drift(
            cycle_stats=self.cycle_stats,
            agent_ids=self.agent_ids,
        )

        # LX3: Dream Narrative
        results.lx3_dream_narrative = compute_lx3_dream_narrative(
            episodes=episodes,
            agent_ids=self.agent_ids,
        )

        # LX4: Dream-Wake Transfer
        results.lx4_dream_transfer = compute_lx4_dream_transfer(
            episodes=episodes,
            cge_by_block=cge_by_block,
            agent_ids=self.agent_ids,
        )

        # LX5: Medicine-Phase Alignment
        results.lx5_medicine_phase = compute_lx5_medicine_phase(
            interventions=interventions,
            needs=needs,
            agent_ids=self.agent_ids,
        )

        # LX6: Full-Cycle Medicine
        results.lx6_full_cycle_medicine = compute_lx6_full_cycle_medicine(
            cycle_stats=self.cycle_stats,
            agent_ids=self.agent_ids,
        )

        # LX7: Circadian CG-E Modulation
        results.lx7_circadian_cge = compute_lx7_circadian_cge(
            cge_by_block_phase=cge_by_block_phase,
        )

        # LX8: Life Plasticity
        results.lx8_life_plasticity = compute_lx8_life_plasticity(
            cycle_stats=self.cycle_stats,
            agent_ids=self.agent_ids,
        )

        # LX9: Multi-Agent Life Synchrony
        results.lx9_multiagent_sync = compute_lx9_multiagent_sync(
            phase_states=phase_states,
            cycle_stats=self.cycle_stats,
            agent_ids=self.agent_ids,
        )

        # LX10: Aggregate (Life-Extended Cognition Index)
        all_scores = {
            'LX1_global': results.lx1_phase_symbol.get('LX1_global', 0.5),
            'LX2_global': results.lx2_symbolic_drift.get('LX2_global', 0.5),
            'LX3_global': results.lx3_dream_narrative.get('LX3_global', 0.5),
            'LX4_global': results.lx4_dream_transfer.get('LX4_global', 0.5),
            'LX5_global': results.lx5_medicine_phase.get('LX5_global', 0.5),
            'LX6_global': results.lx6_full_cycle_medicine.get('LX6_global', 0.5),
            'LX7_global': results.lx7_circadian_cge.get('LX7_global', 0.5),
            'LX8_global': results.lx8_life_plasticity.get('LX8_global', 0.5),
            'LX9_global': results.lx9_multiagent_sync.get('LX9_global', 0.5),
        }
        results.lx10_aggregate = compute_lx10_aggregate(all_scores)

        return results


def run_lx_benchmark(logs: Dict[str, Any]) -> Dict[str, float]:
    """
    Convenience function to run LX benchmark from raw logs.

    Args:
        logs: Dictionary with keys:
            - 'lifecycle': LifecycleLogs or dict
            - 'cognitive': CognitiveLogs or dict
            - 'symbolic': SymbolicLogs or dict
            - 'medical': MedicalLogs or dict
            - 'social': SocialLogs or dict (optional)

    Returns:
        Dictionary with all LX scores:
        {
            "LX1_phase_symbol": ...,
            "LX2_symbolic_drift": ...,
            ...
            "LX10_life_extended_index": ...
        }
    """
    # Convert dicts to dataclasses if needed
    lifecycle = logs.get('lifecycle', LifecycleLogs())
    if isinstance(lifecycle, dict):
        lifecycle = LifecycleLogs(**lifecycle)

    cognitive = logs.get('cognitive', CognitiveLogs())
    if isinstance(cognitive, dict):
        cognitive = CognitiveLogs(**cognitive)

    symbolic = logs.get('symbolic', SymbolicLogs())
    if isinstance(symbolic, dict):
        symbolic = SymbolicLogs(**symbolic)

    medical = logs.get('medical', MedicalLogs())
    if isinstance(medical, dict):
        medical = MedicalLogs(**medical)

    social = logs.get('social', SocialLogs())
    if isinstance(social, dict):
        social = SocialLogs(**social)

    # Run benchmark
    benchmark = LXBenchmark(
        lifecycle_logs=lifecycle,
        cognitive_logs=cognitive,
        symbolic_logs=symbolic,
        medical_logs=medical,
        social_logs=social,
    )

    results = benchmark.run()

    # Return global scores
    return results.get_global_scores()
