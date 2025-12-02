"""
BENCHMARK MODULE
================

Two benchmark suites for measuring cognitive capabilities:

AGI-X BENCHMARK (S1-S10):
    Suite de 10 tests estructurales para medir inteligencia general interna.
    S1: Adaptación a regímenes
    S2: Generalización a nuevas tareas
    S3: Planeamiento multipaso
    S4: Auto-modelo (Self-Model)
    S5: Theory of Mind
    S6: Emergencia de normas
    S7: Curiosidad estructural
    S8: Ética estructural
    S9: Continuidad narrativa
    S10: Madurez vital
    Score: AGI-X = (1/10) Σ S_i

LX BENCHMARK (LX1-LX10): Life-Extended Cognition
    Suite de 10 tests para medir cognición extendida en el ciclo de vida.
    LX1: Phase-Symbol Specialization - Symbol affinity to circadian phases
    LX2: Circadian Symbolic Drift - Symbol evolution across life cycles
    LX3: Dream Narrative - Narrative coherence during DREAM phase
    LX4: Dream-Wake Transfer - Learning transfer from dreams to waking
    LX5: Medicine-Phase Alignment - Medical intervention timing accuracy
    LX6: Full-Cycle Medicine - Cumulative medical impact on life trajectory
    LX7: Circadian CG-E Modulation - Phase structure in global coherence
    LX8: Life Plasticity - Structural change while maintaining identity
    LX9: Multi-Agent Life Synchrony - Alignment of agent life cycles
    LX10: Life-Extended Cognition Index - Aggregate of LX1-LX9
    Score: LX10 = Σ w_i × LX_i (inverse-variance weighted)

All metrics are 100% ENDOGENOUS:
- Windows: L_t = max(3, floor(sqrt(t)))
- Percentiles: Q_p computed on agent's own history
- Ranks: rank(x) in [0,1] from internal distribution
"""

# AGI-X Benchmark
from .run_benchmark import run_benchmark, BenchmarkResults

# LX Benchmark - Life-Extended Cognition
from .lx_metrics import (
    # Core utilities
    endogenous_window,
    endogenous_rank,
    endogenous_percentile,
    # Individual metrics
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
    # Data structures
    CircadianPhase,
    SymbolActivation,
    PhaseState,
    Episode,
    CycleStats,
)

from .lx_benchmark import (
    LXBenchmark,
    LXResults,
    LifecycleLogs,
    CognitiveLogs,
    SymbolicLogs,
    MedicalLogs,
    SocialLogs,
    run_lx_benchmark,
)

__all__ = [
    # AGI-X
    'run_benchmark',
    'BenchmarkResults',
    # LX Utilities
    'endogenous_window',
    'endogenous_rank',
    'endogenous_percentile',
    # LX Metrics
    'compute_lx1_phase_symbol',
    'compute_lx2_symbolic_drift',
    'compute_lx3_dream_narrative',
    'compute_lx4_dream_transfer',
    'compute_lx5_medicine_phase',
    'compute_lx6_full_cycle_medicine',
    'compute_lx7_circadian_cge',
    'compute_lx8_life_plasticity',
    'compute_lx9_multiagent_sync',
    'compute_lx10_aggregate',
    # LX Data Structures
    'CircadianPhase',
    'SymbolActivation',
    'PhaseState',
    'Episode',
    'CycleStats',
    # LX Benchmark
    'LXBenchmark',
    'LXResults',
    'LifecycleLogs',
    'CognitiveLogs',
    'SymbolicLogs',
    'MedicalLogs',
    'SocialLogs',
    'run_lx_benchmark',
]
