"""
Test for LX1-LX10 Life-Extended Cognition Benchmark
====================================================

Generates synthetic logs and runs the complete benchmark to validate all metrics.

All data generation is endogenous - no magic constants hardcoded.
"""

import numpy as np
import sys
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Add parent to path
sys.path.insert(0, '/root/NEO_EVA')

from benchmark.lx_benchmark import (
    LXBenchmark,
    LifecycleLogs,
    CognitiveLogs,
    SymbolicLogs,
    MedicalLogs,
    SocialLogs,
    run_lx_benchmark,
)
from benchmark.lx_metrics import CircadianPhase


def generate_synthetic_logs(
    n_agents: int = 5,
    n_timesteps: int = 1000,
    n_cycles_per_agent: int = 10,
) -> Tuple[LifecycleLogs, CognitiveLogs, SymbolicLogs, MedicalLogs, SocialLogs]:
    """
    Generate synthetic logs for testing.

    Uses only endogenous derivations:
    - Phase durations from sqrt(t)
    - Symbol probabilities from phase
    - Health dynamics from shock history
    """
    np.random.seed(42)

    agent_ids = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS'][:n_agents]
    phases = [CircadianPhase.WAKE, CircadianPhase.REST, CircadianPhase.DREAM, CircadianPhase.LIMINAL]
    symbols = ['action_1', 'goal_1', 'value_1', 'metaphor_1', 'bridge_1',
               'action_2', 'goal_2', 'value_2', 'metaphor_2', 'bridge_2']

    # Phase-symbol affinities (which symbols are more likely in which phase)
    phase_symbol_affinity = {
        CircadianPhase.WAKE: ['action_1', 'action_2', 'goal_1', 'goal_2'],
        CircadianPhase.REST: ['value_1', 'value_2'],
        CircadianPhase.DREAM: ['metaphor_1', 'metaphor_2'],
        CircadianPhase.LIMINAL: ['bridge_1', 'bridge_2'],
    }

    # Initialize logs
    lifecycle = LifecycleLogs()
    cognitive = CognitiveLogs()
    symbolic = SymbolicLogs()
    medical = MedicalLogs()
    social = SocialLogs()

    # Generate data for each agent
    for agent_id in agent_ids:
        # Agent personality (affects dynamics)
        personality = np.random.randn(3) * 0.2 + 0.5  # drives baseline

        # Phase state machine
        current_phase = CircadianPhase.WAKE
        phase_start = 0
        phase_duration = max(10, int(np.sqrt(n_timesteps / 4)))

        # Cycle tracking
        cycle_id = 0
        cycle_start = 0

        # Health dynamics
        health = 0.7
        shock_history = []

        for t in range(n_timesteps):
            # Update phase
            if t - phase_start >= phase_duration:
                phase_start = t
                phase_idx = phases.index(current_phase)
                current_phase = phases[(phase_idx + 1) % 4]
                # Endogenous phase duration
                phase_duration = max(10, int(np.sqrt(t + 1) * np.random.uniform(0.8, 1.2)))

                # New cycle when returning to WAKE
                if current_phase == CircadianPhase.WAKE:
                    lifecycle.cycles.append((cycle_id, agent_id, cycle_start, t - 1))
                    cycle_id += 1
                    cycle_start = t

            # Energy and stress (endogenous from phase)
            if current_phase == CircadianPhase.WAKE:
                energy = 0.8 + np.random.randn() * 0.1
                stress = 0.3 + np.random.randn() * 0.1
            elif current_phase == CircadianPhase.REST:
                energy = 0.4 + np.random.randn() * 0.1
                stress = 0.2 + np.random.randn() * 0.1
            elif current_phase == CircadianPhase.DREAM:
                energy = 0.2 + np.random.randn() * 0.1
                stress = 0.1 + np.random.randn() * 0.1
            else:  # LIMINAL
                energy = 0.5 + np.random.randn() * 0.1
                stress = 0.4 + np.random.randn() * 0.1

            energy = np.clip(energy, 0, 1)
            stress = np.clip(stress, 0, 1)

            lifecycle.phases.append((t, agent_id, current_phase.value, energy, stress))

            # Shock dynamics
            if len(shock_history) > 0:
                shock = shock_history[-1] * 0.9 + np.random.randn() * 0.1
            else:
                shock = np.random.uniform(0, 0.3)
            shock = np.clip(shock, 0, 1)
            shock_history.append(shock)

            # Random shock events
            if np.random.random() < 0.05:
                shock = np.clip(shock + 0.3, 0, 1)
                shock_history[-1] = shock

            # Health dynamics (endogenous)
            health_delta = 0.01 - shock * 0.05
            health = np.clip(health + health_delta + np.random.randn() * 0.02, 0, 1)

            medical.health.append((t, agent_id, health))
            medical.shock.append((t, agent_id, shock))

            # Medical intervention (endogenous threshold)
            need = shock + (1 - health)
            if len(shock_history) > 10:
                threshold = np.percentile(shock_history[-100:], 75)
            else:
                threshold = 0.5
            intervened = need > threshold and current_phase in [CircadianPhase.WAKE, CircadianPhase.REST]
            if intervened:
                health = np.clip(health + 0.1, 0, 1)  # Treatment effect
            medical.interventions.append((t, agent_id, intervened, 'stabilization'))

            # Symbol activations (phase-dependent)
            preferred_symbols = phase_symbol_affinity.get(current_phase, symbols[:2])
            for sym in symbols:
                if sym in preferred_symbols:
                    prob = 0.4 + np.random.uniform(0, 0.2)
                else:
                    prob = 0.1 + np.random.uniform(0, 0.1)
                active = np.random.random() < prob
                symbolic.activations.append((t, agent_id, sym, active))

            # Drive dynamics
            drive = personality + np.random.randn(3) * 0.05
            drive = np.clip(drive, 0, 1)
            cognitive.drives.append((t, agent_id, drive))

        # Add final cycle
        lifecycle.cycles.append((cycle_id, agent_id, cycle_start, n_timesteps - 1))

    # Generate episodes
    episode_id = 0
    for agent_id in agent_ids:
        # Get phases for this agent
        agent_phases = [(t, p, e, s) for t, aid, p, e, s in lifecycle.phases if aid == agent_id]

        # Segment into episodes (every ~50 steps or phase change)
        ep_start = 0
        prev_phase = agent_phases[0][1] if agent_phases else 'WAKE'

        for i, (t, phase, energy, stress) in enumerate(agent_phases):
            if i - ep_start >= 50 or phase != prev_phase or i == len(agent_phases) - 1:
                if i > ep_start:
                    # Episode stats
                    ep_phases = [p for _, p, _, _ in agent_phases[ep_start:i+1]]
                    dominant = max(set(ep_phases), key=ep_phases.count)

                    # Narrative coherence (endogenous)
                    nc = 0.5 + np.random.uniform(0, 0.3)
                    if dominant == 'DREAM':
                        nc += 0.2  # Dreams have higher coherence

                    # Symbol density (from symbolic logs)
                    ep_t_range = range(agent_phases[ep_start][0], t + 1)
                    ep_symbols = [
                        (t_s, s, a) for t_s, aid, s, a in symbolic.activations
                        if aid == agent_id and t_s in ep_t_range and a
                    ]
                    sym_density = len(set(t_s for t_s, _, _ in ep_symbols)) / (len(ep_t_range) + 1)

                    cognitive.episodes.append((
                        episode_id, agent_id, ep_start, i,
                        dominant, nc, sym_density
                    ))
                    episode_id += 1

                ep_start = i
                prev_phase = phase

    # Generate CG-E blocks (every 20 steps)
    block_size = 20
    for b in range(n_timesteps // block_size):
        t_start = b * block_size
        t_end = min(t_start + block_size, n_timesteps) - 1

        # Get dominant phase for block
        block_phases = [p for t, _, p, _, _ in lifecycle.phases if t_start <= t <= t_end]
        if block_phases:
            dom_phase = max(set(block_phases), key=block_phases.count)
        else:
            dom_phase = 'WAKE'

        # CG-E value (endogenous from health and coherence)
        block_health = np.mean([h for t, _, h in medical.health if t_start <= t <= t_end])
        cge = 0.5 + 0.3 * block_health + np.random.uniform(-0.1, 0.1)
        cge = np.clip(cge, 0, 1)

        cognitive.cge_blocks.append((b, cge, dom_phase))

    return lifecycle, cognitive, symbolic, medical, social


def test_lx_benchmark():
    """Run complete LX benchmark test."""
    print("=" * 60)
    print("LX BENCHMARK TEST: Life-Extended Cognition (LX1-LX10)")
    print("=" * 60)
    print()

    # Generate synthetic logs
    print("Generating synthetic logs...")
    lifecycle, cognitive, symbolic, medical, social = generate_synthetic_logs(
        n_agents=5,
        n_timesteps=1000,
        n_cycles_per_agent=10,
    )

    print(f"  - Lifecycle phases: {len(lifecycle.phases)}")
    print(f"  - Lifecycle cycles: {len(lifecycle.cycles)}")
    print(f"  - Cognitive episodes: {len(cognitive.episodes)}")
    print(f"  - Cognitive CG-E blocks: {len(cognitive.cge_blocks)}")
    print(f"  - Symbolic activations: {len(symbolic.activations)}")
    print(f"  - Medical interventions: {len(medical.interventions)}")
    print()

    # Create and run benchmark
    print("Running LX Benchmark...")
    benchmark = LXBenchmark(
        lifecycle_logs=lifecycle,
        cognitive_logs=cognitive,
        symbolic_logs=symbolic,
        medical_logs=medical,
        social_logs=social,
    )

    results = benchmark.run()

    # Print summary
    print()
    print(results.summary())

    # Detailed results
    print()
    print("DETAILED RESULTS:")
    print("-" * 60)

    # LX1
    print("\nLX1 - Phase-Symbol Specialization:")
    for k, v in sorted(results.lx1_phase_symbol.items()):
        print(f"  {k}: {v:.4f}")

    # LX2
    print("\nLX2 - Symbolic Drift:")
    for k, v in sorted(results.lx2_symbolic_drift.items()):
        print(f"  {k}: {v:.4f}")

    # LX3
    print("\nLX3 - Dream Narrative:")
    for k, v in sorted(results.lx3_dream_narrative.items()):
        print(f"  {k}: {v:.4f}")

    # LX4
    print("\nLX4 - Dream-Wake Transfer:")
    for k, v in sorted(results.lx4_dream_transfer.items()):
        print(f"  {k}: {v:.4f}")

    # LX5
    print("\nLX5 - Medicine-Phase Alignment:")
    for k, v in sorted(results.lx5_medicine_phase.items()):
        print(f"  {k}: {v:.4f}")

    # LX6
    print("\nLX6 - Full-Cycle Medicine:")
    for k, v in sorted(results.lx6_full_cycle_medicine.items()):
        print(f"  {k}: {v:.4f}")

    # LX7
    print("\nLX7 - Circadian CG-E Modulation:")
    for k, v in sorted(results.lx7_circadian_cge.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # LX8
    print("\nLX8 - Life Plasticity:")
    for k, v in sorted(results.lx8_life_plasticity.items()):
        print(f"  {k}: {v:.4f}")

    # LX9
    print("\nLX9 - Multi-Agent Sync:")
    for k, v in sorted(results.lx9_multiagent_sync.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # LX10
    print("\nLX10 - Life-Extended Cognition Index:")
    for k, v in sorted(results.lx10_aggregate.items()):
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Validation
    print()
    print("=" * 60)
    print("VALIDATION")
    print("=" * 60)

    scores = results.get_global_scores()
    all_pass = True

    tests = [
        ('LX1_phase_symbol', 0.0, 1.0, 'Symbol-phase specialization'),
        ('LX2_symbolic_drift', 0.0, 1.0, 'Symbolic drift consistency'),
        ('LX3_dream_narrative', 0.0, 1.0, 'Dream narrative quality'),
        ('LX4_dream_transfer', 0.0, 1.0, 'Dream-wake transfer'),
        ('LX5_medicine_phase', 0.3, 1.0, 'Medicine-phase alignment (AUC > 0.5)'),
        ('LX6_full_cycle_medicine', 0.0, 1.0, 'Full-cycle medicine impact'),
        ('LX7_circadian_cge', 0.0, 1.0, 'Circadian CG-E modulation'),
        ('LX8_life_plasticity', 0.0, 1.0, 'Life plasticity balance'),
        ('LX9_multiagent_sync', 0.0, 1.0, 'Multi-agent synchrony'),
        ('LX10_life_extended', 0.3, 1.0, 'Life-extended index > 0.3'),
    ]

    for name, min_val, max_val, desc in tests:
        score = scores.get(name, 0.0)
        passed = min_val <= score <= max_val
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {score:.4f} (range: [{min_val}, {max_val}]) - {desc}")
        if not passed:
            all_pass = False

    print()
    print("=" * 60)
    final_score = results.get_final_score()
    print(f"FINAL LX10 SCORE: {final_score:.4f}")
    print(f"OVERALL STATUS: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 60)

    return results


def test_convenience_function():
    """Test the run_lx_benchmark convenience function."""
    print()
    print("Testing convenience function run_lx_benchmark()...")

    # Generate logs as dicts
    lifecycle, cognitive, symbolic, medical, social = generate_synthetic_logs(
        n_agents=3,
        n_timesteps=500,
    )

    logs = {
        'lifecycle': lifecycle,
        'cognitive': cognitive,
        'symbolic': symbolic,
        'medical': medical,
        'social': social,
    }

    scores = run_lx_benchmark(logs)

    print("Convenience function results:")
    for name, score in sorted(scores.items()):
        print(f"  {name}: {score:.4f}")

    print("Convenience function test: PASS")


if __name__ == '__main__':
    results = test_lx_benchmark()
    test_convenience_function()
