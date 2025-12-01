"""
Phase W1: WORLD-1 Core Dynamics Validation

Verify that WORLD-1 operates autonomously with:
- Non-degenerate variance
- Detectable regime changes
- Meaningful entropy
- Endogenous effective dimension
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from datetime import datetime

from world1.world1_core import World1Core
from world1.world1_entities import EntityPopulation
from world1.world1_metrics import WorldMetrics
from world1.world1_regimes import RegimeDetector


def run_phase_w1(steps: int = 1000):
    """
    Run Phase W1: Validate WORLD-1 autonomous dynamics.

    Tests:
    1. Variance not degenerate (neither 0 nor explosive)
    2. Entropy in intermediate range
    3. Regime changes detectable
    4. Irreversibility > null model
    5. Effective dimension meaningful
    """
    print("=" * 70)
    print("PHASE W1: WORLD-1 AUTONOMOUS DYNAMICS")
    print("=" * 70)

    # Initialize world
    world = World1Core(n_fields=4, n_entities=5, n_resources=3, n_modes=3)
    entities = EntityPopulation(n_entities=5, position_dim=3, state_dim=4)
    metrics = WorldMetrics(world.D)
    regime_detector = RegimeDetector(world.D)

    print(f"\nWorld Configuration:")
    print(f"  Total dimension D = {world.D}")
    print(f"  Fields: {world.n_fields}")
    print(f"  Entities: {world.n_entities}")
    print(f"  Resources: {world.n_resources}")
    print(f"  Modes: {world.n_modes}")
    print(f"\nRunning {steps} steps...")

    # Storage for analysis
    history = {
        'variance': [],
        'entropy': [],
        'health': [],
        'phi_world': [],
        'd_eff': [],
        'regime': [],
        'shock': []
    }

    # Run simulation
    for t in range(steps):
        # Entity perturbation to world
        entity_state = entities.get_population_vector()
        entity_perturbation = np.zeros(world.D)
        # Map entity state to world (partial mapping)
        entity_perturbation[:min(len(entity_state), world.D)] = \
            entity_state[:min(len(entity_state), world.D)] * 0.01

        # World step
        w = world.step({'entities': entity_perturbation})

        # Entity step (influenced by world fields)
        field_effect = world.get_state().fields
        entities.step(field_effect)

        # Compute metrics
        state = world.get_state()
        snapshot = metrics.compute_all(w, state.modes)
        regime = regime_detector.detect_regime(w)

        # Record
        history['variance'].append(float(np.var(w)))
        history['entropy'].append(snapshot.entropy)
        history['health'].append(snapshot.health)
        history['phi_world'].append(snapshot.phi_world)
        history['d_eff'].append(world.d_eff)
        history['regime'].append(regime)
        history['shock'].append(snapshot.shock_magnitude)

        # Progress
        if (t + 1) % 200 == 0:
            print(f"\n  t={t+1}:")
            print(f"    d_eff = {world.d_eff}")
            print(f"    Entropy = {snapshot.entropy:.3f}")
            print(f"    Health = {snapshot.health:.3f}")
            print(f"    Phi_w = {snapshot.phi_world:.3f}")
            print(f"    Regime = {regime}")

    # Validation tests
    print("\n" + "=" * 70)
    print("VALIDATION TESTS")
    print("=" * 70)

    results = {
        'tests': {},
        'metrics': {}
    }

    def to_json_safe(val):
        """Convert numpy types to JSON-serializable."""
        if isinstance(val, (np.bool_, bool)):
            return bool(val)
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return float(val)
        return val

    # Test 1: Non-degenerate variance
    variance_array = np.array(history['variance'])
    mean_var = np.mean(variance_array)
    min_var = np.min(variance_array)
    max_var = np.max(variance_array)

    test1_pass = 0.001 < mean_var < 10 and min_var > 1e-6 and max_var < 100
    results['tests']['variance_non_degenerate'] = to_json_safe(test1_pass)
    print(f"\n1. Variance non-degenerate:")
    print(f"   Mean: {mean_var:.4f}, Min: {min_var:.4f}, Max: {max_var:.4f}")
    print(f"   Status: {'PASS' if test1_pass else 'FAIL'}")

    # Test 2: Entropy in intermediate range
    entropy_array = np.array(history['entropy'])
    mean_entropy = np.mean(entropy_array)
    entropy_std = np.std(entropy_array)

    test2_pass = 0.1 < mean_entropy < 0.9 and entropy_std > 0.01
    results['tests']['entropy_intermediate'] = to_json_safe(test2_pass)
    print(f"\n2. Entropy in intermediate range:")
    print(f"   Mean: {mean_entropy:.3f}, Std: {entropy_std:.3f}")
    print(f"   Status: {'PASS' if test2_pass else 'FAIL'}")

    # Test 3: Regime changes detected
    regime_array = np.array(history['regime'])
    regime_changes = np.sum(np.diff(regime_array) != 0)
    n_regimes = len(set(regime_array))

    test3_pass = regime_changes > 5 and n_regimes >= 2
    results['tests']['regime_changes'] = to_json_safe(test3_pass)
    print(f"\n3. Regime changes detected:")
    print(f"   Total changes: {regime_changes}")
    print(f"   Unique regimes: {n_regimes}")
    print(f"   Status: {'PASS' if test3_pass else 'FAIL'}")

    # Test 4: Health meaningful (not stuck at extremes)
    health_array = np.array(history['health'])
    mean_health = np.mean(health_array)
    health_std = np.std(health_array)

    test4_pass = 0.2 < mean_health < 0.8 and health_std > 0.01
    results['tests']['health_meaningful'] = to_json_safe(test4_pass)
    print(f"\n4. Health meaningful:")
    print(f"   Mean: {mean_health:.3f}, Std: {health_std:.3f}")
    print(f"   Status: {'PASS' if test4_pass else 'FAIL'}")

    # Test 5: Effective dimension adapts
    d_eff_array = np.array(history['d_eff'])
    d_eff_changes = np.sum(np.diff(d_eff_array) != 0)
    final_d_eff = d_eff_array[-1]

    test5_pass = 1 < final_d_eff < world.D and d_eff_changes > 0
    results['tests']['d_eff_adapts'] = to_json_safe(test5_pass)
    print(f"\n5. Effective dimension adapts:")
    print(f"   Final d_eff: {final_d_eff} / {world.D}")
    print(f"   Changes: {d_eff_changes}")
    print(f"   Status: {'PASS' if test5_pass else 'FAIL'}")

    # Test 6: Phi_world not trivial
    phi_array = np.array(history['phi_world'])
    mean_phi = np.mean(phi_array)
    phi_std = np.std(phi_array)

    test6_pass = 0.1 < mean_phi < 0.9 and phi_std > 0.01
    results['tests']['phi_world_meaningful'] = to_json_safe(test6_pass)
    print(f"\n6. Phi_world meaningful:")
    print(f"   Mean: {mean_phi:.3f}, Std: {phi_std:.3f}")
    print(f"   Status: {'PASS' if test6_pass else 'FAIL'}")

    # Summary
    tests_passed = sum(results['tests'].values())
    total_tests = len(results['tests'])

    print("\n" + "=" * 70)
    print(f"PHASE W1 SUMMARY: {tests_passed}/{total_tests} tests passed")
    print("=" * 70)

    # Store metrics
    results['metrics'] = {
        'mean_variance': float(mean_var),
        'mean_entropy': float(mean_entropy),
        'mean_health': float(mean_health),
        'mean_phi_world': float(mean_phi),
        'final_d_eff': int(final_d_eff),
        'n_regime_changes': int(regime_changes),
        'n_unique_regimes': int(n_regimes)
    }

    results['phase'] = 'W1'
    results['status'] = 'PASS' if tests_passed == total_tests else 'PARTIAL'
    results['timestamp'] = datetime.now().isoformat()

    # Save results
    os.makedirs('results/world1', exist_ok=True)
    with open('results/world1/phaseW1_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to results/world1/phaseW1_results.json")

    return results


if __name__ == "__main__":
    run_phase_w1(steps=1000)
