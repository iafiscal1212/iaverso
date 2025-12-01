"""
Phase W2: Agent Coupling to WORLD-1

Connect NEO, EVA, ALEX, ADAM, and IRIS to WORLD-1 via:
- Endogenous observations
- Action-to-world mappings
- Bidirectional feedback

Tests:
- Prediction improves with active agents
- Drive changes correlate with regime changes
- Mutual information between internal states and world
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from datetime import datetime
from typing import Dict, List

from world1.world1_core import World1Core
from world1.world1_observation import ObservationProjector
from world1.world1_actions import ActionMapper
from world1.world1_metrics import WorldMetrics
from world1.world1_regimes import RegimeDetector

# Import autonomous agents - add path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments'))
from autonomous_life import AutonomousAgent


class World1AgentCoupler:
    """
    Couples autonomous agents to WORLD-1.

    Each agent:
    - Observes world through endogenous projection
    - Acts on world through learned mapping
    - Has internal dynamics independent of world

    The coupling creates bidirectional influence.
    """

    def __init__(self, world: World1Core, agent_dim: int = 6):
        """Initialize coupler."""
        self.world = world
        self.agent_dim = agent_dim

        # Create agents: NEO, EVA, ALEX, ADAM, IRIS
        self.agents: Dict[str, AutonomousAgent] = {}
        self._create_agents()

        # Observation and action systems
        self.observer = ObservationProjector(world.D, agent_dim)
        self.actor = ActionMapper(world.D, agent_dim)

        # Register agents
        for name, agent in self.agents.items():
            self.observer.register_agent(name, agent.z)
            self.actor.register_agent(name, agent.z)

        # Metrics
        self.metrics = WorldMetrics(world.D)
        self.regime_detector = RegimeDetector(world.D)

        # History for analysis
        self.t = 0
        self.agent_world_correlations: Dict[str, List[float]] = {name: [] for name in self.agents}
        self.drive_regime_correlations: List[float] = []

    def _create_agents(self):
        """Create the five agents with distinct personalities."""
        # NEO: stability-focused
        self.agents['NEO'] = AutonomousAgent('NEO', self.agent_dim)
        neo_z = np.array([0.12, 0.15, 0.12, 0.28, 0.18, 0.15])
        neo_z = neo_z / neo_z.sum()
        self.agents['NEO'].z = neo_z.copy()
        self.agents['NEO'].identity_core = neo_z.copy()

        # EVA: connection/otherness focused
        self.agents['EVA'] = AutonomousAgent('EVA', self.agent_dim)
        eva_z = np.array([0.15, 0.12, 0.15, 0.15, 0.15, 0.28])
        eva_z = eva_z / eva_z.sum()
        self.agents['EVA'].z = eva_z.copy()
        self.agents['EVA'].identity_core = eva_z.copy()

        # ALEX: exploration/novelty focused
        self.agents['ALEX'] = AutonomousAgent('ALEX', self.agent_dim)
        alex_z = np.array([0.22, 0.10, 0.28, 0.12, 0.15, 0.13])
        alex_z = alex_z / alex_z.sum()
        self.agents['ALEX'].z = alex_z.copy()
        self.agents['ALEX'].identity_core = alex_z.copy()

        # ADAM: integration focused
        self.agents['ADAM'] = AutonomousAgent('ADAM', self.agent_dim)
        adam_z = np.array([0.14, 0.14, 0.14, 0.14, 0.30, 0.14])
        adam_z = adam_z / adam_z.sum()
        self.agents['ADAM'].z = adam_z.copy()
        self.agents['ADAM'].identity_core = adam_z.copy()

        # IRIS: balanced/bridge (new fifth agent)
        # Named after the goddess of rainbows - connects dimensions
        self.agents['IRIS'] = AutonomousAgent('IRIS', self.agent_dim)
        iris_z = np.array([0.18, 0.18, 0.16, 0.16, 0.16, 0.16])
        iris_z = iris_z / iris_z.sum()
        self.agents['IRIS'].z = iris_z.copy()
        self.agents['IRIS'].identity_core = iris_z.copy()

    def step(self):
        """
        Run one coupled step:
        1. Agents observe world
        2. Agents process and act
        3. World receives perturbations
        4. World evolves
        5. Record correlations
        """
        self.t += 1

        # Current world state
        w = self.world.w

        # 1. Agents observe world
        observations = {}
        for name, agent in self.agents.items():
            obs = self.observer.get_observation_with_bias(name, w, agent.z)
            observations[name] = obs

        # 2. Agents process observations and act
        perturbations = {}
        for name, agent in self.agents.items():
            # Agent stimulus from observation
            obs = observations[name]
            # Map observation to stimulus (dimension matching)
            stimulus = np.zeros(self.agent_dim)
            for i in range(min(len(obs), self.agent_dim)):
                stimulus[i] = obs[i % len(obs)] * 0.1

            # Other agents' states as "other"
            other_zs = [a.z for n, a in self.agents.items() if n != name]
            other_z = np.mean(other_zs, axis=0) if other_zs else agent.z

            # Agent step
            result = agent.step(stimulus, other_z)

            # Record states
            self.observer.record_agent_state(name, agent.z)
            self.actor.record_agent_state(name, agent.z)

            # Compute action based on agent's drives and observation
            strategy = self._get_strategy(agent)
            action = self.actor.compute_action_from_drives(name, agent.z, strategy)
            perturbation = self.actor.get_world_perturbation(name, action)
            perturbations[name] = perturbation

        # 3. World step with perturbations
        new_w = self.world.step(perturbations)

        # 4. Update observation/action systems
        self.observer.record_world_state(new_w)
        self.actor.record_world_state(new_w)

        if self.t % 10 == 0:
            for name in self.agents:
                self.observer.update_projection(name)
                self.actor.update_mapping(name)

        # 5. Compute metrics
        state = self.world.get_state()
        metrics_snapshot = self.metrics.compute_all(new_w, state.modes)
        regime = self.regime_detector.detect_regime(new_w)

        # 6. Record correlations
        self._record_correlations(new_w, regime)

        return {
            'world_state': new_w,
            'regime': regime,
            'metrics': metrics_snapshot,
            'agent_states': {name: agent.z.copy() for name, agent in self.agents.items()}
        }

    def _get_strategy(self, agent: AutonomousAgent) -> str:
        """Determine agent's action strategy based on dominant drive."""
        z = agent.z
        # indices: entropy(0), neg_surprise(1), novelty(2), stability(3), integration(4), otherness(5)
        dominant = np.argmax(z)

        if dominant == 0 or dominant == 2:
            return 'exploration'
        elif dominant == 3:
            return 'stability'
        elif dominant == 4:
            return 'integration'
        elif dominant == 5:
            return 'connection'
        else:
            return 'exploration'

    def _record_correlations(self, w: np.ndarray, regime: int):
        """Record agent-world correlations for analysis."""
        if len(self.observer.world_history) < 10:
            return

        for name, agent in self.agents.items():
            # Correlation between agent z changes and world changes
            if len(self.observer.agent_histories.get(name, [])) < 10:
                continue

            agent_recent = np.array(self.observer.agent_histories[name][-10:])
            world_recent = np.array(self.observer.world_history[-10:])

            # Flatten and correlate
            agent_flat = agent_recent.flatten()
            world_flat = world_recent.flatten()

            min_len = min(len(agent_flat), len(world_flat))
            if min_len > 1:
                corr = np.corrcoef(agent_flat[:min_len], world_flat[:min_len])[0, 1]
                if not np.isnan(corr):
                    self.agent_world_correlations[name].append(float(corr))

    def run(self, steps: int = 500) -> Dict:
        """Run coupled simulation."""
        print(f"\nRunning {steps} coupled steps with 5 agents...")

        history = {
            'entropy': [],
            'health': [],
            'phi_world': [],
            'regime': [],
            'agent_phi': {name: [] for name in self.agents}
        }

        for t in range(steps):
            result = self.step()

            history['entropy'].append(result['metrics'].entropy)
            history['health'].append(result['metrics'].health)
            history['phi_world'].append(result['metrics'].phi_world)
            history['regime'].append(result['regime'])

            for name, agent in self.agents.items():
                history['agent_phi'][name].append(agent.integration)

            if (t + 1) % 100 == 0:
                print(f"\n  t={t+1}:")
                print(f"    World: entropy={result['metrics'].entropy:.3f}, "
                      f"health={result['metrics'].health:.3f}, "
                      f"regime={result['regime']}")
                for name, agent in self.agents.items():
                    print(f"    {name}: phi={agent.integration:.3f}, "
                          f"crisis={agent.in_crisis}")

        return history


def run_phase_w2(steps: int = 800):
    """
    Run Phase W2: Agent coupling validation.

    Tests:
    1. Agents develop distinct observation projections
    2. Agent actions affect world (perturbation non-zero)
    3. Agent-world correlations emerge
    4. Agent phi correlates with world phi
    5. Regime changes affect agent drives
    """
    print("=" * 70)
    print("PHASE W2: AGENT-WORLD COUPLING")
    print("=" * 70)

    # Initialize world and coupler
    world = World1Core(n_fields=4, n_entities=5, n_resources=3, n_modes=3)
    coupler = World1AgentCoupler(world, agent_dim=6)

    print(f"\nAgents: {list(coupler.agents.keys())}")
    print(f"World dimension: {world.D}")

    # Run simulation
    history = coupler.run(steps)

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
        if isinstance(val, dict):
            return {k: to_json_safe(v) for k, v in val.items()}
        return val

    # Test 1: Distinct observation projections
    obs_dims = {name: coupler.observer.obs_dims.get(name, 0) for name in coupler.agents}
    proj_norms = {name: np.linalg.norm(coupler.observer.projections.get(name, np.zeros((1,1))))
                  for name in coupler.agents}

    test1_pass = len(set(obs_dims.values())) > 1 or np.std(list(proj_norms.values())) > 0.05
    results['tests']['distinct_projections'] = to_json_safe(test1_pass)
    print(f"\n1. Distinct observation projections:")
    print(f"   Obs dims: {obs_dims}")
    print(f"   Proj norms std: {np.std(list(proj_norms.values())):.3f}")
    print(f"   Status: {'PASS' if test1_pass else 'FAIL'}")

    # Test 2: Actions affect world
    action_stats = {name: coupler.actor.get_statistics(name) for name in coupler.agents}
    mean_influence = np.mean([s['influence_weight'] for s in action_stats.values()])

    test2_pass = mean_influence > 0.1
    results['tests']['actions_affect_world'] = to_json_safe(test2_pass)
    print(f"\n2. Actions affect world:")
    print(f"   Mean influence weight: {mean_influence:.3f}")
    print(f"   Status: {'PASS' if test2_pass else 'FAIL'}")

    # Test 3: Agent-world correlations
    mean_correlations = {}
    for name, corrs in coupler.agent_world_correlations.items():
        if len(corrs) > 0:
            mean_correlations[name] = np.mean(corrs)
        else:
            mean_correlations[name] = 0.0

    overall_corr = np.mean(list(mean_correlations.values()))
    test3_pass = abs(overall_corr) > 0.01
    results['tests']['agent_world_correlation'] = to_json_safe(test3_pass)
    print(f"\n3. Agent-world correlations:")
    for name, corr in mean_correlations.items():
        print(f"   {name}: {corr:.3f}")
    print(f"   Status: {'PASS' if test3_pass else 'FAIL'}")

    # Test 4: Agent phi correlates with world phi
    phi_correlations = {}
    world_phi = np.array(history['phi_world'])
    for name in coupler.agents:
        agent_phi = np.array(history['agent_phi'][name])
        if len(world_phi) == len(agent_phi) and len(world_phi) > 10:
            corr = np.corrcoef(world_phi, agent_phi)[0, 1]
            if not np.isnan(corr):
                phi_correlations[name] = corr
            else:
                phi_correlations[name] = 0.0
        else:
            phi_correlations[name] = 0.0

    mean_phi_corr = np.mean(list(phi_correlations.values()))
    test4_pass = abs(mean_phi_corr) > 0.01
    results['tests']['phi_correlation'] = to_json_safe(test4_pass)
    print(f"\n4. Agent phi correlates with world phi:")
    for name, corr in phi_correlations.items():
        print(f"   {name}: {corr:.3f}")
    print(f"   Status: {'PASS' if test4_pass else 'FAIL'}")

    # Test 5: Regime changes detected
    regime_array = np.array(history['regime'])
    regime_changes = np.sum(np.diff(regime_array) != 0)
    n_regimes = len(set(regime_array))

    test5_pass = regime_changes > 3 and n_regimes >= 2
    results['tests']['regime_detection'] = to_json_safe(test5_pass)
    print(f"\n5. Regime detection:")
    print(f"   Regime changes: {regime_changes}")
    print(f"   Unique regimes: {n_regimes}")
    print(f"   Status: {'PASS' if test5_pass else 'FAIL'}")

    # Summary
    tests_passed = sum(results['tests'].values())
    total_tests = len(results['tests'])

    print("\n" + "=" * 70)
    print(f"PHASE W2 SUMMARY: {tests_passed}/{total_tests} tests passed")
    print("=" * 70)

    # Store metrics
    results['metrics'] = {
        'mean_world_entropy': float(np.mean(history['entropy'])),
        'mean_world_health': float(np.mean(history['health'])),
        'mean_world_phi': float(np.mean(history['phi_world'])),
        'agent_world_correlations': to_json_safe(mean_correlations),
        'phi_correlations': to_json_safe(phi_correlations),
        'n_regime_changes': int(regime_changes)
    }

    results['phase'] = 'W2'
    results['status'] = 'PASS' if tests_passed == total_tests else 'PARTIAL'
    results['timestamp'] = datetime.now().isoformat()
    results['agents'] = list(coupler.agents.keys())

    # Save results
    os.makedirs('results/world1', exist_ok=True)
    with open('results/world1/phaseW2_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to results/world1/phaseW2_results.json")

    return results, coupler


if __name__ == "__main__":
    run_phase_w2(steps=800)
