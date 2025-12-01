"""
Phase W3: Long Life in WORLD-1 with Quantum Games

NEO, EVA, ALEX, ADAM, IRIS living in WORLD-1 AND playing
quantum coalition games to gain experience.

The complete cycle:
1. Agents observe WORLD-1
2. Agents play quantum coalition game
3. Game payoffs affect agent drives
4. Agent actions affect WORLD-1
5. WORLD-1 evolves
6. Repeat
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass

from world1.world1_core import World1Core
from world1.world1_observation import ObservationProjector
from world1.world1_actions import ActionMapper
from world1.world1_metrics import WorldMetrics
from world1.world1_regimes import RegimeDetector

# Import autonomous agents
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments'))
from autonomous_life import AutonomousAgent

# Import quantum game
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'quantum_game', 'endogenous'))
from coalition_game_qg1 import CoalitionGameQG1, AgentGameState


@dataclass
class GameExperience:
    """Experience gained from playing."""
    total_games: int = 0
    total_payoff: float = 0.0
    coalitions_formed: int = 0
    wins: int = 0
    cooperation_rate: float = 0.0


class World1GameIntegration:
    """
    Full integration: WORLD-1 + Quantum Games + 5 Agents.

    Agents live in the world AND play games to gain experience.
    """

    def __init__(self, agent_dim: int = 6):
        """Initialize the complete system."""
        self.agent_dim = agent_dim

        # Create WORLD-1
        self.world = World1Core(n_fields=4, n_entities=5, n_resources=3, n_modes=3)

        # Create agents with personalities
        self.agents: Dict[str, AutonomousAgent] = {}
        self._create_agents()

        # World coupling systems
        self.observer = ObservationProjector(self.world.D, agent_dim)
        self.actor = ActionMapper(self.world.D, agent_dim)

        for name, agent in self.agents.items():
            self.observer.register_agent(name, agent.z)
            self.actor.register_agent(name, agent.z)

        # World metrics
        self.world_metrics = WorldMetrics(self.world.D)
        self.regime_detector = RegimeDetector(self.world.D)

        # Create quantum game with 5 agents
        agent_names = list(self.agents.keys())
        self.game = CoalitionGameQG1(agent_names=agent_names, dim=agent_dim)
        self._sync_agents_to_game()

        # Experience tracking per agent
        self.experience: Dict[str, GameExperience] = {
            name: GameExperience() for name in self.agents
        }

        # History
        self.t = 0
        self.history = {
            'world_entropy': [],
            'world_health': [],
            'world_phi': [],
            'regime': [],
            'game_payoffs': [],
            'entanglement': [],
            'agent_integration': {name: [] for name in self.agents},
            'agent_crisis': {name: [] for name in self.agents}
        }

        # Game frequency (play every N steps)
        self.game_frequency = 10

    def _create_agents(self):
        """Create the five agents with distinct personalities."""
        personalities = {
            'NEO': {'stability': 0.28, 'connection': 0.15, 'exploration': 0.12, 'integration': 0.18},
            'EVA': {'stability': 0.15, 'connection': 0.28, 'exploration': 0.15, 'integration': 0.15},
            'ALEX': {'stability': 0.12, 'connection': 0.13, 'exploration': 0.28, 'integration': 0.15},
            'ADAM': {'stability': 0.14, 'connection': 0.14, 'exploration': 0.14, 'integration': 0.30},
            'IRIS': {'stability': 0.16, 'connection': 0.18, 'exploration': 0.16, 'integration': 0.18}
        }

        for name, traits in personalities.items():
            agent = AutonomousAgent(name, self.agent_dim)
            # z = [entropy, neg_surprise, novelty, stability, integration, otherness]
            z = np.array([
                traits['exploration'] * 0.8,  # entropy
                0.12,                          # neg_surprise
                traits['exploration'],         # novelty
                traits['stability'],           # stability
                traits['integration'],         # integration
                traits['connection']           # otherness
            ])
            z = z / z.sum()
            agent.z = z.copy()
            agent.identity_core = z.copy()
            self.agents[name] = agent

    def _sync_agents_to_game(self):
        """Sync autonomous agent states to game agents."""
        for name, agent in self.agents.items():
            if name in self.game.agents:
                game_agent = self.game.agents[name]

                # Sync drives
                game_agent.drives = agent.z.copy()

                # Sync crisis state
                game_agent.in_crisis = agent.in_crisis

                # Sync identity
                game_agent.identity = agent.identity_core.copy()

    def _apply_game_to_agents(self, round_data: Dict, payoffs: np.ndarray):
        """Apply game results to agent drives."""
        agent_names = list(self.agents.keys())

        # Learning rate decreases with experience
        base_lr = 0.1 / np.sqrt(self.t + 1)

        for i, name in enumerate(agent_names):
            if i >= len(payoffs):
                continue

            agent = self.agents[name]
            payoff = float(payoffs[i])  # Ensure scalar
            exp = self.experience[name]

            # Update experience
            exp.total_games += 1
            exp.total_payoff += payoff
            if payoff > 0:
                exp.wins += 1

            # Normalize payoff
            mean_payoff = exp.total_payoff / max(1, exp.total_games)
            normalized = (payoff - mean_payoff) / (abs(mean_payoff) + 0.1)

            # Apply to drives
            boost = base_lr * np.tanh(normalized)

            # Positive payoff reinforces current strategy
            if payoff > 0:
                dominant_drive = int(np.argmax(agent.z))
                agent.z[dominant_drive] *= (1 + boost)
            else:
                # Negative payoff: explore other drives
                weakest_drive = int(np.argmin(agent.z))
                agent.z[weakest_drive] *= (1 + abs(boost) * 0.5)

            # Renormalize
            agent.z = np.clip(agent.z, 0.05, None)
            agent.z = agent.z / agent.z.sum()

    def step(self) -> Dict:
        """
        Complete step:
        1. Observe world
        2. Maybe play game
        3. Apply game effects
        4. Act on world
        5. World evolves
        """
        self.t += 1

        # Current world state
        w = self.world.w

        # 1. Agents observe world
        observations = {}
        for name, agent in self.agents.items():
            obs = self.observer.get_observation_with_bias(name, w, agent.z)
            observations[name] = obs

        # 2. Maybe play quantum game
        game_result = None
        if self.t % self.game_frequency == 0:
            # Sync agents to game
            self._sync_agents_to_game()

            # Play one round
            game_round = self.game.play_round()
            game_result = game_round

            # Extract payoffs from metric_deltas
            if hasattr(game_round, 'metric_deltas') and game_round.metric_deltas:
                # Payoff = sum of positive metric changes
                payoffs = []
                for name in self.agents.keys():
                    if name in game_round.metric_deltas:
                        deltas = game_round.metric_deltas[name]
                        # Sum all delta values, handling arrays
                        total = 0.0
                        if deltas:
                            for v in deltas.values():
                                if isinstance(v, np.ndarray):
                                    total += float(np.sum(v))
                                else:
                                    total += float(v)
                        payoffs.append(total)
                    else:
                        payoffs.append(0.0)
                payoffs = np.array(payoffs, dtype=float)

                self._apply_game_to_agents({'round': game_round}, payoffs)
                self.history['game_payoffs'].append(float(np.mean(payoffs)))

                # Compute entanglement from quantum state
                ent = self.game.quantum_state.compute_entanglement() if hasattr(self.game, 'quantum_state') else 0
                self.history['entanglement'].append(float(ent))

        # 3. Process observations and compute actions
        perturbations = {}
        for name, agent in self.agents.items():
            obs = observations[name]

            # Stimulus from observation
            stimulus = np.zeros(self.agent_dim)
            for i in range(min(len(obs), self.agent_dim)):
                stimulus[i] = obs[i % len(obs)] * 0.1

            # Other agents influence
            other_zs = [a.z for n, a in self.agents.items() if n != name]
            other_z = np.mean(other_zs, axis=0)

            # Agent step
            agent.step(stimulus, other_z)

            # Record for systems
            self.observer.record_agent_state(name, agent.z)
            self.actor.record_agent_state(name, agent.z)

            # Compute action
            strategy = self._get_strategy(agent)
            action = self.actor.compute_action_from_drives(name, agent.z, strategy)
            perturbations[name] = self.actor.get_world_perturbation(name, action)

        # 4. World step
        new_w = self.world.step(perturbations)

        # Record world state
        self.observer.record_world_state(new_w)
        self.actor.record_world_state(new_w)

        # Update systems
        if self.t % 10 == 0:
            for name in self.agents:
                self.observer.update_projection(name)
                self.actor.update_mapping(name)

        # 5. Compute metrics
        state = self.world.get_state()
        metrics = self.world_metrics.compute_all(new_w, state.modes)
        regime = self.regime_detector.detect_regime(new_w)

        # Record history
        self.history['world_entropy'].append(metrics.entropy)
        self.history['world_health'].append(metrics.health)
        self.history['world_phi'].append(metrics.phi_world)
        self.history['regime'].append(regime)

        for name, agent in self.agents.items():
            self.history['agent_integration'][name].append(agent.integration)
            self.history['agent_crisis'][name].append(agent.in_crisis)

        return {
            'world_metrics': metrics,
            'regime': regime,
            'game_result': game_result,
            'agent_states': {name: agent.z.copy() for name, agent in self.agents.items()}
        }

    def _get_strategy(self, agent: AutonomousAgent) -> str:
        """Determine agent's strategy from drives."""
        z = agent.z
        dominant = np.argmax(z)

        if dominant in [0, 2]:  # entropy, novelty
            return 'exploration'
        elif dominant == 3:  # stability
            return 'stability'
        elif dominant == 4:  # integration
            return 'integration'
        elif dominant == 5:  # otherness
            return 'connection'
        return 'exploration'

    def run(self, steps: int = 1000) -> Dict:
        """Run the complete simulation."""
        print(f"\nRunning WORLD-1 + Games for {steps} steps...")
        print(f"  Agents: {list(self.agents.keys())}")
        print(f"  Game frequency: every {self.game_frequency} steps")

        for t in range(steps):
            result = self.step()

            if (t + 1) % 200 == 0:
                print(f"\n  t={t+1}:")
                print(f"    World: regime={result['regime']}, "
                      f"health={result['world_metrics'].health:.3f}")

                # Game stats
                if len(self.history['game_payoffs']) > 0:
                    recent_payoffs = self.history['game_payoffs'][-20:]
                    recent_entanglement = self.history['entanglement'][-20:]
                    print(f"    Games: mean_payoff={np.mean(recent_payoffs):.3f}, "
                          f"entanglement={np.mean(recent_entanglement):.3f}")

                # Agent stats
                for name, agent in self.agents.items():
                    exp = self.experience[name]
                    crisis_rate = np.mean(self.history['agent_crisis'][name][-100:]) * 100
                    print(f"    {name}: games={exp.total_games}, "
                          f"wins={exp.wins}, "
                          f"crisis={crisis_rate:.0f}%")

        return self.history

    def get_report(self) -> Dict:
        """Generate comprehensive report."""
        report = {
            't': self.t,
            'world': {
                'mean_entropy': float(np.mean(self.history['world_entropy'])),
                'mean_health': float(np.mean(self.history['world_health'])),
                'mean_phi': float(np.mean(self.history['world_phi'])),
                'n_regime_changes': int(np.sum(np.diff(self.history['regime']) != 0))
            },
            'games': {
                'total_rounds': len(self.history['game_payoffs']),
                'mean_payoff': float(np.mean(self.history['game_payoffs'])) if self.history['game_payoffs'] else 0,
                'mean_entanglement': float(np.mean(self.history['entanglement'])) if self.history['entanglement'] else 0
            },
            'agents': {}
        }

        for name, agent in self.agents.items():
            exp = self.experience[name]
            crisis_history = self.history['agent_crisis'][name]
            integration_history = self.history['agent_integration'][name]

            report['agents'][name] = {
                'total_games': exp.total_games,
                'total_payoff': float(exp.total_payoff),
                'wins': exp.wins,
                'win_rate': float(exp.wins / max(1, exp.total_games)),
                'coalitions': exp.coalitions_formed,
                'crisis_rate': float(np.mean(crisis_history)) if crisis_history else 0,
                'mean_integration': float(np.mean(integration_history)) if integration_history else 0,
                'final_drives': agent.z.tolist()
            }

        return report


def run_phase_w3(steps: int = 1500):
    """
    Run Phase W3: Long life in WORLD-1 with games.

    Tests:
    1. All agents play games
    2. Experience accumulates
    3. Entanglement emerges
    4. World-game correlation
    5. Agent adaptation
    """
    print("=" * 70)
    print("PHASE W3: LONG LIFE IN WORLD-1 WITH QUANTUM GAMES")
    print("=" * 70)

    system = World1GameIntegration(agent_dim=6)
    history = system.run(steps)
    report = system.get_report()

    # Validation
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    results = {'tests': {}, 'report': report}

    # Test 1: All agents played
    games_played = [report['agents'][n]['total_games'] for n in system.agents]
    test1 = all(g > 0 for g in games_played)
    results['tests']['all_played'] = test1
    print(f"\n1. All agents played games:")
    for name in system.agents:
        print(f"   {name}: {report['agents'][name]['total_games']} games")
    print(f"   Status: {'PASS' if test1 else 'FAIL'}")

    # Test 2: Experience accumulated
    total_exp = sum(report['agents'][n]['total_payoff'] for n in system.agents)
    test2 = abs(total_exp) > 0
    results['tests']['experience_accumulated'] = test2
    print(f"\n2. Experience accumulated:")
    print(f"   Total payoff across agents: {total_exp:.2f}")
    print(f"   Status: {'PASS' if test2 else 'FAIL'}")

    # Test 3: Entanglement emerged
    mean_ent = report['games']['mean_entanglement']
    test3 = mean_ent > 0.1
    results['tests']['entanglement_emerged'] = test3
    print(f"\n3. Entanglement emerged:")
    print(f"   Mean entanglement: {mean_ent:.3f}")
    print(f"   Status: {'PASS' if test3 else 'FAIL'}")

    # Test 4: World health maintained
    mean_health = report['world']['mean_health']
    test4 = 0.3 < mean_health < 0.9
    results['tests']['world_healthy'] = test4
    print(f"\n4. World health maintained:")
    print(f"   Mean health: {mean_health:.3f}")
    print(f"   Status: {'PASS' if test4 else 'FAIL'}")

    # Test 5: Some agents won games
    total_wins = sum(report['agents'][n]['wins'] for n in system.agents)
    test5 = total_wins > 10
    results['tests']['agents_won'] = test5
    print(f"\n5. Agents won games:")
    print(f"   Total wins: {total_wins}")
    for name in system.agents:
        wr = report['agents'][name]['win_rate']
        print(f"   {name}: {report['agents'][name]['wins']} wins ({wr*100:.0f}%)")
    print(f"   Status: {'PASS' if test5 else 'FAIL'}")

    # Summary
    tests_passed = sum(results['tests'].values())
    total_tests = len(results['tests'])

    print("\n" + "=" * 70)
    print(f"PHASE W3 SUMMARY: {tests_passed}/{total_tests} tests passed")
    print("=" * 70)

    # Final agent summary
    print("\n📊 AGENT FINAL STATUS:")
    for name in system.agents:
        a = report['agents'][name]
        print(f"  {name}:")
        print(f"    Games: {a['total_games']}, Wins: {a['wins']} ({a['win_rate']*100:.0f}%)")
        print(f"    Integration: {a['mean_integration']:.3f}, Crisis: {a['crisis_rate']*100:.0f}%")

    results['phase'] = 'W3'
    results['status'] = 'PASS' if tests_passed == total_tests else 'PARTIAL'
    results['timestamp'] = datetime.now().isoformat()

    # Save
    os.makedirs('results/world1', exist_ok=True)

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        if isinstance(obj, bool):
            return bool(obj)
        return obj

    results = convert(results)

    with open('results/world1/phaseW3_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to results/world1/phaseW3_results.json")

    return results, system


if __name__ == "__main__":
    run_phase_w3(steps=1500)
