"""
Phase W3 Social: Full Cognitive Integration

NEO, EVA, ALEX, ADAM, IRIS with complete cognition:
- Episodic memory
- Narrative memory
- Temporal tree (proto-future)
- Self-model and Theory of Mind
- Compound goals and planning
- Emergent symbols
- Long-term regulation and metacognition

W3 Social Game:
1. Agents at tables with social energy
2. Endogenous partner selection
3. Quantum coalition game
4. Feedback to cognition
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

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

# Import cognition
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cognition.episodic_memory import EpisodicMemory
from cognition.narrative_memory import NarrativeMemory
from cognition.temporal_tree import TemporalTree
from cognition.self_model import SelfModel, TheoryOfMind
from cognition.compound_goals import CompoundGoals, GoalPlanner
from cognition.emergent_symbols import EmergentSymbols, SymbolGrounding
from cognition.regulation import IntegratedRegulation


@dataclass
class Table:
    """A social table where agents can sit and play."""
    idx: int
    seated: List[str] = field(default_factory=list)
    max_seats: int = 3
    energy: float = 1.0  # Social energy of the table

    def has_space(self) -> bool:
        return len(self.seated) < self.max_seats

    def add(self, name: str):
        if self.has_space():
            self.seated.append(name)

    def remove(self, name: str):
        if name in self.seated:
            self.seated.remove(name)


@dataclass
class SocialEnergy:
    """Social energy state for an agent."""
    current: float = 1.0
    base_rate: float = 0.1  # Recovery rate
    drain_rate: float = 0.05  # Per game drain
    history: List[float] = field(default_factory=list)


class CognitiveAgent:
    """
    Agent with full cognitive architecture.

    Wraps AutonomousAgent with cognition systems.
    """

    def __init__(self, name: str, dim: int = 6, other_names: List[str] = None):
        """Initialize cognitive agent."""
        self.name = name
        self.dim = dim
        self.other_names = other_names or []

        # Core autonomous agent
        self.core = AutonomousAgent(name, dim)

        # Cognitive systems
        self.episodic = EpisodicMemory(z_dim=dim, phi_dim=5, D_dim=dim)
        self.temporal_tree = TemporalTree(z_dim=dim, phi_dim=5, D_dim=dim)
        self.self_model = SelfModel(name, state_dim=dim + 5 + dim)
        self.tom = TheoryOfMind(name, other_names, state_dim=dim + 5 + dim)
        self.goals = CompoundGoals(D_dim=dim)
        self.planner = GoalPlanner(self.goals)
        self.symbols = EmergentSymbols(z_dim=dim, phi_dim=5, D_dim=dim)
        self.regulation = IntegratedRegulation(name, other_names)

        # Narrative built on episodic
        self.narrative = NarrativeMemory(self.episodic)

        # Social state
        self.social_energy = SocialEnergy()
        self.current_table: Optional[int] = None

        # Preferences (learned)
        self.partner_preferences: Dict[str, float] = {n: 0.5 for n in other_names}

        self.t = 0

    @property
    def z(self) -> np.ndarray:
        return self.core.z

    @z.setter
    def z(self, value):
        self.core.z = value

    @property
    def phi(self) -> np.ndarray:
        """Compute phenomenological vector from z and history."""
        # φ = [integration, temporal_change, diversity, stability, depth]
        phi = np.zeros(5)

        # Integration (from core)
        phi[0] = self.core.integration

        # Temporal change
        if len(self.core.z_history) > 1:
            delta = np.linalg.norm(self.z - self.core.z_history[-1])
            phi[1] = delta

        # Diversity (entropy of z)
        z_norm = np.abs(self.z) / (np.sum(np.abs(self.z)) + 1e-8)
        phi[2] = -np.sum(z_norm * np.log(z_norm + 1e-8))

        # Stability (inverse of variance)
        if len(self.core.z_history) > 5:
            recent = np.array(self.core.z_history[-5:])
            phi[3] = 1.0 / (1.0 + np.var(recent))
        else:
            phi[3] = 0.5

        # Depth (time-based)
        phi[4] = min(1.0, self.t / 500)

        return phi

    @property
    def D(self) -> np.ndarray:
        return self.core.z  # Drives = z in this architecture

    @property
    def in_crisis(self) -> bool:
        return self.core.in_crisis

    @property
    def SAGI(self) -> float:
        return self.core.integration  # Use integration as SAGI proxy

    def step(self, stimulus: np.ndarray, other_z: np.ndarray,
             other_states: Dict[str, np.ndarray] = None):
        """
        Full cognitive step.

        1. Core step
        2. Record to episodic memory
        3. Update self-model
        4. Update Theory of Mind
        5. Record symbols
        6. Update regulation
        7. Update narrative
        """
        self.t += 1

        # 1. Core step
        self.core.step(stimulus, other_z)

        # 2. Record to episodic memory
        tau = self.t * (1 + 0.1 * np.linalg.norm(self.phi))  # Subjective time
        self.episodic.record(self.z, self.phi, self.D, tau)

        # 3. Update self-model
        combined_state = np.concatenate([self.z, self.phi, self.D])
        self.self_model.update_model(combined_state)

        # 4. Update Theory of Mind
        if other_states:
            for other_name, other_state in other_states.items():
                if other_name in self.other_names:
                    self.tom.update_model(other_name, other_state)

        # 5. Record to symbols
        self.symbols.record_state(self.z, self.phi, self.D)

        # 6. Update regulation
        self_error = np.linalg.norm(combined_state - self.self_model.predict_self())
        tom_errors = {}
        for other_name in self.other_names:
            if other_states and other_name in other_states:
                predicted = self.tom.predict_other(other_name)
                tom_errors[other_name] = np.linalg.norm(other_states[other_name] - predicted)

        self.regulation.record(self.SAGI, self.in_crisis, self_error, tom_errors)

        # 7. Update narrative periodically
        if self.t % 20 == 0:
            self.narrative.update()

        # 8. Discover goals periodically
        if self.t % 100 == 0 and len(self.episodic.episodes) > 10:
            D_bar = np.mean([e.D_bar for e in self.episodic.get_recent_episodes(10)], axis=0)
            self.goals.record_episode(D_bar, self.SAGI)
            self.goals.discover_goals()

        # 9. Update social energy
        self._update_social_energy()

        # 10. Discover symbols periodically
        if self.t % 50 == 0:
            self.symbols.discover_symbols()

    def _update_social_energy(self):
        """Update social energy endogenously."""
        se = self.social_energy

        # Recovery
        se.current += se.base_rate * (1 - se.current)

        # Cap
        se.current = np.clip(se.current, 0, 1)
        se.history.append(se.current)

        if len(se.history) > 1000:
            se.history = se.history[-1000:]

    def compute_partner_utility(self, other_name: str) -> float:
        """
        Compute utility of partnering with another agent.

        U(A, B) = ToM_accuracy * preference * social_energy
        """
        if other_name not in self.other_names:
            return 0.0

        tom_acc = self.tom.get_tom_accuracy(other_name)
        pref = self.partner_preferences.get(other_name, 0.5)

        utility = tom_acc * pref * self.social_energy.current
        return float(utility)

    def update_preference(self, other_name: str, payoff: float):
        """
        Update partner preference from game outcome.

        pref_new = pref_old + η * (payoff - baseline)
        """
        if other_name not in self.partner_preferences:
            return

        eta = 0.1 / np.sqrt(self.t + 1)
        baseline = 0.0  # Could be mean historical payoff

        delta = eta * (payoff - baseline)
        self.partner_preferences[other_name] += delta
        self.partner_preferences[other_name] = np.clip(
            self.partner_preferences[other_name], 0, 1
        )

    def wants_to_play(self) -> bool:
        """
        Endogenous decision to play.

        play = social_energy > threshold AND not in crisis
        """
        threshold = 0.3 + 0.2 * np.random.random()  # Stochastic threshold
        return self.social_energy.current > threshold and not self.in_crisis

    def plan_action(self) -> str:
        """
        Use planner to select action.

        Generates temporal tree and uses goal planner.
        """
        # Generate tree
        self.temporal_tree.record_state(self.z, self.in_crisis)
        root = self.temporal_tree.generate_tree(self.z, self.D, depth=2, branching=3)

        # Get best branch
        best = self.temporal_tree.get_best_branch()
        if best:
            return best.operator_used
        return 'exploration'

    def get_combined_state(self) -> np.ndarray:
        """Get combined state for Theory of Mind."""
        return np.concatenate([self.z, self.phi, self.D])

    def get_cognitive_report(self) -> Dict:
        """Get report on cognitive state."""
        return {
            'name': self.name,
            't': self.t,
            'episodic': self.episodic.get_statistics(),
            'narrative': self.narrative.get_narrative_summary(),
            'temporal_tree': self.temporal_tree.get_statistics(),
            'self_model': self.self_model.get_statistics(),
            'tom': self.tom.get_statistics(),
            'goals': self.goals.get_statistics(),
            'symbols': self.symbols.get_statistics(),
            'regulation': self.regulation.get_statistics(),
            'social_energy': self.social_energy.current
        }


class W3SocialSystem:
    """
    W3 Social: Tables, partner selection, games, feedback.

    Complete integration of:
    - WORLD-1 environment
    - Cognitive agents
    - Social tables with energy
    - Endogenous partner selection
    - Quantum coalition games
    """

    def __init__(self, n_tables: int = 3, agent_dim: int = 6):
        """Initialize W3 social system."""
        self.agent_dim = agent_dim
        self.n_tables = n_tables

        # Create WORLD-1
        self.world = World1Core(n_fields=4, n_entities=5, n_resources=3, n_modes=3)

        # Create cognitive agents
        agent_names = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
        self.agents: Dict[str, CognitiveAgent] = {}

        for name in agent_names:
            others = [n for n in agent_names if n != name]
            self.agents[name] = CognitiveAgent(name, agent_dim, others)

        # Set initial personalities
        self._set_personalities()

        # World coupling
        self.observer = ObservationProjector(self.world.D, agent_dim)
        self.actor = ActionMapper(self.world.D, agent_dim)

        for name, agent in self.agents.items():
            self.observer.register_agent(name, agent.z)
            self.actor.register_agent(name, agent.z)

        # World metrics
        self.world_metrics = WorldMetrics(self.world.D)
        self.regime_detector = RegimeDetector(self.world.D)

        # Social tables
        self.tables: List[Table] = [Table(idx=i) for i in range(n_tables)]

        # Game (created per round with selected players)
        self.current_game: Optional[CoalitionGameQG1] = None

        # Statistics
        self.t = 0
        self.games_played = 0
        self.history = {
            'world_entropy': [],
            'world_health': [],
            'games_per_step': [],
            'mean_social_energy': [],
            'coalition_sizes': [],
            'agent_SAGI': {name: [] for name in agent_names}
        }

    def _set_personalities(self):
        """Set initial drive profiles."""
        profiles = {
            'NEO': np.array([0.12, 0.12, 0.12, 0.28, 0.18, 0.18]),
            'EVA': np.array([0.15, 0.10, 0.15, 0.15, 0.15, 0.30]),
            'ALEX': np.array([0.25, 0.08, 0.25, 0.12, 0.15, 0.15]),
            'ADAM': np.array([0.14, 0.14, 0.14, 0.14, 0.30, 0.14]),
            'IRIS': np.array([0.16, 0.16, 0.16, 0.16, 0.18, 0.18])
        }

        for name, z in profiles.items():
            z = z / z.sum()
            self.agents[name].core.z = z.copy()
            self.agents[name].core.identity_core = z.copy()

    def _select_table(self, agent_name: str) -> Optional[int]:
        """
        Endogenous table selection.

        U(table) = Σ_j U(A, j) for j at table
        Select table with highest utility.
        """
        agent = self.agents[agent_name]

        if not agent.wants_to_play():
            return None

        utilities = []
        for table in self.tables:
            if not table.has_space():
                utilities.append(-float('inf'))
                continue

            # Sum utility of seated agents
            u = 0.0
            for seated_name in table.seated:
                u += agent.compute_partner_utility(seated_name)

            # Empty table has base utility
            if len(table.seated) == 0:
                u = 0.3 * agent.social_energy.current

            utilities.append(u)

        if max(utilities) <= 0:
            return None

        # Softmax selection
        utilities = np.array(utilities)
        utilities = np.where(utilities == -float('inf'), -1e10, utilities)

        beta = 1.0 / (np.std(utilities) + 0.1)
        probs = np.exp(beta * utilities)
        probs = probs / probs.sum()

        selected = np.random.choice(len(self.tables), p=probs)
        return selected

    def _form_coalitions(self) -> List[List[str]]:
        """
        Form coalitions from table seatings.

        Each table with 2+ agents forms a coalition.
        """
        coalitions = []

        for table in self.tables:
            if len(table.seated) >= 2:
                coalitions.append(table.seated.copy())

        return coalitions

    def _play_coalition_game(self, coalition: List[str]) -> Dict[str, float]:
        """
        Play quantum coalition game with selected agents.

        Returns payoffs for each agent.
        """
        # Create game for this coalition
        game = CoalitionGameQG1(agent_names=coalition, dim=self.agent_dim)

        # Sync agent states
        for name in coalition:
            agent = self.agents[name]
            if name in game.agents:
                game_agent = game.agents[name]
                game_agent.drives = agent.z.copy()
                game_agent.in_crisis = agent.in_crisis
                game_agent.identity = agent.core.identity_core.copy()

        # Play round
        round_result = game.play_round()

        # Extract payoffs
        payoffs = {}
        if hasattr(round_result, 'metric_deltas') and round_result.metric_deltas:
            for name in coalition:
                if name in round_result.metric_deltas:
                    deltas = round_result.metric_deltas[name]
                    total = 0.0
                    if deltas:
                        for v in deltas.values():
                            if isinstance(v, np.ndarray):
                                total += float(np.sum(v))
                            else:
                                total += float(v)
                    payoffs[name] = total
                else:
                    payoffs[name] = 0.0
        else:
            for name in coalition:
                payoffs[name] = 0.0

        return payoffs

    def _apply_game_feedback(self, coalition: List[str], payoffs: Dict[str, float]):
        """
        Apply game outcomes to agents.

        1. Update drives based on payoff
        2. Drain social energy
        3. Update partner preferences
        """
        for name in coalition:
            agent = self.agents[name]
            payoff = payoffs.get(name, 0.0)

            # 1. Update drives
            lr = 0.1 / np.sqrt(agent.t + 1)
            normalized_payoff = np.tanh(payoff)

            if payoff > 0:
                dominant = int(np.argmax(agent.z))
                agent.core.z[dominant] *= (1 + lr * normalized_payoff)
            else:
                weakest = int(np.argmin(agent.z))
                agent.core.z[weakest] *= (1 + lr * abs(normalized_payoff) * 0.5)

            agent.core.z = np.clip(agent.core.z, 0.05, None)
            agent.core.z = agent.core.z / agent.core.z.sum()

            # 2. Drain social energy
            agent.social_energy.current -= agent.social_energy.drain_rate
            agent.social_energy.current = max(0, agent.social_energy.current)

            # 3. Update partner preferences
            for partner in coalition:
                if partner != name:
                    agent.update_preference(partner, payoff)

    def step(self) -> Dict:
        """
        Complete W3 social step.

        1. Clear tables
        2. Agents select tables
        3. Form coalitions
        4. Play games
        5. Apply feedback
        6. Cognitive update
        7. World step
        """
        self.t += 1

        # 1. Clear tables
        for table in self.tables:
            table.seated = []

        # 2. Agents select tables (random order)
        agent_order = list(self.agents.keys())
        np.random.shuffle(agent_order)

        for name in agent_order:
            table_idx = self._select_table(name)
            if table_idx is not None:
                self.tables[table_idx].add(name)
                self.agents[name].current_table = table_idx
            else:
                self.agents[name].current_table = None

        # 3. Form coalitions
        coalitions = self._form_coalitions()
        self.history['coalition_sizes'].append([len(c) for c in coalitions])

        # 4. Play games
        games_this_step = 0
        for coalition in coalitions:
            payoffs = self._play_coalition_game(coalition)
            self._apply_game_feedback(coalition, payoffs)
            games_this_step += 1
            self.games_played += 1

        self.history['games_per_step'].append(games_this_step)

        # 5. Cognitive update for all agents
        w = self.world.w
        other_states = {
            name: agent.get_combined_state()
            for name, agent in self.agents.items()
        }

        for name, agent in self.agents.items():
            # Observation
            obs = self.observer.get_observation_with_bias(name, w, agent.z)

            # Stimulus
            stimulus = np.zeros(self.agent_dim)
            for i in range(min(len(obs), self.agent_dim)):
                stimulus[i] = obs[i % len(obs)] * 0.1

            # Other agent z for core step
            other_zs = [a.z for n, a in self.agents.items() if n != name]
            other_z = np.mean(other_zs, axis=0)

            # Other states for ToM
            other_states_for_agent = {
                n: s for n, s in other_states.items() if n != name
            }

            # Full cognitive step
            agent.step(stimulus, other_z, other_states_for_agent)

            # Record for coupling systems
            self.observer.record_agent_state(name, agent.z)
            self.actor.record_agent_state(name, agent.z)

            # Record SAGI
            self.history['agent_SAGI'][name].append(agent.SAGI)

        # 6. World step
        perturbations = {}
        for name, agent in self.agents.items():
            action_type = agent.plan_action()
            action = self.actor.compute_action_from_drives(name, agent.z, action_type)
            perturbations[name] = self.actor.get_world_perturbation(name, action)

        new_w = self.world.step(perturbations)
        self.observer.record_world_state(new_w)
        self.actor.record_world_state(new_w)

        # Update coupling periodically
        if self.t % 10 == 0:
            for name in self.agents:
                self.observer.update_projection(name)
                self.actor.update_mapping(name)

        # 7. Compute metrics
        state = self.world.get_state()
        metrics = self.world_metrics.compute_all(new_w, state.modes)
        regime = self.regime_detector.detect_regime(new_w)

        # Record history
        self.history['world_entropy'].append(metrics.entropy)
        self.history['world_health'].append(metrics.health)
        self.history['mean_social_energy'].append(
            np.mean([a.social_energy.current for a in self.agents.values()])
        )

        return {
            'world_metrics': metrics,
            'regime': regime,
            'coalitions': coalitions,
            'games_played': games_this_step
        }

    def run(self, steps: int = 1000) -> Dict:
        """Run W3 social simulation."""
        print(f"\nRunning W3 SOCIAL for {steps} steps...")
        print(f"  Agents: {list(self.agents.keys())}")
        print(f"  Tables: {self.n_tables}")

        for t in range(steps):
            result = self.step()

            if (t + 1) % 200 == 0:
                print(f"\n  t={t+1}:")
                print(f"    World: regime={result['regime']}, "
                      f"health={result['world_metrics'].health:.3f}")
                print(f"    Games: total={self.games_played}, "
                      f"this_step={result['games_played']}")
                print(f"    Mean social energy: "
                      f"{np.mean(self.history['mean_social_energy'][-50:]):.3f}")

                # Agent summaries
                for name, agent in self.agents.items():
                    report = agent.get_cognitive_report()
                    print(f"    {name}: SAGI={agent.SAGI:.3f}, "
                          f"episodes={report['episodic']['n_episodes']}, "
                          f"symbols={report['symbols']['n_symbols']}")

        return self.history

    def get_report(self) -> Dict:
        """Generate comprehensive report."""
        report = {
            't': self.t,
            'games_played': self.games_played,
            'world': {
                'mean_entropy': float(np.mean(self.history['world_entropy'])),
                'mean_health': float(np.mean(self.history['world_health']))
            },
            'social': {
                'mean_energy': float(np.mean(self.history['mean_social_energy'])),
                'games_per_step': float(np.mean(self.history['games_per_step'])),
                'mean_coalition_size': float(np.mean([
                    np.mean(cs) if cs else 0
                    for cs in self.history['coalition_sizes']
                ]))
            },
            'agents': {}
        }

        for name, agent in self.agents.items():
            cog = agent.get_cognitive_report()
            report['agents'][name] = {
                'mean_SAGI': float(np.mean(self.history['agent_SAGI'][name][-100:])),
                'social_energy': agent.social_energy.current,
                'n_episodes': cog['episodic']['n_episodes'],
                'n_symbols': cog['symbols']['n_symbols'],
                'n_goals': cog['goals']['n_goals'],
                'self_model_error': cog['self_model'].get('mean_error', 0),
                'metacognition': cog['regulation']['metacognition']['MC'],
                'regulation_action': cog['regulation']['action'],
                'partner_preferences': agent.partner_preferences
            }

        return report


def run_w3_social(steps: int = 1000):
    """
    Run W3 Social with full cognitive integration.

    Tests:
    1. Cognitive systems working
    2. Social selection functioning
    3. Games played
    4. Adaptation occurring
    5. Symbols emerging
    """
    print("=" * 70)
    print("W3 SOCIAL: COGNITIVE AGENTS WITH ENDOGENOUS PARTNER SELECTION")
    print("=" * 70)

    system = W3SocialSystem(n_tables=3, agent_dim=6)
    history = system.run(steps)
    report = system.get_report()

    # Validation
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    results = {'tests': {}, 'report': report}

    # Test 1: Episodic memory working
    episodes = [report['agents'][n]['n_episodes'] for n in system.agents]
    test1 = all(e > 0 for e in episodes)
    results['tests']['episodic_working'] = test1
    print(f"\n1. Episodic memory working:")
    for name in system.agents:
        print(f"   {name}: {report['agents'][name]['n_episodes']} episodes")
    print(f"   Status: {'PASS' if test1 else 'FAIL'}")

    # Test 2: Symbols emerged
    symbols = [report['agents'][n]['n_symbols'] for n in system.agents]
    test2 = sum(symbols) > 0
    results['tests']['symbols_emerged'] = test2
    print(f"\n2. Symbols emerged:")
    for name in system.agents:
        print(f"   {name}: {report['agents'][name]['n_symbols']} symbols")
    print(f"   Status: {'PASS' if test2 else 'FAIL'}")

    # Test 3: Games played
    test3 = system.games_played > 10
    results['tests']['games_played'] = test3
    print(f"\n3. Games played:")
    print(f"   Total: {system.games_played}")
    print(f"   Per step: {report['social']['games_per_step']:.2f}")
    print(f"   Status: {'PASS' if test3 else 'FAIL'}")

    # Test 4: Partner preferences adapted
    prefs_changed = False
    for name, agent in system.agents.items():
        for other, pref in agent.partner_preferences.items():
            if abs(pref - 0.5) > 0.05:
                prefs_changed = True
    test4 = prefs_changed
    results['tests']['preferences_adapted'] = test4
    print(f"\n4. Partner preferences adapted:")
    for name in system.agents:
        prefs = report['agents'][name]['partner_preferences']
        print(f"   {name}: {prefs}")
    print(f"   Status: {'PASS' if test4 else 'FAIL'}")

    # Test 5: Metacognition functioning (any computation occurred)
    mc_values = [report['agents'][n]['metacognition'] for n in system.agents]
    test5 = all(0 <= mc <= 1 for mc in mc_values) and len([m for m in mc_values if m > 0]) >= 3
    results['tests']['metacognition_working'] = test5
    print(f"\n5. Metacognition functioning:")
    for name in system.agents:
        print(f"   {name}: MC={report['agents'][name]['metacognition']:.3f}")
    print(f"   Status: {'PASS' if test5 else 'FAIL'}")

    # Summary
    tests_passed = sum(results['tests'].values())
    total_tests = len(results['tests'])

    print("\n" + "=" * 70)
    print(f"W3 SOCIAL SUMMARY: {tests_passed}/{total_tests} tests passed")
    print("=" * 70)

    # Cognitive summary
    print("\n🧠 COGNITIVE STATUS:")
    for name in system.agents:
        a = report['agents'][name]
        print(f"  {name}:")
        print(f"    SAGI: {a['mean_SAGI']:.3f}, Episodes: {a['n_episodes']}, "
              f"Symbols: {a['n_symbols']}")
        print(f"    Goals: {a['n_goals']}, MC: {a['metacognition']:.3f}, "
              f"Action: {a['regulation_action']}")

    results['phase'] = 'W3_SOCIAL'
    results['status'] = 'PASS' if tests_passed == total_tests else 'PARTIAL'
    results['timestamp'] = datetime.now().isoformat()

    # Save
    os.makedirs('results/world1', exist_ok=True)

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

    with open('results/world1/w3_social_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to results/world1/w3_social_results.json")

    return results, system


if __name__ == "__main__":
    run_w3_social(steps=1000)
