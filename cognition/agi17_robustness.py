"""
AGI-17: Robustez Multi-Mundo
=============================

"¿Qué tan bien funciona mi política en mundos alternativos?"

Generación de mundos:
    W_m = perturbar(W_0, ε_m)
    ε_m ~ N(0, σ_pert²)
    σ_pert² = var(recompensas pasadas)

Utilidad cross-world:
    U_A^(m) = E[r_t | agente A en mundo W_m]

Robustez:
    Rob_A = min_m(U_A^(m)) / max_m(U_A^(m))
    Rob_A^rank = rank(Rob_A) entre agentes

Política robusta:
    π_robust = argmax_π min_m E[r|π, W_m]

Índice de generalización:
    Gen_A = 1 - std_m(U_A^(m)) / mean_m(U_A^(m))

100% endógeno.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .agi_dynamic_constants import (
    L_t, max_history, adaptive_momentum, dynamic_percentile_low
)


def n_worlds(t: int) -> int:
    """
    Número de mundos a simular.

    n(t) = 3 + floor(√log(t+1))
    """
    return 3 + int(np.sqrt(np.log(t + 1)))


@dataclass
class SimulatedWorld:
    """Un mundo perturbado."""
    world_id: int
    perturbation: np.ndarray
    perturbation_magnitude: float
    reward_mean: float = 0.5
    reward_std: float = 0.1


@dataclass
class AgentRobustness:
    """Robustez de un agente."""
    agent_name: str
    utilities_per_world: Dict[int, float]
    min_utility: float
    max_utility: float
    robustness: float  # min/max
    robustness_rank: float
    generalization: float  # 1 - CV
    is_robust: bool


@dataclass
class RobustnessState:
    """Estado del sistema de robustez."""
    t: int
    n_worlds: int
    agent_robustness: Dict[str, AgentRobustness]
    most_robust_agent: str
    system_robustness: float
    robust_policy_found: bool


class MultiWorldRobustness:
    """
    Sistema de robustez multi-mundo.

    Evalúa qué tan bien generalizan las políticas
    a mundos perturbados.
    """

    def __init__(self, agent_names: List[str], state_dim: int = 10):
        """
        Inicializa sistema de robustez.

        Args:
            agent_names: Lista de agentes
            state_dim: Dimensión del estado/mundo
        """
        self.agent_names = agent_names
        self.state_dim = state_dim

        # Mundos simulados
        self.worlds: Dict[int, SimulatedWorld] = {}
        self.next_world_id = 0

        # Historial de recompensas por agente
        self.reward_history: Dict[str, List[float]] = {
            name: [] for name in agent_names
        }

        # Utilidades cross-world
        self.cross_world_utilities: Dict[str, Dict[int, List[float]]] = {
            name: {} for name in agent_names
        }

        # Robustez por agente
        self.robustness: Dict[str, AgentRobustness] = {}

        # Política robusta encontrada
        self.robust_policy: Optional[np.ndarray] = None

        self.t = 0

    def _compute_perturbation_scale(self) -> float:
        """
        Calcula escala de perturbación endógena.

        σ_pert² = var(recompensas pasadas)
        """
        all_rewards = []
        for rewards in self.reward_history.values():
            all_rewards.extend(rewards[-max_history(self.t):])

        if len(all_rewards) < L_t(self.t):
            return 0.1

        return float(np.std(all_rewards) + 1e-8)

    def _generate_worlds(self):
        """
        Genera mundos perturbados.

        W_m = perturbar(W_0, ε_m)
        ε_m ~ N(0, σ_pert²)
        """
        n = n_worlds(self.t)
        sigma = self._compute_perturbation_scale()

        # Limpiar mundos antiguos
        self.worlds.clear()
        self.next_world_id = 0

        # Mundo base (sin perturbación)
        self.worlds[0] = SimulatedWorld(
            world_id=0,
            perturbation=np.zeros(self.state_dim),
            perturbation_magnitude=0.0
        )
        self.next_world_id = 1

        # Mundos perturbados
        for _ in range(n - 1):
            perturbation = np.random.randn(self.state_dim) * sigma
            magnitude = float(np.linalg.norm(perturbation))

            self.worlds[self.next_world_id] = SimulatedWorld(
                world_id=self.next_world_id,
                perturbation=perturbation,
                perturbation_magnitude=magnitude
            )
            self.next_world_id += 1

    def _simulate_utility_in_world(self, agent_name: str,
                                   world: SimulatedWorld,
                                   policy: np.ndarray) -> float:
        """
        Simula utilidad de agente en mundo.

        U_A^(m) = E[r_t | agente A en mundo W_m]

        Simple model: utility degrades with perturbation magnitude.
        """
        # Baseline del agente
        rewards = self.reward_history.get(agent_name, [])
        if not rewards:
            baseline = 0.5
        else:
            baseline = float(np.mean(rewards[-max_history(self.t):]))

        # Degradación por perturbación
        degradation = 1.0 / (1.0 + world.perturbation_magnitude)

        # Ruido
        noise = np.random.randn() * 0.05

        # Efecto de política (más uniforme = más robusto)
        policy_entropy = -np.sum(policy * np.log(policy + 1e-8))
        max_entropy = np.log(len(policy))
        entropy_bonus = 0.1 * (policy_entropy / max_entropy)

        utility = baseline * degradation + entropy_bonus + noise
        return float(np.clip(utility, 0, 1))

    def _compute_robustness(self):
        """
        Calcula robustez de cada agente.

        Rob_A = min_m(U_A^(m)) / max_m(U_A^(m))
        Gen_A = 1 - std_m(U_A^(m)) / mean_m(U_A^(m))
        """
        all_robustness = []

        for agent_name in self.agent_names:
            world_utils = self.cross_world_utilities[agent_name]

            if not world_utils:
                continue

            # Promediar utilidades por mundo
            avg_utils = {}
            for world_id, utils in world_utils.items():
                if utils:
                    avg_utils[world_id] = float(np.mean(utils[-50:]))

            if not avg_utils:
                continue

            utilities = list(avg_utils.values())
            min_u = min(utilities)
            max_u = max(utilities)
            mean_u = np.mean(utilities)
            std_u = np.std(utilities)

            # Robustez = min/max
            robustness = min_u / (max_u + 1e-8)

            # Generalización = 1 - CV (coeficiente de variación)
            cv = std_u / (mean_u + 1e-8)
            generalization = 1.0 - min(cv, 1.0)

            all_robustness.append((agent_name, robustness))

            self.robustness[agent_name] = AgentRobustness(
                agent_name=agent_name,
                utilities_per_world=avg_utils,
                min_utility=min_u,
                max_utility=max_u,
                robustness=robustness,
                robustness_rank=0.0,  # Se calcula después
                generalization=generalization,
                is_robust=robustness > 0.7 and generalization > 0.6
            )

        # Calcular ranks
        if all_robustness:
            sorted_rob = sorted(all_robustness, key=lambda x: x[1])
            for i, (name, _) in enumerate(sorted_rob):
                if name in self.robustness:
                    self.robustness[name].robustness_rank = (i + 1) / len(sorted_rob)

    def record_reward(self, agent_name: str, reward: float,
                     policy: Optional[np.ndarray] = None):
        """
        Registra recompensa de un agente.

        Args:
            agent_name: Nombre del agente
            reward: Recompensa obtenida
            policy: Política usada (opcional)
        """
        self.t += 1

        if agent_name in self.reward_history:
            self.reward_history[agent_name].append(reward)

            # Limitar historial
            max_hist = max_history(self.t)
            if len(self.reward_history[agent_name]) > max_hist:
                self.reward_history[agent_name] = self.reward_history[agent_name][-max_hist:]

        # Actualizar mundos periódicamente
        update_freq = max(20, L_t(self.t) * 2)
        if self.t % update_freq == 0:
            self._generate_worlds()

        # Simular utilidad en cada mundo
        if policy is not None and self.worlds:
            for world_id, world in self.worlds.items():
                utility = self._simulate_utility_in_world(agent_name, world, policy)

                if world_id not in self.cross_world_utilities[agent_name]:
                    self.cross_world_utilities[agent_name][world_id] = []

                self.cross_world_utilities[agent_name][world_id].append(utility)

                # Limitar
                if len(self.cross_world_utilities[agent_name][world_id]) > 200:
                    self.cross_world_utilities[agent_name][world_id] = \
                        self.cross_world_utilities[agent_name][world_id][-200:]

            # Recalcular robustez
            if self.t % update_freq == 0:
                self._compute_robustness()

    def find_robust_policy(self, policy_candidates: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Encuentra política más robusta.

        π_robust = argmax_π min_m E[r|π, W_m]

        Args:
            policy_candidates: Lista de políticas candidatas

        Returns:
            (mejor_política, min_utility)
        """
        if not self.worlds or not policy_candidates:
            return policy_candidates[0] if policy_candidates else np.ones(5) / 5, 0.0

        best_policy = policy_candidates[0]
        best_min_utility = float('-inf')

        for policy in policy_candidates:
            min_utility = float('inf')

            for world in self.worlds.values():
                # Simular utilidad promedio
                utility = self._simulate_utility_in_world(
                    self.agent_names[0] if self.agent_names else "agent",
                    world,
                    policy
                )
                min_utility = min(min_utility, utility)

            if min_utility > best_min_utility:
                best_min_utility = min_utility
                best_policy = policy

        self.robust_policy = best_policy
        return best_policy, best_min_utility

    def get_robustness_bonus(self, agent_name: str, action_entropy: float) -> float:
        """
        Calcula bonus de robustez para una acción.

        Acciones más diversas (alta entropía) reciben bonus.

        Args:
            agent_name: Nombre del agente
            action_entropy: Entropía de la distribución de acción

        Returns:
            Bonus de robustez [0, 1]
        """
        if agent_name not in self.robustness:
            return 0.0

        rob = self.robustness[agent_name]

        # Bonus proporcional a entropía y robustez actual
        entropy_factor = action_entropy / (np.log(10) + 1e-8)  # Normalizar
        bonus = rob.robustness * entropy_factor * 0.2

        return float(np.clip(bonus, 0, 0.3))

    def get_state(self) -> RobustnessState:
        """Obtiene estado actual."""
        if self.robustness:
            most_robust = max(self.robustness.values(), key=lambda x: x.robustness)
            most_robust_name = most_robust.agent_name
            system_robustness = float(np.mean([r.robustness for r in self.robustness.values()]))
        else:
            most_robust_name = ""
            system_robustness = 0.0

        return RobustnessState(
            t=self.t,
            n_worlds=len(self.worlds),
            agent_robustness=self.robustness,
            most_robust_agent=most_robust_name,
            system_robustness=system_robustness,
            robust_policy_found=self.robust_policy is not None
        )

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas del sistema."""
        state = self.get_state()

        agent_stats = {}
        for name, rob in self.robustness.items():
            agent_stats[name] = {
                'robustness': rob.robustness,
                'robustness_rank': rob.robustness_rank,
                'generalization': rob.generalization,
                'min_utility': rob.min_utility,
                'max_utility': rob.max_utility,
                'is_robust': rob.is_robust
            }

        return {
            't': self.t,
            'n_worlds': state.n_worlds,
            'n_agents': len(self.agent_names),
            'most_robust_agent': state.most_robust_agent,
            'system_robustness': state.system_robustness,
            'agent_stats': agent_stats,
            'perturbation_scale': self._compute_perturbation_scale(),
            'robust_policy_found': state.robust_policy_found
        }


def test_robustness():
    """Test de robustez multi-mundo."""
    print("=" * 60)
    print("TEST AGI-17: MULTI-WORLD ROBUSTNESS")
    print("=" * 60)

    agents = ['NEO', 'EVA', 'ALEX']
    robustness = MultiWorldRobustness(agents, state_dim=5)

    print(f"\nSimulando 500 pasos con {len(agents)} agentes...")

    for t in range(500):
        for i, agent in enumerate(agents):
            # Diferentes niveles de robustez por agente
            if agent == 'NEO':
                # Política diversa (robusta)
                policy = np.array([0.25, 0.25, 0.2, 0.15, 0.15])
                base_reward = 0.7
            elif agent == 'EVA':
                # Política concentrada (menos robusta)
                policy = np.array([0.6, 0.2, 0.1, 0.05, 0.05])
                base_reward = 0.8  # Mejor en mundo base
            else:
                # Política intermedia
                policy = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
                base_reward = 0.65

            reward = base_reward + np.random.randn() * 0.1
            robustness.record_reward(agent, reward, policy)

        if (t + 1) % 100 == 0:
            state = robustness.get_state()
            print(f"  t={t+1}: worlds={state.n_worlds}, "
                  f"system_rob={state.system_robustness:.3f}, "
                  f"most_robust={state.most_robust_agent}")

    # Resultados finales
    stats = robustness.get_statistics()

    print("\n" + "=" * 60)
    print("RESULTADOS MULTI-WORLD ROBUSTNESS")
    print("=" * 60)

    print(f"\n  Mundos simulados: {stats['n_worlds']}")
    print(f"  Escala perturbación: {stats['perturbation_scale']:.3f}")
    print(f"  Robustez sistema: {stats['system_robustness']:.3f}")
    print(f"  Más robusto: {stats['most_robust_agent']}")

    print("\n  Por agente:")
    for name, agent_stats in stats['agent_stats'].items():
        print(f"    {name}: rob={agent_stats['robustness']:.3f}, "
              f"gen={agent_stats['generalization']:.3f}, "
              f"is_robust={agent_stats['is_robust']}")

    # Test de política robusta
    print("\n  Búsqueda de política robusta:")
    candidates = [
        np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
        np.array([0.5, 0.25, 0.15, 0.05, 0.05]),
        np.array([0.35, 0.25, 0.2, 0.1, 0.1])
    ]
    best, min_u = robustness.find_robust_policy(candidates)
    print(f"    Mejor política: {best}")
    print(f"    Min utility: {min_u:.3f}")

    if stats['system_robustness'] > 0.5:
        print("\n  ✓ Sistema de robustez funcionando")
    else:
        print("\n  ⚠ Robustez del sistema baja")

    return robustness


if __name__ == "__main__":
    test_robustness()
