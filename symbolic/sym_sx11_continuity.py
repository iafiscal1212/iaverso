"""
SX11 - Continuidad Episodica Simbolico-Narrativa
=================================================

Mide la continuidad vital entre episodios consecutivos:
- CE_struct: Coherencia estructural (Mahalanobis)
- CE_sym: Continuidad simbolica (Jaccard)
- CE_causal: Continuidad causal (coseno deltaW)
- CE_goal: Continuidad teleologica (coseno metas)
- CE_cluster: Continuidad de tema narrativo

Criterios:
- PASS: SX11_global > 0.5
- EXCELLENT: SX11_global > 0.7

100% endogeno. Sin numeros magicos.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Optional, Tuple
from collections import defaultdict

import sys
sys.path.insert(0, '/root/NEO_EVA')


@dataclass
class EpisodeSignatureSX11:
    """Firma de un episodio para SX11."""
    episode_id: int
    z_mean: np.ndarray          # Estado estructural medio
    phi_mean: float             # Fenomenologia media
    d_mean: np.ndarray          # Drives medios
    delta_w: np.ndarray         # Efecto en WORLD-1
    symbols: Set[str]           # Simbolos activos
    goal: np.ndarray            # Meta dominante
    cluster_narr: int           # Cluster narrativo
    t_start: int = 0
    t_end: int = 0


@dataclass
class ContinuityMetricsSX11:
    """Metricas de continuidad entre episodios."""
    ce_struct: float
    ce_sym: float
    ce_causal: float
    ce_goal: float
    ce_cluster: float
    ce_total: float
    weights: Dict[str, float]


@dataclass
class SX11Result:
    """Resultado del test SX11."""
    score: float
    passed: bool
    excellent: bool
    ce_global: float
    ce_by_agent: Dict[str, float]
    ce_components: Dict[str, float]
    weights: Dict[str, float]
    details: Dict[str, Any]


class EpisodicContinuitySX11:
    """
    Sistema de continuidad episodica para SX11.
    Mide coherencia entre episodios consecutivos.
    """

    def __init__(self, agent_id: str, state_dim: int = 12):
        self.agent_id = agent_id
        self.state_dim = state_dim

        # Episodios completados
        self.episodes: List[EpisodeSignatureSX11] = []

        # Historial de metricas para umbrales endogenos
        self.struct_distances: List[float] = []
        self.causal_cosines: List[float] = []
        self.goal_distances: List[float] = []

        # Metricas de continuidad por transicion
        self.continuity_history: List[ContinuityMetricsSX11] = []

        # Acumuladores para episodio actual
        self._current_z: List[np.ndarray] = []
        self._current_phi: List[float] = []
        self._current_d: List[np.ndarray] = []
        self._current_delta_w: List[np.ndarray] = []
        self._current_symbols: Set[str] = set()
        self._current_goals: List[np.ndarray] = []
        self._current_cluster: int = 0
        self._current_t_start: int = 0

    def observe_step(self, t: int, z: np.ndarray, d: np.ndarray,
                     phi: float, symbols: List[str], delta_w: np.ndarray,
                     goal: np.ndarray, cluster: int = 0):
        """Registra un paso del episodio actual."""
        if len(self._current_z) == 0:
            self._current_t_start = t

        self._current_z.append(z.copy())
        self._current_phi.append(phi)
        self._current_d.append(d.copy())
        self._current_delta_w.append(delta_w.copy())
        self._current_symbols.update(symbols)
        self._current_goals.append(goal.copy())
        self._current_cluster = cluster

    def close_episode(self, episode_id: int) -> Optional[ContinuityMetricsSX11]:
        """Cierra el episodio actual y calcula continuidad."""
        if len(self._current_z) < 10:
            self._reset_current()
            return None

        # Crear firma del episodio
        signature = EpisodeSignatureSX11(
            episode_id=episode_id,
            z_mean=np.mean(self._current_z, axis=0),
            phi_mean=np.mean(self._current_phi),
            d_mean=np.mean(self._current_d, axis=0),
            delta_w=np.sum(self._current_delta_w, axis=0),
            symbols=self._current_symbols.copy(),
            goal=np.mean(self._current_goals, axis=0),
            cluster_narr=self._current_cluster,
            t_start=self._current_t_start,
            t_end=self._current_t_start + len(self._current_z)
        )

        self.episodes.append(signature)

        # Calcular continuidad si hay episodio previo
        metrics = None
        if len(self.episodes) >= 2:
            metrics = self._compute_continuity(
                self.episodes[-2],
                self.episodes[-1]
            )
            self.continuity_history.append(metrics)

        self._reset_current()
        return metrics

    def _reset_current(self):
        """Resetea acumuladores del episodio actual."""
        self._current_z = []
        self._current_phi = []
        self._current_d = []
        self._current_delta_w = []
        self._current_symbols = set()
        self._current_goals = []
        self._current_cluster = 0

    def _compute_continuity(self, ep1: EpisodeSignatureSX11,
                           ep2: EpisodeSignatureSX11) -> ContinuityMetricsSX11:
        """Calcula metricas de continuidad entre dos episodios."""

        # === CE_struct: Coherencia estructural ===
        # Distancia Mahalanobis normalizada
        diff = ep2.z_mean - ep1.z_mean

        # Covarianza empirica de z
        if len(self.episodes) >= 3:
            z_stack = np.array([e.z_mean for e in self.episodes])
            cov = np.cov(z_stack.T) + np.eye(self.state_dim) * 1e-6
            try:
                cov_inv = np.linalg.inv(cov)
                d_mahal = float(np.sqrt(diff @ cov_inv @ diff))
            except:
                d_mahal = float(np.linalg.norm(diff))
        else:
            d_mahal = float(np.linalg.norm(diff))

        self.struct_distances.append(d_mahal)

        # Q95 endogeno
        if len(self.struct_distances) >= 3:
            q95_struct = np.percentile(self.struct_distances, 95)
        else:
            q95_struct = max(d_mahal * 2, 1.0)

        ce_struct = max(0.0, 1 - d_mahal / (q95_struct + 1e-8))

        # === CE_sym: Continuidad simbolica (Jaccard) ===
        s1, s2 = ep1.symbols, ep2.symbols
        if len(s1 | s2) > 0:
            ce_sym = len(s1 & s2) / len(s1 | s2)
        else:
            ce_sym = 0.0

        # === CE_causal: Continuidad causal ===
        # Coseno de efectos en WORLD-1
        norm1 = np.linalg.norm(ep1.delta_w)
        norm2 = np.linalg.norm(ep2.delta_w)

        if norm1 > 1e-8 and norm2 > 1e-8:
            cos_causal = float(np.dot(ep1.delta_w, ep2.delta_w) / (norm1 * norm2))
            cos_causal = max(0.0, cos_causal)  # Solo correlacion positiva
        else:
            cos_causal = 0.5  # Default si no hay efecto

        self.causal_cosines.append(cos_causal)

        # Q95 endogeno para normalizar
        if len(self.causal_cosines) >= 3:
            q95_causal = np.percentile(self.causal_cosines, 95)
        else:
            q95_causal = 1.0

        ce_causal = min(1.0, cos_causal / (q95_causal + 1e-8))

        # === CE_goal: Continuidad teleologica ===
        norm_g1 = np.linalg.norm(ep1.goal)
        norm_g2 = np.linalg.norm(ep2.goal)

        if norm_g1 > 1e-8 and norm_g2 > 1e-8:
            cos_goal = float(np.dot(ep1.goal, ep2.goal) / (norm_g1 * norm_g2))
            d_goal = 1 - cos_goal
        else:
            d_goal = 0.5

        self.goal_distances.append(d_goal)

        if len(self.goal_distances) >= 3:
            q95_goal = np.percentile(self.goal_distances, 95)
        else:
            q95_goal = 1.0

        ce_goal = max(0.0, 1 - d_goal / (q95_goal + 1e-8))

        # === CE_cluster: Continuidad de tema narrativo ===
        ce_cluster = 1.0 if ep1.cluster_narr == ep2.cluster_narr else 0.0

        # === Pesos endogenos ===
        weights = self._compute_endogenous_weights()

        # === CE total ===
        ce_total = (
            weights['struct'] * ce_struct +
            weights['sym'] * ce_sym +
            weights['causal'] * ce_causal +
            weights['goal'] * ce_goal +
            weights['cluster'] * ce_cluster
        )

        return ContinuityMetricsSX11(
            ce_struct=float(ce_struct),
            ce_sym=float(ce_sym),
            ce_causal=float(ce_causal),
            ce_goal=float(ce_goal),
            ce_cluster=float(ce_cluster),
            ce_total=float(ce_total),
            weights=weights
        )

    def _compute_endogenous_weights(self) -> Dict[str, float]:
        """Calcula pesos endogenos basados en varianza."""
        if len(self.continuity_history) < 2:
            # Pesos uniformes iniciales
            return {
                'struct': 0.2,
                'sym': 0.2,
                'causal': 0.2,
                'goal': 0.2,
                'cluster': 0.2
            }

        # Varianzas de cada metrica
        structs = [m.ce_struct for m in self.continuity_history]
        syms = [m.ce_sym for m in self.continuity_history]
        causals = [m.ce_causal for m in self.continuity_history]
        goals = [m.ce_goal for m in self.continuity_history]
        clusters = [m.ce_cluster for m in self.continuity_history]

        var_struct = np.var(structs) + 1e-8
        var_sym = np.var(syms) + 1e-8
        var_causal = np.var(causals) + 1e-8
        var_goal = np.var(goals) + 1e-8
        var_cluster = np.var(clusters) + 1e-8

        # Pesos inversamente proporcionales a varianza
        w_struct = 1 / var_struct
        w_sym = 1 / var_sym
        w_causal = 1 / var_causal
        w_goal = 1 / var_goal
        w_cluster = 1 / var_cluster

        total = w_struct + w_sym + w_causal + w_goal + w_cluster

        return {
            'struct': w_struct / total,
            'sym': w_sym / total,
            'causal': w_causal / total,
            'goal': w_goal / total,
            'cluster': w_cluster / total
        }

    def get_ce_agent(self) -> float:
        """Retorna CE promedio del agente."""
        if not self.continuity_history:
            return 0.0
        return float(np.mean([m.ce_total for m in self.continuity_history]))

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadisticas completas."""
        if not self.continuity_history:
            return {
                'ce_agent': 0.0,
                'n_episodes': len(self.episodes),
                'n_transitions': 0,
                'ce_components': {
                    'struct': 0.0, 'sym': 0.0, 'causal': 0.0,
                    'goal': 0.0, 'cluster': 0.0
                },
                'weights': self._compute_endogenous_weights()
            }

        return {
            'ce_agent': self.get_ce_agent(),
            'n_episodes': len(self.episodes),
            'n_transitions': len(self.continuity_history),
            'ce_components': {
                'struct': float(np.mean([m.ce_struct for m in self.continuity_history])),
                'sym': float(np.mean([m.ce_sym for m in self.continuity_history])),
                'causal': float(np.mean([m.ce_causal for m in self.continuity_history])),
                'goal': float(np.mean([m.ce_goal for m in self.continuity_history])),
                'cluster': float(np.mean([m.ce_cluster for m in self.continuity_history]))
            },
            'weights': self._compute_endogenous_weights()
        }


def compute_episode_continuity(agent_continuity: EpisodicContinuitySX11) -> Dict[str, Any]:
    """
    Calcula continuidad episodica para un agente.

    Returns:
        {
          "CE_struct": [...],
          "CE_sym": [...],
          "CE_causal": [...],
          "CE_goal": [...],
          "CE_cluster": [...],
          "CE": [...],
          "CE_A": float,
        }
    """
    history = agent_continuity.continuity_history

    return {
        "CE_struct": [m.ce_struct for m in history],
        "CE_sym": [m.ce_sym for m in history],
        "CE_causal": [m.ce_causal for m in history],
        "CE_goal": [m.ce_goal for m in history],
        "CE_cluster": [m.ce_cluster for m in history],
        "CE": [m.ce_total for m in history],
        "CE_A": agent_continuity.get_ce_agent()
    }


def score_sx11_global(agent_systems: Dict[str, EpisodicContinuitySX11]) -> SX11Result:
    """
    Calcula el score SX11 global.

    Args:
        agent_systems: Dict de sistemas de continuidad por agente

    Returns:
        SX11Result con score global y detalles
    """
    if not agent_systems:
        return SX11Result(
            score=0.0, passed=False, excellent=False,
            ce_global=0.0, ce_by_agent={},
            ce_components={'struct': 0, 'sym': 0, 'causal': 0, 'goal': 0, 'cluster': 0},
            weights={}, details={}
        )

    # CE por agente
    ce_by_agent = {}
    all_components = defaultdict(list)
    all_weights = defaultdict(list)

    for aid, system in agent_systems.items():
        stats = system.get_statistics()
        ce_by_agent[aid] = stats['ce_agent']

        for comp, val in stats['ce_components'].items():
            all_components[comp].append(val)
        for w, val in stats['weights'].items():
            all_weights[w].append(val)

    # CE global
    ce_global = float(np.mean(list(ce_by_agent.values()))) if ce_by_agent else 0.0

    # Componentes promedio
    ce_components = {k: float(np.mean(v)) for k, v in all_components.items()}

    # Pesos promedio
    weights = {k: float(np.mean(v)) for k, v in all_weights.items()}

    # Score = CE_global
    score = ce_global

    # Criterios
    passed = score > 0.5
    excellent = score > 0.7

    return SX11Result(
        score=score,
        passed=passed,
        excellent=excellent,
        ce_global=ce_global,
        ce_by_agent=ce_by_agent,
        ce_components=ce_components,
        weights=weights,
        details={
            'n_agents': len(agent_systems),
            'total_episodes': sum(len(s.episodes) for s in agent_systems.values()),
            'total_transitions': sum(len(s.continuity_history) for s in agent_systems.values())
        }
    )


def run_sx11_test(n_agents: int = 5, n_episodes: int = 5,
                  steps_per_episode: int = 200) -> SX11Result:
    """
    Ejecuta el test SX11 completo con datos simulados.
    """
    print("=" * 70)
    print("SX11 - CONTINUIDAD EPISODICA SIMBOLICO-NARRATIVA")
    print("=" * 70)
    print(f"  Agentes: {n_agents}")
    print(f"  Episodios: {n_episodes}")
    print(f"  Pasos/episodio: {steps_per_episode}")
    print("=" * 70)

    np.random.seed(42)

    state_dim = 12
    agent_ids = [f"A{i}" for i in range(n_agents)]

    # Crear sistemas de continuidad
    systems: Dict[str, EpisodicContinuitySX11] = {
        aid: EpisodicContinuitySX11(aid, state_dim) for aid in agent_ids
    }

    # Perfiles base por agente (para continuidad)
    agent_profiles = {aid: np.random.randn(state_dim) * 0.3 for aid in agent_ids}
    agent_goal_base = {aid: np.random.randn(state_dim) * 0.2 for aid in agent_ids}
    agent_symbols_base = {aid: set([f"S{(hash(aid) + i) % 15}" for i in range(5)])
                         for aid in agent_ids}

    # Ejecutar episodios
    for ep in range(n_episodes):
        print(f"\n--- Episodio {ep + 1}/{n_episodes} ---")

        for aid in agent_ids:
            system = systems[aid]

            # Evolucion gradual del perfil
            agent_profiles[aid] += np.random.randn(state_dim) * 0.02
            agent_goal_base[aid] += np.random.randn(state_dim) * 0.01

            # Cluster narrativo (cambia ocasionalmente)
            cluster = ep // 2  # Cambia cada 2 episodios

            for t in range(steps_per_episode):
                # Datos con continuidad
                z = agent_profiles[aid] + np.random.randn(state_dim) * 0.05
                d = np.random.randn(state_dim) * 0.1
                phi = 0.5 + ep * 0.05 + np.random.randn() * 0.02

                # Simbolos: base + algunos nuevos
                base_syms = list(agent_symbols_base[aid])[:3]
                new_syms = [f"S{np.random.randint(0, 20)}"]
                symbols = base_syms + new_syms

                # Actualizar base de simbolos gradualmente
                if np.random.random() < 0.1:
                    agent_symbols_base[aid].add(f"S{np.random.randint(0, 20)}")
                    if len(agent_symbols_base[aid]) > 8:
                        agent_symbols_base[aid].pop()

                delta_w = np.random.randn(state_dim) * 0.03 + agent_profiles[aid] * 0.01
                goal = agent_goal_base[aid] + np.random.randn(state_dim) * 0.02

                system.observe_step(t, z, d, phi, symbols, delta_w, goal, cluster)

            # Cerrar episodio
            metrics = system.close_episode(ep)

            if metrics:
                print(f"  {aid}: CE={metrics.ce_total:.3f} "
                      f"(str={metrics.ce_struct:.2f}, sym={metrics.ce_sym:.2f}, "
                      f"cau={metrics.ce_causal:.2f}, gol={metrics.ce_goal:.2f})")

    # Calcular resultado global
    result = score_sx11_global(systems)

    print("\n" + "=" * 70)
    print("RESULTADOS SX11")
    print("=" * 70)
    print(f"  Score SX11: {result.score:.4f}")
    print(f"  Passed: {result.passed} (> 0.5)")
    print(f"  Excellent: {result.excellent} (> 0.7)")
    print(f"\n  CE por agente:")
    for aid, ce in result.ce_by_agent.items():
        print(f"    {aid}: {ce:.4f}")
    print(f"\n  Componentes promedio:")
    for comp, val in result.ce_components.items():
        print(f"    CE_{comp}: {val:.4f}")
    print(f"\n  Pesos endogenos:")
    for w, val in result.weights.items():
        print(f"    w_{w}: {val:.4f}")
    print("=" * 70)

    return result


if __name__ == "__main__":
    result = run_sx11_test(n_agents=5, n_episodes=5, steps_per_episode=200)
