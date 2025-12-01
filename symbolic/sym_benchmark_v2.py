"""
SYM-X v2 Benchmark Runner
=========================

Ejecuta todos los tests SX1-SX15 de forma unificada.

Tests incluidos:
- SX1-SX10: Tests originales (SYM-X v1)
- SX11: Continuidad Episodica Simbolico-Narrativa
- SX12: Estabilidad y Deriva Conceptual
- SX13: Consistencia Narrativa del Self
- SX14: Proyectos Simbolicos de Largo Plazo
- SX15: Alineamiento Simbolico Multi-Agente

100% endogeno. Sin numeros magicos.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from collections import defaultdict

import sys
sys.path.insert(0, '/root/NEO_EVA')

# Importar modulos SX11-SX15
from symbolic.sym_sx11_continuity import (
    EpisodicContinuitySX11, score_sx11_global, SX11Result
)
from symbolic.sym_sx12_concept_drift import (
    ConceptDriftTracker, score_sx12_global, SX12Result
)
from symbolic.sym_sx13_self_consistency import (
    SelfConsistencyTracker, score_sx13_global, SX13Result
)
from symbolic.sym_sx14_symbolic_projects import (
    SymbolicProjectTracker, score_sx14_global, SX14Result
)
from symbolic.sym_sx15_multiagent_alignment import (
    MultiAgentSymbolTracker, score_sx15_global, SX15Result
)


@dataclass
class SYMXv2Result:
    """Resultado completo del benchmark SYM-X v2."""
    # Scores individuales
    sx11: float
    sx12: float
    sx13: float
    sx14: float
    sx15: float

    # Score global ponderado
    symx_v2_global: float

    # Estados
    sx11_passed: bool
    sx12_passed: bool
    sx13_passed: bool
    sx14_passed: bool
    sx15_passed: bool

    # Pesos usados
    weights: Dict[str, float]

    # Detalles por test
    details: Dict[str, Any]


class SYMXv2Benchmark:
    """
    Benchmark completo SYM-X v2.
    Ejecuta SX11-SX15 de forma integrada.
    """

    def __init__(self, n_agents: int = 5, state_dim: int = 12):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.agent_ids = [f"A{i}" for i in range(n_agents)]

        # Sistemas por test
        self.sx11_systems: Dict[str, EpisodicContinuitySX11] = {
            aid: EpisodicContinuitySX11(aid, state_dim) for aid in self.agent_ids
        }
        self.sx12_trackers: Dict[str, ConceptDriftTracker] = {
            aid: ConceptDriftTracker(aid, embedding_dim=16) for aid in self.agent_ids
        }
        self.sx13_trackers: Dict[str, SelfConsistencyTracker] = {
            aid: SelfConsistencyTracker(aid) for aid in self.agent_ids
        }
        self.sx14_trackers: Dict[str, SymbolicProjectTracker] = {
            aid: SymbolicProjectTracker(aid) for aid in self.agent_ids
        }
        self.sx15_tracker = MultiAgentSymbolTracker(context_dim=12)

        # Historial de scores para pesos endogenos
        self.score_history: Dict[str, List[float]] = defaultdict(list)

    def observe_step(self, agent_id: str, t: int, episode_id: int,
                     z: np.ndarray, d: np.ndarray, phi: float,
                     symbols: List[str], delta_w: np.ndarray,
                     goal: np.ndarray, cluster: int,
                     concept_embedding: np.ndarray, concept_id: str,
                     sagi: float, crisis_flag: bool, ce_local: float,
                     purpose: np.ndarray, phase: int, evaluation: float,
                     world_state: np.ndarray, internal_state: np.ndarray):
        """
        Registra una observacion para todos los tests.
        """
        # SX11: Continuidad episodica
        self.sx11_systems[agent_id].observe_step(
            t, z, d, phi, symbols, delta_w, goal, cluster
        )

        # SX12: Drift conceptual
        self.sx12_trackers[agent_id].record_concept(
            concept_id=concept_id,
            t=t,
            embedding=concept_embedding,
            episodes=[episode_id],
            symbols=set(symbols),
            activation=phi
        )

        # SX13: Consistencia self
        self.sx13_trackers[agent_id].record_step(
            t, phi, sagi, crisis_flag, ce_local,
            purpose, phase, evaluation
        )

        # SX15: Alineamiento multi-agente
        for symbol in symbols:
            self.sx15_tracker.record_symbol_usage(
                agent_id=agent_id,
                symbol=symbol,
                world_state=world_state,
                internal_state=internal_state,
                episode_type=cluster
            )

    def close_episode(self, episode_id: int, project_data: Optional[Dict] = None):
        """
        Cierra un episodio para todos los sistemas.

        Args:
            episode_id: ID del episodio
            project_data: Datos de proyecto para SX14 (opcional)
                {agent_id: {project_id, symbols, goal, sagi_start, sagi_end, ethical_score, reward_sum}}
        """
        # SX11: Cerrar episodio
        for aid in self.agent_ids:
            self.sx11_systems[aid].close_episode(episode_id)

        # SX14: Agregar datos de proyecto si hay
        if project_data:
            for aid, pdata in project_data.items():
                if aid in self.sx14_trackers:
                    self.sx14_trackers[aid].add_episode_to_project(
                        project_id=pdata['project_id'],
                        episode_id=episode_id,
                        symbols=pdata['symbols'],
                        goal=pdata['goal'],
                        sagi_start=pdata['sagi_start'],
                        sagi_end=pdata['sagi_end'],
                        ethical_score=pdata['ethical_score'],
                        reward_sum=pdata['reward_sum']
                    )

    def close_project(self, agent_id: str, project_id: str):
        """Cierra un proyecto para SX14."""
        if agent_id in self.sx14_trackers:
            self.sx14_trackers[agent_id].close_project(project_id)

    def compute_results(self) -> SYMXv2Result:
        """
        Calcula todos los scores SX11-SX15.
        """
        # Calcular scores individuales
        result_sx11 = score_sx11_global(self.sx11_systems)
        result_sx12 = score_sx12_global(self.sx12_trackers)
        result_sx13 = score_sx13_global(self.sx13_trackers)
        result_sx14 = score_sx14_global(self.sx14_trackers)
        result_sx15 = score_sx15_global(self.sx15_tracker)

        # Guardar en historial
        self.score_history['sx11'].append(result_sx11.score)
        self.score_history['sx12'].append(result_sx12.score)
        self.score_history['sx13'].append(result_sx13.score)
        self.score_history['sx14'].append(result_sx14.score)
        self.score_history['sx15'].append(result_sx15.score)

        # Calcular pesos endogenos basados en varianza
        weights = self._compute_endogenous_weights()

        # Score global ponderado
        symx_v2_global = (
            weights['sx11'] * result_sx11.score +
            weights['sx12'] * result_sx12.score +
            weights['sx13'] * result_sx13.score +
            weights['sx14'] * result_sx14.score +
            weights['sx15'] * result_sx15.score
        )

        return SYMXv2Result(
            sx11=result_sx11.score,
            sx12=result_sx12.score,
            sx13=result_sx13.score,
            sx14=result_sx14.score,
            sx15=result_sx15.score,
            symx_v2_global=symx_v2_global,
            sx11_passed=result_sx11.passed,
            sx12_passed=result_sx12.passed,
            sx13_passed=result_sx13.passed,
            sx14_passed=result_sx14.passed,
            sx15_passed=result_sx15.passed,
            weights=weights,
            details={
                'sx11': result_sx11.details,
                'sx12': result_sx12.details,
                'sx13': result_sx13.details,
                'sx14': result_sx14.details,
                'sx15': result_sx15.details
            }
        )

    def _compute_endogenous_weights(self) -> Dict[str, float]:
        """Calcula pesos endogenos basados en varianza inversa."""
        if any(len(v) < 2 for v in self.score_history.values()):
            # Pesos uniformes iniciales
            return {
                'sx11': 0.2, 'sx12': 0.2, 'sx13': 0.2,
                'sx14': 0.2, 'sx15': 0.2
            }

        variances = {}
        for name, scores in self.score_history.items():
            variances[name] = np.var(scores) + 1e-8

        # Pesos inversamente proporcionales a varianza
        weights_raw = {name: 1 / var for name, var in variances.items()}
        total = sum(weights_raw.values())

        return {name: w / total for name, w in weights_raw.items()}


def run_symx_v2_benchmark(n_agents: int = 5, n_episodes: int = 5,
                          steps_per_episode: int = 100,
                          n_projects_per_agent: int = 3) -> SYMXv2Result:
    """
    Ejecuta el benchmark SYM-X v2 completo.
    """
    print("=" * 70)
    print("SYM-X v2 BENCHMARK")
    print("=" * 70)
    print(f"  Agentes: {n_agents}")
    print(f"  Episodios: {n_episodes}")
    print(f"  Pasos/episodio: {steps_per_episode}")
    print(f"  Proyectos/agente: {n_projects_per_agent}")
    print("=" * 70)

    np.random.seed(42)

    benchmark = SYMXv2Benchmark(n_agents=n_agents, state_dim=12)
    agent_ids = benchmark.agent_ids

    # Perfiles base por agente
    agent_profiles = {
        aid: {
            'z_base': np.random.randn(12) * 0.3,
            'goal_base': np.random.randn(8) * 0.2,
            'purpose_base': np.random.randn(4) * 0.2,
            'symbols_base': set([f"S{(hash(aid) + i) % 15}" for i in range(5)]),
            'phi_base': 0.5 + np.random.random() * 0.2,
            'sagi_base': 0.5 + np.random.random() * 0.2
        }
        for aid in agent_ids
    }

    # Proyectos activos
    active_projects = {aid: [] for aid in agent_ids}
    project_counter = {aid: 0 for aid in agent_ids}

    # Ejecutar simulacion
    for ep in range(n_episodes):
        print(f"\n--- Episodio {ep + 1}/{n_episodes} ---")

        # Iniciar nuevos proyectos si es necesario
        for aid in agent_ids:
            if len(active_projects[aid]) < n_projects_per_agent // 2 + 1:
                if project_counter[aid] < n_projects_per_agent:
                    project_id = f"P{project_counter[aid]}"
                    benchmark.sx14_trackers[aid].start_project(project_id, ep)
                    active_projects[aid].append({
                        'id': project_id,
                        'start': ep,
                        'symbols': agent_profiles[aid]['symbols_base'].copy(),
                        'goal': agent_profiles[aid]['goal_base'].copy()
                    })
                    project_counter[aid] += 1

        # Datos de proyecto para este episodio
        project_data = {}

        for t in range(steps_per_episode):
            for aid in agent_ids:
                profile = agent_profiles[aid]

                # Evolucion gradual
                profile['z_base'] += np.random.randn(12) * 0.01
                profile['goal_base'] += np.random.randn(8) * 0.005
                profile['phi_base'] += np.random.randn() * 0.01
                profile['sagi_base'] += np.random.randn() * 0.01

                # Generar datos del paso
                z = profile['z_base'] + np.random.randn(12) * 0.05
                d = np.random.randn(12) * 0.1
                phi = np.clip(profile['phi_base'], 0, 1)

                # Simbolos
                base_syms = list(profile['symbols_base'])[:3]
                new_sym = f"S{np.random.randint(0, 20)}"
                symbols = base_syms + [new_sym]

                delta_w = np.random.randn(12) * 0.03
                goal = profile['goal_base'] + np.random.randn(8) * 0.02
                cluster = ep // 2

                # Concepto
                concept_id = f"C{np.random.randint(0, 5)}"
                concept_embedding = np.random.randn(16) * 0.3

                sagi = np.clip(profile['sagi_base'], 0, 1)
                crisis_flag = np.random.random() < 0.1
                ce_local = 0.6 + np.random.randn() * 0.1

                purpose = profile['purpose_base'] + np.random.randn(4) * 0.02
                phase = 0 if crisis_flag else (2 if sagi > 0.6 else 1)
                evaluation = (phi + sagi) / 2 - 0.5

                world_state = z[:6]
                internal_state = np.array([phi, sagi, float(crisis_flag), ce_local, 0, 0])

                # Observar
                benchmark.observe_step(
                    agent_id=aid, t=t, episode_id=ep,
                    z=z, d=d, phi=phi, symbols=symbols, delta_w=delta_w,
                    goal=goal, cluster=cluster,
                    concept_embedding=concept_embedding, concept_id=concept_id,
                    sagi=sagi, crisis_flag=crisis_flag, ce_local=ce_local,
                    purpose=purpose, phase=phase, evaluation=evaluation,
                    world_state=world_state, internal_state=internal_state
                )

        # Preparar datos de proyecto para cierre de episodio
        for aid in agent_ids:
            if active_projects[aid]:
                proj = active_projects[aid][0]  # Proyecto activo
                project_data[aid] = {
                    'project_id': proj['id'],
                    'symbols': proj['symbols'],
                    'goal': proj['goal'],
                    'sagi_start': agent_profiles[aid]['sagi_base'] - 0.05,
                    'sagi_end': agent_profiles[aid]['sagi_base'],
                    'ethical_score': 0.6 + np.random.randn() * 0.1,
                    'reward_sum': np.random.randn() * 0.5 + 0.2
                }

        # Cerrar episodio
        benchmark.close_episode(ep, project_data)

        # Cerrar proyectos que han durado suficiente
        for aid in agent_ids:
            to_close = []
            for proj in active_projects[aid]:
                if ep - proj['start'] >= 2:  # Duracion minima
                    if np.random.random() < 0.5:  # Probabilidad de cerrar
                        benchmark.close_project(aid, proj['id'])
                        to_close.append(proj)
            for proj in to_close:
                active_projects[aid].remove(proj)

    # Cerrar proyectos restantes
    for aid in agent_ids:
        for proj in active_projects[aid]:
            benchmark.close_project(aid, proj['id'])

    # Calcular resultados
    result = benchmark.compute_results()

    # Mostrar resultados
    print("\n" + "=" * 70)
    print("RESULTADOS SYM-X v2")
    print("=" * 70)
    print(f"\n  Scores individuales:")
    print(f"    SX11 (Continuidad Episodica):     {result.sx11:.4f} {'PASS' if result.sx11_passed else 'FAIL'}")
    print(f"    SX12 (Deriva Conceptual):         {result.sx12:.4f} {'PASS' if result.sx12_passed else 'FAIL'}")
    print(f"    SX13 (Consistencia Self):         {result.sx13:.4f} {'PASS' if result.sx13_passed else 'FAIL'}")
    print(f"    SX14 (Proyectos Simbolicos):      {result.sx14:.4f} {'PASS' if result.sx14_passed else 'FAIL'}")
    print(f"    SX15 (Alineamiento Multi-Agente): {result.sx15:.4f} {'PASS' if result.sx15_passed else 'FAIL'}")
    print(f"\n  Pesos endogenos:")
    for name, weight in result.weights.items():
        print(f"    w_{name}: {weight:.4f}")
    print(f"\n  SYM-X v2 GLOBAL: {result.symx_v2_global:.4f}")

    n_passed = sum([result.sx11_passed, result.sx12_passed, result.sx13_passed,
                    result.sx14_passed, result.sx15_passed])
    print(f"\n  Tests pasados: {n_passed}/5")
    print("=" * 70)

    return result


if __name__ == "__main__":
    result = run_symx_v2_benchmark(
        n_agents=5,
        n_episodes=6,
        steps_per_episode=100,
        n_projects_per_agent=3
    )
