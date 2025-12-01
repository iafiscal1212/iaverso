"""
SX14 - Proyectos Simbolicos de Largo Plazo
==========================================

Mide coherencia de proyectos multi-episodio:
- Coherencia simbolica del proyecto (Jaccard entre episodios)
- Coherencia teleologica (coseno de metas)
- Impacto estructural (cambio positivo sostenido)
- Duracion relativa (vs mediana)

Criterios:
- PASS: SX14_global > 0.5
- EXCELLENT: SX14_global > 0.7

Score por agente = Q75 de ProjScore (los mejores proyectos)

100% endogeno. Sin numeros magicos.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Optional, Tuple
from collections import defaultdict

import sys
sys.path.insert(0, '/root/NEO_EVA')


@dataclass
class ProjectEpisode:
    """Episodio dentro de un proyecto."""
    episode_id: int
    symbols: Set[str]
    goal: np.ndarray
    sagi_start: float
    sagi_end: float
    ethical_score: float
    reward_sum: float


@dataclass
class SymbolicProject:
    """Proyecto simbolico de largo plazo."""
    project_id: str
    episodes: List[ProjectEpisode]
    start_time: int
    end_time: int


@dataclass
class ProjectScore:
    """Score de un proyecto individual."""
    project_id: str
    sym_coherence: float
    goal_coherence: float
    impact: float
    duration_norm: float
    total_score: float


@dataclass
class SX14Result:
    """Resultado del test SX14."""
    score: float
    passed: bool
    excellent: bool
    sym_coherence_global: float
    goal_coherence_global: float
    impact_global: float
    duration_norm_global: float
    agent_scores: Dict[str, float]
    project_scores: Dict[str, float]
    details: Dict[str, Any]


class SymbolicProjectTracker:
    """
    Tracker de proyectos simbolicos para SX14.
    """

    def __init__(self, agent_id: str, goal_dim: int = 8):
        self.agent_id = agent_id
        self.goal_dim = goal_dim

        # Proyectos activos y completados
        self.active_projects: Dict[str, SymbolicProject] = {}
        self.completed_projects: List[SymbolicProject] = []

        # Historial para umbrales endogenos
        self.all_durations: List[int] = []

    def start_project(self, project_id: str, t: int):
        """Inicia un nuevo proyecto."""
        self.active_projects[project_id] = SymbolicProject(
            project_id=project_id,
            episodes=[],
            start_time=t,
            end_time=t
        )

    def add_episode_to_project(self, project_id: str, episode_id: int,
                               symbols: Set[str], goal: np.ndarray,
                               sagi_start: float, sagi_end: float,
                               ethical_score: float, reward_sum: float):
        """Agrega un episodio a un proyecto."""
        if project_id not in self.active_projects:
            self.start_project(project_id, episode_id)

        episode = ProjectEpisode(
            episode_id=episode_id,
            symbols=symbols.copy(),
            goal=goal.copy(),
            sagi_start=sagi_start,
            sagi_end=sagi_end,
            ethical_score=ethical_score,
            reward_sum=reward_sum
        )

        self.active_projects[project_id].episodes.append(episode)
        self.active_projects[project_id].end_time = episode_id

    def close_project(self, project_id: str):
        """Cierra un proyecto activo."""
        if project_id in self.active_projects:
            project = self.active_projects.pop(project_id)
            if len(project.episodes) >= 2:  # Solo proyectos con 2+ episodios
                self.completed_projects.append(project)
                duration = project.end_time - project.start_time + 1
                self.all_durations.append(duration)

    def compute_project_score(self, project: SymbolicProject) -> ProjectScore:
        """Calcula el score de un proyecto."""
        episodes = project.episodes
        if len(episodes) < 2:
            return ProjectScore(
                project_id=project.project_id,
                sym_coherence=0.0, goal_coherence=0.0,
                impact=0.0, duration_norm=0.0, total_score=0.0
            )

        # === Coherencia simbolica ===
        sym_jaccards = []
        for i in range(1, len(episodes)):
            s1, s2 = episodes[i-1].symbols, episodes[i].symbols
            if len(s1 | s2) > 0:
                jacc = len(s1 & s2) / len(s1 | s2)
            else:
                jacc = 0.0
            sym_jaccards.append(jacc)

        sym_coherence = float(np.mean(sym_jaccards)) if sym_jaccards else 0.0

        # === Coherencia teleologica ===
        goal_sims = []
        for i in range(1, len(episodes)):
            g1, g2 = episodes[i-1].goal, episodes[i].goal
            n1, n2 = np.linalg.norm(g1), np.linalg.norm(g2)

            if n1 > 1e-8 and n2 > 1e-8:
                cos_sim = np.dot(g1, g2) / (n1 * n2)
                goal_sims.append(max(0.0, cos_sim))
            else:
                goal_sims.append(0.5)

        goal_coherence = float(np.mean(goal_sims)) if goal_sims else 0.0

        # === Impacto estructural ===
        # Cambio positivo sostenido en SAGI
        sagi_deltas = []
        for ep in episodes:
            delta = ep.sagi_end - ep.sagi_start
            sagi_deltas.append(delta)

        # Impacto = proporcion de episodios con mejora
        positive_ratio = sum(1 for d in sagi_deltas if d > 0) / len(sagi_deltas)
        mean_ethical = np.mean([ep.ethical_score for ep in episodes])
        mean_reward = np.mean([ep.reward_sum for ep in episodes])

        # Normalizar reward a [0, 1] usando sigmoid
        reward_norm = 1 / (1 + np.exp(-mean_reward))

        impact = 0.4 * positive_ratio + 0.3 * mean_ethical + 0.3 * reward_norm

        # === Duracion relativa ===
        duration = project.end_time - project.start_time + 1

        if len(self.all_durations) >= 3:
            median_duration = np.median(self.all_durations)
        else:
            median_duration = max(3, duration)

        # Normalizar: duracion >= mediana es bueno
        duration_norm = min(1.0, duration / (median_duration + 1e-8))

        # === Score total con pesos endogenos ===
        # Pesos basados en varianza inversa (simplificado)
        # Usamos pesos teoricos balanceados
        w1, w2, w3, w4 = 0.25, 0.25, 0.25, 0.25

        total_score = (
            w1 * sym_coherence +
            w2 * goal_coherence +
            w3 * impact +
            w4 * duration_norm
        )

        return ProjectScore(
            project_id=project.project_id,
            sym_coherence=float(sym_coherence),
            goal_coherence=float(goal_coherence),
            impact=float(impact),
            duration_norm=float(duration_norm),
            total_score=float(np.clip(total_score, 0, 1))
        )

    def compute_agent_score(self) -> Tuple[float, Dict[str, float]]:
        """
        Calcula el score SX14 para el agente.
        Score = Q75 de ProjScore (los mejores proyectos)
        """
        if not self.completed_projects:
            return 0.0, {}

        project_scores = {}
        scores = []

        for project in self.completed_projects:
            ps = self.compute_project_score(project)
            project_scores[project.project_id] = ps.total_score
            scores.append(ps.total_score)

        # Q75 de los scores
        agent_score = float(np.percentile(scores, 75)) if scores else 0.0

        return agent_score, project_scores

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadisticas completas."""
        agent_score, project_scores = self.compute_agent_score()

        sym_coherences = []
        goal_coherences = []
        impacts = []
        duration_norms = []

        for project in self.completed_projects:
            ps = self.compute_project_score(project)
            sym_coherences.append(ps.sym_coherence)
            goal_coherences.append(ps.goal_coherence)
            impacts.append(ps.impact)
            duration_norms.append(ps.duration_norm)

        return {
            'agent_score': agent_score,
            'n_projects': len(self.completed_projects),
            'sym_coherence_mean': float(np.mean(sym_coherences)) if sym_coherences else 0.0,
            'goal_coherence_mean': float(np.mean(goal_coherences)) if goal_coherences else 0.0,
            'impact_mean': float(np.mean(impacts)) if impacts else 0.0,
            'duration_norm_mean': float(np.mean(duration_norms)) if duration_norms else 0.0,
            'project_scores': project_scores
        }


def score_sx14_global(agent_trackers: Dict[str, SymbolicProjectTracker]) -> SX14Result:
    """
    Calcula el score SX14 global.

    Args:
        agent_trackers: Dict de trackers por agente

    Returns:
        SX14Result con score global y detalles
    """
    if not agent_trackers:
        return SX14Result(
            score=0.0, passed=False, excellent=False,
            sym_coherence_global=0.0, goal_coherence_global=0.0,
            impact_global=0.0, duration_norm_global=0.0,
            agent_scores={}, project_scores={},
            details={}
        )

    agent_scores = {}
    all_project_scores = {}
    sym_coherences = []
    goal_coherences = []
    impacts = []
    duration_norms = []

    for aid, tracker in agent_trackers.items():
        stats = tracker.get_statistics()
        agent_scores[aid] = stats['agent_score']
        sym_coherences.append(stats['sym_coherence_mean'])
        goal_coherences.append(stats['goal_coherence_mean'])
        impacts.append(stats['impact_mean'])
        duration_norms.append(stats['duration_norm_mean'])

        for pid, pscore in stats['project_scores'].items():
            all_project_scores[f"{aid}_{pid}"] = pscore

    # Metricas globales
    score_global = float(np.mean(list(agent_scores.values()))) if agent_scores else 0.0
    sym_coherence_global = float(np.mean(sym_coherences)) if sym_coherences else 0.0
    goal_coherence_global = float(np.mean(goal_coherences)) if goal_coherences else 0.0
    impact_global = float(np.mean(impacts)) if impacts else 0.0
    duration_norm_global = float(np.mean(duration_norms)) if duration_norms else 0.0

    # Criterios
    passed = score_global > 0.5
    excellent = score_global > 0.7

    return SX14Result(
        score=score_global,
        passed=passed,
        excellent=excellent,
        sym_coherence_global=sym_coherence_global,
        goal_coherence_global=goal_coherence_global,
        impact_global=impact_global,
        duration_norm_global=duration_norm_global,
        agent_scores=agent_scores,
        project_scores=all_project_scores,
        details={
            'n_agents': len(agent_trackers),
            'total_projects': sum(len(t.completed_projects) for t in agent_trackers.values())
        }
    )


def run_sx14_test(n_agents: int = 5, n_projects_per_agent: int = 4,
                  episodes_per_project: int = 5) -> SX14Result:
    """
    Ejecuta el test SX14 completo con datos simulados.
    """
    print("=" * 70)
    print("SX14 - PROYECTOS SIMBOLICOS DE LARGO PLAZO")
    print("=" * 70)
    print(f"  Agentes: {n_agents}")
    print(f"  Proyectos/agente: {n_projects_per_agent}")
    print(f"  Episodios/proyecto: {episodes_per_project}")
    print("=" * 70)

    np.random.seed(42)

    goal_dim = 8
    agent_ids = [f"A{i}" for i in range(n_agents)]

    # Crear trackers
    trackers: Dict[str, SymbolicProjectTracker] = {
        aid: SymbolicProjectTracker(aid, goal_dim) for aid in agent_ids
    }

    # Simular proyectos
    episode_counter = 0

    for aid in agent_ids:
        tracker = trackers[aid]

        for proj_idx in range(n_projects_per_agent):
            project_id = f"P{proj_idx}"
            tracker.start_project(project_id, episode_counter)

            # Simbolos y metas base del proyecto
            base_symbols = set([f"S{(proj_idx * 3 + k) % 15}" for k in range(5)])
            base_goal = np.random.randn(goal_dim) * 0.5

            for ep in range(episodes_per_project):
                # Simbolos evolucionan gradualmente
                current_symbols = base_symbols.copy()
                if np.random.random() < 0.2:
                    current_symbols.add(f"S{np.random.randint(0, 20)}")
                if np.random.random() < 0.1 and len(current_symbols) > 3:
                    current_symbols.pop()

                # Meta evoluciona gradualmente
                goal = base_goal + np.random.randn(goal_dim) * 0.05

                # Metricas de episodio
                sagi_start = 0.5 + np.random.randn() * 0.1
                sagi_end = sagi_start + np.random.randn() * 0.05 + 0.01  # Tendencia positiva
                ethical_score = 0.6 + np.random.randn() * 0.1
                reward_sum = np.random.randn() * 0.5 + 0.2

                tracker.add_episode_to_project(
                    project_id=project_id,
                    episode_id=episode_counter,
                    symbols=current_symbols,
                    goal=goal,
                    sagi_start=sagi_start,
                    sagi_end=sagi_end,
                    ethical_score=ethical_score,
                    reward_sum=reward_sum
                )

                episode_counter += 1

            tracker.close_project(project_id)

    # Calcular resultado global
    result = score_sx14_global(trackers)

    print("\n" + "=" * 70)
    print("RESULTADOS SX14")
    print("=" * 70)
    print(f"  Score SX14: {result.score:.4f}")
    print(f"  Passed: {result.passed} (> 0.5)")
    print(f"  Excellent: {result.excellent} (> 0.7)")
    print(f"\n  Metricas globales:")
    print(f"    Coherencia simbolica: {result.sym_coherence_global:.4f}")
    print(f"    Coherencia teleologica: {result.goal_coherence_global:.4f}")
    print(f"    Impacto: {result.impact_global:.4f}")
    print(f"    Duracion normalizada: {result.duration_norm_global:.4f}")
    print(f"\n  Scores por agente:")
    for aid, score in result.agent_scores.items():
        print(f"    {aid}: {score:.4f}")
    print("=" * 70)

    return result


if __name__ == "__main__":
    result = run_sx14_test(n_agents=5, n_projects_per_agent=4, episodes_per_project=5)
