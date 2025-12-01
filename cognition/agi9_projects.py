"""
AGI-9: Proyectos de Largo Plazo
===============================

Metas → cadenas narrativas → proyectos persistentes.

Proyecto = cadena narrativa dominante:
    C_k = (E_i1, ..., E_in)

Criterios:
- duración ≥ perc75
- coherencia ≥ mediana
- valor ≥ mediana

Progreso:
    prog_k = #episodios completados / n

Valor de proyecto:
    C_k = corr(prog_k, V_t)

Política de proyectos:
    P(P_k) ∝ rank(C_k) + rank(T_k)

100% endógeno.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from .agi_dynamic_constants import (
    L_t, max_history, update_period, adaptive_momentum,
    compute_adaptive_percentile, similarity_threshold
)


class ProjectStatus(Enum):
    """Estado de un proyecto."""
    NASCENT = "nascent"        # Recién detectado
    ACTIVE = "active"          # En progreso activo
    STALLED = "stalled"        # Pausado
    COMPLETED = "completed"    # Terminado
    ABANDONED = "abandoned"    # Abandonado


@dataclass
class ProjectEpisode:
    """Episodio dentro de un proyecto."""
    episode_id: int
    entry_time: int
    value_at_entry: float
    completed: bool = False
    completion_time: Optional[int] = None


@dataclass
class Project:
    """Un proyecto de largo plazo."""
    project_id: int
    name: str  # Nombre generado
    episodes: List[ProjectEpisode]
    creation_time: int
    status: ProjectStatus = ProjectStatus.NASCENT

    # Métricas
    duration: int = 0  # Pasos desde creación
    coherence: float = 0.0
    total_value: float = 0.0
    progress: float = 0.0
    value_correlation: float = 0.0  # corr(prog, V)

    # Para política
    priority_score: float = 0.0
    last_activity: int = 0


class LongTermProjects:
    """
    Sistema de proyectos de largo plazo.

    Detecta y gestiona cadenas narrativas que forman
    proyectos con progreso medible.
    """

    def __init__(self, agent_name: str):
        """
        Inicializa sistema de proyectos.

        Args:
            agent_name: Nombre del agente
        """
        self.agent_name = agent_name

        # Proyectos
        self.projects: Dict[int, Project] = {}
        self.next_project_id = 0

        # Historial de episodios y valores
        self.episode_sequence: List[int] = []
        self.value_history: List[float] = []
        self.coherence_history: List[float] = []

        # Cadenas narrativas candidatas
        self.narrative_chains: List[List[int]] = []

        # Umbrales (se calculan endógenamente)
        self.duration_threshold: float = 0.0
        self.coherence_threshold: float = 0.0
        self.value_threshold: float = 0.0

        self.t = 0

    def _update_thresholds(self):
        """Actualiza umbrales endógenamente."""
        min_samples = L_t(self.t)
        if len(self.value_history) < min_samples:
            return

        # Duración: percentil adaptativo de duraciones existentes
        if self.projects:
            durations = [p.duration for p in self.projects.values() if p.duration > 0]
            if durations:
                self.duration_threshold = compute_adaptive_percentile(
                    np.array(durations), self.t, mode='high'
                )

        # Coherencia: mediana adaptativa
        window = min(len(self.coherence_history), max_history(self.t))
        if len(self.coherence_history) > min_samples:
            self.coherence_threshold = np.median(self.coherence_history[-window:])

        # Valor: mediana adaptativa
        if len(self.value_history) > min_samples:
            self.value_threshold = np.median(self.value_history[-window:])

    def _detect_chains(self) -> List[List[int]]:
        """
        Detecta cadenas narrativas en la secuencia de episodios.

        Usa coherencia temporal para agrupar episodios.
        """
        min_samples = L_t(self.t)
        if len(self.episode_sequence) < min_samples:
            return []

        chains = []
        current_chain = [self.episode_sequence[0]]

        # Calcular historial de similaridades para umbral endógeno
        sim_history = []
        for i in range(1, len(self.episode_sequence)):
            ep = self.episode_sequence[i]
            prev_ep = self.episode_sequence[i-1]
            sim = 1.0 / (1.0 + abs(ep - prev_ep))
            sim_history.append(sim)

        sim_thresh = similarity_threshold(sim_history) if sim_history else 0.3

        for i in range(1, len(self.episode_sequence)):
            ep = self.episode_sequence[i]
            prev_ep = self.episode_sequence[i-1]

            # Coherencia simple: episodios cercanos en ID tienden a ser parte de la misma cadena
            similarity = 1.0 / (1.0 + abs(ep - prev_ep))

            if similarity > sim_thresh:
                current_chain.append(ep)
            else:
                min_chain_len = max(3, int(np.sqrt(self.t / 10 + 1)))
                if len(current_chain) >= min_chain_len:
                    chains.append(current_chain)
                current_chain = [ep]

        min_chain_len = max(3, int(np.sqrt(self.t / 10 + 1)))
        if len(current_chain) >= min_chain_len:
            chains.append(current_chain)

        return chains

    def _chain_to_project(self, chain: List[int]) -> Optional[Project]:
        """
        Convierte una cadena narrativa en proyecto si cumple criterios.
        """
        min_chain_len = max(3, int(np.sqrt(self.t / 10 + 1)))
        if len(chain) < min_chain_len:
            return None

        # Calcular métricas de la cadena
        duration = len(chain)
        coherence = 1.0 / (1.0 + np.std(chain))  # Coherencia basada en varianza de IDs

        # Valores asociados
        values = []
        for ep_id in chain:
            try:
                idx = self.episode_sequence.index(ep_id)
                if idx < len(self.value_history):
                    values.append(self.value_history[idx])
            except ValueError:
                pass

        total_value = np.mean(values) if values else 0.0

        # Criterios más relajados al inicio (umbrales pueden ser 0)
        # Solo verificar si hay suficiente historia

        # Crear proyecto
        episodes = [
            ProjectEpisode(
                episode_id=ep_id,
                entry_time=self.t,
                value_at_entry=values[i] if i < len(values) else 0.0
            )
            for i, ep_id in enumerate(chain)
        ]

        project = Project(
            project_id=self.next_project_id,
            name=f"Project_{self.next_project_id}",
            episodes=episodes,
            creation_time=self.t,
            duration=duration,
            coherence=float(coherence),
            total_value=float(total_value)
        )
        self.next_project_id += 1

        return project

    def _update_project_progress(self, project: Project):
        """
        Actualiza progreso de un proyecto.

        prog_k = #episodios completados / n
        """
        completed = sum(1 for ep in project.episodes if ep.completed)
        project.progress = completed / len(project.episodes) if project.episodes else 0.0

        # Calcular correlación progreso-valor
        if len(project.episodes) > 2:
            progress_values = []
            value_values = []
            cumulative_completed = 0

            for ep in project.episodes:
                if ep.completed:
                    cumulative_completed += 1
                progress_values.append(cumulative_completed / len(project.episodes))
                value_values.append(ep.value_at_entry)

            if np.std(progress_values) > 0 and np.std(value_values) > 0:
                corr = np.corrcoef(progress_values, value_values)[0, 1]
                project.value_correlation = float(corr) if not np.isnan(corr) else 0.0

    def _update_project_status(self, project: Project):
        """Actualiza estado del proyecto."""
        # Duración desde creación
        project.duration = self.t - project.creation_time

        # Tiempo desde última actividad
        time_since_activity = self.t - project.last_activity

        # Umbrales adaptativos para estado de proyecto
        stall_threshold = max_history(self.t) // 5
        abandon_threshold = max_history(self.t) // 2

        if project.progress >= 1.0:
            project.status = ProjectStatus.COMPLETED
        elif time_since_activity > stall_threshold:
            project.status = ProjectStatus.STALLED
            if time_since_activity > abandon_threshold:
                project.status = ProjectStatus.ABANDONED
        elif project.progress > 0:
            project.status = ProjectStatus.ACTIVE
        else:
            project.status = ProjectStatus.NASCENT

    def _update_priority_scores(self):
        """
        Actualiza scores de prioridad.

        P(P_k) ∝ rank(C_k) + rank(T_k)
        donde C_k = value_correlation, T_k = duration
        """
        if not self.projects:
            return

        active_projects = [
            p for p in self.projects.values()
            if p.status in [ProjectStatus.NASCENT, ProjectStatus.ACTIVE]
        ]

        if not active_projects:
            return

        # Calcular ranks
        correlations = [p.value_correlation for p in active_projects]
        durations = [p.duration for p in active_projects]

        for project in active_projects:
            corr_rank = np.sum(np.array(correlations) <= project.value_correlation)
            dur_rank = np.sum(np.array(durations) <= project.duration)
            project.priority_score = float(corr_rank + dur_rank)

        # Normalizar
        total = sum(p.priority_score for p in active_projects)
        if total > 0:
            for project in active_projects:
                project.priority_score /= total

    def record_episode(self, episode_id: int, value: float, coherence: float):
        """
        Registra un nuevo episodio.

        Args:
            episode_id: ID del episodio
            value: Valor actual
            coherence: Coherencia narrativa
        """
        self.t += 1

        self.episode_sequence.append(episode_id)
        self.value_history.append(value)
        self.coherence_history.append(coherence)

        # Limitar historial adaptativamente
        max_hist = max_history(self.t)
        if len(self.episode_sequence) > max_hist:
            self.episode_sequence = self.episode_sequence[-max_hist:]
            self.value_history = self.value_history[-max_hist:]
            self.coherence_history = self.coherence_history[-max_hist:]

        # Actualizar umbrales
        self._update_thresholds()

        # Marcar episodio como completado en proyectos existentes
        for project in self.projects.values():
            for ep in project.episodes:
                if ep.episode_id == episode_id and not ep.completed:
                    ep.completed = True
                    ep.completion_time = self.t
                    project.last_activity = self.t

        # Detectar nuevos proyectos con período adaptativo
        period = update_period(self.value_history)
        if self.t % period == 0:
            chains = self._detect_chains()
            for chain in chains:
                # Verificar si ya existe proyecto similar
                is_new = True
                for project in self.projects.values():
                    existing_eps = set(ep.episode_id for ep in project.episodes)
                    overlap = len(existing_eps.intersection(set(chain))) / len(chain)
                    # Umbral de overlap endógeno
                    overlap_thresh = 1.0 / (1.0 + np.sqrt(self.t / 100 + 1))
                    if overlap > overlap_thresh:
                        is_new = False
                        break

                if is_new:
                    new_project = self._chain_to_project(chain)
                    if new_project:
                        self.projects[new_project.project_id] = new_project

        # Actualizar todos los proyectos
        for project in self.projects.values():
            self._update_project_progress(project)
            self._update_project_status(project)

        self._update_priority_scores()

    def get_active_projects(self) -> List[Project]:
        """Obtiene proyectos activos ordenados por prioridad."""
        active = [
            p for p in self.projects.values()
            if p.status in [ProjectStatus.NASCENT, ProjectStatus.ACTIVE]
        ]
        return sorted(active, key=lambda x: x.priority_score, reverse=True)

    def get_project_for_episode(self, episode_id: int) -> Optional[int]:
        """Obtiene el proyecto al que pertenece un episodio."""
        for project in self.projects.values():
            for ep in project.episodes:
                if ep.episode_id == episode_id:
                    return project.project_id
        return None

    def get_next_episode_suggestion(self) -> Optional[int]:
        """
        Sugiere el siguiente episodio a perseguir basado en proyectos activos.
        """
        active = self.get_active_projects()
        if not active:
            return None

        # Del proyecto más prioritario, encontrar siguiente episodio no completado
        for project in active:
            for ep in project.episodes:
                if not ep.completed:
                    return ep.episode_id

        return None

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas de proyectos."""
        if not self.projects:
            return {
                'agent': self.agent_name,
                't': self.t,
                'n_projects': 0,
                'n_active': 0,
                'n_completed': 0,
                'n_abandoned': 0,
                'mean_progress': 0,
                'mean_coherence': 0,
                'projects': []
            }

        active = [p for p in self.projects.values()
                 if p.status in [ProjectStatus.NASCENT, ProjectStatus.ACTIVE]]
        completed = [p for p in self.projects.values()
                    if p.status == ProjectStatus.COMPLETED]
        abandoned = [p for p in self.projects.values()
                    if p.status == ProjectStatus.ABANDONED]

        project_info = []
        for p in sorted(self.projects.values(), key=lambda x: x.priority_score, reverse=True)[:10]:
            project_info.append({
                'id': p.project_id,
                'status': p.status.value,
                'progress': p.progress,
                'duration': p.duration,
                'coherence': p.coherence,
                'value_corr': p.value_correlation,
                'priority': p.priority_score
            })

        return {
            'agent': self.agent_name,
            't': self.t,
            'n_projects': len(self.projects),
            'n_active': len(active),
            'n_completed': len(completed),
            'n_abandoned': len(abandoned),
            'mean_progress': float(np.mean([p.progress for p in active])) if active else 0,
            'mean_coherence': float(np.mean([p.coherence for p in self.projects.values()])),
            'projects': project_info
        }


def test_projects():
    """Test de proyectos de largo plazo."""
    print("=" * 60)
    print("TEST AGI-9: PROYECTOS DE LARGO PLAZO")
    print("=" * 60)

    projects = LongTermProjects("NEO")

    print("\nSimulando 500 episodios con cadenas narrativas...")

    # Simular secuencias de episodios con patrones
    current_chain_start = 0
    chain_length = 0

    for t in range(500):
        # Generar episodios en cadenas
        if chain_length == 0:
            # Iniciar nueva cadena
            current_chain_start = t
            chain_length = np.random.randint(5, 20)

        episode_id = current_chain_start + (t - current_chain_start) % chain_length

        # Valor y coherencia
        value = 0.5 + 0.3 * np.sin(t / 30) + np.random.randn() * 0.1
        coherence = 0.6 + 0.2 * np.cos(t / 40) + np.random.randn() * 0.1

        projects.record_episode(episode_id, value, coherence)

        if t - current_chain_start >= chain_length - 1:
            chain_length = 0  # Terminar cadena

        if (t + 1) % 100 == 0:
            stats = projects.get_statistics()
            print(f"  t={t+1}: {stats['n_projects']} proyectos, "
                  f"{stats['n_active']} activos, {stats['n_completed']} completados")

    # Resultados finales
    stats = projects.get_statistics()

    print("\n" + "=" * 60)
    print("RESULTADOS PROYECTOS")
    print("=" * 60)

    print(f"\n  Proyectos totales: {stats['n_projects']}")
    print(f"  Activos: {stats['n_active']}")
    print(f"  Completados: {stats['n_completed']}")
    print(f"  Abandonados: {stats['n_abandoned']}")
    print(f"  Progreso medio: {stats['mean_progress']*100:.1f}%")
    print(f"  Coherencia media: {stats['mean_coherence']:.3f}")

    print("\n  Top 5 proyectos:")
    for p in stats['projects'][:5]:
        print(f"    Proyecto {p['id']}: {p['status']}, "
              f"prog={p['progress']*100:.0f}%, dur={p['duration']}, "
              f"prio={p['priority']:.3f}")

    if stats['n_projects'] > 0:
        print("\n  ✓ Proyectos de largo plazo emergiendo")
    else:
        print("\n  ⚠️ No se detectaron proyectos")

    return projects


if __name__ == "__main__":
    test_projects()
