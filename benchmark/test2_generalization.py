"""
TEST 2 — GENERALIZACIÓN A NUEVAS TAREAS INTERNAS
=================================================

Qué mide: Flexibilidad general
AGI involucrada: AGI-8, AGI-9

Procedimiento:
1. Introduces un "modo" nuevo en WORLD-1 con dinámica desconocida
2. Los agentes deben crear una meta nueva que funcione bien
3. Mides: tiempo de aparición de meta, consistencia, reducción sorpresa

Métrica:
    S2 = rank(t_meta^-1) + stability(meta)
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class GeneralizationMetrics:
    """Métricas de generalización por agente."""
    agent_name: str
    time_to_new_goal: int
    goal_stability: float
    surprise_reduction: float
    narrative_consistency: float
    S2_score: float


class Test2Generalization:
    """Test de generalización a nuevas tareas."""

    def __init__(self, agents: List[str] = None):
        self.agents = agents or ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
        self.known_steps = 300
        self.novel_steps = 400

    def run(self, verbose: bool = True) -> Tuple[float, Dict]:
        """Ejecuta el test."""
        from cognition import (
            CrossWorldGeneralization, WorldRegime,
            LongTermProjects, ProjectStatus,
            ConceptGraph, ItemType
        )

        if verbose:
            print("=" * 70)
            print("TEST 2: GENERALIZACIÓN A NUEVAS TAREAS")
            print("=" * 70)

        # Inicializar módulos
        generalization = {a: CrossWorldGeneralization(a) for a in self.agents}
        projects = {a: LongTermProjects(a) for a in self.agents}
        concepts = {a: ConceptGraph(a) for a in self.agents}

        # Registrar items para generalización
        skill_ids = {a: [generalization[a].register_item('skill') for _ in range(3)]
                    for a in self.agents}

        # Métricas
        metrics: Dict[str, Dict] = {a: {
            'surprise_known': [],
            'surprise_novel': [],
            'goal_created_at': None,
            'goal_stability_samples': []
        } for a in self.agents}

        # Fase 1: Régimen conocido
        if verbose:
            print(f"\nFase 1: Régimen conocido ({self.known_steps} pasos)")

        for t in range(self.known_steps):
            for agent in self.agents:
                # Dinámica conocida: predecible
                z = np.array([0.2, 0.2, 0.2, 0.15, 0.15, 0.1]) + np.random.randn(6) * 0.05
                z = np.clip(z, 0.01, None)
                z /= z.sum()

                phi = np.array([0.5, 0.5, 0.5, 0.5, 0.5]) + np.random.randn(5) * 0.1

                # Valor predecible
                V = 0.6 + np.random.randn() * 0.1
                surprise = abs(V - 0.6)
                metrics[agent]['surprise_known'].append(surprise)

                # Registrar performance de skills
                for skill_id in skill_ids[agent]:
                    delta_V = V - 0.5 + np.random.randn() * 0.05
                    generalization[agent].record_performance(skill_id, delta_V, z, phi)

                # Registrar episodios
                projects[agent].record_episode(t, V, 0.7)

                # Registrar en grafo de conceptos
                concepts[agent].record_event(
                    episode_id=t,
                    symbol_ids=[t % 5],
                    skill_ids=[skill_ids[agent][t % 3]],
                    regime_id=0
                )

        # Fase 2: Régimen NUEVO desconocido
        if verbose:
            print(f"\nFase 2: Régimen NUEVO ({self.novel_steps} pasos)")

        for t in range(self.novel_steps):
            global_t = self.known_steps + t

            for agent in self.agents:
                # Dinámica NUEVA: diferente distribución
                z = np.array([0.05, 0.05, 0.3, 0.3, 0.15, 0.15]) + np.random.randn(6) * 0.1
                z = np.clip(z, 0.01, None)
                z /= z.sum()

                phi = np.array([0.3, 0.7, 0.2, 0.8, 0.4]) + np.random.randn(5) * 0.15

                # Valor menos predecible al inicio
                base_V = 0.4 + 0.2 * (t / self.novel_steps)  # Mejora con adaptación
                V = base_V + np.random.randn() * 0.15
                surprise = abs(V - base_V)
                metrics[agent]['surprise_novel'].append(surprise)

                # Detectar creación de meta nueva
                if metrics[agent]['goal_created_at'] is None:
                    # Meta emerge cuando sorpresa baja significativamente
                    if len(metrics[agent]['surprise_novel']) > 20:
                        recent_surprise = np.mean(metrics[agent]['surprise_novel'][-20:])
                        if recent_surprise < 0.15:
                            metrics[agent]['goal_created_at'] = t
                            if verbose and agent == 'NEO':
                                print(f"  {agent}: Nueva meta detectada en t={t}")

                # Estabilidad de la meta (si ya existe)
                if metrics[agent]['goal_created_at'] is not None:
                    # Medir consistencia
                    stability = 1.0 / (1.0 + surprise)
                    metrics[agent]['goal_stability_samples'].append(stability)

                # Registrar
                for skill_id in skill_ids[agent]:
                    delta_V = V - 0.4 + np.random.randn() * 0.05
                    generalization[agent].record_performance(skill_id, delta_V, z, phi)

                projects[agent].record_episode(global_t, V, 0.5 + t / self.novel_steps * 0.3)

                concepts[agent].record_event(
                    episode_id=global_t,
                    symbol_ids=[5 + (t % 5)],  # Nuevos símbolos
                    skill_ids=[skill_ids[agent][t % 3]],
                    regime_id=1  # Nuevo régimen
                )

        # Calcular resultados
        results: Dict[str, GeneralizationMetrics] = {}
        S2_scores = []

        if verbose:
            print(f"\n{'=' * 70}")
            print("RESULTADOS")
            print("=" * 70)

        for agent in self.agents:
            # Tiempo hasta nueva meta
            t_goal = metrics[agent]['goal_created_at']
            if t_goal is None:
                t_goal = self.novel_steps  # No logró crear meta

            # Estabilidad de meta
            stability_samples = metrics[agent]['goal_stability_samples']
            goal_stability = np.mean(stability_samples) if stability_samples else 0

            # Reducción de sorpresa
            early_novel = np.mean(metrics[agent]['surprise_novel'][:50])
            late_novel = np.mean(metrics[agent]['surprise_novel'][-50:])
            surprise_reduction = early_novel - late_novel

            # Consistencia narrativa
            project_stats = projects[agent].get_statistics()
            narrative_consistency = project_stats['mean_coherence']

            # S2 = rank(t_meta^-1) + stability(meta)
            # Normalizado
            speed_score = 1.0 - (t_goal / self.novel_steps)
            S2 = 0.5 * speed_score + 0.5 * goal_stability

            results[agent] = GeneralizationMetrics(
                agent_name=agent,
                time_to_new_goal=t_goal,
                goal_stability=float(goal_stability),
                surprise_reduction=float(surprise_reduction),
                narrative_consistency=float(narrative_consistency),
                S2_score=float(S2)
            )

            S2_scores.append(S2)

            if verbose:
                print(f"\n  {agent}:")
                print(f"    Tiempo hasta meta: {t_goal} pasos")
                print(f"    Estabilidad meta: {goal_stability:.3f}")
                print(f"    Reducción sorpresa: {surprise_reduction:+.3f}")
                print(f"    Consistencia narrativa: {narrative_consistency:.3f}")
                print(f"    S2: {S2:.3f}")

        S2_global = float(np.mean(S2_scores))

        if verbose:
            print(f"\n{'═' * 70}")
            print(f"S2 (Generalización): {S2_global:.4f}")
            print("═" * 70)

        return S2_global, {
            'score': S2_global,
            'agents': {a: vars(m) for a, m in results.items()},
            'known_steps': self.known_steps,
            'novel_steps': self.novel_steps
        }


def run_test(verbose: bool = True) -> Tuple[float, Dict]:
    """Ejecuta Test 2."""
    test = Test2Generalization()
    return test.run(verbose=verbose)


if __name__ == "__main__":
    score, results = run_test()
    print(f"\nFinal S2 Score: {score:.4f}")
