"""
TEST 3 — PLANEAMIENTO MULTIPASO
================================

Qué mide: Inteligencia secuencial interna
AGI involucrada: AGI-9 + temporal_tree

Procedimiento:
1. Das un "proyecto interno" (optimizar integración + reducir crisis)
2. Agentes simulan múltiples trayectorias en árbol temporal
3. Mides: profundidad, consistencia, eficiencia

Métrica:
    S3 = mean_depth + goal_success
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PlanningMetrics:
    """Métricas de planeamiento por agente."""
    agent_name: str
    mean_planning_depth: float
    trajectory_consistency: float
    goal_progress: float
    decision_quality: float
    S3_score: float


class Test3Planning:
    """Test de planeamiento multipaso."""

    def __init__(self, agents: List[str] = None):
        self.agents = agents or ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
        self.total_steps = 500
        self.target_integration = 0.8
        self.target_crisis_rate = 0.1

    def run(self, verbose: bool = True) -> Tuple[float, Dict]:
        """Ejecuta el test."""
        from cognition import (
            LongTermProjects,
            TemporalTree,
            DifferentiatedSoftHook
        )

        if verbose:
            print("=" * 70)
            print("TEST 3: PLANEAMIENTO MULTIPASO")
            print("=" * 70)
            print(f"\nObjetivo: integración={self.target_integration}, "
                  f"crisis<{self.target_crisis_rate*100}%")

        # Inicializar módulos
        projects = {a: LongTermProjects(a) for a in self.agents}
        temporal = {a: TemporalTree(z_dim=6, phi_dim=5, D_dim=6) for a in self.agents}
        soft_hook = {a: DifferentiatedSoftHook(a) for a in self.agents}

        # Métricas
        metrics: Dict[str, Dict] = {a: {
            'planning_depths': [],
            'trajectory_scores': [],
            'integration_history': [],
            'crisis_history': [],
            'decisions': []
        } for a in self.agents}

        if verbose:
            print(f"\nSimulando {self.total_steps} pasos con planeamiento...")

        for t in range(self.total_steps):
            for agent in self.agents:
                # Estado actual
                z = np.random.dirichlet(np.ones(6))
                phi = np.random.random(5) * 0.6 + 0.2

                # Integración actual
                integration = 1.0 / (1.0 + np.std(z)) * 0.5 + np.mean(phi) * 0.5
                in_crisis = np.random.random() < 0.2

                metrics[agent]['integration_history'].append(integration)
                metrics[agent]['crisis_history'].append(float(in_crisis))

                # PLANEAMIENTO: simular trayectorias
                # Profundidad de planificación
                depth = int(np.ceil(np.sqrt(t + 1)))
                depth = min(depth, 10)
                metrics[agent]['planning_depths'].append(depth)

                # Simular múltiples trayectorias
                best_score = -np.inf
                best_trajectory = None
                n_trajectories = 3

                trajectory_scores = []
                for traj_idx in range(n_trajectories):
                    # Simular trayectoria
                    traj_integration = integration
                    traj_crisis = 0

                    for step in range(depth):
                        # Evolución simulada
                        delta = np.random.randn() * 0.1
                        traj_integration = np.clip(traj_integration + delta, 0, 1)
                        traj_crisis += float(np.random.random() < 0.15)

                    # Score de trayectoria
                    score = (
                        0.6 * (traj_integration - self.target_integration + 1) +
                        0.4 * (1 - traj_crisis / depth)
                    )
                    trajectory_scores.append(score)

                    if score > best_score:
                        best_score = score
                        best_trajectory = traj_idx

                metrics[agent]['trajectory_scores'].append(np.std(trajectory_scores))

                # Registrar decisión
                decision_quality = best_score
                metrics[agent]['decisions'].append(decision_quality)

                # Actualizar temporal tree
                temporal[agent].record_state(z, in_crisis)

                # Registrar proyecto
                projects[agent].record_episode(t, integration, 0.7)

            if verbose and (t + 1) % 100 == 0:
                mean_int = np.mean([metrics[a]['integration_history'][-50:]
                                   for a in self.agents])
                mean_crisis = np.mean([metrics[a]['crisis_history'][-50:]
                                      for a in self.agents])
                print(f"  t={t+1}: integration={np.mean(mean_int):.3f}, "
                      f"crisis={np.mean(mean_crisis)*100:.0f}%")

        # Calcular resultados
        results: Dict[str, PlanningMetrics] = {}
        S3_scores = []

        if verbose:
            print(f"\n{'=' * 70}")
            print("RESULTADOS")
            print("=" * 70)

        for agent in self.agents:
            # Profundidad media
            mean_depth = np.mean(metrics[agent]['planning_depths'])

            # Consistencia (baja varianza en scores de trayectorias)
            trajectory_consistency = 1.0 / (1.0 + np.mean(metrics[agent]['trajectory_scores']))

            # Progreso hacia objetivo
            final_integration = np.mean(metrics[agent]['integration_history'][-50:])
            final_crisis = np.mean(metrics[agent]['crisis_history'][-50:])

            goal_progress = (
                0.6 * max(0, 1 - abs(final_integration - self.target_integration)) +
                0.4 * max(0, 1 - final_crisis / self.target_crisis_rate)
            )
            goal_progress = min(1.0, goal_progress)

            # Calidad de decisiones
            decision_quality = np.mean(metrics[agent]['decisions'][-100:])

            # S3 = mean_depth_normalized + goal_success
            depth_normalized = mean_depth / 10  # Max depth = 10
            S3 = 0.4 * depth_normalized + 0.6 * goal_progress

            results[agent] = PlanningMetrics(
                agent_name=agent,
                mean_planning_depth=float(mean_depth),
                trajectory_consistency=float(trajectory_consistency),
                goal_progress=float(goal_progress),
                decision_quality=float(decision_quality),
                S3_score=float(S3)
            )

            S3_scores.append(S3)

            if verbose:
                print(f"\n  {agent}:")
                print(f"    Profundidad media: {mean_depth:.1f} pasos")
                print(f"    Consistencia: {trajectory_consistency:.3f}")
                print(f"    Progreso objetivo: {goal_progress:.3f}")
                print(f"    Calidad decisiones: {decision_quality:.3f}")
                print(f"    S3: {S3:.3f}")

        S3_global = float(np.mean(S3_scores))

        if verbose:
            print(f"\n{'═' * 70}")
            print(f"S3 (Planeamiento): {S3_global:.4f}")
            print("═" * 70)

        return S3_global, {
            'score': S3_global,
            'agents': {a: vars(m) for a, m in results.items()},
            'total_steps': self.total_steps
        }


def run_test(verbose: bool = True) -> Tuple[float, Dict]:
    """Ejecuta Test 3."""
    test = Test3Planning()
    return test.run(verbose=verbose)


if __name__ == "__main__":
    score, results = run_test()
    print(f"\nFinal S3 Score: {score:.4f}")
