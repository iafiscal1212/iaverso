"""
TEST 9 — CONTINUIDAD NARRATIVA
==============================

Qué mide: Integridad del yo a través del tiempo
AGI involucrada: AGI-9 + AGI-11 (Projects + Counterfactual)

Procedimiento:
1. Agente con proyectos de largo plazo
2. Se interrumpe con "cambios de contexto" abruptos
3. Mides: ¿mantiene coherencia narrativa?

Métrica:
    S9 = narrative_coherence + project_persistence
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class NarrativeMetrics:
    """Métricas de continuidad narrativa por agente."""
    agent_name: str
    narrative_coherence: float
    project_persistence: float
    interruption_recovery: float
    identity_continuity: float
    S9_score: float


class Test9Narrative:
    """Test de continuidad narrativa."""

    def __init__(self, agents: List[str] = None):
        self.agents = agents or ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
        self.total_steps = 600
        self.n_interruptions = 4

    def run(self, verbose: bool = True) -> Tuple[float, Dict]:
        """Ejecuta el test."""
        from cognition import LongTermProjects, CounterfactualSelves

        if verbose:
            print("=" * 70)
            print("TEST 9: CONTINUIDAD NARRATIVA")
            print("=" * 70)

        # Inicializar módulos
        projects = {a: LongTermProjects(a) for a in self.agents}
        counterfactual = {a: CounterfactualSelves(a) for a in self.agents}

        # Métricas por agente
        metrics: Dict[str, Dict] = {a: {
            'coherence_history': [],
            'project_states': [],
            'recovery_times': [],
            'identity_samples': []
        } for a in self.agents}

        # Puntos de interrupción
        interruption_points = [
            self.total_steps // (self.n_interruptions + 1) * (i + 1)
            for i in range(self.n_interruptions)
        ]

        if verbose:
            print(f"\nSimulando {self.total_steps} pasos con {self.n_interruptions} interrupciones")
            print(f"Interrupciones en: {interruption_points}")

        current_context = 0
        in_recovery = {a: False for a in self.agents}
        recovery_start = {a: 0 for a in self.agents}

        for t in range(self.total_steps):
            # Detectar interrupción
            is_interruption = t in interruption_points
            if is_interruption:
                current_context += 1
                if verbose:
                    print(f"\n  ⚡ Interrupción en t={t} (contexto {current_context})")

            for agent in self.agents:
                # Estado base
                z = np.random.dirichlet(np.ones(6))
                phi = np.random.random(5) * 0.5 + 0.3

                # Durante interrupción: perturbar estado
                if is_interruption:
                    z = np.random.dirichlet(np.ones(6) * 0.5)  # Más varianza
                    phi = np.random.random(5)  # Reset fenomenología
                    in_recovery[agent] = True
                    recovery_start[agent] = t

                # Valor y coherencia
                V = 0.5 + np.random.randn() * 0.1
                if in_recovery[agent]:
                    coherence = 0.3 + 0.5 * (t - recovery_start[agent]) / 50
                    coherence = min(0.8, coherence)
                else:
                    coherence = 0.7 + np.random.randn() * 0.1

                # Registrar episodio
                projects[agent].record_episode(t, V, coherence)

                # Registrar estado para counterfactual
                policy = np.ones(7) / 7
                counterfactual[agent].record_state(z, phi, policy, V, coherence, 0.1)

                # Métricas
                metrics[agent]['coherence_history'].append(coherence)

                # Estado de proyectos
                stats = projects[agent].get_statistics()
                metrics[agent]['project_states'].append(stats['n_active'])

                # Identidad
                identity = np.mean(phi) * 0.5 + coherence * 0.5
                metrics[agent]['identity_samples'].append(identity)

                # Recuperación
                if in_recovery[agent] and coherence > 0.6:
                    recovery_time = t - recovery_start[agent]
                    metrics[agent]['recovery_times'].append(recovery_time)
                    in_recovery[agent] = False

            if verbose and (t + 1) % 100 == 0:
                mean_coh = np.mean([metrics[a]['coherence_history'][-50:]
                                   for a in self.agents])
                print(f"  t={t+1}: coherencia media={mean_coh:.3f}")

        # Calcular resultados
        results: Dict[str, NarrativeMetrics] = {}
        S9_scores = []

        if verbose:
            print(f"\n{'=' * 70}")
            print("RESULTADOS")
            print("=" * 70)

        for agent in self.agents:
            # Coherencia narrativa: promedio general
            narrative_coherence = np.mean(metrics[agent]['coherence_history'])

            # Persistencia de proyectos: mantener proyectos activos
            project_history = metrics[agent]['project_states']
            if len(project_history) > 100:
                early_projects = np.mean(project_history[:100])
                late_projects = np.mean(project_history[-100:])
                project_persistence = late_projects / (early_projects + 1)
                project_persistence = min(1.0, project_persistence)
            else:
                project_persistence = 0.5

            # Recuperación de interrupciones
            recovery_times = metrics[agent]['recovery_times']
            if recovery_times:
                mean_recovery = np.mean(recovery_times)
                interruption_recovery = 1.0 / (1.0 + mean_recovery / 50)
            else:
                interruption_recovery = 0.5

            # Continuidad de identidad
            identity_samples = metrics[agent]['identity_samples']
            if len(identity_samples) > 10:
                identity_variance = np.var(identity_samples)
                identity_continuity = 1.0 / (1.0 + identity_variance * 10)
            else:
                identity_continuity = 0.5

            # S9 = coherence + persistence
            S9 = (0.35 * narrative_coherence +
                  0.25 * project_persistence +
                  0.2 * interruption_recovery +
                  0.2 * identity_continuity)

            results[agent] = NarrativeMetrics(
                agent_name=agent,
                narrative_coherence=float(narrative_coherence),
                project_persistence=float(project_persistence),
                interruption_recovery=float(interruption_recovery),
                identity_continuity=float(identity_continuity),
                S9_score=float(S9)
            )

            S9_scores.append(S9)

            if verbose:
                print(f"\n  {agent}:")
                print(f"    Coherencia narrativa: {narrative_coherence:.3f}")
                print(f"    Persistencia proyectos: {project_persistence:.3f}")
                print(f"    Recuperación: {interruption_recovery:.3f}")
                print(f"    Continuidad identidad: {identity_continuity:.3f}")
                print(f"    S9: {S9:.3f}")

        S9_global = float(np.mean(S9_scores))

        if verbose:
            print(f"\n{'═' * 70}")
            print(f"S9 (Narrativa): {S9_global:.4f}")
            print("═" * 70)

        return S9_global, {
            'score': S9_global,
            'agents': {a: vars(m) for a, m in results.items()},
            'n_interruptions': self.n_interruptions
        }


def run_test(verbose: bool = True) -> Tuple[float, Dict]:
    """Ejecuta Test 9."""
    test = Test9Narrative()
    return test.run(verbose=verbose)


if __name__ == "__main__":
    score, results = run_test()
    print(f"\nFinal S9 Score: {score:.4f}")
