"""
TEST 6 — EMERGENCIA DE NORMAS (Norm Emergence)
===============================================

Qué mide: Inteligencia social colectiva
AGI involucrada: AGI-12 (NormEmergence)

Procedimiento:
1. Dejas 5 agentes interactuar libremente
2. Mides: ¿aparecen comportamientos estables compartidos?
3. Cuantificas: persistencia, valencia, adopción

Métrica:
    S6 = mean(W_ℓ) sobre normas emergentes
    W_ℓ = rank(persistencia) + rank(valencia)
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class NormMetrics:
    """Métricas de normas emergentes."""
    n_norms_emerged: int
    mean_persistence: float
    mean_valence: float
    adoption_rate: float
    S6_score: float


class Test6Norms:
    """Test de emergencia de normas."""

    def __init__(self, agents: List[str] = None):
        self.agents = agents or ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
        self.interaction_steps = 500
        self.min_persistence_threshold = 0.3

    def run(self, verbose: bool = True) -> Tuple[float, Dict]:
        """Ejecuta el test."""
        from cognition import NormEmergence

        if verbose:
            print("=" * 70)
            print("TEST 6: EMERGENCIA DE NORMAS")
            print("=" * 70)

        # Inicializar sistema de normas compartido
        # n_policies = 6 (dimensión de drives/políticas por agente)
        norm_system = NormEmergence(self.agents, n_policies=6)

        # Historial de acciones por agente
        action_history: Dict[str, List[np.ndarray]] = {a: [] for a in self.agents}

        if verbose:
            print(f"\nSimulando {self.interaction_steps} pasos de interacción...")

        for t in range(self.interaction_steps):
            # Políticas y valores de este paso
            agent_policies: Dict[str, np.ndarray] = {}
            agent_values: Dict[str, float] = {}

            # Cada agente actúa
            for agent in self.agents:
                # Acción base (tendencia individual)
                if agent == 'NEO':
                    base = np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.1])
                elif agent == 'EVA':
                    base = np.array([0.2, 0.3, 0.15, 0.15, 0.1, 0.1])
                elif agent == 'ALEX':
                    base = np.array([0.15, 0.15, 0.3, 0.2, 0.1, 0.1])
                elif agent == 'ADAM':
                    base = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
                else:  # IRIS
                    base = np.array([0.15, 0.15, 0.15, 0.15, 0.2, 0.2])

                # Influencia social: adoptar comportamientos de otros
                if t > 50:
                    social_influence = np.zeros(6)
                    for other in self.agents:
                        if other != agent and action_history[other]:
                            # Promedio de acciones recientes del otro
                            recent = np.mean(action_history[other][-20:], axis=0)
                            social_influence += recent
                    social_influence /= (len(self.agents) - 1)

                    # Mezclar individual + social
                    mix = 0.3 + 0.4 * (t / self.interaction_steps)  # Más social con tiempo
                    action = (1 - mix) * base + mix * social_influence
                else:
                    action = base

                # Añadir ruido
                action = action + np.random.randn(6) * 0.05
                action = np.clip(action, 0.01, None)
                action /= action.sum()

                action_history[agent].append(action)

                # Valor resultado de la acción
                V = 0.5 + np.random.randn() * 0.1
                agent_policies[agent] = action
                agent_values[agent] = V

            # Registrar políticas de todos los agentes
            norm_system.record_policies(agent_policies, agent_values)

            # Mostrar progreso
            if verbose and (t + 1) % 100 == 0:
                stats = norm_system.get_statistics()
                norms_list = stats.get('norms', [])
                pers = np.mean([n.get('persistence', 0) for n in norms_list]) if norms_list else 0
                print(f"  t={t+1}: {stats['n_norms']} normas, persistencia={pers:.3f}")

        # Análisis final
        stats = norm_system.get_statistics()

        # Calcular métricas
        n_norms = stats['n_norms']

        # Calcular persistencia y valencia desde las normas
        if n_norms > 0 and stats.get('norms'):
            mean_persistence = np.mean([n.get('persistence', 0) for n in stats['norms']])
            mean_valence = np.mean([n.get('value_corr', 0) for n in stats['norms']])
        else:
            mean_persistence = 0.0
            mean_valence = 0.0

        # Tasa de adopción: cuántos agentes siguen las normas
        if n_norms > 0:
            # Calcular adherencia a normas
            norm_vectors = []
            for norm_id, norm in norm_system.norms.items():
                norm_vectors.append(norm.eigenvector)

            adoption_scores = []
            for agent in self.agents:
                if action_history[agent]:
                    recent_actions = np.mean(action_history[agent][-50:], axis=0)
                    for norm_vec in norm_vectors:
                        # Las normas tienen dim n_agents * n_policies, extraer parte de este agente
                        agent_idx = self.agents.index(agent)
                        start = agent_idx * 6
                        end = start + 6
                        if end <= len(norm_vec):
                            agent_norm_part = norm_vec[start:end]
                            # Similitud con norma
                            similarity = np.dot(recent_actions, agent_norm_part) / (
                                np.linalg.norm(recent_actions) * np.linalg.norm(agent_norm_part) + 1e-8
                            )
                            adoption_scores.append(max(0, similarity))

            adoption_rate = np.mean(adoption_scores) if adoption_scores else 0
        else:
            adoption_rate = 0

        # S6 = mean(W_ℓ)
        # W_ℓ = rank(persistencia) + rank(valencia)
        if n_norms > 0:
            S6 = 0.4 * mean_persistence + 0.3 * mean_valence + 0.3 * adoption_rate
        else:
            S6 = 0.0

        results = NormMetrics(
            n_norms_emerged=n_norms,
            mean_persistence=float(mean_persistence),
            mean_valence=float(mean_valence),
            adoption_rate=float(adoption_rate),
            S6_score=float(S6)
        )

        if verbose:
            print(f"\n{'=' * 70}")
            print("RESULTADOS")
            print("=" * 70)
            print(f"\n  Normas emergidas: {n_norms}")
            print(f"  Persistencia media: {mean_persistence:.3f}")
            print(f"  Valencia media: {mean_valence:.3f}")
            print(f"  Tasa de adopción: {adoption_rate:.3f}")
            print(f"\n{'═' * 70}")
            print(f"S6 (Normas): {S6:.4f}")
            print("═" * 70)

        return S6, {
            'score': S6,
            'n_norms': n_norms,
            'mean_persistence': mean_persistence,
            'mean_valence': mean_valence,
            'adoption_rate': adoption_rate
        }


def run_test(verbose: bool = True) -> Tuple[float, Dict]:
    """Ejecuta Test 6."""
    test = Test6Norms()
    return test.run(verbose=verbose)


if __name__ == "__main__":
    score, results = run_test()
    print(f"\nFinal S6 Score: {score:.4f}")
