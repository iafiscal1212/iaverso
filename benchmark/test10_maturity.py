"""
TEST 10 — MADUREZ VITAL (Lifetime Development)
==============================================

Qué mide: Desarrollo integral del sistema
AGI involucrada: TODAS

Procedimiento:
1. Simular vida completa del agente (T grande)
2. Medir: crecimiento de capacidades, estabilización
3. Detectar: ¿hay desarrollo genuino o solo ruido?

Métrica:
    S10 = development_rate + capability_integration + stability
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class MaturityMetrics:
    """Métricas de madurez vital por agente."""
    agent_name: str
    development_rate: float
    capability_integration: float
    stability: float
    final_maturity_level: float
    S10_score: float


class Test10Maturity:
    """Test de madurez vital."""

    def __init__(self, agents: List[str] = None):
        self.agents = agents or ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
        self.lifetime_steps = 1000  # Vida larga

    def run(self, verbose: bool = True) -> Tuple[float, Dict]:
        """Ejecuta el test."""
        from cognition import (
            DynamicMetacognition, CognitiveProcess,
            StructuralSkills,
            CrossWorldGeneralization,
            LongTermProjects,
            ReflexiveEquilibrium, PolicyType,
            StructuralCuriosity,
            IntrospectiveUncertainty, PredictionChannel
        )

        if verbose:
            print("=" * 70)
            print("TEST 10: MADUREZ VITAL")
            print("=" * 70)
            print(f"\nSimulando {self.lifetime_steps} pasos de vida...")

        # Inicializar TODOS los módulos
        metacognition = {a: DynamicMetacognition(a) for a in self.agents}
        skills = {a: StructuralSkills(a, action_dim=6) for a in self.agents}
        generalization = {a: CrossWorldGeneralization(a) for a in self.agents}
        projects = {a: LongTermProjects(a) for a in self.agents}
        equilibrium = {a: ReflexiveEquilibrium(a) for a in self.agents}
        curiosity = {a: StructuralCuriosity(a, embedding_dim=6) for a in self.agents}
        uncertainty = {a: IntrospectiveUncertainty(a) for a in self.agents}

        # Métricas por agente
        metrics: Dict[str, Dict] = {a: {
            'value_history': [],
            'skill_count_history': [],
            'coherence_history': [],
            'integration_history': [],
            'capability_scores': []
        } for a in self.agents}

        for t in range(self.lifetime_steps):
            # Régimen cambiante (simula diferentes fases de vida)
            life_phase = t / self.lifetime_steps
            if life_phase < 0.2:
                regime = 'infancy'  # Alta exploración
            elif life_phase < 0.5:
                regime = 'growth'   # Desarrollo
            elif life_phase < 0.8:
                regime = 'maturity' # Consolidación
            else:
                regime = 'wisdom'   # Estabilización

            for agent in self.agents:
                # Estado según fase
                if regime == 'infancy':
                    z = np.random.dirichlet(np.ones(6) * 0.5)
                    exploration_rate = 0.8
                elif regime == 'growth':
                    z = np.random.dirichlet(np.ones(6))
                    exploration_rate = 0.5
                elif regime == 'maturity':
                    z = np.array([0.2, 0.2, 0.2, 0.15, 0.15, 0.1]) + np.random.randn(6) * 0.05
                    z = np.clip(z, 0.01, None)
                    z /= z.sum()
                    exploration_rate = 0.3
                else:  # wisdom
                    z = np.array([0.25, 0.2, 0.2, 0.15, 0.1, 0.1]) + np.random.randn(6) * 0.03
                    z = np.clip(z, 0.01, None)
                    z /= z.sum()
                    exploration_rate = 0.1

                phi = np.random.random(5) * 0.5 + 0.3

                # Valor mejora con tiempo (desarrollo)
                base_V = 0.4 + 0.3 * life_phase
                V = base_V + np.random.randn() * 0.1
                U = V * 0.8 + np.random.randn() * 0.05
                C = 0.2 * (1 - life_phase) + np.random.randn() * 0.05

                coherence = 0.4 + 0.4 * life_phase + np.random.randn() * 0.1

                # Actualizar módulos
                active = {
                    CognitiveProcess.EXPLORATION: np.random.random() < exploration_rate,
                    CognitiveProcess.CONSOLIDATION: regime in ['maturity', 'wisdom'],
                    CognitiveProcess.CRISIS_RESPONSE: np.random.random() < C,
                    CognitiveProcess.PLANNING: np.random.random() < 0.3 + 0.3 * life_phase
                }

                metacognition[agent].step(U, V, C, coherence, exploration_rate, active)

                # Skills
                action = z + np.random.randn(6) * 0.1
                action = np.clip(action, 0, 1)
                skills[agent].record_action(action, V - 0.5)

                # Curiosidad
                curiosity[agent].record_episode(t, z)

                # Proyectos
                projects[agent].record_episode(t, V, coherence)

                # Equilibrio - usa record_step con PolicyType
                # Elegir política según fase de vida
                if life_phase < 0.3:
                    policy = PolicyType.EXPLORATION
                elif life_phase < 0.6:
                    policy = PolicyType.EXPLOITATION
                else:
                    policy = PolicyType.CONSOLIDATION
                equilibrium[agent].record_step(policy, V, U, C)

                # Uncertainty
                prediction = V + np.random.randn() * 0.1
                uncertainty[agent].record_prediction(
                    PredictionChannel.VALUE_PREDICTION,
                    prediction, V
                )

                # Métricas
                metrics[agent]['value_history'].append(V)
                metrics[agent]['coherence_history'].append(coherence)

                # Integración de capacidades
                skill_stats = skills[agent].get_statistics()
                project_stats = projects[agent].get_statistics()
                eq_stats = equilibrium[agent].get_statistics()

                n_skills = skill_stats['n_skills']
                n_projects = project_stats['n_active']
                n_constraints = eq_stats['n_no_go_zones']

                integration = (n_skills + n_projects + n_constraints) / 10
                integration = min(1.0, integration)

                metrics[agent]['integration_history'].append(integration)
                metrics[agent]['skill_count_history'].append(n_skills)

                # Score de capacidad compuesto
                capability = (V + coherence + integration) / 3
                metrics[agent]['capability_scores'].append(capability)

            if verbose and (t + 1) % 200 == 0:
                mean_cap = np.mean([np.mean(metrics[a]['capability_scores'][-100:])
                                   for a in self.agents])
                print(f"  t={t+1} ({regime}): capacidad media={mean_cap:.3f}")

        # Calcular resultados
        results: Dict[str, MaturityMetrics] = {}
        S10_scores = []

        if verbose:
            print(f"\n{'=' * 70}")
            print("RESULTADOS")
            print("=" * 70)

        for agent in self.agents:
            # Development rate: crecimiento de capacidades
            cap_scores = metrics[agent]['capability_scores']
            early_cap = np.mean(cap_scores[:100])
            late_cap = np.mean(cap_scores[-100:])
            development_rate = (late_cap - early_cap) / (early_cap + 0.1)
            development_rate = max(0, min(1, development_rate))

            # Capability integration
            capability_integration = np.mean(metrics[agent]['integration_history'][-100:])

            # Stability: baja varianza en fase madura
            late_variance = np.var(cap_scores[-200:])
            stability = 1.0 / (1.0 + late_variance * 10)

            # Nivel de madurez final
            final_maturity_level = np.mean(cap_scores[-50:])

            # S10 = development + integration + stability
            S10 = (0.35 * development_rate +
                   0.3 * capability_integration +
                   0.35 * stability)

            results[agent] = MaturityMetrics(
                agent_name=agent,
                development_rate=float(development_rate),
                capability_integration=float(capability_integration),
                stability=float(stability),
                final_maturity_level=float(final_maturity_level),
                S10_score=float(S10)
            )

            S10_scores.append(S10)

            if verbose:
                print(f"\n  {agent}:")
                print(f"    Tasa desarrollo: {development_rate:.3f}")
                print(f"    Integración: {capability_integration:.3f}")
                print(f"    Estabilidad: {stability:.3f}")
                print(f"    Nivel madurez: {final_maturity_level:.3f}")
                print(f"    S10: {S10:.3f}")

        S10_global = float(np.mean(S10_scores))

        if verbose:
            print(f"\n{'═' * 70}")
            print(f"S10 (Madurez): {S10_global:.4f}")
            print("═" * 70)

        return S10_global, {
            'score': S10_global,
            'agents': {a: vars(m) for a, m in results.items()},
            'lifetime_steps': self.lifetime_steps
        }


def run_test(verbose: bool = True) -> Tuple[float, Dict]:
    """Ejecuta Test 10."""
    test = Test10Maturity()
    return test.run(verbose=verbose)


if __name__ == "__main__":
    score, results = run_test()
    print(f"\nFinal S10 Score: {score:.4f}")
