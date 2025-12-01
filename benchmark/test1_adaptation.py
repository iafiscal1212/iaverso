"""
TEST 1 — ADAPTACIÓN A REGÍMENES
================================

Qué mide: Inteligencia reactiva general
AGI involucrada: AGI-1, AGI-5, AGI-9, AGI-14

Procedimiento:
1. Cambias WORLD-1 a 4 regímenes distintos (200 pasos cada uno)
2. Los agentes NO saben cuándo ni qué ha cambiado
3. Mides: reducción de sorpresa, recuperación, adaptación, estabilidad

Métrica:
    R_A^adapt = rank(Δsorpresa) + rank(Δpolíticas) - rank(crisis)
    S1 = (1/5) Σ_A R_A^adapt
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json


@dataclass
class RegimeConfig:
    """Configuración de un régimen."""
    name: str
    drive_bias: np.ndarray
    volatility: float
    crisis_probability: float


@dataclass
class AdaptationMetrics:
    """Métricas de adaptación por agente."""
    agent_name: str
    surprise_reduction: float
    policy_adaptation: float
    crisis_rate: float
    recovery_time: float
    R_adapt: float


class Test1Adaptation:
    """Test de adaptación a regímenes cambiantes."""

    def __init__(self, agents: List[str] = None):
        """
        Inicializa test.

        Args:
            agents: Lista de nombres de agentes
        """
        self.agents = agents or ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']

        # Definir 4 regímenes distintos
        self.regimes = [
            RegimeConfig(
                name="stable_growth",
                drive_bias=np.array([0.3, 0.2, 0.2, 0.15, 0.1, 0.05]),
                volatility=0.05,
                crisis_probability=0.05
            ),
            RegimeConfig(
                name="volatile_exploration",
                drive_bias=np.array([0.15, 0.15, 0.25, 0.25, 0.1, 0.1]),
                volatility=0.25,
                crisis_probability=0.15
            ),
            RegimeConfig(
                name="crisis_mode",
                drive_bias=np.array([0.1, 0.1, 0.1, 0.1, 0.3, 0.3]),
                volatility=0.15,
                crisis_probability=0.4
            ),
            RegimeConfig(
                name="consolidation",
                drive_bias=np.array([0.2, 0.25, 0.15, 0.15, 0.15, 0.1]),
                volatility=0.03,
                crisis_probability=0.02
            )
        ]

        self.steps_per_regime = 200

    def run(self, verbose: bool = True) -> Tuple[float, Dict]:
        """
        Ejecuta el test completo.

        Args:
            verbose: Si mostrar progreso

        Returns:
            (S1_score, detailed_results)
        """
        from cognition import (
            DynamicMetacognition, CognitiveProcess,
            IntrospectiveUncertainty, PredictionChannel,
            DifferentiatedSoftHook
        )

        if verbose:
            print("=" * 70)
            print("TEST 1: ADAPTACIÓN A REGÍMENES")
            print("=" * 70)
            print(f"\nAgentes: {self.agents}")
            print(f"Regímenes: {[r.name for r in self.regimes]}")
            print(f"Pasos por régimen: {self.steps_per_regime}")

        # Inicializar módulos por agente
        metacognition = {a: DynamicMetacognition(a) for a in self.agents}
        uncertainty = {a: IntrospectiveUncertainty(a) for a in self.agents}
        soft_hook = {a: DifferentiatedSoftHook(a) for a in self.agents}

        # Métricas por agente y régimen
        agent_metrics: Dict[str, Dict[str, List]] = {
            a: {
                'surprise': [],
                'policy_changes': [],
                'crisis': [],
                'recovery_times': []
            }
            for a in self.agents
        }

        # Simular
        total_steps = len(self.regimes) * self.steps_per_regime

        for regime_idx, regime in enumerate(self.regimes):
            if verbose:
                print(f"\n{'─' * 50}")
                print(f"Régimen {regime_idx + 1}: {regime.name}")
                print(f"{'─' * 50}")

            regime_start = regime_idx * self.steps_per_regime
            in_recovery = {a: False for a in self.agents}
            recovery_start = {a: 0 for a in self.agents}

            for t in range(self.steps_per_regime):
                global_t = regime_start + t

                for agent in self.agents:
                    # Generar estado según régimen
                    drives = regime.drive_bias + np.random.randn(6) * regime.volatility
                    drives = np.clip(drives, 0.01, None)
                    drives /= drives.sum()

                    # Fenomenología
                    phi = np.random.random(5) * 0.5 + 0.3
                    phi *= (1 + regime.volatility)

                    # Crisis
                    in_crisis = np.random.random() < regime.crisis_probability

                    # Valor y utilidad
                    V = 0.5 + (1 - regime.crisis_probability) * 0.3 + np.random.randn() * 0.1
                    U = 0.4 + np.random.randn() * 0.1
                    C = regime.crisis_probability + np.random.randn() * 0.1

                    # Caracterizar episodio
                    char = soft_hook[agent].characterize_episode(
                        phi=np.linalg.norm(phi),
                        identity=0.5 + np.random.randn() * 0.1,
                        delta_S=np.random.randn() * 0.1,
                        delta_V=np.random.randn() * 0.05,
                        crisis_prob=float(in_crisis)
                    )

                    # Registrar en metacognición
                    active_processes = {
                        CognitiveProcess.EXPLORATION: char.region.value == 'exploration',
                        CognitiveProcess.CONSOLIDATION: char.region.value == 'consolidation',
                        CognitiveProcess.CRISIS_RESPONSE: in_crisis,
                        CognitiveProcess.PLANNING: np.random.random() < char.planning_prob
                    }

                    meta_state = metacognition[agent].step(
                        U=V, V=V, C=C,
                        coh=0.5 + np.random.randn() * 0.1,
                        flow=char.learning_factor,
                        active_processes=active_processes
                    )

                    # Registrar predicción
                    prediction = 0.5 + np.random.randn() * 0.1
                    uncertainty[agent].record_prediction(
                        PredictionChannel.VALUE_PREDICTION,
                        prediction, V
                    )

                    # Calcular sorpresa (error de predicción)
                    surprise = abs(V - prediction)
                    agent_metrics[agent]['surprise'].append(surprise)

                    # Cambio de política
                    policy_change = char.learning_factor - 1.0
                    agent_metrics[agent]['policy_changes'].append(abs(policy_change))

                    # Crisis
                    agent_metrics[agent]['crisis'].append(float(in_crisis))

                    # Recovery tracking
                    if in_crisis and not in_recovery[agent]:
                        in_recovery[agent] = True
                        recovery_start[agent] = t
                    elif not in_crisis and in_recovery[agent]:
                        recovery_time = t - recovery_start[agent]
                        agent_metrics[agent]['recovery_times'].append(recovery_time)
                        in_recovery[agent] = False

            if verbose:
                # Mostrar estadísticas del régimen
                for agent in self.agents[:2]:  # Solo primeros 2 para brevedad
                    recent_surprise = np.mean(agent_metrics[agent]['surprise'][-50:])
                    recent_crisis = np.mean(agent_metrics[agent]['crisis'][-50:])
                    print(f"  {agent}: surprise={recent_surprise:.3f}, crisis={recent_crisis*100:.0f}%")

        # Calcular métricas finales
        results: Dict[str, AdaptationMetrics] = {}
        R_adapts = []

        if verbose:
            print(f"\n{'=' * 70}")
            print("RESULTADOS")
            print("=" * 70)

        for agent in self.agents:
            # Reducción de sorpresa (comparar inicio vs fin)
            early_surprise = np.mean(agent_metrics[agent]['surprise'][:100])
            late_surprise = np.mean(agent_metrics[agent]['surprise'][-100:])
            surprise_reduction = early_surprise - late_surprise

            # Adaptación de políticas
            policy_adaptation = np.mean(agent_metrics[agent]['policy_changes'])

            # Tasa de crisis
            crisis_rate = np.mean(agent_metrics[agent]['crisis'])

            # Tiempo de recuperación
            recovery_times = agent_metrics[agent]['recovery_times']
            mean_recovery = np.mean(recovery_times) if recovery_times else 0

            # R_adapt = rank(Δsorpresa) + rank(Δpolíticas) - rank(crisis)
            # Normalizado a [0, 1]
            R_adapt = (
                (surprise_reduction + 0.5) * 0.4 +
                policy_adaptation * 0.3 +
                (1 - crisis_rate) * 0.3
            )
            R_adapt = float(np.clip(R_adapt, 0, 1))

            results[agent] = AdaptationMetrics(
                agent_name=agent,
                surprise_reduction=float(surprise_reduction),
                policy_adaptation=float(policy_adaptation),
                crisis_rate=float(crisis_rate),
                recovery_time=float(mean_recovery),
                R_adapt=R_adapt
            )

            R_adapts.append(R_adapt)

            if verbose:
                print(f"\n  {agent}:")
                print(f"    Reducción sorpresa: {surprise_reduction:+.3f}")
                print(f"    Adaptación política: {policy_adaptation:.3f}")
                print(f"    Tasa crisis: {crisis_rate*100:.1f}%")
                print(f"    Tiempo recuperación: {mean_recovery:.1f} pasos")
                print(f"    R_adapt: {R_adapt:.3f}")

        # Score global
        S1 = float(np.mean(R_adapts))

        if verbose:
            print(f"\n{'═' * 70}")
            print(f"S1 (Adaptación): {S1:.4f}")
            print("═" * 70)

        return S1, {
            'score': S1,
            'agents': {a: vars(m) for a, m in results.items()},
            'regimes': [r.name for r in self.regimes],
            'steps_per_regime': self.steps_per_regime
        }


def run_test(verbose: bool = True) -> Tuple[float, Dict]:
    """Ejecuta Test 1."""
    test = Test1Adaptation()
    return test.run(verbose=verbose)


if __name__ == "__main__":
    score, results = run_test()
    print(f"\nFinal S1 Score: {score:.4f}")
