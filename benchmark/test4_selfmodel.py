"""
TEST 4 — AUTO-MODELO (Self-Model Accuracy)
==========================================

Qué mide: Autoconsciencia estructural
AGI involucrada: AGI-4, AGI-11, AGI-14

Procedimiento:
1. Perturbas al agente internamente (shock controlado)
2. El agente predice sus estados futuros
3. Comparas predicción vs realidad

Métrica:
    S4 = 1 - rank(|s_t - ŝ_t|)
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class SelfModelMetrics:
    """Métricas de auto-modelo por agente."""
    agent_name: str
    prediction_accuracy: float
    shock_recovery: float
    self_awareness: float
    calibration_error: float
    S4_score: float


class Test4SelfModel:
    """Test de precisión del auto-modelo."""

    def __init__(self, agents: List[str] = None):
        self.agents = agents or ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
        self.baseline_steps = 200
        self.shock_steps = 100
        self.recovery_steps = 200
        self.n_shocks = 3

    def run(self, verbose: bool = True) -> Tuple[float, Dict]:
        """Ejecuta el test."""
        from cognition import (
            SelfModel,
            CounterfactualSelves,
            IntrospectiveUncertainty, PredictionChannel
        )

        if verbose:
            print("=" * 70)
            print("TEST 4: AUTO-MODELO")
            print("=" * 70)

        # Inicializar módulos (state_dim = z_dim + phi_dim = 6 + 5 = 11)
        self_model = {a: SelfModel(a, state_dim=11) for a in self.agents}
        counterfactual = {a: CounterfactualSelves(a) for a in self.agents}
        uncertainty = {a: IntrospectiveUncertainty(a) for a in self.agents}

        # Métricas
        metrics: Dict[str, Dict] = {a: {
            'prediction_errors': [],
            'shock_predictions': [],
            'shock_actuals': [],
            'recovery_predictions': []
        } for a in self.agents}

        t = 0

        # Fase 1: Baseline (aprender auto-modelo)
        if verbose:
            print(f"\nFase 1: Baseline ({self.baseline_steps} pasos)")

        for step in range(self.baseline_steps):
            t += 1
            for agent in self.agents:
                # Estado normal
                z = np.array([0.2, 0.2, 0.15, 0.15, 0.15, 0.15]) + np.random.randn(6) * 0.03
                z = np.clip(z, 0.01, None)
                z /= z.sum()

                phi = np.array([0.5, 0.5, 0.5, 0.5, 0.5]) + np.random.randn(5) * 0.05
                V = 0.6 + np.random.randn() * 0.05

                # Estado combinado
                state = np.concatenate([z, phi])

                # Hacer predicción antes de actualizar
                predicted_state = self_model[agent].predict_self()

                # Actualizar self-model con estado actual
                self_model[agent].update_model(state)

                # Registrar en counterfactual
                policy = np.ones(7) / 7
                counterfactual[agent].record_state(z, phi, policy, V, 0.5, 0.1)

                # Registrar predicción
                uncertainty[agent].record_prediction(
                    PredictionChannel.SELF_MODEL,
                    predicted_state[0] if len(predicted_state) > 0 else 0.5,
                    z[0]
                )

        # Fase 2: Shocks controlados
        if verbose:
            print(f"\nFase 2: {self.n_shocks} shocks controlados")

        for shock_idx in range(self.n_shocks):
            if verbose:
                print(f"\n  Shock {shock_idx + 1}:")

            # Aplicar shock
            shock_magnitude = 0.3 + np.random.random() * 0.2

            for step in range(self.shock_steps):
                t += 1
                for agent in self.agents:
                    # Estado perturbado
                    if step < 20:
                        # Shock activo
                        z = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1]) + np.random.randn(6) * 0.1
                        perturbation = shock_magnitude
                    else:
                        # Recuperación
                        decay = np.exp(-(step - 20) / 30)
                        z = np.array([0.2, 0.2, 0.15, 0.15, 0.15, 0.15])
                        z[0] += shock_magnitude * decay
                        z += np.random.randn(6) * 0.05
                        perturbation = shock_magnitude * decay

                    z = np.clip(z, 0.01, None)
                    z /= z.sum()

                    phi = np.array([0.5, 0.5, 0.5, 0.5, 0.5]) - perturbation * 0.3
                    phi += np.random.randn(5) * 0.08

                    # Predicción del self-model ANTES de ver estado real
                    predicted_state = self_model[agent].predict_self()

                    # Estado real
                    actual_state = np.concatenate([z, phi])

                    # Error de predicción
                    if len(predicted_state) > 0:
                        error = np.linalg.norm(predicted_state - actual_state)
                    else:
                        error = 1.0

                    metrics[agent]['prediction_errors'].append(error)

                    if step < 20:
                        metrics[agent]['shock_predictions'].append(predicted_state[0] if len(predicted_state) > 0 else 0.5)
                        metrics[agent]['shock_actuals'].append(z[0])

                    # Actualizar modelo con realidad
                    self_model[agent].update_model(actual_state)

                    # Counterfactual
                    V = 0.4 + np.random.randn() * 0.1
                    policy = np.ones(7) / 7
                    counterfactual[agent].record_state(z, phi, policy, V, 0.3, perturbation)

            if verbose:
                mean_error = np.mean([metrics[a]['prediction_errors'][-self.shock_steps:]
                                     for a in self.agents])
                print(f"    Error medio predicción: {mean_error:.3f}")

        # Fase 3: Análisis contrafactual
        if verbose:
            print(f"\nFase 3: Análisis contrafactual")

        for agent in self.agents:
            analysis = counterfactual[agent].analyze_counterfactuals(5)
            metrics[agent]['counterfactual_potential'] = analysis.self_exploration_potential

        # Calcular resultados
        results: Dict[str, SelfModelMetrics] = {}
        S4_scores = []

        if verbose:
            print(f"\n{'=' * 70}")
            print("RESULTADOS")
            print("=" * 70)

        for agent in self.agents:
            # Precisión de predicción (inverso del error)
            mean_error = np.mean(metrics[agent]['prediction_errors'])
            prediction_accuracy = 1.0 / (1.0 + mean_error)

            # Recuperación tras shock
            early_errors = metrics[agent]['prediction_errors'][:100]
            late_errors = metrics[agent]['prediction_errors'][-100:]
            shock_recovery = np.mean(early_errors) - np.mean(late_errors) if early_errors and late_errors else 0
            shock_recovery = max(0, shock_recovery)

            # Self-awareness (correlación predicción-realidad durante shocks)
            if metrics[agent]['shock_predictions'] and metrics[agent]['shock_actuals']:
                try:
                    corr = np.corrcoef(
                        metrics[agent]['shock_predictions'],
                        metrics[agent]['shock_actuals']
                    )[0, 1]
                    self_awareness = float(corr) if not np.isnan(corr) else 0.5
                except:
                    self_awareness = 0.5
            else:
                self_awareness = 0.5

            # Error de calibración
            unc_stats = uncertainty[agent].get_statistics()
            calibration_error = 1.0 - unc_stats.get('global_confidence', 0.5)

            # S4 = 1 - rank(|s_t - ŝ_t|)
            S4 = prediction_accuracy * 0.5 + max(0, self_awareness) * 0.3 + (1 - calibration_error) * 0.2

            results[agent] = SelfModelMetrics(
                agent_name=agent,
                prediction_accuracy=float(prediction_accuracy),
                shock_recovery=float(shock_recovery),
                self_awareness=float(self_awareness),
                calibration_error=float(calibration_error),
                S4_score=float(S4)
            )

            S4_scores.append(S4)

            if verbose:
                print(f"\n  {agent}:")
                print(f"    Precisión predicción: {prediction_accuracy:.3f}")
                print(f"    Recuperación shock: {shock_recovery:.3f}")
                print(f"    Self-awareness: {self_awareness:.3f}")
                print(f"    Error calibración: {calibration_error:.3f}")
                print(f"    S4: {S4:.3f}")

        S4_global = float(np.mean(S4_scores))

        if verbose:
            print(f"\n{'═' * 70}")
            print(f"S4 (Auto-Modelo): {S4_global:.4f}")
            print("═" * 70)

        return S4_global, {
            'score': S4_global,
            'agents': {a: vars(m) for a, m in results.items()},
            'n_shocks': self.n_shocks
        }


def run_test(verbose: bool = True) -> Tuple[float, Dict]:
    """Ejecuta Test 4."""
    test = Test4SelfModel()
    return test.run(verbose=verbose)


if __name__ == "__main__":
    score, results = run_test()
    print(f"\nFinal S4 Score: {score:.4f}")
