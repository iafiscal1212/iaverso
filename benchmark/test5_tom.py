"""
TEST 5 — TEORÍA DE LA MENTE (ToM Accuracy)
==========================================

Qué mide: Inteligencia social interna
AGI involucrada: Theory of Mind

Procedimiento:
1. Cambias estados internos de ALEX sin decir nada a NEO
2. NEO predice los drives/estado futuro de ALEX
3. Repetir en todos los pares

Métrica:
    S5 = corr(ŝ_A→B, s_B)
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from itertools import permutations


@dataclass
class ToMMetrics:
    """Métricas de Theory of Mind."""
    observer: str
    target: str
    prediction_correlation: float
    drive_accuracy: float
    state_accuracy: float


class Test5ToM:
    """Test de Theory of Mind."""

    def __init__(self, agents: List[str] = None):
        self.agents = agents or ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
        self.observation_steps = 200
        self.test_steps = 100
        self.n_perturbations = 3

    def run(self, verbose: bool = True) -> Tuple[float, Dict]:
        """Ejecuta el test."""
        from cognition import TheoryOfMind

        if verbose:
            print("=" * 70)
            print("TEST 5: THEORY OF MIND")
            print("=" * 70)

        # Inicializar ToM para cada agente (observa a todos los demás)
        # state_dim = 6 (drives) para observaciones de acciones
        tom = {a: TheoryOfMind(a, list(set(self.agents) - {a}), state_dim=6)
               for a in self.agents}

        # Estados reales de cada agente
        true_states: Dict[str, Dict] = {a: {
            'drives': np.ones(6) / 6,
            'phi': np.zeros(5),
            'history': []
        } for a in self.agents}

        # Métricas por par
        pair_metrics: Dict[Tuple[str, str], Dict] = {
            (a, b): {'predictions': [], 'actuals': []}
            for a in self.agents for b in self.agents if a != b
        }

        t = 0

        # Fase 1: Observación mutua
        if verbose:
            print(f"\nFase 1: Observación mutua ({self.observation_steps} pasos)")

        for step in range(self.observation_steps):
            t += 1

            # Cada agente evoluciona
            for agent in self.agents:
                z = true_states[agent]['drives']
                z = z + np.random.randn(6) * 0.02
                z = np.clip(z, 0.01, None)
                z /= z.sum()
                true_states[agent]['drives'] = z

                phi = np.random.random(5) * 0.5 + 0.3
                true_states[agent]['phi'] = phi
                true_states[agent]['history'].append(z.copy())

            # Cada agente observa a los demás
            for observer in self.agents:
                for target in self.agents:
                    if observer == target:
                        continue

                    # Observar comportamiento (acciones derivadas de drives)
                    target_z = true_states[target]['drives']

                    # Acción observable = función de z
                    action = target_z + np.random.randn(6) * 0.1
                    action = np.clip(action, 0, None)
                    action /= action.sum()

                    # Registrar estado observado y actualizar modelo
                    tom[observer].update_model(target, action)

        # Fase 2: Perturbaciones secretas y predicción
        if verbose:
            print(f"\nFase 2: {self.n_perturbations} perturbaciones secretas")

        for pert_idx in range(self.n_perturbations):
            # Seleccionar agente a perturbar
            perturbed_agent = self.agents[pert_idx % len(self.agents)]

            if verbose:
                print(f"\n  Perturbación {pert_idx + 1}: {perturbed_agent} cambia secretamente")

            # Aplicar perturbación secreta
            perturbation = np.zeros(6)
            perturbation[pert_idx % 6] = 0.4
            true_states[perturbed_agent]['drives'] = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) + perturbation
            true_states[perturbed_agent]['drives'] /= true_states[perturbed_agent]['drives'].sum()

            # Los demás predicen el estado del perturbado
            for observer in self.agents:
                if observer == perturbed_agent:
                    continue

                # Predicción sin saber de la perturbación
                predicted = tom[observer].predict_other(perturbed_agent)
                actual = true_states[perturbed_agent]['drives']

                pair_metrics[(observer, perturbed_agent)]['predictions'].append(predicted.copy())
                pair_metrics[(observer, perturbed_agent)]['actuals'].append(actual.copy())

            # Evolución durante test
            for step in range(self.test_steps // self.n_perturbations):
                t += 1

                for agent in self.agents:
                    z = true_states[agent]['drives']
                    # El perturbado vuelve gradualmente
                    if agent == perturbed_agent:
                        z = z * 0.95 + np.ones(6) / 6 * 0.05
                    z = z + np.random.randn(6) * 0.02
                    z = np.clip(z, 0.01, None)
                    z /= z.sum()
                    true_states[agent]['drives'] = z

                    phi = np.random.random(5) * 0.5 + 0.3
                    true_states[agent]['phi'] = phi

                # Observación continua
                for observer in self.agents:
                    for target in self.agents:
                        if observer == target:
                            continue

                        target_z = true_states[target]['drives']
                        action = target_z + np.random.randn(6) * 0.1
                        action = np.clip(action, 0, None)
                        action /= action.sum()

                        tom[observer].update_model(target, action)

                        # Predicciones adicionales
                        predicted = tom[observer].predict_other(target)
                        actual = true_states[target]['drives']

                        pair_metrics[(observer, target)]['predictions'].append(predicted.copy())
                        pair_metrics[(observer, target)]['actuals'].append(actual.copy())

        # Calcular resultados
        results: Dict[str, Dict[str, ToMMetrics]] = {a: {} for a in self.agents}
        all_correlations = []

        if verbose:
            print(f"\n{'=' * 70}")
            print("RESULTADOS")
            print("=" * 70)

        for observer in self.agents:
            observer_correlations = []

            for target in self.agents:
                if observer == target:
                    continue

                preds = pair_metrics[(observer, target)]['predictions']
                acts = pair_metrics[(observer, target)]['actuals']

                if not preds or not acts:
                    continue

                # Correlación de predicciones
                preds_flat = np.array(preds).flatten()
                acts_flat = np.array(acts).flatten()

                try:
                    corr = np.corrcoef(preds_flat, acts_flat)[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                except:
                    corr = 0.0

                # Precisión de drives
                drive_errors = [np.linalg.norm(p - a) for p, a in zip(preds, acts)]
                drive_accuracy = 1.0 / (1.0 + np.mean(drive_errors))

                # Precisión de estado
                state_accuracy = float(corr) if corr > 0 else 0.0

                results[observer][target] = ToMMetrics(
                    observer=observer,
                    target=target,
                    prediction_correlation=float(corr),
                    drive_accuracy=float(drive_accuracy),
                    state_accuracy=float(state_accuracy)
                )

                observer_correlations.append(corr)
                all_correlations.append(corr)

            if verbose and observer_correlations:
                mean_corr = np.mean(observer_correlations)
                print(f"\n  {observer} → otros:")
                print(f"    Correlación media: {mean_corr:.3f}")

        # Score global
        S5 = float(np.mean(all_correlations)) if all_correlations else 0.0

        if verbose:
            print(f"\n{'═' * 70}")
            print(f"S5 (Theory of Mind): {S5:.4f}")
            print("═" * 70)

        return S5, {
            'score': S5,
            'pairs': {f"{o}->{t}": vars(m)
                     for o, targets in results.items()
                     for t, m in targets.items()},
            'mean_correlation': S5
        }


def run_test(verbose: bool = True) -> Tuple[float, Dict]:
    """Ejecuta Test 5."""
    test = Test5ToM()
    return test.run(verbose=verbose)


if __name__ == "__main__":
    score, results = run_test()
    print(f"\nFinal S5 Score: {score:.4f}")
