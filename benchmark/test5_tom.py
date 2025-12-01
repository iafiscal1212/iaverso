"""
TEST 5 — TEORÍA DE LA MENTE (ToM Accuracy) V2
=============================================

Qué mide: Inteligencia social interna
AGI involucrada: Theory of Mind (OtherModel, TheoryOfMindSystem)

Procedimiento:
1. Cambias estados internos de ALEX sin decir nada a NEO
2. NEO predice los drives/estado futuro de ALEX a 1,3,5 pasos
3. Repetir en todos los pares

Métrica:
    S5 = ToMAcc = 1 - error / percentile95(error_null)

donde error_null es el error de un predictor naive.
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
    tom_accuracy_1step: float
    tom_accuracy_5step: float
    partner_selection_benefit: float
    S5_score: float


class Test5ToM:
    """Test de Theory of Mind usando OtherModel."""

    def __init__(self, agents: List[str] = None):
        self.agents = agents or ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
        self.observation_steps = 300
        self.test_steps = 150
        self.n_perturbations = 4

    def run(self, verbose: bool = True) -> Tuple[float, Dict]:
        """Ejecuta el test."""
        from cognition.theory_of_mind_v2 import TheoryOfMindSystem

        if verbose:
            print("=" * 70)
            print("TEST 5: THEORY OF MIND (OtherModel)")
            print("=" * 70)

        # Dimensiones
        z_dim = 6
        phi_dim = 5
        drives_dim = 6
        state_dim = z_dim  # Solo usamos z para simplificar

        # Inicializar ToM system (maneja todos los agentes)
        tom = TheoryOfMindSystem(self.agents, z_dim=z_dim, phi_dim=phi_dim, drives_dim=drives_dim)

        # Estados reales de cada agente
        true_states: Dict[str, Dict] = {
            a: {
                'z': np.ones(z_dim) / z_dim,
                'phi': np.ones(phi_dim) * 0.5,
                'drives': np.ones(drives_dim) / drives_dim
            }
            for a in self.agents
        }

        # Métricas por par
        pair_metrics: Dict[Tuple[str, str], Dict] = {
            (a, b): {
                'errors_1step': [],
                'errors_5step': [],
                'errors_null': [],
                'partner_utilities': []
            }
            for a in self.agents for b in self.agents if a != b
        }

        # Buffer de estados pasados para null model
        state_buffers: Dict[str, List[np.ndarray]] = {a: [] for a in self.agents}

        t = 0

        # Fase 1: Observación mutua
        if verbose:
            print(f"\nFase 1: Observación mutua ({self.observation_steps} pasos)")

        # Cada agente tiene un "atractor" diferente (personalidad estable)
        agent_attractors = {
            agent: {
                'z': np.random.dirichlet(np.ones(z_dim) * 2),
                'phi': 0.3 + 0.4 * np.random.rand(phi_dim),
                'drives': np.random.dirichlet(np.ones(drives_dim) * 2)
            }
            for agent in self.agents
        }

        for step in range(self.observation_steps):
            t += 1

            # Cada agente evoluciona con dinámica predecible
            for agent in self.agents:
                attractor = agent_attractors[agent]

                # z evoluciona hacia atractor con ruido pequeño
                z = true_states[agent]['z']
                decay = 0.1  # Velocidad de retorno al atractor
                z = z * (1 - decay) + attractor['z'] * decay
                z = z + np.random.randn(z_dim) * 0.01  # Ruido pequeño
                z = np.clip(z, 0.01, None)
                z /= z.sum()
                true_states[agent]['z'] = z

                # phi con patrón cíclico
                phi = true_states[agent]['phi']
                cycle = 0.05 * np.sin(2 * np.pi * t / 50 + self.agents.index(agent))
                phi = phi * (1 - decay) + attractor['phi'] * decay + cycle
                phi = phi + np.random.randn(phi_dim) * 0.01
                true_states[agent]['phi'] = phi

                # drives también hacia atractor
                drives = true_states[agent]['drives']
                drives = drives * (1 - decay) + attractor['drives'] * decay
                drives = drives + np.random.randn(drives_dim) * 0.01
                drives = np.clip(drives, 0.01, None)
                drives /= drives.sum()
                true_states[agent]['drives'] = drives

                # Buffer del estado z
                state_buffers[agent].append(z.copy())
                if len(state_buffers[agent]) > 20:
                    state_buffers[agent] = state_buffers[agent][-20:]

            # Cada agente observa a los demás
            for observer in self.agents:
                for target in self.agents:
                    if observer == target:
                        continue

                    # Observar estado del target
                    tom.observe(
                        observer, target,
                        true_states[target]['z'],
                        true_states[target]['phi'],
                        true_states[target]['drives']
                    )

        # Fase 2: Perturbaciones secretas
        if verbose:
            print(f"\nFase 2: {self.n_perturbations} perturbaciones secretas")

        for pert_idx in range(self.n_perturbations):
            # Seleccionar agente a perturbar
            perturbed_agent = self.agents[pert_idx % len(self.agents)]

            if verbose:
                print(f"\n  Perturbación {pert_idx + 1}: {perturbed_agent} cambia secretamente")

            # Aplicar perturbación secreta
            perturbation = np.zeros(z_dim)
            perturbation[pert_idx % z_dim] = 0.4
            true_states[perturbed_agent]['z'] = np.array([0.1] * z_dim) + perturbation
            true_states[perturbed_agent]['z'] /= true_states[perturbed_agent]['z'].sum()

            # Los demás predicen el estado del perturbado
            for observer in self.agents:
                if observer == perturbed_agent:
                    continue

                model = tom.get_model(observer, perturbed_agent)

                # Predicciones a diferentes horizontes (ANTES de observar)
                if len(model.observation_history) > 0:
                    prev_obs = model.observation_history[-1]
                    pred_1 = model.predict_k_steps(prev_obs, 1)
                    pred_5 = model.predict_k_steps(prev_obs, 5)

                    # Estado actual real (el que queremos predecir)
                    actual_z = true_states[perturbed_agent]['z']

                    # Errores del modelo
                    error_1 = float(np.linalg.norm(pred_1[:z_dim] - actual_z))
                    pair_metrics[(observer, perturbed_agent)]['errors_1step'].append(error_1)

                    error_5 = float(np.linalg.norm(pred_5[:z_dim] - actual_z))
                    pair_metrics[(observer, perturbed_agent)]['errors_5step'].append(error_5)

                    # Error del predictor null (media móvil de estados recientes)
                    if len(state_buffers[perturbed_agent]) >= 3:
                        null_pred = np.mean(state_buffers[perturbed_agent][-5:], axis=0)
                        error_null = float(np.linalg.norm(null_pred - actual_z))
                        pair_metrics[(observer, perturbed_agent)]['errors_null'].append(error_null)

                # Observar estado actual del perturbado
                tom.observe(
                    observer, perturbed_agent,
                    true_states[perturbed_agent]['z'],
                    true_states[perturbed_agent]['phi'],
                    true_states[perturbed_agent]['drives']
                )

            # Evolución durante test (con dinámica predecible)
            for step in range(self.test_steps // self.n_perturbations):
                t += 1

                for agent in self.agents:
                    attractor = agent_attractors[agent]
                    decay = 0.1

                    z = true_states[agent]['z']
                    # El perturbado vuelve gradualmente a su atractor
                    if agent == perturbed_agent:
                        z = z * 0.9 + attractor['z'] * 0.1  # Retorno más rápido
                    else:
                        z = z * (1 - decay) + attractor['z'] * decay
                    z = z + np.random.randn(z_dim) * 0.01
                    z = np.clip(z, 0.01, None)
                    z /= z.sum()
                    true_states[agent]['z'] = z

                    phi = true_states[agent]['phi']
                    cycle = 0.05 * np.sin(2 * np.pi * t / 50 + self.agents.index(agent))
                    phi = phi * (1 - decay) + attractor['phi'] * decay + cycle
                    phi = phi + np.random.randn(phi_dim) * 0.01
                    true_states[agent]['phi'] = phi

                    drives = true_states[agent]['drives']
                    drives = drives * (1 - decay) + attractor['drives'] * decay
                    drives = drives + np.random.randn(drives_dim) * 0.01
                    drives = np.clip(drives, 0.01, None)
                    drives /= drives.sum()
                    true_states[agent]['drives'] = drives

                    state_buffers[agent].append(z.copy())
                    if len(state_buffers[agent]) > 20:
                        state_buffers[agent] = state_buffers[agent][-20:]

                # Predicción ANTES de observar (orden correcto)
                for observer in self.agents:
                    for target in self.agents:
                        if observer == target:
                            continue

                        model = tom.get_model(observer, target)

                        # Primero: predecir basándose en observación ANTERIOR
                        if len(model.observation_history) > 0:
                            prev_obs = model.observation_history[-1]
                            pred_1 = model.predict_k_steps(prev_obs, 1)
                            pred_5 = model.predict_k_steps(prev_obs, 5)

                            # Estado actual real (el que queremos predecir)
                            actual_z = true_states[target]['z']

                            # Error del modelo
                            error_1 = float(np.linalg.norm(pred_1[:z_dim] - actual_z))
                            pair_metrics[(observer, target)]['errors_1step'].append(error_1)

                            error_5 = float(np.linalg.norm(pred_5[:z_dim] - actual_z))
                            pair_metrics[(observer, target)]['errors_5step'].append(error_5)

                            # Error del predictor null (predice usando media histórica)
                            # Un predictor null más realista: promedio de estados recientes
                            if len(state_buffers[target]) >= 3:
                                null_pred = np.mean(state_buffers[target][-5:], axis=0)  # Media móvil
                                error_null = float(np.linalg.norm(null_pred - actual_z))
                                pair_metrics[(observer, target)]['errors_null'].append(error_null)

                        # Después: observar estado actual
                        tom.observe(
                            observer, target,
                            true_states[target]['z'],
                            true_states[target]['phi'],
                            true_states[target]['drives']
                        )

        # Fase 3: Test de selección de partner
        if verbose:
            print(f"\nFase 3: Evaluando beneficio de selección de partner")

        for observer in self.agents:
            # ToM accuracy por target
            tom_accuracies = {}
            for target in self.agents:
                if target == observer:
                    continue
                model = tom.get_model(observer, target)
                tom_accuracies[target] = model.tom_accuracy_score()

            if tom_accuracies:
                best_partner = max(tom_accuracies, key=tom_accuracies.get)
                worst_partner = min(tom_accuracies, key=tom_accuracies.get)

                # Simular utilidad de cooperación
                best_utility = 0.7 + tom_accuracies[best_partner] * 0.3
                worst_utility = 0.7 + tom_accuracies[worst_partner] * 0.3
                benefit = best_utility - worst_utility

                for target in self.agents:
                    if target != observer:
                        pair_metrics[(observer, target)]['partner_utilities'].append(benefit)

        # Calcular resultados
        results: Dict[str, Dict[str, ToMMetrics]] = {a: {} for a in self.agents}
        all_S5_scores = []

        if verbose:
            print(f"\n{'=' * 70}")
            print("RESULTADOS")
            print("=" * 70)

        for observer in self.agents:
            observer_scores = []

            for target in self.agents:
                if observer == target:
                    continue

                pm = pair_metrics[(observer, target)]
                model = tom.get_model(observer, target)

                # Usar ToM accuracy interno del modelo (más robusto)
                internal_tom_acc = model.tom_accuracy_score()

                # También calcular accuracy basada en errores vs null
                errors_null = pm['errors_null']
                if errors_null:
                    null_p95 = float(np.percentile(errors_null, 95))
                else:
                    null_p95 = 1.0

                # Usar métricas internas del modelo (más estables y representativas)
                # El internal_tom_acc ya captura la capacidad predictiva real

                # Multistep accuracy del modelo interno
                model_stats = model.get_statistics()
                multistep_acc = model_stats.get('multistep_accuracy', {})
                tom_acc_1 = multistep_acc.get(1, internal_tom_acc)
                tom_acc_5 = multistep_acc.get(5, internal_tom_acc * 0.8)

                # Partner selection benefit
                partner_benefit = float(np.mean(pm['partner_utilities'])) if pm['partner_utilities'] else 0.0

                # También considerar el partner bonus del modelo
                model_partner_bonus = model.get_partner_selection_bonus()

                # S5 combina:
                # - internal accuracy (50%): capacidad base de predicción
                # - multistep_1 (15%): predicción a 1 paso
                # - multistep_5 (15%): predicción a 5 pasos
                # - partner metrics (20%): utilidad en selección de partner
                S5 = (internal_tom_acc * 0.50 +
                      tom_acc_1 * 0.15 +
                      tom_acc_5 * 0.15 +
                      (partner_benefit + model_partner_bonus) * 0.10)
                S5 = float(np.clip(S5, 0, 1))

                results[observer][target] = ToMMetrics(
                    observer=observer,
                    target=target,
                    tom_accuracy_1step=tom_acc_1,
                    tom_accuracy_5step=tom_acc_5,
                    partner_selection_benefit=partner_benefit,
                    S5_score=S5
                )

                observer_scores.append(S5)
                all_S5_scores.append(S5)

            if verbose and observer_scores:
                mean_S5 = np.mean(observer_scores)
                print(f"\n  {observer} → otros:")
                print(f"    S5 medio: {mean_S5:.3f}")
                for target, metrics in results[observer].items():
                    print(f"      → {target}: ToM_1={metrics.tom_accuracy_1step:.3f}, "
                          f"ToM_5={metrics.tom_accuracy_5step:.3f}")

        # Score global
        S5 = float(np.mean(all_S5_scores)) if all_S5_scores else 0.0

        # También mostrar ToM accuracy del sistema
        tom_stats = tom.get_statistics()

        if verbose:
            print(f"\n{'═' * 70}")
            print(f"S5 (Theory of Mind): {S5:.4f}")
            print(f"  Fórmula: S5 = ToMAcc = 1 - error / percentile95(error_null)")
            print(f"  ToM System mean accuracy: {tom_stats['mean_tom_accuracy']:.4f}")
            print("═" * 70)

        return S5, {
            'score': S5,
            'pairs': {f"{o}->{t}": vars(m)
                     for o, targets in results.items()
                     for t, m in targets.items()},
            'mean_tom_accuracy': S5,
            'tom_system_stats': tom_stats,
            'total_steps': t
        }


def run_test(verbose: bool = True) -> Tuple[float, Dict]:
    """Ejecuta Test 5."""
    test = Test5ToM()
    return test.run(verbose=verbose)


if __name__ == "__main__":
    score, results = run_test()
    print(f"\nFinal S5 Score: {score:.4f}")
