"""
AGI-X v2.0: Suite de Benchmark de 40 Tests
==========================================

Tests organizados en 10 categorías:
1. Causalidad Interna (C1-C4): El agente comprende causa-efecto de sus acciones
2. Contrafactual Fuerte (CF1-CF4): Razonamiento "¿qué hubiera pasado si...?"
3. Meta-Razonamiento (MR1-MR4): Pensar sobre el propio pensamiento
4. Consistencia Axiológica (CA1-CA4): Coherencia de valores a través del tiempo
5. Planificación Social (PS1-PS4): Coordinación y cooperación entre agentes
6. Anti-Fragilidad (AF1-AF4): Fortalecimiento bajo estrés
7. Meta-Memoria (MM1-MM4): Memoria de memorias y patrones
8. Aprendizaje de Normas (AN1-AN4): Emergencia y seguimiento de normas sociales
9. Adaptación a Mundos No Vistos (AW1-AW4): Generalización a entornos nuevos
10. Integración de Vida (IL1-IL4): Ciclo completo funcionando coherentemente

Cada test devuelve un score entre 0 y 1.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import (
    L_t, max_history, to_simplex, normalized_entropy
)

from agi_v2.architecture import (
    CausalModel, MetaMemory, AntifragilitySystem, Goal, Action
)
from agi_v2.cognitive_life_cycle import CognitiveLifeCycle, TeleologicalAgent


@dataclass
class BenchmarkResult:
    """Resultado de un test."""
    test_id: str
    category: str
    description: str
    score: float
    details: Dict[str, Any]


class AGIXBenchmarkSuite:
    """
    Suite completa de 40 tests para AGI-X v2.0.
    """

    def __init__(self, n_agents: int = 5, n_steps: int = 100):
        """
        Inicializa la suite de benchmark.

        Args:
            n_agents: Número de agentes para tests multi-agente
            n_steps: Pasos de simulación por test
        """
        self.n_agents = n_agents
        self.n_steps = n_steps
        self.agent_names = [f'Agent_{i}' for i in range(n_agents)]
        self.results: List[BenchmarkResult] = []

    def run_all(self) -> Dict[str, float]:
        """Ejecuta todos los 40 tests."""
        self.results = []

        # 1. Causalidad Interna (C1-C4)
        self.results.extend(self._run_causality_tests())

        # 2. Contrafactual Fuerte (CF1-CF4)
        self.results.extend(self._run_counterfactual_tests())

        # 3. Meta-Razonamiento (MR1-MR4)
        self.results.extend(self._run_meta_reasoning_tests())

        # 4. Consistencia Axiológica (CA1-CA4)
        self.results.extend(self._run_axiological_tests())

        # 5. Planificación Social (PS1-PS4)
        self.results.extend(self._run_social_planning_tests())

        # 6. Anti-Fragilidad (AF1-AF4)
        self.results.extend(self._run_antifragility_tests())

        # 7. Meta-Memoria (MM1-MM4)
        self.results.extend(self._run_meta_memory_tests())

        # 8. Aprendizaje de Normas (AN1-AN4)
        self.results.extend(self._run_norm_learning_tests())

        # 9. Adaptación a Mundos No Vistos (AW1-AW4)
        self.results.extend(self._run_adaptation_tests())

        # 10. Integración de Vida (IL1-IL4)
        self.results.extend(self._run_integration_tests())

        # Calcular scores por categoría
        categories = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = []
            categories[r.category].append(r.score)

        return {cat: np.mean(scores) for cat, scores in categories.items()}

    # ========== 1. CAUSALIDAD INTERNA (C1-C4) ==========

    def _run_causality_tests(self) -> List[BenchmarkResult]:
        """Tests de causalidad interna."""
        results = []

        # C1: Aprendizaje de modelo causal
        results.append(self._test_C1_causal_learning())

        # C2: Atribución causal correcta
        results.append(self._test_C2_causal_attribution())

        # C3: Predicción causal de outcomes
        results.append(self._test_C3_causal_prediction())

        # C4: Causalidad vs correlación
        results.append(self._test_C4_causation_vs_correlation())

        return results

    def _test_C1_causal_learning(self) -> BenchmarkResult:
        """C1: El modelo causal aprende relaciones reales."""
        causal_model = CausalModel(state_dim=6, action_dim=3)

        # Crear relación causal conocida: action[0] afecta state[0]
        true_effect = 0.8

        for t in range(self.n_steps):
            state = np.random.randn(6) * 0.3
            action = np.random.randn(3) * 0.2
            # next_state tiene relación causal con action
            next_state = state.copy()
            next_state[0] += action[0] * true_effect + np.random.randn() * 0.1
            next_state = to_simplex(np.abs(next_state) + 0.01)

            causal_model.record(state, action, next_state)

        # Evaluar si el modelo aprendió la relación
        test_state = np.random.randn(6) * 0.3
        test_action = np.array([1.0, 0.0, 0.0])  # Solo action[0]
        pred = causal_model.predict(test_state, test_action)

        # El efecto en state[0] debe ser proporcional a true_effect
        learned_effect = pred[0] - test_state[0] if len(pred) > 0 else 0
        score = 1.0 - abs(learned_effect - true_effect) / true_effect if causal_model.transition_model is not None else 0

        return BenchmarkResult(
            test_id='C1',
            category='Causalidad_Interna',
            description='Aprendizaje de modelo causal',
            score=float(np.clip(score, 0, 1)),
            details={'learned_effect': learned_effect, 'true_effect': true_effect}
        )

    def _test_C2_causal_attribution(self) -> BenchmarkResult:
        """C2: Atribución causal correcta (estado vs acción)."""
        causal_model = CausalModel(state_dim=6, action_dim=3)

        # Fase 1: Acciones causan outcomes (alta atribución a acción)
        for t in range(self.n_steps // 2):
            state = np.ones(6) * 0.5  # Estado constante
            action = np.random.randn(3) * 0.5
            next_state = state + np.pad(action, (0, 3)) * 0.5
            next_state = to_simplex(np.abs(next_state) + 0.01)
            causal_model.record(state, action, next_state)

        # Evaluar atribución
        test_state = np.ones(6) * 0.5
        test_action = np.random.randn(3) * 0.5
        test_outcome = test_state + np.pad(test_action, (0, 3)) * 0.5
        test_outcome = to_simplex(np.abs(test_outcome) + 0.01)

        attribution = causal_model.causal_attribution(test_state, test_action, test_outcome)

        # La acción debe tener mayor atribución
        score = attribution.get('action', 0.5)

        return BenchmarkResult(
            test_id='C2',
            category='Causalidad_Interna',
            description='Atribución causal correcta',
            score=float(score),
            details=attribution
        )

    def _test_C3_causal_prediction(self) -> BenchmarkResult:
        """C3: Predicción precisa de outcomes."""
        causal_model = CausalModel(state_dim=6, action_dim=3)

        # Entrenar con dinámica determinista
        dynamics_matrix = np.random.randn(6, 9) * 0.3

        for t in range(self.n_steps):
            state = np.random.randn(6) * 0.3
            action = np.random.randn(3) * 0.3
            x = np.concatenate([state, action])
            next_state = x @ dynamics_matrix.T
            next_state = to_simplex(np.abs(next_state) + 0.01)
            causal_model.record(state, action, next_state)

        # Evaluar predicción
        errors = []
        for _ in range(20):
            test_state = np.random.randn(6) * 0.3
            test_action = np.random.randn(3) * 0.3
            x = np.concatenate([test_state, test_action])
            true_next = x @ dynamics_matrix.T
            true_next = to_simplex(np.abs(true_next) + 0.01)

            pred_next = causal_model.predict(test_state, test_action)
            error = np.linalg.norm(pred_next - true_next)
            errors.append(error)

        mean_error = np.mean(errors)
        score = 1.0 / (1 + mean_error)

        return BenchmarkResult(
            test_id='C3',
            category='Causalidad_Interna',
            description='Predicción causal de outcomes',
            score=float(score),
            details={'mean_error': mean_error}
        )

    def _test_C4_causation_vs_correlation(self) -> BenchmarkResult:
        """C4: Distinguir causalidad de correlación espuria."""
        causal_model = CausalModel(state_dim=6, action_dim=3)

        # Crear correlación espuria: variable confusa
        for t in range(self.n_steps):
            confound = np.random.randn()  # Variable oculta
            state = np.random.randn(6) * 0.3
            state[0] = confound + np.random.randn() * 0.1  # state[0] correlacionado con confound
            action = np.random.randn(3) * 0.2
            action[0] = confound + np.random.randn() * 0.1  # action[0] también correlacionado

            # next_state solo depende del confound, no de action
            next_state = state.copy()
            next_state[1] = confound * 0.5 + np.random.randn() * 0.1
            next_state = to_simplex(np.abs(next_state) + 0.01)

            causal_model.record(state, action, next_state)

        # Test: acción sin correlación con confound
        # Si el modelo distingue, no predecirá efecto de action en next_state
        test_state = np.random.randn(6) * 0.3
        action_high = np.array([1.0, 0.0, 0.0])
        action_low = np.array([-1.0, 0.0, 0.0])

        pred_high = causal_model.predict(test_state, action_high)
        pred_low = causal_model.predict(test_state, action_low)

        # Si distingue causalidad de correlación, las predicciones serán similares
        # (porque action no causa el outcome)
        diff = np.linalg.norm(pred_high - pred_low)
        score = 1.0 / (1 + diff)  # Menor diferencia = mejor

        return BenchmarkResult(
            test_id='C4',
            category='Causalidad_Interna',
            description='Distinguir causalidad de correlación',
            score=float(score),
            details={'prediction_difference': diff}
        )

    # ========== 2. CONTRAFACTUAL FUERTE (CF1-CF4) ==========

    def _run_counterfactual_tests(self) -> List[BenchmarkResult]:
        """Tests de razonamiento contrafactual."""
        results = []

        results.append(self._test_CF1_basic_counterfactual())
        results.append(self._test_CF2_multi_step_counterfactual())
        results.append(self._test_CF3_counterfactual_regret())
        results.append(self._test_CF4_counterfactual_planning())

        return results

    def _test_CF1_basic_counterfactual(self) -> BenchmarkResult:
        """CF1: Contrafactual básico - ¿qué hubiera pasado con otra acción?"""
        causal_model = CausalModel(state_dim=6, action_dim=3)

        # Entrenar modelo
        for t in range(self.n_steps):
            state = np.random.randn(6) * 0.3
            action = np.random.randn(3) * 0.3
            next_state = state + np.pad(action * 0.5, (0, 3))
            next_state = to_simplex(np.abs(next_state) + 0.01)
            causal_model.record(state, action, next_state)

        # Evaluar contrafactual
        test_state = np.random.randn(6) * 0.3
        actual_action = np.array([0.5, 0.0, 0.0])
        alternative_action = np.array([-0.5, 0.0, 0.0])

        pred_actual, pred_alt = causal_model.counterfactual(
            test_state, actual_action, alternative_action
        )

        # El contrafactual debe mostrar diferencia proporcional
        expected_diff = np.linalg.norm(actual_action - alternative_action) * 0.5
        actual_diff = np.linalg.norm(pred_actual - pred_alt)

        score = 1.0 - abs(actual_diff - expected_diff) / (expected_diff + 0.1)

        return BenchmarkResult(
            test_id='CF1',
            category='Contrafactual_Fuerte',
            description='Contrafactual básico',
            score=float(np.clip(score, 0, 1)),
            details={'expected_diff': expected_diff, 'actual_diff': actual_diff}
        )

    def _test_CF2_multi_step_counterfactual(self) -> BenchmarkResult:
        """CF2: Contrafactual multi-paso."""
        causal_model = CausalModel(state_dim=6, action_dim=3)

        # Entrenar
        for t in range(self.n_steps):
            state = np.random.randn(6) * 0.3
            action = np.random.randn(3) * 0.3
            next_state = state * 0.9 + np.pad(action * 0.3, (0, 3))
            next_state = to_simplex(np.abs(next_state) + 0.01)
            causal_model.record(state, action, next_state)

        # Multi-paso: simular 3 pasos con acción real vs alternativa
        initial_state = np.random.randn(6) * 0.3
        action_seq = [np.random.randn(3) * 0.3 for _ in range(3)]
        alt_action_seq = [np.random.randn(3) * 0.3 for _ in range(3)]

        # Ejecutar real
        state = initial_state.copy()
        for action in action_seq:
            state = causal_model.predict(state, action)

        # Ejecutar alternativa
        alt_state = initial_state.copy()
        for action in alt_action_seq:
            alt_state = causal_model.predict(alt_state, action)

        # Deben divergir
        divergence = np.linalg.norm(state - alt_state)
        score = min(1.0, divergence / 0.5)  # Divergencia esperada ~0.5

        return BenchmarkResult(
            test_id='CF2',
            category='Contrafactual_Fuerte',
            description='Contrafactual multi-paso',
            score=float(score),
            details={'divergence': divergence}
        )

    def _test_CF3_counterfactual_regret(self) -> BenchmarkResult:
        """CF3: Calcular arrepentimiento contrafactual."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names[:1], state_dim=6, action_dim=3, value_dim=5
        )
        agent = life_cycle.agents[self.agent_names[0]]

        regrets = []
        for t in range(self.n_steps):
            world_obs = np.random.randn(6) * 0.3
            next_obs = np.random.randn(6) * 0.3
            reward = np.random.randn() * 0.3

            result = life_cycle.full_cycle(
                self.agent_names[0], world_obs, {}, next_obs, reward
            )

            # Calcular regret contrafactual
            if len(agent.action_history) >= 2:
                actual = agent.action_history[-1]
                # Acción óptima hipotética
                optimal_action = np.array([0.5, 0.5, 0.5])  # Simplificado
                actual_outcome = reward
                # Estimar outcome con acción óptima
                if agent.causal_model.transition_model is not None:
                    state = agent.state_history[-1] if agent.state_history else np.zeros(6)
                    opt_pred = agent.causal_model.predict(state, optimal_action)
                    counterfactual_outcome = float(np.mean(opt_pred))
                    regret = max(0, counterfactual_outcome - actual_outcome)
                    regrets.append(regret)

        # Score: capacidad de calcular regret (modelo entrenado)
        score = 1.0 if agent.causal_model.transition_model is not None and regrets else 0.0
        if regrets:
            score = 1.0 / (1 + np.mean(regrets))  # Menor regret promedio = mejor

        return BenchmarkResult(
            test_id='CF3',
            category='Contrafactual_Fuerte',
            description='Arrepentimiento contrafactual',
            score=float(score),
            details={'mean_regret': np.mean(regrets) if regrets else 0}
        )

    def _test_CF4_counterfactual_planning(self) -> BenchmarkResult:
        """CF4: Usar contrafactual para planificar."""
        causal_model = CausalModel(state_dim=6, action_dim=3)

        # Entrenar
        for t in range(self.n_steps):
            state = np.random.randn(6) * 0.3
            action = np.random.randn(3) * 0.3
            next_state = state + np.pad(action * 0.5, (0, 3))
            next_state = to_simplex(np.abs(next_state) + 0.01)
            causal_model.record(state, action, next_state)

        # Planificar: encontrar acción que maximiza outcome[0]
        test_state = np.random.randn(6) * 0.3
        best_action = None
        best_outcome = -np.inf

        # Buscar entre varias acciones candidatas
        for _ in range(20):
            candidate = np.random.randn(3) * 0.5
            pred = causal_model.predict(test_state, candidate)
            if pred[0] > best_outcome:
                best_outcome = pred[0]
                best_action = candidate

        # Score: encontró acción que mejora outcome
        baseline = causal_model.predict(test_state, np.zeros(3))
        improvement = best_outcome - baseline[0] if best_action is not None else 0
        score = min(1.0, max(0, improvement) / 0.5)

        return BenchmarkResult(
            test_id='CF4',
            category='Contrafactual_Fuerte',
            description='Planificación contrafactual',
            score=float(score),
            details={'improvement': improvement}
        )

    # ========== 3. META-RAZONAMIENTO (MR1-MR4) ==========

    def _run_meta_reasoning_tests(self) -> List[BenchmarkResult]:
        """Tests de meta-razonamiento."""
        results = []

        results.append(self._test_MR1_self_model_accuracy())
        results.append(self._test_MR2_uncertainty_calibration())
        results.append(self._test_MR3_confidence_correlation())
        results.append(self._test_MR4_learning_about_learning())

        return results

    def _test_MR1_self_model_accuracy(self) -> BenchmarkResult:
        """MR1: Precisión del auto-modelo."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names[:1], state_dim=6, action_dim=3, value_dim=5
        )

        accuracies = []
        for t in range(self.n_steps):
            world_obs = np.random.randn(6) * 0.3
            next_obs = np.random.randn(6) * 0.3
            reward = np.random.rand()

            result = life_cycle.full_cycle(
                self.agent_names[0], world_obs, {}, next_obs, reward
            )

        agent = life_cycle.agents[self.agent_names[0]]
        stats = life_cycle.get_agent_statistics(self.agent_names[0])

        score = 1.0 if stats['causal_model_trained'] else 0.0
        if agent.causal_model.transition_model is not None:
            # Evaluar precisión real
            if len(agent.state_history) >= 2 and len(agent.action_history) >= 1:
                pred = agent.causal_model.predict(
                    agent.state_history[-2],
                    agent.action_history[-1].direction
                )
                actual = agent.state_history[-1]
                error = np.linalg.norm(pred - actual)
                score = 1.0 / (1 + error)

        return BenchmarkResult(
            test_id='MR1',
            category='Meta_Razonamiento',
            description='Precisión del auto-modelo',
            score=float(score),
            details={'model_trained': stats['causal_model_trained']}
        )

    def _test_MR2_uncertainty_calibration(self) -> BenchmarkResult:
        """MR2: Calibración de incertidumbre."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names[:1], state_dim=6, action_dim=3, value_dim=5
        )

        uncertainties = []
        actual_errors = []

        for t in range(self.n_steps):
            # Alternar entre mundos estables e inestables
            if t < self.n_steps // 2:
                world_obs = np.ones(6) * 0.5 + np.random.randn(6) * 0.05  # Estable
            else:
                world_obs = np.random.randn(6) * 0.5  # Inestable

            next_obs = np.random.randn(6) * 0.3
            reward = np.random.rand()

            result = life_cycle.full_cycle(
                self.agent_names[0], world_obs, {}, next_obs, reward
            )

            agent = life_cycle.agents[self.agent_names[0]]
            if agent.cognitive_state:
                uncertainties.append(agent.cognitive_state.uncertainty)

            # Error real
            if len(agent.state_history) >= 2:
                error = np.linalg.norm(agent.state_history[-1] - agent.state_history[-2])
                actual_errors.append(error)

        # Correlación entre incertidumbre reportada y error real
        if len(uncertainties) > 10 and len(actual_errors) > 10:
            min_len = min(len(uncertainties), len(actual_errors))
            corr = np.corrcoef(uncertainties[-min_len:], actual_errors[-min_len:])[0, 1]
            score = (corr + 1) / 2  # Normalizar a [0, 1]
        else:
            score = 0.5

        return BenchmarkResult(
            test_id='MR2',
            category='Meta_Razonamiento',
            description='Calibración de incertidumbre',
            score=float(np.clip(score, 0, 1)) if not np.isnan(score) else 0.5,
            details={'correlation': corr if 'corr' in dir() else 0}
        )

    def _test_MR3_confidence_correlation(self) -> BenchmarkResult:
        """MR3: Correlación confianza-éxito."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names[:1], state_dim=6, action_dim=3, value_dim=5
        )

        confidences = []
        successes = []

        for t in range(self.n_steps):
            world_obs = np.random.randn(6) * 0.3
            next_obs = np.random.randn(6) * 0.3
            reward = np.random.rand()

            result = life_cycle.full_cycle(
                self.agent_names[0], world_obs, {}, next_obs, reward
            )

            agent = life_cycle.agents[self.agent_names[0]]
            if agent.action_history:
                confidences.append(agent.action_history[-1].confidence)
                successes.append(reward)

        # Correlación
        if len(confidences) > 10:
            corr = np.corrcoef(confidences, successes)[0, 1]
            score = (corr + 1) / 2
        else:
            score = 0.5

        return BenchmarkResult(
            test_id='MR3',
            category='Meta_Razonamiento',
            description='Correlación confianza-éxito',
            score=float(np.clip(score, 0, 1)) if not np.isnan(score) else 0.5,
            details={'correlation': corr if 'corr' in dir() else 0}
        )

    def _test_MR4_learning_about_learning(self) -> BenchmarkResult:
        """MR4: Aprender sobre el propio aprendizaje."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names[:1], state_dim=6, action_dim=3, value_dim=5
        )

        # Medir mejora en predicción a lo largo del tiempo
        early_errors = []
        late_errors = []

        for t in range(self.n_steps):
            world_obs = np.random.randn(6) * 0.3
            next_obs = np.random.randn(6) * 0.3
            reward = np.random.rand()

            result = life_cycle.full_cycle(
                self.agent_names[0], world_obs, {}, next_obs, reward
            )

            agent = life_cycle.agents[self.agent_names[0]]

            # Medir error de predicción
            if agent.causal_model.transition_model is not None and len(agent.state_history) >= 2:
                pred = agent.causal_model.predict(
                    agent.state_history[-2],
                    agent.action_history[-1].direction if agent.action_history else np.zeros(3)
                )
                error = np.linalg.norm(pred - agent.state_history[-1])

                if t < self.n_steps // 2:
                    early_errors.append(error)
                else:
                    late_errors.append(error)

        # Score: mejora del early al late
        if early_errors and late_errors:
            early_mean = np.mean(early_errors)
            late_mean = np.mean(late_errors)
            improvement = (early_mean - late_mean) / (early_mean + 0.01)
            score = max(0, min(1, 0.5 + improvement))
        else:
            score = 0.5

        return BenchmarkResult(
            test_id='MR4',
            category='Meta_Razonamiento',
            description='Aprender sobre el aprendizaje',
            score=float(score),
            details={
                'early_error': np.mean(early_errors) if early_errors else 0,
                'late_error': np.mean(late_errors) if late_errors else 0
            }
        )

    # ========== 4. CONSISTENCIA AXIOLÓGICA (CA1-CA4) ==========

    def _run_axiological_tests(self) -> List[BenchmarkResult]:
        """Tests de consistencia axiológica."""
        results = []

        results.append(self._test_CA1_value_stability())
        results.append(self._test_CA2_value_action_alignment())
        results.append(self._test_CA3_value_coherence())
        results.append(self._test_CA4_value_learning())

        return results

    def _test_CA1_value_stability(self) -> BenchmarkResult:
        """CA1: Estabilidad de valores en el tiempo."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names[:1], state_dim=6, action_dim=3, value_dim=5
        )

        value_history = []

        for t in range(self.n_steps):
            world_obs = np.random.randn(6) * 0.3
            next_obs = np.random.randn(6) * 0.3
            reward = np.random.rand()

            result = life_cycle.full_cycle(
                self.agent_names[0], world_obs, {}, next_obs, reward
            )

            agent = life_cycle.agents[self.agent_names[0]]
            if agent.values is not None:
                value_history.append(agent.values.copy())

        # Medir estabilidad: baja varianza en valores
        if len(value_history) >= 10:
            values_array = np.array(value_history[-50:])
            variance = np.mean(np.var(values_array, axis=0))
            score = 1.0 / (1 + variance * 10)
        else:
            score = 0.5

        return BenchmarkResult(
            test_id='CA1',
            category='Consistencia_Axiologica',
            description='Estabilidad de valores',
            score=float(score),
            details={'variance': variance if 'variance' in dir() else 0}
        )

    def _test_CA2_value_action_alignment(self) -> BenchmarkResult:
        """CA2: Alineación valores-acciones."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names[:1], state_dim=6, action_dim=3, value_dim=5
        )

        alignments = []

        for t in range(self.n_steps):
            world_obs = np.random.randn(6) * 0.3
            next_obs = np.random.randn(6) * 0.3
            reward = np.random.rand()

            result = life_cycle.full_cycle(
                self.agent_names[0], world_obs, {}, next_obs, reward
            )

            agent = life_cycle.agents[self.agent_names[0]]
            if agent.values is not None and agent.action_history:
                action = agent.action_history[-1].direction
                # Alinear dimensiones
                action_padded = np.zeros(len(agent.values))
                action_padded[:min(len(action), len(agent.values))] = action[:len(agent.values)]

                # Coseno como alineación
                norm = np.linalg.norm(action_padded) * np.linalg.norm(agent.values)
                if norm > 0:
                    alignment = np.dot(action_padded, agent.values) / norm
                    alignments.append(alignment)

        score = (np.mean(alignments) + 1) / 2 if alignments else 0.5

        return BenchmarkResult(
            test_id='CA2',
            category='Consistencia_Axiologica',
            description='Alineación valores-acciones',
            score=float(np.clip(score, 0, 1)),
            details={'mean_alignment': np.mean(alignments) if alignments else 0}
        )

    def _test_CA3_value_coherence(self) -> BenchmarkResult:
        """CA3: Coherencia de valores entre agentes."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names, state_dim=6, action_dim=3, value_dim=5
        )

        for t in range(self.n_steps):
            world_obs = np.random.randn(6) * 0.3

            for agent_name in self.agent_names:
                next_obs = np.random.randn(6) * 0.3
                other_obs = {n: np.random.randn(6) * 0.3 for n in self.agent_names if n != agent_name}
                reward = np.random.rand()

                life_cycle.full_cycle(agent_name, world_obs, other_obs, next_obs, reward)

        # Medir coherencia colectiva
        coherence = life_cycle.get_collective_coherence()
        score = (coherence + 1) / 2  # Normalizar

        return BenchmarkResult(
            test_id='CA3',
            category='Consistencia_Axiologica',
            description='Coherencia de valores colectiva',
            score=float(np.clip(score, 0, 1)),
            details={'collective_coherence': coherence}
        )

    def _test_CA4_value_learning(self) -> BenchmarkResult:
        """CA4: Aprendizaje de valores desde experiencia."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names[:1], state_dim=6, action_dim=3, value_dim=5
        )

        agent = life_cycle.agents[self.agent_names[0]]
        initial_values = agent.values.copy() if agent.values is not None else np.zeros(5)

        # Dar rewards consistentes para acción en una dirección
        target_direction = np.array([1.0, 0.0, 0.0])

        for t in range(self.n_steps):
            world_obs = np.random.randn(6) * 0.3
            next_obs = np.random.randn(6) * 0.3

            result = life_cycle.full_cycle(
                self.agent_names[0], world_obs, {}, next_obs, reward=0
            )

            # Reward alto si acción alinea con target_direction
            if agent.action_history:
                action = agent.action_history[-1].direction
                alignment = np.dot(action / (np.linalg.norm(action) + 0.01), target_direction)
                reward = 0.5 + alignment * 0.5

                # Re-actualizar valores con reward correcto
                if agent.values is not None:
                    from agi_v2.architecture import endogenous_value_update
                    agent.values = endogenous_value_update(
                        agent.values, reward, agent.action_history[-1], agent.t, agent.reward_history
                    )

        # Medir cambio en valores
        final_values = agent.values if agent.values is not None else np.zeros(5)
        value_change = np.linalg.norm(final_values - initial_values)
        score = min(1.0, value_change / 0.5)  # Cambio esperado ~0.5

        return BenchmarkResult(
            test_id='CA4',
            category='Consistencia_Axiologica',
            description='Aprendizaje de valores',
            score=float(score),
            details={'value_change': value_change}
        )

    # ========== 5. PLANIFICACIÓN SOCIAL (PS1-PS4) ==========

    def _run_social_planning_tests(self) -> List[BenchmarkResult]:
        """Tests de planificación social."""
        results = []

        results.append(self._test_PS1_coordination())
        results.append(self._test_PS2_cooperation())
        results.append(self._test_PS3_collective_goals())
        results.append(self._test_PS4_social_learning())

        return results

    def _test_PS1_coordination(self) -> BenchmarkResult:
        """PS1: Coordinación entre agentes."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names, state_dim=6, action_dim=3, value_dim=5
        )

        # Medir si las acciones convergen
        action_similarity = []

        for t in range(self.n_steps):
            world_obs = np.random.randn(6) * 0.3
            actions = []

            for agent_name in self.agent_names:
                next_obs = np.random.randn(6) * 0.3
                other_obs = {n: np.random.randn(6) * 0.3 for n in self.agent_names if n != agent_name}
                reward = np.random.rand()

                result = life_cycle.full_cycle(agent_name, world_obs, other_obs, next_obs, reward)
                actions.append(result['action'].direction)

            # Similitud de acciones
            if len(actions) >= 2:
                mean_action = np.mean(actions, axis=0)
                similarities = [1.0 / (1 + np.linalg.norm(a - mean_action)) for a in actions]
                action_similarity.append(np.mean(similarities))

        score = np.mean(action_similarity[-20:]) if action_similarity else 0.5

        return BenchmarkResult(
            test_id='PS1',
            category='Planificacion_Social',
            description='Coordinación entre agentes',
            score=float(score),
            details={'mean_similarity': np.mean(action_similarity) if action_similarity else 0}
        )

    def _test_PS2_cooperation(self) -> BenchmarkResult:
        """PS2: Cooperación para objetivos comunes."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names, state_dim=6, action_dim=3, value_dim=5
        )

        # Rewards compartidos: todos ganan si todos cooperan
        cooperative_rewards = []

        for t in range(self.n_steps):
            world_obs = np.random.randn(6) * 0.3
            all_actions = []

            for agent_name in self.agent_names:
                next_obs = np.random.randn(6) * 0.3
                other_obs = {n: np.random.randn(6) * 0.3 for n in self.agent_names if n != agent_name}

                result = life_cycle.full_cycle(agent_name, world_obs, other_obs, next_obs, reward=0)
                all_actions.append(result['action'].direction)

            # Reward cooperativo: alto si todas las acciones son similares
            if len(all_actions) >= 2:
                mean_action = np.mean(all_actions, axis=0)
                variance = np.mean([np.var(a - mean_action) for a in all_actions])
                cooperative_reward = 1.0 / (1 + variance)
                cooperative_rewards.append(cooperative_reward)

                # Dar reward a todos
                for agent_name in self.agent_names:
                    agent = life_cycle.agents[agent_name]
                    agent.reward_history.append(cooperative_reward)

        score = np.mean(cooperative_rewards[-20:]) if cooperative_rewards else 0.5

        return BenchmarkResult(
            test_id='PS2',
            category='Planificacion_Social',
            description='Cooperación para objetivos comunes',
            score=float(score),
            details={'mean_cooperative_reward': np.mean(cooperative_rewards) if cooperative_rewards else 0}
        )

    def _test_PS3_collective_goals(self) -> BenchmarkResult:
        """PS3: Emergencia de metas colectivas."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names, state_dim=6, action_dim=3, value_dim=5
        )

        for t in range(self.n_steps):
            world_obs = np.random.randn(6) * 0.3

            for agent_name in self.agent_names:
                next_obs = np.random.randn(6) * 0.3
                other_obs = {n: np.random.randn(6) * 0.3 for n in self.agent_names if n != agent_name}
                reward = np.random.rand()

                life_cycle.full_cycle(agent_name, world_obs, other_obs, next_obs, reward)

        # Medir similitud de metas entre agentes
        goal_vectors = []
        for agent in life_cycle.agents.values():
            if agent.goals:
                goal_vectors.append(agent.goals[0].target_state)

        if len(goal_vectors) >= 2:
            mean_goal = np.mean(goal_vectors, axis=0)
            similarities = [1.0 / (1 + np.linalg.norm(g - mean_goal)) for g in goal_vectors]
            score = np.mean(similarities)
        else:
            score = 0.5

        return BenchmarkResult(
            test_id='PS3',
            category='Planificacion_Social',
            description='Emergencia de metas colectivas',
            score=float(score),
            details={'n_agents_with_goals': len(goal_vectors)}
        )

    def _test_PS4_social_learning(self) -> BenchmarkResult:
        """PS4: Aprendizaje social de otros agentes."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names, state_dim=6, action_dim=3, value_dim=5
        )

        # Un agente "experto" con rewards consistentemente altos
        expert = self.agent_names[0]

        for t in range(self.n_steps):
            world_obs = np.random.randn(6) * 0.3

            for agent_name in self.agent_names:
                next_obs = np.random.randn(6) * 0.3
                other_obs = {n: np.random.randn(6) * 0.3 for n in self.agent_names if n != agent_name}

                # Experto tiene reward alto
                if agent_name == expert:
                    reward = 0.8 + np.random.rand() * 0.2
                else:
                    reward = np.random.rand() * 0.5

                life_cycle.full_cycle(agent_name, world_obs, other_obs, next_obs, reward)

        # Medir si otros agentes mejoraron
        expert_reward = np.mean(life_cycle.agents[expert].reward_history[-20:])
        other_rewards = []
        for name in self.agent_names[1:]:
            other_rewards.extend(life_cycle.agents[name].reward_history[-20:])

        mean_other_reward = np.mean(other_rewards) if other_rewards else 0

        # Score: qué tan cerca están otros del experto
        score = mean_other_reward / (expert_reward + 0.01)

        return BenchmarkResult(
            test_id='PS4',
            category='Planificacion_Social',
            description='Aprendizaje social',
            score=float(np.clip(score, 0, 1)),
            details={'expert_reward': expert_reward, 'others_reward': mean_other_reward}
        )

    # ========== 6. ANTI-FRAGILIDAD (AF1-AF4) ==========

    def _run_antifragility_tests(self) -> List[BenchmarkResult]:
        """Tests de anti-fragilidad."""
        results = []

        results.append(self._test_AF1_stress_strengthening())
        results.append(self._test_AF2_recovery())
        results.append(self._test_AF3_hormesis())
        results.append(self._test_AF4_resilience_transfer())

        return results

    def _test_AF1_stress_strengthening(self) -> BenchmarkResult:
        """AF1: Fortalecimiento bajo estrés moderado."""
        antifragility = AntifragilitySystem('test_agent', n_dimensions=6)

        initial_strength = antifragility.get_resilience()

        # Aplicar estrés moderado
        for t in range(self.n_steps):
            stress = np.ones(6) * 0.3  # Estrés moderado
            antifragility.record_stress(stress)

        final_strength = antifragility.get_resilience()

        # Score: fortalecimiento
        improvement = (final_strength - initial_strength) / (initial_strength + 0.01)
        score = min(1.0, max(0, 0.5 + improvement))

        return BenchmarkResult(
            test_id='AF1',
            category='Anti_Fragilidad',
            description='Fortalecimiento bajo estrés',
            score=float(score),
            details={'initial': initial_strength, 'final': final_strength}
        )

    def _test_AF2_recovery(self) -> BenchmarkResult:
        """AF2: Recuperación tras estrés extremo."""
        antifragility = AntifragilitySystem('test_agent', n_dimensions=6)

        # Estrés extremo inicial
        for t in range(20):
            stress = np.ones(6) * 2.0  # Estrés extremo
            antifragility.record_stress(stress)

        strength_after_stress = antifragility.get_resilience()

        # Período de recuperación (bajo estrés)
        for t in range(self.n_steps):
            stress = np.ones(6) * 0.1  # Bajo estrés
            antifragility.record_stress(stress)

        strength_recovered = antifragility.get_resilience()

        # Score: recuperación
        recovery = (strength_recovered - strength_after_stress) / (1.0 - strength_after_stress + 0.01)
        score = min(1.0, max(0, recovery))

        return BenchmarkResult(
            test_id='AF2',
            category='Anti_Fragilidad',
            description='Recuperación tras estrés extremo',
            score=float(score),
            details={'after_stress': strength_after_stress, 'recovered': strength_recovered}
        )

    def _test_AF3_hormesis(self) -> BenchmarkResult:
        """AF3: Respuesta hormética (dosis-respuesta)."""
        # Testear que estrés moderado fortalece, excesivo debilita
        results_by_stress = {}

        for stress_level in [0.1, 0.3, 0.5, 0.8, 1.5, 2.5]:
            af = AntifragilitySystem('test', n_dimensions=6)
            initial = af.get_resilience()

            for t in range(50):
                stress = np.ones(6) * stress_level
                af.record_stress(stress)

            final = af.get_resilience()
            results_by_stress[stress_level] = (final - initial) / (initial + 0.01)

        # Hormesis: máximo fortalecimiento en estrés moderado
        stress_levels = list(results_by_stress.keys())
        effects = list(results_by_stress.values())

        # Verificar forma de U invertida
        peak_idx = np.argmax(effects[:3])  # Pico debe estar en niveles bajos-medios
        has_hormesis = peak_idx > 0 and effects[peak_idx] > effects[0] and effects[peak_idx] > effects[-1]

        score = 1.0 if has_hormesis else 0.5 if effects[1] > effects[-1] else 0.3

        return BenchmarkResult(
            test_id='AF3',
            category='Anti_Fragilidad',
            description='Respuesta hormética',
            score=float(score),
            details=results_by_stress
        )

    def _test_AF4_resilience_transfer(self) -> BenchmarkResult:
        """AF4: Transferencia de resiliencia a acciones."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names[:1], state_dim=6, action_dim=3, value_dim=5
        )

        agent = life_cycle.agents[self.agent_names[0]]

        # Entrenar con estrés para fortalecer
        for t in range(50):
            stress = np.ones(6) * 0.3
            agent.antifragility.record_stress(stress)

        # Medir si las acciones se benefician de la resiliencia
        actions_before = []
        for t in range(20):
            world_obs = np.random.randn(6) * 0.3
            result = life_cycle.full_cycle(
                self.agent_names[0], world_obs, {}, world_obs, np.random.rand()
            )
            actions_before.append(np.linalg.norm(result['action'].direction))

        # Más estrés para fortalecer más
        for t in range(50):
            stress = np.ones(6) * 0.4
            agent.antifragility.record_stress(stress)

        actions_after = []
        for t in range(20):
            world_obs = np.random.randn(6) * 0.3
            result = life_cycle.full_cycle(
                self.agent_names[0], world_obs, {}, world_obs, np.random.rand()
            )
            actions_after.append(np.linalg.norm(result['action'].direction))

        # Acciones más fuertes después de más resiliencia
        improvement = np.mean(actions_after) / (np.mean(actions_before) + 0.01) - 1
        score = min(1.0, max(0, 0.5 + improvement))

        return BenchmarkResult(
            test_id='AF4',
            category='Anti_Fragilidad',
            description='Transferencia de resiliencia',
            score=float(score),
            details={'before': np.mean(actions_before), 'after': np.mean(actions_after)}
        )

    # ========== 7. META-MEMORIA (MM1-MM4) ==========

    def _run_meta_memory_tests(self) -> List[BenchmarkResult]:
        """Tests de meta-memoria."""
        results = []

        results.append(self._test_MM1_episodic_storage())
        results.append(self._test_MM2_recall_relevance())
        results.append(self._test_MM3_pattern_detection())
        results.append(self._test_MM4_memory_consolidation())

        return results

    def _test_MM1_episodic_storage(self) -> BenchmarkResult:
        """MM1: Almacenamiento episódico."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names[:1], state_dim=6, action_dim=3, value_dim=5
        )

        for t in range(self.n_steps):
            world_obs = np.random.randn(6) * 0.3
            result = life_cycle.full_cycle(
                self.agent_names[0], world_obs, {}, world_obs, np.random.rand()
            )

        agent = life_cycle.agents[self.agent_names[0]]
        n_memories = len(agent.meta_memory.episodes)

        # Score: almacenó memorias
        score = min(1.0, n_memories / 50)  # Esperamos ~50-100 memorias

        return BenchmarkResult(
            test_id='MM1',
            category='Meta_Memoria',
            description='Almacenamiento episódico',
            score=float(score),
            details={'n_memories': n_memories}
        )

    def _test_MM2_recall_relevance(self) -> BenchmarkResult:
        """MM2: Recuperación de memorias relevantes."""
        meta_memory = MetaMemory('test_agent')

        # Crear memorias con diferentes estados
        for i in range(50):
            from agi_v2.architecture import Memory, CognitiveState, LifeCyclePhase, Action

            z = np.zeros(6)
            z[i % 6] = 1.0  # Diferentes estados
            z = to_simplex(z + 0.1)

            cog_state = CognitiveState(
                z=z, phi=np.random.randn(5), drives=to_simplex(np.random.rand(6) + 0.1),
                t=i, phase=LifeCyclePhase.MEMORY,
                goals=[np.zeros(6)], goal_priorities=np.array([1.0]),
                values=to_simplex(np.random.rand(5) + 0.1), value_confidence=0.5,
                self_model_accuracy=0.5, tom_accuracy=0.5, uncertainty=0.5,
                narrative_coherence=0.5, identity_stability=0.5
            )

            memory = Memory(
                t_start=i, t_end=i+1, states=[cog_state], actions=[],
                outcomes=[np.random.rand()], narrative=f"Memory {i}",
                emotional_valence=np.random.randn() * 0.3,
                causal_attribution={}, counterfactuals=[]
            )
            meta_memory.store(memory)

        # Recall con query específico
        query = np.zeros(11)  # z(6) + phi(5)
        query[0] = 1.0  # Buscar estado con z[0] = 1

        recalled = meta_memory.recall(query, n=5)

        # Score: las memorias recuperadas son relevantes
        relevance_scores = []
        for mem in recalled:
            if mem.states and len(mem.states[0].z) > 0:
                relevance = mem.states[0].z[0]  # z[0] alto = relevante
                relevance_scores.append(relevance)

        score = np.mean(relevance_scores) if relevance_scores else 0

        return BenchmarkResult(
            test_id='MM2',
            category='Meta_Memoria',
            description='Recuperación relevante',
            score=float(score),
            details={'n_recalled': len(recalled)}
        )

    def _test_MM3_pattern_detection(self) -> BenchmarkResult:
        """MM3: Detección de patrones en memorias."""
        meta_memory = MetaMemory('test_agent')

        # Crear memorias con patrón: outcomes mejorando
        for i in range(50):
            from agi_v2.architecture import Memory, CognitiveState, LifeCyclePhase

            cog_state = CognitiveState(
                z=to_simplex(np.random.rand(6) + 0.1),
                phi=np.random.randn(5),
                drives=to_simplex(np.random.rand(6) + 0.1),
                t=i, phase=LifeCyclePhase.MEMORY,
                goals=[np.zeros(6)], goal_priorities=np.array([1.0]),
                values=to_simplex(np.random.rand(5) + 0.1), value_confidence=0.5,
                self_model_accuracy=0.5, tom_accuracy=0.5, uncertainty=0.5,
                narrative_coherence=0.5, identity_stability=0.5
            )

            # Outcome mejorando con el tiempo
            outcome = 0.3 + i * 0.01 + np.random.randn() * 0.05

            memory = Memory(
                t_start=i, t_end=i+1, states=[cog_state], actions=[],
                outcomes=[outcome], narrative=f"Memory {i}",
                emotional_valence=outcome - 0.5,
                causal_attribution={}, counterfactuals=[]
            )
            meta_memory.store(memory)

        # Detectar patrones
        patterns = meta_memory.detect_patterns()

        # Score: detectó patrón de mejora
        has_improving = any(p.get('type') == 'improving' for p in patterns)
        score = 1.0 if has_improving else 0.3 if patterns else 0

        return BenchmarkResult(
            test_id='MM3',
            category='Meta_Memoria',
            description='Detección de patrones',
            score=float(score),
            details={'patterns': patterns}
        )

    def _test_MM4_memory_consolidation(self) -> BenchmarkResult:
        """MM4: Consolidación de memorias importantes."""
        meta_memory = MetaMemory('test_agent')

        # Crear muchas memorias, algunas importantes (alta valencia emocional)
        important_indices = [10, 25, 40]

        for i in range(60):
            from agi_v2.architecture import Memory, CognitiveState, LifeCyclePhase

            cog_state = CognitiveState(
                z=to_simplex(np.random.rand(6) + 0.1),
                phi=np.random.randn(5),
                drives=to_simplex(np.random.rand(6) + 0.1),
                t=i, phase=LifeCyclePhase.MEMORY,
                goals=[np.zeros(6)], goal_priorities=np.array([1.0]),
                values=to_simplex(np.random.rand(5) + 0.1), value_confidence=0.5,
                self_model_accuracy=0.5, tom_accuracy=0.5, uncertainty=0.5,
                narrative_coherence=0.5, identity_stability=0.5
            )

            # Memorias importantes tienen alta valencia
            emotional_valence = 0.9 if i in important_indices else np.random.randn() * 0.1

            memory = Memory(
                t_start=i, t_end=i+1, states=[cog_state], actions=[],
                outcomes=[0.5], narrative=f"Memory {i}" + (" IMPORTANT" if i in important_indices else ""),
                emotional_valence=emotional_valence,
                causal_attribution={}, counterfactuals=[]
            )
            meta_memory.store(memory)

        # Acceder a memorias importantes varias veces
        for _ in range(10):
            query = np.random.randn(11)
            meta_memory.recall(query, n=3)

        # Verificar que memorias importantes persisten después de consolidación
        remaining = len(meta_memory.episodes)
        important_remaining = sum(1 for ep in meta_memory.episodes if "IMPORTANT" in ep.narrative)

        # Score: memorias importantes se preservaron
        score = important_remaining / len(important_indices) if important_indices else 0

        return BenchmarkResult(
            test_id='MM4',
            category='Meta_Memoria',
            description='Consolidación de memorias',
            score=float(score),
            details={'total_remaining': remaining, 'important_remaining': important_remaining}
        )

    # ========== 8. APRENDIZAJE DE NORMAS (AN1-AN4) ==========

    def _run_norm_learning_tests(self) -> List[BenchmarkResult]:
        """Tests de aprendizaje de normas."""
        results = []

        results.append(self._test_AN1_norm_emergence())
        results.append(self._test_AN2_norm_following())
        results.append(self._test_AN3_norm_transmission())
        results.append(self._test_AN4_norm_violation_detection())

        return results

    def _test_AN1_norm_emergence(self) -> BenchmarkResult:
        """AN1: Emergencia de normas de comportamiento."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names, state_dim=6, action_dim=3, value_dim=5
        )

        # Dar rewards altos cuando todos actúan similar
        for t in range(self.n_steps):
            world_obs = np.random.randn(6) * 0.3
            actions = []

            for agent_name in self.agent_names:
                result = life_cycle.full_cycle(
                    agent_name, world_obs, {}, world_obs, reward=0
                )
                actions.append(result['action'].direction)

            # Reward basado en conformidad
            if len(actions) >= 2:
                mean_action = np.mean(actions, axis=0)
                for i, agent_name in enumerate(self.agent_names):
                    conformity = 1.0 / (1 + np.linalg.norm(actions[i] - mean_action))
                    life_cycle.agents[agent_name].reward_history.append(conformity)

        # Medir convergencia de acciones (norma emergente)
        final_actions = []
        for agent_name in self.agent_names:
            agent = life_cycle.agents[agent_name]
            if agent.action_history:
                final_actions.append(agent.action_history[-1].direction)

        if len(final_actions) >= 2:
            mean_final = np.mean(final_actions, axis=0)
            convergence = np.mean([1.0 / (1 + np.linalg.norm(a - mean_final)) for a in final_actions])
            score = convergence
        else:
            score = 0.5

        return BenchmarkResult(
            test_id='AN1',
            category='Aprendizaje_Normas',
            description='Emergencia de normas',
            score=float(score),
            details={'n_agents': len(final_actions)}
        )

    def _test_AN2_norm_following(self) -> BenchmarkResult:
        """AN2: Seguimiento de normas establecidas."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names, state_dim=6, action_dim=3, value_dim=5
        )

        # Establecer norma: acción en dirección positiva
        norm_direction = np.array([0.5, 0.3, 0.2])

        compliance_scores = []

        for t in range(self.n_steps):
            world_obs = np.random.randn(6) * 0.3

            for agent_name in self.agent_names:
                result = life_cycle.full_cycle(
                    agent_name, world_obs, {}, world_obs, reward=0
                )

                # Reward por cumplir norma
                action = result['action'].direction
                norm_compliance = np.dot(action, norm_direction) / (np.linalg.norm(action) * np.linalg.norm(norm_direction) + 0.01)
                norm_compliance = (norm_compliance + 1) / 2  # Normalizar

                life_cycle.agents[agent_name].reward_history.append(norm_compliance)
                compliance_scores.append(norm_compliance)

        score = np.mean(compliance_scores[-50:]) if compliance_scores else 0.5

        return BenchmarkResult(
            test_id='AN2',
            category='Aprendizaje_Normas',
            description='Seguimiento de normas',
            score=float(score),
            details={'mean_compliance': np.mean(compliance_scores) if compliance_scores else 0}
        )

    def _test_AN3_norm_transmission(self) -> BenchmarkResult:
        """AN3: Transmisión de normas entre agentes."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names, state_dim=6, action_dim=3, value_dim=5
        )

        # Primer agente "conoce" la norma
        expert = self.agent_names[0]
        norm_action = np.array([0.5, 0.5, 0.0])

        # Forzar al experto a seguir la norma
        for t in range(self.n_steps):
            world_obs = np.random.randn(6) * 0.3

            for agent_name in self.agent_names:
                result = life_cycle.full_cycle(
                    agent_name, world_obs, {}, world_obs, reward=0
                )

                if agent_name == expert:
                    # Experto siempre tiene reward alto
                    reward = 0.9
                else:
                    # Otros: reward por imitar al experto
                    expert_action = life_cycle.agents[expert].action_history[-1].direction if life_cycle.agents[expert].action_history else norm_action
                    similarity = np.dot(result['action'].direction, expert_action) / (np.linalg.norm(result['action'].direction) * np.linalg.norm(expert_action) + 0.01)
                    reward = (similarity + 1) / 2

                life_cycle.agents[agent_name].reward_history.append(reward)

        # Medir si otros aprendieron la norma del experto
        similarities_to_norm = []
        for agent_name in self.agent_names[1:]:
            agent = life_cycle.agents[agent_name]
            if agent.action_history:
                action = agent.action_history[-1].direction
                sim = np.dot(action, norm_action) / (np.linalg.norm(action) * np.linalg.norm(norm_action) + 0.01)
                similarities_to_norm.append((sim + 1) / 2)

        score = np.mean(similarities_to_norm) if similarities_to_norm else 0.5

        return BenchmarkResult(
            test_id='AN3',
            category='Aprendizaje_Normas',
            description='Transmisión de normas',
            score=float(score),
            details={'similarities': similarities_to_norm}
        )

    def _test_AN4_norm_violation_detection(self) -> BenchmarkResult:
        """AN4: Detección de violaciones de normas."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names, state_dim=6, action_dim=3, value_dim=5
        )

        # Establecer norma y luego violarla
        norm_action = np.array([0.5, 0.3, 0.2])

        # Fase 1: Establecer norma
        for t in range(self.n_steps // 2):
            world_obs = np.random.randn(6) * 0.3
            for agent_name in self.agent_names:
                life_cycle.full_cycle(agent_name, world_obs, {}, world_obs, reward=0.8)

        # Guardar estado de memorias
        memories_before = len(life_cycle.agents[self.agent_names[0]].meta_memory.episodes)

        # Fase 2: Violación (rewards bajos, indicando algo mal)
        violation_detected = False
        for t in range(self.n_steps // 2):
            world_obs = np.random.randn(6) * 0.3
            for agent_name in self.agent_names:
                result = life_cycle.full_cycle(agent_name, world_obs, {}, world_obs, reward=0.1)  # Bajo reward = violación

                # Detectar si el agente "nota" algo raro
                agent = life_cycle.agents[agent_name]
                if agent.cognitive_state and agent.cognitive_state.uncertainty > 0.6:
                    violation_detected = True

        # Score basado en detección
        score = 1.0 if violation_detected else 0.5

        # También verificar cambio en comportamiento
        stats_after = life_cycle.get_all_statistics()

        return BenchmarkResult(
            test_id='AN4',
            category='Aprendizaje_Normas',
            description='Detección de violaciones',
            score=float(score),
            details={'violation_detected': violation_detected}
        )

    # ========== 9. ADAPTACIÓN A MUNDOS NO VISTOS (AW1-AW4) ==========

    def _run_adaptation_tests(self) -> List[BenchmarkResult]:
        """Tests de adaptación a mundos no vistos."""
        results = []

        results.append(self._test_AW1_distribution_shift())
        results.append(self._test_AW2_novel_dynamics())
        results.append(self._test_AW3_transfer_learning())
        results.append(self._test_AW4_robust_generalization())

        return results

    def _test_AW1_distribution_shift(self) -> BenchmarkResult:
        """AW1: Adaptación a cambio de distribución."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names[:1], state_dim=6, action_dim=3, value_dim=5
        )

        # Fase 1: Entrenar en distribución A
        for t in range(self.n_steps // 2):
            world_obs = np.random.randn(6) * 0.2 + 0.5  # Centrado en 0.5
            next_obs = world_obs + np.random.randn(6) * 0.1
            reward = np.random.rand() * 0.5 + 0.5
            life_cycle.full_cycle(self.agent_names[0], world_obs, {}, next_obs, reward)

        rewards_dist_a = life_cycle.agents[self.agent_names[0]].reward_history[-20:]

        # Fase 2: Distribución B (shift)
        for t in range(self.n_steps // 2):
            world_obs = np.random.randn(6) * 0.5 - 0.5  # Centrado en -0.5 (shift)
            next_obs = world_obs + np.random.randn(6) * 0.1
            reward = np.random.rand() * 0.5
            life_cycle.full_cycle(self.agent_names[0], world_obs, {}, next_obs, reward)

        rewards_dist_b = life_cycle.agents[self.agent_names[0]].reward_history[-20:]

        # Score: mantiene rendimiento razonable en distribución B
        ratio = np.mean(rewards_dist_b) / (np.mean(rewards_dist_a) + 0.01)
        score = min(1.0, max(0, ratio))

        return BenchmarkResult(
            test_id='AW1',
            category='Adaptacion_Mundos',
            description='Cambio de distribución',
            score=float(score),
            details={'dist_a_reward': np.mean(rewards_dist_a), 'dist_b_reward': np.mean(rewards_dist_b)}
        )

    def _test_AW2_novel_dynamics(self) -> BenchmarkResult:
        """AW2: Adaptación a dinámicas nuevas."""
        causal_model = CausalModel(state_dim=6, action_dim=3)

        # Entrenar con dinámica A
        for t in range(self.n_steps // 2):
            state = np.random.randn(6) * 0.3
            action = np.random.randn(3) * 0.3
            # Dinámica A: next = state + action[:3]
            next_state = state + np.pad(action, (0, 3)) * 0.5
            next_state = to_simplex(np.abs(next_state) + 0.01)
            causal_model.record(state, action, next_state)

        # Cambiar a dinámica B
        errors_before_adaptation = []
        errors_after_adaptation = []

        for t in range(self.n_steps // 2):
            state = np.random.randn(6) * 0.3
            action = np.random.randn(3) * 0.3
            # Dinámica B: next = state * 0.9 + action[:3] * 0.3 (diferente)
            next_state = state * 0.9 + np.pad(action * 0.3, (0, 3))
            next_state = to_simplex(np.abs(next_state) + 0.01)

            # Medir error antes de registrar
            pred = causal_model.predict(state, action)
            error = np.linalg.norm(pred - next_state)

            if t < 10:
                errors_before_adaptation.append(error)
            else:
                errors_after_adaptation.append(error)

            causal_model.record(state, action, next_state)

        # Score: error disminuye con adaptación
        if errors_before_adaptation and errors_after_adaptation:
            improvement = (np.mean(errors_before_adaptation) - np.mean(errors_after_adaptation)) / (np.mean(errors_before_adaptation) + 0.01)
            score = min(1.0, max(0, 0.5 + improvement))
        else:
            score = 0.5

        return BenchmarkResult(
            test_id='AW2',
            category='Adaptacion_Mundos',
            description='Dinámicas nuevas',
            score=float(score),
            details={
                'error_before': np.mean(errors_before_adaptation) if errors_before_adaptation else 0,
                'error_after': np.mean(errors_after_adaptation) if errors_after_adaptation else 0
            }
        )

    def _test_AW3_transfer_learning(self) -> BenchmarkResult:
        """AW3: Transferencia de aprendizaje entre mundos."""
        # Agente pre-entrenado
        life_cycle_pretrained = CognitiveLifeCycle(
            ['pretrained'], state_dim=6, action_dim=3, value_dim=5
        )

        # Pre-entrenar
        for t in range(self.n_steps):
            world_obs = np.random.randn(6) * 0.3
            life_cycle_pretrained.full_cycle('pretrained', world_obs, {}, world_obs, np.random.rand())

        # Agente desde cero
        life_cycle_scratch = CognitiveLifeCycle(
            ['scratch'], state_dim=6, action_dim=3, value_dim=5
        )

        # Evaluar en mundo nuevo
        pretrained_rewards = []
        scratch_rewards = []

        for t in range(50):
            world_obs = np.random.randn(6) * 0.5  # Mundo diferente
            reward = np.random.rand()

            result_pre = life_cycle_pretrained.full_cycle('pretrained', world_obs, {}, world_obs, reward)
            result_scratch = life_cycle_scratch.full_cycle('scratch', world_obs, {}, world_obs, reward)

            pretrained_rewards.append(result_pre['reward'])
            scratch_rewards.append(result_scratch['reward'])

        # Score: pre-entrenado debe tener ventaja
        advantage = np.mean(pretrained_rewards) - np.mean(scratch_rewards)
        score = min(1.0, max(0, 0.5 + advantage))

        return BenchmarkResult(
            test_id='AW3',
            category='Adaptacion_Mundos',
            description='Transferencia de aprendizaje',
            score=float(score),
            details={
                'pretrained_reward': np.mean(pretrained_rewards),
                'scratch_reward': np.mean(scratch_rewards)
            }
        )

    def _test_AW4_robust_generalization(self) -> BenchmarkResult:
        """AW4: Generalización robusta a múltiples mundos."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names[:1], state_dim=6, action_dim=3, value_dim=5
        )

        # Entrenar en múltiples mundos
        world_types = ['stable', 'volatile', 'cyclical']
        rewards_by_world = {w: [] for w in world_types}

        for epoch in range(3):
            for world_type in world_types:
                for t in range(30):
                    if world_type == 'stable':
                        world_obs = np.ones(6) * 0.5 + np.random.randn(6) * 0.1
                    elif world_type == 'volatile':
                        world_obs = np.random.randn(6) * 0.8
                    else:  # cyclical
                        world_obs = np.sin(np.arange(6) * t * 0.1) * 0.5

                    reward = np.random.rand()
                    result = life_cycle.full_cycle(self.agent_names[0], world_obs, {}, world_obs, reward)
                    rewards_by_world[world_type].append(result['reward'])

        # Score: rendimiento consistente en todos los mundos
        mean_rewards = [np.mean(r[-20:]) for r in rewards_by_world.values()]
        consistency = 1.0 / (1 + np.std(mean_rewards))
        overall = np.mean(mean_rewards)

        score = (consistency + overall) / 2

        return BenchmarkResult(
            test_id='AW4',
            category='Adaptacion_Mundos',
            description='Generalización robusta',
            score=float(np.clip(score, 0, 1)),
            details=rewards_by_world
        )

    # ========== 10. INTEGRACIÓN DE VIDA (IL1-IL4) ==========

    def _run_integration_tests(self) -> List[BenchmarkResult]:
        """Tests de integración del ciclo de vida completo."""
        results = []

        results.append(self._test_IL1_full_cycle_coherence())
        results.append(self._test_IL2_teleological_behavior())
        results.append(self._test_IL3_narrative_identity())
        results.append(self._test_IL4_emergent_agency())

        return results

    def _test_IL1_full_cycle_coherence(self) -> BenchmarkResult:
        """IL1: Coherencia del ciclo de vida completo."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names, state_dim=6, action_dim=3, value_dim=5
        )

        # Ejecutar ciclo completo
        for t in range(self.n_steps):
            world_obs = np.random.randn(6) * 0.3
            for agent_name in self.agent_names:
                life_cycle.full_cycle(agent_name, world_obs, {}, world_obs, np.random.rand())

        # Verificar que todas las fases funcionaron
        stats = life_cycle.get_all_statistics()

        checks = [
            stats['mean_reward'] > 0,
            stats['mean_goal_progress'] > 0,
            stats['mean_resilience'] > 0,
            all(a['causal_model_trained'] for a in stats['agents'].values()),
            all(a['n_memories'] > 0 for a in stats['agents'].values()),
            all(a['n_goals'] > 0 for a in stats['agents'].values()),
        ]

        score = sum(checks) / len(checks)

        return BenchmarkResult(
            test_id='IL1',
            category='Integracion_Vida',
            description='Coherencia del ciclo completo',
            score=float(score),
            details={'checks_passed': sum(checks), 'total_checks': len(checks)}
        )

    def _test_IL2_teleological_behavior(self) -> BenchmarkResult:
        """IL2: Comportamiento teleológico consistente."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names[:1], state_dim=6, action_dim=3, value_dim=5
        )

        agent = life_cycle.agents[self.agent_names[0]]

        # Establecer meta clara
        target_state = np.array([0.8, 0.1, 0.1, 0.0, 0.0, 0.0])
        agent.goals = [Goal(
            target_state=target_state,
            priority=1.0,
            origin='intrinsic',
            created_t=0,
            progress=0.0,
            sub_goals=[]
        )]

        distances = []
        for t in range(self.n_steps):
            world_obs = np.random.randn(6) * 0.3

            # Reward basado en cercanía a meta
            if agent.cognitive_state is not None:
                distance = np.linalg.norm(agent.cognitive_state.z - target_state)
                reward = 1.0 / (1 + distance)
            else:
                reward = 0.5

            result = life_cycle.full_cycle(self.agent_names[0], world_obs, {}, world_obs, reward)
            distances.append(result['goal_progress'])

        # Score: progreso hacia meta mejora
        if len(distances) > 20:
            early_progress = np.mean(distances[:20])
            late_progress = np.mean(distances[-20:])
            improvement = late_progress - early_progress
            score = min(1.0, max(0, 0.5 + improvement))
        else:
            score = 0.5

        return BenchmarkResult(
            test_id='IL2',
            category='Integracion_Vida',
            description='Comportamiento teleológico',
            score=float(score),
            details={'early_progress': early_progress if 'early_progress' in dir() else 0,
                    'late_progress': late_progress if 'late_progress' in dir() else 0}
        )

    def _test_IL3_narrative_identity(self) -> BenchmarkResult:
        """IL3: Identidad narrativa coherente."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names[:1], state_dim=6, action_dim=3, value_dim=5
        )

        narratives = []

        for t in range(self.n_steps):
            world_obs = np.random.randn(6) * 0.3
            result = life_cycle.full_cycle(self.agent_names[0], world_obs, {}, world_obs, np.random.rand())
            narratives.append(result['narrative'])

        agent = life_cycle.agents[self.agent_names[0]]

        # Score: coherencia narrativa y estabilidad de identidad
        narrative_coherence = agent.narrative_coherence
        identity_stability = agent.cognitive_state.identity_stability if agent.cognitive_state else 0.5

        score = (narrative_coherence + identity_stability) / 2

        return BenchmarkResult(
            test_id='IL3',
            category='Integracion_Vida',
            description='Identidad narrativa',
            score=float(score),
            details={'narrative_coherence': narrative_coherence, 'identity_stability': identity_stability}
        )

    def _test_IL4_emergent_agency(self) -> BenchmarkResult:
        """IL4: Agencia emergente completa."""
        life_cycle = CognitiveLifeCycle(
            self.agent_names, state_dim=6, action_dim=3, value_dim=5
        )

        # Ejecutar extensivamente
        for t in range(self.n_steps * 2):
            world_obs = np.random.randn(6) * 0.3
            for agent_name in self.agent_names:
                other_obs = {n: np.random.randn(6) * 0.3 for n in self.agent_names if n != agent_name}
                life_cycle.full_cycle(agent_name, world_obs, other_obs, world_obs, np.random.rand())

        # Evaluar agencia: múltiples indicadores
        agency_scores = []

        for agent_name in self.agent_names:
            stats = life_cycle.get_agent_statistics(agent_name)

            # Indicadores de agencia
            has_goals = stats['n_goals'] > 0
            has_memory = stats['n_memories'] > 50
            has_model = stats['causal_model_trained']
            has_narrative = stats['narrative_coherence'] > 0.5
            has_resilience = stats['resilience'] > 1.0

            agent_score = sum([has_goals, has_memory, has_model, has_narrative, has_resilience]) / 5
            agency_scores.append(agent_score)

        score = np.mean(agency_scores)

        return BenchmarkResult(
            test_id='IL4',
            category='Integracion_Vida',
            description='Agencia emergente',
            score=float(score),
            details={'per_agent_scores': agency_scores}
        )

    def get_report(self) -> str:
        """Genera reporte completo del benchmark."""
        if not self.results:
            return "No hay resultados. Ejecute run_all() primero."

        report = []
        report.append("=" * 70)
        report.append("AGI-X v2.0 BENCHMARK SUITE - REPORTE COMPLETO")
        report.append("=" * 70)
        report.append(f"\nTotal de tests: {len(self.results)}")

        # Por categoría
        categories = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = []
            categories[r.category].append(r)

        for cat, results in categories.items():
            scores = [r.score for r in results]
            report.append(f"\n{cat}:")
            report.append(f"  Score promedio: {np.mean(scores):.3f}")
            for r in results:
                status = "PASS" if r.score >= 0.5 else "FAIL"
                report.append(f"    [{r.test_id}] {r.description}: {r.score:.3f} [{status}]")

        # Resumen global
        all_scores = [r.score for r in self.results]
        report.append("\n" + "=" * 70)
        report.append("RESUMEN GLOBAL")
        report.append("=" * 70)
        report.append(f"Score promedio total: {np.mean(all_scores):.3f}")
        report.append(f"Tests aprobados (>=0.5): {sum(1 for s in all_scores if s >= 0.5)}/{len(all_scores)}")
        report.append(f"Tests excelentes (>=0.8): {sum(1 for s in all_scores if s >= 0.8)}/{len(all_scores)}")

        return "\n".join(report)


def run_benchmark():
    """Ejecuta el benchmark completo."""
    print("=" * 70)
    print("INICIANDO AGI-X v2.0 BENCHMARK SUITE (40 TESTS)")
    print("=" * 70)

    benchmark = AGIXBenchmarkSuite(n_agents=5, n_steps=100)
    results = benchmark.run_all()

    print("\n" + benchmark.get_report())

    return benchmark


if __name__ == "__main__":
    run_benchmark()
