"""
Cognitive Action Layer: Capa de Decisión Cognitiva
==================================================

Acciones moduladas por todos los módulos AGI:
- Self-Model → confianza en predicciones propias
- ToM → anticipación de otros
- Ethics → filtro de acciones
- Planning → metas de largo plazo
- Curiosity → exploración
- Norms → respeto a normas emergentes
- Meta-Rules → adaptación de estrategia

La acción final es el resultado de la cognición integrada.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import (
    L_t, max_history, adaptive_learning_rate, adaptive_momentum,
    to_simplex, softmax, normalized_entropy
)


@dataclass
class CognitiveDecision:
    """Decisión cognitiva integrada."""
    agent_name: str
    direction: np.ndarray         # Dirección de movimiento/acción
    magnitude: float              # Intensidad de la acción
    confidence: float             # Confianza en la decisión
    target_agent: Optional[str]   # Agente objetivo (si aplica)
    goal_alignment: float         # Alineación con meta actual
    ethical_score: float          # Evaluación ética
    curiosity_bonus: float        # Bonus por exploración
    tom_anticipation: Dict[str, float]  # Anticipación de reacciones
    reasoning: str                # Explicación de la decisión
    t: int


@dataclass
class ActionOutcome:
    """Resultado de una acción en el mundo."""
    agent_name: str
    intended_direction: np.ndarray
    actual_change: np.ndarray
    reward_signal: float
    surprise: float
    goal_progress: float
    social_feedback: Dict[str, float]
    t: int


class CognitiveActionLayer:
    """
    Capa que integra cognición AGI para generar acciones.

    Cada decisión combina:
    1. Self-Model: ¿Qué predigo que pasará si hago esto?
    2. ToM: ¿Cómo reaccionarán los otros?
    3. Ethics: ¿Es esta acción aceptable?
    4. Planning: ¿Me acerca a mi meta?
    5. Curiosity: ¿Exploro o exploto?
    6. Norms: ¿Viola normas del grupo?
    7. Reconfiguration: ¿Qué módulo priorizo?
    """

    def __init__(self, agent_name: str):
        """
        Inicializa capa cognitiva para un agente.

        Args:
            agent_name: Nombre del agente
        """
        self.agent_name = agent_name

        # Módulos cognitivos (se conectarán externamente)
        self.self_model = None
        self.tom_system = None
        self.ethics_module = None
        self.planning_module = None
        self.curiosity_module = None
        self.norm_system = None
        self.reconfig_system = None

        # Estado interno
        self.current_goal: Optional[np.ndarray] = None
        self.goal_history: List[np.ndarray] = []

        # Historial de decisiones y outcomes
        self.decision_history: List[CognitiveDecision] = []
        self.outcome_history: List[ActionOutcome] = []

        # Pesos de módulos (se ajustan por reconfiguration)
        self.module_weights = {
            'self_model': 1.0,
            'tom': 1.0,
            'ethics': 1.0,
            'planning': 1.0,
            'curiosity': 1.0,
            'norms': 1.0
        }

        # Estadísticas para aprendizaje
        self.reward_history: List[float] = []
        self.surprise_history: List[float] = []

        self.t = 0

    def connect_modules(self, self_model=None, tom_system=None,
                       ethics_module=None, planning_module=None,
                       curiosity_module=None, norm_system=None,
                       reconfig_system=None):
        """Conecta módulos cognitivos."""
        self.self_model = self_model
        self.tom_system = tom_system
        self.ethics_module = ethics_module
        self.planning_module = planning_module
        self.curiosity_module = curiosity_module
        self.norm_system = norm_system
        self.reconfig_system = reconfig_system

    def set_goal(self, goal: np.ndarray):
        """Establece meta actual."""
        self.current_goal = goal.copy()
        self.goal_history.append(goal.copy())
        max_hist = max_history(self.t)
        if len(self.goal_history) > max_hist:
            self.goal_history = self.goal_history[-max_hist:]

    def _compute_self_model_prediction(self, state: Dict, action_candidate: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Usa self-model para predecir resultado de acción.

        Returns:
            (confidence, predicted_next_state)
        """
        if self.self_model is None:
            return 0.5, state.get('z', np.zeros(6))

        # Construir estado actual
        z = state.get('z', np.zeros(6))
        phi = state.get('phi', np.zeros(5))
        drives = state.get('drives', np.zeros(6))

        current_state = np.concatenate([z, phi, drives])

        # Predecir siguiente estado
        predicted = self.self_model.predict_k_steps(current_state, 1)
        confidence = self.self_model.confidence()

        return confidence, predicted

    def _compute_tom_anticipation(self, state: Dict,
                                  others: Dict[str, Dict],
                                  action_candidate: np.ndarray) -> Dict[str, float]:
        """
        Usa ToM para anticipar reacciones de otros.

        Returns:
            Dict[other_name, anticipated_reaction_magnitude]
        """
        anticipations = {}

        if self.tom_system is None:
            for other in others:
                anticipations[other] = 0.5
            return anticipations

        for other_name, other_state in others.items():
            model = self.tom_system.get_model(self.agent_name, other_name)

            if model and len(model.observation_history) > 0:
                # Predecir cómo cambiará el otro
                last_obs = model.observation_history[-1]
                predicted = model.predict_k_steps(last_obs, 1)

                # Anticipación = magnitud del cambio predicho
                change = np.linalg.norm(predicted - last_obs)
                tom_acc = model.tom_accuracy_score()

                # Ponderar por ToM accuracy
                anticipations[other_name] = float(change * tom_acc)
            else:
                anticipations[other_name] = 0.5

        return anticipations

    def _compute_ethical_score(self, action_candidate: np.ndarray,
                               state: Dict, others: Dict) -> float:
        """
        Evalúa acción éticamente.

        Returns:
            Score ético [0, 1], mayor = más ético
        """
        if self.ethics_module is None:
            # Sin módulo: evaluación básica
            # Evitar acciones muy extremas
            magnitude = np.linalg.norm(action_candidate)
            return float(1.0 - min(1.0, magnitude / 2.0))

        # Con módulo: usar evaluación estructural
        # TODO: Integrar con StructuralEthics
        return 0.8

    def _compute_goal_alignment(self, action_candidate: np.ndarray,
                                current_position: np.ndarray) -> float:
        """
        Calcula alineación de acción con meta.

        Returns:
            Alineación [-1, 1], mayor = más alineado
        """
        if self.current_goal is None:
            return 0.0

        # Dirección hacia la meta
        to_goal = self.current_goal - current_position
        to_goal_norm = to_goal / (np.linalg.norm(to_goal) + 1e-8)

        # Dirección de la acción
        action_norm = action_candidate / (np.linalg.norm(action_candidate) + 1e-8)

        # Producto punto = coseno del ángulo
        alignment = float(np.dot(to_goal_norm[:len(action_norm)], action_norm))

        return alignment

    def _compute_curiosity_bonus(self, state: Dict,
                                 action_candidate: np.ndarray) -> float:
        """
        Calcula bonus de curiosidad por exploración.

        Returns:
            Bonus [0, 1]
        """
        if self.curiosity_module is None:
            # Sin módulo: bonus basado en novedad de la posición
            if len(self.decision_history) < 5:
                return 0.5

            # Comparar con acciones recientes
            recent_directions = [d.direction for d in self.decision_history[-10:]]
            if not recent_directions:
                return 0.5

            mean_direction = np.mean(recent_directions, axis=0)
            novelty = np.linalg.norm(action_candidate - mean_direction)

            return float(min(1.0, novelty))

        # Con módulo: usar curiosidad estructural
        # TODO: Integrar con StructuralCuriosity
        return 0.5

    def _compute_norm_compliance(self, action_candidate: np.ndarray,
                                 state: Dict) -> float:
        """
        Evalúa cumplimiento de normas emergentes.

        Returns:
            Compliance [0, 1]
        """
        if self.norm_system is None:
            return 1.0  # Sin normas, todo permitido

        # Con módulo: evaluar contra normas activas
        # TODO: Integrar con NormSystem
        return 0.9

    def _update_module_weights(self):
        """Actualiza pesos de módulos basado en resultados."""
        if self.reconfig_system is None:
            return

        # Obtener pesos del sistema de reconfiguración
        for module_name in self.module_weights:
            weight = self.reconfig_system.get_module_weight(module_name)
            if weight > 0:
                self.module_weights[module_name] = weight

    def generate_action_candidates(self, state: Dict, n_candidates: int = 5) -> List[np.ndarray]:
        """
        Genera candidatos de acción.

        Returns:
            Lista de vectores de acción candidatos
        """
        candidates = []

        position = state.get('z', np.zeros(6))[:3]

        # Candidato 1: hacia la meta (si existe)
        if self.current_goal is not None:
            to_goal = self.current_goal[:3] - position
            to_goal = to_goal / (np.linalg.norm(to_goal) + 1e-8) * 0.1
            candidates.append(to_goal)

        # Candidato 2: continuar dirección actual
        if self.decision_history:
            last_dir = self.decision_history[-1].direction
            candidates.append(last_dir * 0.9)

        # Candidatos aleatorios (exploración)
        while len(candidates) < n_candidates:
            random_dir = np.random.randn(3)
            random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-8) * 0.1
            candidates.append(random_dir)

        return candidates

    def decide(self, state: Dict, others: Dict[str, Dict]) -> CognitiveDecision:
        """
        Genera decisión cognitiva integrada.

        Args:
            state: Estado cognitivo propio (z, phi, drives, context)
            others: Estados de otros agentes para ToM

        Returns:
            CognitiveDecision
        """
        self.t += 1

        # Actualizar pesos de módulos
        self._update_module_weights()

        # Generar candidatos
        candidates = self.generate_action_candidates(state)

        # Evaluar cada candidato
        best_score = -np.inf
        best_candidate = candidates[0]
        best_eval = {}

        position = state.get('z', np.zeros(6))[:3]

        for candidate in candidates:
            # 1. Self-Model: confianza
            sm_confidence, predicted = self._compute_self_model_prediction(state, candidate)

            # 2. ToM: anticipación
            tom_anticipation = self._compute_tom_anticipation(state, others, candidate)

            # 3. Ethics: evaluación
            ethical_score = self._compute_ethical_score(candidate, state, others)

            # 4. Planning: alineación con meta
            goal_alignment = self._compute_goal_alignment(candidate, position)

            # 5. Curiosity: bonus exploración
            curiosity_bonus = self._compute_curiosity_bonus(state, candidate)

            # 6. Norms: cumplimiento
            norm_compliance = self._compute_norm_compliance(candidate, state)

            # Score compuesto (ponderado por module_weights)
            score = (
                self.module_weights['self_model'] * sm_confidence +
                self.module_weights['tom'] * (1 - np.mean(list(tom_anticipation.values()))) +
                self.module_weights['ethics'] * ethical_score +
                self.module_weights['planning'] * (goal_alignment + 1) / 2 +
                self.module_weights['curiosity'] * curiosity_bonus +
                self.module_weights['norms'] * norm_compliance
            )

            if score > best_score:
                best_score = score
                best_candidate = candidate
                best_eval = {
                    'confidence': sm_confidence,
                    'tom_anticipation': tom_anticipation,
                    'ethical_score': ethical_score,
                    'goal_alignment': goal_alignment,
                    'curiosity_bonus': curiosity_bonus,
                    'norm_compliance': norm_compliance
                }

        # Construir decisión
        decision = CognitiveDecision(
            agent_name=self.agent_name,
            direction=best_candidate,
            magnitude=float(np.linalg.norm(best_candidate)),
            confidence=best_eval.get('confidence', 0.5),
            target_agent=None,
            goal_alignment=best_eval.get('goal_alignment', 0.0),
            ethical_score=best_eval.get('ethical_score', 0.8),
            curiosity_bonus=best_eval.get('curiosity_bonus', 0.5),
            tom_anticipation=best_eval.get('tom_anticipation', {}),
            reasoning=self._generate_reasoning(best_eval),
            t=self.t
        )

        # Guardar en historial
        self.decision_history.append(decision)
        max_hist = max_history(self.t)
        if len(self.decision_history) > max_hist:
            self.decision_history = self.decision_history[-max_hist:]

        return decision

    def _generate_reasoning(self, evaluation: Dict) -> str:
        """Genera explicación de la decisión."""
        parts = []

        if evaluation.get('goal_alignment', 0) > 0.5:
            parts.append("hacia meta")
        elif evaluation.get('goal_alignment', 0) < -0.3:
            parts.append("alejándose de meta")

        if evaluation.get('curiosity_bonus', 0) > 0.7:
            parts.append("explorando")

        if evaluation.get('ethical_score', 1) < 0.5:
            parts.append("ética cuestionable")

        if evaluation.get('confidence', 0) > 0.8:
            parts.append("alta confianza")

        return "; ".join(parts) if parts else "decisión estándar"

    def process_outcome(self, actual_change: np.ndarray,
                        reward: float, surprise: float,
                        social_feedback: Dict[str, float]) -> ActionOutcome:
        """
        Procesa el resultado de la acción.

        Args:
            actual_change: Cambio real en el mundo
            reward: Señal de recompensa
            surprise: Nivel de sorpresa
            social_feedback: Feedback de otros agentes

        Returns:
            ActionOutcome
        """
        if not self.decision_history:
            return None

        last_decision = self.decision_history[-1]

        # Calcular progreso hacia meta
        goal_progress = 0.0
        if self.current_goal is not None and len(actual_change) >= 3:
            # Reducción de distancia a meta
            goal_progress = -np.linalg.norm(actual_change[:3])  # Simplificado

        outcome = ActionOutcome(
            agent_name=self.agent_name,
            intended_direction=last_decision.direction,
            actual_change=actual_change,
            reward_signal=reward,
            surprise=surprise,
            goal_progress=goal_progress,
            social_feedback=social_feedback,
            t=self.t
        )

        # Guardar en historiales
        self.outcome_history.append(outcome)
        self.reward_history.append(reward)
        self.surprise_history.append(surprise)

        max_hist = max_history(self.t)
        if len(self.outcome_history) > max_hist:
            self.outcome_history = self.outcome_history[-max_hist:]
        if len(self.reward_history) > max_hist:
            self.reward_history = self.reward_history[-max_hist:]
        if len(self.surprise_history) > max_hist:
            self.surprise_history = self.surprise_history[-max_hist:]

        # Actualizar módulos con feedback
        self._update_from_outcome(outcome)

        return outcome

    def _update_from_outcome(self, outcome: ActionOutcome):
        """Actualiza módulos cognitivos con el resultado."""
        # Si hay sistema de reconfiguración, registrar activaciones
        if self.reconfig_system is not None:
            activations = {
                'self_model': outcome.surprise,  # Alta sorpresa = self-model falló
                'tom': np.mean(list(outcome.social_feedback.values())) if outcome.social_feedback else 0.5,
                'planning': max(0, outcome.goal_progress),
                'curiosity': outcome.surprise,  # Sorpresa = novedad
                'norms': 1 - abs(outcome.reward_signal)  # Simplificado
            }
            self.reconfig_system.record_activations(activations, outcome.reward_signal)

    def get_statistics(self) -> Dict:
        """Estadísticas de la capa cognitiva."""
        return {
            'agent': self.agent_name,
            't': self.t,
            'n_decisions': len(self.decision_history),
            'n_outcomes': len(self.outcome_history),
            'mean_confidence': np.mean([d.confidence for d in self.decision_history[-20:]]) if self.decision_history else 0,
            'mean_reward': np.mean(self.reward_history[-20:]) if self.reward_history else 0,
            'mean_surprise': np.mean(self.surprise_history[-20:]) if self.surprise_history else 0,
            'module_weights': self.module_weights.copy(),
            'has_goal': self.current_goal is not None
        }


def test_cognitive_action_layer():
    """Test de CognitiveActionLayer."""
    print("=" * 60)
    print("TEST: COGNITIVE ACTION LAYER")
    print("=" * 60)

    # Crear capa cognitiva para NEO
    neo_brain = CognitiveActionLayer("NEO")

    # Establecer meta
    goal = np.array([0.8, 0.8, 0.8])
    neo_brain.set_goal(goal)

    print(f"\nAgente: NEO")
    print(f"Meta: {goal}")

    # Simular decisiones
    for t in range(50):
        # Estado simulado
        state = {
            'z': np.random.rand(6),
            'phi': np.random.rand(5),
            'drives': np.random.rand(6),
            'context': np.random.rand(10)
        }

        # Otros agentes simulados
        others = {
            'EVA': {'z': np.random.rand(6), 'phi': np.random.rand(5), 'drives': np.random.rand(6)},
            'ALEX': {'z': np.random.rand(6), 'phi': np.random.rand(5), 'drives': np.random.rand(6)}
        }

        # Decidir
        decision = neo_brain.decide(state, others)

        # Simular outcome
        actual_change = decision.direction + np.random.randn(3) * 0.02
        reward = decision.goal_alignment * 0.5 + np.random.randn() * 0.1
        surprise = float(np.linalg.norm(actual_change - decision.direction))

        social_feedback = {
            'EVA': np.random.rand(),
            'ALEX': np.random.rand()
        }

        outcome = neo_brain.process_outcome(actual_change, reward, surprise, social_feedback)

        if (t + 1) % 10 == 0:
            stats = neo_brain.get_statistics()
            print(f"\n  t={t+1}:")
            print(f"    Confianza media: {stats['mean_confidence']:.3f}")
            print(f"    Recompensa media: {stats['mean_reward']:.3f}")
            print(f"    Último razonamiento: {decision.reasoning}")

    print("\n" + "=" * 60)
    print("COGNITIVE ACTION LAYER TEST COMPLETADO")
    print("=" * 60)

    return neo_brain


if __name__ == "__main__":
    test_cognitive_action_layer()
