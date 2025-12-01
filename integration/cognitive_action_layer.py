"""
Cognitive Action Layer: Capa de Decisión Cognitiva
==================================================

Acciones moduladas por todos los módulos AGI:
- Self-Model (AGI-4) → confianza en predicciones propias
- ToM (AGI-5) → anticipación de otros
- Ethics (AGI-15) → filtro de acciones
- Planning (AGI-6) → metas de largo plazo
- Curiosity (AGI-13) → exploración
- Norms (AGI-12) → respeto a normas emergentes
- Meta-Rules (AGI-16) → adaptación de estrategia por contexto
- Robustness (AGI-17) → generalización multi-mundo
- Reconfiguration (AGI-18) → ajuste dinámico de pesos
- Collective Intent (AGI-19) → alineación con intenciones colectivas

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

# Importar módulos AGI-16 a AGI-19
from cognition.agi16_meta_rules import StructuralMetaRules
from cognition.agi17_robustness import MultiWorldRobustness
from cognition.agi18_reconfiguration import ReflectiveReconfiguration
from cognition.agi19_collective_intent import CollectiveIntentionality


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
    # AGI-16 a AGI-19 contributions
    meta_rule_policy: int = 0     # Política sugerida por meta-reglas
    robustness_bonus: float = 0.0 # Bonus por robustez
    collective_alignment: float = 0.0  # Alineación con intención colectiva
    t: int = 0


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
    1. Self-Model (AGI-4): ¿Qué predigo que pasará si hago esto?
    2. ToM (AGI-5): ¿Cómo reaccionarán los otros?
    3. Ethics (AGI-15): ¿Es esta acción aceptable?
    4. Planning (AGI-6): ¿Me acerca a mi meta?
    5. Curiosity (AGI-13): ¿Exploro o exploto?
    6. Norms (AGI-12): ¿Viola normas del grupo?
    7. Meta-Rules (AGI-16): ¿Qué política usar en este contexto?
    8. Robustness (AGI-17): ¿Es robusto a perturbaciones?
    9. Reconfiguration (AGI-18): ¿Qué módulo priorizo?
    10. Collective Intent (AGI-19): ¿Me alineo con intención colectiva?
    """

    # Mapeo de políticas a nombres (para AGI-16)
    POLICY_NAMES = [
        'exploit',      # 0: explotar conocimiento
        'explore',      # 1: explorar nuevas opciones
        'cooperate',    # 2: colaborar con otros
        'compete',      # 3: competir
        'conserve',     # 4: conservar recursos
        'adapt',        # 5: adaptarse al cambio
        'balance'       # 6: equilibrio
    ]

    def __init__(self, agent_name: str, all_agent_names: List[str] = None):
        """
        Inicializa capa cognitiva para un agente.

        Args:
            agent_name: Nombre del agente
            all_agent_names: Lista de todos los agentes (para AGI-17, AGI-19)
        """
        self.agent_name = agent_name
        self.all_agent_names = all_agent_names or [agent_name]

        # Módulos cognitivos básicos (se conectarán externamente)
        self.self_model = None
        self.tom_system = None
        self.ethics_module = None
        self.planning_module = None
        self.curiosity_module = None
        self.norm_system = None

        # Nuevos módulos AGI-16 a AGI-19 (propios del agente)
        self.meta_rules = StructuralMetaRules(
            agent_name,
            context_dim=10,
            n_policies=len(self.POLICY_NAMES)
        )
        self.reconfig_system = ReflectiveReconfiguration(
            agent_name,
            module_names=['self_model', 'tom', 'ethics', 'planning',
                         'curiosity', 'norms', 'meta_rules', 'robustness',
                         'collective_intent']
        )

        # AGI-17 y AGI-19 son compartidos (se conectan externamente)
        self.robustness_system: Optional[MultiWorldRobustness] = None
        self.collective_intent: Optional[CollectiveIntentionality] = None

        # Estado interno
        self.current_goal: Optional[np.ndarray] = None
        self.goal_history: List[np.ndarray] = []
        self.current_context: np.ndarray = np.zeros(10)

        # Historial de decisiones y outcomes
        self.decision_history: List[CognitiveDecision] = []
        self.outcome_history: List[ActionOutcome] = []

        # Pesos de módulos (se ajustan por AGI-18 reconfiguration)
        self.module_weights = {
            'self_model': 1.0,
            'tom': 1.0,
            'ethics': 1.0,
            'planning': 1.0,
            'curiosity': 1.0,
            'norms': 1.0,
            'meta_rules': 0.8,
            'robustness': 0.7,
            'collective_intent': 0.6
        }

        # Estadísticas para aprendizaje
        self.reward_history: List[float] = []
        self.surprise_history: List[float] = []

        # Política actual sugerida por meta-reglas
        self.current_policy: int = 6  # balance por defecto

        self.t = 0

    def connect_modules(self, self_model=None, tom_system=None,
                       ethics_module=None, planning_module=None,
                       curiosity_module=None, norm_system=None,
                       robustness_system=None, collective_intent=None):
        """
        Conecta módulos cognitivos externos.

        Args:
            self_model: SelfPredictorV2 (AGI-4)
            tom_system: TheoryOfMindSystem (AGI-5)
            ethics_module: StructuralEthics (AGI-15)
            planning_module: Planning module (AGI-6)
            curiosity_module: StructuralCuriosity (AGI-13)
            norm_system: NormSystem (AGI-12)
            robustness_system: MultiWorldRobustness (AGI-17) - compartido
            collective_intent: CollectiveIntentionality (AGI-19) - compartido
        """
        self.self_model = self_model
        self.tom_system = tom_system
        self.ethics_module = ethics_module
        self.planning_module = planning_module
        self.curiosity_module = curiosity_module
        self.norm_system = norm_system
        # Sistemas compartidos AGI-17 y AGI-19
        self.robustness_system = robustness_system
        self.collective_intent = collective_intent

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
        """
        Actualiza pesos de módulos usando AGI-18 (Reconfiguration).

        Los pesos se ajustan basándose en la correlación de cada
        módulo con la utilidad obtenida.
        """
        # Obtener pesos del sistema de reconfiguración AGI-18
        for module_name in self.module_weights:
            weight = self.reconfig_system.get_module_weight(module_name)
            if weight > 0:
                self.module_weights[module_name] = weight

    def _get_meta_rule_policy(self, context: np.ndarray) -> Tuple[int, float]:
        """
        Usa AGI-16 (Meta-Rules) para obtener política óptima para el contexto.

        Args:
            context: Vector de contexto actual

        Returns:
            (policy_id, confidence)
        """
        policy, confidence = self.meta_rules.get_meta_policy(context)
        return policy, confidence

    def _get_robustness_bonus(self, action_entropy: float) -> float:
        """
        Usa AGI-17 (Robustness) para calcular bonus de robustez.

        Args:
            action_entropy: Entropía de la distribución de acción

        Returns:
            Bonus de robustez [0, 0.3]
        """
        if self.robustness_system is None:
            return 0.0

        return self.robustness_system.get_robustness_bonus(
            self.agent_name, action_entropy
        )

    def _get_collective_alignment(self, action_direction: np.ndarray) -> float:
        """
        Usa AGI-19 (Collective Intent) para calcular alineación con intención colectiva.

        Args:
            action_direction: Dirección de la acción propuesta (dim 3)

        Returns:
            Bonus por alineación [0, 0.3]
        """
        if self.collective_intent is None:
            return 0.0

        # Expandir action_direction a dim 10 para AGI-19
        action_expanded = np.zeros(10)
        action_expanded[:min(3, len(action_direction))] = action_direction[:3]

        return self.collective_intent.get_alignment_bonus(
            self.agent_name, action_expanded
        )

    def _update_context(self, state: Dict):
        """
        Actualiza contexto actual para AGI-16.

        El contexto combina:
        - Estado cognitivo (z, phi, drives)
        - Régimen del mundo
        - Recursos disponibles
        """
        z = state.get('z', np.zeros(6))
        phi = state.get('phi', np.zeros(5))
        context_raw = state.get('context', np.zeros(10))

        # Combinar en vector de contexto de 10 dims
        self.current_context = np.zeros(10)
        self.current_context[:min(3, len(z))] = z[:3]
        self.current_context[3:3+min(2, len(phi))] = phi[:2]
        self.current_context[5:] = context_raw[:5] if len(context_raw) >= 5 else np.zeros(5)

    def generate_action_candidates(self, state: Dict, n_candidates: int = 7) -> List[np.ndarray]:
        """
        Genera candidatos de acción considerando política de AGI-16.

        Returns:
            Lista de vectores de acción candidatos
        """
        candidates = []

        position = state.get('z', np.zeros(6))[:3]

        # Obtener política óptima de AGI-16
        policy, policy_confidence = self._get_meta_rule_policy(self.current_context)
        self.current_policy = policy

        # Candidato 1: hacia la meta (si existe) - planning
        if self.current_goal is not None:
            to_goal = self.current_goal[:3] - position
            to_goal = to_goal / (np.linalg.norm(to_goal) + 1e-8) * 0.1
            candidates.append(to_goal)

        # Candidato 2: continuar dirección actual - exploit
        if self.decision_history:
            last_dir = self.decision_history[-1].direction
            candidates.append(last_dir * 0.9)

        # Candidato 3: dirección colectiva (AGI-19) - cooperate
        if self.collective_intent is not None:
            collective_dir, coll_conf = self.collective_intent.get_collective_direction()
            if coll_conf > 0.3 and len(collective_dir) >= 3:
                candidates.append(collective_dir[:3] * 0.1)

        # Candidatos según política de AGI-16
        if policy == 0:  # exploit
            if self.decision_history:
                # Amplificar última dirección exitosa
                candidates.append(self.decision_history[-1].direction * 1.1)
        elif policy == 1:  # explore
            # Dirección ortogonal a última
            if self.decision_history:
                last = self.decision_history[-1].direction
                perp = np.array([-last[1], last[0], 0]) if len(last) >= 2 else np.random.randn(3)
                candidates.append(perp / (np.linalg.norm(perp) + 1e-8) * 0.15)
        elif policy == 2:  # cooperate
            # Hacia centroide de otros (si tenemos info)
            pass  # Ya cubierto por collective_dir
        elif policy == 3:  # compete
            # Alejarse de otros
            if self.collective_intent is not None:
                collective_dir, _ = self.collective_intent.get_collective_direction()
                if len(collective_dir) >= 3:
                    candidates.append(-collective_dir[:3] * 0.08)
        elif policy == 4:  # conserve
            # Movimiento mínimo
            candidates.append(np.zeros(3))
        elif policy == 5:  # adapt
            # Dirección basada en cambio reciente del mundo
            context = state.get('context', np.zeros(10))
            adapt_dir = context[:3] / (np.linalg.norm(context[:3]) + 1e-8) * 0.1
            candidates.append(adapt_dir)

        # Candidatos aleatorios (exploración)
        while len(candidates) < n_candidates:
            random_dir = np.random.randn(3)
            random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-8) * 0.1
            candidates.append(random_dir)

        return candidates

    def decide(self, state: Dict, others: Dict[str, Dict]) -> CognitiveDecision:
        """
        Genera decisión cognitiva integrada usando AGI-4 a AGI-19.

        Args:
            state: Estado cognitivo propio (z, phi, drives, context)
            others: Estados de otros agentes para ToM

        Returns:
            CognitiveDecision
        """
        self.t += 1

        # Actualizar contexto para AGI-16
        self._update_context(state)

        # Actualizar pesos de módulos (AGI-18)
        self._update_module_weights()

        # Generar candidatos (considera AGI-16 policy)
        candidates = self.generate_action_candidates(state)

        # Evaluar cada candidato
        best_score = -np.inf
        best_candidate = candidates[0]
        best_eval = {}

        position = state.get('z', np.zeros(6))[:3]

        for candidate in candidates:
            # 1. Self-Model (AGI-4): confianza
            sm_confidence, predicted = self._compute_self_model_prediction(state, candidate)

            # 2. ToM (AGI-5): anticipación
            tom_anticipation = self._compute_tom_anticipation(state, others, candidate)

            # 3. Ethics (AGI-15): evaluación
            ethical_score = self._compute_ethical_score(candidate, state, others)

            # 4. Planning (AGI-6): alineación con meta
            goal_alignment = self._compute_goal_alignment(candidate, position)

            # 5. Curiosity (AGI-13): bonus exploración
            curiosity_bonus = self._compute_curiosity_bonus(state, candidate)

            # 6. Norms (AGI-12): cumplimiento
            norm_compliance = self._compute_norm_compliance(candidate, state)

            # 7. Robustness (AGI-17): bonus por diversidad
            action_entropy = -np.sum(np.abs(candidate) * np.log(np.abs(candidate) + 1e-8))
            robustness_bonus = self._get_robustness_bonus(action_entropy)

            # 8. Collective Intent (AGI-19): alineación colectiva
            collective_alignment = self._get_collective_alignment(candidate)

            # Score compuesto (ponderado por module_weights de AGI-18)
            score = (
                self.module_weights['self_model'] * sm_confidence +
                self.module_weights['tom'] * (1 - np.mean(list(tom_anticipation.values()))) +
                self.module_weights['ethics'] * ethical_score +
                self.module_weights['planning'] * (goal_alignment + 1) / 2 +
                self.module_weights['curiosity'] * curiosity_bonus +
                self.module_weights['norms'] * norm_compliance +
                self.module_weights.get('robustness', 0.7) * robustness_bonus +
                self.module_weights.get('collective_intent', 0.6) * collective_alignment
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
                    'norm_compliance': norm_compliance,
                    'robustness_bonus': robustness_bonus,
                    'collective_alignment': collective_alignment
                }

        # Construir decisión con contribuciones AGI-16 a AGI-19
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
            meta_rule_policy=self.current_policy,
            robustness_bonus=best_eval.get('robustness_bonus', 0.0),
            collective_alignment=best_eval.get('collective_alignment', 0.0),
            t=self.t
        )

        # Guardar en historial
        self.decision_history.append(decision)
        max_hist = max_history(self.t)
        if len(self.decision_history) > max_hist:
            self.decision_history = self.decision_history[-max_hist:]

        return decision

    def _generate_reasoning(self, evaluation: Dict) -> str:
        """Genera explicación de la decisión incluyendo AGI-16 a AGI-19."""
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

        # AGI-16: política
        if self.current_policy < len(self.POLICY_NAMES):
            policy_name = self.POLICY_NAMES[self.current_policy]
            parts.append(f"política:{policy_name}")

        # AGI-17: robustez
        if evaluation.get('robustness_bonus', 0) > 0.15:
            parts.append("robusto")

        # AGI-19: colectivo
        if evaluation.get('collective_alignment', 0) > 0.2:
            parts.append("alineado-colectivo")

        return "; ".join(parts) if parts else "decisión estándar"

    def process_outcome(self, actual_change: np.ndarray,
                        reward: float, surprise: float,
                        social_feedback: Dict[str, float]) -> ActionOutcome:
        """
        Procesa el resultado de la acción y actualiza AGI-16, AGI-17, AGI-18.

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

        # Actualizar módulos con feedback (AGI-16, AGI-17, AGI-18)
        self._update_from_outcome(outcome)

        return outcome

    def _update_from_outcome(self, outcome: ActionOutcome):
        """
        Actualiza módulos cognitivos AGI-16, AGI-17, AGI-18 con el resultado.
        """
        # AGI-16: Registrar observación (contexto, política, utilidad)
        self.meta_rules.record_observation(
            self.current_context,
            self.current_policy,
            outcome.reward_signal
        )

        # AGI-17: Registrar recompensa para robustez
        if self.robustness_system is not None:
            # Construir policy vector simplificado para AGI-17
            policy_vec = np.zeros(7)
            policy_vec[self.current_policy] = 1.0
            self.robustness_system.record_reward(
                self.agent_name,
                outcome.reward_signal,
                policy_vec
            )

        # AGI-18: Registrar activaciones para reconfiguración
        activations = {
            'self_model': 1.0 - outcome.surprise,  # Baja sorpresa = self-model funcionó
            'tom': np.mean(list(outcome.social_feedback.values())) if outcome.social_feedback else 0.5,
            'ethics': 0.8,  # Por ahora constante
            'planning': max(0, outcome.goal_progress + 0.5),
            'curiosity': outcome.surprise,  # Sorpresa = novedad
            'norms': 1 - abs(outcome.reward_signal - 0.5),
            'meta_rules': 0.5 + outcome.reward_signal * 0.3,  # Correlacionado con reward
            'robustness': 0.5,  # Contribución base
            'collective_intent': np.mean(list(outcome.social_feedback.values())) if outcome.social_feedback else 0.5
        }
        self.reconfig_system.record_activations(activations, outcome.reward_signal)

    def get_statistics(self) -> Dict:
        """Estadísticas de la capa cognitiva incluyendo AGI-16 a AGI-19."""
        # AGI-16 stats
        meta_rule_stats = self.meta_rules.get_statistics()

        # AGI-18 stats
        reconfig_stats = self.reconfig_system.get_statistics()

        return {
            'agent': self.agent_name,
            't': self.t,
            'n_decisions': len(self.decision_history),
            'n_outcomes': len(self.outcome_history),
            'mean_confidence': np.mean([d.confidence for d in self.decision_history[-20:]]) if self.decision_history else 0,
            'mean_reward': np.mean(self.reward_history[-20:]) if self.reward_history else 0,
            'mean_surprise': np.mean(self.surprise_history[-20:]) if self.surprise_history else 0,
            'module_weights': self.module_weights.copy(),
            'has_goal': self.current_goal is not None,
            # AGI-16: Meta-Rules
            'current_policy': self.POLICY_NAMES[self.current_policy] if self.current_policy < len(self.POLICY_NAMES) else 'unknown',
            'n_meta_rules': meta_rule_stats.get('n_valid_rules', 0),
            'meta_utility_gain': meta_rule_stats.get('meta_utility_gain', 0),
            # AGI-18: Reconfiguration
            'n_reconfigurations': reconfig_stats.get('n_reconfigurations', 0),
            'config_entropy': reconfig_stats.get('configuration_entropy', 0),
            'most_weighted_module': reconfig_stats.get('most_weighted', ''),
            # AGI-17 y AGI-19 se obtienen del sistema compartido
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
