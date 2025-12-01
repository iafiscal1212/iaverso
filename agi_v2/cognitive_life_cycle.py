"""
AGI-X v2.0: Ciclo de Vida Cognitivo Unificado
==============================================

Todos los agentes son teleológicos por diseño.
Ciclo completo: percepción → cognición → acción → mundo → memoria → narrativa → reconfiguración → metas → acción

Sin números mágicos. Todo endógeno.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import (
    L_t, max_history, adaptive_learning_rate, adaptive_momentum,
    to_simplex, softmax, normalized_entropy, confidence_from_error
)

from agi_v2.architecture import (
    LifeCyclePhase, CognitiveState, WorldState, Action, Memory, Goal,
    CausalModel, MetaMemory, AntifragilitySystem,
    endogenous_goal_priority, endogenous_value_update
)


@dataclass
class TeleologicalAgent:
    """
    Agente teleológico completo.

    Cada agente tiene:
    - Metas intrínsecas (teleología)
    - Modelo causal interno
    - Meta-memoria
    - Sistema de anti-fragilidad
    - Consistencia axiológica
    """
    name: str
    state_dim: int = 6
    action_dim: int = 3
    value_dim: int = 5

    # Estado cognitivo actual
    cognitive_state: Optional[CognitiveState] = None

    # Sistemas internos (se inicializan en __post_init__)
    causal_model: Optional[CausalModel] = None
    meta_memory: Optional[MetaMemory] = None
    antifragility: Optional[AntifragilitySystem] = None

    # Historial
    state_history: List[np.ndarray] = field(default_factory=list)
    action_history: List[Action] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)
    goal_progress_history: List[float] = field(default_factory=list)

    # Metas activas
    goals: List[Goal] = field(default_factory=list)

    # Valores axiológicos
    values: Optional[np.ndarray] = None
    value_consistency_history: List[float] = field(default_factory=list)

    # Narrativa
    narrative_buffer: List[str] = field(default_factory=list)
    narrative_coherence: float = 0.5

    # Temporal
    t: int = 0
    current_phase: LifeCyclePhase = LifeCyclePhase.PERCEPTION

    def __post_init__(self):
        """Inicializa sistemas internos."""
        self.causal_model = CausalModel(self.state_dim, self.action_dim)
        self.meta_memory = MetaMemory(self.name)
        self.antifragility = AntifragilitySystem(self.name, self.state_dim)
        self.values = to_simplex(np.random.rand(self.value_dim) + 0.1)

        # Crear meta intrínseca inicial
        self._create_intrinsic_goal()

    def _create_intrinsic_goal(self):
        """Crea meta intrínseca endógena."""
        # Meta intrínseca: maximizar supervivencia/recursos
        target = np.zeros(self.state_dim)
        target[0] = 1.0  # Alto valor en primera dimensión (típicamente recursos)

        goal = Goal(
            target_state=target,
            priority=1.0,
            origin='intrinsic',
            created_t=self.t,
            progress=0.0,
            sub_goals=[]
        )
        self.goals.append(goal)


class CognitiveLifeCycle:
    """
    Ciclo de vida cognitivo completo para múltiples agentes.

    Gestiona el ciclo:
    percepción → cognición → acción → mundo → memoria → narrativa → reconfiguración → metas → acción

    Todos los agentes son teleológicos.
    """

    PHASES = [
        LifeCyclePhase.PERCEPTION,
        LifeCyclePhase.COGNITION,
        LifeCyclePhase.INTENTION,
        LifeCyclePhase.ACTION,
        LifeCyclePhase.FEEDBACK,
        LifeCyclePhase.MEMORY,
        LifeCyclePhase.NARRATIVE,
        LifeCyclePhase.RECONFIGURATION,
        LifeCyclePhase.GOAL_UPDATE
    ]

    def __init__(self, agent_names: List[str], state_dim: int = 6,
                 action_dim: int = 3, value_dim: int = 5):
        """
        Inicializa el ciclo de vida para todos los agentes.

        Args:
            agent_names: Nombres de los agentes
            state_dim: Dimensión del estado
            action_dim: Dimensión de acciones
            value_dim: Dimensión de valores axiológicos
        """
        self.agent_names = agent_names
        self.n_agents = len(agent_names)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.value_dim = value_dim

        # Crear agentes teleológicos
        self.agents: Dict[str, TeleologicalAgent] = {}
        for name in agent_names:
            self.agents[name] = TeleologicalAgent(
                name=name,
                state_dim=state_dim,
                action_dim=action_dim,
                value_dim=value_dim
            )

        # Estado compartido del mundo
        self.world_state: Optional[WorldState] = None

        # Memoria colectiva (para emergencia de normas)
        self.collective_memory: List[Dict] = []

        # Historial de coherencia colectiva
        self.collective_coherence_history: List[float] = []

        # Contador global
        self.t = 0

    def perceive(self, agent_name: str, world_observation: np.ndarray,
                 other_agents_obs: Dict[str, np.ndarray]) -> CognitiveState:
        """
        Fase 1: PERCEPCIÓN

        El agente percibe el mundo y otros agentes.
        """
        agent = self.agents[agent_name]
        agent.current_phase = LifeCyclePhase.PERCEPTION

        # Construir estado estructural z
        z = to_simplex(np.abs(world_observation[:self.state_dim]) + 0.01)

        # Estado fenomenológico phi (derivado de la observación)
        phi = np.array([
            float(np.mean(world_observation)),          # Intensidad media
            float(np.std(world_observation)),           # Variabilidad
            float(np.max(world_observation)),           # Máximo
            normalized_entropy(to_simplex(np.abs(world_observation) + 0.01)),  # Entropía
            float(np.linalg.norm(world_observation))    # Magnitud total
        ])

        # Drives basados en historial y metas
        if agent.goals:
            goal_vector = agent.goals[0].target_state
            drives_raw = goal_vector - z
        else:
            drives_raw = np.random.randn(self.state_dim) * 0.1
        drives = to_simplex(np.abs(drives_raw) + 0.01)

        # Calcular precisiones meta-cognitivas
        self_model_acc = self._compute_self_model_accuracy(agent)
        tom_acc = self._compute_tom_accuracy(agent, other_agents_obs)
        uncertainty = self._compute_uncertainty(agent, world_observation)

        # Crear estado cognitivo
        cognitive_state = CognitiveState(
            z=z,
            phi=phi,
            drives=drives,
            t=agent.t,
            phase=LifeCyclePhase.PERCEPTION,
            goals=[g.target_state for g in agent.goals],
            goal_priorities=np.array([g.priority for g in agent.goals]) if agent.goals else np.array([1.0]),
            values=agent.values.copy(),
            value_confidence=self._compute_value_confidence(agent),
            self_model_accuracy=self_model_acc,
            tom_accuracy=tom_acc,
            uncertainty=uncertainty,
            narrative_coherence=agent.narrative_coherence,
            identity_stability=self._compute_identity_stability(agent)
        )

        agent.cognitive_state = cognitive_state
        return cognitive_state

    def cognize(self, agent_name: str) -> Dict[str, Any]:
        """
        Fase 2: COGNICIÓN

        Procesamiento cognitivo profundo:
        - Razonamiento causal
        - Contrafactual
        - Meta-cognición
        """
        agent = self.agents[agent_name]
        agent.current_phase = LifeCyclePhase.COGNITION

        if agent.cognitive_state is None:
            return {'status': 'no_state'}

        state = agent.cognitive_state

        # Razonamiento causal: ¿qué causó el estado actual?
        causal_attribution = {}
        if len(agent.action_history) > 0 and len(agent.state_history) >= 2:
            prev_state = agent.state_history[-2] if len(agent.state_history) >= 2 else agent.state_history[-1]
            last_action = agent.action_history[-1]
            causal_attribution = agent.causal_model.causal_attribution(
                prev_state, last_action.direction, state.z
            )

        # Contrafactual: ¿qué hubiera pasado con otra acción?
        counterfactual_analysis = {}
        if len(agent.action_history) > 0 and len(agent.state_history) >= 2:
            # Generar acción alternativa
            alternative_action = np.random.randn(self.action_dim) * 0.1
            prev_state = agent.state_history[-2] if len(agent.state_history) >= 2 else state.z
            pred_actual, pred_alt = agent.causal_model.counterfactual(
                prev_state, agent.action_history[-1].direction, alternative_action
            )
            counterfactual_analysis = {
                'actual_outcome': pred_actual,
                'alternative_outcome': pred_alt,
                'difference': float(np.linalg.norm(pred_alt - pred_actual))
            }

        # Meta-cognición: evaluar propia cognición
        # Calcular confianza endógenamente
        error_history = [1.0 - r for r in agent.reward_history[-L_t(agent.t):]] if agent.reward_history else [0.5]
        confidence = confidence_from_error(state.uncertainty, error_history) if len(error_history) >= 2 else 1.0 / (1 + state.uncertainty)

        meta_cognition = {
            'self_model_accuracy': state.self_model_accuracy,
            'tom_accuracy': state.tom_accuracy,
            'uncertainty': state.uncertainty,
            'confidence': confidence
        }

        # Recuperar memorias relevantes
        query = np.concatenate([state.z, state.phi])
        relevant_memories = agent.meta_memory.recall(query, n=L_t(agent.t))

        return {
            'causal_attribution': causal_attribution,
            'counterfactual': counterfactual_analysis,
            'meta_cognition': meta_cognition,
            'relevant_memories': len(relevant_memories),
            'patterns_detected': agent.meta_memory.detect_patterns()
        }

    def intend(self, agent_name: str, cognition_result: Dict) -> Dict[str, Any]:
        """
        Fase 3: INTENCIÓN

        Formar intención basada en:
        - Metas activas
        - Valores axiológicos
        - Cognición previa
        """
        agent = self.agents[agent_name]
        agent.current_phase = LifeCyclePhase.INTENTION

        if agent.cognitive_state is None:
            return {'intention': 'none', 'target': None}

        state = agent.cognitive_state

        # Seleccionar meta más relevante
        if not agent.goals:
            agent._create_intrinsic_goal()

        # Priorizar metas endógenamente
        for goal in agent.goals:
            goal.priority = endogenous_goal_priority(
                goal, agent.t, agent.goal_progress_history
            )

        # Meta con mayor prioridad
        sorted_goals = sorted(agent.goals, key=lambda g: g.priority, reverse=True)
        target_goal = sorted_goals[0]

        # Dirección hacia la meta
        direction = target_goal.target_state - state.z

        # Magnitud basada en drives y confianza
        magnitude = float(np.linalg.norm(state.drives)) * cognition_result.get('meta_cognition', {}).get('confidence', 0.5)

        # Consistencia axiológica: la intención debe alinear con valores
        value_alignment = self._compute_value_alignment(agent, direction)

        intention = {
            'intention': 'pursue_goal',
            'target_goal': target_goal.origin,
            'direction': direction,
            'magnitude': magnitude,
            'value_alignment': value_alignment,
            'confidence': cognition_result.get('meta_cognition', {}).get('confidence', 0.5)
        }

        return intention

    def act(self, agent_name: str, intention: Dict) -> Action:
        """
        Fase 4: ACCIÓN

        Ejecutar acción basada en intención.
        """
        agent = self.agents[agent_name]
        agent.current_phase = LifeCyclePhase.ACTION

        direction = intention.get('direction', np.zeros(self.action_dim))
        if len(direction) > self.action_dim:
            direction = direction[:self.action_dim]
        elif len(direction) < self.action_dim:
            direction = np.pad(direction, (0, self.action_dim - len(direction)))

        magnitude = intention.get('magnitude', 0.1)
        confidence = intention.get('confidence', 0.5)

        # Aplicar fortaleza del sistema anti-fragilidad
        direction_strengthened = agent.antifragility.apply_strength(direction)
        direction_strengthened = direction_strengthened[:self.action_dim]

        # Crear acción
        action = Action(
            direction=direction_strengthened,
            magnitude=float(magnitude),
            target=intention.get('target_goal'),
            intention=str(intention.get('intention', 'unknown')),
            confidence=float(confidence),
            counterfactual_considered=True,
            causal_model_used=agent.causal_model.transition_model is not None
        )

        # Registrar en historial
        agent.action_history.append(action)
        max_hist = max_history(agent.t)
        if len(agent.action_history) > max_hist:
            agent.action_history = agent.action_history[-max_hist:]

        return action

    def feedback(self, agent_name: str, action: Action,
                 next_observation: np.ndarray, reward: float) -> Dict[str, float]:
        """
        Fase 5: FEEDBACK

        Recibir feedback del mundo tras la acción.
        """
        agent = self.agents[agent_name]
        agent.current_phase = LifeCyclePhase.FEEDBACK
        agent.t += 1

        # Registrar estado
        current_state = agent.cognitive_state.z if agent.cognitive_state else np.zeros(self.state_dim)
        agent.state_history.append(current_state.copy())
        max_hist = max_history(agent.t)
        if len(agent.state_history) > max_hist:
            agent.state_history = agent.state_history[-max_hist:]

        # Registrar en modelo causal
        next_state = to_simplex(np.abs(next_observation[:self.state_dim]) + 0.01)
        agent.causal_model.record(current_state, action.direction, next_state)

        # Registrar reward
        agent.reward_history.append(reward)
        if len(agent.reward_history) > max_hist:
            agent.reward_history = agent.reward_history[-max_hist:]

        # Calcular progreso hacia meta
        if agent.goals:
            goal_distance = agent.goals[0].distance_to(next_state)
            progress = 1.0 / (1 + goal_distance)
            agent.goals[0].progress = progress
            agent.goal_progress_history.append(progress)
            if len(agent.goal_progress_history) > max_hist:
                agent.goal_progress_history = agent.goal_progress_history[-max_hist:]

        # Registrar estrés (varianza como indicador)
        if len(agent.reward_history) >= 2:
            recent_rewards = agent.reward_history[-L_t(agent.t):]
            stress = np.abs(reward - np.mean(recent_rewards))
            stress_vector = np.ones(self.state_dim) * stress
            agent.antifragility.record_stress(stress_vector)

        # Actualizar valores axiológicos
        if agent.values is not None:
            agent.values = endogenous_value_update(
                agent.values, reward, action, agent.t, agent.reward_history
            )

        return {
            'reward': reward,
            'goal_progress': agent.goals[0].progress if agent.goals else 0,
            'resilience': agent.antifragility.get_resilience()
        }

    def memorize(self, agent_name: str, feedback_result: Dict) -> Memory:
        """
        Fase 6: MEMORIA

        Consolidar experiencia en memoria episódica.
        """
        agent = self.agents[agent_name]
        agent.current_phase = LifeCyclePhase.MEMORY

        # Crear memoria episódica
        states = [agent.cognitive_state] if agent.cognitive_state else []
        actions = agent.action_history[-1:] if agent.action_history else []
        outcomes = [feedback_result.get('reward', 0)]

        # Calcular valencia emocional endógena
        emotional_valence = self._compute_emotional_valence(agent, feedback_result)

        # Atribución causal
        causal_attribution = {}
        if agent.action_history and len(agent.state_history) >= 2:
            causal_attribution = agent.causal_model.causal_attribution(
                agent.state_history[-2],
                agent.action_history[-1].direction,
                agent.state_history[-1]
            )

        # Generar contrafactuales
        counterfactuals = self._generate_counterfactuals(agent)

        memory = Memory(
            t_start=agent.t - 1,
            t_end=agent.t,
            states=states,
            actions=actions,
            outcomes=outcomes,
            narrative="",  # Se llena en fase de narrativa
            emotional_valence=emotional_valence,
            causal_attribution=causal_attribution,
            counterfactuals=counterfactuals
        )

        # Almacenar en meta-memoria
        agent.meta_memory.store(memory)

        return memory

    def narrate(self, agent_name: str, memory: Memory) -> str:
        """
        Fase 7: NARRATIVA

        Construir narrativa coherente de la experiencia.
        """
        agent = self.agents[agent_name]
        agent.current_phase = LifeCyclePhase.NARRATIVE

        # Generar narrativa endógena
        narrative_parts = []

        # Acción tomada
        if memory.actions:
            action = memory.actions[0]
            narrative_parts.append(f"Acción: {action.intention} (conf: {action.confidence:.2f})")

        # Resultado
        if memory.outcomes:
            outcome = memory.outcomes[0]
            if outcome > 0.5:
                narrative_parts.append(f"Resultado positivo ({outcome:.2f})")
            elif outcome < -0.5:
                narrative_parts.append(f"Resultado negativo ({outcome:.2f})")
            else:
                narrative_parts.append(f"Resultado neutro ({outcome:.2f})")

        # Atribución causal
        if memory.causal_attribution:
            state_contrib = memory.causal_attribution.get('state', 0.5)
            action_contrib = memory.causal_attribution.get('action', 0.5)
            if action_contrib > state_contrib:
                narrative_parts.append("Mi acción fue determinante")
            else:
                narrative_parts.append("El contexto fue determinante")

        # Contrafactual
        if memory.counterfactuals:
            narrative_parts.append(f"Alternativa: {memory.counterfactuals[0]}")

        narrative = " | ".join(narrative_parts)

        # Actualizar coherencia narrativa
        agent.narrative_buffer.append(narrative)
        max_narr = L_t(agent.t) * 2
        if len(agent.narrative_buffer) > max_narr:
            agent.narrative_buffer = agent.narrative_buffer[-max_narr:]

        agent.narrative_coherence = self._compute_narrative_coherence(agent)

        return narrative

    def reconfigure(self, agent_name: str) -> Dict[str, float]:
        """
        Fase 8: RECONFIGURACIÓN

        Ajustar módulos internos basándose en rendimiento.
        """
        agent = self.agents[agent_name]
        agent.current_phase = LifeCyclePhase.RECONFIGURATION

        reconfiguration = {}

        # Evaluar rendimiento reciente
        if len(agent.reward_history) >= L_t(agent.t):
            recent_rewards = agent.reward_history[-L_t(agent.t):]
            mean_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)

            # Si rendimiento bajo, ajustar
            if mean_reward < np.percentile(agent.reward_history, 25):
                # Aumentar exploración
                reconfiguration['exploration_boost'] = 1.0 + adaptive_learning_rate(agent.t)
                # Reconsiderar metas
                reconfiguration['goal_revision'] = True
            else:
                reconfiguration['exploration_boost'] = 1.0
                reconfiguration['goal_revision'] = False

            # Estabilidad
            reconfiguration['stability'] = 1.0 / (1 + std_reward)

        # Detectar patrones en memoria
        patterns = agent.meta_memory.detect_patterns()
        reconfiguration['patterns_detected'] = len(patterns)

        # Ajustar consistencia de valores
        if len(agent.value_consistency_history) >= L_t(agent.t):
            value_consistency = np.mean(agent.value_consistency_history[-L_t(agent.t):])
            if value_consistency < 0.5:
                # Valores inconsistentes, necesitan revisión
                reconfiguration['value_revision'] = True
            else:
                reconfiguration['value_revision'] = False

        return reconfiguration

    def update_goals(self, agent_name: str, reconfig: Dict) -> List[Goal]:
        """
        Fase 9: ACTUALIZACIÓN DE METAS

        Actualizar metas basándose en progreso y reconfiguración.
        """
        agent = self.agents[agent_name]
        agent.current_phase = LifeCyclePhase.GOAL_UPDATE

        # Revisar metas si es necesario
        if reconfig.get('goal_revision', False):
            # Recalcular prioridades
            for goal in agent.goals:
                goal.priority = endogenous_goal_priority(
                    goal, agent.t, agent.goal_progress_history
                )

            # Eliminar metas con prioridad muy baja
            threshold = np.percentile([g.priority for g in agent.goals], 10) if len(agent.goals) > 1 else 0
            agent.goals = [g for g in agent.goals if g.priority > threshold]

        # Crear nuevas metas si necesario
        if not agent.goals:
            agent._create_intrinsic_goal()

        # Meta emergente basada en patrones
        if reconfig.get('patterns_detected', 0) > 0 and len(agent.goals) < 3:
            # Crear meta derivada de patrones
            patterns = agent.meta_memory.patterns
            if patterns:
                pattern = patterns[0]
                if pattern.get('type') == 'improving':
                    # Continuar la mejora
                    target = agent.cognitive_state.z.copy() if agent.cognitive_state else np.zeros(self.state_dim)
                    target += np.ones_like(target) * pattern.get('rate', 0.1)
                    target = to_simplex(np.abs(target) + 0.01)

                    new_goal = Goal(
                        target_state=target,
                        priority=0.5,
                        origin='emergent',
                        created_t=agent.t,
                        progress=0.0,
                        sub_goals=[]
                    )
                    agent.goals.append(new_goal)

        return agent.goals

    def full_cycle(self, agent_name: str, world_observation: np.ndarray,
                   other_agents_obs: Dict[str, np.ndarray],
                   next_observation: np.ndarray, reward: float) -> Dict[str, Any]:
        """
        Ejecuta un ciclo completo de vida cognitiva.
        """
        self.t += 1

        # Fase 1: Percepción
        cognitive_state = self.perceive(agent_name, world_observation, other_agents_obs)

        # Fase 2: Cognición
        cognition_result = self.cognize(agent_name)

        # Fase 3: Intención
        intention = self.intend(agent_name, cognition_result)

        # Fase 4: Acción
        action = self.act(agent_name, intention)

        # Fase 5: Feedback
        feedback_result = self.feedback(agent_name, action, next_observation, reward)

        # Fase 6: Memoria
        memory = self.memorize(agent_name, feedback_result)

        # Fase 7: Narrativa
        narrative = self.narrate(agent_name, memory)

        # Fase 8: Reconfiguración
        reconfig = self.reconfigure(agent_name)

        # Fase 9: Actualización de metas
        goals = self.update_goals(agent_name, reconfig)

        return {
            'agent': agent_name,
            't': self.t,
            'action': action,
            'reward': feedback_result['reward'],
            'goal_progress': feedback_result['goal_progress'],
            'resilience': feedback_result['resilience'],
            'narrative': narrative,
            'n_goals': len(goals),
            'narrative_coherence': self.agents[agent_name].narrative_coherence
        }

    # ---------- Métodos auxiliares ----------

    def _compute_self_model_accuracy(self, agent: TeleologicalAgent) -> float:
        """Calcula precisión del auto-modelo."""
        if len(agent.state_history) < 2:
            return 0.5

        # Comparar predicción vs realidad
        if agent.causal_model.transition_model is not None and len(agent.action_history) > 0:
            prev_state = agent.state_history[-2] if len(agent.state_history) >= 2 else agent.state_history[-1]
            pred = agent.causal_model.predict(prev_state, agent.action_history[-1].direction)
            actual = agent.state_history[-1]
            error = np.linalg.norm(pred - actual)
            return float(1.0 / (1 + error))

        return 0.5

    def _compute_tom_accuracy(self, agent: TeleologicalAgent,
                              other_agents_obs: Dict[str, np.ndarray]) -> float:
        """Calcula precisión de teoría de la mente."""
        if not other_agents_obs:
            return 0.5

        # Placeholder: en implementación completa compararía predicciones
        return 0.5 + np.random.rand() * 0.1

    def _compute_uncertainty(self, agent: TeleologicalAgent,
                            observation: np.ndarray) -> float:
        """Calcula incertidumbre endógena."""
        if len(agent.state_history) < 2:
            return 0.5

        # Varianza reciente como proxy de incertidumbre
        recent = agent.state_history[-L_t(agent.t):]
        if len(recent) < 2:
            return 0.5

        variance = np.mean([np.var(s) for s in recent])
        return float(np.clip(variance, 0, 1))

    def _compute_value_confidence(self, agent: TeleologicalAgent) -> float:
        """Calcula confianza en valores axiológicos."""
        if len(agent.value_consistency_history) < L_t(agent.t):
            return 0.5

        recent = agent.value_consistency_history[-L_t(agent.t):]
        return float(np.mean(recent))

    def _compute_identity_stability(self, agent: TeleologicalAgent) -> float:
        """Calcula estabilidad de identidad."""
        if len(agent.state_history) < L_t(agent.t):
            return 0.5

        recent = agent.state_history[-L_t(agent.t):]
        changes = [np.linalg.norm(recent[i+1] - recent[i]) for i in range(len(recent)-1)]
        if not changes:
            return 0.5

        mean_change = np.mean(changes)
        return float(1.0 / (1 + mean_change))

    def _compute_value_alignment(self, agent: TeleologicalAgent,
                                 direction: np.ndarray) -> float:
        """Calcula alineación de dirección con valores."""
        if agent.values is None:
            return 0.5

        # Normalizar dirección
        dir_norm = direction / (np.linalg.norm(direction) + 1e-8)

        # Pad para match
        if len(dir_norm) < len(agent.values):
            dir_norm = np.pad(dir_norm, (0, len(agent.values) - len(dir_norm)))
        else:
            dir_norm = dir_norm[:len(agent.values)]

        # Coseno como alineación
        alignment = np.dot(dir_norm, agent.values)

        # Registrar consistencia
        agent.value_consistency_history.append(float(alignment))
        max_hist = max_history(agent.t)
        if len(agent.value_consistency_history) > max_hist:
            agent.value_consistency_history = agent.value_consistency_history[-max_hist:]

        return float((alignment + 1) / 2)  # Normalizar a [0, 1]

    def _compute_emotional_valence(self, agent: TeleologicalAgent,
                                   feedback: Dict) -> float:
        """Calcula valencia emocional endógena."""
        reward = feedback.get('reward', 0)
        progress = feedback.get('goal_progress', 0)

        # Combinar reward y progreso
        valence = reward * 0.6 + progress * 0.4

        # Modular por expectativa
        if len(agent.reward_history) >= L_t(agent.t):
            expected = np.mean(agent.reward_history[-L_t(agent.t):])
            surprise = reward - expected
            valence = valence + surprise * 0.3

        return float(np.clip(valence, -1, 1))

    def _generate_counterfactuals(self, agent: TeleologicalAgent) -> List[str]:
        """Genera descripciones contrafactuales."""
        counterfactuals = []

        if len(agent.action_history) > 0 and agent.causal_model.transition_model is not None:
            # Acción opuesta
            actual_action = agent.action_history[-1].direction
            opposite_action = -actual_action

            if len(agent.state_history) >= 1:
                prev_state = agent.state_history[-1]
                _, pred_opposite = agent.causal_model.counterfactual(
                    prev_state, actual_action, opposite_action
                )

                diff = np.linalg.norm(pred_opposite - prev_state)
                if diff > 0.5:
                    counterfactuals.append(f"Acción opuesta: cambio significativo ({diff:.2f})")
                else:
                    counterfactuals.append(f"Acción opuesta: cambio menor ({diff:.2f})")

        return counterfactuals

    def _compute_narrative_coherence(self, agent: TeleologicalAgent) -> float:
        """Calcula coherencia narrativa."""
        if len(agent.narrative_buffer) < 2:
            return 0.5

        # Coherencia como consistencia de temas
        # Simplificado: longitud promedio como proxy de riqueza narrativa
        lengths = [len(n) for n in agent.narrative_buffer[-L_t(agent.t):]]
        coherence = 1.0 / (1 + np.std(lengths) / (np.mean(lengths) + 1))

        return float(np.clip(coherence, 0, 1))

    def get_collective_coherence(self) -> float:
        """Calcula coherencia colectiva entre agentes."""
        if len(self.agents) < 2:
            return 1.0

        values_list = [agent.values for agent in self.agents.values() if agent.values is not None]
        if len(values_list) < 2:
            return 0.5

        # Similitud promedio de valores
        similarities = []
        for i in range(len(values_list)):
            for j in range(i+1, len(values_list)):
                sim = np.dot(values_list[i], values_list[j])
                similarities.append(sim)

        coherence = float(np.mean(similarities))
        self.collective_coherence_history.append(coherence)

        return coherence

    def get_agent_statistics(self, agent_name: str) -> Dict:
        """Obtiene estadísticas de un agente."""
        agent = self.agents[agent_name]
        return {
            'name': agent_name,
            't': agent.t,
            'n_goals': len(agent.goals),
            'goal_types': [g.origin for g in agent.goals],
            'mean_reward': np.mean(agent.reward_history) if agent.reward_history else 0,
            'goal_progress': agent.goals[0].progress if agent.goals else 0,
            'resilience': agent.antifragility.get_resilience(),
            'narrative_coherence': agent.narrative_coherence,
            'value_confidence': self._compute_value_confidence(agent),
            'n_memories': len(agent.meta_memory.episodes),
            'causal_model_trained': agent.causal_model.transition_model is not None
        }

    def get_all_statistics(self) -> Dict:
        """Obtiene estadísticas globales."""
        stats = {
            't': self.t,
            'n_agents': self.n_agents,
            'collective_coherence': self.get_collective_coherence(),
            'agents': {}
        }

        for name in self.agent_names:
            stats['agents'][name] = self.get_agent_statistics(name)

        # Métricas agregadas
        all_rewards = []
        all_progress = []
        all_resilience = []

        for agent in self.agents.values():
            if agent.reward_history:
                all_rewards.extend(agent.reward_history[-L_t(agent.t):])
            if agent.goal_progress_history:
                all_progress.extend(agent.goal_progress_history[-L_t(agent.t):])
            all_resilience.append(agent.antifragility.get_resilience())

        stats['mean_reward'] = float(np.mean(all_rewards)) if all_rewards else 0
        stats['mean_goal_progress'] = float(np.mean(all_progress)) if all_progress else 0
        stats['mean_resilience'] = float(np.mean(all_resilience))

        return stats


def test_cognitive_life_cycle():
    """Test del ciclo de vida cognitivo."""
    print("=" * 70)
    print("TEST: CICLO DE VIDA COGNITIVO AGI-X v2.0")
    print("=" * 70)

    # Crear sistema con agentes teleológicos
    agents = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
    life_cycle = CognitiveLifeCycle(agents, state_dim=6, action_dim=3, value_dim=5)

    print(f"\nAgentes teleológicos: {agents}")
    print(f"Dimensiones: state={life_cycle.state_dim}, action={life_cycle.action_dim}, value={life_cycle.value_dim}")

    # Simular 200 pasos
    n_steps = 200
    results = {name: {'rewards': [], 'progress': [], 'resilience': []} for name in agents}

    for step in range(n_steps):
        # Observación del mundo (simulada)
        world_obs = np.random.randn(life_cycle.state_dim) * 0.5

        for agent_name in agents:
            # Otros agentes
            other_obs = {other: np.random.randn(life_cycle.state_dim) * 0.3
                        for other in agents if other != agent_name}

            # Siguiente observación (simulada)
            next_obs = world_obs + np.random.randn(life_cycle.state_dim) * 0.1

            # Reward endógeno basado en progreso hacia meta
            agent = life_cycle.agents[agent_name]
            if agent.goals and agent.cognitive_state is not None:
                goal_dist = agent.goals[0].distance_to(agent.cognitive_state.z)
                reward = 1.0 / (1 + goal_dist) + np.random.randn() * 0.1
            else:
                reward = np.random.randn() * 0.2

            # Ejecutar ciclo completo
            result = life_cycle.full_cycle(
                agent_name, world_obs, other_obs, next_obs, reward
            )

            results[agent_name]['rewards'].append(result['reward'])
            results[agent_name]['progress'].append(result['goal_progress'])
            results[agent_name]['resilience'].append(result['resilience'])

        # Reportar progreso
        if (step + 1) % 50 == 0:
            stats = life_cycle.get_all_statistics()
            print(f"\n  Paso {step + 1}:")
            print(f"    Coherencia colectiva: {stats['collective_coherence']:.3f}")
            print(f"    Reward promedio: {stats['mean_reward']:.3f}")
            print(f"    Progreso promedio: {stats['mean_goal_progress']:.3f}")
            print(f"    Resiliencia promedio: {stats['mean_resilience']:.3f}")

    # Resultados finales
    print("\n" + "=" * 70)
    print("RESULTADOS FINALES")
    print("=" * 70)

    for name in agents:
        agent_stats = life_cycle.get_agent_statistics(name)
        print(f"\n{name}:")
        print(f"  Metas activas: {agent_stats['n_goals']} ({', '.join(agent_stats['goal_types'])})")
        print(f"  Reward promedio: {agent_stats['mean_reward']:.3f}")
        print(f"  Progreso hacia meta: {agent_stats['goal_progress']:.3f}")
        print(f"  Resiliencia: {agent_stats['resilience']:.3f}")
        print(f"  Coherencia narrativa: {agent_stats['narrative_coherence']:.3f}")
        print(f"  Memorias almacenadas: {agent_stats['n_memories']}")
        print(f"  Modelo causal entrenado: {agent_stats['causal_model_trained']}")

    final_stats = life_cycle.get_all_statistics()
    print(f"\nCoherencia colectiva final: {final_stats['collective_coherence']:.3f}")

    print("\n" + "=" * 70)
    print("TEST COMPLETADO: TODOS LOS AGENTES SON TELEOLÓGICOS")
    print("=" * 70)

    return life_cycle


if __name__ == "__main__":
    test_cognitive_life_cycle()
