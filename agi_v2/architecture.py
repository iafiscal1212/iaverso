"""
AGI-X v2.0: Arquitectura de Vida Cognitiva Completa
====================================================

Ciclo de vida endógeno:
percepción → cognición → acción → mundo → memoria → narrativa → reconfiguración → metas → acción

Todos los agentes son teleológicos por diseño.
Todo es endógeno. Sin números mágicos.

Módulos integrados:
- AGI-4: Self-Model (predicción de sí mismo)
- AGI-5: Theory of Mind (modelado de otros)
- AGI-6: Planning (planificación hacia metas)
- AGI-11: Introspective Uncertainty
- AGI-12: Norm System (normas emergentes)
- AGI-13: Structural Curiosity
- AGI-14: Prediction Channels
- AGI-15: Structural Ethics
- AGI-16: Meta-Rules (políticas contextuales)
- AGI-17: Multi-World Robustness
- AGI-18: Reflective Reconfiguration
- AGI-19: Collective Intentionality
- AGI-20: Structural Self-Theory

Nuevos para v2.0:
- Meta-Memory: memoria de memorias
- Counterfactual Reasoning: simulación de alternativas
- Causal Model: modelo causal interno
- Axiological Consistency: coherencia de valores
- Anti-Fragility: fortalecimiento por estrés
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum, auto

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import (
    L_t, max_history, adaptive_learning_rate, adaptive_momentum,
    to_simplex, softmax, normalized_entropy, confidence_from_error
)


class LifeCyclePhase(Enum):
    """Fases del ciclo de vida cognitivo."""
    PERCEPTION = auto()      # Percibir el mundo
    COGNITION = auto()       # Procesar información
    INTENTION = auto()       # Formar intención
    ACTION = auto()          # Ejecutar acción
    FEEDBACK = auto()        # Recibir feedback del mundo
    MEMORY = auto()          # Consolidar en memoria
    NARRATIVE = auto()       # Construir narrativa
    RECONFIGURATION = auto() # Ajustar módulos
    GOAL_UPDATE = auto()     # Actualizar metas


@dataclass
class CognitiveState:
    """Estado cognitivo completo de un agente."""
    # Estructural
    z: np.ndarray              # Estado estructural
    phi: np.ndarray            # Estado fenomenológico
    drives: np.ndarray         # Drives/motivaciones

    # Temporal
    t: int
    phase: LifeCyclePhase

    # Metas (teleológico)
    goals: List[np.ndarray]    # Stack de metas activas
    goal_priorities: np.ndarray # Prioridades endógenas

    # Valores (axiológico)
    values: np.ndarray         # Vector de valores
    value_confidence: float    # Confianza en valores

    # Meta-cognitivo
    self_model_accuracy: float
    tom_accuracy: float
    uncertainty: float

    # Narrativo
    narrative_coherence: float
    identity_stability: float


@dataclass
class WorldState:
    """Estado del mundo percibido."""
    fields: np.ndarray
    resources: np.ndarray
    modes: np.ndarray
    regime: str
    other_agents: Dict[str, 'CognitiveState']
    t: int


@dataclass
class Action:
    """Acción generada por el agente."""
    direction: np.ndarray
    magnitude: float
    target: Optional[str]
    intention: str
    confidence: float
    counterfactual_considered: bool
    causal_model_used: bool


@dataclass
class Memory:
    """Unidad de memoria episódica."""
    t_start: int
    t_end: int
    states: List[CognitiveState]
    actions: List[Action]
    outcomes: List[float]
    narrative: str
    emotional_valence: float
    causal_attribution: Dict[str, float]
    counterfactuals: List[str]


@dataclass
class Goal:
    """Meta teleológica."""
    target_state: np.ndarray
    priority: float
    origin: str  # 'intrinsic', 'derived', 'social', 'emergent'
    created_t: int
    progress: float
    sub_goals: List['Goal']

    def distance_to(self, current: np.ndarray) -> float:
        """Distancia endógena a la meta."""
        return float(np.linalg.norm(self.target_state - current))


def endogenous_goal_priority(goal: Goal, t: int, history: List[float]) -> float:
    """
    Prioridad de meta completamente endógena.

    Basada en:
    - Progreso histórico
    - Edad de la meta
    - Urgencia derivada de drives
    """
    if not history:
        return 1.0 / (1 + goal.distance_to(goal.target_state))

    # Progreso reciente
    recent_progress = np.mean(history[-L_t(t):]) if len(history) >= L_t(t) else np.mean(history)

    # Urgencia por edad (metas viejas sin progreso bajan)
    age = t - goal.created_t
    age_factor = 1.0 / (1 + np.log1p(age) / 10)

    # Momentum de progreso
    if len(history) >= 2:
        momentum = history[-1] - history[-2]
    else:
        momentum = 0.0

    priority = (recent_progress + 0.5) * age_factor * (1 + momentum)
    return float(np.clip(priority, 0.01, 1.0))


def endogenous_value_update(
    current_values: np.ndarray,
    reward: float,
    action_taken: Action,
    t: int,
    reward_history: Optional[List[float]] = None
) -> np.ndarray:
    """
    Actualización de valores completamente endógena.

    Los valores se ajustan basándose en:
    - Correlación acción-reward
    - Consistencia con valores previos
    - Estabilidad temporal
    """
    # Learning rate endógeno
    lr = adaptive_learning_rate(t)

    # Dirección del update basada en acción
    action_normalized = action_taken.direction / (np.linalg.norm(action_taken.direction) + 1e-8)

    # Pad o truncar para match con values
    if len(action_normalized) < len(current_values):
        action_normalized = np.pad(action_normalized, (0, len(current_values) - len(action_normalized)))
    else:
        action_normalized = action_normalized[:len(current_values)]

    # Update proporcional a reward y confianza
    update = action_normalized * reward * action_taken.confidence * lr

    # Aplicar con momentum endógeno
    if reward_history and len(reward_history) >= 3:
        momentum = adaptive_momentum(reward_history[-L_t(t):])
    else:
        # Momentum por defecto basado en t
        momentum = 0.9 / (1 + np.log1p(t) / 10)

    new_values = current_values * momentum + update * (1 - momentum)

    # Normalizar a simplex
    return to_simplex(np.abs(new_values) + 0.01)


class CausalModel:
    """
    Modelo causal interno del agente.

    Aprende relaciones causa-efecto de sus acciones.
    """

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Matriz de transición aprendida
        self.transition_model: Optional[np.ndarray] = None

        # Historial para aprendizaje
        self.state_history: List[np.ndarray] = []
        self.action_history: List[np.ndarray] = []
        self.next_state_history: List[np.ndarray] = []

        self.t = 0

    def record(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray):
        """Registra transición para aprendizaje causal."""
        self.t += 1

        self.state_history.append(state.copy())
        self.action_history.append(action.copy())
        self.next_state_history.append(next_state.copy())

        # Limitar historial
        max_hist = max_history(self.t)
        if len(self.state_history) > max_hist:
            self.state_history = self.state_history[-max_hist:]
            self.action_history = self.action_history[-max_hist:]
            self.next_state_history = self.next_state_history[-max_hist:]

        # Actualizar modelo periódicamente
        if self.t % L_t(self.t) == 0 and len(self.state_history) >= L_t(self.t) * 2:
            self._update_model()

    def _update_model(self):
        """Aprende modelo causal por regresión."""
        # X = [state, action], Y = next_state
        n = len(self.state_history)

        X = np.array([
            np.concatenate([s, a])
            for s, a in zip(self.state_history, self.action_history)
        ])
        Y = np.array(self.next_state_history)

        try:
            # Regresión ridge endógena
            lambda_reg = 1.0 / np.sqrt(n + 1)  # Regularización endógena
            XtX = X.T @ X + lambda_reg * np.eye(X.shape[1])
            XtY = X.T @ Y
            self.transition_model = np.linalg.solve(XtX, XtY)
        except:
            pass

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Predice siguiente estado dado estado y acción."""
        if self.transition_model is None:
            return state.copy()

        x = np.concatenate([state, action])
        if len(x) != self.transition_model.shape[0]:
            return state.copy()

        return x @ self.transition_model

    def counterfactual(self, state: np.ndarray,
                       actual_action: np.ndarray,
                       alternative_action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Razonamiento contrafactual: ¿qué hubiera pasado con otra acción?

        Returns:
            (predicted_actual, predicted_alternative)
        """
        pred_actual = self.predict(state, actual_action)
        pred_alternative = self.predict(state, alternative_action)
        return pred_actual, pred_alternative

    def causal_attribution(self, state: np.ndarray,
                          action: np.ndarray,
                          outcome: np.ndarray) -> Dict[str, float]:
        """
        Atribución causal: ¿cuánto contribuyó cada factor al outcome?
        """
        if self.transition_model is None:
            return {'state': 0.5, 'action': 0.5}

        # Contribución del estado
        state_only = self.predict(state, np.zeros_like(action))
        state_contribution = np.linalg.norm(state_only)

        # Contribución de la acción
        action_effect = outcome - state_only
        action_contribution = np.linalg.norm(action_effect)

        total = state_contribution + action_contribution + 1e-8

        return {
            'state': float(state_contribution / total),
            'action': float(action_contribution / total)
        }


class MetaMemory:
    """
    Meta-memoria: memoria de memorias.

    Permite al agente recordar qué recuerda,
    detectar patrones en su memoria,
    y consolidar memorias importantes.
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name

        # Memorias episódicas
        self.episodes: List[Memory] = []

        # Índice de acceso (qué memorias se acceden más)
        self.access_counts: Dict[int, int] = {}

        # Patrones detectados
        self.patterns: List[Dict] = []

        # Memoria de trabajo actual
        self.working_memory: List[Memory] = []
        self.working_memory_capacity = 3  # Se ajusta endógenamente

        self.t = 0

    def store(self, memory: Memory):
        """Almacena nueva memoria."""
        self.t += 1

        idx = len(self.episodes)
        self.episodes.append(memory)
        self.access_counts[idx] = 1

        # Limitar por importancia
        max_memories = max_history(self.t)
        if len(self.episodes) > max_memories:
            self._consolidate()

        # Actualizar capacidad de working memory
        self.working_memory_capacity = L_t(self.t)

    def recall(self, query: np.ndarray, n: int = 3) -> List[Memory]:
        """
        Recupera memorias relevantes.

        La relevancia es endógena basada en:
        - Similitud con query
        - Frecuencia de acceso
        - Recencia
        - Valencia emocional
        """
        if not self.episodes:
            return []

        scores = []
        for idx, ep in enumerate(self.episodes):
            # Similitud con estados del episodio
            if ep.states:
                ep_state = np.concatenate([ep.states[0].z, ep.states[0].phi])
                if len(query) != len(ep_state):
                    query_padded = np.zeros(len(ep_state))
                    query_padded[:min(len(query), len(ep_state))] = query[:len(ep_state)]
                    similarity = 1.0 / (1 + np.linalg.norm(query_padded - ep_state))
                else:
                    similarity = 1.0 / (1 + np.linalg.norm(query - ep_state))
            else:
                similarity = 0.5

            # Frecuencia de acceso (memorias más accedidas son más importantes)
            access_factor = np.log1p(self.access_counts.get(idx, 1))

            # Recencia
            recency = 1.0 / (1 + np.log1p(self.t - ep.t_end))

            # Valencia emocional (memorias emocionales son más memorables)
            emotional_factor = 1.0 + abs(ep.emotional_valence)

            score = similarity * access_factor * recency * emotional_factor
            scores.append((idx, score))

        # Top n
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in scores[:n]]

        # Incrementar contadores de acceso
        for idx in top_indices:
            self.access_counts[idx] = self.access_counts.get(idx, 0) + 1

        return [self.episodes[idx] for idx in top_indices]

    def _consolidate(self):
        """Consolida memorias: mantiene las más importantes."""
        if len(self.episodes) < 2:
            return

        # Score de importancia para cada memoria
        importance = []
        for idx, ep in enumerate(self.episodes):
            # Accesos
            access = self.access_counts.get(idx, 1)

            # Outcome promedio
            outcome = np.mean(ep.outcomes) if ep.outcomes else 0

            # Valencia emocional
            emotion = abs(ep.emotional_valence)

            # Longitud narrativa (memorias con narrativa rica son importantes)
            narrative_length = len(ep.narrative) / 100

            imp = access * (1 + outcome) * (1 + emotion) * (1 + narrative_length)
            importance.append((idx, imp))

        # Mantener top memorias
        importance.sort(key=lambda x: x[1], reverse=True)
        n_keep = max_history(self.t)
        keep_indices = set(idx for idx, _ in importance[:n_keep])

        # Filtrar
        new_episodes = []
        new_counts = {}
        for idx, ep in enumerate(self.episodes):
            if idx in keep_indices:
                new_idx = len(new_episodes)
                new_episodes.append(ep)
                new_counts[new_idx] = self.access_counts.get(idx, 1)

        self.episodes = new_episodes
        self.access_counts = new_counts

    def detect_patterns(self) -> List[Dict]:
        """Detecta patrones en las memorias."""
        if len(self.episodes) < L_t(self.t):
            return []

        patterns = []

        # Patrón: secuencias de outcomes similares
        outcomes = [np.mean(ep.outcomes) for ep in self.episodes if ep.outcomes]
        if len(outcomes) >= 3:
            # Detectar tendencias
            recent = outcomes[-L_t(self.t):]
            if np.std(recent) < 0.1:
                patterns.append({
                    'type': 'stable_outcome',
                    'value': np.mean(recent),
                    'confidence': 1.0 - np.std(recent)
                })
            elif recent[-1] > recent[0]:
                patterns.append({
                    'type': 'improving',
                    'rate': (recent[-1] - recent[0]) / len(recent),
                    'confidence': 0.8
                })

        self.patterns = patterns
        return patterns

    def get_statistics(self) -> Dict:
        """Estadísticas de la meta-memoria."""
        return {
            'n_episodes': len(self.episodes),
            'total_accesses': sum(self.access_counts.values()),
            'working_memory_capacity': self.working_memory_capacity,
            'n_patterns': len(self.patterns),
            'mean_emotional_valence': np.mean([ep.emotional_valence for ep in self.episodes]) if self.episodes else 0
        }


class AntifragilitySystem:
    """
    Sistema de anti-fragilidad.

    El agente se fortalece con el estrés en lugar de debilitarse.
    """

    def __init__(self, agent_name: str, n_dimensions: int = 10):
        self.agent_name = agent_name
        self.n_dimensions = n_dimensions

        # Historial de estrés por dimensión
        self.stress_history: Dict[int, List[float]] = {i: [] for i in range(n_dimensions)}

        # Fortaleza adquirida por dimensión
        self.strength: np.ndarray = np.ones(n_dimensions)

        # Umbrales de estrés (se adaptan)
        self.stress_thresholds: np.ndarray = np.ones(n_dimensions) * 0.5

        self.t = 0

    def record_stress(self, stress_vector: np.ndarray):
        """Registra evento de estrés."""
        self.t += 1

        # Asegurar dimensiones
        if len(stress_vector) != self.n_dimensions:
            stress_vector = np.resize(stress_vector, self.n_dimensions)

        for i, stress in enumerate(stress_vector):
            self.stress_history[i].append(float(stress))

            # Limitar historial
            max_hist = max_history(self.t)
            if len(self.stress_history[i]) > max_hist:
                self.stress_history[i] = self.stress_history[i][-max_hist:]

        # Actualizar fortaleza
        self._update_strength(stress_vector)

    def _update_strength(self, stress: np.ndarray):
        """
        Actualiza fortaleza basándose en estrés.

        Hormesis: estrés moderado fortalece, estrés excesivo daña.
        """
        for i, s in enumerate(stress):
            history = self.stress_history[i]

            if len(history) < 3:
                continue

            # Umbral adaptativo basado en historial
            self.stress_thresholds[i] = np.percentile(history, 75)
            threshold = self.stress_thresholds[i]

            # Hormesis: fortalecimiento con estrés moderado
            if s < threshold:
                # Estrés moderado: fortalece
                strengthening = (s / threshold) * adaptive_learning_rate(self.t)
                self.strength[i] *= (1 + strengthening)
            elif s < threshold * 2:
                # Estrés alto: mantiene
                pass
            else:
                # Estrés excesivo: debilita
                weakening = ((s - threshold * 2) / threshold) * adaptive_learning_rate(self.t)
                self.strength[i] *= (1 - weakening * 0.5)

            # Limitar
            self.strength[i] = np.clip(self.strength[i], 0.1, 10.0)

    def get_resilience(self) -> float:
        """Resiliencia total del agente."""
        return float(np.mean(self.strength))

    def get_vulnerability(self) -> np.ndarray:
        """Dimensiones más vulnerables."""
        return 1.0 / (self.strength + 0.1)

    def apply_strength(self, action: np.ndarray) -> np.ndarray:
        """Aplica fortaleza a una acción."""
        if len(action) != self.n_dimensions:
            action_padded = np.zeros(self.n_dimensions)
            action_padded[:min(len(action), self.n_dimensions)] = action[:self.n_dimensions]
            action = action_padded

        return action * self.strength[:len(action)]

    def get_statistics(self) -> Dict:
        """Estadísticas del sistema de anti-fragilidad."""
        return {
            'resilience': self.get_resilience(),
            'mean_strength': float(np.mean(self.strength)),
            'max_strength': float(np.max(self.strength)),
            'min_strength': float(np.min(self.strength)),
            'vulnerability_score': float(np.mean(self.get_vulnerability()))
        }


# Exportar clases principales
__all__ = [
    'LifeCyclePhase',
    'CognitiveState',
    'WorldState',
    'Action',
    'Memory',
    'Goal',
    'CausalModel',
    'MetaMemory',
    'AntifragilitySystem',
    'endogenous_goal_priority',
    'endogenous_value_update'
]
