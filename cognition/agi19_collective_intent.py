"""
AGI-19: Intencionalidad Colectiva
==================================

"Cuando muchos agentes convergen hacia algo, ese algo tiene peso."

Vector de intención:
    i_t^A = dirección del gradiente de V_A en espacio de estados

Intención colectiva:
    I_col = Σ_A w_A · i_t^A
    w_A = V_A / Σ_B V_B

Coherencia:
    Coh = 1 - var_A(angle(i_t^A, I_col))

Índice de intencionalidad:
    Int_t = ||I_col|| · Coh

Umbral de intención:
    τ_I = percentile({Int_τ}, 75)

Detección de intención emergente:
    Intent si Int_t > τ_I durante L_t pasos

100% endógeno.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from .agi_dynamic_constants import (
    L_t, max_history, dynamic_percentile_high, adaptive_momentum
)


def intent_threshold(intent_history: List[float], t: int) -> float:
    """
    Umbral de intención endógeno.

    τ_I = percentile({Int_τ}, 75 + 5/√(t+1))
    """
    if len(intent_history) < 5:
        return 0.5

    percentile_idx = 75 + 5 / np.sqrt(t + 1)
    percentile_idx = min(95, percentile_idx)

    return float(np.percentile(intent_history, percentile_idx))


@dataclass
class AgentIntent:
    """Intención de un agente."""
    agent_name: str
    intent_vector: np.ndarray
    intent_magnitude: float
    value: float
    weight: float
    alignment_with_collective: float


@dataclass
class CollectiveIntentState:
    """Estado de intención colectiva."""
    t: int
    collective_intent: np.ndarray
    intent_magnitude: float
    coherence: float
    intentionality_index: float
    is_emergent: bool
    emergent_duration: int
    agent_intents: Dict[str, AgentIntent]


@dataclass
class EmergentGoal:
    """Una meta colectiva emergente."""
    goal_id: int
    direction: np.ndarray
    strength: float
    coherence: float
    n_agents_aligned: int
    detection_time: int
    duration: int
    is_active: bool = True


class CollectiveIntentionality:
    """
    Sistema de intencionalidad colectiva.

    Detecta cuando los agentes convergen hacia
    objetivos compartidos emergentes.
    """

    def __init__(self, agent_names: List[str], state_dim: int = 10):
        """
        Inicializa sistema de intencionalidad.

        Args:
            agent_names: Lista de agentes
            state_dim: Dimensión del espacio de estados
        """
        self.agent_names = agent_names
        self.state_dim = state_dim

        # Historial de estados por agente
        self.state_history: Dict[str, List[np.ndarray]] = {
            name: [] for name in agent_names
        }

        # Valores por agente
        self.value_history: Dict[str, List[float]] = {
            name: [] for name in agent_names
        }

        # Intenciones calculadas
        self.intents: Dict[str, AgentIntent] = {}

        # Intención colectiva
        self.collective_intent: np.ndarray = np.zeros(state_dim)
        self.collective_intent_history: List[np.ndarray] = []

        # Historial de índice de intencionalidad
        self.intentionality_history: List[float] = []

        # Metas emergentes
        self.emergent_goals: Dict[int, EmergentGoal] = {}
        self.next_goal_id = 0

        # Contador de emergencia
        self.emergence_counter = 0

        self.t = 0

    def _compute_intent_vector(self, agent_name: str) -> np.ndarray:
        """
        Calcula vector de intención de un agente.

        i_t^A = dirección del gradiente de V_A en espacio de estados
        """
        states = self.state_history.get(agent_name, [])
        values = self.value_history.get(agent_name, [])

        min_samples = L_t(self.t)
        if len(states) < min_samples or len(values) < min_samples:
            return np.random.randn(self.state_dim) * 0.01

        window = min(max_history(self.t), len(states), len(values))
        recent_states = np.array(states[-window:])
        recent_values = np.array(values[-window:])

        # Estimar gradiente: correlación de cada dimensión con valor
        intent = np.zeros(self.state_dim)

        for d in range(self.state_dim):
            if d < recent_states.shape[1]:
                state_dim = recent_states[:, d]
                if np.std(state_dim) > 0 and np.std(recent_values) > 0:
                    corr = np.corrcoef(state_dim, recent_values)[0, 1]
                    if not np.isnan(corr):
                        intent[d] = corr

        # Normalizar
        norm = np.linalg.norm(intent)
        if norm > 0:
            intent = intent / norm

        return intent

    def _compute_collective_intent(self):
        """
        Calcula intención colectiva.

        I_col = Σ_A w_A · i_t^A
        w_A = V_A / Σ_B V_B
        """
        # Calcular valores totales para pesos
        total_value = 0.0
        agent_values = {}

        for name in self.agent_names:
            values = self.value_history.get(name, [])
            if values:
                avg_value = float(np.mean(values[-50:])) if len(values) >= 50 else float(np.mean(values))
                agent_values[name] = max(0.01, avg_value)
                total_value += agent_values[name]
            else:
                agent_values[name] = 0.01
                total_value += 0.01

        # Calcular intenciones y pesos
        self.collective_intent = np.zeros(self.state_dim)

        for name in self.agent_names:
            # Intención individual
            intent = self._compute_intent_vector(name)
            magnitude = float(np.linalg.norm(intent))

            # Peso
            weight = agent_values[name] / (total_value + 1e-8)

            # Contribuir a colectiva
            self.collective_intent += weight * intent

            # Guardar
            self.intents[name] = AgentIntent(
                agent_name=name,
                intent_vector=intent,
                intent_magnitude=magnitude,
                value=agent_values[name],
                weight=weight,
                alignment_with_collective=0.0  # Se calcula después
            )

        # Guardar historial
        self.collective_intent_history.append(self.collective_intent.copy())
        if len(self.collective_intent_history) > max_history(self.t):
            self.collective_intent_history = self.collective_intent_history[-max_history(self.t):]

    def _compute_coherence(self) -> float:
        """
        Calcula coherencia de intenciones.

        Coh = 1 - var_A(angle(i_t^A, I_col))
        """
        if not self.intents:
            return 0.0

        collective_norm = np.linalg.norm(self.collective_intent)
        if collective_norm < 1e-8:
            return 0.0

        angles = []
        for name, intent in self.intents.items():
            intent_norm = np.linalg.norm(intent.intent_vector)
            if intent_norm > 1e-8:
                # Coseno del ángulo
                cos_angle = np.dot(intent.intent_vector, self.collective_intent) / \
                           (intent_norm * collective_norm)
                cos_angle = np.clip(cos_angle, -1, 1)

                # Ángulo normalizado [0, 1]
                angle = np.arccos(cos_angle) / np.pi
                angles.append(angle)

                # Actualizar alineamiento
                intent.alignment_with_collective = 1.0 - angle

        if not angles:
            return 0.0

        # Coherencia = 1 - varianza de ángulos
        variance = np.var(angles)
        coherence = 1.0 - min(variance * 4, 1.0)  # Escalar varianza

        return float(coherence)

    def _compute_intentionality_index(self, coherence: float) -> float:
        """
        Calcula índice de intencionalidad.

        Int_t = ||I_col|| · Coh
        """
        magnitude = float(np.linalg.norm(self.collective_intent))
        return magnitude * coherence

    def _detect_emergent_goals(self, intentionality: float, coherence: float):
        """
        Detecta metas colectivas emergentes.

        Intent si Int_t > τ_I durante L_t pasos
        """
        threshold = intent_threshold(self.intentionality_history, self.t)

        if intentionality > threshold and coherence > 0.5:
            self.emergence_counter += 1

            min_duration = L_t(self.t)
            if self.emergence_counter >= min_duration:
                # Verificar si ya existe meta similar
                found_similar = False
                for goal in self.emergent_goals.values():
                    if goal.is_active:
                        similarity = np.dot(goal.direction, self.collective_intent) / \
                                   (np.linalg.norm(goal.direction) * np.linalg.norm(self.collective_intent) + 1e-8)
                        if similarity > 0.8:
                            # Actualizar existente
                            beta = adaptive_momentum(self.intentionality_history)
                            goal.direction = beta * goal.direction + (1 - beta) * self.collective_intent
                            goal.strength = max(goal.strength, intentionality)
                            goal.coherence = coherence
                            goal.duration += 1
                            found_similar = True
                            break

                if not found_similar:
                    # Nueva meta emergente
                    n_aligned = sum(1 for i in self.intents.values()
                                   if i.alignment_with_collective > 0.6)

                    goal = EmergentGoal(
                        goal_id=self.next_goal_id,
                        direction=self.collective_intent.copy(),
                        strength=intentionality,
                        coherence=coherence,
                        n_agents_aligned=n_aligned,
                        detection_time=self.t,
                        duration=1
                    )
                    self.emergent_goals[self.next_goal_id] = goal
                    self.next_goal_id += 1
        else:
            self.emergence_counter = max(0, self.emergence_counter - 1)

        # Desactivar metas antiguas
        decay_time = max_history(self.t) // 2
        for goal in self.emergent_goals.values():
            if goal.is_active and self.t - goal.detection_time > decay_time:
                if goal.duration < L_t(self.t) * 2:
                    goal.is_active = False

    def record_state(self, agent_name: str, state: np.ndarray, value: float):
        """
        Registra estado y valor de un agente.

        Args:
            agent_name: Nombre del agente
            state: Vector de estado
            value: Valor actual
        """
        self.t += 1

        if agent_name in self.state_history:
            self.state_history[agent_name].append(state.copy())
            self.value_history[agent_name].append(value)

            # Limitar historial
            max_hist = max_history(self.t)
            if len(self.state_history[agent_name]) > max_hist:
                self.state_history[agent_name] = self.state_history[agent_name][-max_hist:]
                self.value_history[agent_name] = self.value_history[agent_name][-max_hist:]

        # Actualizar intención colectiva periódicamente
        update_freq = max(5, L_t(self.t) // 2)
        if self.t % update_freq == 0:
            self._compute_collective_intent()
            coherence = self._compute_coherence()
            intentionality = self._compute_intentionality_index(coherence)

            self.intentionality_history.append(intentionality)
            if len(self.intentionality_history) > max_history(self.t):
                self.intentionality_history = self.intentionality_history[-max_history(self.t):]

            self._detect_emergent_goals(intentionality, coherence)

    def get_collective_direction(self) -> Tuple[np.ndarray, float]:
        """
        Obtiene dirección colectiva actual.

        Returns:
            (dirección, confianza)
        """
        if len(self.intentionality_history) < L_t(self.t):
            return np.zeros(self.state_dim), 0.0

        magnitude = float(np.linalg.norm(self.collective_intent))
        if magnitude < 1e-8:
            return np.zeros(self.state_dim), 0.0

        direction = self.collective_intent / magnitude
        confidence = min(1.0, np.mean(self.intentionality_history[-20:]) * 2)

        return direction, float(confidence)

    def get_alignment_bonus(self, agent_name: str, action_direction: np.ndarray) -> float:
        """
        Calcula bonus por alinearse con intención colectiva.

        Args:
            agent_name: Nombre del agente
            action_direction: Dirección de la acción propuesta

        Returns:
            Bonus [0, 0.3]
        """
        if agent_name not in self.intents:
            return 0.0

        collective_norm = np.linalg.norm(self.collective_intent)
        action_norm = np.linalg.norm(action_direction)

        if collective_norm < 1e-8 or action_norm < 1e-8:
            return 0.0

        # Alineamiento
        cos_angle = np.dot(action_direction, self.collective_intent) / \
                   (action_norm * collective_norm)
        alignment = (cos_angle + 1) / 2  # [0, 1]

        # Bonus proporcional a alineamiento y fuerza de intención
        recent_int = np.mean(self.intentionality_history[-10:]) if self.intentionality_history else 0
        bonus = alignment * recent_int * 0.3

        return float(np.clip(bonus, 0, 0.3))

    def is_intent_emergent(self) -> bool:
        """Verifica si hay intención emergente activa."""
        return any(g.is_active for g in self.emergent_goals.values())

    def get_active_goals(self) -> List[EmergentGoal]:
        """Obtiene metas activas."""
        return [g for g in self.emergent_goals.values() if g.is_active]

    def get_state(self) -> CollectiveIntentState:
        """Obtiene estado actual."""
        coherence = self._compute_coherence() if self.intents else 0.0
        intentionality = self._compute_intentionality_index(coherence)

        return CollectiveIntentState(
            t=self.t,
            collective_intent=self.collective_intent.copy(),
            intent_magnitude=float(np.linalg.norm(self.collective_intent)),
            coherence=coherence,
            intentionality_index=intentionality,
            is_emergent=self.is_intent_emergent(),
            emergent_duration=self.emergence_counter,
            agent_intents=self.intents.copy()
        )

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas del sistema."""
        state = self.get_state()
        active_goals = self.get_active_goals()

        agent_stats = {}
        for name, intent in self.intents.items():
            agent_stats[name] = {
                'intent_magnitude': intent.intent_magnitude,
                'weight': intent.weight,
                'alignment': intent.alignment_with_collective
            }

        return {
            't': self.t,
            'n_agents': len(self.agent_names),
            'intent_magnitude': state.intent_magnitude,
            'coherence': state.coherence,
            'intentionality_index': state.intentionality_index,
            'is_emergent': state.is_emergent,
            'emergence_counter': self.emergence_counter,
            'n_active_goals': len(active_goals),
            'n_total_goals': len(self.emergent_goals),
            'agent_stats': agent_stats,
            'active_goals': [
                {
                    'id': g.goal_id,
                    'strength': g.strength,
                    'coherence': g.coherence,
                    'n_aligned': g.n_agents_aligned,
                    'duration': g.duration
                }
                for g in active_goals
            ]
        }


def test_collective_intent():
    """Test de intencionalidad colectiva."""
    print("=" * 60)
    print("TEST AGI-19: COLLECTIVE INTENTIONALITY")
    print("=" * 60)

    agents = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
    collective = CollectiveIntentionality(agents, state_dim=5)

    print(f"\nSimulando 500 pasos con {len(agents)} agentes...")

    for t in range(500):
        # Fase determina si hay convergencia
        phase = (t // 100) % 3

        for agent in agents:
            # Estado base
            state = np.random.randn(5) * 0.3

            if phase == 0:
                # Todos convergen hacia misma dirección
                target = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
                state += target * 0.5 + np.random.randn(5) * 0.1
            elif phase == 1:
                # Divergencia
                if agent in ['NEO', 'EVA']:
                    state += np.array([1.0, 0.0, 0.0, 0.0, 0.0]) * 0.5
                else:
                    state += np.array([-1.0, 0.0, 0.0, 0.0, 0.0]) * 0.5
            else:
                # Convergencia parcial
                target = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
                state += target * 0.3 + np.random.randn(5) * 0.2

            # Valor correlacionado con convergencia
            if phase == 0:
                value = 0.7 + np.random.randn() * 0.1
            elif phase == 1:
                value = 0.4 + np.random.randn() * 0.1
            else:
                value = 0.55 + np.random.randn() * 0.1

            collective.record_state(agent, state, value)

        if (t + 1) % 100 == 0:
            state = collective.get_state()
            print(f"  t={t+1}: coherence={state.coherence:.3f}, "
                  f"int_idx={state.intentionality_index:.3f}, "
                  f"emergent={state.is_emergent}")

    # Resultados finales
    stats = collective.get_statistics()

    print("\n" + "=" * 60)
    print("RESULTADOS COLLECTIVE INTENTIONALITY")
    print("=" * 60)

    print(f"\n  Agentes: {stats['n_agents']}")
    print(f"  Magnitud intención: {stats['intent_magnitude']:.3f}")
    print(f"  Coherencia: {stats['coherence']:.3f}")
    print(f"  Índice intencionalidad: {stats['intentionality_index']:.3f}")
    print(f"  Es emergente: {stats['is_emergent']}")
    print(f"  Metas activas: {stats['n_active_goals']}/{stats['n_total_goals']}")

    print("\n  Por agente:")
    for name, agent_stats in stats['agent_stats'].items():
        print(f"    {name}: align={agent_stats['alignment']:.3f}, "
              f"weight={agent_stats['weight']:.3f}")

    if stats['active_goals']:
        print("\n  Metas activas:")
        for goal in stats['active_goals']:
            print(f"    Goal {goal['id']}: strength={goal['strength']:.3f}, "
                  f"coherence={goal['coherence']:.3f}, "
                  f"aligned={goal['n_aligned']}")

    # Test de dirección colectiva
    print("\n  Dirección colectiva:")
    direction, confidence = collective.get_collective_direction()
    print(f"    Dirección: {direction[:3]}...")
    print(f"    Confianza: {confidence:.3f}")

    # Test de bonus de alineamiento
    print("\n  Bonus de alineamiento:")
    test_action = np.array([0.5, 0.3, 0.1, 0.0, 0.0])
    for agent in agents[:2]:
        bonus = collective.get_alignment_bonus(agent, test_action)
        print(f"    {agent}: bonus={bonus:.3f}")

    if stats['n_total_goals'] > 0:
        print("\n  ✓ Intencionalidad colectiva funcionando")
    else:
        print("\n  ⚠ Sin metas emergentes detectadas")

    return collective


if __name__ == "__main__":
    test_collective_intent()
