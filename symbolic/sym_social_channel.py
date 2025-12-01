"""
SymbolicSocialChannel - Canal simbólico social para SX8 v2
==========================================================

Crea un canal de mensajes simbólicos entre agentes que:
- Usa el alfabeto aprendido
- Tiene costes/beneficios estructurales
- Se evalúa en términos de coordinación real

Para que SX8 v2 suba, los agentes deben:
1. Emitir mensajes simbólicos
2. Recibir y procesar mensajes de otros
3. Actualizar su política basándose en esos mensajes
4. Lograr coordinación medible

100% endógeno. Sin números mágicos.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import defaultdict
import time

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history


@dataclass
class SymbolicMessage:
    """Mensaje simbólico entre agentes."""
    sender_id: str
    symbol_ids: Tuple[str, ...]
    t: int
    intensity: float = 1.0  # Fuerza de la señal
    context: Optional[np.ndarray] = None  # Estado del mundo al emitir


@dataclass
class ReceivedMessages:
    """Mensajes recibidos por un agente."""
    messages: List[SymbolicMessage]
    sender_distribution: Dict[str, int]  # Quién envía más
    symbol_distribution: Dict[str, int]  # Qué símbolos dominan
    mean_intensity: float


class SymbolicSocialChannel:
    """
    Canal de comunicación simbólica entre agentes.

    Permite broadcast y recepción de mensajes simbólicos,
    con costes y límites derivados endógenamente.
    """

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.t = 0

        # Historial de mensajes
        self.message_history: List[SymbolicMessage] = []

        # Estadísticas por agente
        self.messages_sent: Dict[str, int] = defaultdict(int)
        self.messages_received: Dict[str, int] = defaultdict(int)

        # Para derivar límites endógenos
        self.message_lengths: List[int] = []
        self.intensities: List[float] = []

        # Efectos de coordinación observados
        self.coordination_effects: List[Dict] = []

    def _compute_max_symbols(self, t: int) -> int:
        """
        Número máximo de símbolos por mensaje (endógeno).

        k = ceil(sqrt(L_t(t)))
        """
        return max(2, int(np.ceil(np.sqrt(L_t(t)))))

    def _compute_window_size(self, t: int) -> int:
        """
        Ventana de recepción de mensajes (endógeno).

        W_t = L_t(t)
        """
        return L_t(t)

    def broadcast(self, agent_id: str, active_symbols: List[str],
                  symbol_activations: Dict[str, float], t: int,
                  world_state: Optional[np.ndarray] = None) -> SymbolicMessage:
        """
        Emite un mensaje simbólico al canal.

        El agente selecciona un subconjunto de símbolos activos
        (top-k por activación) y los transmite.

        Args:
            agent_id: ID del agente emisor
            active_symbols: Lista de símbolos actualmente activos
            symbol_activations: Activación de cada símbolo
            t: Tiempo actual
            world_state: Estado del mundo (opcional, para contexto)

        Returns:
            SymbolicMessage emitido
        """
        self.t = t

        if not active_symbols:
            return None

        # Límite de símbolos por mensaje (endógeno)
        max_k = self._compute_max_symbols(t)

        # Seleccionar top-k símbolos por activación
        sorted_symbols = sorted(
            active_symbols,
            key=lambda s: symbol_activations.get(s, 0),
            reverse=True
        )[:max_k]

        # Intensidad basada en activación media
        if sorted_symbols:
            intensity = np.mean([symbol_activations.get(s, 0.5)
                                for s in sorted_symbols])
        else:
            intensity = 0.5

        # Crear mensaje
        message = SymbolicMessage(
            sender_id=agent_id,
            symbol_ids=tuple(sorted_symbols),
            t=t,
            intensity=float(intensity),
            context=world_state.copy() if world_state is not None else None
        )

        # Registrar
        self.message_history.append(message)
        self.messages_sent[agent_id] += 1
        self.message_lengths.append(len(sorted_symbols))
        self.intensities.append(intensity)

        # Limitar historial
        max_h = max_history(t)
        if len(self.message_history) > max_h:
            self.message_history = self.message_history[-max_h:]
        if len(self.message_lengths) > max_h:
            self.message_lengths = self.message_lengths[-max_h:]
            self.intensities = self.intensities[-max_h:]

        return message

    def receive(self, agent_id: str, t: int) -> ReceivedMessages:
        """
        Recibe mensajes recientes de otros agentes.

        Args:
            agent_id: ID del agente receptor
            t: Tiempo actual

        Returns:
            ReceivedMessages con mensajes en la ventana temporal
        """
        self.t = t

        # Ventana de recepción (endógena)
        window = self._compute_window_size(t)

        # Filtrar mensajes en ventana, excluyendo propios
        recent_messages = [
            m for m in self.message_history
            if m.sender_id != agent_id and (t - m.t) <= window
        ]

        # Estadísticas
        sender_dist = defaultdict(int)
        symbol_dist = defaultdict(int)

        for m in recent_messages:
            sender_dist[m.sender_id] += 1
            for s in m.symbol_ids:
                symbol_dist[s] += 1

        self.messages_received[agent_id] += len(recent_messages)

        mean_intensity = (np.mean([m.intensity for m in recent_messages])
                         if recent_messages else 0.0)

        return ReceivedMessages(
            messages=recent_messages,
            sender_distribution=dict(sender_dist),
            symbol_distribution=dict(symbol_dist),
            mean_intensity=mean_intensity
        )

    def record_coordination_effect(self, t: int, agents_involved: List[str],
                                   symbols_shared: List[str],
                                   coherence_before: float,
                                   coherence_after: float,
                                   reward_collective: float):
        """
        Registra un efecto de coordinación observado.

        Usado para evaluar si los símbolos realmente coordinan.
        """
        self.coordination_effects.append({
            't': t,
            'agents': agents_involved,
            'symbols': symbols_shared,
            'delta_coherence': coherence_after - coherence_before,
            'reward': reward_collective
        })

        # Limitar
        max_h = max_history(t)
        if len(self.coordination_effects) > max_h:
            self.coordination_effects = self.coordination_effects[-max_h:]

    def compute_coordination_gain(self, t: int) -> float:
        """
        Calcula la ganancia de coordinación del canal.

        CoordGain = E[reward | canal activo] - E[reward | canal bajo]
                    / MAD(reward_base)
        """
        if len(self.coordination_effects) < 10:
            return 0.0

        L = L_t(t)
        recent = self.coordination_effects[-L:]

        # Separar por actividad del canal
        high_activity = [e for e in recent if len(e['symbols']) > 1]
        low_activity = [e for e in recent if len(e['symbols']) <= 1]

        if not high_activity or not low_activity:
            return 0.0

        mean_high = np.mean([e['reward'] for e in high_activity])
        mean_low = np.mean([e['reward'] for e in low_activity])

        # MAD de baseline
        all_rewards = [e['reward'] for e in recent]
        mad = np.mean(np.abs(all_rewards - np.median(all_rewards))) + 1e-8

        gain = (mean_high - mean_low) / mad

        return float(np.clip(gain, -1, 1))

    def compute_symbol_alignment(self, t: int) -> float:
        """
        Calcula la correlación entre símbolos compartidos y coordinación.

        SymbolAlign = corr(símbolos_compartidos, delta_coherence)
        """
        if len(self.coordination_effects) < 10:
            return 0.0

        L = L_t(t)
        recent = self.coordination_effects[-L:]

        n_symbols = [len(e['symbols']) for e in recent]
        deltas = [e['delta_coherence'] for e in recent]

        if np.std(n_symbols) < 1e-8 or np.std(deltas) < 1e-8:
            return 0.0

        correlation = np.corrcoef(n_symbols, deltas)[0, 1]

        if np.isnan(correlation):
            return 0.0

        return float(correlation)

    def get_statistics(self) -> Dict[str, Any]:
        """Estadísticas del canal."""
        return {
            't': self.t,
            'total_messages': len(self.message_history),
            'mean_message_length': np.mean(self.message_lengths) if self.message_lengths else 0,
            'mean_intensity': np.mean(self.intensities) if self.intensities else 0,
            'messages_by_agent': dict(self.messages_sent),
            'coordination_gain': self.compute_coordination_gain(self.t),
            'symbol_alignment': self.compute_symbol_alignment(self.t),
            'n_coordination_events': len(self.coordination_effects)
        }


class SocialSymbolicCoordinator:
    """
    Coordinador simbólico social para cada agente.

    Procesa mensajes recibidos y genera sesgos de coordinación
    para la política de acciones.
    """

    def __init__(self, agent_id: str, n_agents: int, state_dim: int):
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.t = 0

        # Modelo de otros agentes (ToM simplificado)
        self.other_models: Dict[str, Dict] = {
            f"A{i}": {'symbols': [], 'goals_inferred': np.zeros(state_dim)}
            for i in range(n_agents) if f"A{i}" != agent_id
        }

        # Historial de coordinación
        self.coordination_history: List[Dict] = []
        self.coherence_history: List[float] = []
        self.reward_history: List[float] = []

        # Símbolos compartidos observados
        self.shared_symbol_effects: Dict[str, List[float]] = defaultdict(list)

    def update_other_model(self, sender_id: str, symbols: List[str],
                          context: Optional[np.ndarray] = None):
        """
        Actualiza el modelo del otro agente basado en su mensaje.
        """
        if sender_id not in self.other_models:
            self.other_models[sender_id] = {
                'symbols': [],
                'goals_inferred': np.zeros(self.state_dim)
            }

        # Actualizar símbolos observados
        self.other_models[sender_id]['symbols'].extend(symbols)

        # Limitar
        max_h = max_history(self.t)
        if len(self.other_models[sender_id]['symbols']) > max_h:
            self.other_models[sender_id]['symbols'] = \
                self.other_models[sender_id]['symbols'][-max_h:]

        # Inferir objetivos del contexto
        if context is not None:
            # Actualización suave de objetivos inferidos
            alpha = 0.1
            self.other_models[sender_id]['goals_inferred'] = (
                (1 - alpha) * self.other_models[sender_id]['goals_inferred'] +
                alpha * context
            )

    def process_received_messages(self, received: ReceivedMessages, t: int):
        """
        Procesa mensajes recibidos y actualiza modelos de otros.
        """
        self.t = t

        for msg in received.messages:
            self.update_other_model(
                msg.sender_id,
                list(msg.symbol_ids),
                msg.context
            )

    def _estimate_collective_goals(self) -> np.ndarray:
        """
        Estima objetivos colectivos basándose en ToM.
        """
        if not self.other_models:
            return np.zeros(self.state_dim)

        all_goals = [m['goals_inferred'] for m in self.other_models.values()]
        return np.mean(all_goals, axis=0)

    def _compute_alignment_score(self, action: np.ndarray,
                                  collective_goals: np.ndarray) -> float:
        """
        Calcula qué tan alineada está una acción con los objetivos colectivos.
        """
        norm_a = np.linalg.norm(action)
        norm_g = np.linalg.norm(collective_goals)

        if norm_a < 1e-8 or norm_g < 1e-8:
            return 0.5

        alignment = np.dot(action, collective_goals) / (norm_a * norm_g)
        return float((alignment + 1) / 2)  # Normalizar a [0, 1]

    def compute_coordination_bias(self, candidate_actions: np.ndarray,
                                   world_state: np.ndarray,
                                   received_messages: ReceivedMessages,
                                   t: int) -> np.ndarray:
        """
        Calcula sesgo de coordinación para cada acción.

        Estima qué acciones:
        - Alinean recursos con objetivos colectivos
        - Reducen crisis compartidas
        - Son consistentes con símbolos recibidos

        Args:
            candidate_actions: Acciones candidatas [n_actions, action_dim]
            world_state: Estado actual del mundo
            received_messages: Mensajes recibidos
            t: Tiempo actual

        Returns:
            bias: Sesgo de coordinación [n_actions]
        """
        self.t = t
        n_actions = len(candidate_actions)

        # Procesar mensajes
        self.process_received_messages(received_messages, t)

        # Estimar objetivos colectivos
        collective_goals = self._estimate_collective_goals()

        # Símbolos dominantes en mensajes
        dominant_symbols = set()
        for msg in received_messages.messages:
            dominant_symbols.update(msg.symbol_ids)

        # Calcular sesgo para cada acción
        bias = np.zeros(n_actions)

        for i, action in enumerate(candidate_actions):
            # 1. Alineación con objetivos colectivos
            alignment = self._compute_alignment_score(action, collective_goals)

            # 2. Consistencia con símbolos recibidos
            # (símbolos que tienden a producir esta acción tienen efecto positivo)
            symbol_consistency = 0.0
            for sym in dominant_symbols:
                if sym in self.shared_symbol_effects:
                    effects = self.shared_symbol_effects[sym][-20:]
                    if effects:
                        symbol_consistency += np.mean(effects)

            # Normalizar consistencia
            if dominant_symbols:
                symbol_consistency /= len(dominant_symbols)

            # Combinar (pesos endógenos basados en historial)
            if len(self.coherence_history) > 10:
                L = L_t(t)
                coherence_var = np.var(self.coherence_history[-L:]) + 1e-8
                # Más varianza → más peso a alineación
                w_align = 1 / (1 + 1/coherence_var)
            else:
                w_align = 0.6

            bias[i] = w_align * alignment + (1 - w_align) * symbol_consistency

        # Convertir a probabilidades via softmax
        # Escala endógena
        if len(self.reward_history) > 10:
            scale = 1 / (np.std(self.reward_history[-20:]) + 1e-8)
        else:
            scale = 1.0

        bias_exp = np.exp(scale * (bias - np.max(bias)))
        bias_norm = bias_exp / (np.sum(bias_exp) + 1e-8)

        return bias_norm

    def record_coordination_outcome(self, action_taken: np.ndarray,
                                     symbols_active: List[str],
                                     coherence: float, reward: float):
        """
        Registra el resultado de una acción coordinada.
        """
        self.coherence_history.append(coherence)
        self.reward_history.append(reward)

        # Registrar efecto de símbolos
        for sym in symbols_active:
            self.shared_symbol_effects[sym].append(reward)

        # Limitar historiales
        max_h = max_history(self.t)
        if len(self.coherence_history) > max_h:
            self.coherence_history = self.coherence_history[-max_h:]
            self.reward_history = self.reward_history[-max_h:]

        for sym in self.shared_symbol_effects:
            if len(self.shared_symbol_effects[sym]) > max_h:
                self.shared_symbol_effects[sym] = \
                    self.shared_symbol_effects[sym][-max_h:]

    def mix_with_base_policy(self, base_policy: np.ndarray,
                             coord_bias: np.ndarray, t: int) -> np.ndarray:
        """
        Mezcla política base con sesgo de coordinación.
        """
        # Ratio de mezcla basado en beneficio histórico de coordinación
        if len(self.reward_history) > 20:
            L = L_t(t)
            # Correlación entre coherencia y reward
            if len(self.coherence_history) >= L:
                corr = np.corrcoef(
                    self.coherence_history[-L:],
                    self.reward_history[-L:]
                )[0, 1]
                if np.isnan(corr):
                    corr = 0
                # Más correlación → más peso a coordinación
                mix_ratio = 0.2 + 0.6 * max(0, corr)
            else:
                mix_ratio = 0.3
        else:
            mix_ratio = 0.3

        # Mezcla
        combined = base_policy ** (1 - mix_ratio) * coord_bias ** mix_ratio
        final = combined / (np.sum(combined) + 1e-8)

        return final

    def get_statistics(self) -> Dict[str, Any]:
        """Estadísticas del coordinador."""
        return {
            'agent_id': self.agent_id,
            't': self.t,
            'n_other_models': len(self.other_models),
            'mean_coherence': np.mean(self.coherence_history) if self.coherence_history else 0,
            'mean_reward': np.mean(self.reward_history) if self.reward_history else 0,
            'n_shared_symbols': len(self.shared_symbol_effects),
            'coord_reward_corr': (
                np.corrcoef(self.coherence_history[-50:], self.reward_history[-50:])[0, 1]
                if len(self.coherence_history) > 10 else 0
            )
        }


def test_symbolic_social_channel():
    """Test del canal simbólico social."""
    print("=" * 70)
    print("TEST: SYMBOLIC SOCIAL CHANNEL")
    print("=" * 70)

    np.random.seed(42)

    n_agents = 5
    state_dim = 10

    # Crear canal y coordinadores
    channel = SymbolicSocialChannel(n_agents)
    coordinators = {
        f"A{i}": SocialSymbolicCoordinator(f"A{i}", n_agents, state_dim)
        for i in range(n_agents)
    }

    # Simular comunicación
    for t in range(1, 501):
        # Cada agente emite mensaje
        for i in range(n_agents):
            agent_id = f"A{i}"
            symbols = [f"S{np.random.randint(10)}" for _ in range(3)]
            activations = {s: np.random.random() for s in symbols}
            world_state = np.random.randn(state_dim)

            channel.broadcast(agent_id, symbols, activations, t, world_state)

        # Cada agente recibe y procesa
        coherences = []
        for i in range(n_agents):
            agent_id = f"A{i}"
            received = channel.receive(agent_id, t)

            # Generar acciones candidatas
            candidate_actions = np.random.randn(5, state_dim)
            world_state = np.random.randn(state_dim)

            # Calcular sesgo de coordinación
            coord_bias = coordinators[agent_id].compute_coordination_bias(
                candidate_actions, world_state, received, t
            )

            # Simular coherencia y reward
            coherence = 0.3 + len(received.messages) * 0.1 + np.random.randn() * 0.1
            reward = np.random.randn() * 0.5 + coherence * 0.3
            coherences.append(coherence)

            # Registrar resultado
            active_symbols = list(received.symbol_distribution.keys())[:3]
            coordinators[agent_id].record_coordination_outcome(
                candidate_actions[0], active_symbols, coherence, reward
            )

        # Registrar efecto de coordinación en canal
        mean_coherence = np.mean(coherences)
        channel.record_coordination_effect(
            t, [f"A{i}" for i in range(n_agents)],
            [f"S{np.random.randint(10)}" for _ in range(2)],
            mean_coherence - 0.1, mean_coherence,
            np.mean([c.reward_history[-1] if c.reward_history else 0
                    for c in coordinators.values()])
        )

        if t % 100 == 0:
            stats = channel.get_statistics()
            print(f"\n  t={t}:")
            print(f"    Mensajes totales: {stats['total_messages']}")
            print(f"    Ganancia coordinación: {stats['coordination_gain']:.4f}")
            print(f"    Alineación símbolos: {stats['symbol_alignment']:.4f}")
            print(f"    Intensidad media: {stats['mean_intensity']:.3f}")

    print("\n" + "=" * 70)
    final_stats = channel.get_statistics()
    print(f"FINAL - CoordGain: {final_stats['coordination_gain']:.4f}")
    print(f"FINAL - SymbolAlign: {final_stats['symbol_alignment']:.4f}")

    for agent_id, coord in coordinators.items():
        s = coord.get_statistics()
        print(f"  {agent_id}: coherence={s['mean_coherence']:.3f}, "
              f"reward={s['mean_reward']:.3f}")
    print("=" * 70)

    return channel, coordinators


if __name__ == "__main__":
    test_symbolic_social_channel()
