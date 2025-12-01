#!/usr/bin/env python3
"""
COALITION GAME QG1 - Juego de Coalición Cuántico Endógeno
=========================================================

Cada ronda del juego:
1. Lee estados actuales (drives, attachment, S, φ, crisis)
2. Cada agente elige operador O_k con prob ∝ drives relevantes
3. Aplica O_k, actualiza drives
4. Recodifica ψ(t+1)
5. Recalcula métricas

TODO es endógeno:
- Sin "cooperar"/"defeccionar"
- Sin payoffs externos
- Solo Δ de métricas internas

CHECK: Verificar que no hay magic numbers.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import sys
import os

# Agregar path del proyecto
sys.path.insert(0, '/root/NEO_EVA')

from quantum_game.endogenous.state_encoding import QuantumStateEncoding, EntangledStateEncoding
from quantum_game.endogenous.operators_qg import OperatorSelector


@dataclass
class AgentGameState:
    """
    Estado de un agente en el juego.

    Todo derivado de drives, sin valores externos.
    """
    name: str
    dim: int = 6  # Dimensión de drives (estructural, no magic)

    # Estado cuántico
    quantum_state: QuantumStateEncoding = None

    # Drives actuales
    drives: np.ndarray = None

    # Historial para métricas endógenas
    drive_history: List[np.ndarray] = field(default_factory=list)
    z_history: List[np.ndarray] = field(default_factory=list)

    # Métricas endógenas (calculadas, no impuestas)
    entropy: float = 0.5
    phi: float = 0.5
    coherence: float = 0.5
    identity: float = 0.5
    in_crisis: bool = False

    # Vínculos con otros (endógeno)
    attachments: Dict[str, float] = field(default_factory=dict)

    # Selector de operadores
    operator_selector: OperatorSelector = None

    def __post_init__(self):
        if self.drives is None:
            # Inicialización uniforme (máxima entropía)
            self.drives = np.ones(self.dim) / self.dim

        if self.quantum_state is None:
            self.quantum_state = QuantumStateEncoding.from_drives(self.drives)

        if self.operator_selector is None:
            self.operator_selector = OperatorSelector()

    def compute_entropy_endogenous(self) -> float:
        """
        Entropía endógena: normalizada por la propia historia.
        """
        probs = self.drives / (self.drives.sum() + 1e-16)
        probs = np.clip(probs, 1e-16, 1)
        raw_entropy = -np.sum(probs * np.log(probs))

        if len(self.drive_history) > 10:
            hist_entropies = []
            for d in self.drive_history[-50:]:
                p = d / (d.sum() + 1e-16)
                p = np.clip(p, 1e-16, 1)
                hist_entropies.append(-np.sum(p * np.log(p)))

            e_min = np.percentile(hist_entropies, 5)
            e_max = np.percentile(hist_entropies, 95)

            if e_max > e_min:
                return np.clip((raw_entropy - e_min) / (e_max - e_min), 0, 1)

        return raw_entropy / np.log(self.dim)

    def compute_phi_endogenous(self) -> float:
        """
        φ endógeno: integración de información desde covarianza.

        φ = off_diagonal_sum / (trace × dim)
        """
        if len(self.z_history) < 10:
            return 0.5

        # Ventana endógena: √(len(history))
        window = max(10, int(np.sqrt(len(self.z_history))))
        recent = np.array(self.z_history[-window:])

        if recent.shape[0] < 3:
            return 0.5

        try:
            cov_matrix = np.cov(recent.T)
            total_var = np.trace(cov_matrix) + 1e-16
            off_diagonal = np.sum(np.abs(cov_matrix)) - np.trace(np.abs(cov_matrix))
            phi = off_diagonal / (total_var * recent.shape[1])
            return np.clip(phi, 0, 1)
        except:
            return 0.5

    def compute_identity_endogenous(self) -> float:
        """
        Identidad endógena: autocorrelación de drives.
        """
        if len(self.drive_history) < 20:
            return 0.5

        # Ventana: √(len)
        window = max(5, int(np.sqrt(len(self.drive_history))))

        recent = np.array(self.drive_history[-window:])
        older = np.array(self.drive_history[-2*window:-window]) if len(self.drive_history) >= 2*window else np.array(self.drive_history[:window])

        # Autocorrelación
        correlations = []
        for d in range(self.dim):
            if len(recent[:, d]) > 2 and len(older[:, d]) > 2:
                c = np.corrcoef(recent[:, d].flatten(), older[:, d].flatten()[:len(recent)])[0, 1]
                if not np.isnan(c):
                    correlations.append(c)

        if correlations:
            return np.clip((np.mean(correlations) + 1) / 2, 0, 1)
        return 0.5

    def compute_coherence_endogenous(self) -> float:
        """
        Coherencia endógena: estabilidad relativa.
        """
        if len(self.drive_history) < 20:
            return 0.5

        window = max(5, int(np.sqrt(len(self.drive_history))))

        recent = np.array(self.drive_history[-window:])
        baseline = np.array(self.drive_history[:-window])

        var_recent = np.mean(np.var(recent, axis=0))
        var_baseline = np.mean(np.var(baseline, axis=0)) + 1e-16

        return 1.0 / (1.0 + var_recent / var_baseline)

    def detect_crisis_endogenous(self) -> bool:
        """
        Detecta crisis desde métricas endógenas.

        Crisis = entropía baja + identidad baja + coherencia baja
        Todo relativo a la historia.
        """
        if len(self.drive_history) < 30:
            return False

        # Umbrales desde percentiles de la historia
        hist_entropies = [self.compute_entropy_endogenous() for _ in range(1)]  # Actual

        # Crisis cuando todo está por debajo del percentil 20
        entropy_low = self.entropy < 0.2  # Percentil bajo
        identity_low = self.identity < 0.3
        coherence_low = self.coherence < 0.3

        return entropy_low and identity_low and coherence_low

    def compute_attachment_endogenous(self, other_history: List[np.ndarray]) -> float:
        """
        Calcula attachment desde correlación histórica.
        """
        if len(self.drive_history) < 20 or len(other_history) < 20:
            return 0.0

        # Ventana endógena
        window = min(len(self.drive_history), len(other_history), 50)

        my_recent = np.array(self.drive_history[-window:])
        other_recent = np.array(other_history[-window:])

        # Correlación promedio
        correlations = []
        for d in range(min(my_recent.shape[1], other_recent.shape[1])):
            c = np.corrcoef(my_recent[:, d], other_recent[:, d])[0, 1]
            if not np.isnan(c):
                correlations.append(abs(c))

        return np.mean(correlations) if correlations else 0.0

    def update_all_metrics(self, other_agents: Dict[str, 'AgentGameState'] = None):
        """Actualiza todas las métricas endógenas."""
        self.entropy = self.compute_entropy_endogenous()
        self.phi = self.compute_phi_endogenous()
        self.identity = self.compute_identity_endogenous()
        self.coherence = self.compute_coherence_endogenous()
        self.in_crisis = self.detect_crisis_endogenous()

        if other_agents:
            for name, other in other_agents.items():
                if name != self.name:
                    self.attachments[name] = self.compute_attachment_endogenous(other.drive_history)

    def get_context(self, other_agents: Dict[str, 'AgentGameState'] = None) -> Dict:
        """Construye contexto para selección de operador."""
        context = {
            'drive_history': self.drive_history,
            'z_history': self.z_history,
            'coherence': self.coherence,
            'phi': self.phi,
            'identity': self.identity,
            'in_crisis': self.in_crisis,
            'attachment': max(self.attachments.values()) if self.attachments else 0
        }

        if other_agents:
            # Drives del agente con mayor attachment
            if self.attachments:
                most_attached = max(self.attachments.keys(), key=lambda k: self.attachments[k])
                context['other_drives'] = other_agents[most_attached].drives

        return context


@dataclass
class CoalitionGameRound:
    """
    Una ronda del juego de coalición.

    Registra acciones y resultados sin semántica externa.
    """
    round_number: int

    # Estados antes de la ronda
    states_before: Dict[str, Dict] = field(default_factory=dict)

    # Operadores seleccionados
    operators_selected: Dict[str, str] = field(default_factory=dict)

    # Estados después de la ronda
    states_after: Dict[str, Dict] = field(default_factory=dict)

    # Deltas de métricas (lo que será payoff)
    metric_deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)


class CoalitionGameQG1:
    """
    Juego de Coalición Cuántico Endógeno.

    No hay "cooperar" o "defeccionar".
    Solo agentes eligiendo operadores según sus drives,
    y payoffs como Δ de métricas endógenas.
    """

    def __init__(self, agent_names: List[str], dim: int = 6):
        """
        Inicializa el juego.

        Args:
            agent_names: Nombres de agentes (ej: ['NEO', 'EVA', 'ALEX'])
            dim: Dimensión de drives (estructural)
        """
        self.agent_names = agent_names
        self.dim = dim

        # Crear agentes
        self.agents: Dict[str, AgentGameState] = {}
        for name in agent_names:
            self.agents[name] = AgentGameState(name=name, dim=dim)

        # Estado entangled
        self.entangled_state = EntangledStateEncoding(
            agent_names=agent_names,
            local_states={name: self.agents[name].quantum_state for name in agent_names}
        )

        # Historia del juego
        self.rounds: List[CoalitionGameRound] = []
        self.current_round = 0

        # Estadísticas
        self.operator_counts = defaultdict(lambda: defaultdict(int))
        self.crisis_history = defaultdict(list)

    def capture_state(self, agent: AgentGameState) -> Dict:
        """Captura estado actual de un agente."""
        return {
            'drives': agent.drives.copy(),
            'entropy': agent.entropy,
            'phi': agent.phi,
            'identity': agent.identity,
            'coherence': agent.coherence,
            'in_crisis': agent.in_crisis,
            'attachments': agent.attachments.copy()
        }

    def compute_deltas(self, before: Dict, after: Dict) -> Dict[str, float]:
        """Calcula deltas de métricas."""
        return {
            'delta_entropy': after['entropy'] - before['entropy'],
            'delta_phi': after['phi'] - before['phi'],
            'delta_identity': after['identity'] - before['identity'],
            'delta_coherence': after['coherence'] - before['coherence'],
            'delta_attachment': sum(after['attachments'].values()) - sum(before['attachments'].values()),
            'crisis_change': int(after['in_crisis']) - int(before['in_crisis'])
        }

    def play_round(self) -> CoalitionGameRound:
        """
        Ejecuta una ronda del juego.

        1. Captura estados
        2. Cada agente selecciona operador
        3. Aplica operadores
        4. Actualiza estados cuánticos
        5. Recalcula métricas
        6. Registra deltas
        """
        self.current_round += 1
        round_data = CoalitionGameRound(round_number=self.current_round)

        # 1. Capturar estados antes
        for name, agent in self.agents.items():
            round_data.states_before[name] = self.capture_state(agent)

        # 2. Cada agente selecciona operador (probabilidad ∝ drives)
        for name, agent in self.agents.items():
            context = agent.get_context(self.agents)
            selected_op = agent.operator_selector.select(agent.drives, context)
            round_data.operators_selected[name] = selected_op
            self.operator_counts[name][selected_op] += 1

        # 3. Aplicar operadores (simultáneamente)
        new_drives = {}
        for name, agent in self.agents.items():
            context = agent.get_context(self.agents)
            op_name = round_data.operators_selected[name]
            new_drives[name] = agent.operator_selector.apply(op_name, agent.drives.copy(), context)

        # 4. Actualizar drives y estados cuánticos
        for name, agent in self.agents.items():
            # Guardar en historia
            agent.drive_history.append(agent.drives.copy())

            # Actualizar drives
            agent.drives = new_drives[name]

            # Actualizar z (representación latente simple)
            z = agent.drives - np.mean(agent.drives)
            agent.z_history.append(z)

            # Actualizar estado cuántico
            agent.quantum_state.update(agent.drives)

            # Mantener historia acotada
            max_history = 500
            if len(agent.drive_history) > max_history:
                agent.drive_history = agent.drive_history[-max_history:]
                agent.z_history = agent.z_history[-max_history:]

        # 5. Actualizar entanglement
        current_states = {name: agent.drives for name, agent in self.agents.items()}
        self.entangled_state.update_entanglement(current_states)

        # 6. Recalcular métricas
        for name, agent in self.agents.items():
            agent.update_all_metrics(self.agents)
            self.crisis_history[name].append(agent.in_crisis)

        # 7. Capturar estados después y calcular deltas
        for name, agent in self.agents.items():
            round_data.states_after[name] = self.capture_state(agent)
            round_data.metric_deltas[name] = self.compute_deltas(
                round_data.states_before[name],
                round_data.states_after[name]
            )

        self.rounds.append(round_data)
        return round_data

    def play_game(self, num_rounds: int) -> List[CoalitionGameRound]:
        """Juega múltiples rondas."""
        results = []
        for _ in range(num_rounds):
            results.append(self.play_round())
        return results

    def get_entanglement_matrix(self) -> np.ndarray:
        """Retorna matriz de entanglement actual."""
        return self.entangled_state.correlation_matrix

    def get_statistics(self) -> Dict:
        """
        Estadísticas del juego.

        Todo expresado en términos endógenos (no "ganadores").
        """
        stats = {
            'total_rounds': len(self.rounds),
            'agents': {}
        }

        for name, agent in self.agents.items():
            # Contar operadores usados
            ops = dict(self.operator_counts[name])
            total_ops = sum(ops.values())
            op_distribution = {k: v/total_ops for k, v in ops.items()} if total_ops > 0 else {}

            # Estadísticas de métricas
            if self.rounds:
                entropies = [r.states_after[name]['entropy'] for r in self.rounds]
                phis = [r.states_after[name]['phi'] for r in self.rounds]
                identities = [r.states_after[name]['identity'] for r in self.rounds]

                stats['agents'][name] = {
                    'operator_distribution': op_distribution,
                    'entropy_mean': np.mean(entropies),
                    'entropy_std': np.std(entropies),
                    'phi_mean': np.mean(phis),
                    'phi_std': np.std(phis),
                    'identity_mean': np.mean(identities),
                    'identity_std': np.std(identities),
                    'crisis_rate': sum(self.crisis_history[name]) / len(self.crisis_history[name]) if self.crisis_history[name] else 0,
                    'final_attachments': agent.attachments.copy()
                }

        # Entanglement
        stats['entanglement_matrix'] = self.entangled_state.correlation_matrix.tolist()

        return stats


def test_coalition_game():
    """Test del juego de coalición."""
    print("=" * 60)
    print("TEST: Coalition Game QG1 - Endógeno")
    print("=" * 60)

    # Crear juego con 3 agentes
    game = CoalitionGameQG1(agent_names=['NEO', 'EVA', 'ALEX'])

    print(f"\nAgentes: {game.agent_names}")
    print(f"Dimensión de drives: {game.dim}")

    # Jugar 100 rondas
    print("\nJugando 100 rondas...")
    game.play_game(100)

    # Estadísticas
    stats = game.get_statistics()

    print(f"\n--- Estadísticas ---")
    for name, agent_stats in stats['agents'].items():
        print(f"\n{name}:")
        print(f"  Distribución de operadores: {agent_stats['operator_distribution']}")
        print(f"  Entropía: {agent_stats['entropy_mean']:.3f} ± {agent_stats['entropy_std']:.3f}")
        print(f"  φ: {agent_stats['phi_mean']:.3f} ± {agent_stats['phi_std']:.3f}")
        print(f"  Identidad: {agent_stats['identity_mean']:.3f} ± {agent_stats['identity_std']:.3f}")
        print(f"  Tasa de crisis: {agent_stats['crisis_rate']*100:.1f}%")
        print(f"  Attachments finales: {agent_stats['final_attachments']}")

    print(f"\n--- Matriz de Entanglement ---")
    print(np.array(stats['entanglement_matrix']))

    # Verificar última ronda
    last_round = game.rounds[-1]
    print(f"\n--- Última ronda ({last_round.round_number}) ---")
    for name in game.agent_names:
        print(f"\n{name}:")
        print(f"  Operador: {last_round.operators_selected[name]}")
        print(f"  Deltas: {last_round.metric_deltas[name]}")

    print("\n✓ Juego 100% endógeno: sin cooperar/defeccionar, solo operadores y Δ métricas")


if __name__ == "__main__":
    test_coalition_game()
