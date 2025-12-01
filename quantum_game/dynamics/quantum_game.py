#!/usr/bin/env python3
"""
Juego Cuántico Endógeno
=======================

Un juego donde:
- Estados = superposiciones de estrategias (drives)
- Payoffs = emergen de resonancia/disonancia
- Estrategias = no se eligen, evolucionan
- Equilibrios = atractores cuánticos

NO HAY:
- Matriz de payoffs predefinida
- Estrategias discretas fijas
- Nash equilibrium clásico

TODO EMERGE de la dinámica cuántica del sistema.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import sys

sys.path.insert(0, '/root/NEO_EVA/quantum_game')

from core.quantum_state import QuantumState, EntangledState
from core.quantum_operators import (
    EndogenousHamiltonian,
    EntanglementOperator,
    MeasurementOperator,
    DecoherenceOperator
)


class GamePhase(Enum):
    """Fases del juego cuántico."""
    SUPERPOSITION = "superposition"  # Estrategias en superposición
    ENTANGLEMENT = "entanglement"    # Jugadores entangled
    MEASUREMENT = "measurement"       # Colapso a estrategia concreta
    PAYOFF = "payoff"                # Cálculo de payoff emergente


@dataclass
class QuantumPlayer:
    """
    Jugador cuántico cuya estrategia es un estado cuántico.
    """
    name: str
    dim: int = 6  # Dimensión del espacio de estrategias

    # Estado cuántico actual
    state: QuantumState = None

    # Historia
    strategy_history: List[np.ndarray] = field(default_factory=list)
    payoff_history: List[float] = field(default_factory=list)
    measurement_history: List[int] = field(default_factory=list)

    # Operadores propios
    hamiltonian: EndogenousHamiltonian = None
    decoherence: DecoherenceOperator = None

    def __post_init__(self):
        # Estado inicial: superposición uniforme
        if self.state is None:
            uniform = np.ones(self.dim) / self.dim
            self.state = QuantumState.from_drives(uniform)

        if self.hamiltonian is None:
            self.hamiltonian = EndogenousHamiltonian(self.dim)

        if self.decoherence is None:
            self.decoherence = DecoherenceOperator()

    def set_initial_strategy(self, strategy: np.ndarray):
        """Establece estrategia inicial (distribución sobre opciones)."""
        strategy = np.clip(strategy, 1e-10, None)
        strategy = strategy / strategy.sum()
        self.state = QuantumState.from_drives(strategy, self.strategy_history)
        self.strategy_history.append(strategy)

    def evolve(self, other_player: 'QuantumPlayer' = None,
               stimulus: np.ndarray = None, dt: float = 0.1):
        """
        Evoluciona el estado del jugador.

        La evolución es UNITARIA (preserva norma).
        """
        other_state = other_player.state if other_player else None

        # Obtener operador de evolución
        U = self.hamiltonian.evolution_operator(
            self.state, dt, other_state, stimulus
        )

        # Evolucionar
        new_amplitudes = U @ self.state.amplitudes
        new_amplitudes = np.abs(new_amplitudes)
        new_amplitudes = new_amplitudes / np.linalg.norm(new_amplitudes)

        # Actualizar estado
        self.state = QuantumState(
            amplitudes=new_amplitudes,
            phase=self.state.phase + dt,  # Fase acumula
            coherence=self.state.coherence,
            amplitude_history=self.state.amplitude_history + [new_amplitudes]
        )

        # Aplicar decoherencia
        self.state = self.decoherence.apply(self.state)

        self.strategy_history.append(self.state.probabilities)

    def measure(self) -> int:
        """
        Mide el estado: colapsa a una estrategia concreta.

        Retorna el índice de la estrategia medida.
        """
        probs = self.state.probabilities
        result = np.random.choice(self.dim, p=probs)

        # Colapsar estado
        collapsed_amp = np.zeros(self.dim)
        collapsed_amp[result] = 1.0

        self.state = QuantumState(
            amplitudes=collapsed_amp,
            phase=self.state.phase,
            coherence=1.0  # Estado puro post-medición
        )

        self.measurement_history.append(result)
        return result

    @property
    def expected_strategy(self) -> np.ndarray:
        """Valor esperado de la estrategia (sin colapsar)."""
        return self.state.probabilities


@dataclass
class QuantumPayoff:
    """
    Sistema de payoffs emergentes.

    Los payoffs NO son una matriz fija.
    EMERGEN de:
    - Resonancia entre estrategias
    - Entanglement
    - Historia de interacciones
    """

    # Pesos de contribución (emergen con el tiempo)
    resonance_weight: float = 1.0
    entanglement_weight: float = 0.5
    history_weight: float = 0.3

    # Historia para aprendizaje
    payoff_history: Dict[str, List[float]] = field(default_factory=dict)

    def compute(self, player1: QuantumPlayer, player2: QuantumPlayer,
                entanglement: float = 0.0) -> Tuple[float, float]:
        """
        Calcula payoffs emergentes para ambos jugadores.

        Payoff = resonancia + bonus_entanglement - costo_decoherencia
        """
        s1 = player1.state
        s2 = player2.state

        # 1. Resonancia: qué tan "alineados" están
        fidelity = s1.fidelity(s2)

        # 2. Bonus por entanglement (cooperación cuántica)
        ent_bonus = entanglement * self.entanglement_weight

        # 3. Costo de decoherencia (pérdida de coherencia)
        decoh_cost1 = (1 - s1.coherence) * 0.1
        decoh_cost2 = (1 - s2.coherence) * 0.1

        # 4. Componente de suma cero parcial (competencia)
        # Diferencia de entropías: menor entropía = más "decidido"
        entropy_diff = s1.entropy - s2.entropy
        competition1 = -0.1 * entropy_diff
        competition2 = 0.1 * entropy_diff

        # Payoffs base
        base_payoff = fidelity * self.resonance_weight

        payoff1 = base_payoff + ent_bonus - decoh_cost1 + competition1
        payoff2 = base_payoff + ent_bonus - decoh_cost2 + competition2

        # Guardar historia
        if player1.name not in self.payoff_history:
            self.payoff_history[player1.name] = []
        if player2.name not in self.payoff_history:
            self.payoff_history[player2.name] = []

        self.payoff_history[player1.name].append(payoff1)
        self.payoff_history[player2.name].append(payoff2)

        # Actualizar pesos endógenamente
        self._update_weights()

        return payoff1, payoff2

    def _update_weights(self):
        """Actualiza pesos de forma endógena basándose en la historia."""
        if len(self.payoff_history) < 2:
            return

        # Si los payoffs son muy desiguales, aumentar resonancia
        all_payoffs = []
        for p in self.payoff_history.values():
            all_payoffs.extend(p[-20:])

        if len(all_payoffs) > 10:
            variance = np.var(all_payoffs)
            # Alta varianza → aumentar peso de resonancia (favorece cooperación)
            if variance > 0.1:
                self.resonance_weight = min(2.0, self.resonance_weight + 0.01)
            else:
                self.resonance_weight = max(0.5, self.resonance_weight - 0.005)


class QuantumGame:
    """
    El Juego Cuántico principal.

    Estructura:
    1. Jugadores en superposición de estrategias
    2. Entanglement emerge de interacción
    3. El juego "mide" periódicamente (crisis = colapso)
    4. Payoffs emergen de resonancia

    NO hay:
    - Turnos discretos fijos
    - Matriz de pagos predefinida
    - Estrategias puras obligatorias
    """

    STRATEGY_NAMES = ['cooperate', 'defect', 'tit-for-tat', 'random', 'generous', 'grudge']

    def __init__(self, player_names: List[str], dim: int = 6):
        self.dim = dim
        self.t = 0

        # Crear jugadores
        self.players = {
            name: QuantumPlayer(name, dim)
            for name in player_names
        }

        # Operadores
        self.entanglement_op = EntanglementOperator()
        self.measurement_op = MeasurementOperator()

        # Sistema de payoffs
        self.payoff_system = QuantumPayoff()

        # Estado entangled del sistema completo
        self.entangled_state: EntangledState = None

        # Historia del juego
        self.game_history: List[Dict] = []
        self.phase_history: List[GamePhase] = []

    def initialize(self, initial_strategies: Dict[str, np.ndarray] = None):
        """Inicializa el juego con estrategias iniciales."""
        for name, player in self.players.items():
            if initial_strategies and name in initial_strategies:
                player.set_initial_strategy(initial_strategies[name])
            else:
                # Estrategia inicial endógena: leve sesgo hacia cooperación
                strategy = np.array([0.25, 0.15, 0.2, 0.15, 0.15, 0.1])
                strategy = strategy / strategy.sum()
                player.set_initial_strategy(strategy)

        self._update_entangled_state()

    def _update_entangled_state(self):
        """Actualiza el estado entangled del sistema."""
        agent_states = {
            name: player.state
            for name, player in self.players.items()
        }

        # Construir historia de interacciones
        interaction_history = []
        min_len = min(len(p.strategy_history) for p in self.players.values())
        for i in range(min_len):
            step_data = {
                name: player.strategy_history[i]
                for name, player in self.players.items()
            }
            interaction_history.append(step_data)

        self.entangled_state = EntangledState.from_agents(
            agent_states, interaction_history
        )

    def step(self, stimulus: np.ndarray = None) -> Dict:
        """
        Un paso del juego cuántico.

        Retorna información sobre lo que pasó.
        """
        self.t += 1
        result = {'t': self.t, 'phase': None, 'events': []}

        # 1. Determinar fase actual
        phase = self._determine_phase()
        self.phase_history.append(phase)
        result['phase'] = phase

        # 2. Ejecutar según fase
        if phase == GamePhase.SUPERPOSITION:
            result['events'].append("Players evolving in superposition")
            self._evolution_step(stimulus)

        elif phase == GamePhase.ENTANGLEMENT:
            result['events'].append("Entanglement forming/strengthening")
            self._entanglement_step()

        elif phase == GamePhase.MEASUREMENT:
            result['events'].append("Measurement/collapse occurring")
            measurements = self._measurement_step()
            result['measurements'] = measurements

        elif phase == GamePhase.PAYOFF:
            result['events'].append("Payoffs being calculated")
            payoffs = self._payoff_step()
            result['payoffs'] = payoffs

        # 3. Actualizar estado entangled
        self._update_entangled_state()

        # 4. Registrar
        result['player_states'] = {
            name: {
                'strategy': player.expected_strategy.tolist(),
                'coherence': player.state.coherence,
                'entropy': player.state.entropy
            }
            for name, player in self.players.items()
        }

        self.game_history.append(result)
        return result

    def _determine_phase(self) -> GamePhase:
        """
        Determina la fase actual del juego.

        Endógeno: basado en coherencia promedio del sistema.
        """
        avg_coherence = np.mean([p.state.coherence for p in self.players.values()])

        # Umbrales endógenos
        if len(self.phase_history) < 20:
            coherence_threshold = 0.3
        else:
            # Umbral = percentil 20 de coherencia histórica
            recent_coherences = []
            for h in self.game_history[-20:]:
                for ps in h.get('player_states', {}).values():
                    recent_coherences.append(ps.get('coherence', 0.5))
            coherence_threshold = np.percentile(recent_coherences, 20) if recent_coherences else 0.3

        if avg_coherence < coherence_threshold:
            return GamePhase.MEASUREMENT
        elif self.t % 5 == 0:  # Cada 5 pasos calculamos payoff
            return GamePhase.PAYOFF
        elif avg_coherence > 0.7:
            return GamePhase.ENTANGLEMENT
        else:
            return GamePhase.SUPERPOSITION

    def _evolution_step(self, stimulus: np.ndarray = None):
        """Paso de evolución: todos los jugadores evolucionan."""
        player_list = list(self.players.values())

        for i, player in enumerate(player_list):
            # Cada jugador ve al "otro promedio"
            others = [p for j, p in enumerate(player_list) if j != i]
            if others:
                other = others[0]  # Simplificación: ve al primero
            else:
                other = None

            player.evolve(other, stimulus)

    def _entanglement_step(self):
        """Paso de entanglement: correlaciones se fortalecen."""
        player_list = list(self.players.values())

        for i in range(len(player_list)):
            for j in range(i+1, len(player_list)):
                p1, p2 = player_list[i], player_list[j]

                # Calcular fuerza de entanglement
                strength = self.entanglement_op.compute_entangling_strength(
                    p1.state, p2.state
                )

                # Si hay resonancia, aumentar coherencia mutua
                if strength > 0.5:
                    p1.state = QuantumState(
                        amplitudes=p1.state.amplitudes,
                        phase=p1.state.phase,
                        coherence=min(1.0, p1.state.coherence + 0.05)
                    )
                    p2.state = QuantumState(
                        amplitudes=p2.state.amplitudes,
                        phase=p2.state.phase,
                        coherence=min(1.0, p2.state.coherence + 0.05)
                    )

    def _measurement_step(self) -> Dict[str, int]:
        """Paso de medición: jugadores con baja coherencia colapsan."""
        measurements = {}

        for name, player in self.players.items():
            if self.measurement_op.should_collapse(player.state):
                result = player.measure()
                measurements[name] = result

        return measurements

    def _payoff_step(self) -> Dict[str, float]:
        """Paso de payoff: calcula pagos emergentes."""
        payoffs = {name: 0.0 for name in self.players}

        player_list = list(self.players.items())

        for i in range(len(player_list)):
            for j in range(i+1, len(player_list)):
                name1, p1 = player_list[i]
                name2, p2 = player_list[j]

                # Entanglement entre este par
                ent = self.entangled_state.entanglement_measure(name1, name2)

                # Payoffs
                pay1, pay2 = self.payoff_system.compute(p1, p2, ent)

                payoffs[name1] += pay1
                payoffs[name2] += pay2

                # Guardar en historial del jugador
                p1.payoff_history.append(pay1)
                p2.payoff_history.append(pay2)

        return payoffs

    def run(self, n_steps: int = 100, stimulus_fn=None) -> List[Dict]:
        """
        Ejecuta el juego por n pasos.

        stimulus_fn: función que genera estímulo en cada paso
        """
        results = []

        for t in range(n_steps):
            stimulus = stimulus_fn(t) if stimulus_fn else None
            result = self.step(stimulus)
            results.append(result)

        return results

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas del juego."""
        total_payoffs = {name: sum(p.payoff_history) for name, p in self.players.items()}

        phase_counts = {}
        for phase in self.phase_history:
            phase_counts[phase.value] = phase_counts.get(phase.value, 0) + 1

        # Estrategia final promedio
        final_strategies = {
            name: player.expected_strategy
            for name, player in self.players.items()
        }

        # Entanglement final
        final_entanglement = {}
        if self.entangled_state:
            names = list(self.players.keys())
            for i, n1 in enumerate(names):
                for j, n2 in enumerate(names):
                    if i < j:
                        ent = self.entangled_state.entanglement_measure(n1, n2)
                        final_entanglement[f"{n1}-{n2}"] = ent

        return {
            'total_steps': self.t,
            'total_payoffs': total_payoffs,
            'phase_distribution': phase_counts,
            'final_strategies': {k: v.tolist() for k, v in final_strategies.items()},
            'final_entanglement': final_entanglement,
            'winner': max(total_payoffs, key=total_payoffs.get)
        }


def test_quantum_game():
    """Test del juego cuántico."""
    print("=" * 60)
    print("TEST: Juego Cuántico Endógeno")
    print("=" * 60)

    # Crear juego con 3 jugadores
    game = QuantumGame(['NEO', 'EVA', 'ALEX'])

    # Estrategias iniciales diferentes
    initial = {
        'NEO': np.array([0.3, 0.1, 0.2, 0.15, 0.15, 0.1]),   # Sesgo cooperativo
        'EVA': np.array([0.15, 0.25, 0.15, 0.2, 0.15, 0.1]), # Sesgo competitivo
        'ALEX': np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])     # Balanceado
    }

    game.initialize(initial)

    print("\nEstrategias iniciales:")
    for name, player in game.players.items():
        print(f"  {name}: {player.expected_strategy}")

    # Ejecutar juego
    print("\nEjecutando 50 pasos...")

    def random_stimulus(t):
        return np.random.dirichlet(np.ones(6))

    results = game.run(50, random_stimulus)

    # Mostrar algunos eventos
    print("\nEventos seleccionados:")
    for r in results[::10]:
        print(f"  t={r['t']}: {r['phase'].value} - {r['events']}")

    # Estadísticas
    stats = game.get_statistics()

    print("\n" + "=" * 60)
    print("ESTADÍSTICAS FINALES")
    print("=" * 60)

    print(f"\nPayoffs totales:")
    for name, payoff in stats['total_payoffs'].items():
        print(f"  {name}: {payoff:.3f}")

    print(f"\nGanador: {stats['winner']}")

    print(f"\nDistribución de fases:")
    for phase, count in stats['phase_distribution'].items():
        print(f"  {phase}: {count}")

    print(f"\nEstrategias finales:")
    for name, strategy in stats['final_strategies'].items():
        dominant = game.STRATEGY_NAMES[np.argmax(strategy)]
        print(f"  {name}: {dominant} ({strategy})")

    print(f"\nEntanglement final:")
    for pair, ent in stats['final_entanglement'].items():
        print(f"  {pair}: {ent:.3f}")


if __name__ == "__main__":
    test_quantum_game()
