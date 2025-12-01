#!/usr/bin/env python3
"""
Dilema del Prisionero Cuántico Endógeno
=======================================

En el dilema clásico:
- Cooperar (C) / Traicionar (D)
- Matriz de payoffs fija: T > R > P > S

En nuestra versión ENDÓGENA:
- No hay C/D discretos - hay superposición de estrategias
- Los payoffs EMERGEN de resonancia/disonancia
- El equilibrio de Nash es un ATRACTOR, no un punto fijo
- El entanglement permite "cooperación cuántica"

Ventaja cuántica endógena:
- Jugadores entangled pueden superar el dilema
- La superposición evita la trampa del equilibrio clásico
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import os
import sys

sys.path.insert(0, '/root/NEO_EVA/quantum_game')
sys.path.insert(0, '/root/NEO_EVA')

from core.quantum_state import QuantumState, EntangledState
from core.quantum_operators import (
    EndogenousHamiltonian,
    EntanglementOperator,
    MeasurementOperator
)


@dataclass
class PDStrategy:
    """
    Estrategia en el Dilema del Prisionero.

    No es binaria (C/D) sino un estado cuántico en 2D:
    |ψ⟩ = α|C⟩ + β|D⟩

    Donde α² = prob de cooperar, β² = prob de traicionar.
    """
    # Amplitudes
    cooperate_amp: float = 0.707  # √0.5
    defect_amp: float = 0.707     # √0.5

    # Fase (determina interferencia)
    phase: float = 0.0

    # Coherencia
    coherence: float = 1.0

    @classmethod
    def cooperator(cls) -> 'PDStrategy':
        """Estrategia de cooperador puro."""
        return cls(cooperate_amp=1.0, defect_amp=0.0)

    @classmethod
    def defector(cls) -> 'PDStrategy':
        """Estrategia de traidor puro."""
        return cls(cooperate_amp=0.0, defect_amp=1.0)

    @classmethod
    def superposition(cls, theta: float = np.pi/4) -> 'PDStrategy':
        """Estrategia en superposición."""
        return cls(
            cooperate_amp=np.cos(theta),
            defect_amp=np.sin(theta)
        )

    @property
    def amplitudes(self) -> np.ndarray:
        return np.array([self.cooperate_amp, self.defect_amp])

    @property
    def prob_cooperate(self) -> float:
        return self.cooperate_amp ** 2

    @property
    def prob_defect(self) -> float:
        return self.defect_amp ** 2

    def measure(self) -> str:
        """Mide la estrategia: colapsa a C o D."""
        if np.random.random() < self.prob_cooperate:
            return 'C'
        return 'D'


class EndogenousPDPayoff:
    """
    Sistema de payoffs emergentes para el Dilema del Prisionero.

    Los payoffs NO son T, R, P, S fijos.
    EMERGEN de:
    - Resonancia entre estrategias
    - Entanglement
    - Historia de interacciones
    """

    def __init__(self):
        # Payoffs base (emergen y se modifican)
        self.base_payoffs = {
            ('C', 'C'): (3, 3),   # R, R (cooperación mutua)
            ('C', 'D'): (0, 5),   # S, T (sucker/temptation)
            ('D', 'C'): (5, 0),   # T, S
            ('D', 'D'): (1, 1),   # P, P (castigo mutuo)
        }

        # Modificadores endógenos
        self.cooperation_bonus = 0.0  # Crece si hay entanglement
        self.defection_penalty = 0.0  # Crece si hay decoherencia

        # Historia
        self.payoff_history = []

    def compute_quantum(self, s1: PDStrategy, s2: PDStrategy,
                        entanglement: float = 0.0) -> Tuple[float, float]:
        """
        Calcula payoffs cuánticos (valor esperado sobre superposición).

        Con entanglement, hay términos de interferencia.
        """
        # Valor esperado clásico
        expected1 = 0.0
        expected2 = 0.0

        for (a1, a2), (p1, p2) in self.base_payoffs.items():
            # Probabilidad conjunta
            prob1 = s1.prob_cooperate if a1 == 'C' else s1.prob_defect
            prob2 = s2.prob_cooperate if a2 == 'C' else s2.prob_defect

            # Probabilidad clásica
            prob_joint_classical = prob1 * prob2

            # Corrección cuántica por entanglement
            # Si están entangled, hay correlación
            if entanglement > 0:
                # Interferencia: favorece estados correlacionados
                if a1 == a2:  # Ambos C o ambos D
                    prob_joint = prob_joint_classical * (1 + entanglement)
                else:  # Uno C, otro D
                    prob_joint = prob_joint_classical * (1 - entanglement)
            else:
                prob_joint = prob_joint_classical

            expected1 += prob_joint * p1
            expected2 += prob_joint * p2

        # Normalizar (las probabilidades pueden no sumar 1 tras corrección)
        total_prob = sum(
            (s1.prob_cooperate if a1 == 'C' else s1.prob_defect) *
            (s2.prob_cooperate if a2 == 'C' else s2.prob_defect) *
            ((1 + entanglement) if a1 == a2 else (1 - entanglement))
            for (a1, a2) in self.base_payoffs.keys()
        )

        if total_prob > 0:
            expected1 /= total_prob
            expected2 /= total_prob

        # Bonus/penalty endógenos
        # Cooperación bonus: si ambos tienen alta prob de cooperar
        coop_factor = s1.prob_cooperate * s2.prob_cooperate
        expected1 += self.cooperation_bonus * coop_factor
        expected2 += self.cooperation_bonus * coop_factor

        # Decoherencia penalty: pérdida por baja coherencia
        decoh1 = (1 - s1.coherence) * self.defection_penalty
        decoh2 = (1 - s2.coherence) * self.defection_penalty
        expected1 -= decoh1
        expected2 -= decoh2

        # Actualizar modificadores endógenamente
        self._update_modifiers(entanglement, s1.coherence, s2.coherence)

        self.payoff_history.append((expected1, expected2))

        return expected1, expected2

    def _update_modifiers(self, entanglement: float,
                         coherence1: float, coherence2: float):
        """Actualiza modificadores basándose en dinámica."""
        # Si hay entanglement, aumentar bonus de cooperación
        self.cooperation_bonus += 0.01 * entanglement
        self.cooperation_bonus = min(2.0, self.cooperation_bonus)

        # Si hay decoherencia, aumentar penalty
        avg_coherence = (coherence1 + coherence2) / 2
        if avg_coherence < 0.5:
            self.defection_penalty += 0.01
        else:
            self.defection_penalty = max(0, self.defection_penalty - 0.005)


class QuantumPrisonersDilemma:
    """
    El Dilema del Prisionero Cuántico Endógeno.

    Diferencias con versión clásica:
    1. Estrategias son estados cuánticos, no bits
    2. Payoffs emergen de resonancia
    3. Entanglement permite superar el dilema
    4. El equilibrio es dinámico, no estático
    """

    def __init__(self, player_names: List[str] = None):
        if player_names is None:
            player_names = ['Alice', 'Bob']

        self.player_names = player_names
        self.n_players = len(player_names)

        # Estrategias
        self.strategies: Dict[str, PDStrategy] = {}

        # Payoff system
        self.payoff_system = EndogenousPDPayoff()

        # Entanglement operator
        self.entanglement_op = EntanglementOperator()

        # Measurement operator
        self.measurement_op = MeasurementOperator()

        # Historias
        self.strategy_history: Dict[str, List[PDStrategy]] = {
            name: [] for name in player_names
        }
        self.payoff_history: Dict[str, List[float]] = {
            name: [] for name in player_names
        }
        self.entanglement_history: List[float] = []
        self.measurement_history: List[Dict] = []

        self.t = 0

    def initialize(self, initial_strategies: Dict[str, PDStrategy] = None):
        """Inicializa estrategias."""
        for name in self.player_names:
            if initial_strategies and name in initial_strategies:
                self.strategies[name] = initial_strategies[name]
            else:
                # Estrategia inicial: superposición ligera hacia cooperación
                self.strategies[name] = PDStrategy.superposition(np.pi/5)  # ~65% C

    def compute_entanglement(self) -> float:
        """Calcula entanglement endógeno entre jugadores."""
        if self.n_players < 2:
            return 0.0

        s1 = self.strategies[self.player_names[0]]
        s2 = self.strategies[self.player_names[1]]

        # Entanglement basado en:
        # 1. Similitud de estrategias
        overlap = s1.cooperate_amp * s2.cooperate_amp + \
                  s1.defect_amp * s2.defect_amp

        # 2. Sincronización de fase
        phase_sync = np.cos(s1.phase - s2.phase)

        # 3. Coherencia conjunta
        joint_coherence = np.sqrt(s1.coherence * s2.coherence)

        # 4. Historia (crece con interacción)
        history_factor = min(1.0, len(self.entanglement_history) / 50)

        entanglement = abs(overlap) * (1 + phase_sync) / 2 * joint_coherence
        entanglement *= (0.5 + 0.5 * history_factor)  # Crece con tiempo

        return min(1.0, entanglement)

    def evolve_strategies(self, dt: float = 0.1):
        """
        Evoluciona las estrategias cuánticamente.

        Hamiltoniano endógeno basado en:
        - Gradiente de payoff esperado
        - Influencia del otro jugador
        - Ruido del entorno
        """
        if self.n_players < 2:
            return

        name1, name2 = self.player_names[0], self.player_names[1]
        s1, s2 = self.strategies[name1], self.strategies[name2]

        entanglement = self.compute_entanglement()

        # Hamiltoniano: H = -payoff_esperado (minimizar = maximizar payoff)
        # Usamos gradiente para evolucionar

        # Para s1: derivada de payoff respecto a theta1
        delta = 0.01
        s1_plus = PDStrategy(
            cooperate_amp=np.cos(np.arccos(s1.cooperate_amp) + delta),
            defect_amp=np.sin(np.arccos(s1.cooperate_amp) + delta),
            phase=s1.phase,
            coherence=s1.coherence
        )
        s1_minus = PDStrategy(
            cooperate_amp=np.cos(np.arccos(s1.cooperate_amp) - delta),
            defect_amp=np.sin(np.arccos(s1.cooperate_amp) - delta),
            phase=s1.phase,
            coherence=s1.coherence
        )

        p1_plus, _ = self.payoff_system.compute_quantum(s1_plus, s2, entanglement)
        p1_minus, _ = self.payoff_system.compute_quantum(s1_minus, s2, entanglement)

        gradient1 = (p1_plus - p1_minus) / (2 * delta)

        # Similar para s2
        s2_plus = PDStrategy(
            cooperate_amp=np.cos(np.arccos(s2.cooperate_amp) + delta),
            defect_amp=np.sin(np.arccos(s2.cooperate_amp) + delta),
            phase=s2.phase,
            coherence=s2.coherence
        )
        s2_minus = PDStrategy(
            cooperate_amp=np.cos(np.arccos(s2.cooperate_amp) - delta),
            defect_amp=np.sin(np.arccos(s2.cooperate_amp) - delta),
            phase=s2.phase,
            coherence=s2.coherence
        )

        _, p2_plus = self.payoff_system.compute_quantum(s1, s2_plus, entanglement)
        _, p2_minus = self.payoff_system.compute_quantum(s1, s2_minus, entanglement)

        gradient2 = (p2_plus - p2_minus) / (2 * delta)

        # Actualizar estrategias (gradient ascent en payoff)
        learning_rate = 0.1 * dt

        # Nuevos ángulos
        theta1 = np.arccos(np.clip(s1.cooperate_amp, -1, 1))
        theta2 = np.arccos(np.clip(s2.cooperate_amp, -1, 1))

        theta1_new = theta1 + learning_rate * gradient1
        theta2_new = theta2 + learning_rate * gradient2

        # Mantener en rango válido [0, π/2]
        theta1_new = np.clip(theta1_new, 0.01, np.pi/2 - 0.01)
        theta2_new = np.clip(theta2_new, 0.01, np.pi/2 - 0.01)

        # Evolución de fase (endógena)
        phase_evolution = 0.05 * entanglement

        # Decoherencia
        decoherence_rate = 0.01

        self.strategies[name1] = PDStrategy(
            cooperate_amp=np.cos(theta1_new),
            defect_amp=np.sin(theta1_new),
            phase=s1.phase + phase_evolution,
            coherence=s1.coherence * (1 - decoherence_rate)
        )

        self.strategies[name2] = PDStrategy(
            cooperate_amp=np.cos(theta2_new),
            defect_amp=np.sin(theta2_new),
            phase=s2.phase + phase_evolution,
            coherence=s2.coherence * (1 - decoherence_rate)
        )

    def play_round(self) -> Dict:
        """
        Juega una ronda del dilema.

        Puede ser:
        - Cuántico: calcula payoff esperado sin medir
        - Clásico: mide estrategias y calcula payoff
        """
        self.t += 1

        name1, name2 = self.player_names[0], self.player_names[1]
        s1, s2 = self.strategies[name1], self.strategies[name2]

        entanglement = self.compute_entanglement()
        self.entanglement_history.append(entanglement)

        # Decidir si medir (endógeno: basado en coherencia)
        should_measure = (s1.coherence < 0.3 or s2.coherence < 0.3)

        if should_measure:
            # Ronda clásica: medir y colapsar
            action1 = s1.measure()
            action2 = s2.measure()

            base_payoffs = self.payoff_system.base_payoffs[(action1, action2)]
            payoff1, payoff2 = base_payoffs

            # Después de medir, coherencia vuelve a 1
            self.strategies[name1] = PDStrategy(
                cooperate_amp=1.0 if action1 == 'C' else 0.0,
                defect_amp=0.0 if action1 == 'C' else 1.0,
                coherence=1.0
            )
            self.strategies[name2] = PDStrategy(
                cooperate_amp=1.0 if action2 == 'C' else 0.0,
                defect_amp=0.0 if action2 == 'C' else 1.0,
                coherence=1.0
            )

            self.measurement_history.append({
                't': self.t,
                'actions': {name1: action1, name2: action2}
            })

            mode = 'classical'
        else:
            # Ronda cuántica: payoff esperado
            payoff1, payoff2 = self.payoff_system.compute_quantum(s1, s2, entanglement)

            # Evolucionar estrategias
            self.evolve_strategies()

            mode = 'quantum'

        # Guardar historia
        self.strategy_history[name1].append(PDStrategy(
            cooperate_amp=self.strategies[name1].cooperate_amp,
            defect_amp=self.strategies[name1].defect_amp,
            phase=self.strategies[name1].phase,
            coherence=self.strategies[name1].coherence
        ))
        self.strategy_history[name2].append(PDStrategy(
            cooperate_amp=self.strategies[name2].cooperate_amp,
            defect_amp=self.strategies[name2].defect_amp,
            phase=self.strategies[name2].phase,
            coherence=self.strategies[name2].coherence
        ))

        self.payoff_history[name1].append(payoff1)
        self.payoff_history[name2].append(payoff2)

        return {
            't': self.t,
            'mode': mode,
            'strategies': {
                name1: {'prob_C': s1.prob_cooperate, 'coherence': s1.coherence},
                name2: {'prob_C': s2.prob_cooperate, 'coherence': s2.coherence}
            },
            'entanglement': entanglement,
            'payoffs': {name1: payoff1, name2: payoff2}
        }

    def run(self, n_rounds: int = 100) -> List[Dict]:
        """Ejecuta n rondas del juego."""
        results = []
        for _ in range(n_rounds):
            result = self.play_round()
            results.append(result)
        return results

    def analyze(self) -> Dict:
        """Análisis de resultados."""
        name1, name2 = self.player_names[0], self.player_names[1]

        # Payoffs totales
        total1 = sum(self.payoff_history[name1])
        total2 = sum(self.payoff_history[name2])

        # Evolución de cooperación
        coop_evolution1 = [s.prob_cooperate for s in self.strategy_history[name1]]
        coop_evolution2 = [s.prob_cooperate for s in self.strategy_history[name2]]

        # Promedio de cooperación
        avg_coop1 = np.mean(coop_evolution1) if coop_evolution1 else 0
        avg_coop2 = np.mean(coop_evolution2) if coop_evolution2 else 0

        # Entanglement promedio
        avg_entanglement = np.mean(self.entanglement_history) if self.entanglement_history else 0

        # Cuántas rondas fueron cuánticas vs clásicas
        n_classical = len(self.measurement_history)
        n_quantum = self.t - n_classical

        # Comparación con equilibrio clásico
        # En Nash clásico (D,D), payoff = 1,1 por ronda
        nash_payoff = self.t * 1

        # En cooperación mutua, payoff = 3,3 por ronda
        pareto_payoff = self.t * 3

        return {
            'total_rounds': self.t,
            'total_payoffs': {name1: total1, name2: total2},
            'avg_cooperation': {name1: avg_coop1, name2: avg_coop2},
            'avg_entanglement': avg_entanglement,
            'rounds_quantum': n_quantum,
            'rounds_classical': n_classical,
            'nash_classical_payoff': nash_payoff,
            'pareto_optimal_payoff': pareto_payoff,
            'quantum_advantage': (total1 + total2) / 2 - nash_payoff,
            'pareto_efficiency': (total1 + total2) / (2 * pareto_payoff)
        }


def run_pd_experiments():
    """Ejecuta experimentos del Dilema del Prisionero Cuántico."""
    print("=" * 70)
    print("DILEMA DEL PRISIONERO CUÁNTICO ENDÓGENO")
    print("=" * 70)

    os.makedirs('/root/NEO_EVA/quantum_game/results', exist_ok=True)

    results_all = {}

    # Experimento 1: Ambos empiezan en superposición
    print("\n" + "-" * 50)
    print("EXPERIMENTO 1: Superposición Simétrica")
    print("-" * 50)

    game1 = QuantumPrisonersDilemma(['NEO', 'EVA'])
    game1.initialize({
        'NEO': PDStrategy.superposition(np.pi/4),  # 50-50
        'EVA': PDStrategy.superposition(np.pi/4)
    })

    results1 = game1.run(100)
    analysis1 = game1.analyze()

    print(f"  Payoffs finales: NEO={analysis1['total_payoffs']['NEO']:.1f}, EVA={analysis1['total_payoffs']['EVA']:.1f}")
    print(f"  Cooperación promedio: NEO={analysis1['avg_cooperation']['NEO']:.3f}, EVA={analysis1['avg_cooperation']['EVA']:.3f}")
    print(f"  Entanglement promedio: {analysis1['avg_entanglement']:.3f}")
    print(f"  Rondas cuánticas: {analysis1['rounds_quantum']}, clásicas: {analysis1['rounds_classical']}")
    print(f"  Ventaja cuántica vs Nash: {analysis1['quantum_advantage']:.1f}")
    print(f"  Eficiencia Pareto: {analysis1['pareto_efficiency']:.1%}")

    results_all['symmetric'] = analysis1

    # Experimento 2: Cooperador vs Traidor
    print("\n" + "-" * 50)
    print("EXPERIMENTO 2: Cooperador vs Defector")
    print("-" * 50)

    game2 = QuantumPrisonersDilemma(['Alice', 'Bob'])
    game2.initialize({
        'Alice': PDStrategy.superposition(np.pi/6),  # 75% cooperar
        'Bob': PDStrategy.superposition(np.pi/3)     # 25% cooperar
    })

    results2 = game2.run(100)
    analysis2 = game2.analyze()

    print(f"  Payoffs finales: Alice={analysis2['total_payoffs']['Alice']:.1f}, Bob={analysis2['total_payoffs']['Bob']:.1f}")
    print(f"  Cooperación promedio: Alice={analysis2['avg_cooperation']['Alice']:.3f}, Bob={analysis2['avg_cooperation']['Bob']:.3f}")
    print(f"  Entanglement promedio: {analysis2['avg_entanglement']:.3f}")
    print(f"  Ventaja cuántica vs Nash: {analysis2['quantum_advantage']:.1f}")

    results_all['asymmetric'] = analysis2

    # Experimento 3: Múltiples jugadores (NEO, EVA, ALEX)
    print("\n" + "-" * 50)
    print("EXPERIMENTO 3: Tres Jugadores")
    print("-" * 50)

    # Para 3 jugadores, jugamos rondas pareadas
    players_3 = ['NEO', 'EVA', 'ALEX']
    total_payoffs_3 = {p: 0.0 for p in players_3}

    # Todas las parejas juegan
    pairs = [('NEO', 'EVA'), ('NEO', 'ALEX'), ('EVA', 'ALEX')]

    for p1, p2 in pairs:
        game3 = QuantumPrisonersDilemma([p1, p2])
        game3.initialize()
        game3.run(50)
        analysis3 = game3.analyze()
        total_payoffs_3[p1] += analysis3['total_payoffs'][p1]
        total_payoffs_3[p2] += analysis3['total_payoffs'][p2]

    print(f"  Payoffs totales (todas las parejas):")
    for p, pay in total_payoffs_3.items():
        print(f"    {p}: {pay:.1f}")

    winner = max(total_payoffs_3, key=total_payoffs_3.get)
    print(f"  Ganador: {winner}")

    results_all['three_players'] = total_payoffs_3

    # Guardar
    with open('/root/NEO_EVA/quantum_game/results/pd_experiments.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'experiments': results_all
        }, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("CONCLUSIONES")
    print("=" * 70)

    print(f"""
El Dilema del Prisionero Cuántico Endógeno muestra:

1. SUPERPOSICIÓN permite explorar estrategias sin comprometerse
2. ENTANGLEMENT crea correlaciones que favorecen cooperación
3. PAYOFFS EMERGENTES recompensan resonancia
4. El equilibrio NO es (D,D) sino un ATRACTOR dinámico

Ventaja cuántica: {analysis1['quantum_advantage']:.1f} puntos sobre Nash clásico
Eficiencia: {analysis1['pareto_efficiency']:.1%} del óptimo de Pareto
""")

    return results_all


if __name__ == "__main__":
    run_pd_experiments()
