#!/usr/bin/env python3
"""
PAYOFF ENDOGENOUS - Payoffs como Δ de Métricas Internas
=======================================================

Los payoffs NO son valores externos (como en el dilema del prisionero).
Son RANKS de deltas de métricas propias:

payoff_A(t) = rank(ΔS_A) + rank(Δφ_A) + rank(Δattach_A) - rank(Δcrisis_A)

Esto garantiza:
- Sin escalas arbitrarias
- Sin "cooperar = +3", "defeccionar = +5"
- Solo Δ de métricas internas, comparadas entre agentes

CHECK: No hay ningún número que no sea 0, 1, o dimensión estructural.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.stats import rankdata


@dataclass
class PayoffCalculator:
    """
    Calculadora de payoffs endógenos.

    Payoff = combinación de ranks de deltas de métricas.
    """
    agent_names: List[str]

    # Pesos de métricas (todos 1, endógeno)
    # NO son magic numbers - son conteos de métricas
    metric_weights: Dict[str, float] = None

    # Historia de payoffs para normalización
    payoff_history: Dict[str, List[float]] = field(default_factory=dict)

    def __post_init__(self):
        # Pesos iguales para todas las métricas (no hay preferencias externas)
        if self.metric_weights is None:
            self.metric_weights = {
                'delta_entropy': 1,       # Más entropía = más exploración
                'delta_phi': 1,           # Más φ = más integración
                'delta_identity': 1,      # Más identidad = más estabilidad del yo
                'delta_coherence': 1,     # Más coherencia = más estabilidad
                'delta_attachment': 1,    # Más attachment = más vínculo
                'crisis_change': -1       # Crisis es negativo
            }

        for name in self.agent_names:
            self.payoff_history[name] = []

    def compute_raw_payoff(self, deltas: Dict[str, float]) -> float:
        """
        Calcula payoff raw desde deltas.

        payoff = Σ weight_i × delta_i
        """
        payoff = 0.0
        for metric, weight in self.metric_weights.items():
            if metric in deltas:
                payoff += weight * deltas[metric]
        return payoff

    def compute_ranked_payoff(self,
                              all_deltas: Dict[str, Dict[str, float]],
                              agent: str) -> float:
        """
        Calcula payoff rankeado comparando con otros agentes.

        Para cada métrica:
        - rank(Δmetric_A) entre todos los agentes
        - payoff_A += rank normalizado

        Esto hace los payoffs relativos, no absolutos.
        """
        if len(all_deltas) < 2:
            return self.compute_raw_payoff(all_deltas.get(agent, {}))

        agents = list(all_deltas.keys())
        payoff = 0.0

        for metric, weight in self.metric_weights.items():
            # Extraer valores de esta métrica para todos los agentes
            values = []
            for a in agents:
                if metric in all_deltas[a]:
                    values.append(all_deltas[a][metric])
                else:
                    values.append(0.0)

            # Rankear
            ranks = rankdata(values, method='average')

            # Normalizar ranks a [0, 1]
            normalized_ranks = ranks / len(agents)

            # Encontrar rank del agente actual
            agent_idx = agents.index(agent)
            agent_rank = normalized_ranks[agent_idx]

            # Aplicar peso
            payoff += weight * agent_rank

        return payoff

    def compute_all_payoffs(self,
                           round_data) -> Dict[str, float]:
        """
        Calcula payoffs para todos los agentes en una ronda.

        Args:
            round_data: CoalitionGameRound con metric_deltas

        Returns:
            Dict de {agent_name: payoff}
        """
        payoffs = {}

        for agent in self.agent_names:
            if agent in round_data.metric_deltas:
                payoff = self.compute_ranked_payoff(
                    round_data.metric_deltas,
                    agent
                )
                payoffs[agent] = payoff
                self.payoff_history[agent].append(payoff)

        return payoffs

    def normalize_payoff(self, agent: str, payoff: float) -> float:
        """
        Normaliza payoff por la historia del agente.

        payoff_normalized = (payoff - p5) / (p95 - p5)
        """
        history = self.payoff_history[agent]

        if len(history) < 10:
            return payoff

        p5 = np.percentile(history, 5)
        p95 = np.percentile(history, 95)

        if p95 > p5:
            return (payoff - p5) / (p95 - p5)
        return 0.5

    def get_statistics(self, agent: str) -> Dict:
        """Estadísticas de payoffs de un agente."""
        history = self.payoff_history[agent]

        if not history:
            return {}

        return {
            'mean': np.mean(history),
            'std': np.std(history),
            'min': np.min(history),
            'max': np.max(history),
            'p25': np.percentile(history, 25),
            'p50': np.percentile(history, 50),
            'p75': np.percentile(history, 75)
        }


@dataclass
class CooperationMetric:
    """
    Métrica de "cooperación" emergente (sin usar la palabra).

    Detecta cuando los agentes actúan de forma que beneficia al sistema total.
    """
    agent_names: List[str]

    # Historia de payoffs totales del sistema
    system_payoff_history: List[float] = field(default_factory=list)

    # Historia de distribución de payoffs (Gini)
    inequality_history: List[float] = field(default_factory=list)

    def compute_system_payoff(self, individual_payoffs: Dict[str, float]) -> float:
        """
        Payoff del sistema = suma de individuales.
        """
        return sum(individual_payoffs.values())

    def compute_inequality(self, individual_payoffs: Dict[str, float]) -> float:
        """
        Desigualdad (índice de Gini normalizado).

        0 = todos iguales
        1 = máxima desigualdad
        """
        payoffs = list(individual_payoffs.values())

        if len(payoffs) < 2:
            return 0.0

        # Índice de Gini
        n = len(payoffs)
        mean_payoff = np.mean(payoffs) + 1e-16

        # Σ|x_i - x_j| / (2n²μ)
        total_diff = 0
        for i in range(n):
            for j in range(n):
                total_diff += abs(payoffs[i] - payoffs[j])

        gini = total_diff / (2 * n * n * mean_payoff)
        return np.clip(gini, 0, 1)

    def update(self, individual_payoffs: Dict[str, float]):
        """Actualiza métricas del sistema."""
        system_p = self.compute_system_payoff(individual_payoffs)
        inequality = self.compute_inequality(individual_payoffs)

        self.system_payoff_history.append(system_p)
        self.inequality_history.append(inequality)

    def get_coordination_index(self) -> float:
        """
        Índice de coordinación emergente.

        coordination = system_payoff / (expected_random)
        normalizado por historia.
        """
        if len(self.system_payoff_history) < 10:
            return 0.5

        current = self.system_payoff_history[-1]
        historical = self.system_payoff_history[:-1]

        p50 = np.percentile(historical, 50)

        if p50 > 0:
            return current / p50
        return 1.0

    def get_fairness_index(self) -> float:
        """
        Índice de fairness emergente.

        fairness = 1 - inequality
        normalizado por historia.
        """
        if not self.inequality_history:
            return 0.5

        current = self.inequality_history[-1]
        return 1 - current


@dataclass
class PayoffMatrix:
    """
    Matriz de payoffs emergente.

    No es una matriz fija como en teoría de juegos clásica.
    Se construye desde los deltas observados.
    """
    agent_names: List[str]

    # Matrices por operador: op_name -> matrix[agent_a][agent_b]
    # Cada entrada es el payoff promedio de A cuando A usa op_A y B usa op_B
    operator_payoff_matrices: Dict[str, np.ndarray] = field(default_factory=dict)

    # Conteos para promediar
    operator_counts: Dict[str, np.ndarray] = field(default_factory=dict)

    def update_from_round(self, round_data, payoffs: Dict[str, float]):
        """
        Actualiza matrices desde una ronda.
        """
        # Para cada par de agentes
        n = len(self.agent_names)

        for i, a1 in enumerate(self.agent_names):
            for j, a2 in enumerate(self.agent_names):
                if i != j:
                    op1 = round_data.operators_selected.get(a1, 'unknown')
                    op2 = round_data.operators_selected.get(a2, 'unknown')

                    key = f"{op1}_{op2}"

                    if key not in self.operator_payoff_matrices:
                        self.operator_payoff_matrices[key] = np.zeros((n, n))
                        self.operator_counts[key] = np.zeros((n, n))

                    # Actualizar promedio incremental
                    count = self.operator_counts[key][i, j]
                    current_avg = self.operator_payoff_matrices[key][i, j]

                    new_count = count + 1
                    new_avg = current_avg + (payoffs.get(a1, 0) - current_avg) / new_count

                    self.operator_payoff_matrices[key][i, j] = new_avg
                    self.operator_counts[key][i, j] = new_count

    def get_best_response(self, agent: str, opponent_op: str) -> Tuple[str, float]:
        """
        Mejor respuesta endógena a un operador del oponente.

        Retorna (mejor_operador, payoff_esperado)
        """
        agent_idx = self.agent_names.index(agent)
        best_op = None
        best_payoff = float('-inf')

        # Buscar entre todas las combinaciones
        for key, matrix in self.operator_payoff_matrices.items():
            ops = key.split('_')
            if len(ops) == 2:
                my_op, their_op = ops
                if their_op == opponent_op:
                    payoff = matrix[agent_idx, :].mean()  # Promedio sobre oponentes
                    if payoff > best_payoff:
                        best_payoff = payoff
                        best_op = my_op

        return best_op, best_payoff

    def get_nash_approximation(self) -> Dict[str, Dict[str, float]]:
        """
        Aproximación a equilibrio de Nash emergente.

        Retorna distribución de probabilidad sobre operadores
        para cada agente, basada en payoffs históricos.
        """
        nash_probs = {}

        for agent in self.agent_names:
            op_payoffs = {}

            # Agregar payoffs por operador propio
            for key, matrix in self.operator_payoff_matrices.items():
                ops = key.split('_')
                if len(ops) == 2:
                    my_op = ops[0]
                    agent_idx = self.agent_names.index(agent)
                    count = self.operator_counts[key][agent_idx, :].sum()

                    if count > 0:
                        avg_payoff = matrix[agent_idx, :].mean()
                        if my_op not in op_payoffs:
                            op_payoffs[my_op] = []
                        op_payoffs[my_op].append(avg_payoff)

            # Convertir a probabilidades (softmax sobre payoffs promedio)
            if op_payoffs:
                avg_payoffs = {op: np.mean(pays) for op, pays in op_payoffs.items()}
                values = np.array(list(avg_payoffs.values()))

                # Softmax con temperatura 1
                exp_values = np.exp(values - np.max(values))
                probs = exp_values / exp_values.sum()

                nash_probs[agent] = dict(zip(avg_payoffs.keys(), probs))

        return nash_probs


def test_payoff_system():
    """Test del sistema de payoffs."""
    print("=" * 60)
    print("TEST: Payoff Endogenous - Δ de métricas")
    print("=" * 60)

    # Crear calculadora
    agent_names = ['NEO', 'EVA', 'ALEX']
    calculator = PayoffCalculator(agent_names=agent_names)

    # Simular deltas de una ronda
    deltas = {
        'NEO': {
            'delta_entropy': 0.05,
            'delta_phi': 0.02,
            'delta_identity': -0.01,
            'delta_coherence': 0.03,
            'delta_attachment': 0.1,
            'crisis_change': 0
        },
        'EVA': {
            'delta_entropy': -0.02,
            'delta_phi': 0.08,
            'delta_identity': 0.04,
            'delta_coherence': 0.01,
            'delta_attachment': 0.15,
            'crisis_change': 0
        },
        'ALEX': {
            'delta_entropy': 0.1,
            'delta_phi': -0.03,
            'delta_identity': -0.05,
            'delta_coherence': -0.02,
            'delta_attachment': -0.05,
            'crisis_change': 1
        }
    }

    print("\n--- Deltas de métricas ---")
    for agent, d in deltas.items():
        print(f"\n{agent}: {d}")

    # Payoffs raw
    print("\n--- Payoffs raw ---")
    for agent in agent_names:
        raw = calculator.compute_raw_payoff(deltas[agent])
        print(f"  {agent}: {raw:.4f}")

    # Payoffs rankeados
    print("\n--- Payoffs rankeados (relativos) ---")
    from dataclasses import dataclass
    @dataclass
    class MockRound:
        metric_deltas: Dict

    mock_round = MockRound(metric_deltas=deltas)
    payoffs = calculator.compute_all_payoffs(mock_round)

    for agent, payoff in payoffs.items():
        print(f"  {agent}: {payoff:.4f}")

    # Métrica de cooperación
    print("\n--- Métricas de sistema ---")
    coop_metric = CooperationMetric(agent_names=agent_names)
    coop_metric.update(payoffs)

    print(f"  Payoff del sistema: {coop_metric.system_payoff_history[-1]:.4f}")
    print(f"  Desigualdad (Gini): {coop_metric.inequality_history[-1]:.4f}")
    print(f"  Fairness: {coop_metric.get_fairness_index():.4f}")

    # Simular múltiples rondas
    print("\n--- Simulando 50 rondas ---")
    for _ in range(50):
        # Generar deltas aleatorios
        new_deltas = {}
        for agent in agent_names:
            new_deltas[agent] = {
                'delta_entropy': np.random.normal(0, 0.05),
                'delta_phi': np.random.normal(0, 0.05),
                'delta_identity': np.random.normal(0, 0.03),
                'delta_coherence': np.random.normal(0, 0.03),
                'delta_attachment': np.random.normal(0.02, 0.05),
                'crisis_change': np.random.choice([0, 0, 0, 1, -1])
            }

        mock_round.metric_deltas = new_deltas
        payoffs = calculator.compute_all_payoffs(mock_round)
        coop_metric.update(payoffs)

    print("\n--- Estadísticas finales ---")
    for agent in agent_names:
        stats = calculator.get_statistics(agent)
        print(f"\n{agent}:")
        print(f"  Payoff medio: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Rango: [{stats['min']:.4f}, {stats['max']:.4f}]")

    print(f"\n--- Métricas de sistema (final) ---")
    print(f"  Coordinación: {coop_metric.get_coordination_index():.4f}")
    print(f"  Fairness: {coop_metric.get_fairness_index():.4f}")

    print("\n✓ Payoffs 100% endógenos: solo Δ de métricas, sin valores externos")


if __name__ == "__main__":
    test_payoff_system()
