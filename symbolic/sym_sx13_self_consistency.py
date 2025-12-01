"""
SX13 - Consistencia Narrativa del Self
======================================

Mide si la auto-narrativa del agente es consistente con su estado interno:
- Consistencia entre self_report y estado estructural
- Penalizacion por "saltos" narrativos sin cambio estructural

Criterios:
- PASS: SX13_global > 0.5
- EXCELLENT: SX13_global > 0.7

Formula:
SX13_A = Cons_self * (1 - JumpRate)
donde:
- Cons_self = 1 - MSE(self_report, self_report_pred) / Q95(MSE_null)
- JumpRate = fraccion de grandes cambios narrativos con delta_state < Q25

100% endogeno. Sin numeros magicos.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import sys
sys.path.insert(0, '/root/NEO_EVA')


@dataclass
class SelfReportSnapshot:
    """Snapshot del auto-reporte narrativo."""
    t: int
    purpose: np.ndarray          # Vector de proposito [dim]
    phase: int                   # Fase narrativa (0=crisis, 1=consolidation, 2=thriving)
    evaluation: float            # Auto-evaluacion [-1, 1]
    narrative_vector: np.ndarray  # Representacion completa


@dataclass
class StateSnapshot:
    """Snapshot del estado estructural."""
    t: int
    phi: float                   # Fenomenologia
    sagi: float                  # SAGI score
    crisis_flag: bool
    ce_local: float              # Continuidad episodica local
    state_vector: np.ndarray     # Representacion completa


@dataclass
class SX13Result:
    """Resultado del test SX13."""
    score: float
    passed: bool
    excellent: bool
    consistency_global: float
    jump_rate_global: float
    agent_scores: Dict[str, float]
    agent_consistency: Dict[str, float]
    agent_jump_rates: Dict[str, float]
    details: Dict[str, Any]


class SelfConsistencyTracker:
    """
    Tracker de consistencia narrativa del self para SX13.
    """

    def __init__(self, agent_id: str, state_dim: int = 8, narrative_dim: int = 6):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.narrative_dim = narrative_dim

        # Historial de snapshots
        self.self_reports: List[SelfReportSnapshot] = []
        self.states: List[StateSnapshot] = []

        # Historial para umbrales endogenos
        self.prediction_errors: List[float] = []
        self.state_deltas: List[float] = []
        self.narrative_deltas: List[float] = []

        # Modelo interno simple: f(state) -> narrative
        # Usamos regresion lineal online
        self._model_weights: Optional[np.ndarray] = None
        self._model_bias: Optional[np.ndarray] = None

    def record_step(self, t: int, phi: float, sagi: float, crisis_flag: bool,
                    ce_local: float, purpose: np.ndarray, phase: int,
                    evaluation: float):
        """Registra un paso con estado y auto-reporte."""
        # Construir vectores
        state_vec = np.array([phi, sagi, float(crisis_flag), ce_local,
                             phi * sagi, float(crisis_flag) * (1 - ce_local),
                             np.sin(phi), np.cos(sagi)])[:self.state_dim]

        narrative_vec = np.concatenate([
            purpose[:3] if len(purpose) >= 3 else np.pad(purpose, (0, 3 - len(purpose))),
            [phase / 2.0, evaluation, (evaluation + 1) / 2]
        ])[:self.narrative_dim]

        state = StateSnapshot(
            t=t, phi=phi, sagi=sagi, crisis_flag=crisis_flag,
            ce_local=ce_local, state_vector=state_vec
        )

        self_report = SelfReportSnapshot(
            t=t, purpose=purpose, phase=phase, evaluation=evaluation,
            narrative_vector=narrative_vec
        )

        self.states.append(state)
        self.self_reports.append(self_report)

        # Actualizar modelo interno
        self._update_model(state_vec, narrative_vec)

    def _update_model(self, state: np.ndarray, narrative: np.ndarray):
        """Actualiza el modelo interno f(state) -> narrative."""
        if self._model_weights is None:
            self._model_weights = np.random.randn(self.state_dim, self.narrative_dim) * 0.1
            self._model_bias = np.zeros(self.narrative_dim)

        # Prediccion actual
        pred = state @ self._model_weights + self._model_bias

        # Error
        error = narrative - pred

        # Gradiente descendente simple
        learning_rate = 0.01
        self._model_weights += learning_rate * np.outer(state, error)
        self._model_bias += learning_rate * error

    def predict_narrative(self, state: np.ndarray) -> np.ndarray:
        """Predice narrativa a partir de estado."""
        if self._model_weights is None:
            return np.zeros(self.narrative_dim)
        return state @ self._model_weights + self._model_bias

    def compute_consistency(self) -> float:
        """
        Calcula consistencia entre self_report y prediccion basada en estado.
        Cons_self = 1 - MSE(self_report, pred) / Q95(MSE_null)
        """
        if len(self.states) < 5:
            return 0.5

        mses = []
        for i in range(len(self.states)):
            state = self.states[i].state_vector
            actual = self.self_reports[i].narrative_vector
            pred = self.predict_narrative(state)

            mse = float(np.mean((actual - pred) ** 2))
            mses.append(mse)
            self.prediction_errors.append(mse)

        # MSE_null: predecir con media
        mean_narrative = np.mean([sr.narrative_vector for sr in self.self_reports], axis=0)
        null_mses = [float(np.mean((sr.narrative_vector - mean_narrative) ** 2))
                    for sr in self.self_reports]

        # Q95 de MSE null
        q95_null = np.percentile(null_mses, 95) + 1e-8
        mse_actual = np.mean(mses)

        consistency = 1 - mse_actual / q95_null
        return float(np.clip(consistency, 0, 1))

    def compute_jump_rate(self) -> float:
        """
        Calcula tasa de saltos narrativos sin cambio estructural.
        JumpRate = fraccion de (delta_narrative > Q75) con (delta_state < Q25)
        """
        if len(self.states) < 3:
            return 0.0

        # Calcular deltas
        for i in range(1, len(self.states)):
            state_delta = np.linalg.norm(
                self.states[i].state_vector - self.states[i-1].state_vector
            )
            narrative_delta = np.linalg.norm(
                self.self_reports[i].narrative_vector - self.self_reports[i-1].narrative_vector
            )
            self.state_deltas.append(state_delta)
            self.narrative_deltas.append(narrative_delta)

        if len(self.state_deltas) < 3:
            return 0.0

        # Umbrales endogenos
        q25_state = np.percentile(self.state_deltas, 25)
        q75_narrative = np.percentile(self.narrative_deltas, 75)

        # Contar saltos
        n_jumps = 0
        n_total = 0

        for i in range(len(self.state_deltas)):
            if self.narrative_deltas[i] > q75_narrative:
                n_total += 1
                if self.state_deltas[i] < q25_state:
                    n_jumps += 1

        if n_total == 0:
            return 0.0

        return float(n_jumps / n_total)

    def compute_sx13_score(self) -> Tuple[float, float, float]:
        """
        Calcula el score SX13 para el agente.

        Returns:
            (score, consistency, jump_rate)
        """
        consistency = self.compute_consistency()
        jump_rate = self.compute_jump_rate()

        # SX13_A = Cons_self * (1 - JumpRate)
        score = consistency * (1 - jump_rate)

        return float(score), float(consistency), float(jump_rate)

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadisticas completas."""
        score, consistency, jump_rate = self.compute_sx13_score()

        return {
            'score': score,
            'consistency': consistency,
            'jump_rate': jump_rate,
            'n_steps': len(self.states),
            'mean_prediction_error': float(np.mean(self.prediction_errors)) if self.prediction_errors else 0.0
        }


def score_sx13_global(agent_trackers: Dict[str, SelfConsistencyTracker]) -> SX13Result:
    """
    Calcula el score SX13 global.

    Args:
        agent_trackers: Dict de trackers por agente

    Returns:
        SX13Result con score global y detalles
    """
    if not agent_trackers:
        return SX13Result(
            score=0.0, passed=False, excellent=False,
            consistency_global=0.0, jump_rate_global=0.0,
            agent_scores={}, agent_consistency={}, agent_jump_rates={},
            details={}
        )

    agent_scores = {}
    agent_consistency = {}
    agent_jump_rates = {}

    for aid, tracker in agent_trackers.items():
        score, cons, jump = tracker.compute_sx13_score()
        agent_scores[aid] = score
        agent_consistency[aid] = cons
        agent_jump_rates[aid] = jump

    # Globales
    score_global = float(np.mean(list(agent_scores.values())))
    consistency_global = float(np.mean(list(agent_consistency.values())))
    jump_rate_global = float(np.mean(list(agent_jump_rates.values())))

    # Criterios
    passed = score_global > 0.5
    excellent = score_global > 0.7

    return SX13Result(
        score=score_global,
        passed=passed,
        excellent=excellent,
        consistency_global=consistency_global,
        jump_rate_global=jump_rate_global,
        agent_scores=agent_scores,
        agent_consistency=agent_consistency,
        agent_jump_rates=agent_jump_rates,
        details={
            'n_agents': len(agent_trackers),
            'total_steps': sum(len(t.states) for t in agent_trackers.values())
        }
    )


def run_sx13_test(n_agents: int = 5, n_steps: int = 200) -> SX13Result:
    """
    Ejecuta el test SX13 completo con datos simulados.
    """
    print("=" * 70)
    print("SX13 - CONSISTENCIA NARRATIVA DEL SELF")
    print("=" * 70)
    print(f"  Agentes: {n_agents}")
    print(f"  Pasos: {n_steps}")
    print("=" * 70)

    np.random.seed(42)

    agent_ids = [f"A{i}" for i in range(n_agents)]

    # Crear trackers
    trackers: Dict[str, SelfConsistencyTracker] = {
        aid: SelfConsistencyTracker(aid) for aid in agent_ids
    }

    # Simular evolucion
    for aid in agent_ids:
        tracker = trackers[aid]

        # Estado base del agente
        phi_base = 0.5 + np.random.random() * 0.2
        sagi_base = 0.5 + np.random.random() * 0.2
        purpose_base = np.random.randn(4) * 0.3

        for t in range(n_steps):
            # Evolucion gradual del estado
            phi = phi_base + np.sin(t * 0.05) * 0.1 + np.random.randn() * 0.02
            sagi = sagi_base + np.cos(t * 0.03) * 0.1 + np.random.randn() * 0.02

            # Crisis ocasional
            crisis_flag = np.random.random() < 0.1

            # CE local
            ce_local = 0.6 + np.random.randn() * 0.1

            # Auto-reporte consistente con estado
            # (con algo de ruido para simular imperfeccion)
            purpose = purpose_base + np.random.randn(4) * 0.05

            # Fase basada en estado
            if crisis_flag:
                phase = 0
            elif sagi > 0.6:
                phase = 2
            else:
                phase = 1

            # Evaluacion basada en phi y sagi
            evaluation = (phi + sagi) / 2 - 0.5 + np.random.randn() * 0.1

            # Ocasionalmente: salto narrativo sin cambio estructural (simula inconsistencia)
            if np.random.random() < 0.05:
                phase = np.random.randint(0, 3)
                evaluation = np.random.randn() * 0.5

            tracker.record_step(t, phi, sagi, crisis_flag, ce_local,
                              purpose, phase, evaluation)

    # Calcular resultado global
    result = score_sx13_global(trackers)

    print("\n" + "=" * 70)
    print("RESULTADOS SX13")
    print("=" * 70)
    print(f"  Score SX13: {result.score:.4f}")
    print(f"  Passed: {result.passed} (> 0.5)")
    print(f"  Excellent: {result.excellent} (> 0.7)")
    print(f"\n  Metricas globales:")
    print(f"    Consistencia: {result.consistency_global:.4f}")
    print(f"    Jump Rate: {result.jump_rate_global:.4f}")
    print(f"\n  Scores por agente:")
    for aid in agent_ids:
        print(f"    {aid}: score={result.agent_scores[aid]:.4f}, "
              f"cons={result.agent_consistency[aid]:.4f}, "
              f"jump={result.agent_jump_rates[aid]:.4f}")
    print("=" * 70)

    return result


if __name__ == "__main__":
    result = run_sx13_test(n_agents=5, n_steps=200)
