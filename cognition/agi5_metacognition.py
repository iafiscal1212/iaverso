"""
AGI-5: Metacognición Dinámica Jerárquica
=========================================

El sistema observa qué procesos cognitivos le funcionan mejor
en su propia historia y ajusta dinámicamente sus preferencias.

Vector metacognitivo:
    M_t = [ΔU_t, ΔV_t, -ΔC_t, coh_t, flow_t]

Auto-evaluación de procesos:
    score_p = corr(M_t, I_t^p)

Política metacognitiva:
    P(p) = rank(score_p) / Σ_q rank(score_q)

100% endógeno.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class CognitiveProcess(Enum):
    """Procesos cognitivos monitoreados."""
    EPISODIC_ENCODING = "episodic_encoding"
    NARRATIVE_UPDATE = "narrative_update"
    GOAL_PURSUIT = "goal_pursuit"
    PLANNING = "planning"
    EXPLORATION = "exploration"
    CONSOLIDATION = "consolidation"
    CRISIS_RESPONSE = "crisis_response"
    SOCIAL_MODELING = "social_modeling"
    TEMPORAL_PROJECTION = "temporal_projection"
    REGULATION = "regulation"


@dataclass
class MetacognitiveState:
    """Estado metacognitivo en un momento."""
    t: int
    M_vector: np.ndarray  # [ΔU, ΔV, -ΔC, coh, flow]
    active_processes: Dict[CognitiveProcess, bool]
    process_scores: Dict[CognitiveProcess, float]
    best_process: CognitiveProcess
    metacognitive_confidence: float


class DynamicMetacognition:
    """
    Metacognición jerárquica dinámica.

    Observa qué procesos cognitivos generan mejor vida
    y ajusta preferencias endógenamente.
    """

    def __init__(self, agent_name: str):
        """
        Inicializa metacognición.

        Args:
            agent_name: Nombre del agente
        """
        self.agent_name = agent_name

        # Historiales para cada componente de M
        self.U_history: List[float] = []  # Utilidad
        self.V_history: List[float] = []  # Valor
        self.C_history: List[float] = []  # Crisis/Coste
        self.coh_history: List[float] = []  # Coherencia
        self.flow_history: List[float] = []  # Flow

        # Historial de activación de procesos
        self.process_history: Dict[CognitiveProcess, List[bool]] = {
            p: [] for p in CognitiveProcess
        }

        # Scores acumulados
        self.process_scores: Dict[CognitiveProcess, float] = {
            p: 0.5 for p in CognitiveProcess
        }

        # Política actual
        self.process_policy: Dict[CognitiveProcess, float] = {
            p: 1.0 / len(CognitiveProcess) for p in CognitiveProcess
        }

        self.t = 0

    def _zscore_normalize(self, value: float, history: List[float]) -> float:
        """Normaliza valor con z-score histórico."""
        if len(history) < 10:
            return 0.0

        mu = np.mean(history)
        sigma = np.std(history) + 1e-8
        return (value - mu) / sigma

    def _compute_M_vector(self, U: float, V: float, C: float,
                          coh: float, flow: float) -> np.ndarray:
        """
        Computa vector metacognitivo M_t.

        M_t = [ΔU_t, ΔV_t, -ΔC_t, coh_t, flow_t]
        """
        # Calcular deltas
        if len(self.U_history) > 0:
            delta_U = U - self.U_history[-1]
            delta_V = V - self.V_history[-1]
            delta_C = C - self.C_history[-1]
        else:
            delta_U = delta_V = delta_C = 0.0

        # Normalizar con z-score
        delta_U_norm = self._zscore_normalize(delta_U,
                                              np.diff(self.U_history).tolist() if len(self.U_history) > 1 else [0])
        delta_V_norm = self._zscore_normalize(delta_V,
                                              np.diff(self.V_history).tolist() if len(self.V_history) > 1 else [0])
        delta_C_norm = self._zscore_normalize(delta_C,
                                              np.diff(self.C_history).tolist() if len(self.C_history) > 1 else [0])
        coh_norm = self._zscore_normalize(coh, self.coh_history)
        flow_norm = self._zscore_normalize(flow, self.flow_history)

        return np.array([delta_U_norm, delta_V_norm, -delta_C_norm, coh_norm, flow_norm])

    def _compute_process_scores(self) -> Dict[CognitiveProcess, float]:
        """
        Calcula score de cada proceso: corr(M_t, I_t^p)
        """
        scores = {}
        window = int(np.ceil(np.sqrt(self.t + 1)))

        for process in CognitiveProcess:
            if len(self.process_history[process]) < window:
                scores[process] = 0.5
                continue

            # Obtener activaciones recientes
            I_p = np.array(self.process_history[process][-window:], dtype=float)

            # Construir M reciente
            if len(self.U_history) < window:
                scores[process] = 0.5
                continue

            M_recent = []
            for i in range(-window, 0):
                if abs(i) >= len(self.U_history):
                    continue

                # Deltas
                if abs(i) < len(self.U_history) - 1:
                    dU = self.U_history[i] - self.U_history[i-1]
                    dV = self.V_history[i] - self.V_history[i-1]
                    dC = self.C_history[i] - self.C_history[i-1]
                else:
                    dU = dV = dC = 0

                coh = self.coh_history[i] if abs(i) < len(self.coh_history) else 0
                flow = self.flow_history[i] if abs(i) < len(self.flow_history) else 0

                M_recent.append([dU, dV, -dC, coh, flow])

            if len(M_recent) < 5:
                scores[process] = 0.5
                continue

            M_arr = np.array(M_recent)

            # Correlación media con cada componente de M
            correlations = []
            for dim in range(min(M_arr.shape[1], len(I_p))):
                if len(M_arr[:, dim]) != len(I_p[:len(M_arr)]):
                    continue
                try:
                    corr = np.corrcoef(M_arr[:len(I_p), dim], I_p[:len(M_arr)])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
                except:
                    pass

            if correlations:
                scores[process] = float(np.mean(correlations))
            else:
                scores[process] = 0.0

        return scores

    def _update_policy(self, scores: Dict[CognitiveProcess, float]):
        """
        Actualiza política metacognitiva.

        P(p) = rank(score_p) / Σ_q rank(score_q)
        """
        # Calcular ranks
        score_values = list(scores.values())
        processes = list(scores.keys())

        ranks = {}
        sorted_indices = np.argsort(score_values)
        for rank_idx, idx in enumerate(sorted_indices):
            ranks[processes[idx]] = rank_idx + 1

        # Normalizar
        total_rank = sum(ranks.values())
        for process in CognitiveProcess:
            self.process_policy[process] = ranks[process] / total_rank

    def step(self, U: float, V: float, C: float,
             coh: float, flow: float,
             active_processes: Dict[CognitiveProcess, bool]) -> MetacognitiveState:
        """
        Paso de metacognición.

        Args:
            U: Utilidad actual
            V: Valor actual
            C: Crisis/Coste actual
            coh: Coherencia narrativa
            flow: Estado de flow
            active_processes: Procesos activos en este paso

        Returns:
            MetacognitiveState con evaluación
        """
        self.t += 1

        # Registrar en historiales
        self.U_history.append(U)
        self.V_history.append(V)
        self.C_history.append(C)
        self.coh_history.append(coh)
        self.flow_history.append(flow)

        # Registrar activaciones
        for process in CognitiveProcess:
            self.process_history[process].append(
                active_processes.get(process, False)
            )

        # Limitar historiales
        max_hist = 1000
        if len(self.U_history) > max_hist:
            self.U_history = self.U_history[-max_hist:]
            self.V_history = self.V_history[-max_hist:]
            self.C_history = self.C_history[-max_hist:]
            self.coh_history = self.coh_history[-max_hist:]
            self.flow_history = self.flow_history[-max_hist:]
            for p in CognitiveProcess:
                self.process_history[p] = self.process_history[p][-max_hist:]

        # Computar vector M
        M = self._compute_M_vector(U, V, C, coh, flow)

        # Evaluar procesos
        scores = self._compute_process_scores()
        self.process_scores = scores

        # Actualizar política
        self._update_policy(scores)

        # Mejor proceso
        best_process = max(scores, key=scores.get)

        # Confianza metacognitiva = varianza de scores (alta varianza = buena discriminación)
        score_values = list(scores.values())
        confidence = np.std(score_values) / (np.mean(np.abs(score_values)) + 1e-8)
        confidence = float(np.clip(confidence, 0, 1))

        return MetacognitiveState(
            t=self.t,
            M_vector=M,
            active_processes=active_processes,
            process_scores=scores,
            best_process=best_process,
            metacognitive_confidence=confidence
        )

    def get_process_preference(self, process: CognitiveProcess) -> float:
        """Obtiene preferencia actual por un proceso."""
        return self.process_policy.get(process, 0.1)

    def should_activate(self, process: CognitiveProcess) -> bool:
        """
        Decide si activar un proceso basado en política.

        Usa sampling proporcional a la política.
        """
        return np.random.random() < self.process_policy[process]

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas metacognitivas."""
        return {
            'agent': self.agent_name,
            't': self.t,
            'process_scores': {p.value: s for p, s in self.process_scores.items()},
            'process_policy': {p.value: pr for p, pr in self.process_policy.items()},
            'best_processes': self._get_top_processes(3),
            'worst_processes': self._get_bottom_processes(3),
            'mean_U': float(np.mean(self.U_history[-50:])) if self.U_history else 0,
            'mean_V': float(np.mean(self.V_history[-50:])) if self.V_history else 0,
            'mean_coh': float(np.mean(self.coh_history[-50:])) if self.coh_history else 0
        }

    def _get_top_processes(self, n: int) -> List[str]:
        """Obtiene los n mejores procesos."""
        sorted_procs = sorted(self.process_scores.items(),
                             key=lambda x: x[1], reverse=True)
        return [p.value for p, _ in sorted_procs[:n]]

    def _get_bottom_processes(self, n: int) -> List[str]:
        """Obtiene los n peores procesos."""
        sorted_procs = sorted(self.process_scores.items(),
                             key=lambda x: x[1])
        return [p.value for p, _ in sorted_procs[:n]]


def test_metacognition():
    """Test de metacognición dinámica."""
    print("=" * 60)
    print("TEST AGI-5: METACOGNICIÓN DINÁMICA")
    print("=" * 60)

    meta = DynamicMetacognition("NEO")

    # Simular 500 pasos
    for t in range(500):
        # Estado base que evoluciona
        U = 0.5 + 0.3 * np.sin(t / 50) + np.random.randn() * 0.1
        V = 0.4 + 0.2 * np.cos(t / 40) + np.random.randn() * 0.1
        C = max(0, 0.2 + 0.1 * np.sin(t / 30) + np.random.randn() * 0.05)
        coh = 0.6 + 0.2 * np.sin(t / 60) + np.random.randn() * 0.1
        flow = 0.5 + 0.3 * np.cos(t / 45) + np.random.randn() * 0.1

        # Activar procesos con correlación a U/V
        active = {}
        for process in CognitiveProcess:
            if process == CognitiveProcess.EXPLORATION:
                # Exploración correlaciona con bajo V
                active[process] = V < 0.4 and np.random.random() < 0.6
            elif process == CognitiveProcess.CONSOLIDATION:
                # Consolidación correlaciona con alto coh
                active[process] = coh > 0.6 and np.random.random() < 0.7
            elif process == CognitiveProcess.CRISIS_RESPONSE:
                # Crisis response correlaciona con alto C
                active[process] = C > 0.3 and np.random.random() < 0.8
            elif process == CognitiveProcess.PLANNING:
                # Planning correlaciona con alto U
                active[process] = U > 0.5 and np.random.random() < 0.5
            else:
                active[process] = np.random.random() < 0.3

        state = meta.step(U, V, C, coh, flow, active)

        if (t + 1) % 100 == 0:
            print(f"\n  t={t+1}:")
            print(f"    Best process: {state.best_process.value}")
            print(f"    Confidence: {state.metacognitive_confidence:.3f}")

    # Resultados finales
    stats = meta.get_statistics()

    print("\n" + "=" * 60)
    print("RESULTADOS METACOGNICIÓN")
    print("=" * 60)

    print("\n  Top 3 procesos:")
    for p in stats['best_processes']:
        print(f"    - {p}: score={stats['process_scores'][p]:.3f}, "
              f"policy={stats['process_policy'][p]:.3f}")

    print("\n  Bottom 3 procesos:")
    for p in stats['worst_processes']:
        print(f"    - {p}: score={stats['process_scores'][p]:.3f}, "
              f"policy={stats['process_policy'][p]:.3f}")

    # Verificar que hay diferenciación
    scores = list(stats['process_scores'].values())
    if np.std(scores) > 0.05:
        print("\n  ✓ Metacognición diferenciando procesos correctamente")
    else:
        print("\n  ⚠️ Poca diferenciación entre procesos")

    return meta


if __name__ == "__main__":
    test_metacognition()
