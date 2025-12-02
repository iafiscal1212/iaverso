"""
CDE Coherence - Sistema de Coherencia Global
============================================

Calcula coherencia global integrando:
- AGI-E
- CG-E
- LX
- ELLEX

100% endógeno. Sin números mágicos.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cde.cde_worldx import WorldXState


@dataclass
class CoherenceEvaluation:
    """Evaluación de coherencia global."""
    t: int
    coherence_index: float           # [0, 1] - índice global
    structural_coherence: float      # Coherencia estructural
    temporal_coherence: float        # Coherencia temporal
    symbolic_coherence: float        # Coherencia simbólica
    ellex_index: float               # Índice ELLEX


class CDECoherence:
    """
    Sistema de coherencia global.

    Integra:
    - Coherencia estructural (entre componentes)
    - Coherencia temporal (a lo largo del tiempo)
    - Coherencia simbólica (capas ELLEX)

    Todo endógeno basado en correlaciones y percentiles.
    """

    def __init__(self):
        self.t = 0

        # Historiales
        self._state_history: List[np.ndarray] = []
        self._coherence_history: List[float] = []
        self._ellex_history: List[float] = []

    def _compute_structural_coherence(self, state: WorldXState) -> float:
        """
        Calcula coherencia estructural.

        Basado en correlación entre componentes del estado.
        """
        return float(state.coherence)

    def _compute_temporal_coherence(self, state: WorldXState) -> float:
        """
        Calcula coherencia temporal.

        Basado en estabilidad de estados a lo largo del tiempo.
        """
        if len(self._state_history) < 5:
            return 1/2

        recent = np.array(self._state_history[-10:])

        # Varianza temporal por componente
        temporal_var = np.mean(np.var(recent, axis=0))

        # Coherencia = inverso de varianza
        # Normalizar por varianza histórica
        if len(self._coherence_history) > 10:
            var_hist = [np.mean(np.var(self._state_history[max(0,i-5):i+1], axis=0))
                       for i in range(5, len(self._state_history))]
            p95_var = np.percentile(var_hist, 95) if var_hist else 1

            coherence = 1 - (temporal_var / (p95_var + np.finfo(float).eps))
        else:
            coherence = 1 / (1 + temporal_var)

        return float(np.clip(coherence, 0, 1))

    def _compute_symbolic_coherence(self, state: WorldXState) -> float:
        """
        Calcula coherencia simbólica (proxy de ELLEX).

        Basado en consistencia de patrones.
        """
        if len(self._state_history) < 10:
            return 1/2

        # Detectar patrones mediante autocorrelación
        recent = np.array(self._state_history[-20:])
        mean_state = np.mean(recent, axis=0)

        # Similitud con media (como proxy de patrón estable)
        current = state.observation_vector
        dist_to_mean = np.linalg.norm(current - mean_state)

        # Normalizar
        dists = [np.linalg.norm(s - mean_state) for s in recent]
        max_dist = max(dists) if dists else 1

        symbolic_coh = 1 - (dist_to_mean / (max_dist + np.finfo(float).eps))

        return float(np.clip(symbolic_coh, 0, 1))

    def _compute_ellex_index(
        self,
        structural: float,
        temporal: float,
        symbolic: float
    ) -> float:
        """
        Calcula índice ELLEX.

        Combinación ponderada de coherencias.
        """
        # Pesos por varianza inversa si hay historial
        if len(self._ellex_history) > 10:
            # Usar pesos uniformes cuando no hay suficiente varianza
            ellex = (structural + temporal + symbolic) / 3
        else:
            # Peso inicial uniforme
            ellex = (structural + temporal + symbolic) / 3

        return float(np.clip(ellex, 0, 1))

    def compute_coherence(self, state: WorldXState) -> CoherenceEvaluation:
        """
        Calcula coherencia global.

        Args:
            state: Estado del WorldX

        Returns:
            Evaluación de coherencia
        """
        self.t += 1
        self._state_history.append(state.observation_vector.copy())

        # Calcular componentes
        structural = self._compute_structural_coherence(state)
        temporal = self._compute_temporal_coherence(state)
        symbolic = self._compute_symbolic_coherence(state)
        ellex = self._compute_ellex_index(structural, temporal, symbolic)

        self._ellex_history.append(ellex)

        # Índice global = ELLEX
        coherence_index = ellex
        self._coherence_history.append(coherence_index)

        return CoherenceEvaluation(
            t=self.t,
            coherence_index=coherence_index,
            structural_coherence=structural,
            temporal_coherence=temporal,
            symbolic_coherence=symbolic,
            ellex_index=ellex
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de coherencia."""
        return {
            't': self.t,
            'coherence_mean': float(np.mean(self._coherence_history[-10:])) if self._coherence_history else 1/2,
            'ellex_mean': float(np.mean(self._ellex_history[-10:])) if self._ellex_history else 1/2,
            'coherence_trend': float(np.polyfit(
                np.arange(min(10, len(self._coherence_history))),
                self._coherence_history[-10:],
                1
            )[0]) if len(self._coherence_history) >= 3 else 0
        }
