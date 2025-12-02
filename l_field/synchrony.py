"""
Synchrony - Latent Synchrony Index (LSI)
========================================

Mide cuánto coinciden las fases internas entre agentes
SIN interacción directa.

LSI(t) = (1 / N(N-1)) * Σ_{i≠j} cos(θ_i(t) - θ_j(t))

Donde θ_i(t) se deriva de:
- Fase circadiana
- Fase cuántica (Q-field)
- Fase narrativa

LSI alto = coherencia grupal espontánea
LSI bajo = fragmentación

100% endógeno. Sin números mágicos. Solo observa.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class SynchronySnapshot:
    """Estado de sincronía en un instante."""
    t: int
    LSI: float                          # Latent Synchrony Index
    phase_coherence: float              # Coherencia de fase global
    n_synchronized_pairs: int           # Pares con alta sincronía
    dominant_phase: float               # Fase dominante
    phase_dispersion: float             # Dispersión de fases


class LatentSynchrony:
    """
    Calcula el Índice de Sincronía Latente (LSI).

    Observa correlaciones de fase entre agentes sin influirlos.
    Las fases se derivan de estados internos (circadiano, cuántico, narrativo).
    """

    def __init__(self):
        """Inicializa el observador de sincronía."""
        self.eps = np.finfo(float).eps
        self.t = 0

        # Historial de fases por agente
        self._phase_history: Dict[str, List[float]] = {}

        # Historial de LSI
        self._lsi_history: List[float] = []

        # Snapshots
        self._snapshots: List[SynchronySnapshot] = []

    def _extract_phase(
        self,
        circadian_phase: float = 0.0,
        quantum_phase: float = 0.0,
        narrative_phase: float = 0.0
    ) -> float:
        """
        Extrae fase combinada de las tres fuentes.

        Combina las fases usando promedio circular (no aritmético).
        Sin pesos humanos - cada fase contribuye 1/3.
        """
        # Número de componentes
        K = 3

        # Convertir a vectores unitarios en círculo
        phases = [circadian_phase, quantum_phase, narrative_phase]

        # Promedio circular: sumar vectores unitarios
        x_sum = sum(np.cos(p) for p in phases)
        y_sum = sum(np.sin(p) for p in phases)

        # Fase resultante
        combined_phase = np.arctan2(y_sum / K, x_sum / K)

        return float(combined_phase)

    def _compute_lsi(self, phases: Dict[str, float]) -> float:
        """
        Calcula Latent Synchrony Index.

        LSI(t) = (1 / N(N-1)) * Σ_{i≠j} cos(θ_i - θ_j)

        Rango: [-1, 1]
        - 1 = perfecta sincronía
        - 0 = aleatorio
        - -1 = anti-sincronía perfecta
        """
        agents = list(phases.keys())
        N = len(agents)

        if N < 2:
            return 0.0

        # Sumar cosenos de diferencias de fase
        cos_sum = 0.0
        n_pairs = 0

        for i in range(N):
            for j in range(i + 1, N):
                phase_diff = phases[agents[i]] - phases[agents[j]]
                cos_sum += np.cos(phase_diff)
                n_pairs += 1

        # Normalizar por N(N-1)/2 pares únicos
        # Pero la fórmula usa N(N-1), así que multiplicamos por 2
        LSI = (2 * cos_sum) / (N * (N - 1) + self.eps)

        return float(LSI)

    def _compute_phase_coherence(self, phases: Dict[str, float]) -> float:
        """
        Calcula coherencia de fase global (orden parameter).

        R = |1/N * Σ e^{iθ_j}|

        Rango: [0, 1]
        - 1 = todas las fases alineadas
        - 0 = fases uniformemente distribuidas
        """
        if not phases:
            return 0.0

        N = len(phases)
        phase_values = list(phases.values())

        # Sumar vectores unitarios
        x_sum = sum(np.cos(p) for p in phase_values)
        y_sum = sum(np.sin(p) for p in phase_values)

        # Magnitud normalizada
        R = np.sqrt(x_sum**2 + y_sum**2) / N

        return float(R)

    def _count_synchronized_pairs(
        self,
        phases: Dict[str, float],
        threshold: float = None
    ) -> int:
        """
        Cuenta pares con alta sincronía.

        Umbral endógeno: cos(diff) > 1/2 (60 grados)
        """
        if threshold is None:
            threshold = 1 / 2  # Endógeno

        agents = list(phases.keys())
        N = len(agents)
        count = 0

        for i in range(N):
            for j in range(i + 1, N):
                phase_diff = phases[agents[i]] - phases[agents[j]]
                if np.cos(phase_diff) > threshold:
                    count += 1

        return count

    def observe(
        self,
        agent_states: Dict[str, Dict[str, float]]
    ) -> SynchronySnapshot:
        """
        Observa sincronía latente entre agentes.

        Args:
            agent_states: {agent_id: {
                'circadian_phase': float,  # Fase circadiana [0, 2π]
                'quantum_phase': float,    # Fase cuántica
                'narrative_phase': float   # Fase narrativa (derivada de H_narr)
            }}

        Returns:
            SynchronySnapshot con métricas de sincronía
        """
        self.t += 1

        # Extraer fases combinadas
        phases: Dict[str, float] = {}

        for agent_id, state in agent_states.items():
            phase = self._extract_phase(
                circadian_phase=state.get('circadian_phase', 0.0),
                quantum_phase=state.get('quantum_phase', 0.0),
                narrative_phase=state.get('narrative_phase', 0.0)
            )
            phases[agent_id] = phase

            # Guardar historial
            if agent_id not in self._phase_history:
                self._phase_history[agent_id] = []
            self._phase_history[agent_id].append(phase)

        # Calcular métricas
        LSI = self._compute_lsi(phases)
        phase_coherence = self._compute_phase_coherence(phases)
        n_sync_pairs = self._count_synchronized_pairs(phases)

        # Fase dominante (promedio circular)
        if phases:
            x_mean = np.mean([np.cos(p) for p in phases.values()])
            y_mean = np.mean([np.sin(p) for p in phases.values()])
            dominant_phase = np.arctan2(y_mean, x_mean)
        else:
            dominant_phase = 0.0

        # Dispersión de fases (varianza circular)
        phase_dispersion = 1 - phase_coherence  # Complemento de coherencia

        # Crear snapshot
        snapshot = SynchronySnapshot(
            t=self.t,
            LSI=LSI,
            phase_coherence=phase_coherence,
            n_synchronized_pairs=n_sync_pairs,
            dominant_phase=float(dominant_phase),
            phase_dispersion=phase_dispersion
        )

        self._lsi_history.append(LSI)
        self._snapshots.append(snapshot)

        # Limitar historial
        max_history = 500
        if len(self._snapshots) > max_history:
            self._snapshots = self._snapshots[-max_history:]
        if len(self._lsi_history) > max_history:
            self._lsi_history = self._lsi_history[-max_history:]

        return snapshot

    def get_lsi_history(self) -> List[float]:
        """Retorna historial de LSI."""
        return self._lsi_history.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de sincronía."""
        if not self._lsi_history:
            return {'t': 0, 'n_observations': 0}

        lsi_arr = np.array(self._lsi_history)

        return {
            't': self.t,
            'n_observations': len(self._lsi_history),
            'LSI_mean': float(np.mean(lsi_arr)),
            'LSI_std': float(np.std(lsi_arr)),
            'LSI_current': self._lsi_history[-1] if self._lsi_history else 0,
            'LSI_max': float(np.max(lsi_arr)),
            'LSI_min': float(np.min(lsi_arr)),
            'high_sync_ratio': float(np.mean(lsi_arr > 1/2)),  # % tiempo en alta sincronía
            'agents_tracked': len(self._phase_history)
        }
