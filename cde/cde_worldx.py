"""
CDE WorldX - Mundo Interno del Sistema
=======================================

WORLD-X = versión adaptada de WORLD-1 para el software real.

Contiene:
- Campos: métricas del sistema
- Entidades: módulos
- Recursos: CPU, RAM, I/O
- Regímenes: "sano", "estresado", "fragmentado"

100% endógeno. Sin números mágicos.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class Regime(Enum):
    """Regímenes del sistema."""
    SANO = "sano"
    ESTRESADO = "estresado"
    FRAGMENTADO = "fragmentado"
    RECUPERANDO = "recuperando"


@dataclass
class WorldXState:
    """Estado del mundo interno."""
    t: int
    observation_vector: np.ndarray
    regime: Regime
    stress_level: float
    fragmentation: float
    coherence: float


class WorldX:
    """
    Mundo interno del sistema (WORLD-X).

    Mantiene:
    - Estado de campos (métricas)
    - Estado de entidades (módulos)
    - Estado de recursos
    - Régimen actual

    Todo endógeno basado en percentiles históricos.
    """

    def __init__(self, dimension: int = 7):
        """
        Args:
            dimension: Dimensión del vector de observación
        """
        self.dimension = dimension
        self.t = 0

        # Estado actual
        self._state: Optional[np.ndarray] = None
        self._regime = Regime.SANO

        # Historiales
        self._state_history: List[np.ndarray] = []
        self._regime_history: List[Regime] = []
        self._stress_history: List[float] = []
        self._fragmentation_history: List[float] = []

    def _compute_stress(self, observation: np.ndarray) -> float:
        """
        Calcula nivel de estrés endógeno.

        Basado en:
        - Carga de recursos (cpu, ram, load)
        - Longitud de cola
        """
        # Índices: cpu=3, ram=4, load=5, queue=6
        resource_stress = np.mean(observation[3:6])
        queue_stress = observation[6]

        # Combinar con peso por varianza inversa si hay historial
        if len(self._stress_history) > 5:
            var_resource = np.var([np.mean(s[3:6]) for s in self._state_history[-10:]])
            var_queue = np.var([s[6] for s in self._state_history[-10:]])

            EPS = np.finfo(float).eps
            w_resource = 1 / (var_resource + EPS)
            w_queue = 1 / (var_queue + EPS)
            w_total = w_resource + w_queue

            stress = (w_resource * resource_stress + w_queue * queue_stress) / w_total
        else:
            stress = (resource_stress + queue_stress) / 2

        return float(np.clip(stress, 0, 1))

    def _compute_fragmentation(self, observation: np.ndarray) -> float:
        """
        Calcula fragmentación endógena.

        Basado en varianza entre componentes.
        """
        if len(self._state_history) < 3:
            return 0

        # Varianza de las diferencias entre estados consecutivos
        deltas = []
        for i in range(1, len(self._state_history)):
            delta = self._state_history[i] - self._state_history[i-1]
            deltas.append(np.var(delta))

        if not deltas:
            return 0

        # Normalizar por percentil
        current_var = np.var(observation - self._state_history[-1]) if self._state_history else 0
        p95 = np.percentile(deltas, 95) if len(deltas) > 5 else max(deltas)

        fragmentation = current_var / (p95 + np.finfo(float).eps)

        return float(np.clip(fragmentation, 0, 1))

    def _compute_coherence(self, observation: np.ndarray) -> float:
        """
        Calcula coherencia interna.

        Basado en correlación entre componentes.
        """
        if len(self._state_history) < 10:
            return 1/2

        recent = np.array(self._state_history[-10:])

        try:
            corr = np.corrcoef(recent.T)
            mask = ~np.eye(self.dimension, dtype=bool)
            correlations = corr[mask]
            correlations = correlations[~np.isnan(correlations)]

            if len(correlations) == 0:
                return 1/2

            # Mayor correlación absoluta = mayor coherencia
            coherence = np.mean(np.abs(correlations))
        except:
            coherence = 1/2

        return float(np.clip(coherence, 0, 1))

    def infer_regime(self) -> Regime:
        """
        Infiere el régimen actual del sistema.

        Basado en umbrales endógenos (percentiles).
        """
        if len(self._stress_history) < 10 or len(self._fragmentation_history) < 10:
            return Regime.SANO

        stress = self._stress_history[-1]
        fragmentation = self._fragmentation_history[-1]

        # Umbrales endógenos
        stress_high = np.percentile(self._stress_history, 75)
        frag_high = np.percentile(self._fragmentation_history, 75)
        stress_low = np.percentile(self._stress_history, 25)

        # Lógica de régimen
        if fragmentation > frag_high:
            return Regime.FRAGMENTADO
        elif stress > stress_high:
            return Regime.ESTRESADO
        elif stress < stress_low and self._regime in [Regime.ESTRESADO, Regime.FRAGMENTADO]:
            return Regime.RECUPERANDO
        elif self._regime == Regime.RECUPERANDO and stress < stress_low:
            return Regime.SANO

        return self._regime

    def step(self, observation: np.ndarray) -> WorldXState:
        """
        Ejecuta un paso del mundo interno.

        Args:
            observation: Vector de observación del CDEObserver

        Returns:
            Estado actual del mundo
        """
        self.t += 1
        self._state = observation.copy()
        self._state_history.append(observation.copy())

        # Calcular métricas
        stress = self._compute_stress(observation)
        fragmentation = self._compute_fragmentation(observation)
        coherence = self._compute_coherence(observation)

        self._stress_history.append(stress)
        self._fragmentation_history.append(fragmentation)

        # Inferir régimen
        self._regime = self.infer_regime()
        self._regime_history.append(self._regime)

        return WorldXState(
            t=self.t,
            observation_vector=observation.copy(),
            regime=self._regime,
            stress_level=stress,
            fragmentation=fragmentation,
            coherence=coherence
        )

    def get_state(self) -> Optional[WorldXState]:
        """Retorna estado actual."""
        if self._state is None:
            return None

        return WorldXState(
            t=self.t,
            observation_vector=self._state.copy(),
            regime=self._regime,
            stress_level=self._stress_history[-1] if self._stress_history else 0,
            fragmentation=self._fragmentation_history[-1] if self._fragmentation_history else 0,
            coherence=self._compute_coherence(self._state) if self._state is not None else 1/2
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas del mundo."""
        return {
            't': self.t,
            'regime': self._regime.value,
            'stress_mean': float(np.mean(self._stress_history[-10:])) if self._stress_history else 0,
            'fragmentation_mean': float(np.mean(self._fragmentation_history[-10:])) if self._fragmentation_history else 0,
            'regime_distribution': {
                r.value: sum(1 for h in self._regime_history if h == r) / max(len(self._regime_history), 1)
                for r in Regime
            }
        }
