"""
Layer Emergence: Base Classes for Existential Layers
=====================================================

Cada capa existencial emerge de metricas internas del agente.
Todas las capas comparten:
    - Estado actual [0,1]
    - Historial para calculos endogenos
    - Varianza para pesos adaptativos
    - Metodo de actualizacion

100% endogeno.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from enum import Enum

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history


class LayerType(Enum):
    """Tipos de capas existenciales."""
    COGNITIVE = "L1_cognitive"
    SYMBOLIC = "L2_symbolic"
    NARRATIVE = "L3_narrative"
    LIFE = "L4_life"
    HEALTH = "L5_health"
    SOCIAL = "L6_social"
    TENSION = "L7_tension"
    IDENTITY = "L8_identity"
    PHASE = "L9_phase"
    INTEGRATED = "L10_integrated"


@dataclass
class LayerState:
    """Estado de una capa existencial."""
    layer_type: LayerType
    value: float                    # [0, 1] valor actual
    variance: float                 # Varianza reciente
    trend: float                    # Tendencia [-1, 1]
    stability: float                # Estabilidad [0, 1]
    components: Dict[str, float]    # Sub-componentes
    t: int                          # Timestep


@dataclass
class LayerHistory:
    """Historial de una capa para calculos endogenos."""
    values: List[float] = field(default_factory=list)
    timestamps: List[int] = field(default_factory=list)
    components_history: List[Dict[str, float]] = field(default_factory=list)

    def add(self, value: float, t: int, components: Dict[str, float] = None):
        """Agrega valor al historial."""
        self.values.append(value)
        self.timestamps.append(t)
        if components:
            self.components_history.append(components)

    def trim(self, max_len: int):
        """Recorta historial a longitud maxima."""
        if len(self.values) > max_len:
            self.values = self.values[-max_len:]
            self.timestamps = self.timestamps[-max_len:]
            if self.components_history:
                self.components_history = self.components_history[-max_len:]

    def get_recent(self, window: int) -> List[float]:
        """Obtiene valores recientes."""
        return self.values[-window:] if self.values else []

    def get_variance(self, window: int = None) -> float:
        """Calcula varianza de valores recientes."""
        if not self.values:
            return 0.0
        if window is None:
            window = len(self.values)
        recent = self.values[-window:]
        if len(recent) < 2:
            return 0.0
        return float(np.var(recent))

    def get_trend(self, window: int = None) -> float:
        """Calcula tendencia (slope normalizado)."""
        if not self.values:
            return 0.0
        if window is None:
            window = len(self.values)
        recent = self.values[-window:]
        if len(recent) < 3:
            return 0.0

        # Regresion lineal simple
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]

        # Normalizar por rango
        range_val = max(recent) - min(recent) + 1e-8
        normalized = slope / range_val

        return float(np.clip(normalized, -1, 1))

    def get_stability(self, window: int = None) -> float:
        """Calcula estabilidad (1 - normalized_std)."""
        if not self.values:
            return 1.0
        if window is None:
            window = len(self.values)
        recent = self.values[-window:]
        if len(recent) < 2:
            return 1.0

        std = np.std(recent)
        mean = np.mean(recent)

        # Coeficiente de variacion invertido
        cv = std / (mean + 1e-8)
        stability = 1 / (1 + cv)

        return float(np.clip(stability, 0, 1))


class ExistentialLayer(ABC):
    """
    Clase base abstracta para capas existenciales.

    Cada capa:
        - Tiene un tipo (L1-L10)
        - Mantiene historial para calculos endogenos
        - Calcula su valor de observaciones internas
        - Reporta varianza para pesos adaptativos
    """

    def __init__(self, agent_id: str, layer_type: LayerType):
        """
        Inicializa capa existencial.

        Args:
            agent_id: ID del agente
            layer_type: Tipo de capa
        """
        self.agent_id = agent_id
        self.layer_type = layer_type
        self.history = LayerHistory()
        self.t = 0

        # Estado actual
        self._current_value = 0.5
        self._current_components: Dict[str, float] = {}

    @abstractmethod
    def compute(self, observations: Dict[str, Any]) -> float:
        """
        Calcula valor de la capa desde observaciones.

        Args:
            observations: Metricas observadas del agente

        Returns:
            Valor de coherencia/estado [0, 1]
        """
        pass

    def update(self, observations: Dict[str, Any]) -> LayerState:
        """
        Actualiza capa con nuevas observaciones.

        Args:
            observations: Metricas observadas

        Returns:
            Estado actualizado
        """
        self.t += 1

        # Calcular valor
        value = self.compute(observations)
        self._current_value = value

        # Agregar a historial
        self.history.add(value, self.t, self._current_components.copy())

        # Recortar historial
        self.history.trim(max_history(self.t))

        return self.get_state()

    def get_state(self) -> LayerState:
        """Obtiene estado actual de la capa."""
        window = L_t(self.t)

        return LayerState(
            layer_type=self.layer_type,
            value=self._current_value,
            variance=self.history.get_variance(window),
            trend=self.history.get_trend(window),
            stability=self.history.get_stability(window),
            components=self._current_components.copy(),
            t=self.t
        )

    def get_weight(self) -> float:
        """
        Obtiene peso endogeno para agregacion.

        Peso = 1 / (varianza + epsilon)
        Capas mas estables tienen mas peso.
        """
        window = L_t(self.t)
        variance = self.history.get_variance(window)

        # Peso inverso a varianza
        weight = 1.0 / (variance + 0.01)

        return weight

    def get_value(self) -> float:
        """Obtiene valor actual."""
        return self._current_value

    def get_statistics(self) -> Dict[str, Any]:
        """Estadisticas de la capa."""
        window = L_t(self.t)

        return {
            'layer_type': self.layer_type.value,
            'agent_id': self.agent_id,
            't': self.t,
            'value': self._current_value,
            'variance': self.history.get_variance(window),
            'trend': self.history.get_trend(window),
            'stability': self.history.get_stability(window),
            'weight': self.get_weight(),
            'history_length': len(self.history.values),
            'components': self._current_components
        }


def aggregate_layers(
    layers: List[ExistentialLayer],
    method: str = 'inverse_variance'
) -> float:
    """
    Agrega multiples capas en un valor unico.

    Args:
        layers: Lista de capas
        method: Metodo de agregacion
            'inverse_variance': Peso por 1/var (default)
            'uniform': Peso uniforme
            'stability': Peso por estabilidad

    Returns:
        Valor agregado [0, 1]
    """
    if not layers:
        return 0.5

    values = []
    weights = []

    for layer in layers:
        values.append(layer.get_value())

        if method == 'inverse_variance':
            weights.append(layer.get_weight())
        elif method == 'stability':
            state = layer.get_state()
            weights.append(state.stability)
        else:  # uniform
            weights.append(1.0)

    # Normalizar pesos
    total_weight = sum(weights)
    if total_weight < 1e-8:
        weights = [1.0 / len(weights)] * len(weights)
    else:
        weights = [w / total_weight for w in weights]

    # Agregacion ponderada
    aggregated = sum(v * w for v, w in zip(values, weights))

    return float(np.clip(aggregated, 0, 1))


def compute_layer_tension(layers: List[ExistentialLayer]) -> float:
    """
    Calcula tension entre capas.

    Tension alta = capas muy diferentes entre si.

    Returns:
        Tension [0, 1]
    """
    if len(layers) < 2:
        return 0.0

    values = [layer.get_value() for layer in layers]

    # Varianza de valores = tension
    variance = np.var(values)

    # Normalizar a [0, 1] usando sigmoid
    tension = 2 / (1 + np.exp(-5 * variance)) - 1

    return float(np.clip(tension, 0, 1))
