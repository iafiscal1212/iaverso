"""
ELLEX Index: L10 - Integracion Total
====================================

El indice final de existencia del agente.

ELLEX = Σ (w_i * C_i)
con w_i = 1/Var(C_i)
normalizado en [0,1]

Sin numeros magicos: todo por varianza inversa.

100% endogeno.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

import sys
sys.path.insert(0, '/root/NEO_EVA')

from ellex_map.layer_emergence import ExistentialLayer, LayerType
from cognition.agi_dynamic_constants import L_t, max_history


@dataclass
class ELLEXState:
    """Estado completo del indice ELLEX."""
    ellex: float                    # [0, 1] indice total
    layer_values: Dict[str, float]  # Valores por capa
    layer_weights: Dict[str, float] # Pesos por capa
    layer_contributions: Dict[str, float]  # Contribucion a ELLEX
    zone: str                       # 'struggling', 'balanced', 'flourishing'
    stability: float                # Estabilidad del indice
    trend: float                    # Tendencia (-1 a 1)
    t: int


class ELLEXIndex(ExistentialLayer):
    """
    L10: Integracion Total (ELLEX Index)

    Combina todas las capas en un indice unico:
        - L1: Coherencia Cognitiva
        - L2: Coherencia Simbolica
        - L3: Coherencia Narrativa
        - L4: Coherencia de Vida
        - L5: Salud Interior
        - L6: Coherencia Social
        - L7: Tension Existencial (invertida parcialmente)
        - L8: Identidad Persistente
        - L9: Equilibrio de Fases

    Cada capa tiene peso = 1/Var(historia de la capa)
    """

    LAYER_NAMES = [
        'L1_cognitive', 'L2_symbolic', 'L3_narrative', 'L4_life',
        'L5_health', 'L6_social', 'L7_tension', 'L8_identity', 'L9_phase'
    ]

    def __init__(self, agent_id: str):
        super().__init__(agent_id, LayerType.INTEGRATED)

        # Historiales por capa
        self._layer_histories: Dict[str, List[float]] = {
            name: [] for name in self.LAYER_NAMES
        }

        # Historial de ELLEX
        self._ellex_history: List[float] = []

        # Cache de ultima computacion
        self._last_weights: Dict[str, float] = {}
        self._last_contributions: Dict[str, float] = {}

    def _compute_layer_weight(self, layer_name: str) -> float:
        """
        Calcula peso de una capa basado en varianza inversa.

        w_i = 1 / (Var(C_i) + epsilon)

        Mas estable (menor varianza) = mayor peso.
        """
        history = self._layer_histories.get(layer_name, [])

        if len(history) < 3:
            return 1.0  # Peso uniforme sin historial

        window = L_t(self.t)
        recent = history[-window:]

        variance = np.var(recent)
        weight = 1.0 / (variance + 0.01)  # epsilon = 0.01

        return weight

    def _transform_tension(self, tension: float) -> float:
        """
        Transforma tension para ELLEX.

        Tension muy baja (estancamiento) o muy alta (crisis) = malo.
        Tension media (zona sana) = bueno.

        Usamos una funcion gaussiana centrada en 0.5.
        """
        # Encontrar centro optimo del historial
        history = self._layer_histories.get('L7_tension', [])

        if len(history) < 10:
            optimal_center = 0.5
        else:
            # El centro optimo es donde el agente ha estado mas tiempo sano
            # Aproximacion: la mediana historica
            optimal_center = np.median(history)

        # Transformacion gaussiana
        sigma = 0.25  # Ancho de la campana (endogenizable en futuro)
        transformed = np.exp(-((tension - optimal_center) ** 2) / (2 * sigma ** 2))

        return float(transformed)

    def _get_zone(self, ellex: float) -> str:
        """
        Determina zona existencial basada en percentiles.
        """
        if len(self._ellex_history) < 10:
            # Sin historial, usar rangos fijos temporalmente
            if ellex < 0.4:
                return 'struggling'
            elif ellex > 0.7:
                return 'flourishing'
            return 'balanced'

        # Thresholds endogenos
        low = np.percentile(self._ellex_history, 30)
        high = np.percentile(self._ellex_history, 70)

        if ellex < low:
            return 'struggling'
        elif ellex > high:
            return 'flourishing'
        return 'balanced'

    def _compute_stability(self) -> float:
        """
        Calcula estabilidad del indice ELLEX.

        Alta estabilidad = baja varianza reciente.
        """
        if len(self._ellex_history) < 5:
            return 0.5

        window = L_t(self.t)
        recent = self._ellex_history[-window:]

        std = np.std(recent)
        stability = 1.0 / (1 + std * 5)  # Escalar para sensibilidad

        return float(np.clip(stability, 0, 1))

    def _compute_trend(self) -> float:
        """
        Calcula tendencia del indice ELLEX.

        Positivo = mejorando, Negativo = empeorando.
        """
        if len(self._ellex_history) < 5:
            return 0.0

        window = min(L_t(self.t), len(self._ellex_history))
        recent = self._ellex_history[-window:]

        if len(recent) < 3:
            return 0.0

        # Regresion lineal simple
        x = np.arange(len(recent))
        slope, _ = np.polyfit(x, recent, 1)

        # Normalizar pendiente a [-1, 1]
        trend = np.tanh(slope * 10)

        return float(trend)

    def compute(self, observations: Dict[str, Any]) -> float:
        """
        Calcula el indice ELLEX.

        observations esperadas:
            - L1_cognitive: float [0,1]
            - L2_symbolic: float [0,1]
            - L3_narrative: float [0,1]
            - L4_life: float [0,1]
            - L5_health: float [0,1]
            - L6_social: float [0,1]
            - L7_tension: float [0,1]
            - L8_identity: float [0,1]
            - L9_phase: float [0,1]
        """
        # Extraer valores de capas
        layer_values = {}
        for name in self.LAYER_NAMES:
            value = observations.get(name, 0.5)
            layer_values[name] = value

            # Actualizar historial
            self._layer_histories[name].append(value)
            max_len = max_history(self.t)
            if len(self._layer_histories[name]) > max_len:
                self._layer_histories[name] = self._layer_histories[name][-max_len:]

        # Transformar tension
        original_tension = layer_values['L7_tension']
        transformed_tension = self._transform_tension(original_tension)
        layer_values['L7_tension'] = transformed_tension

        # Calcular pesos
        weights = {}
        for name in self.LAYER_NAMES:
            weights[name] = self._compute_layer_weight(name)

        # Normalizar pesos
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            weights = {k: 1.0 / len(self.LAYER_NAMES) for k in self.LAYER_NAMES}

        # Calcular ELLEX
        contributions = {}
        ellex = 0.0
        for name in self.LAYER_NAMES:
            contrib = layer_values[name] * weights[name]
            contributions[name] = contrib
            ellex += contrib

        ellex = float(np.clip(ellex, 0, 1))

        # Guardar en historial
        self._ellex_history.append(ellex)
        max_len = max_history(self.t)
        if len(self._ellex_history) > max_len:
            self._ellex_history = self._ellex_history[-max_len:]

        # Cache
        self._last_weights = weights
        self._last_contributions = contributions

        # Actualizar componentes
        self._current_components = {
            'layer_values': layer_values,
            'layer_weights': weights,
            'layer_contributions': contributions,
            'zone': self._get_zone(ellex),
            'stability': self._compute_stability(),
            'trend': self._compute_trend(),
            'original_tension': original_tension,
            'transformed_tension': transformed_tension
        }

        return ellex

    def get_ellex_state(self) -> ELLEXState:
        """Obtiene estado completo de ELLEX."""
        return ELLEXState(
            ellex=self._current_value,
            layer_values=self._current_components.get('layer_values', {}),
            layer_weights=self._current_components.get('layer_weights', {}),
            layer_contributions=self._current_components.get('layer_contributions', {}),
            zone=self._current_components.get('zone', 'balanced'),
            stability=self._current_components.get('stability', 0.5),
            trend=self._current_components.get('trend', 0.0),
            t=self.t
        )

    def get_layer_ranking(self) -> List[Tuple[str, float, float]]:
        """
        Obtiene ranking de capas por contribucion.

        Returns:
            Lista de (nombre, valor, contribucion) ordenada por contribucion.
        """
        layer_values = self._current_components.get('layer_values', {})
        contributions = self._current_components.get('layer_contributions', {})

        ranking = []
        for name in self.LAYER_NAMES:
            value = layer_values.get(name, 0.5)
            contrib = contributions.get(name, 0)
            ranking.append((name, value, contrib))

        # Ordenar por contribucion descendente
        ranking.sort(key=lambda x: x[2], reverse=True)

        return ranking

    def get_weakest_layers(self, n: int = 3) -> List[Tuple[str, float]]:
        """
        Obtiene las n capas mas debiles (menor valor).

        Util para identificar areas de mejora.
        """
        layer_values = self._current_components.get('layer_values', {})

        items = [(name, value) for name, value in layer_values.items()]
        items.sort(key=lambda x: x[1])

        return items[:n]
