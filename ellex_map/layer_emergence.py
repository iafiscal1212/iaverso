"""
Emergencia de Capas: Clases Base para Capas Existenciales
=========================================================

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


class TipoCapa(Enum):
    """Tipos de capas existenciales."""
    COGNITIVA = "L1_cognitiva"
    SIMBOLICA = "L2_simbolica"
    NARRATIVA = "L3_narrativa"
    VITAL = "L4_vital"
    SALUD = "L5_salud"
    SOCIAL = "L6_social"
    TENSION = "L7_tension"
    IDENTIDAD = "L8_identidad"
    FASE = "L9_fase"
    INTEGRADA = "L10_integrada"


# Alias para compatibilidad
LayerType = TipoCapa


@dataclass
class EstadoCapa:
    """Estado de una capa existencial."""
    tipo_capa: TipoCapa
    valor: float                    # [0, 1] valor actual
    varianza: float                 # Varianza reciente
    tendencia: float                # Tendencia [-1, 1]
    estabilidad: float              # Estabilidad [0, 1]
    componentes: Dict[str, float]   # Sub-componentes
    t: int                          # Paso temporal

    # Alias para compatibilidad
    @property
    def value(self) -> float:
        return self.valor

    @property
    def layer_type(self) -> TipoCapa:
        return self.tipo_capa


# Alias para compatibilidad
LayerState = EstadoCapa


@dataclass
class HistorialCapa:
    """Historial de una capa para calculos endogenos."""
    valores: List[float] = field(default_factory=list)
    tiempos: List[int] = field(default_factory=list)
    historial_componentes: List[Dict[str, float]] = field(default_factory=list)

    # Alias para compatibilidad
    @property
    def values(self) -> List[float]:
        return self.valores

    @property
    def timestamps(self) -> List[int]:
        return self.tiempos

    def agregar(self, valor: float, t: int, componentes: Dict[str, float] = None):
        """Agrega valor al historial."""
        self.valores.append(valor)
        self.tiempos.append(t)
        if componentes:
            self.historial_componentes.append(componentes)

    # Alias
    def add(self, valor: float, t: int, componentes: Dict[str, float] = None):
        self.agregar(valor, t, componentes)

    def recortar(self, max_len: int):
        """Recorta historial a longitud maxima."""
        if len(self.valores) > max_len:
            self.valores = self.valores[-max_len:]
            self.tiempos = self.tiempos[-max_len:]
            if self.historial_componentes:
                self.historial_componentes = self.historial_componentes[-max_len:]

    # Alias
    def trim(self, max_len: int):
        self.recortar(max_len)

    def obtener_recientes(self, ventana: int) -> List[float]:
        """Obtiene valores recientes."""
        return self.valores[-ventana:] if self.valores else []

    # Alias
    def get_recent(self, ventana: int) -> List[float]:
        return self.obtener_recientes(ventana)

    def obtener_varianza(self, ventana: int = None) -> float:
        """Calcula varianza de valores recientes."""
        if not self.valores:
            return 0.0
        if ventana is None:
            ventana = len(self.valores)
        recientes = self.valores[-ventana:]
        if len(recientes) < 2:
            return 0.0
        return float(np.var(recientes))

    # Alias
    def get_variance(self, ventana: int = None) -> float:
        return self.obtener_varianza(ventana)

    def obtener_tendencia(self, ventana: int = None) -> float:
        """Calcula tendencia (pendiente normalizada)."""
        if not self.valores:
            return 0.0
        if ventana is None:
            ventana = len(self.valores)
        recientes = self.valores[-ventana:]
        if len(recientes) < 3:
            return 0.0

        # Regresion lineal simple
        x = np.arange(len(recientes))
        pendiente = np.polyfit(x, recientes, 1)[0]

        # Normalizar por rango
        rango = max(recientes) - min(recientes) + 1e-8
        normalizada = pendiente / rango

        return float(np.clip(normalizada, -1, 1))

    # Alias
    def get_trend(self, ventana: int = None) -> float:
        return self.obtener_tendencia(ventana)

    def obtener_estabilidad(self, ventana: int = None) -> float:
        """Calcula estabilidad (1 - desviacion_normalizada)."""
        if not self.valores:
            return 1.0
        if ventana is None:
            ventana = len(self.valores)
        recientes = self.valores[-ventana:]
        if len(recientes) < 2:
            return 1.0

        std = np.std(recientes)
        media = np.mean(recientes)

        # Coeficiente de variacion invertido
        cv = std / (media + 1e-8)
        estabilidad = 1 / (1 + cv)

        return float(np.clip(estabilidad, 0, 1))

    # Alias
    def get_stability(self, ventana: int = None) -> float:
        return self.obtener_estabilidad(ventana)


# Alias para compatibilidad
LayerHistory = HistorialCapa


class CapaExistencial(ABC):
    """
    Clase base abstracta para capas existenciales.

    Cada capa:
        - Tiene un tipo (L1-L10)
        - Mantiene historial para calculos endogenos
        - Calcula su valor de observaciones internas
        - Reporta varianza para pesos adaptativos
    """

    def __init__(self, id_agente: str, tipo_capa: TipoCapa):
        """
        Inicializa capa existencial.

        Args:
            id_agente: ID del agente
            tipo_capa: Tipo de capa
        """
        self.id_agente = id_agente
        self.tipo_capa = tipo_capa
        self.historial = HistorialCapa()
        self.t = 0

        # Estado actual
        self._valor_actual = 0.5
        self._componentes_actuales: Dict[str, float] = {}

        # Alias para compatibilidad
        self.agent_id = id_agente
        self.layer_type = tipo_capa
        self.history = self.historial
        self._current_value = self._valor_actual
        self._current_components = self._componentes_actuales

    @abstractmethod
    def calcular(self, observaciones: Dict[str, Any]) -> float:
        """
        Calcula valor de la capa desde observaciones.

        Args:
            observaciones: Metricas observadas del agente

        Returns:
            Valor de coherencia/estado [0, 1]
        """
        pass

    # Alias
    def compute(self, observaciones: Dict[str, Any]) -> float:
        return self.calcular(observaciones)

    def actualizar(self, observaciones: Dict[str, Any]) -> EstadoCapa:
        """
        Actualiza capa con nuevas observaciones.

        Args:
            observaciones: Metricas observadas

        Returns:
            Estado actualizado
        """
        self.t += 1

        # Calcular valor
        valor = self.calcular(observaciones)
        self._valor_actual = valor
        self._current_value = valor

        # Agregar a historial
        self.historial.agregar(valor, self.t, self._componentes_actuales.copy())

        # Recortar historial
        self.historial.recortar(max_history(self.t))

        return self.obtener_estado()

    # Alias
    def update(self, observaciones: Dict[str, Any]) -> EstadoCapa:
        return self.actualizar(observaciones)

    def obtener_estado(self) -> EstadoCapa:
        """Obtiene estado actual de la capa."""
        ventana = L_t(self.t)

        return EstadoCapa(
            tipo_capa=self.tipo_capa,
            valor=self._valor_actual,
            varianza=self.historial.obtener_varianza(ventana),
            tendencia=self.historial.obtener_tendencia(ventana),
            estabilidad=self.historial.obtener_estabilidad(ventana),
            componentes=self._componentes_actuales.copy(),
            t=self.t
        )

    # Alias
    def get_state(self) -> EstadoCapa:
        return self.obtener_estado()

    def obtener_peso(self) -> float:
        """
        Obtiene peso endogeno para agregacion.

        Peso = 1 / (varianza + epsilon)
        Capas mas estables tienen mas peso.
        """
        ventana = L_t(self.t)
        varianza = self.historial.obtener_varianza(ventana)

        # Peso inverso a varianza
        peso = 1.0 / (varianza + 0.01)

        return peso

    # Alias
    def get_weight(self) -> float:
        return self.obtener_peso()

    def obtener_valor(self) -> float:
        """Obtiene valor actual."""
        return self._valor_actual

    # Alias
    def get_value(self) -> float:
        return self.obtener_valor()

    def obtener_estadisticas(self) -> Dict[str, Any]:
        """Estadisticas de la capa."""
        ventana = L_t(self.t)

        return {
            'tipo_capa': self.tipo_capa.value,
            'id_agente': self.id_agente,
            't': self.t,
            'valor': self._valor_actual,
            'varianza': self.historial.obtener_varianza(ventana),
            'tendencia': self.historial.obtener_tendencia(ventana),
            'estabilidad': self.historial.obtener_estabilidad(ventana),
            'peso': self.obtener_peso(),
            'longitud_historial': len(self.historial.valores),
            'componentes': self._componentes_actuales
        }

    # Alias
    def get_statistics(self) -> Dict[str, Any]:
        return self.obtener_estadisticas()


# Alias para compatibilidad
ExistentialLayer = CapaExistencial


def agregar_capas(
    capas: List[CapaExistencial],
    metodo: str = 'varianza_inversa'
) -> float:
    """
    Agrega multiples capas en un valor unico.

    Args:
        capas: Lista de capas
        metodo: Metodo de agregacion
            'varianza_inversa': Peso por 1/var (default)
            'uniforme': Peso uniforme
            'estabilidad': Peso por estabilidad

    Returns:
        Valor agregado [0, 1]
    """
    if not capas:
        return 0.5

    valores = []
    pesos = []

    for capa in capas:
        valores.append(capa.obtener_valor())

        if metodo == 'varianza_inversa' or metodo == 'inverse_variance':
            pesos.append(capa.obtener_peso())
        elif metodo == 'estabilidad' or metodo == 'stability':
            estado = capa.obtener_estado()
            pesos.append(estado.estabilidad)
        else:  # uniforme
            pesos.append(1.0)

    # Normalizar pesos
    peso_total = sum(pesos)
    if peso_total < 1e-8:
        pesos = [1.0 / len(pesos)] * len(pesos)
    else:
        pesos = [p / peso_total for p in pesos]

    # Agregacion ponderada
    agregado = sum(v * p for v, p in zip(valores, pesos))

    return float(np.clip(agregado, 0, 1))


# Alias
def aggregate_layers(capas: List[CapaExistencial], metodo: str = 'inverse_variance') -> float:
    return agregar_capas(capas, metodo)


def calcular_tension_entre_capas(capas: List[CapaExistencial]) -> float:
    """
    Calcula tension entre capas.

    Tension alta = capas muy diferentes entre si.

    Returns:
        Tension [0, 1]
    """
    if len(capas) < 2:
        return 0.0

    valores = [capa.obtener_valor() for capa in capas]

    # Varianza de valores = tension
    varianza = np.var(valores)

    # Normalizar a [0, 1] usando sigmoid
    tension = 2 / (1 + np.exp(-5 * varianza)) - 1

    return float(np.clip(tension, 0, 1))


# Alias
def compute_layer_tension(capas: List[CapaExistencial]) -> float:
    return calcular_tension_entre_capas(capas)
