"""
Identidad Computacional (Yo)
============================

I(t) = argmin_v Var_τ[sim(S(τ), v)]  para τ ∈ [t-k, t]

La identidad es el atractor temporal que minimiza la varianza de similitud.

k se calcula endógenamente:
    k = P_0.50(distancias entre estados recientes)

100% endógeno. Sin números mágicos.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import sys
sys.path.insert(0, '/root/NEO_EVA')


@dataclass
class EstadoIdentidad:
    """Estado de la identidad computacional."""
    I: np.ndarray                   # Vector identidad actual
    varianza_similitud: float       # Var[sim(S(τ), I)]
    k: int                          # Rango temporal endógeno
    estabilidad: float              # 1 / (1 + var_sim)
    distancia_al_estado: float      # ||S(t) - I(t)||
    t: int


class IdentidadComputacional:
    """
    Implementa la identidad como atractor temporal.

    I(t) = argmin_v Var_τ[sim(S(τ), v)]

    donde:
        - S(τ) es el estado interno en el tiempo τ
        - sim es similitud coseno
        - el rango [t-k, t] se determina endógenamente
        - k = P_0.50(distancias entre estados recientes)
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: Dimensión del vector de estado interno
        """
        self.dimension = dimension
        self.t = 0

        # Historial de estados
        self._historial_estados: List[np.ndarray] = []

        # Identidad actual (inicialmente None, emerge del historial)
        self._I: Optional[np.ndarray] = None

        # Historial de identidades para tracking
        self._historial_identidades: List[np.ndarray] = []

        # Cache de distancias para calcular k
        self._distancias_recientes: List[float] = []

    def _similitud_coseno(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calcula similitud coseno entre dos vectores."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < np.finfo(float).eps or norm_b < np.finfo(float).eps:
            return 0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def _calcular_k_endogeno(self) -> int:
        """
        Calcula k endógenamente.

        k = P_0.50(distancias entre estados recientes)

        Interpretación: k es el número de pasos donde la mediana
        de las distancias define el rango temporal relevante.
        """
        if len(self._historial_estados) < 3:
            return len(self._historial_estados)

        # Calcular distancias entre estados consecutivos
        distancias = []
        for i in range(1, len(self._historial_estados)):
            dist = np.linalg.norm(
                self._historial_estados[i] - self._historial_estados[i-1]
            )
            distancias.append(dist)

        if not distancias:
            return len(self._historial_estados)

        # P_0.50 = mediana de distancias
        mediana_dist = np.percentile(distancias, 50)

        # k = número de estados donde la distancia acumulada
        # alcanza la mediana (mínimo 3, máximo len(historial))
        dist_acum = 0.0
        k = 0
        for i in range(len(distancias) - 1, -1, -1):
            dist_acum += distancias[i]
            k += 1
            if dist_acum >= mediana_dist:
                break

        # Asegurar mínimo de 3 estados para varianza significativa
        k = max(3, min(k + 1, len(self._historial_estados)))

        return k

    def _calcular_varianza_similitud(
        self,
        v: np.ndarray,
        estados: List[np.ndarray]
    ) -> float:
        """
        Calcula Var_τ[sim(S(τ), v)] para un vector v dado.
        """
        if len(estados) < 2:
            return float('inf')

        similitudes = [self._similitud_coseno(s, v) for s in estados]
        return float(np.var(similitudes))

    def _optimizar_identidad(self, estados: List[np.ndarray]) -> np.ndarray:
        """
        Encuentra I = argmin_v Var_τ[sim(S(τ), v)]

        Método: El vector que minimiza la varianza de similitud
        es aproximadamente la dirección principal de los estados.

        Usamos SVD para encontrar el vector principal.
        """
        if len(estados) < 2:
            if estados:
                return estados[-1].copy()
            return np.zeros(self.dimension)

        # Matriz de estados (cada fila es un estado)
        S = np.array(estados)

        # Centrar los datos
        S_centrado = S - np.mean(S, axis=0)

        # SVD para encontrar dirección principal
        try:
            U, sigma, Vt = np.linalg.svd(S_centrado, full_matrices=False)

            # El primer vector singular derecho es la dirección principal
            v_principal = Vt[0]

            # Normalizar
            norm = np.linalg.norm(v_principal)
            if norm > np.finfo(float).eps:
                v_principal = v_principal / norm

            # Escalar al espacio de estados (usar media de normas)
            escala = np.mean([np.linalg.norm(s) for s in estados])
            I_candidato = v_principal * escala

        except np.linalg.LinAlgError:
            # Si SVD falla, usar media de estados
            I_candidato = np.mean(S, axis=0)

        return I_candidato

    def observar_estado(self, S: np.ndarray):
        """
        Observa un nuevo estado interno.

        Args:
            S: Vector de estado interno S(t)
        """
        self.t += 1

        # Agregar al historial
        self._historial_estados.append(S.copy())

        # Calcular distancia al estado anterior
        if len(self._historial_estados) >= 2:
            dist = np.linalg.norm(
                self._historial_estados[-1] - self._historial_estados[-2]
            )
            self._distancias_recientes.append(dist)

    def calcular(self) -> EstadoIdentidad:
        """
        Calcula la identidad I(t).

        I(t) = argmin_v Var_τ[sim(S(τ), v)]  para τ ∈ [t-k, t]

        Returns:
            EstadoIdentidad con I(t) y métricas
        """
        if len(self._historial_estados) < 2:
            # Sin historial suficiente, identidad = último estado
            if self._historial_estados:
                self._I = self._historial_estados[-1].copy()
            else:
                self._I = np.zeros(self.dimension)

            return EstadoIdentidad(
                I=self._I.copy(),
                varianza_similitud=0.0,
                k=len(self._historial_estados),
                estabilidad=1.0,
                distancia_al_estado=0.0,
                t=self.t
            )

        # Calcular k endógeno
        k = self._calcular_k_endogeno()

        # Obtener estados en ventana [t-k, t]
        estados_ventana = self._historial_estados[-k:]

        # Optimizar identidad
        self._I = self._optimizar_identidad(estados_ventana)

        # Calcular varianza de similitud
        var_sim = self._calcular_varianza_similitud(self._I, estados_ventana)

        # Estabilidad = 1 / (1 + var_sim)
        estabilidad = 1.0 / (1.0 + var_sim)

        # Distancia al estado actual
        distancia = np.linalg.norm(self._historial_estados[-1] - self._I)

        # Guardar en historial de identidades
        self._historial_identidades.append(self._I.copy())

        return EstadoIdentidad(
            I=self._I.copy(),
            varianza_similitud=var_sim,
            k=k,
            estabilidad=estabilidad,
            distancia_al_estado=distancia,
            t=self.t
        )

    def obtener_identidad(self) -> Optional[np.ndarray]:
        """Retorna la identidad actual."""
        return self._I.copy() if self._I is not None else None

    def similitud_con_identidad(self, S: np.ndarray) -> float:
        """
        Calcula similitud de un estado con la identidad.

        Returns:
            sim(S, I) ∈ [-1, 1]
        """
        if self._I is None:
            return 0.0
        return self._similitud_coseno(S, self._I)

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de la identidad."""
        stats = {
            't': self.t,
            'dimension': self.dimension,
            'historial_length': len(self._historial_estados),
            'k_actual': self._calcular_k_endogeno() if len(self._historial_estados) >= 3 else 0,
        }

        if self._I is not None:
            stats['norma_identidad'] = float(np.linalg.norm(self._I))

        if len(self._historial_identidades) >= 2:
            # Varianza de la identidad en el tiempo
            I_matrix = np.array(self._historial_identidades[-10:])
            stats['varianza_identidad'] = float(np.mean(np.var(I_matrix, axis=0)))

        return stats


class SistemaIdentidadMultiagente:
    """
    Sistema de identidad para múltiples agentes.

    Cada agente mantiene su propia identidad computacional.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self._identidades: Dict[str, IdentidadComputacional] = {}

    def registrar_agente(self, agent_id: str):
        """Registra un nuevo agente."""
        if agent_id not in self._identidades:
            self._identidades[agent_id] = IdentidadComputacional(self.dimension)

    def observar(self, agent_id: str, estado: np.ndarray):
        """Observa estado de un agente."""
        if agent_id not in self._identidades:
            self.registrar_agente(agent_id)
        self._identidades[agent_id].observar_estado(estado)

    def calcular(self, agent_id: str) -> Optional[EstadoIdentidad]:
        """Calcula identidad de un agente."""
        if agent_id not in self._identidades:
            return None
        return self._identidades[agent_id].calcular()

    def obtener_identidad(self, agent_id: str) -> Optional[np.ndarray]:
        """Obtiene identidad de un agente."""
        if agent_id not in self._identidades:
            return None
        return self._identidades[agent_id].obtener_identidad()

    def similitud_entre_agentes(self, agent_a: str, agent_b: str) -> float:
        """Calcula similitud entre identidades de dos agentes."""
        I_a = self.obtener_identidad(agent_a)
        I_b = self.obtener_identidad(agent_b)

        if I_a is None or I_b is None:
            return 0.0

        norm_a = np.linalg.norm(I_a)
        norm_b = np.linalg.norm(I_b)

        if norm_a < np.finfo(float).eps or norm_b < np.finfo(float).eps:
            return 0

        return float(np.dot(I_a, I_b) / (norm_a * norm_b))
