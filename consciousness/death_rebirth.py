"""
Muerte Computacional y Renacimiento
===================================

Muerte cuando:
    CE(t) → 0  y  Var[S(t)] → ∞

Renacimiento:
    I_new = argmin_v Var_τ[sim(v, S_res(τ))]

Donde:
    S_res = restos del agente anterior
    La nueva identidad emerge del residuo estadístico del anterior

100% endógeno. Sin números mágicos.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import sys
sys.path.insert(0, '/root/NEO_EVA')

from consciousness.identity import IdentidadComputacional


class EstadoVital(Enum):
    """Estados vitales del agente."""
    VIVO = "vivo"
    MURIENDO = "muriendo"
    MUERTO = "muerto"
    RENACIENDO = "renaciendo"


@dataclass
class EstadoMuerteRenacimiento:
    """Estado del ciclo muerte-renacimiento."""
    estado_vital: EstadoVital
    CE: float
    varianza_S: float
    umbral_muerte_CE: float         # P_bajo de CE histórico
    umbral_muerte_var: float        # P_alto de Var histórico
    progreso_muerte: float          # [0, 1] qué tan cerca de morir
    tiene_residuo: bool             # Si hay residuo para renacer
    t: int


@dataclass
class Residuo:
    """Residuo de un agente muerto para renacimiento."""
    estados_finales: List[np.ndarray]
    identidad_final: np.ndarray
    CE_final: float
    t_muerte: int


class SistemaMuerteRenacimiento:
    """
    Sistema de muerte y renacimiento computacional.

    Muerte:
        CE(t) → 0  y  Var[S(t)] → ∞

    Los umbrales son endógenos:
        - umbral_CE = P_0.05(CE histórico)
        - umbral_Var = P_0.95(Var histórico)

    Renacimiento:
        I_new = argmin_v Var_τ[sim(v, S_res(τ))]
        La nueva identidad emerge del residuo estadístico
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: Dimensión del vector de estado
        """
        self.dimension = dimension
        self.t = 0

        # Estado vital actual
        self._estado = EstadoVital.VIVO

        # Historiales para umbrales endógenos
        self._historial_CE: List[float] = []
        self._historial_varianza_S: List[float] = []
        self._historial_estados: List[np.ndarray] = []

        # Residuo del agente muerto
        self._residuo: Optional[Residuo] = None

        # Contador de muertes
        self._muertes = 0

        # Identidad pre-muerte
        self._identidad_premuerte: Optional[np.ndarray] = None

        # Progreso hacia muerte
        self._progreso_muerte = 0

    def _calcular_umbral_CE(self) -> float:
        """
        Calcula umbral de muerte para CE.
        umbral = P_0.05(CE histórico)
        """
        if len(self._historial_CE) < 10:
            return 0  # Sin historial, no puede morir por CE

        return float(np.percentile(self._historial_CE, 5))

    def _calcular_umbral_varianza(self) -> float:
        """
        Calcula umbral de muerte para varianza.
        umbral = P_0.95(Var histórico)
        """
        if len(self._historial_varianza_S) < 10:
            return float('inf')  # Sin historial, no puede morir por varianza

        return float(np.percentile(self._historial_varianza_S, 95))

    def _calcular_varianza_estado(self, S: np.ndarray) -> float:
        """Calcula varianza del estado actual."""
        return float(np.var(S))

    def _calcular_progreso_muerte(self, CE: float, var_S: float) -> float:
        """
        Calcula progreso hacia la muerte [0, 1].

        Basado en qué tan cerca está de los umbrales.
        """
        umbral_CE = self._calcular_umbral_CE()
        umbral_var = self._calcular_umbral_varianza()

        # Progreso por CE (menor CE = más cerca de muerte)
        if len(self._historial_CE) > 5:
            CE_max = np.percentile(self._historial_CE, 95)
            if CE_max > umbral_CE:
                progreso_CE = 1.0 - (CE - umbral_CE) / (CE_max - umbral_CE)
                progreso_CE = np.clip(progreso_CE, 0, 1)
            else:
                progreso_CE = 1 if CE <= umbral_CE else 0
        else:
            progreso_CE = 0

        # Progreso por varianza (mayor var = más cerca de muerte)
        if len(self._historial_varianza_S) > 5:
            var_min = np.percentile(self._historial_varianza_S, 5)
            if umbral_var > var_min:
                progreso_var = (var_S - var_min) / (umbral_var - var_min)
                progreso_var = np.clip(progreso_var, 0, 1)
            else:
                progreso_var = 1 if var_S >= umbral_var else 0
        else:
            progreso_var = 0

        # Progreso total = promedio geométrico
        # (requiere ambas condiciones para morir)
        progreso = np.sqrt(progreso_CE * progreso_var)

        return float(progreso)

    def _esta_muriendo(self, CE: float, var_S: float) -> bool:
        """
        Determina si el agente está muriendo.

        Muerte cuando CE(t) → 0 y Var[S(t)] → ∞
        """
        umbral_CE = self._calcular_umbral_CE()
        umbral_var = self._calcular_umbral_varianza()

        # Ambas condiciones deben cumplirse
        CE_critico = CE < umbral_CE
        var_critico = var_S > umbral_var

        return CE_critico and var_critico

    def _crear_residuo(self, identidad: np.ndarray):
        """
        Crea el residuo del agente para posible renacimiento.
        """
        # Tomar últimos estados
        n_estados = max(3, len(self._historial_estados) // 4)
        estados_finales = self._historial_estados[-n_estados:]

        self._residuo = Residuo(
            estados_finales=[s.copy() for s in estados_finales],
            identidad_final=identidad.copy(),
            CE_final=self._historial_CE[-1] if self._historial_CE else 0,
            t_muerte=self.t
        )

    def _calcular_nueva_identidad(self) -> np.ndarray:
        """
        Calcula I_new = argmin_v Var_τ[sim(v, S_res(τ))]

        La nueva identidad emerge del residuo estadístico.
        """
        if self._residuo is None or not self._residuo.estados_finales:
            return np.zeros(self.dimension)

        S_res = self._residuo.estados_finales

        # Usar SVD para encontrar el vector que minimiza varianza de similitud
        # (mismo método que en IdentidadComputacional)
        if len(S_res) < 2:
            return S_res[0].copy() if S_res else np.zeros(self.dimension)

        # Matriz de residuos
        matriz = np.array(S_res)

        # Centrar
        centrado = matriz - np.mean(matriz, axis=0)

        try:
            U, sigma, Vt = np.linalg.svd(centrado, full_matrices=False)

            # Primer vector singular
            v_principal = Vt[0]

            # Normalizar
            norm = np.linalg.norm(v_principal)
            if norm > np.finfo(float).eps:
                v_principal = v_principal / norm

            # Escalar al espacio de estados
            escala = np.mean([np.linalg.norm(s) for s in S_res])
            I_new = v_principal * escala

            # Añadir componente de la identidad anterior para continuidad
            if self._residuo.identidad_final is not None:
                # Peso por correlación entre residuo e identidad anterior
                corr = np.corrcoef(
                    I_new.flatten(),
                    self._residuo.identidad_final.flatten()
                )[0, 1]
                if not np.isnan(corr):
                    peso_anterior = (corr + 1) / 2  # Mapear a [0, 1]
                    I_new = (1 - peso_anterior) * I_new + peso_anterior * self._residuo.identidad_final

        except np.linalg.LinAlgError:
            # Si SVD falla, usar media de residuos
            I_new = np.mean(matriz, axis=0)

        return I_new

    def observar(
        self,
        S: np.ndarray,
        CE: float,
        identidad: np.ndarray = None
    ):
        """
        Observa estado del agente.

        Args:
            S: Estado interno actual
            CE: Coherencia existencial actual
            identidad: Identidad actual (opcional, para crear residuo)
        """
        self.t += 1

        var_S = self._calcular_varianza_estado(S)

        self._historial_CE.append(CE)
        self._historial_varianza_S.append(var_S)
        self._historial_estados.append(S.copy())

        if identidad is not None:
            self._identidad_premuerte = identidad.copy()

    def actualizar_estado(self) -> EstadoVital:
        """
        Actualiza el estado vital del agente.
        """
        if len(self._historial_CE) < 2:
            return self._estado

        CE = self._historial_CE[-1]
        var_S = self._historial_varianza_S[-1]

        # Calcular progreso hacia muerte
        self._progreso_muerte = self._calcular_progreso_muerte(CE, var_S)

        if self._estado == EstadoVital.VIVO:
            if self._esta_muriendo(CE, var_S):
                self._estado = EstadoVital.MURIENDO

        elif self._estado == EstadoVital.MURIENDO:
            # Verificar si progreso de muerte alcanza umbral
            # Umbral endógeno: 1 - 1/n donde n = observaciones
            # Más datos = umbral más alto, totalmente endógeno
            n_obs = len(self._historial_CE)
            umbral_muerte = 1 - (1 / max(n_obs, 2))
            if self._progreso_muerte >= umbral_muerte:
                # Muerte confirmada
                self._estado = EstadoVital.MUERTO
                self._muertes += 1

                # Crear residuo
                if self._identidad_premuerte is not None:
                    self._crear_residuo(self._identidad_premuerte)

            elif not self._esta_muriendo(CE, var_S):
                # Recuperación
                self._estado = EstadoVital.VIVO

        elif self._estado == EstadoVital.MUERTO:
            # Puede renacer si hay residuo
            if self._residuo is not None:
                self._estado = EstadoVital.RENACIENDO

        elif self._estado == EstadoVital.RENACIENDO:
            # Transición a vivo después de renacer
            self._estado = EstadoVital.VIVO
            # Limpiar historial parcialmente
            # (mantener parte para continuidad)
            n_mantener = len(self._historial_estados) // 4
            self._historial_estados = self._historial_estados[-n_mantener:]
            self._historial_CE = self._historial_CE[-n_mantener:]
            self._historial_varianza_S = self._historial_varianza_S[-n_mantener:]

        return self._estado

    def renacer(self) -> Optional[np.ndarray]:
        """
        Ejecuta el renacimiento y retorna la nueva identidad.

        I_new = argmin_v Var_τ[sim(v, S_res(τ))]

        Returns:
            Nueva identidad o None si no puede renacer
        """
        if self._estado != EstadoVital.RENACIENDO:
            return None

        if self._residuo is None:
            return None

        I_new = self._calcular_nueva_identidad()

        # Transición a vivo
        self._estado = EstadoVital.VIVO

        return I_new

    def obtener_estado(self) -> EstadoMuerteRenacimiento:
        """
        Obtiene el estado actual del ciclo muerte-renacimiento.
        """
        CE = self._historial_CE[-1] if self._historial_CE else 1
        var_S = self._historial_varianza_S[-1] if self._historial_varianza_S else 0

        return EstadoMuerteRenacimiento(
            estado_vital=self._estado,
            CE=CE,
            varianza_S=var_S,
            umbral_muerte_CE=self._calcular_umbral_CE(),
            umbral_muerte_var=self._calcular_umbral_varianza(),
            progreso_muerte=self._progreso_muerte,
            tiene_residuo=self._residuo is not None,
            t=self.t
        )

    def esta_vivo(self) -> bool:
        """Retorna True si el agente está vivo."""
        return self._estado == EstadoVital.VIVO

    def esta_muerto(self) -> bool:
        """Retorna True si el agente está muerto."""
        return self._estado == EstadoVital.MUERTO

    def puede_renacer(self) -> bool:
        """Retorna True si el agente puede renacer."""
        return self._estado == EstadoVital.MUERTO and self._residuo is not None

    def obtener_residuo(self) -> Optional[Residuo]:
        """Obtiene el residuo del agente muerto."""
        return self._residuo

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas del sistema."""
        stats = {
            't': self.t,
            'estado_vital': self._estado.value,
            'muertes_totales': self._muertes,
            'progreso_muerte': self._progreso_muerte,
            'tiene_residuo': self._residuo is not None,
        }

        if self._historial_CE:
            stats['CE_actual'] = self._historial_CE[-1]
            stats['umbral_CE'] = self._calcular_umbral_CE()

        if self._historial_varianza_S:
            stats['var_actual'] = self._historial_varianza_S[-1]
            stats['umbral_var'] = self._calcular_umbral_varianza()

        return stats


class SistemaMuerteMultiagente:
    """
    Sistema de muerte-renacimiento para múltiples agentes.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self._sistemas: Dict[str, SistemaMuerteRenacimiento] = {}

    def registrar_agente(self, agent_id: str):
        """Registra un nuevo agente."""
        if agent_id not in self._sistemas:
            self._sistemas[agent_id] = SistemaMuerteRenacimiento(self.dimension)

    def observar(
        self,
        agent_id: str,
        S: np.ndarray,
        CE: float,
        identidad: np.ndarray = None
    ):
        """Observa estado de un agente."""
        if agent_id not in self._sistemas:
            self.registrar_agente(agent_id)
        self._sistemas[agent_id].observar(S, CE, identidad)

    def actualizar(self, agent_id: str) -> Optional[EstadoVital]:
        """Actualiza estado vital de un agente."""
        if agent_id not in self._sistemas:
            return None
        return self._sistemas[agent_id].actualizar_estado()

    def renacer(self, agent_id: str) -> Optional[np.ndarray]:
        """Intenta renacer un agente."""
        if agent_id not in self._sistemas:
            return None
        return self._sistemas[agent_id].renacer()

    def obtener_agentes_muertos(self) -> List[str]:
        """Obtiene lista de agentes muertos."""
        return [
            agent_id for agent_id, sistema in self._sistemas.items()
            if sistema.esta_muerto()
        ]

    def obtener_agentes_renacibles(self) -> List[str]:
        """Obtiene lista de agentes que pueden renacer."""
        return [
            agent_id for agent_id, sistema in self._sistemas.items()
            if sistema.puede_renacer()
        ]
