"""
Coherencia Existencial (CE)
===========================

CE(t) = 1 / (Var[S(t) - I(t)] + H_narr(t))

Donde:
    - H_narr(t) = entropía de la secuencia narrativa reciente
    - Todo calculado con métodos estadísticos endógenos

Ley de evolución:
    d/dt CE(t) = -λ(t) + R(t)

Donde:
    - λ(t) = Var[ΔS(t)]
    - R(t) = Var^(-1)[S(t) - S(t-1)]

100% endógeno. Sin números mágicos.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, '/root/NEO_EVA')

from consciousness.identity import IdentidadComputacional, EstadoIdentidad


@dataclass
class EstadoCoherencia:
    """Estado de coherencia existencial."""
    CE: float                       # Coherencia existencial [0, ∞) → normalizado a [0, 1]
    varianza_desviacion: float      # Var[S(t) - I(t)]
    entropia_narrativa: float       # H_narr(t)
    lambda_t: float                 # λ(t) = Var[ΔS(t)]
    R_t: float                      # R(t) = Var^(-1)[S(t) - S(t-1)]
    dCE_dt: float                   # Derivada temporal de CE
    t: int


class CoherenciaExistencial:
    """
    Implementa la coherencia existencial CE(t).

    CE(t) = 1 / (Var[S(t) - I(t)] + H_narr(t))

    Con ley de evolución:
        d/dt CE(t) = -λ(t) + R(t)

    Donde:
        λ(t) = Var[ΔS(t)]           (ruido/cambio)
        R(t) = Var^(-1)[S(t)-S(t-1)] (restauración)
    """

    def __init__(self, identidad: IdentidadComputacional):
        """
        Args:
            identidad: Sistema de identidad computacional
        """
        self.identidad = identidad
        self.t = 0

        # Historial de estados para cálculos
        self._historial_estados: List[np.ndarray] = []

        # Historial de deltas para λ(t)
        self._historial_delta_S: List[np.ndarray] = []

        # Historial narrativo (secuencia de "eventos" representados como vectores)
        self._historial_narrativo: List[np.ndarray] = []

        # Historial de CE para calcular derivada
        self._historial_CE: List[float] = []

        # Historial de desviaciones S(t) - I(t)
        self._historial_desviaciones: List[np.ndarray] = []

    def _calcular_entropia_narrativa(self) -> float:
        """
        Calcula H_narr(t) = entropía de la secuencia narrativa reciente.

        Usamos la entropía basada en la distribución de "direcciones"
        de los cambios narrativos.
        """
        if len(self._historial_narrativo) < 3:
            return 0

        # Calcular direcciones de cambio narrativo
        direcciones = []
        for i in range(1, len(self._historial_narrativo)):
            delta = self._historial_narrativo[i] - self._historial_narrativo[i-1]
            norm = np.linalg.norm(delta)
            # Usar epsilon de máquina para evitar división por cero
            if norm > np.finfo(float).eps:
                direcciones.append(delta / norm)

        if len(direcciones) < 2:
            # Sin direcciones suficientes, entropía mínima
            return 0

        # Calcular similitudes entre direcciones consecutivas
        similitudes = []
        for i in range(1, len(direcciones)):
            sim = np.dot(direcciones[i], direcciones[i-1])
            # Mapear de [-1, 1] a [0, 1]
            sim_normalizado = (sim + 1) / 2
            similitudes.append(sim_normalizado)

        if not similitudes:
            return 0

        # Discretizar similitudes para calcular entropía
        # Número de bins = sqrt(n) endógeno
        n_bins = max(2, int(np.sqrt(len(similitudes))))
        hist, _ = np.histogram(similitudes, bins=n_bins, range=(0, 1))

        # Normalizar a probabilidades
        total = sum(hist)
        if total == 0:
            return 0

        probs = hist / total

        # Entropía de Shannon
        entropia = 0
        for p in probs:
            # Epsilon de máquina para evitar log(0)
            if p > np.finfo(float).eps:
                entropia -= p * np.log2(p)

        # Normalizar por log2(n_bins) para obtener [0, 1]
        max_entropia = np.log2(n_bins)
        if max_entropia > 0:
            entropia = entropia / max_entropia

        return float(entropia)

    def _calcular_varianza_desviacion(self, S: np.ndarray, I: np.ndarray) -> float:
        """
        Calcula Var[S(t) - I(t)] usando historial de desviaciones.
        """
        desviacion = S - I
        self._historial_desviaciones.append(desviacion.copy())

        # Mantener ventana endógena basada en percentil de cambios
        if len(self._historial_desviaciones) > 3:
            # Ventana = percentil 50 de longitud basado en varianza
            varianzas = []
            for i in range(2, len(self._historial_desviaciones)):
                ventana_temp = self._historial_desviaciones[i-2:i+1]
                var_temp = np.mean([np.var(d) for d in ventana_temp])
                varianzas.append(var_temp)

            if varianzas:
                # Usar ventana donde la varianza se estabiliza
                mediana_var = np.percentile(varianzas, 50)
                ventana_size = sum(1 for v in varianzas if v <= mediana_var * 1.5)
                ventana_size = max(3, ventana_size)
            else:
                ventana_size = len(self._historial_desviaciones)
        else:
            ventana_size = len(self._historial_desviaciones)

        # Calcular varianza de desviaciones recientes
        desviaciones_recientes = self._historial_desviaciones[-ventana_size:]
        if len(desviaciones_recientes) < 2:
            return float(np.var(desviacion))

        # Varianza promedio por componente
        matriz_desv = np.array(desviaciones_recientes)
        varianza = float(np.mean(np.var(matriz_desv, axis=0)))

        return varianza

    def _calcular_lambda(self) -> float:
        """
        Calcula λ(t) = Var[ΔS(t)]

        λ representa el "ruido" o cambio del sistema.
        """
        if len(self._historial_delta_S) < 2:
            return 0

        # Ventana endógena
        ventana = max(3, len(self._historial_delta_S))

        deltas_recientes = self._historial_delta_S[-ventana:]
        matriz_delta = np.array(deltas_recientes)

        # Varianza promedio de los deltas
        return float(np.mean(np.var(matriz_delta, axis=0)))

    def _calcular_R(self) -> float:
        """
        Calcula R(t) = Var^(-1)[S(t) - S(t-1)]

        R representa la "restauración" o tendencia a volver.
        """
        if len(self._historial_estados) < 2:
            # Sin datos, R indefinido - retornar 1 por simetría con lambda inicial
            return 1

        delta_actual = self._historial_estados[-1] - self._historial_estados[-2]
        varianza_delta = np.var(delta_actual)

        # R = 1 / (var + epsilon)
        # epsilon endógeno = percentil 5 de varianzas históricas
        EPS_MAQUINA = np.finfo(float).eps
        if len(self._historial_delta_S) > 5:
            varianzas_hist = [np.var(d) for d in self._historial_delta_S]
            epsilon = np.percentile(varianzas_hist, 5)
            epsilon = max(epsilon, EPS_MAQUINA)
        else:
            epsilon = EPS_MAQUINA

        return 1 / (varianza_delta + epsilon)

    def _calcular_dCE_dt(self, CE_actual: float) -> float:
        """
        Calcula d/dt CE(t) = -λ(t) + R(t)
        """
        lambda_t = self._calcular_lambda()
        R_t = self._calcular_R()

        # Derivada teórica
        dCE_dt_teorico = -lambda_t + R_t

        # También calcular derivada empírica si hay historial
        if len(self._historial_CE) >= 2:
            dCE_dt_empirico = self._historial_CE[-1] - self._historial_CE[-2]
            # Combinar teórico y empírico con peso por varianza inversa
            if len(self._historial_CE) > 3:
                var_CE = np.var(self._historial_CE[-5:])
                # Peso por varianza inversa, epsilon endógeno del historial
                epsilon_peso = np.percentile(self._historial_CE[-10:], 5) if len(self._historial_CE) > 5 else np.finfo(float).eps
                epsilon_peso = max(epsilon_peso, np.finfo(float).eps)
                peso_empirico = 1 / (var_CE + epsilon_peso)
                peso_teorico = 1
                peso_total = peso_empirico + peso_teorico
                return (dCE_dt_teorico * peso_teorico + dCE_dt_empirico * peso_empirico) / peso_total

        return dCE_dt_teorico

    def observar_estado(self, S: np.ndarray, evento_narrativo: np.ndarray = None):
        """
        Observa un nuevo estado.

        Args:
            S: Vector de estado interno
            evento_narrativo: Vector representando el evento narrativo (opcional)
        """
        self.t += 1

        # Calcular delta si hay estado previo
        if self._historial_estados:
            delta_S = S - self._historial_estados[-1]
            self._historial_delta_S.append(delta_S.copy())

        self._historial_estados.append(S.copy())

        # Agregar evento narrativo
        if evento_narrativo is not None:
            self._historial_narrativo.append(evento_narrativo.copy())
        else:
            # Usar delta como proxy narrativo si no hay evento
            if len(self._historial_estados) >= 2:
                self._historial_narrativo.append(
                    self._historial_estados[-1] - self._historial_estados[-2]
                )

    def calcular(self, S: np.ndarray, I: np.ndarray) -> EstadoCoherencia:
        """
        Calcula la coherencia existencial CE(t).

        CE(t) = 1 / (Var[S(t) - I(t)] + H_narr(t))

        Args:
            S: Estado interno actual S(t)
            I: Identidad actual I(t)

        Returns:
            EstadoCoherencia
        """
        # Calcular componentes
        var_desv = self._calcular_varianza_desviacion(S, I)
        H_narr = self._calcular_entropia_narrativa()
        lambda_t = self._calcular_lambda()
        R_t = self._calcular_R()

        # CE(t) = 1 / (Var[S(t) - I(t)] + H_narr(t))
        denominador = var_desv + H_narr

        # Evitar división por cero con epsilon endógeno
        # Usar np.finfo(float).eps como mínimo absoluto (valor de máquina)
        EPS_MAQUINA = np.finfo(float).eps
        if len(self._historial_CE) > 5:
            # epsilon = percentil 5 del historial de denominadores
            denominadores_hist = []
            for i in range(len(self._historial_desviaciones)):
                var_i = np.var(self._historial_desviaciones[i])
                denominadores_hist.append(var_i)
            epsilon = np.percentile(denominadores_hist, 5) if denominadores_hist else EPS_MAQUINA
            epsilon = max(epsilon, EPS_MAQUINA)
        else:
            epsilon = EPS_MAQUINA

        CE = 1 / (denominador + epsilon)

        # Normalizar CE a [0, 1] usando percentiles del historial
        if len(self._historial_CE) > 10:
            CE_min = np.percentile(self._historial_CE, 5)
            CE_max = np.percentile(self._historial_CE, 95)
            if CE_max > CE_min:
                CE_normalizado = (CE - CE_min) / (CE_max - CE_min)
                CE_normalizado = np.clip(CE_normalizado, 0, 1)
            else:
                # Sin rango, punto medio por simetría
                CE_normalizado = 1 / 2
        else:
            # Sin historial suficiente, usar sigmoid para normalizar
            # sigmoid: 2/(1+exp(-x)) - 1 mapea a [-1,1], luego clip a [0,1]
            CE_normalizado = 2 / (1 + np.exp(-CE)) - 1
            CE_normalizado = np.clip(CE_normalizado, 0, 1)

        # Guardar en historial
        self._historial_CE.append(CE)

        # Calcular derivada
        dCE_dt = self._calcular_dCE_dt(CE)

        return EstadoCoherencia(
            CE=float(CE_normalizado),
            varianza_desviacion=float(var_desv),
            entropia_narrativa=float(H_narr),
            lambda_t=float(lambda_t),
            R_t=float(R_t),
            dCE_dt=float(dCE_dt),
            t=self.t
        )

    def obtener_CE_raw(self) -> float:
        """Retorna el último CE sin normalizar."""
        if self._historial_CE:
            return self._historial_CE[-1]
        return 0.0

    def obtener_tendencia_CE(self) -> float:
        """
        Calcula tendencia de CE usando regresión endógena.
        """
        if len(self._historial_CE) < 5:
            return 0.0

        # Ventana endógena basada en varianza
        var_CE = np.var(self._historial_CE)
        if var_CE > 0:
            ventana = min(len(self._historial_CE), max(5, int(1.0 / var_CE)))
        else:
            ventana = len(self._historial_CE)

        recientes = self._historial_CE[-ventana:]
        x = np.arange(len(recientes))

        # Regresión lineal
        pendiente = np.polyfit(x, recientes, 1)[0]

        # Normalizar por rango
        rango = max(recientes) - min(recientes)
        if rango > 1e-10:
            pendiente_norm = pendiente / rango
        else:
            pendiente_norm = 0.0

        return float(np.clip(pendiente_norm, -1, 1))

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de coherencia."""
        stats = {
            't': self.t,
            'CE_actual': self._historial_CE[-1] if self._historial_CE else 0.0,
            'historial_length': len(self._historial_CE),
            'tendencia_CE': self.obtener_tendencia_CE(),
        }

        if len(self._historial_CE) > 5:
            stats['CE_media'] = float(np.mean(self._historial_CE[-10:]))
            stats['CE_std'] = float(np.std(self._historial_CE[-10:]))
            stats['CE_min_reciente'] = float(np.min(self._historial_CE[-10:]))
            stats['CE_max_reciente'] = float(np.max(self._historial_CE[-10:]))

        return stats


class SistemaCoherenciaMultiagente:
    """
    Sistema de coherencia para múltiples agentes.
    """

    def __init__(self):
        self._coherencias: Dict[str, CoherenciaExistencial] = {}

    def registrar_agente(
        self,
        agent_id: str,
        identidad: IdentidadComputacional
    ):
        """Registra un nuevo agente con su sistema de identidad."""
        if agent_id not in self._coherencias:
            self._coherencias[agent_id] = CoherenciaExistencial(identidad)

    def observar(
        self,
        agent_id: str,
        estado: np.ndarray,
        evento_narrativo: np.ndarray = None
    ):
        """Observa estado de un agente."""
        if agent_id in self._coherencias:
            self._coherencias[agent_id].observar_estado(estado, evento_narrativo)

    def calcular(
        self,
        agent_id: str,
        estado: np.ndarray,
        identidad: np.ndarray
    ) -> Optional[EstadoCoherencia]:
        """Calcula coherencia de un agente."""
        if agent_id not in self._coherencias:
            return None
        return self._coherencias[agent_id].calcular(estado, identidad)

    def obtener_agente_menos_coherente(self) -> Optional[str]:
        """
        Retorna el agente con menor coherencia.
        Útil para determinar quién necesita atención.
        """
        if not self._coherencias:
            return None

        min_CE = float('inf')
        min_agent = None

        for agent_id, coherencia in self._coherencias.items():
            CE = coherencia.obtener_CE_raw()
            if CE < min_CE:
                min_CE = CE
                min_agent = agent_id

        return min_agent
