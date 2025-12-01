#!/usr/bin/env python3
"""
WEAVER Global Indices
=====================

Índices globales del sistema:
- MSI: Multi-Scale Integration
- SCI: Structural Coherence Index
- EGI: Emergent Goal Index

100% ENDÓGENO - Sin constantes mágicas
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class IndexValues:
    """Valores actuales de índices globales."""
    MSI: float  # Multi-Scale Integration
    SCI: float  # Structural Coherence Index
    EGI: float  # Emergent Goal Index
    timestamp: int


class GlobalIndices:
    """
    Calcula y mantiene índices globales del sistema.

    100% Endógeno:
    - MSI = correlación entre vistas de distintas escalas
    - SCI = coherencia de dependencias (basado en TE)
    - EGI = consistencia de goals estructurales
    """

    def __init__(self):
        # Historia de índices
        self.MSI_history: List[float] = []
        self.SCI_history: List[float] = []
        self.EGI_history: List[float] = []

        # Valores actuales
        self.current: Optional[IndexValues] = None

        # Buffers para cálculo
        self.scale_data: List[List[float]] = [[], [], []]  # 3 escalas
        self.coherence_data: List[float] = []
        self.goal_data: List[float] = []

    def update_scale_data(self, scale_idx: int, value: float) -> None:
        """Actualiza datos de una escala temporal."""
        if 0 <= scale_idx < len(self.scale_data):
            self.scale_data[scale_idx].append(value)

    def update_coherence_data(self, value: float) -> None:
        """Actualiza datos de coherencia."""
        self.coherence_data.append(value)

    def update_goal_data(self, value: float) -> None:
        """Actualiza datos de goals."""
        self.goal_data.append(value)

    def compute_MSI(self) -> float:
        """
        Calcula Multi-Scale Integration.

        MSI = correlación media entre pares de escalas
        100% endógeno
        """
        n_scales = len(self.scale_data)
        correlations = []

        for i in range(n_scales):
            for j in range(i + 1, n_scales):
                X = self.scale_data[i]
                Y = self.scale_data[j]

                min_len = min(len(X), len(Y))
                if min_len < 3:
                    continue

                X = np.array(X[-min_len:])
                Y = np.array(Y[-min_len:])

                try:
                    corr = np.corrcoef(X, Y)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                except Exception:
                    pass

        if not correlations:
            return 0.0

        return float(np.mean(correlations))

    def compute_SCI(self) -> float:
        """
        Calcula Structural Coherence Index.

        SCI = 1 - variabilidad de coherencia (normalizada)
        100% endógeno
        """
        if len(self.coherence_data) < 2:
            return 1.0

        data = np.array(self.coherence_data)
        mean_val = np.mean(data)
        std_val = np.std(data)

        # SCI alto = poca variabilidad relativa
        if mean_val > 0:
            cv = std_val / mean_val  # Coeficiente de variación
            sci = 1.0 / (1.0 + cv)  # Transformar a [0, 1]
        else:
            sci = 0.0

        return float(sci)

    def compute_EGI(self) -> float:
        """
        Calcula Emergent Goal Index.

        EGI = tendencia de goals + consistencia
        100% endógeno
        """
        if len(self.goal_data) < 3:
            return 0.0

        data = np.array(self.goal_data)

        # Tendencia (regresión lineal)
        x = np.arange(len(data))
        try:
            slope = np.polyfit(x, data, 1)[0]
        except Exception:
            slope = 0.0

        # Consistencia (1 - variabilidad)
        std_val = np.std(data)
        mean_val = np.mean(np.abs(data))
        if mean_val > 0:
            consistency = 1.0 / (1.0 + std_val / mean_val)
        else:
            consistency = 0.0

        # EGI = combinación de tendencia positiva y consistencia
        trend_component = np.tanh(slope)  # [-1, 1]
        trend_component = (trend_component + 1) / 2  # [0, 1]

        egi = 0.5 * trend_component + 0.5 * consistency

        return float(egi)

    def compute_all(self, t: int) -> IndexValues:
        """Calcula todos los índices."""
        msi = self.compute_MSI()
        sci = self.compute_SCI()
        egi = self.compute_EGI()

        self.MSI_history.append(msi)
        self.SCI_history.append(sci)
        self.EGI_history.append(egi)

        self.current = IndexValues(
            MSI=msi,
            SCI=sci,
            EGI=egi,
            timestamp=t
        )

        return self.current

    def get_trends(self) -> Dict[str, float]:
        """
        Calcula tendencias de cada índice.

        100% endógeno: regresión sobre historia
        """
        trends = {}

        for name, history in [
            ('MSI', self.MSI_history),
            ('SCI', self.SCI_history),
            ('EGI', self.EGI_history)
        ]:
            if len(history) < 3:
                trends[f'{name}_trend'] = 0.0
            else:
                x = np.arange(len(history))
                try:
                    trends[f'{name}_trend'] = float(np.polyfit(x, history, 1)[0])
                except Exception:
                    trends[f'{name}_trend'] = 0.0

        return trends

    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumen de índices."""
        if self.current is None:
            return {'computed': False}

        return {
            'computed': True,
            'MSI': self.current.MSI,
            'SCI': self.current.SCI,
            'EGI': self.current.EGI,
            'MSI_mean': np.mean(self.MSI_history) if self.MSI_history else 0.0,
            'SCI_mean': np.mean(self.SCI_history) if self.SCI_history else 0.0,
            'EGI_mean': np.mean(self.EGI_history) if self.EGI_history else 0.0,
            'trends': self.get_trends()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            'current': {
                'MSI': self.current.MSI if self.current else 0.0,
                'SCI': self.current.SCI if self.current else 0.0,
                'EGI': self.current.EGI if self.current else 0.0
            },
            'history_length': len(self.MSI_history),
            'summary': self.get_summary()
        }
