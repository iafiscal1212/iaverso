#!/usr/bin/env python3
"""
WEAVER Multiscale Views
=======================

Vistas multi-escala temporales del sistema.
100% ENDÓGENO - Sin constantes mágicas

Escalas: W_k = √t, 2√t, 4√t
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ScaleView:
    """Vista a una escala temporal específica."""
    scale_idx: int
    window_size: int
    S_mean: float
    S_std: float
    S_trend: float  # Pendiente de tendencia
    z_centroid: np.ndarray
    z_dispersion: float


class MultiscaleViews:
    """
    Vistas multi-escala del sistema.

    100% Endógeno:
    - Escalas W_k = √t, 2√t, 4√t
    - Estadísticas derivadas de la historia
    - Tendencias por regresión lineal
    """

    def __init__(self, n_scales: int = 3):
        self.n_scales = n_scales

        # Historia por escala
        self.views_history: List[List[ScaleView]] = []

        # Buffer de datos
        self.S_buffer: List[float] = []
        self.z_buffer: List[np.ndarray] = []

    def _get_window_sizes(self, t: int) -> List[int]:
        """Calcula tamaños de ventana endógenos: √t, 2√t, 4√t."""
        sqrt_t = max(1, int(np.sqrt(t + 1)))
        return [sqrt_t * (2 ** k) for k in range(self.n_scales)]

    def update(self, z: np.ndarray, S: float) -> List[ScaleView]:
        """
        Actualiza vistas multi-escala con nuevo dato.

        Returns:
            Lista de vistas para cada escala
        """
        self.S_buffer.append(S)
        self.z_buffer.append(z.copy())

        t = len(self.S_buffer)
        windows = self._get_window_sizes(t)

        views = []
        for k, w in enumerate(windows):
            view = self._compute_scale_view(k, w)
            views.append(view)

        self.views_history.append(views)
        return views

    def _compute_scale_view(self, scale_idx: int, window: int) -> ScaleView:
        """Computa vista para una escala específica."""
        # Ajustar ventana a datos disponibles
        actual_window = min(window, len(self.S_buffer))

        if actual_window == 0:
            return ScaleView(
                scale_idx=scale_idx,
                window_size=0,
                S_mean=0.0,
                S_std=0.0,
                S_trend=0.0,
                z_centroid=np.zeros(3),
                z_dispersion=0.0
            )

        # Extraer datos de ventana
        S_window = self.S_buffer[-actual_window:]
        z_window = self.z_buffer[-actual_window:]

        # Estadísticas de S
        S_mean = np.mean(S_window)
        S_std = np.std(S_window) if len(S_window) > 1 else 0.0

        # Tendencia (regresión lineal simple)
        if len(S_window) > 1:
            x = np.arange(len(S_window))
            S_trend = np.polyfit(x, S_window, 1)[0]
        else:
            S_trend = 0.0

        # Centroide de z
        z_array = np.array(z_window)
        z_centroid = np.mean(z_array, axis=0)

        # Dispersión de z (distancia media al centroide)
        if len(z_window) > 1:
            distances = [np.linalg.norm(z - z_centroid) for z in z_window]
            z_dispersion = np.mean(distances)
        else:
            z_dispersion = 0.0

        return ScaleView(
            scale_idx=scale_idx,
            window_size=actual_window,
            S_mean=S_mean,
            S_std=S_std,
            S_trend=S_trend,
            z_centroid=z_centroid,
            z_dispersion=z_dispersion
        )

    def get_cross_scale_features(self) -> Dict[str, Any]:
        """
        Extrae features que comparan entre escalas.

        100% endógeno: ratios entre escalas
        """
        if not self.views_history:
            return {}

        current_views = self.views_history[-1]

        if len(current_views) < 2:
            return {}

        features = {}

        # Ratios de tendencia entre escalas
        for k in range(len(current_views) - 1):
            v1 = current_views[k]
            v2 = current_views[k + 1]

            # Ratio de variabilidad
            if v2.S_std > 0:
                features[f'var_ratio_{k}_{k+1}'] = v1.S_std / v2.S_std
            else:
                features[f'var_ratio_{k}_{k+1}'] = 1.0

            # Diferencia de tendencia (multi-escala)
            features[f'trend_diff_{k}_{k+1}'] = v1.S_trend - v2.S_trend

            # Ratio de dispersión z
            if v2.z_dispersion > 0:
                features[f'disp_ratio_{k}_{k+1}'] = v1.z_dispersion / v2.z_dispersion
            else:
                features[f'disp_ratio_{k}_{k+1}'] = 1.0

        # Coherencia multi-escala: correlación de tendencias
        trends = [v.S_trend for v in current_views]
        features['trend_coherence'] = 1.0 - np.std(trends) / (np.mean(np.abs(trends)) + 1e-10)

        return features

    def get_dominant_scale(self) -> int:
        """
        Identifica escala dominante (mayor varianza explicada).

        Returns:
            Índice de escala dominante
        """
        if not self.views_history:
            return 0

        current_views = self.views_history[-1]
        variances = [v.S_std ** 2 for v in current_views]

        return int(np.argmax(variances))

    def to_dict(self) -> Dict[str, Any]:
        """Serializa estado a diccionario."""
        if not self.views_history:
            return {'n_updates': 0, 'views': []}

        current = self.views_history[-1]
        return {
            'n_updates': len(self.views_history),
            'views': [
                {
                    'scale': v.scale_idx,
                    'window': v.window_size,
                    'S_mean': v.S_mean,
                    'S_std': v.S_std,
                    'S_trend': v.S_trend,
                    'z_dispersion': v.z_dispersion
                }
                for v in current
            ],
            'dominant_scale': self.get_dominant_scale(),
            'cross_scale': self.get_cross_scale_features()
        }
