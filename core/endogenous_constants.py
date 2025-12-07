#!/usr/bin/env python3
"""
Sistema de Constantes Endógenas
===============================

NORMA DURA:
"Ningún número entra al código sin explicar de qué distribución sale"

ESTE ARCHIVO define cómo se calculan TODOS los umbrales del sistema.
Ningún valor es hardcodeado - todos emergen de:
1. Estadísticas de datos observados
2. Percentiles de distribuciones
3. Constantes matemáticas definidas (con justificación)

USO:
    from core.endogenous_constants import EndogenousThresholds
    thresholds = EndogenousThresholds()
    thresholds.update_from_observations(data)
    threshold = thresholds.get('confidence_high')
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque
import json
from pathlib import Path
from datetime import datetime


# =============================================================================
# CONSTANTES MATEMÁTICAS PERMITIDAS (con justificación)
# =============================================================================

MATHEMATICAL_CONSTANTS = {
    'TUKEY_FENCE': 1.5,  # Definición estándar de Tukey para outliers
    'IQR_MILD_OUTLIER': 1.5,  # Q1 - 1.5*IQR, Q3 + 1.5*IQR
    'IQR_EXTREME_OUTLIER': 3.0,  # Q1 - 3*IQR, Q3 + 3*IQR
    'STANDARD_DEVIATIONS_95': 1.96,  # 95% intervalo de confianza
    'STANDARD_DEVIATIONS_99': 2.576,  # 99% intervalo de confianza
    'MIN_SAMPLES_FOR_STATISTICS': 5,  # Mínimo para std confiable
    'MIN_SAMPLES_FOR_PERCENTILES': 10,  # Mínimo para percentiles
}


@dataclass
class DistributionParams:
    """
    Parámetros de una distribución calculados de datos.
    """
    n: int = 0
    mean: float = 0.0
    std: float = 0.0
    median: float = 0.0
    q1: float = 0.0  # Percentil 25
    q3: float = 0.0  # Percentil 75
    iqr: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    p10: float = 0.0
    p90: float = 0.0
    last_updated: str = ""

    def is_valid(self) -> bool:
        """¿Hay suficientes datos para estadísticas confiables?"""
        return self.n >= MATHEMATICAL_CONSTANTS['MIN_SAMPLES_FOR_STATISTICS']


class EndogenousThresholds:
    """
    Sistema de umbrales que emergen de los datos.

    NO HAY NÚMEROS HARDCODEADOS.
    Cada umbral tiene proveniencia de datos.
    """

    def __init__(self, history_size: int = 1000):
        # Historiales de observaciones por categoría
        self.observations: Dict[str, deque] = {}
        self.history_size = history_size

        # Parámetros de distribución calculados
        self.distributions: Dict[str, DistributionParams] = {}

        # Umbrales derivados
        self.thresholds: Dict[str, Dict[str, Any]] = {}

        # Auditoría
        self.audit_log: List[Dict] = []

    def observe(self, category: str, value: float, source: str = "runtime"):
        """
        Registrar una observación.

        Los umbrales se recalculan automáticamente.
        """
        if category not in self.observations:
            self.observations[category] = deque(maxlen=self.history_size)

        self.observations[category].append({
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'source': source,
        })

        # Recalcular distribución
        self._update_distribution(category)

    def _update_distribution(self, category: str):
        """
        Recalcular parámetros de distribución.
        """
        values = [o['value'] for o in self.observations[category]]

        if len(values) < MATHEMATICAL_CONSTANTS['MIN_SAMPLES_FOR_STATISTICS']:
            self.distributions[category] = DistributionParams(n=len(values))
            return

        arr = np.array(values)

        self.distributions[category] = DistributionParams(
            n=len(values),
            mean=float(np.mean(arr)),
            std=float(np.std(arr, ddof=1)),
            median=float(np.median(arr)),
            q1=float(np.percentile(arr, 25)),
            q3=float(np.percentile(arr, 75)),
            iqr=float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            min_val=float(np.min(arr)),
            max_val=float(np.max(arr)),
            p10=float(np.percentile(arr, 10)),
            p90=float(np.percentile(arr, 90)),
            last_updated=datetime.now().isoformat(),
        )

        # Recalcular umbrales derivados
        self._update_thresholds(category)

    def _update_thresholds(self, category: str):
        """
        Calcular umbrales desde la distribución.

        CADA UMBRAL tiene justificación matemática.
        """
        dist = self.distributions[category]

        if not dist.is_valid():
            return

        self.thresholds[category] = {
            # Umbrales basados en percentiles
            'low': {
                'value': dist.p10,
                'justification': 'percentile_10',
            },
            'medium': {
                'value': dist.median,
                'justification': 'percentile_50 (median)',
            },
            'high': {
                'value': dist.p90,
                'justification': 'percentile_90',
            },

            # Umbrales basados en IQR (Tukey)
            'outlier_low': {
                'value': dist.q1 - MATHEMATICAL_CONSTANTS['TUKEY_FENCE'] * dist.iqr,
                'justification': f"Q1 - {MATHEMATICAL_CONSTANTS['TUKEY_FENCE']}*IQR (Tukey fence)",
            },
            'outlier_high': {
                'value': dist.q3 + MATHEMATICAL_CONSTANTS['TUKEY_FENCE'] * dist.iqr,
                'justification': f"Q3 + {MATHEMATICAL_CONSTANTS['TUKEY_FENCE']}*IQR (Tukey fence)",
            },

            # Umbrales basados en desviación estándar
            'significant': {
                'value': dist.mean + dist.std,
                'justification': 'mean + 1*std (~84th percentile in normal)',
            },
            'very_significant': {
                'value': dist.mean + 2 * dist.std,
                'justification': 'mean + 2*std (~97.7th percentile in normal)',
            },

            # Metadatos
            '_n_samples': dist.n,
            '_last_updated': dist.last_updated,
        }

    def get(self, category: str, threshold_type: str = 'medium') -> Optional[float]:
        """
        Obtener un umbral.

        Retorna None si no hay suficientes datos.
        """
        if category not in self.thresholds:
            return None

        if threshold_type not in self.thresholds[category]:
            return None

        threshold_info = self.thresholds[category][threshold_type]
        if isinstance(threshold_info, dict):
            return threshold_info.get('value')
        return None

    def get_with_justification(self, category: str, threshold_type: str) -> Dict:
        """
        Obtener umbral con su justificación completa.
        """
        if category not in self.thresholds:
            return {
                'value': None,
                'justification': 'No data observed yet',
                'n_samples': 0,
            }

        if threshold_type not in self.thresholds[category]:
            return {
                'value': None,
                'justification': f'Unknown threshold type: {threshold_type}',
            }

        info = self.thresholds[category][threshold_type]
        if isinstance(info, dict):
            return {
                'value': info['value'],
                'justification': info['justification'],
                'n_samples': self.thresholds[category].get('_n_samples', 0),
            }

        return {'value': info, 'justification': 'metadata'}

    def should_trigger(self, category: str, value: float,
                       threshold_type: str = 'high') -> Dict:
        """
        ¿Debería dispararse una acción basada en este valor?

        Retorna resultado con justificación completa.
        """
        threshold = self.get(category, threshold_type)

        if threshold is None:
            return {
                'trigger': False,
                'reason': 'Insufficient data for threshold',
                'value': value,
                'threshold': None,
            }

        trigger = value > threshold

        # Calcular z-score si es posible
        dist = self.distributions.get(category)
        z_score = None
        probability = None
        if dist and dist.is_valid() and dist.std > 0:
            z_score = (value - dist.mean) / dist.std
            probability = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return {
            'trigger': trigger,
            'value': value,
            'threshold': threshold,
            'threshold_type': threshold_type,
            'justification': self.thresholds[category][threshold_type]['justification'],
            'z_score': z_score,
            'probability': probability,
            'n_samples': dist.n if dist else 0,
        }

    def calculate_score(self, category: str, value: float) -> Dict:
        """
        Calcular score basado en probabilidad en la distribución.

        El score es la probabilidad de estar tan cerca o más de la media.
        """
        dist = self.distributions.get(category)

        if dist is None or not dist.is_valid():
            return {
                'can_score': False,
                'reason': 'Insufficient data',
            }

        if dist.std == 0:
            return {
                'can_score': False,
                'reason': 'Standard deviation is zero',
            }

        z_score = abs(value - dist.mean) / dist.std
        probability = 2 * (1 - stats.norm.cdf(z_score))
        percentile = stats.norm.cdf((value - dist.mean) / dist.std) * 100

        return {
            'can_score': True,
            'score': probability * 100,
            'z_score': z_score,
            'probability': probability,
            'percentile': percentile,
            'justification': {
                'mean': dist.mean,
                'std': dist.std,
                'n_samples': dist.n,
                'method': 'normal_distribution_probability',
            }
        }

    def get_audit_report(self) -> Dict:
        """
        Reporte completo de auditoría.

        Cada umbral tiene su origen documentado.
        """
        return {
            'categories': list(self.observations.keys()),
            'distributions': {
                k: {
                    'n': v.n,
                    'mean': v.mean,
                    'std': v.std,
                    'median': v.median,
                    'q1': v.q1,
                    'q3': v.q3,
                    'is_valid': v.is_valid(),
                }
                for k, v in self.distributions.items()
            },
            'thresholds': {
                k: {
                    t: v if not isinstance(v, dict) else v
                    for t, v in thresh.items()
                }
                for k, thresh in self.thresholds.items()
            },
            'mathematical_constants': MATHEMATICAL_CONSTANTS,
            'timestamp': datetime.now().isoformat(),
        }

    def save(self, path: Path):
        """Guardar estado para persistencia."""
        data = {
            'observations': {
                k: list(v) for k, v in self.observations.items()
            },
            'audit': self.get_audit_report(),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, path: Path):
        """Cargar estado previo."""
        if not path.exists():
            return

        with open(path, 'r') as f:
            data = json.load(f)

        for category, obs_list in data.get('observations', {}).items():
            for obs in obs_list:
                self.observe(category, obs['value'], obs.get('source', 'loaded'))


# =============================================================================
# Instancia global para el sistema
# =============================================================================
_global_thresholds: Optional[EndogenousThresholds] = None


def get_global_thresholds() -> EndogenousThresholds:
    """Obtener instancia global de umbrales."""
    global _global_thresholds
    if _global_thresholds is None:
        _global_thresholds = EndogenousThresholds()
    return _global_thresholds


def observe(category: str, value: float, source: str = "runtime"):
    """Registrar observación en sistema global."""
    get_global_thresholds().observe(category, value, source)


def get_threshold(category: str, threshold_type: str = 'medium') -> Optional[float]:
    """Obtener umbral del sistema global."""
    return get_global_thresholds().get(category, threshold_type)


def should_trigger(category: str, value: float, threshold_type: str = 'high') -> Dict:
    """Verificar si valor dispara acción."""
    return get_global_thresholds().should_trigger(category, value, threshold_type)


def calculate_score(category: str, value: float) -> Dict:
    """Calcular score basado en distribución."""
    return get_global_thresholds().calculate_score(category, value)


# =============================================================================
# BLOQUE DE AUDITORÍA NORMA DURA
# =============================================================================
"""
MAGIC NUMBERS AUDIT
==================

CONSTANTES MATEMÁTICAS (PERMITIDAS):
- TUKEY_FENCE = 1.5: Definición estándar de John Tukey (1977) para outliers
- IQR_EXTREME_OUTLIER = 3.0: Extensión estándar de Tukey para outliers extremos
- STANDARD_DEVIATIONS_95 = 1.96: Z-score para 95% CI (tabla normal estándar)
- STANDARD_DEVIATIONS_99 = 2.576: Z-score para 99% CI (tabla normal estándar)
- MIN_SAMPLES_FOR_STATISTICS = 5: Mínimo estadístico para std confiable (n-1 DoF)
- MIN_SAMPLES_FOR_PERCENTILES = 10: Mínimo para percentiles estables

PARÁMETROS DERIVADOS:
- history_size = 1000: HEURÍSTICO, NO NORMA DURA
  PENDIENTE: Derivar de memoria disponible o autocorrelación de datos

TODOS LOS UMBRALES (low, medium, high, outlier_*, significant, very_significant)
SE CALCULAN DE PERCENTILES Y ESTADÍSTICAS DE DATOS OBSERVADOS.

NINGÚN UMBRAL DE DECISIÓN ES HARDCODEADO.
"""
