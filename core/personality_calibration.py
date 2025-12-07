#!/usr/bin/env python3
"""
Sistema de Calibración de Personalidades - NORMA DURA
=====================================================

Este módulo proporciona umbrales para rasgos de personalidad
basados en:
1. Distribución uniforme U(0,1) como prior
2. Percentiles para clasificación (p10, p25, p50, p75, p90)

NORMA DURA: No hay números mágicos.
Todos los umbrales son percentiles de U(0,1).

Para calibración real, este sistema debería ser alimentado con:
- Datos de estudios de personalidad (Big Five, OCEAN)
- Distribuciones observadas en poblaciones reales
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class TraitThresholds:
    """
    Umbrales para clasificar un rasgo.

    Basados en percentiles de distribución.
    """
    very_low: float   # p10
    low: float        # p25
    medium: float     # p50 (mediana)
    high: float       # p75
    very_high: float  # p90


class PersonalityCalibration:
    """
    Sistema de calibración de personalidades.

    NORMA DURA: Todos los umbrales son percentiles.
    """

    def __init__(self):
        # Datos observados de población para calibración
        self._observed_traits: Dict[str, list] = {}

        # Umbrales calibrados
        self._calibrated_thresholds: Dict[str, TraitThresholds] = {}

        # Usar umbrales de distribución uniforme U(0,1) como prior
        # ORIGEN: Percentiles de U(0,1)
        self._uniform_thresholds = TraitThresholds(
            very_low=0.1,    # ORIGEN: percentil 10 de U(0,1)
            low=0.25,        # ORIGEN: percentil 25 de U(0,1)
            medium=0.5,      # ORIGEN: percentil 50 de U(0,1)
            high=0.75,       # ORIGEN: percentil 75 de U(0,1)
            very_high=0.9    # ORIGEN: percentil 90 de U(0,1)
        )

    def observe_trait(self, trait_name: str, value: float):
        """Registrar observación de un rasgo."""
        if trait_name not in self._observed_traits:
            self._observed_traits[trait_name] = []
        self._observed_traits[trait_name].append(value)

        # Recalibrar si hay suficientes datos
        # ORIGEN: MIN_SAMPLES = 10 (mínimo para percentiles)
        MIN_SAMPLES = 10
        if len(self._observed_traits[trait_name]) >= MIN_SAMPLES:
            self._calibrate_trait(trait_name)

    def _calibrate_trait(self, trait_name: str):
        """Calibrar umbrales basado en observaciones."""
        values = np.array(self._observed_traits[trait_name])

        self._calibrated_thresholds[trait_name] = TraitThresholds(
            very_low=float(np.percentile(values, 10)),
            low=float(np.percentile(values, 25)),
            medium=float(np.percentile(values, 50)),
            high=float(np.percentile(values, 75)),
            very_high=float(np.percentile(values, 90))
        )

    def get_thresholds(self, trait_name: str) -> TraitThresholds:
        """
        Obtener umbrales para un rasgo.

        Si no hay calibración, usa prior uniforme.
        """
        if trait_name in self._calibrated_thresholds:
            return self._calibrated_thresholds[trait_name]
        return self._uniform_thresholds

    def classify_trait(self, trait_name: str, value: float) -> str:
        """
        Clasificar un valor de rasgo.

        Returns: 'very_low', 'low', 'medium', 'high', 'very_high'
        """
        thresholds = self.get_thresholds(trait_name)

        if value < thresholds.very_low:
            return 'very_low'
        elif value < thresholds.low:
            return 'low'
        elif value < thresholds.high:
            return 'medium'
        elif value < thresholds.very_high:
            return 'high'
        else:
            return 'very_high'

    def is_high(self, trait_name: str, value: float) -> bool:
        """¿El valor está por encima del p75?"""
        thresholds = self.get_thresholds(trait_name)
        return value > thresholds.high

    def is_low(self, trait_name: str, value: float) -> bool:
        """¿El valor está por debajo del p25?"""
        thresholds = self.get_thresholds(trait_name)
        return value < thresholds.low

    def is_very_high(self, trait_name: str, value: float) -> bool:
        """¿El valor está por encima del p90?"""
        thresholds = self.get_thresholds(trait_name)
        return value > thresholds.very_high

    def is_very_low(self, trait_name: str, value: float) -> bool:
        """¿El valor está por debajo del p10?"""
        thresholds = self.get_thresholds(trait_name)
        return value < thresholds.very_low

    def get_audit_report(self) -> Dict:
        """Reporte de auditoría."""
        return {
            'n_traits_observed': len(self._observed_traits),
            'n_traits_calibrated': len(self._calibrated_thresholds),
            'traits': {
                name: len(values) for name, values in self._observed_traits.items()
            },
            'thresholds': {
                name: {
                    'very_low': t.very_low,
                    'low': t.low,
                    'medium': t.medium,
                    'high': t.high,
                    'very_high': t.very_high,
                }
                for name, t in self._calibrated_thresholds.items()
            },
            'uniform_prior': {
                'very_low': 0.1,
                'low': 0.25,
                'medium': 0.5,
                'high': 0.75,
                'very_high': 0.9,
                'origin': 'percentiles of U(0,1)'
            }
        }


# Instancia global para el sistema
_global_personality_calibration: PersonalityCalibration = None


def get_personality_calibration() -> PersonalityCalibration:
    """Obtener instancia global."""
    global _global_personality_calibration
    if _global_personality_calibration is None:
        _global_personality_calibration = PersonalityCalibration()
    return _global_personality_calibration


# =============================================================================
# BLOQUE DE AUDITORÍA NORMA DURA
# =============================================================================
"""
MAGIC NUMBERS AUDIT
==================

UMBRALES DE PRIOR (basados en U(0,1)):
- very_low = 0.10: ORIGEN: percentil 10 de U(0,1)
- low = 0.25: ORIGEN: percentil 25 de U(0,1) (Q1)
- medium = 0.50: ORIGEN: percentil 50 de U(0,1) (mediana)
- high = 0.75: ORIGEN: percentil 75 de U(0,1) (Q3)
- very_high = 0.90: ORIGEN: percentil 90 de U(0,1)

CONSTANTES:
- MIN_SAMPLES = 10: ORIGEN: Mínimo para percentiles estables

CALIBRACIÓN:
- Los umbrales se recalculan como percentiles de datos observados
- Mientras no haya datos, se usa el prior uniforme

TODAS LAS DECISIONES TIENEN ORIGEN DOCUMENTADO.
"""
