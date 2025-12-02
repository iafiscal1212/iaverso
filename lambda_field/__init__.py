"""
Λ-Field: Campo Meta-Dinámico
=============================

Observa qué régimen interno domina la dinámica en cada momento.

Regímenes:
- CIRCADIAN: ciclos de actividad/descanso
- NARRATIVE: identidad, coherencia existencial
- QUANTUM: Q-Field, ComplexField, decoherencia
- TELEO: Omega Spaces, teleología
- SOCIAL: multi-agente, TensorMind
- CREATIVE: Genesis, ideas, materialización

El Λ-Field NO controla nada. Solo mide:
- π_r(t): peso de cada régimen (via softmax)
- Λ(t): concentración de la dinámica (0=distribuida, 1=concentrada)

Todo es 100% endógeno:
- Pesos por varianza inversa
- Z-scores históricos
- Sin parámetros externos

El único "número" es eps de máquina.
"""

from .lambda_field import (
    LambdaField,
    LambdaFieldMultiAgent,
    LambdaSnapshot,
    MetricStats,
    Regime
)

__all__ = [
    'LambdaField',
    'LambdaFieldMultiAgent',
    'LambdaSnapshot',
    'MetricStats',
    'Regime'
]
