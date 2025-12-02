"""
CDE - Cerebro Digital Ético
===========================

Núcleo de Coherencia, Ética y Salud Interna para Sistemas Críticos.

Un componente autónomo, cerrado y ético que:
- Mantiene la coherencia del sistema
- Detecta daño estructural
- Se autocura
- Regula carga, estrés, fases, normas, valores
- Explica decisiones sin exponer datos sensibles
- Trabaja 100% desde dentro, sin tocar el mundo externo

100% endógeno. Sin números mágicos.
"""

from .cde_observer import CDEObserver
from .cde_worldx import WorldX
from .cde_ethics import CDEEthics
from .cde_health import CDEHealth
from .cde_coherence import CDECoherence
from .cde_report import CDEReport
from .cde_daemon import CDEDaemon

__all__ = [
    'CDEObserver',
    'WorldX',
    'CDEEthics',
    'CDEHealth',
    'CDECoherence',
    'CDEReport',
    'CDEDaemon'
]
