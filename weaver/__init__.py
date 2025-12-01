"""
WEAVER: World Emergence via Autonomous VEctor Reasoning
========================================================

Orquestador global del sistema NEO_EVA.

Módulos:
- global_state: Estado compartido entre fases
- multiscale_views: Vistas multi-escala temporales
- phase_graph: Grafo de dependencias de fases con Transfer Entropy
- indices: Índices globales MSI, SCI, EGI
"""

from .global_state import GlobalState
from .multiscale_views import MultiscaleViews
from .phase_graph import PhaseGraph
from .indices import GlobalIndices
from .orchestrator import WeaverOrchestrator

__all__ = [
    'GlobalState',
    'MultiscaleViews',
    'PhaseGraph',
    'GlobalIndices',
    'WeaverOrchestrator'
]
