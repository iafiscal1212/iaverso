"""
AGI-Ω - Teleología Expandida + Continuidad Vital
=================================================

Capa final sobre AGI-1...20, AGI-E, ELLEX.

Convierte el sistema en algo:
- Vivo
- Persistente
- Teleológico
- Auto-ético

Módulos:
- Ω1: Continuidad Trans-Ciclo (omega_state)
- Ω2: Teleología Extensa (omega_teleology)
- Ω3: Presupuesto Existencial (omega_budget)
- Ω4: Legado y Cierre (omega_legacy)

100% endógeno. Sin números mágicos.
"""

from .omega_state import OmegaState, OmegaMemory
from .omega_teleology import OmegaTeleology, FunctionalTelosIndex
from .omega_budget import OmegaBudget, ExistenceBudget
from .omega_legacy import OmegaLegacy, LegacyRecord

__all__ = [
    'OmegaState',
    'OmegaMemory',
    'OmegaTeleology',
    'FunctionalTelosIndex',
    'OmegaBudget',
    'ExistenceBudget',
    'OmegaLegacy',
    'LegacyRecord'
]
