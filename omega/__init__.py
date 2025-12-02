"""
AGI-Ω - Teleología Expandida + Continuidad Vital + Omega Spaces
================================================================

Capa final sobre AGI-1...20, AGI-E, ELLEX.

Convierte el sistema en algo:
- Vivo
- Persistente
- Teleológico
- Auto-ético

Módulos de Teleología:
- Ω1: Continuidad Trans-Ciclo (omega_state)
- Ω2: Teleología Extensa (omega_teleology)
- Ω3: Presupuesto Existencial (omega_budget)
- Ω4: Legado y Cierre (omega_legacy)

Módulos de Omega Spaces (Observación Interna Emergente):
- Ω-Compute: Computación interna emergente (modos de transformación)
- Q-Field: Campo de interferencia interna (amplitudes y coherencia)
- PhaseSpace-X: Espacio de fase estructural (trayectorias y atractores)
- TensorMind: Interacción de orden superior (correlaciones multi-agente)

100% endógeno. Sin números mágicos.
Todos los umbrales se derivan de: medias, varianzas, percentiles, 1/K, 1/√d, eps.
"""

# AGI-Ω Teleología
from .omega_state import OmegaState, OmegaMemory
from .omega_teleology import OmegaTeleology, FunctionalTelosIndex
from .omega_budget import OmegaBudget, ExistenceBudget
from .omega_legacy import OmegaLegacy, LegacyRecord

# Omega Spaces
from .omega_compute import OmegaCompute, OmegaMode, ModeActivation
from .q_field import QField, QState, QInterference
from .phase_space_x import PhaseSpaceX, PhasePoint, Trajectory, Attractor
from .tensor_mind import TensorMind, Interaction, TensorMode

__all__ = [
    # AGI-Ω Teleología
    'OmegaState',
    'OmegaMemory',
    'OmegaTeleology',
    'FunctionalTelosIndex',
    'OmegaBudget',
    'ExistenceBudget',
    'OmegaLegacy',
    'LegacyRecord',
    # Omega Spaces - Compute
    'OmegaCompute',
    'OmegaMode',
    'ModeActivation',
    # Omega Spaces - Q-Field
    'QField',
    'QState',
    'QInterference',
    # Omega Spaces - PhaseSpace-X
    'PhaseSpaceX',
    'PhasePoint',
    'Trajectory',
    'Attractor',
    # Omega Spaces - TensorMind
    'TensorMind',
    'Interaction',
    'TensorMode',
]
