"""
ELLEX-MAP: Existential Life Layer Explorer
===========================================

El mapa vivo definitivo de coherencia, tension, identidad,
ritmos, medicina y simbolos.

10 CAPAS EXISTENCIALES:
    L1: Coherencia Cognitiva Interna (AGI-X)
    L2: Coherencia Simbolica (SYM-X + STX)
    L3: Coherencia Narrativa (episodic + narrative)
    L4: Coherencia Vital (LX1-LX10)
    L5: Salud Interior (MED-X)
    L6: Coherencia Social (SX5 + AGI-19)
    L7: Tension Existencial
    L8: Estabilidad de Identidad (AGI-20)
    L9: Equilibrio de Fases (circadian)
    L10: Integracion Total (ELLEX Index)

100% ENDOGENO:
    - Pesos por varianza inversa
    - Thresholds por percentiles
    - Ventanas por L_t = sqrt(t)
    - Sin numeros magicos

ELLEX = sum(w_i * C_i) donde w_i = 1/Var(C_i)
"""

# Layer system
from .layer_emergence import (
    ExistentialLayer,
    LayerState,
    LayerHistory,
)

# Coherence surfaces
from .coherence_surface import (
    CoherenceSurface,
    CognitiveCoherence,
    SymbolicCoherence,
    NarrativeCoherence,
    LifeCoherence,
    SocialCoherence,
)

# Tension and identity
from .existential_tension import (
    ExistentialTension,
    TensionState,
)

# Health
from .health_equilibrium import (
    HealthEquilibrium,
    HealthState,
)

# Circadian
from .circadian_phase_space import (
    CircadianPhaseSpace,
    PhaseEquilibrium,
)

# Narrative
from .narrative_waveform import (
    NarrativeWaveform,
    WaveformState,
)

# Symbolic
from .symbolic_cohesion import (
    SymbolicCohesion,
    CohesionState,
)

# ELLEX Index
from .ellex_index import (
    ELLEXIndex,
    ELLEXState,
)

# Main orchestrator
from .ellex_map import (
    ELLEXMap,
    ELLEXMapState,
    IdentityLayer,
    IdentityState,
)

# Visualizer
from .ellex_visualizer import (
    ELLEXVisualizer,
    VisualizationData,
)

__all__ = [
    # Layers
    'ExistentialLayer',
    'LayerState',
    'LayerHistory',
    # Coherences
    'CoherenceSurface',
    'CognitiveCoherence',
    'SymbolicCoherence',
    'NarrativeCoherence',
    'LifeCoherence',
    'SocialCoherence',
    # Tension
    'ExistentialTension',
    'TensionState',
    # Health
    'HealthEquilibrium',
    'HealthState',
    # Circadian
    'CircadianPhaseSpace',
    'PhaseEquilibrium',
    # Narrative
    'NarrativeWaveform',
    'WaveformState',
    # Symbolic
    'SymbolicCohesion',
    'CohesionState',
    # Index
    'ELLEXIndex',
    'ELLEXState',
    # Map
    'ELLEXMap',
    'ELLEXMapState',
    'IdentityLayer',
    'IdentityState',
    # Visualizer
    'ELLEXVisualizer',
    'VisualizationData',
]
