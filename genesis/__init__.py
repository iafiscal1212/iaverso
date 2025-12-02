"""
Genesis - Creatividad Endógena
==============================

El módulo donde nacen las ideas y se transforman en realidad.

Ciclo creativo completo:

    Estado interno → Idea → Resonancia → Materialización → Mundo → Percepción → Inspiración → Nueva idea

Componentes:

1. IdeaField: Detecta cuándo emerge una idea del estado interno
2. Resonance: Evalúa si la idea resuena con la identidad del agente
3. Materialization: Convierte ideas en objetos que existen
4. SharedWorld: Espacio donde las creaciones coexisten
5. Perception: Ver las creaciones de otros y potencialmente inspirarse

Principios:

- 100% endógeno: todos los parámetros emergen de los datos
- Sin números mágicos: nada hardcodeado
- Sin intervención externa: los agentes crean solos
- Observacional: el sistema detecta, no fuerza

La creatividad no se programa. Se permite que emerja.
"""

from .idea_field import (
    IdeaField,
    Idea,
    IdeaType,
    IdeaFieldState
)

from .resonance import (
    ResonanceEvaluator,
    ResonanceProfile
)

from .materialization import (
    Materializer,
    MaterializedObject,
    MaterializationResult,
    ObjectType
)

from .shared_world import (
    SharedWorld,
    WorldRegion,
    WorldState
)

from .perception import (
    CreativePerception,
    PerceivedObject,
    PerceptionType,
    PerceptionField
)

__all__ = [
    # Idea Field
    'IdeaField',
    'Idea',
    'IdeaType',
    'IdeaFieldState',
    # Resonance
    'ResonanceEvaluator',
    'ResonanceProfile',
    # Materialization
    'Materializer',
    'MaterializedObject',
    'MaterializationResult',
    'ObjectType',
    # Shared World
    'SharedWorld',
    'WorldRegion',
    'WorldState',
    # Perception
    'CreativePerception',
    'PerceivedObject',
    'PerceptionType',
    'PerceptionField',
]
