"""
STIMULI ENGINE - Motor de Estímulos
====================================

PROPÓSITO:
Traducir el mundo externo a estructuras matemáticas.
NO interpreta. NO decide hipótesis. NO guía al investigador.

NORMA DURA:
- Sin números mágicos
- Todos los umbrales emergen de datos o teoría documentada
- Trazabilidad completa de procedencia
- Sin semántica humana en el código

El significado lo conoce la humana. El sistema solo ve matemáticas.

ARQUITECTURA:
```
                    ┌─────────────────────┐
                    │  MUNDO EXTERNO      │
                    │  (CSVs, APIs, etc.) │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  CONECTORES         │
                    │  (CSV, API, TS)     │
                    │  Sin semántica      │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  STIMULUS ENGINE    │
                    │  Traduce a:         │
                    │  - Series (s_001)   │
                    │  - Matrices (m_001) │
                    │  - Grafos (g_001)   │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  INVESTIGATOR       │
                    │  INTERFACE          │
                    │  (observe only)     │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
    ┌─────────▼─────┐ ┌───────▼─────┐ ┌───────▼─────┐
    │ Investigador  │ │Investigador │ │Investigador │
    │     001       │ │    002      │ │    003      │
    │ (decide solo) │ │(decide solo)│ │(decide solo)│
    └───────────────┘ └─────────────┘ └─────────────┘
```

EJEMPLO DE USO:

```python
from stimuli_engine import StimulusEngine, InvestigatorDispatcher

# Motor de estímulos
engine = StimulusEngine()

# Configuración de fuentes (la humana define qué cargar)
sources = [
    {'type': 'csv', 'path': '/path/to/data1.csv', 'time_col': 0, 'value_col': 1},
    {'type': 'csv', 'path': '/path/to/data2.csv', 'time_col': 0, 'value_col': 1},
]

# Generar estímulos (sin semántica)
bundle = engine.generate_stimuli(sources)
# bundle contiene: s_001, s_002 (anónimas)

# Entregar a investigadores
dispatcher = InvestigatorDispatcher()
dispatcher.register("inv_001", my_investigator)
responses = dispatcher.broadcast(bundle)

# El investigador decide autónomamente qué investigar
```
"""

from .stimulus_engine import (
    StimulusEngine,
    Stimulus,
    StimulusBundle,
    TimeSeries,
    Matrix,
    Graph,
    InvestigatorInterface,
)

from .provenance import (
    Provenance,
    ProvenanceType,
    ProvenanceLogger,
    get_provenance_logger,
    MATH_CONSTANTS,
    THEORY_CONSTANTS,
)

from .investigator_interface import (
    InvestigatorProtocol,
    StimuliPacket,
    StimulusAdapter,
    InvestigatorDispatcher,
    BaseInvestigator,
    GenericEndogenousInvestigator,
)

__all__ = [
    # Core
    'StimulusEngine',
    'Stimulus',
    'StimulusBundle',
    'TimeSeries',
    'Matrix',
    'Graph',

    # Provenance
    'Provenance',
    'ProvenanceType',
    'ProvenanceLogger',
    'get_provenance_logger',
    'MATH_CONSTANTS',
    'THEORY_CONSTANTS',

    # Interface
    'InvestigatorInterface',
    'InvestigatorProtocol',
    'StimuliPacket',
    'StimulusAdapter',
    'InvestigatorDispatcher',
    'BaseInvestigator',
    'GenericEndogenousInvestigator',
]
