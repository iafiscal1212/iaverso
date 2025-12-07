"""
SISTEMA DE DOMINIOS - NEO_EVA

NORMA DURA EXTENDIDA:
=====================
Claude es el ARQUITECTO de infraestructura, NO el experto de dominio.
Los agentes APRENDEN de los datos, no de reglas hardcodeadas.

Dominios disponibles:
- medicine: Datos clínicos, laboratorio, imaging
- finance: Mercados, fundamentales, opciones, crypto
- cosmology: Astronómico, espectral, ondas gravitacionales
- engineering: Sensores, manufactura, mantenimiento predictivo

Uso básico:
-----------
    from domains import get_engine

    engine = get_engine()
    data = engine.load_data("medicine", "synthetic", n_samples=1000)
    result = engine.analyze(data["data_key"], "correlation")

Agregar nuevo dominio:
----------------------
    1. Crear directorio domains/nuevo_dominio/
    2. Crear nuevo_dominio_connector.py con:
       - Schema(s) definiendo variables
       - Connector heredando de DomainConnector
    3. Registrar en domain_engine.py

PROHIBIDO en conectores:
- Reglas de dominio (if X > threshold then diagnosis)
- Umbrales hardcodeados
- Conocimiento experto embebido
- Constantes físicas/médicas/financieras predefinidas

PERMITIDO en conectores:
- Definición de variables y sus unidades
- Carga de datos de múltiples fuentes
- Cálculos matemáticos genéricos
- Transformaciones de coordenadas/unidades
"""

from .domain_engine import DomainEngine, get_engine, DomainRegistry
from .core.domain_base import (
    DomainSchema,
    DomainConnector,
    DomainAnalyzer,
    VariableDefinition,
    VariableType,
    VariableRole,
    Hypothesis,
    HypothesisEngine,
)

__all__ = [
    # Motor principal
    "DomainEngine",
    "get_engine",
    "DomainRegistry",

    # Base classes
    "DomainSchema",
    "DomainConnector",
    "DomainAnalyzer",
    "VariableDefinition",
    "VariableType",
    "VariableRole",

    # Hipótesis
    "Hypothesis",
    "HypothesisEngine",
]

__version__ = "1.0.0"
