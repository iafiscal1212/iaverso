"""
Cosmology domain - astronomical data infrastructure.

NO physical constants hardcoded, only data infrastructure.
Exception: Mathematical constants (pi, e) are allowed as definitions.
"""

from .cosmology_connector import (
    CosmologyConnector,
    create_astronomical_schema,
    create_spectral_schema,
    create_gravitational_wave_schema,
    create_particle_physics_schema,
    PI,
    TAU,
)

__all__ = [
    "CosmologyConnector",
    "create_astronomical_schema",
    "create_spectral_schema",
    "create_gravitational_wave_schema",
    "create_particle_physics_schema",
    "PI",
    "TAU",
]
