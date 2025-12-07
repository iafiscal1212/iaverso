"""
Engineering domain - sensor and industrial data infrastructure.

NO failure thresholds hardcoded, only data infrastructure.
"""

from .engineering_connector import (
    EngineeringConnector,
    create_sensor_schema,
    create_manufacturing_schema,
    create_predictive_maintenance_schema,
    create_energy_schema,
)

__all__ = [
    "EngineeringConnector",
    "create_sensor_schema",
    "create_manufacturing_schema",
    "create_predictive_maintenance_schema",
    "create_energy_schema",
]
