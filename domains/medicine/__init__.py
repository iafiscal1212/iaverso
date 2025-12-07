"""
Medicine domain - clinical data infrastructure.

NO domain rules, only data infrastructure.
"""

from .medicine_connector import (
    MedicineConnector,
    create_clinical_schema,
    create_longitudinal_schema,
    create_imaging_schema,
)

__all__ = [
    "MedicineConnector",
    "create_clinical_schema",
    "create_longitudinal_schema",
    "create_imaging_schema",
]
