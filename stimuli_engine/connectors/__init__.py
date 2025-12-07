"""
CONNECTORS - Conectores Genéricos
==================================

Conectores para diferentes fuentes de datos.
Todos anónimos - no contienen semántica.
"""

from .csv_connector import CSVConnector
from .timeseries_connector import TimeseriesConnector
from .api_connector import APIConnector

__all__ = [
    'CSVConnector',
    'TimeseriesConnector',
    'APIConnector',
]
