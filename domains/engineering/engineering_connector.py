"""
DOMINIO: INGENIERÍA / SISTEMAS

NORMA DURA EXTENDIDA:
- NO hay umbrales de fallo hardcodeados
- NO hay reglas de mantenimiento predefinidas
- SOLO infraestructura para que agentes aprendan de datos

El agente que use esto APRENDERÁ de sensores, logs, métricas,
y formulará hipótesis que serán falsificadas contra datos reales.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
from datetime import datetime
import numpy as np

from ..core.domain_base import (
    DomainSchema, DomainConnector, VariableDefinition,
    VariableType, VariableRole
)


# =============================================================================
# SCHEMA: Define QUÉ variables existen, NO reglas de ingeniería
# =============================================================================

def create_sensor_schema() -> DomainSchema:
    """
    Schema para datos de sensores genéricos.

    NOTA: No definimos qué es "normal" o "anómalo".
    Eso lo aprende el agente de los datos históricos.
    """
    variables = [
        # Identificación
        VariableDefinition(
            name="timestamp",
            var_type=VariableType.TEMPORAL,
            role=VariableRole.INDEX,
            description="Momento de la medición"
        ),
        VariableDefinition(
            name="sensor_id",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.INDEX,
            description="Identificador del sensor"
        ),
        VariableDefinition(
            name="equipment_id",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.COVARIATE,
            description="Identificador del equipo"
        ),

        # Variables físicas genéricas
        VariableDefinition(
            name="temperature",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="°C",
            description="Temperatura"
        ),
        VariableDefinition(
            name="pressure",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="bar",
            description="Presión"
        ),
        VariableDefinition(
            name="vibration",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="mm/s",
            description="Vibración"
        ),
        VariableDefinition(
            name="current",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="A",
            description="Corriente eléctrica"
        ),
        VariableDefinition(
            name="voltage",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="V",
            description="Voltaje"
        ),
        VariableDefinition(
            name="rpm",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="rev/min",
            description="Revoluciones por minuto"
        ),
        VariableDefinition(
            name="flow_rate",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="m³/h",
            description="Caudal"
        ),

        # Estado (etiquetado por operadores, no reglas)
        VariableDefinition(
            name="operational_status",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.OUTCOME,
            categories=["normal", "warning", "fault", "maintenance"],
            description="Estado operacional (etiquetado)"
        ),
    ]

    return DomainSchema(
        domain_name="engineering_sensor",
        version="1.0.0",
        variables=variables,
        metadata={
            "tipo": "sensores_genéricos",
            "nota": "El agente aprende umbrales de datos históricos"
        }
    )


def create_manufacturing_schema() -> DomainSchema:
    """Schema para datos de manufactura/producción."""
    variables = [
        VariableDefinition(
            name="batch_id",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.INDEX,
            description="Identificador de lote"
        ),
        VariableDefinition(
            name="timestamp",
            var_type=VariableType.TEMPORAL,
            role=VariableRole.INDEX,
            description="Momento de producción"
        ),
        VariableDefinition(
            name="product_type",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.COVARIATE,
            description="Tipo de producto"
        ),
        VariableDefinition(
            name="machine_id",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.COVARIATE,
            description="Máquina de producción"
        ),
        VariableDefinition(
            name="cycle_time",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="seconds",
            description="Tiempo de ciclo"
        ),
        VariableDefinition(
            name="units_produced",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            description="Unidades producidas"
        ),
        VariableDefinition(
            name="defect_count",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            description="Conteo de defectos"
        ),
        VariableDefinition(
            name="energy_consumption",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="kWh",
            description="Consumo energético"
        ),
        VariableDefinition(
            name="quality_score",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.OUTCOME,
            description="Score de calidad (0-1)"
        ),
    ]

    return DomainSchema(
        domain_name="engineering_manufacturing",
        version="1.0.0",
        variables=variables,
        metadata={"tipo": "manufactura"}
    )


# =============================================================================
# CONNECTOR: Carga datos de diferentes fuentes
# =============================================================================

class EngineeringConnector(DomainConnector):
    """
    Conector para datos de ingeniería/sensores.

    Soporta múltiples formatos y protocolos industriales.
    NO interpreta los datos - solo los carga y calcula derivados.
    """

    def __init__(self):
        self.sensor_schema = create_sensor_schema()
        self.manufacturing_schema = create_manufacturing_schema()
        super().__init__(schema=self.sensor_schema)

    def load_data(self, source: str, **kwargs) -> np.ndarray:
        """Carga datos desde una fuente."""
        if source == "csv":
            result = self.load_csv(Path(kwargs["path"]), kwargs.get("column_mapping"))
            return result["data"].values if hasattr(result["data"], 'values') else result["data"]
        elif source == "synthetic":
            result = self.load_synthetic_for_testing(
                kwargs.get("n_samples", 1000),
                kwargs.get("seed"),
                kwargs.get("include_anomalies", True),
                kwargs.get("anomaly_fraction", 0.05)
            )
            return result["data"].values if hasattr(result["data"], 'values') else result["data"]
        else:
            raise ValueError(f"Fuente '{source}' no soportada")

    def get_available_sources(self) -> List[str]:
        """Lista fuentes de datos disponibles."""
        return ["csv", "synthetic", "influxdb", "opcua"]

    def load_csv(self, path: Path, column_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Carga datos desde CSV."""
        import pandas as pd

        df = pd.read_csv(path)

        if column_mapping:
            df = df.rename(columns=column_mapping)

        # Convertir timestamp si existe
        for col in ['timestamp', 'time', 'datetime']:
            if col in df.columns:
                df['timestamp'] = pd.to_datetime(df[col])
                if col != 'timestamp':
                    df = df.drop(columns=[col])
                break

        return {
            "data": df,
            "source": str(path),
            "n_records": len(df),
        }

    def load_from_influxdb(
        self,
        host: str,
        database: str,
        measurement: str,
        time_range: Tuple[str, str],
        credentials: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Carga datos desde InfluxDB.

        Requiere: influxdb-client instalado
        """
        try:
            from influxdb_client import InfluxDBClient
        except ImportError:
            raise ImportError("Instalar influxdb-client: pip install influxdb-client")

        # Placeholder - implementación real dependería de la versión de InfluxDB
        raise NotImplementedError("Implementar para versión específica de InfluxDB")

    def load_from_opcua(
        self,
        endpoint: str,
        node_ids: List[str],
        time_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """
        Carga datos desde servidor OPC-UA.

        Requiere: opcua-asyncio instalado
        """
        try:
            from asyncua import Client
        except ImportError:
            raise ImportError("Instalar asyncua: pip install asyncua")

        raise NotImplementedError("Implementar cliente OPC-UA")

    def load_synthetic_for_testing(
        self,
        n_samples: int,
        seed: Optional[int] = None,
        include_anomalies: bool = True,
        anomaly_fraction: float = 0.05
    ) -> Dict[str, Any]:
        """
        Genera datos sintéticos para testing.

        Args:
            n_samples: Número de muestras
            seed: Semilla para reproducibilidad
            include_anomalies: Si incluir anomalías sintéticas
            anomaly_fraction: Fracción de anomalías (solo para testing)

        IMPORTANTE: Estos son datos de prueba, NO representan
        comportamiento real de sistemas.
        """
        import pandas as pd

        if seed is not None:
            np.random.seed(seed)

        # Generar datos base con distribuciones genéricas
        n_sensors = 5  # ORIGEN: número arbitrario para testing

        data = {
            "timestamp": pd.date_range(
                start="2024-01-01",
                periods=n_samples,
                freq="1min"
            ),
            "sensor_id": np.random.choice([f"SENS_{i:03d}" for i in range(n_sensors)], n_samples),
            "equipment_id": np.random.choice(["EQ_A", "EQ_B", "EQ_C"], n_samples),
            "temperature": np.random.normal(loc=50, scale=5, size=n_samples),
            "pressure": np.random.normal(loc=10, scale=1, size=n_samples),
            "vibration": np.abs(np.random.normal(loc=2, scale=0.5, size=n_samples)),
            "current": np.random.normal(loc=100, scale=10, size=n_samples),
            "voltage": np.random.normal(loc=220, scale=5, size=n_samples),
            "rpm": np.random.normal(loc=1500, scale=50, size=n_samples),
            "flow_rate": np.abs(np.random.normal(loc=10, scale=2, size=n_samples)),
            "operational_status": ["normal"] * n_samples,
        }

        # Añadir anomalías sintéticas si se solicita
        if include_anomalies:
            n_anomalies = int(n_samples * anomaly_fraction)
            anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)

            for idx in anomaly_indices:
                # Perturbar algunas variables
                anomaly_type = np.random.choice(["temp", "vib", "pressure"])
                if anomaly_type == "temp":
                    data["temperature"][idx] += np.random.uniform(20, 50)
                    data["operational_status"][idx] = "warning"
                elif anomaly_type == "vib":
                    data["vibration"][idx] *= np.random.uniform(3, 5)
                    data["operational_status"][idx] = "warning"
                else:
                    data["pressure"][idx] += np.random.uniform(5, 10)
                    data["operational_status"][idx] = "fault"

        df = pd.DataFrame(data)

        return {
            "data": df,
            "source": "synthetic_for_testing",
            "n_records": n_samples,
            "n_anomalies": int(n_samples * anomaly_fraction) if include_anomalies else 0,
            "note": "Datos sintéticos - NO representan sistemas reales"
        }

    def calculate_derived_features(
        self,
        df,
        window_sizes: List[int] = None
    ) -> Dict[str, Any]:
        """
        Calcula features derivados genéricos para señales de sensores.

        NOTA: Estos son cálculos matemáticos puros, no reglas de ingeniería.
        El agente decide qué features son útiles para detección.
        """
        import pandas as pd

        if window_sizes is None:
            window_sizes = [10, 30, 60]  # ventanas en minutos, típicas pero no reglas

        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in ['timestamp']:
                continue

            for w in window_sizes:
                # Rolling statistics
                result[f'{col}_mean_{w}'] = result[col].rolling(w).mean()
                result[f'{col}_std_{w}'] = result[col].rolling(w).std()
                result[f'{col}_min_{w}'] = result[col].rolling(w).min()
                result[f'{col}_max_{w}'] = result[col].rolling(w).max()

                # Rate of change
                result[f'{col}_delta_{w}'] = result[col].diff(w)

        features_added = [
            f'{col}_{stat}_{w}'
            for col in numeric_cols if col != 'timestamp'
            for w in window_sizes
            for stat in ['mean', 'std', 'min', 'max', 'delta']
        ]

        return {
            "data": result,
            "features_added": features_added
        }

    def resample_timeseries(
        self,
        df,
        freq: str,
        agg_functions: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Resamplea serie temporal a una frecuencia diferente.

        Args:
            df: DataFrame con columna 'timestamp'
            freq: Frecuencia objetivo ('1H', '1D', etc.)
            agg_functions: Diccionario {columna: función} para agregación
        """
        import pandas as pd

        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame debe tener columna 'timestamp'")

        df_indexed = df.set_index('timestamp')

        if agg_functions is None:
            # Usar mean para numéricas, first para categóricas
            numeric_cols = df_indexed.select_dtypes(include=[np.number]).columns
            cat_cols = df_indexed.select_dtypes(include=['object', 'category']).columns

            agg_functions = {}
            for col in numeric_cols:
                agg_functions[col] = 'mean'
            for col in cat_cols:
                agg_functions[col] = 'first'

        resampled = df_indexed.resample(freq).agg(agg_functions)

        return {
            "data": resampled.reset_index(),
            "original_freq": "inferred",
            "target_freq": freq,
            "n_records": len(resampled)
        }


# =============================================================================
# SCHEMA ADICIONALES: Subtipos de ingeniería
# =============================================================================

def create_predictive_maintenance_schema() -> DomainSchema:
    """Schema para datos de mantenimiento predictivo."""
    variables = [
        VariableDefinition(
            name="equipment_id",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.INDEX,
            description="Identificador del equipo"
        ),
        VariableDefinition(
            name="timestamp",
            var_type=VariableType.TEMPORAL,
            role=VariableRole.INDEX,
            description="Momento del registro"
        ),
        VariableDefinition(
            name="operating_hours",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="hours",
            description="Horas de operación acumuladas"
        ),
        VariableDefinition(
            name="cycles",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            description="Ciclos acumulados"
        ),
        VariableDefinition(
            name="last_maintenance",
            var_type=VariableType.TEMPORAL,
            role=VariableRole.COVARIATE,
            description="Fecha último mantenimiento"
        ),
        VariableDefinition(
            name="failure_event",
            var_type=VariableType.BINARY,
            role=VariableRole.OUTCOME,
            description="¿Ocurrió falla? (etiqueta real)"
        ),
        VariableDefinition(
            name="remaining_useful_life",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.OUTCOME,
            unit="hours",
            description="Vida útil remanente (si conocida)"
        ),
    ]

    return DomainSchema(
        domain_name="engineering_pdm",
        version="1.0.0",
        variables=variables,
        metadata={"tipo": "mantenimiento_predictivo"}
    )


def create_energy_schema() -> DomainSchema:
    """Schema para datos de sistemas energéticos."""
    variables = [
        VariableDefinition(
            name="timestamp",
            var_type=VariableType.TEMPORAL,
            role=VariableRole.INDEX,
            description="Momento de la medición"
        ),
        VariableDefinition(
            name="node_id",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.INDEX,
            description="Nodo de la red"
        ),
        VariableDefinition(
            name="power_active",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="MW",
            description="Potencia activa"
        ),
        VariableDefinition(
            name="power_reactive",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="MVAr",
            description="Potencia reactiva"
        ),
        VariableDefinition(
            name="voltage_magnitude",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="kV",
            description="Magnitud de voltaje"
        ),
        VariableDefinition(
            name="frequency",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="Hz",
            description="Frecuencia del sistema"
        ),
        VariableDefinition(
            name="generation_type",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.COVARIATE,
            categories=["solar", "wind", "hydro", "thermal", "nuclear"],
            description="Tipo de generación"
        ),
    ]

    return DomainSchema(
        domain_name="engineering_energy",
        version="1.0.0",
        variables=variables,
        metadata={"tipo": "sistemas_energéticos"}
    )
