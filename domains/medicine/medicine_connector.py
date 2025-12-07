"""
DOMINIO: MEDICINA / INVESTIGACIÓN CLÍNICA

NORMA DURA EXTENDIDA:
- NO hay reglas diagnósticas hardcodeadas
- NO hay umbrales clínicos (glucosa > 126, etc.)
- SOLO infraestructura para que agentes aprendan de datos

El agente que use esto APRENDERÁ de papers, datasets clínicos,
y formulará hipótesis que serán falsificadas contra datos reales.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

from ..core.domain_base import (
    DomainSchema, DomainConnector, VariableDefinition,
    VariableType, VariableRole
)


# =============================================================================
# SCHEMA: Define QUÉ variables existen, NO qué significan clínicamente
# =============================================================================

def create_clinical_schema() -> DomainSchema:
    """
    Schema para datos clínicos genéricos.

    NOTA: No definimos qué es "normal" o "patológico".
    Eso lo aprende el agente de los datos.
    """
    variables = [
        # Variables de laboratorio (sin rangos - los aprende el agente)
        VariableDefinition(
            name="glucose",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="mg/dL",
            description="Glucosa en sangre"
        ),
        VariableDefinition(
            name="hemoglobin_a1c",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="%",
            description="Hemoglobina glicosilada"
        ),
        VariableDefinition(
            name="blood_pressure_systolic",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="mmHg",
            description="Presión arterial sistólica"
        ),
        VariableDefinition(
            name="blood_pressure_diastolic",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="mmHg",
            description="Presión arterial diastólica"
        ),
        VariableDefinition(
            name="cholesterol_total",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="mg/dL",
            description="Colesterol total"
        ),
        VariableDefinition(
            name="cholesterol_ldl",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="mg/dL",
            description="Colesterol LDL"
        ),
        VariableDefinition(
            name="cholesterol_hdl",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="mg/dL",
            description="Colesterol HDL"
        ),
        VariableDefinition(
            name="triglycerides",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="mg/dL",
            description="Triglicéridos"
        ),
        VariableDefinition(
            name="creatinine",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="mg/dL",
            description="Creatinina sérica"
        ),
        VariableDefinition(
            name="bmi",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="kg/m²",
            description="Índice de masa corporal"
        ),

        # Variables demográficas
        VariableDefinition(
            name="age",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.COVARIATE,
            unit="years",
            description="Edad del paciente"
        ),
        VariableDefinition(
            name="sex",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.COVARIATE,
            categories=["M", "F"],
            description="Sexo biológico"
        ),

        # Variable objetivo (sin definir qué es outcome positivo/negativo)
        VariableDefinition(
            name="outcome",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.OUTCOME,
            description="Resultado clínico (definido por dataset)"
        ),

        # Tiempo (para estudios longitudinales)
        VariableDefinition(
            name="timestamp",
            var_type=VariableType.TEMPORAL,
            role=VariableRole.INDEX,
            description="Momento de la medición"
        ),
    ]

    return DomainSchema(
        domain_name="medicine",
        version="1.0.0",
        variables=variables,
        metadata={
            "tipo": "clínico_genérico",
            "nota": "El agente aprende rangos normales de los datos"
        }
    )


# =============================================================================
# CONNECTOR: Carga datos de diferentes fuentes
# =============================================================================

class MedicineConnector(DomainConnector):
    """
    Conector para datos médicos/clínicos.

    Soporta múltiples formatos y fuentes.
    NO interpreta los datos - solo los carga y valida schema.
    """

    def __init__(self):
        schema = create_clinical_schema()
        super().__init__(schema=schema)

    def load_data(self, source: str, **kwargs) -> np.ndarray:
        """Carga datos desde una fuente."""
        if source == "csv":
            result = self.load_csv(Path(kwargs["path"]), kwargs.get("column_mapping"))
            return result["data"].values if hasattr(result["data"], 'values') else result["data"]
        elif source == "synthetic":
            result = self.load_synthetic_for_testing(kwargs.get("n_samples", 1000), kwargs.get("seed"))
            return result["data"].values if hasattr(result["data"], 'values') else result["data"]
        else:
            raise ValueError(f"Fuente '{source}' no soportada")

    def get_available_sources(self) -> List[str]:
        """Lista fuentes de datos disponibles."""
        return ["csv", "synthetic", "api"]

    def load_csv(self, path: Path, column_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Carga datos desde CSV.

        Args:
            path: Ruta al archivo CSV
            column_mapping: Mapeo de nombres de columnas del CSV a variables del schema
                           Ejemplo: {"glucemia_ayunas": "glucose"}

        Returns:
            Dict con datos cargados y metadata
        """
        import pandas as pd

        df = pd.read_csv(path)

        # Aplicar mapeo si se proporciona
        if column_mapping:
            df = df.rename(columns=column_mapping)

        # Validar que las columnas correspondan al schema
        schema_vars = {v.name for v in self.schema.variables}
        loaded_vars = set(df.columns)

        return {
            "data": df,
            "source": str(path),
            "n_records": len(df),
            "variables_loaded": list(loaded_vars & schema_vars),
            "variables_missing": list(schema_vars - loaded_vars),
            "variables_extra": list(loaded_vars - schema_vars),
        }

    def load_from_api(self, endpoint: str, credentials: Dict[str, str]) -> Dict[str, Any]:
        """
        Carga datos desde API (FHIR, HL7, etc.)

        NOTA: Implementación específica depende del sistema.
        Esta es la estructura base.
        """
        # Placeholder - implementación real dependería del sistema específico
        raise NotImplementedError(
            "Implementar para sistema específico (FHIR, HL7, etc.)"
        )

    def load_synthetic_for_testing(self, n_samples: int, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Genera datos sintéticos para testing.

        IMPORTANTE: Los rangos aquí son SOLO para generar datos de prueba.
        NO representan conocimiento médico - el agente debe aprender
        los patrones reales de datasets clínicos reales.
        """
        import pandas as pd

        if seed is not None:
            np.random.seed(seed)

        # Generar datos con distribuciones genéricas (NO clínicamente significativas)
        # El agente aprenderá las distribuciones reales de datos reales
        data = {
            "glucose": np.random.lognormal(mean=np.log(100), sigma=0.3, size=n_samples),
            "hemoglobin_a1c": np.random.normal(loc=6.0, scale=1.5, size=n_samples),
            "blood_pressure_systolic": np.random.normal(loc=120, scale=20, size=n_samples),
            "blood_pressure_diastolic": np.random.normal(loc=80, scale=12, size=n_samples),
            "cholesterol_total": np.random.normal(loc=200, scale=40, size=n_samples),
            "cholesterol_ldl": np.random.normal(loc=100, scale=30, size=n_samples),
            "cholesterol_hdl": np.random.normal(loc=50, scale=15, size=n_samples),
            "triglycerides": np.random.lognormal(mean=np.log(150), sigma=0.4, size=n_samples),
            "creatinine": np.random.lognormal(mean=np.log(1.0), sigma=0.3, size=n_samples),
            "bmi": np.random.normal(loc=26, scale=5, size=n_samples),
            "age": np.random.uniform(low=18, high=90, size=n_samples),
            "sex": np.random.choice(["M", "F"], size=n_samples),
            "outcome": np.random.choice(["positive", "negative"], size=n_samples),
        }

        df = pd.DataFrame(data)

        return {
            "data": df,
            "source": "synthetic_for_testing",
            "n_records": n_samples,
            "note": "Datos sintéticos - NO usar para entrenamiento real"
        }


# =============================================================================
# SCHEMA ADICIONALES: Subtipos clínicos
# =============================================================================

def create_longitudinal_schema() -> DomainSchema:
    """Schema para estudios longitudinales/seguimiento."""
    base_schema = create_clinical_schema()

    # Añadir variables específicas de seguimiento
    additional = [
        VariableDefinition(
            name="visit_number",
            var_type=VariableType.ORDINAL,
            role=VariableRole.INDEX,
            description="Número de visita en el seguimiento"
        ),
        VariableDefinition(
            name="days_since_baseline",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.INDEX,
            unit="days",
            description="Días desde visita basal"
        ),
        VariableDefinition(
            name="treatment_arm",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.TREATMENT,
            description="Brazo de tratamiento (si aplica)"
        ),
    ]

    return DomainSchema(
        domain_name="medicine_longitudinal",
        version="1.0.0",
        variables=base_schema.variables + additional,
        metadata={"tipo": "longitudinal"}
    )


def create_imaging_schema() -> DomainSchema:
    """Schema para datos de imagen médica."""
    variables = [
        VariableDefinition(
            name="image_path",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.PREDICTOR,
            description="Ruta a la imagen"
        ),
        VariableDefinition(
            name="modality",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.COVARIATE,
            categories=["CT", "MRI", "X-ray", "Ultrasound", "PET"],
            description="Modalidad de imagen"
        ),
        VariableDefinition(
            name="body_part",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.COVARIATE,
            description="Parte del cuerpo"
        ),
        VariableDefinition(
            name="finding",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.OUTCOME,
            description="Hallazgo (etiquetado por radiólogo)"
        ),
    ]

    return DomainSchema(
        domain_name="medicine_imaging",
        version="1.0.0",
        variables=variables,
        metadata={"tipo": "imagen_médica"}
    )
