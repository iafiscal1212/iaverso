"""
DOMINIO: COSMOLOGÍA / FÍSICA

NORMA DURA EXTENDIDA:
- NO hay constantes físicas hardcodeadas (c, G, h)
- NO hay modelos cosmológicos predefinidos
- SOLO infraestructura para que agentes aprendan de datos

El agente que use esto APRENDERÁ de observaciones astronómicas,
y formulará hipótesis que serán falsificadas contra datos reales.

EXCEPCIÓN MATEMÁTICA: Constantes matemáticas puras (π, e) SÍ permitidas
ya que son definiciones, no mediciones.
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
# CONSTANTES MATEMÁTICAS (permitidas - son definiciones, no mediciones)
# =============================================================================

PI = np.pi  # ORIGEN: definición matemática, razón circunferencia/diámetro
TAU = 2 * np.pi  # ORIGEN: definición matemática, 2π


# =============================================================================
# SCHEMA: Define QUÉ variables existen, NO teorías cosmológicas
# =============================================================================

def create_astronomical_schema() -> DomainSchema:
    """
    Schema para datos astronómicos genéricos.

    NOTA: No definimos cosmología (Big Bang, constantes físicas).
    Eso lo aprende el agente de los datos observacionales.
    """
    variables = [
        # Identificación
        VariableDefinition(
            name="object_id",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.INDEX,
            description="Identificador del objeto"
        ),
        VariableDefinition(
            name="object_type",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.COVARIATE,
            categories=["star", "galaxy", "quasar", "nebula", "planet", "asteroid", "unknown"],
            description="Tipo de objeto"
        ),

        # Coordenadas
        VariableDefinition(
            name="ra",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="degrees",
            description="Ascensión recta"
        ),
        VariableDefinition(
            name="dec",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="degrees",
            description="Declinación"
        ),
        VariableDefinition(
            name="redshift",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            description="Corrimiento al rojo"
        ),

        # Fotometría
        VariableDefinition(
            name="magnitude_u",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="mag",
            description="Magnitud banda U"
        ),
        VariableDefinition(
            name="magnitude_g",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="mag",
            description="Magnitud banda G"
        ),
        VariableDefinition(
            name="magnitude_r",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="mag",
            description="Magnitud banda R"
        ),
        VariableDefinition(
            name="magnitude_i",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="mag",
            description="Magnitud banda I"
        ),
        VariableDefinition(
            name="magnitude_z",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="mag",
            description="Magnitud banda Z"
        ),

        # Errores (importante para física)
        VariableDefinition(
            name="redshift_err",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.COVARIATE,
            description="Error en redshift"
        ),

        # Temporal
        VariableDefinition(
            name="observation_time",
            var_type=VariableType.TEMPORAL,
            role=VariableRole.INDEX,
            description="Momento de observación"
        ),
    ]

    return DomainSchema(
        domain_name="cosmology_astronomical",
        version="1.0.0",
        variables=variables,
        metadata={
            "tipo": "observaciones_astronómicas",
            "nota": "El agente aprende relaciones distancia-redshift de datos"
        }
    )


def create_spectral_schema() -> DomainSchema:
    """Schema para datos espectroscópicos."""
    variables = [
        VariableDefinition(
            name="object_id",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.INDEX,
            description="Identificador del objeto"
        ),
        VariableDefinition(
            name="wavelength",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.INDEX,
            unit="angstrom",
            description="Longitud de onda"
        ),
        VariableDefinition(
            name="flux",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="erg/s/cm²/Å",
            description="Flujo espectral"
        ),
        VariableDefinition(
            name="flux_error",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.COVARIATE,
            description="Error en flujo"
        ),
        VariableDefinition(
            name="sky",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.COVARIATE,
            description="Contribución del cielo"
        ),
    ]

    return DomainSchema(
        domain_name="cosmology_spectral",
        version="1.0.0",
        variables=variables,
        metadata={"tipo": "espectroscopia"}
    )


# =============================================================================
# CONNECTOR: Carga datos de diferentes fuentes
# =============================================================================

class CosmologyConnector(DomainConnector):
    """
    Conector para datos cosmológicos/astronómicos.

    Soporta formatos estándar (FITS, VOTable) y catálogos públicos.
    NO interpreta los datos - solo los carga.
    """

    def __init__(self):
        self.astronomical_schema = create_astronomical_schema()
        self.spectral_schema = create_spectral_schema()
        super().__init__(schema=self.astronomical_schema)

    def load_data(self, source: str, **kwargs) -> np.ndarray:
        """Carga datos desde una fuente."""
        if source == "fits":
            result = self.load_fits(Path(kwargs["path"]), kwargs.get("extension", 1))
            return result["data"].values if hasattr(result["data"], 'values') else result["data"]
        elif source == "synthetic":
            result = self.load_synthetic_for_testing(kwargs.get("n_samples", 1000), kwargs.get("seed"))
            return result["data"].values if hasattr(result["data"], 'values') else result["data"]
        elif source == "votable":
            result = self.load_votable(Path(kwargs["path"]))
            return result["data"].values if hasattr(result["data"], 'values') else result["data"]
        else:
            raise ValueError(f"Fuente '{source}' no soportada")

    def get_available_sources(self) -> List[str]:
        """Lista fuentes de datos disponibles."""
        return ["fits", "votable", "synthetic", "sdss"]

    def load_fits(self, path: Path, extension: int = 1) -> Dict[str, Any]:
        """
        Carga datos desde archivo FITS.

        Requiere: astropy instalado
        """
        try:
            from astropy.io import fits
            from astropy.table import Table
            import pandas as pd
        except ImportError:
            raise ImportError("Instalar astropy: pip install astropy")

        with fits.open(path) as hdul:
            table = Table.read(hdul, hdu=extension)
            df = table.to_pandas()

        return {
            "data": df,
            "source": str(path),
            "n_records": len(df),
            "columns": list(df.columns),
        }

    def load_votable(self, path: Path) -> Dict[str, Any]:
        """Carga datos desde VOTable."""
        try:
            from astropy.io.votable import parse
            import pandas as pd
        except ImportError:
            raise ImportError("Instalar astropy: pip install astropy")

        votable = parse(path)
        table = votable.get_first_table().to_table()
        df = table.to_pandas()

        return {
            "data": df,
            "source": str(path),
            "n_records": len(df),
        }

    def load_from_sdss(
        self,
        ra_range: Tuple[float, float],
        dec_range: Tuple[float, float],
        object_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Carga datos desde SDSS (Sloan Digital Sky Survey).

        NOTA: Requiere conexión a internet y astroquery.
        """
        try:
            from astroquery.sdss import SDSS
            from astropy.coordinates import SkyCoord
            import astropy.units as u
        except ImportError:
            raise ImportError("Instalar astroquery: pip install astroquery")

        # Construir query SQL
        ra_min, ra_max = ra_range
        dec_min, dec_max = dec_range

        sql_query = f"""
        SELECT TOP 10000
            objID, ra, dec, type,
            psfMag_u, psfMag_g, psfMag_r, psfMag_i, psfMag_z,
            z as redshift, zErr as redshift_err
        FROM PhotoObj
        WHERE ra BETWEEN {ra_min} AND {ra_max}
            AND dec BETWEEN {dec_min} AND {dec_max}
        """

        if object_type:
            type_map = {"star": 6, "galaxy": 3}
            if object_type.lower() in type_map:
                sql_query += f" AND type = {type_map[object_type.lower()]}"

        result = SDSS.query_sql(sql_query)

        if result is None:
            return {
                "data": None,
                "source": "SDSS",
                "n_records": 0,
                "error": "No results found"
            }

        df = result.to_pandas()

        # Renombrar a schema estándar
        rename_map = {
            'objID': 'object_id',
            'psfMag_u': 'magnitude_u',
            'psfMag_g': 'magnitude_g',
            'psfMag_r': 'magnitude_r',
            'psfMag_i': 'magnitude_i',
            'psfMag_z': 'magnitude_z',
        }
        df = df.rename(columns=rename_map)

        return {
            "data": df,
            "source": "SDSS",
            "n_records": len(df),
            "query": sql_query,
        }

    def load_synthetic_for_testing(
        self,
        n_samples: int,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Genera datos sintéticos para testing.

        IMPORTANTE: NO representan física real.
        El agente debe aprender de datos observacionales reales.
        """
        import pandas as pd

        if seed is not None:
            np.random.seed(seed)

        # Distribuciones genéricas (no calibradas a datos reales)
        data = {
            "object_id": [f"SYN_{i:06d}" for i in range(n_samples)],
            "object_type": np.random.choice(
                ["star", "galaxy", "quasar"],
                size=n_samples,
                p=[0.6, 0.35, 0.05]
            ),
            "ra": np.random.uniform(0, 360, n_samples),
            "dec": np.random.uniform(-90, 90, n_samples),
            "redshift": np.abs(np.random.exponential(scale=0.5, size=n_samples)),
            "magnitude_u": np.random.normal(20, 3, n_samples),
            "magnitude_g": np.random.normal(19, 3, n_samples),
            "magnitude_r": np.random.normal(18, 3, n_samples),
            "magnitude_i": np.random.normal(17.5, 3, n_samples),
            "magnitude_z": np.random.normal(17, 3, n_samples),
            "redshift_err": np.abs(np.random.normal(0, 0.01, n_samples)),
        }

        df = pd.DataFrame(data)

        return {
            "data": df,
            "source": "synthetic_for_testing",
            "n_records": n_samples,
            "note": "Datos sintéticos - NO representan física real"
        }

    def angular_to_cartesian(self, ra: np.ndarray, dec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convierte coordenadas angulares a cartesianas (unitarias).

        NOTA: Transformación geométrica pura, no física.
        """
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)

        x = np.cos(dec_rad) * np.cos(ra_rad)
        y = np.cos(dec_rad) * np.sin(ra_rad)
        z = np.sin(dec_rad)

        return x, y, z

    def angular_separation(
        self,
        ra1: float, dec1: float,
        ra2: float, dec2: float
    ) -> float:
        """
        Calcula separación angular entre dos puntos (fórmula haversine).

        Returns:
            Separación en grados
        """
        ra1_rad, dec1_rad = np.radians(ra1), np.radians(dec1)
        ra2_rad, dec2_rad = np.radians(ra2), np.radians(dec2)

        # Fórmula haversine
        dra = ra2_rad - ra1_rad
        ddec = dec2_rad - dec1_rad

        a = np.sin(ddec/2)**2 + np.cos(dec1_rad) * np.cos(dec2_rad) * np.sin(dra/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return np.degrees(c)


# =============================================================================
# SCHEMA ADICIONALES: Subtipos físicos
# =============================================================================

def create_gravitational_wave_schema() -> DomainSchema:
    """Schema para datos de ondas gravitacionales."""
    variables = [
        VariableDefinition(
            name="event_id",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.INDEX,
            description="Identificador del evento"
        ),
        VariableDefinition(
            name="timestamp",
            var_type=VariableType.TEMPORAL,
            role=VariableRole.INDEX,
            description="Tiempo GPS del evento"
        ),
        VariableDefinition(
            name="strain",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            description="Strain h(t)"
        ),
        VariableDefinition(
            name="frequency",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="Hz",
            description="Frecuencia característica"
        ),
        VariableDefinition(
            name="snr",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            description="Signal-to-noise ratio"
        ),
        VariableDefinition(
            name="detector",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.COVARIATE,
            categories=["H1", "L1", "V1"],
            description="Detector (Hanford, Livingston, Virgo)"
        ),
    ]

    return DomainSchema(
        domain_name="physics_gw",
        version="1.0.0",
        variables=variables,
        metadata={"tipo": "ondas_gravitacionales"}
    )


def create_particle_physics_schema() -> DomainSchema:
    """Schema para datos de física de partículas."""
    variables = [
        VariableDefinition(
            name="event_id",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.INDEX,
            description="Identificador del evento"
        ),
        VariableDefinition(
            name="run_number",
            var_type=VariableType.ORDINAL,
            role=VariableRole.INDEX,
            description="Número de run"
        ),
        VariableDefinition(
            name="n_jets",
            var_type=VariableType.ORDINAL,
            role=VariableRole.PREDICTOR,
            description="Número de jets"
        ),
        VariableDefinition(
            name="n_leptons",
            var_type=VariableType.ORDINAL,
            role=VariableRole.PREDICTOR,
            description="Número de leptones"
        ),
        VariableDefinition(
            name="missing_et",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="GeV",
            description="Energía transversal faltante"
        ),
        VariableDefinition(
            name="invariant_mass",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="GeV",
            description="Masa invariante"
        ),
        VariableDefinition(
            name="label",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.OUTCOME,
            description="Clasificación (signal/background)"
        ),
    ]

    return DomainSchema(
        domain_name="physics_particles",
        version="1.0.0",
        variables=variables,
        metadata={"tipo": "física_partículas"}
    )
