"""
TIMESERIES CONNECTOR - Conector Genérico para Series Temporales
================================================================

Procesa series temporales de múltiples formatos.
NO conoce la semántica de los datos.

NORMA DURA:
- Interpolación y resampling con procedencia
- Sin parámetros mágicos
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from provenance import Provenance, ProvenanceType, get_provenance_logger, THEORY_CONSTANTS


@dataclass
class TimeseriesMetadata:
    """
    Metadatos de una serie temporal.

    Solo información estructural, no semántica.
    """
    n_points: int
    t_min: float
    t_max: float
    resolution: float           # Derivado de datos
    has_gaps: bool
    n_nans: int
    source_hash: str


class TimeseriesConnector:
    """
    Conector para series temporales.

    RESPONSABILIDADES:
    - Cargar series de diferentes formatos
    - Interpolar y resamplear
    - Detectar gaps y anomalías estructurales
    - Documentar procedencia

    NO HACE:
    - Interpretar qué significa la serie
    - Nombrar con semántica
    """

    def __init__(self):
        self.logger = get_provenance_logger()

    def analyze(
        self,
        t: np.ndarray,
        values: np.ndarray
    ) -> TimeseriesMetadata:
        """
        Analiza estructura de una serie temporal.

        NORMA DURA: Todos los umbrales derivados de datos.
        """
        n = len(t)

        if n == 0:
            return TimeseriesMetadata(
                n_points=0,
                t_min=0,
                t_max=0,
                resolution=0,
                has_gaps=False,
                n_nans=0,
                source_hash="empty"
            )

        t_min = float(np.min(t))
        t_max = float(np.max(t))

        # Resolución = mediana de diferencias
        if n > 1:
            diffs = np.diff(t)
            resolution = float(np.median(diffs))

            # Detectar gaps: diferencia > 3 * mediana
            # ORIGEN: 3 = convención para outliers (similar a 3-sigma)
            gap_threshold = resolution * 3
            has_gaps = bool(np.any(diffs > gap_threshold))

            self.logger.log_from_data(
                value=gap_threshold,
                source="median(dt) * 3",
                statistic="gap_threshold",
                context="TimeseriesConnector.analyze"
            )
        else:
            resolution = 0
            has_gaps = False

        n_nans = int(np.sum(np.isnan(values)))

        # Hash de la serie
        source_hash = f"{hash(t.tobytes()) % 10000:04d}"

        return TimeseriesMetadata(
            n_points=n,
            t_min=t_min,
            t_max=t_max,
            resolution=resolution,
            has_gaps=has_gaps,
            n_nans=n_nans,
            source_hash=source_hash
        )

    def interpolate_nans(
        self,
        t: np.ndarray,
        values: np.ndarray
    ) -> Tuple[np.ndarray, Provenance]:
        """
        Interpola valores NaN.

        ORIGEN: Interpolación lineal, método estándar.
        """
        result = values.copy()
        nan_mask = np.isnan(result)

        if not np.any(nan_mask):
            prov = self.logger.log_from_theory(
                value="no_interpolation_needed",
                source="No NaN values found"
            )
            return result, prov

        # Interpolación lineal
        valid_idx = ~nan_mask
        if np.sum(valid_idx) < 2:
            # No hay suficientes puntos válidos
            result[nan_mask] = 0
            prov = self.logger.log_from_theory(
                value="zero_fill",
                source="Insufficient valid points for interpolation"
            )
            return result, prov

        result[nan_mask] = np.interp(
            t[nan_mask],
            t[valid_idx],
            values[valid_idx]
        )

        prov = self.logger.log_from_theory(
            value="linear_interpolation",
            source="np.interp (linear)",
            reference="Método estándar de interpolación",
            context="TimeseriesConnector.interpolate_nans"
        )

        return result, prov

    def resample(
        self,
        t: np.ndarray,
        values: np.ndarray,
        n_points: Optional[int] = None,
        resolution: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, Provenance]:
        """
        Resamplea serie a nueva resolución.

        Args:
            t: Timestamps originales
            values: Valores originales
            n_points: Número de puntos objetivo (opcional)
            resolution: Resolución objetivo (opcional)

        NORMA DURA:
        - Si no se especifica, deriva de datos
        - n_points = √len(original) * factor de Nyquist
        """
        if len(t) < 2:
            prov = self.logger.log_from_theory(
                value="no_resample",
                source="Insufficient points"
            )
            return t, values, prov

        t_min = np.min(t)
        t_max = np.max(t)

        if n_points is None and resolution is None:
            # ORIGEN: √n como aproximación razonable
            # Preserva información sin sobrecargar
            n_points = int(np.sqrt(len(t))) * 2

            self.logger.log_from_data(
                value=n_points,
                source="sqrt(n_original) * 2",
                statistic="derived_n_points",
                context="TimeseriesConnector.resample"
            )

        if n_points is not None:
            t_new = np.linspace(t_min, t_max, n_points)
        else:
            t_new = np.arange(t_min, t_max, resolution)

        # Interpolar
        values_new = np.interp(t_new, t, values)

        prov = self.logger.log_from_theory(
            value="linear_resample",
            source=f"Resampled from {len(t)} to {len(t_new)} points",
            reference="np.interp (linear)",
            context="TimeseriesConnector.resample"
        )

        return t_new, values_new, prov

    def align_multiple(
        self,
        series_list: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[np.ndarray, List[np.ndarray], Provenance]:
        """
        Alinea múltiples series a timestamps comunes.

        ORIGEN: Usa la serie con más puntos como referencia.
        """
        if not series_list:
            return np.array([]), [], self.logger.log_from_theory(
                value="empty_alignment",
                source="No series provided"
            )

        # Encontrar rango común
        t_min = max(np.min(t) for t, _ in series_list)
        t_max = min(np.max(t) for t, _ in series_list)

        # Usar número de puntos de la serie más densa
        max_density = max(
            len(t) / (np.max(t) - np.min(t)) if np.max(t) > np.min(t) else 0
            for t, _ in series_list
        )

        # Número de puntos para el rango común
        if t_max > t_min and max_density > 0:
            n_points = int((t_max - t_min) * max_density)
            n_points = max(10, min(n_points, 10000))  # Límites razonables
        else:
            n_points = 100

        t_common = np.linspace(t_min, t_max, n_points)

        # Interpolar todas las series
        aligned_values = []
        for t, values in series_list:
            values_aligned = np.interp(t_common, t, values)
            aligned_values.append(values_aligned)

        prov = self.logger.log_from_data(
            value=f"aligned[{len(series_list)} series, {n_points} points]",
            source="common_timeline_interpolation",
            statistic=f"t_range=[{t_min:.4f}, {t_max:.4f}]",
            context="TimeseriesConnector.align_multiple"
        )

        return t_common, aligned_values, prov

    def detect_changepoints(
        self,
        t: np.ndarray,
        values: np.ndarray
    ) -> Tuple[List[int], Provenance]:
        """
        Detecta puntos de cambio en la serie.

        NORMA DURA:
        - Umbral = Tukey fence sobre derivadas
        """
        if len(values) < THEORY_CONSTANTS['min_samples_clt'].value:
            return [], self.logger.log_from_theory(
                value="insufficient_data",
                source=f"n < {THEORY_CONSTANTS['min_samples_clt'].value}"
            )

        # Calcular derivadas (diferencias)
        diffs = np.diff(values)

        # Tukey fence sobre derivadas
        q1 = np.percentile(diffs, 25)
        q3 = np.percentile(diffs, 75)
        iqr = q3 - q1
        k = THEORY_CONSTANTS['tukey_k'].value

        lower = q1 - k * iqr
        upper = q3 + k * iqr

        # Puntos donde la derivada es outlier
        changepoints = []
        for i, d in enumerate(diffs):
            if d < lower or d > upper:
                changepoints.append(i + 1)  # +1 porque diff reduce longitud

        prov = self.logger.log_from_theory(
            value={'n_changepoints': len(changepoints), 'k': k},
            source="Tukey fences on differences",
            reference="Tukey, J. W. (1977)",
            context="TimeseriesConnector.detect_changepoints"
        )

        return changepoints, prov

    def compute_statistics(
        self,
        values: np.ndarray
    ) -> Tuple[Dict[str, float], Provenance]:
        """
        Computa estadísticas descriptivas.

        ORIGEN: Estadísticas estándar.
        """
        valid = values[~np.isnan(values)]

        if len(valid) == 0:
            return {}, self.logger.log_from_theory(
                value="no_valid_data",
                source="All values are NaN"
            )

        stats = {
            'n': len(valid),
            'mean': float(np.mean(valid)),
            'std': float(np.std(valid, ddof=1)) if len(valid) > 1 else 0,
            'min': float(np.min(valid)),
            'max': float(np.max(valid)),
            'q25': float(np.percentile(valid, 25)),
            'median': float(np.median(valid)),
            'q75': float(np.percentile(valid, 75)),
            'iqr': float(np.percentile(valid, 75) - np.percentile(valid, 25)),
        }

        prov = self.logger.log_from_theory(
            value="descriptive_statistics",
            source="Standard statistical measures",
            context="TimeseriesConnector.compute_statistics"
        )

        return stats, prov

    def load_from_json(
        self,
        path: Path,
        t_key: str = 't',
        values_key: str = 'values'
    ) -> Tuple[np.ndarray, np.ndarray, Provenance]:
        """
        Carga serie desde JSON.

        Args:
            path: Ruta al archivo
            t_key: Clave para timestamps
            values_key: Clave para valores

        Returns:
            (t, values, provenance)
        """
        with open(path, 'r') as f:
            data = json.load(f)

        t = np.array(data.get(t_key, []))
        values = np.array(data.get(values_key, []))

        prov = self.logger.log_from_data(
            value=f"json[{len(t)} points]",
            source=f"file_hash:{hash(str(path)) % 10000:04d}",
            dataset="json_timeseries",
            context="TimeseriesConnector.load_from_json"
        )

        return t, values, prov

    def load_from_numpy(
        self,
        path: Path
    ) -> Tuple[np.ndarray, Provenance]:
        """
        Carga array desde .npy

        Returns:
            (array, provenance)
        """
        arr = np.load(path)

        prov = self.logger.log_from_data(
            value=f"numpy[{arr.shape}]",
            source=f"file_hash:{hash(str(path)) % 10000:04d}",
            dataset="numpy_array",
            context="TimeseriesConnector.load_from_numpy"
        )

        return arr, prov
