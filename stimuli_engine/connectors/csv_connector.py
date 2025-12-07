"""
CSV CONNECTOR - Conector Genérico para CSV
===========================================

Lee archivos CSV y los convierte a estructuras matemáticas.
NO interpreta el contenido. NO conoce la semántica.

NORMA DURA:
- Todos los parámetros derivados tienen procedencia
- Sin números mágicos
"""

import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from provenance import Provenance, ProvenanceType, get_provenance_logger


@dataclass
class CSVSchema:
    """
    Schema inferido de un CSV.

    Detectado automáticamente, no impuesto.
    """
    n_columns: int
    n_rows: int
    column_types: List[str]         # 'numeric', 'temporal', 'categorical'
    has_header: bool
    delimiter: str
    file_hash: str                  # Hash anónimo del archivo


class CSVConnector:
    """
    Conector genérico para archivos CSV.

    RESPONSABILIDADES:
    - Leer CSV sin conocer su semántica
    - Inferir tipos de columnas
    - Convertir a arrays numéricos
    - Documentar procedencia

    NO HACE:
    - Interpretar qué significan las columnas
    - Sugerir qué columnas son "importantes"
    - Nombrar las series con nombres semánticos
    """

    def __init__(self):
        self.logger = get_provenance_logger()

    def _compute_file_hash(self, path: Path) -> str:
        """Computa hash anónimo del archivo."""
        with open(path, 'rb') as f:
            content = f.read()
        return hashlib.sha256(content).hexdigest()[:12]

    def _detect_delimiter(self, path: Path) -> str:
        """
        Detecta delimitador del CSV.

        ORIGEN: Usa sniffer de csv, estándar de Python.
        """
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            sample = f.read(4096)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',;\t|')
            return dialect.delimiter
        except csv.Error:
            return ','  # Default CSV

    def _detect_header(self, path: Path, delimiter: str) -> bool:
        """
        Detecta si tiene header.

        ORIGEN: Usa sniffer de csv.
        """
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            sample = f.read(4096)

        try:
            return csv.Sniffer().has_header(sample)
        except csv.Error:
            return True  # Asumir header

    def _infer_column_type(self, values: List[str]) -> str:
        """
        Infiere tipo de columna.

        ORIGEN: Basado en parseo exitoso.
        """
        numeric_count = 0
        temporal_count = 0
        total = len(values)

        if total == 0:
            return 'empty'

        for v in values[:100]:  # Muestra de primeras 100
            v = v.strip()

            # Intentar numérico
            try:
                float(v)
                numeric_count += 1
                continue
            except ValueError:
                pass

            # Intentar temporal (formatos comunes)
            temporal_patterns = [
                '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y',
                '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S',
            ]
            for pattern in temporal_patterns:
                try:
                    datetime.strptime(v, pattern)
                    temporal_count += 1
                    break
                except ValueError:
                    continue

        sample_size = min(100, total)

        # ORIGEN: Mayoría simple (>50%)
        if numeric_count > sample_size * 0.5:
            return 'numeric'
        elif temporal_count > sample_size * 0.5:
            return 'temporal'
        else:
            return 'categorical'

    def analyze(self, path: Path) -> CSVSchema:
        """
        Analiza estructura del CSV sin leer todo.

        Returns:
            CSVSchema con información inferida
        """
        file_hash = self._compute_file_hash(path)
        delimiter = self._detect_delimiter(path)
        has_header = self._detect_header(path, delimiter)

        # Leer primeras filas para inferir tipos
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f, delimiter=delimiter)

            if has_header:
                header = next(reader)
            else:
                header = None

            rows = []
            for i, row in enumerate(reader):
                rows.append(row)
                if i >= 100:  # Muestra de 100 filas
                    break

        if not rows:
            return CSVSchema(
                n_columns=0,
                n_rows=0,
                column_types=[],
                has_header=has_header,
                delimiter=delimiter,
                file_hash=file_hash
            )

        n_columns = len(rows[0])

        # Inferir tipos de cada columna
        column_types = []
        for col_idx in range(n_columns):
            col_values = [row[col_idx] for row in rows if col_idx < len(row)]
            col_type = self._infer_column_type(col_values)
            column_types.append(col_type)

        # Contar filas totales
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            n_rows = sum(1 for _ in f)
            if has_header:
                n_rows -= 1

        self.logger.log_from_data(
            value={'n_cols': n_columns, 'n_rows': n_rows},
            source=f"csv_analysis:{file_hash}",
            statistic="schema_inference",
            context="CSVConnector.analyze"
        )

        return CSVSchema(
            n_columns=n_columns,
            n_rows=n_rows,
            column_types=column_types,
            has_header=has_header,
            delimiter=delimiter,
            file_hash=file_hash
        )

    def load_column(
        self,
        path: Path,
        column_index: int,
        as_numeric: bool = True
    ) -> Tuple[np.ndarray, Provenance]:
        """
        Carga una columna específica.

        Args:
            path: Ruta al archivo
            column_index: Índice de la columna
            as_numeric: Convertir a numérico

        Returns:
            (array, provenance)
        """
        schema = self.analyze(path)

        values = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f, delimiter=schema.delimiter)

            if schema.has_header:
                next(reader)

            for row in reader:
                if column_index < len(row):
                    val = row[column_index].strip()
                    if as_numeric:
                        try:
                            values.append(float(val))
                        except ValueError:
                            values.append(np.nan)
                    else:
                        values.append(val)

        arr = np.array(values) if as_numeric else np.array(values, dtype=object)

        provenance = self.logger.log_from_data(
            value=f"column[{column_index}]",
            source=f"csv:{schema.file_hash}",
            dataset=f"col_{column_index}",
            statistic="raw_load",
            context="CSVConnector.load_column"
        )

        return arr, provenance

    def load_numeric_columns(
        self,
        path: Path
    ) -> Dict[int, Tuple[np.ndarray, Provenance]]:
        """
        Carga todas las columnas numéricas.

        Returns:
            Dict[column_index] -> (array, provenance)
        """
        schema = self.analyze(path)

        result = {}
        for i, ctype in enumerate(schema.column_types):
            if ctype == 'numeric':
                arr, prov = self.load_column(path, i, as_numeric=True)
                result[i] = (arr, prov)

        return result

    def load_as_timeseries(
        self,
        path: Path,
        time_column: int,
        value_column: int
    ) -> Tuple[np.ndarray, np.ndarray, Provenance]:
        """
        Carga dos columnas como serie temporal.

        Args:
            path: Ruta
            time_column: Índice de columna temporal
            value_column: Índice de columna de valores

        Returns:
            (t_array, values_array, provenance)
        """
        schema = self.analyze(path)

        t_values = []
        data_values = []

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f, delimiter=schema.delimiter)

            if schema.has_header:
                next(reader)

            for row in reader:
                if time_column < len(row) and value_column < len(row):
                    # Intentar parsear tiempo
                    t_str = row[time_column].strip()
                    v_str = row[value_column].strip()

                    try:
                        t_val = float(t_str)
                    except ValueError:
                        # Intentar como fecha
                        try:
                            dt = datetime.fromisoformat(t_str.replace('Z', '+00:00'))
                            t_val = dt.timestamp()
                        except ValueError:
                            continue

                    try:
                        v_val = float(v_str)
                    except ValueError:
                        v_val = np.nan

                    t_values.append(t_val)
                    data_values.append(v_val)

        t_arr = np.array(t_values)
        v_arr = np.array(data_values)

        # Ordenar por tiempo
        sort_idx = np.argsort(t_arr)
        t_arr = t_arr[sort_idx]
        v_arr = v_arr[sort_idx]

        provenance = self.logger.log_from_data(
            value=f"timeseries[{len(t_arr)}]",
            source=f"csv:{schema.file_hash}",
            dataset=f"t={time_column},v={value_column}",
            statistic="timeseries_load",
            context="CSVConnector.load_as_timeseries"
        )

        return t_arr, v_arr, provenance
