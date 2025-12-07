#!/usr/bin/env python3
"""
Logger Endógeno - Registro de Parámetros Derivados de Datos
===========================================================

Este módulo registra todos los parámetros que emergen de datos
durante la ejecución, creando una pista de auditoría completa.

NORMA DURA: Todo parámetro derivado de datos debe quedar registrado
            con su origen, distribución y momento de derivación.

Formato de log: JSONL (una línea JSON por entrada)
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

from core.norma_dura_config import ProvenanceTag, PROVENANCE_TAGS


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

LOG_DIR = Path('/root/NEO_EVA/logs/endogenous')
LOG_FILE = LOG_DIR / 'endogenous_params.jsonl'


# =============================================================================
# ESTRUCTURAS DE DATOS
# =============================================================================

@dataclass
class EndogenousParam:
    """Registro de un parámetro endógeno."""

    # Identificación
    name: str                          # Nombre del parámetro
    value: float                       # Valor calculado

    # Procedencia
    provenance: str                    # FROM_DATA, FROM_DIST, FROM_CALIB, FROM_THEORY
    source_description: str            # Descripción del origen

    # Datos de origen (si aplica)
    source_data_shape: Optional[tuple] = None  # Shape de datos de origen
    source_data_stats: Optional[dict] = None   # Estadísticas básicas

    # Método de derivación
    derivation_method: str = ""        # Ej: "percentile_75", "std", "mean"
    derivation_params: Optional[dict] = None   # Parámetros del método

    # Metadatos
    timestamp: str = ""                # Momento de derivación
    module: str = ""                   # Módulo que lo generó
    function: str = ""                 # Función que lo generó

    def to_dict(self) -> dict:
        """Convertir a diccionario."""
        return asdict(self)

    def to_json(self) -> str:
        """Convertir a JSON string."""
        return json.dumps(self.to_dict(), default=str)


# =============================================================================
# LOGGER PRINCIPAL
# =============================================================================

class EndogenousLogger:
    """
    Logger para parámetros endógenos.

    Registra todos los parámetros derivados de datos con su origen
    completo para auditoría de NORMA DURA.
    """

    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file or LOG_FILE
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: List[EndogenousParam] = []
        self._flush_threshold = 10  # Flush cada N registros

    def log_param(
        self,
        name: str,
        value: float,
        provenance: Union[str, ProvenanceTag],
        source_description: str,
        source_data: Optional[np.ndarray] = None,
        derivation_method: str = "",
        derivation_params: Optional[dict] = None,
        module: str = "",
        function: str = ""
    ) -> EndogenousParam:
        """
        Registrar un parámetro endógeno.

        Args:
            name: Nombre del parámetro
            value: Valor calculado
            provenance: Etiqueta de procedencia (FROM_DATA, FROM_DIST, etc.)
            source_description: Descripción del origen
            source_data: Array de datos de origen (opcional)
            derivation_method: Método usado (percentile, std, etc.)
            derivation_params: Parámetros del método
            module: Módulo que genera el parámetro
            function: Función que genera el parámetro

        Returns:
            EndogenousParam registrado
        """
        # Validar procedencia
        if isinstance(provenance, ProvenanceTag):
            provenance = provenance.value
        if provenance not in PROVENANCE_TAGS:
            raise ValueError(f"Procedencia inválida: {provenance}. Use: {PROVENANCE_TAGS}")

        # Calcular estadísticas de datos origen si se proporcionan
        source_data_shape = None
        source_data_stats = None
        if source_data is not None:
            source_data_shape = source_data.shape
            source_data_stats = {
                'n': len(source_data),
                'mean': float(np.mean(source_data)),
                'std': float(np.std(source_data)),
                'min': float(np.min(source_data)),
                'max': float(np.max(source_data)),
                'p25': float(np.percentile(source_data, 25)),
                'p50': float(np.percentile(source_data, 50)),
                'p75': float(np.percentile(source_data, 75)),
            }

        # Crear registro
        param = EndogenousParam(
            name=name,
            value=float(value),
            provenance=provenance,
            source_description=source_description,
            source_data_shape=source_data_shape,
            source_data_stats=source_data_stats,
            derivation_method=derivation_method,
            derivation_params=derivation_params or {},
            timestamp=datetime.now().isoformat(),
            module=module,
            function=function
        )

        # Agregar a buffer
        self._buffer.append(param)

        # Flush si es necesario
        if len(self._buffer) >= self._flush_threshold:
            self.flush()

        return param

    def log_percentile(
        self,
        name: str,
        data: np.ndarray,
        percentile: float,
        module: str = "",
        function: str = ""
    ) -> EndogenousParam:
        """Atajo para registrar un percentil de datos."""
        value = float(np.percentile(data, percentile))
        return self.log_param(
            name=name,
            value=value,
            provenance=ProvenanceTag.FROM_DATA,
            source_description=f"Percentil {percentile} de datos observados",
            source_data=data,
            derivation_method=f"percentile_{int(percentile)}",
            derivation_params={'percentile': percentile},
            module=module,
            function=function
        )

    def log_threshold_from_distribution(
        self,
        name: str,
        percentile: float,
        distribution: str = "U(0,1)",
        module: str = "",
        function: str = ""
    ) -> EndogenousParam:
        """Atajo para registrar umbral de distribución teórica."""
        # Para U(0,1), el percentil p es simplemente p/100
        if distribution == "U(0,1)":
            value = percentile / 100.0
        else:
            raise ValueError(f"Distribución no soportada: {distribution}")

        return self.log_param(
            name=name,
            value=value,
            provenance=ProvenanceTag.FROM_DIST,
            source_description=f"Percentil {percentile} de {distribution}",
            derivation_method=f"theoretical_percentile",
            derivation_params={'percentile': percentile, 'distribution': distribution},
            module=module,
            function=function
        )

    def log_calibrated(
        self,
        name: str,
        value: float,
        calibration_data: np.ndarray,
        calibration_method: str,
        module: str = "",
        function: str = ""
    ) -> EndogenousParam:
        """Atajo para registrar parámetro calibrado."""
        return self.log_param(
            name=name,
            value=value,
            provenance=ProvenanceTag.FROM_CALIB,
            source_description=f"Calibrado via {calibration_method}",
            source_data=calibration_data,
            derivation_method=calibration_method,
            module=module,
            function=function
        )

    def log_theoretical(
        self,
        name: str,
        value: float,
        theory_description: str,
        module: str = "",
        function: str = ""
    ) -> EndogenousParam:
        """Atajo para registrar constante teórica."""
        return self.log_param(
            name=name,
            value=value,
            provenance=ProvenanceTag.FROM_THEORY,
            source_description=theory_description,
            derivation_method="theoretical_constant",
            module=module,
            function=function
        )

    def flush(self):
        """Escribir buffer a archivo."""
        if not self._buffer:
            return

        with open(self.log_file, 'a') as f:
            for param in self._buffer:
                f.write(param.to_json() + '\n')

        self._buffer.clear()

    def close(self):
        """Flush y cerrar."""
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def get_all_params(self) -> List[Dict]:
        """Leer todos los parámetros del log."""
        self.flush()  # Asegurar que todo está escrito

        params = []
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        params.append(json.loads(line))
        return params

    def get_params_by_provenance(self, provenance: str) -> List[Dict]:
        """Filtrar parámetros por procedencia."""
        return [p for p in self.get_all_params() if p['provenance'] == provenance]

    def get_audit_summary(self) -> Dict:
        """Generar resumen de auditoría."""
        params = self.get_all_params()

        by_provenance = {}
        for tag in PROVENANCE_TAGS:
            by_provenance[tag] = len([p for p in params if p['provenance'] == tag])

        by_module = {}
        for p in params:
            module = p.get('module', 'unknown')
            by_module[module] = by_module.get(module, 0) + 1

        return {
            'total_params': len(params),
            'by_provenance': by_provenance,
            'by_module': by_module,
            'log_file': str(self.log_file),
            'timestamp': datetime.now().isoformat()
        }


# =============================================================================
# INSTANCIA GLOBAL
# =============================================================================

_global_logger: Optional[EndogenousLogger] = None


def get_endogenous_logger() -> EndogenousLogger:
    """Obtener instancia global del logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = EndogenousLogger()
    return _global_logger


def log_endogenous_param(*args, **kwargs) -> EndogenousParam:
    """Atajo para registrar parámetro en logger global."""
    return get_endogenous_logger().log_param(*args, **kwargs)


# =============================================================================
# BLOQUE DE AUDITORÍA NORMA DURA
# =============================================================================
"""
MAGIC NUMBERS AUDIT
==================

CONSTANTES EN ESTE ARCHIVO:
- _flush_threshold = 10: ORIGEN: Valor práctico para balance memoria/IO
  (podría derivarse de benchmarks, pero 10 es razonable para la mayoría de casos)

FORMATO DE LOG:
- JSONL: Una línea JSON por parámetro (estándar de industria)

CAMPOS REQUERIDOS:
- name: Identificador del parámetro
- value: Valor numérico
- provenance: FROM_DATA | FROM_DIST | FROM_CALIB | FROM_THEORY
- source_description: Explicación del origen

TODAS LAS DECISIONES TIENEN ORIGEN DOCUMENTADO.
"""
