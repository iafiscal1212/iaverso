"""
PROVENANCE - Sistema de Trazabilidad
=====================================

NORMA DURA: Todo parámetro derivado debe poder loguearse con su procedencia.

Tipos de procedencia:
- FROM_DATA: Derivado de estadísticas de los datos
- FROM_THEORY: Derivado de teoría estadística/matemática
- FROM_MATH: Constante matemática pura
- FROM_CONFIG: Configuración externa (proporcionada por humana)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List
from datetime import datetime
import json
from pathlib import Path


class ProvenanceType(Enum):
    """Tipos de procedencia para parámetros."""
    FROM_DATA = "from_data"       # Percentil, media, std de datos observados
    FROM_THEORY = "from_theory"   # Fisher, Tukey, z=1.96, 1/e, etc.
    FROM_MATH = "from_math"       # pi, e, sqrt(2), etc.
    FROM_CONFIG = "from_config"   # Proporcionado externamente por humana
    UNKNOWN = "unknown"           # No documentado (VIOLACIÓN de NORMA DURA)


@dataclass
class Provenance:
    """
    Registro de procedencia de un valor.

    NORMA DURA: Todo valor numérico usado como umbral o parámetro
    debe tener un Provenance asociado.
    """
    value: Any                              # El valor
    ptype: ProvenanceType                   # Tipo de procedencia
    source: str                             # Descripción de la fuente
    timestamp: str = ""                     # Cuándo se derivó
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            'value': self.value if not hasattr(self.value, 'tolist') else self.value.tolist(),
            'type': self.ptype.value,
            'source': self.source,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
        }

    def __repr__(self) -> str:
        return f"Provenance({self.ptype.value}: {self.source} = {self.value})"


# =============================================================================
# CONSTANTES CON PROCEDENCIA DOCUMENTADA
# =============================================================================

# Constantes matemáticas puras
MATH_CONSTANTS = {
    'pi': Provenance(
        value=3.141592653589793,
        ptype=ProvenanceType.FROM_MATH,
        source="Definición: razón circunferencia/diámetro",
    ),
    'e': Provenance(
        value=2.718281828459045,
        ptype=ProvenanceType.FROM_MATH,
        source="Definición: lim(1+1/n)^n cuando n→∞",
    ),
    'inv_e': Provenance(
        value=0.36787944117144233,
        ptype=ProvenanceType.FROM_MATH,
        source="1/e: tiempo de decorrelación estándar",
    ),
    'sqrt_2': Provenance(
        value=1.4142135623730951,
        ptype=ProvenanceType.FROM_MATH,
        source="√2: diagonal del cuadrado unitario",
    ),
    'phi': Provenance(
        value=1.618033988749895,
        ptype=ProvenanceType.FROM_MATH,
        source="Razón áurea: (1+√5)/2",
    ),
}

# Constantes de teoría estadística
THEORY_CONSTANTS = {
    'z_95': Provenance(
        value=1.96,
        ptype=ProvenanceType.FROM_THEORY,
        source="z-score para 95% CI (distribución normal)",
        metadata={'confidence_level': 0.95},
    ),
    'z_99': Provenance(
        value=2.576,
        ptype=ProvenanceType.FROM_THEORY,
        source="z-score para 99% CI (distribución normal)",
        metadata={'confidence_level': 0.99},
    ),
    'tukey_k': Provenance(
        value=1.5,
        ptype=ProvenanceType.FROM_THEORY,
        source="Tukey fence multiplier para outliers: Q1-1.5*IQR, Q3+1.5*IQR",
        metadata={'reference': 'Tukey, J. W. (1977). Exploratory Data Analysis'},
    ),
    'tukey_k_extreme': Provenance(
        value=3.0,
        ptype=ProvenanceType.FROM_THEORY,
        source="Tukey fence para outliers extremos",
        metadata={'reference': 'Tukey, J. W. (1977). Exploratory Data Analysis'},
    ),
    'min_samples_clt': Provenance(
        value=30,
        ptype=ProvenanceType.FROM_THEORY,
        source="Mínimo para CLT (Teorema Central del Límite)",
        metadata={'note': 'Convención estadística, n≥30 para aproximación normal'},
    ),
    'min_samples_corr': Provenance(
        value=5,
        ptype=ProvenanceType.FROM_THEORY,
        source="Mínimo para correlación: n-1 grados de libertad, n≥5",
    ),
    'fisher_z': Provenance(
        value=None,  # Se calcula: 0.5 * ln((1+r)/(1-r))
        ptype=ProvenanceType.FROM_THEORY,
        source="Fisher z-transform para correlaciones",
        metadata={'formula': "z = 0.5 * ln((1+r)/(1-r))"},
    ),
}


class ProvenanceLogger:
    """
    Logger de procedencia para auditoría NORMA DURA.

    Registra todos los valores derivados con su procedencia.
    """

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("/root/NEO_EVA/logs/provenance")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.entries: List[Provenance] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log(self, provenance: Provenance, context: str = "") -> Provenance:
        """
        Registra un valor con su procedencia.

        Returns:
            El mismo Provenance para encadenamiento
        """
        entry = provenance.to_dict()
        entry['context'] = context
        self.entries.append(provenance)
        return provenance

    def log_from_data(self, value: Any, source: str,
                      dataset: str = "", statistic: str = "",
                      context: str = "") -> Provenance:
        """Registra un valor derivado de datos."""
        p = Provenance(
            value=value,
            ptype=ProvenanceType.FROM_DATA,
            source=source,
            metadata={'dataset': dataset, 'statistic': statistic},
        )
        return self.log(p, context)

    def log_from_theory(self, value: Any, source: str,
                        reference: str = "", context: str = "") -> Provenance:
        """Registra un valor derivado de teoría."""
        p = Provenance(
            value=value,
            ptype=ProvenanceType.FROM_THEORY,
            source=source,
            metadata={'reference': reference},
        )
        return self.log(p, context)

    def save(self):
        """Guarda log a archivo."""
        log_file = self.log_dir / f"provenance_{self.session_id}.json"
        with open(log_file, 'w') as f:
            json.dump([p.to_dict() for p in self.entries], f, indent=2)

    def get_audit_report(self) -> Dict:
        """Genera reporte de auditoría."""
        by_type = {}
        for p in self.entries:
            t = p.ptype.value
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(p.to_dict())

        # Detectar violaciones (UNKNOWN)
        violations = [p for p in self.entries if p.ptype == ProvenanceType.UNKNOWN]

        return {
            'session': self.session_id,
            'total_entries': len(self.entries),
            'by_type': {k: len(v) for k, v in by_type.items()},
            'violations': len(violations),
            'violation_details': [p.to_dict() for p in violations],
        }


# Instancia global del logger
_provenance_logger: Optional[ProvenanceLogger] = None

def get_provenance_logger() -> ProvenanceLogger:
    """Obtiene el logger global de procedencia."""
    global _provenance_logger
    if _provenance_logger is None:
        _provenance_logger = ProvenanceLogger()
    return _provenance_logger
