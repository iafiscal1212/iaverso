#!/usr/bin/env python3
"""
Configuración Central de NORMA DURA
===================================

"Ningún número entra al código sin poder explicar de qué distribución sale"

Este módulo define:
1. ALLOWED_CONSTANTS: Constantes permitidas con justificación matemática
2. PROVENANCE_TAGS: Etiquetas para documentar origen de parámetros
3. Funciones de validación para auditoría

NORMA DURA: Todo número debe tener origen documentado.
"""

import numpy as np
from typing import Dict, List, Set
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# CONSTANTES MATEMÁTICAS PERMITIDAS
# =============================================================================

ALLOWED_CONSTANTS: Dict[str, str] = {
    # Constantes matemáticas fundamentales
    "np.pi": "constante matemática pi (3.14159...)",
    "np.e": "constante matemática e (2.71828...)",
    "math.pi": "constante matemática pi (3.14159...)",
    "math.e": "constante matemática e (2.71828...)",

    # Escalares neutros
    "0": "cero escalar neutro (identidad aditiva)",
    "1": "uno escalar neutro (identidad multiplicativa)",
    "0.0": "cero flotante neutro",
    "1.0": "uno flotante neutro",
    "-1": "menos uno (inversión de signo)",

    # Constantes de precisión numérica
    "np.finfo(float).eps": "épsilon de máquina (~2.2e-16)",
    "sys.float_info.epsilon": "épsilon de máquina",

    # Estadística estándar - Intervalos de confianza
    "1.96": "z-score para 95% intervalo de confianza (dos colas)",
    "2.576": "z-score para 99% intervalo de confianza (dos colas)",
    "1.645": "z-score para 90% intervalo de confianza (dos colas)",
    "3.291": "z-score para 99.9% intervalo de confianza (dos colas)",

    # Estadística estándar - Outliers
    "1.5": "multiplicador de Tukey para outliers moderados (IQR)",
    "3.0": "multiplicador de Tukey para outliers extremos (IQR)",

    # Tiempo de decorrelación
    "1/np.e": "tiempo de decorrelación estándar (τ donde acf < 1/e)",
    "0.368": "aproximación de 1/e para decorrelación",

    # Percentiles de U(0,1) - Base de NORMA DURA
    "0.1": "percentil 10 de U(0,1) - umbral muy bajo",
    "0.25": "percentil 25 de U(0,1) - Q1, umbral bajo",
    "0.5": "percentil 50 de U(0,1) - mediana",
    "0.75": "percentil 75 de U(0,1) - Q3, umbral alto",
    "0.9": "percentil 90 de U(0,1) - umbral muy alto",

    # Muestras mínimas para estadísticas
    "5": "mínimo para std confiable (MIN_SAMPLES_FOR_STATISTICS)",
    "10": "mínimo para percentiles confiables",
    "30": "regla empírica para aproximación normal (CLT)",

    # Correlación
    "2/sqrt(n)": "umbral de significancia para correlación",

    # Fracciones comunes con justificación
    "0.5": "mitad, división equitativa",
    "2": "duplicación, factor par mínimo",
}


# =============================================================================
# ETIQUETAS DE PROCEDENCIA
# =============================================================================

class ProvenanceTag(Enum):
    """Etiquetas para documentar origen de parámetros."""

    FROM_DATA = "FROM_DATA"       # Derivado de datos observados (percentiles, estadísticas)
    FROM_DIST = "FROM_DIST"       # Percentil de distribución teórica (U(0,1), Normal, etc.)
    FROM_CALIB = "FROM_CALIB"     # Calibrado durante fase de entrenamiento
    FROM_THEORY = "FROM_THEORY"   # Constante matemática o física con justificación teórica


PROVENANCE_TAGS: List[str] = [tag.value for tag in ProvenanceTag]


# =============================================================================
# PATRONES REGEX PARA AUDITORÍA
# =============================================================================

# Patrones que detectan magic numbers sospechosos
SUSPICIOUS_PATTERNS: List[str] = [
    r'(?<![a-zA-Z0-9_])0\.[0-9]{1,2}(?![0-9])',  # Decimales como 0.3, 0.85
    r'(?<![a-zA-Z0-9_])[1-9][0-9]?(?![0-9\.])',   # Enteros pequeños sin contexto
    r'>\s*0\.[0-9]+',                              # Comparaciones con decimales
    r'<\s*0\.[0-9]+',                              # Comparaciones con decimales
    r'\*\s*0\.[0-9]+',                             # Multiplicación por decimales
]

# Patrones que indican origen documentado (PERMITIDOS)
DOCUMENTED_PATTERNS: List[str] = [
    r'#\s*ORIGEN:',                    # Comentario de origen
    r'PERCENTILE_\d+',                 # Constante de percentil
    r'np\.percentile\(',               # Cálculo de percentil
    r'np\.finfo\(',                    # Precisión de máquina
    r'np\.pi|np\.e',                   # Constantes matemáticas
    r'ConfidenceLevel\.',              # Niveles de confianza documentados
    r'TraitThresholds',                # Umbrales calibrados
    r'get_correlation_threshold',      # Función de umbral
    r'MATHEMATICAL_CONSTANTS\[',       # Constantes documentadas
]

# Excepciones conocidas (no son magic numbers)
KNOWN_EXCEPTIONS: Set[str] = {
    'range(0,',
    'range(1,',
    'enumerate(',
    'shape[0]',
    'shape[1]',
    'axis=0',
    'axis=1',
    'dim=0',
    'dim=1',
    '== 0',
    '!= 0',
    'len(',
    '// 2',
    '% 2',
    '* 2',
    '/ 2',
    '+ 1',
    '- 1',
    'n_components',
    'n_samples',
    'n_features',
}


# =============================================================================
# CONSTANTES ENDÓGENAS ESTÁNDAR
# =============================================================================

@dataclass
class EndogenousConstants:
    """
    Constantes endógenas basadas en percentiles de U(0,1).

    Estas son las constantes base para NORMA DURA.
    """
    # Percentiles de distribución uniforme U(0,1)
    PERCENTILE_10: float = 0.1
    PERCENTILE_25: float = 0.25
    PERCENTILE_50: float = 0.5
    PERCENTILE_75: float = 0.75
    PERCENTILE_90: float = 0.9

    # Constantes de decorrelación
    DECAY_RATE: float = 1 / np.e  # ≈ 0.368

    # Precisión numérica
    MACHINE_EPS: float = np.finfo(float).eps

    # Muestras mínimas
    MIN_SAMPLES_STATS: int = 5
    MIN_SAMPLES_PERCENTILES: int = 10
    MIN_SAMPLES_CLT: int = 30

    # Intervalos de confianza
    Z_95: float = 1.96
    Z_99: float = 2.576

    # Outliers (Tukey)
    TUKEY_MODERATE: float = 1.5
    TUKEY_EXTREME: float = 3.0


# Instancia global
CONSTANTS = EndogenousConstants()


# =============================================================================
# FUNCIONES DE VALIDACIÓN
# =============================================================================

def is_allowed_constant(value_str: str) -> bool:
    """Verificar si un valor es una constante permitida."""
    return value_str in ALLOWED_CONSTANTS


def get_constant_justification(value_str: str) -> str:
    """Obtener justificación de una constante."""
    return ALLOWED_CONSTANTS.get(value_str, "NO JUSTIFICADO")


def is_documented(line: str) -> bool:
    """Verificar si una línea tiene documentación de origen."""
    import re
    for pattern in DOCUMENTED_PATTERNS:
        if re.search(pattern, line):
            return True
    return False


def is_known_exception(line: str) -> bool:
    """Verificar si la línea contiene una excepción conocida."""
    for exc in KNOWN_EXCEPTIONS:
        if exc in line:
            return True
    return False


def validate_provenance_tag(tag: str) -> bool:
    """Validar que una etiqueta de procedencia es válida."""
    return tag in PROVENANCE_TAGS


# =============================================================================
# BLOQUE DE AUDITORÍA NORMA DURA
# =============================================================================
"""
MAGIC NUMBERS AUDIT
==================

Este archivo define las constantes permitidas y su justificación.

CONSTANTES DE PERCENTILES (U(0,1)):
- 0.1: percentil 10
- 0.25: percentil 25 (Q1)
- 0.5: percentil 50 (mediana)
- 0.75: percentil 75 (Q3)
- 0.9: percentil 90

CONSTANTES ESTADÍSTICAS:
- 1.96: z-score 95% CI (ORIGEN: tabla normal estándar)
- 2.576: z-score 99% CI (ORIGEN: tabla normal estándar)
- 1.5: Tukey fence (ORIGEN: definición estándar de Tukey)
- 3.0: Tukey extreme (ORIGEN: definición estándar de Tukey)

CONSTANTES DE DECORRELACIÓN:
- 1/np.e: tiempo de decorrelación (ORIGEN: definición de autocorrelación)

TODAS LAS CONSTANTES TIENEN ORIGEN DOCUMENTADO.
"""
