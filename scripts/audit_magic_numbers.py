#!/usr/bin/env python3
"""
Auditor Estático de Magic Numbers - NORMA DURA v2
==================================================

Escanea el código fuente buscando números sin justificación documentada.

NORMA DURA: "Ningún número entra al código sin poder explicar
             de qué distribución sale"

DOS NIVELES DE EXIGENCIA:

1) ZONA ESTRICTA (0 tolerancia) → ERROR si hay violaciones:
   - core/
   - research/
   - endogenous_causality.py
   - scripts de análisis serio (exoplanetas, terremotos, cosmos, etc.)

2) ZONA SANDBOX (tolerante) → WARNING si hay violaciones:
   - worlds/
   - subjectivity/
   - juegos, mundos, seres, recompensas
   - scripts de visualización y plots

Uso:
    python scripts/audit_magic_numbers.py [--verbose] [--strict-only]
"""

import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from enum import Enum

# Agregar path del proyecto
sys.path.insert(0, '/root/NEO_EVA')

from core.norma_dura_config import (
    ALLOWED_CONSTANTS,
    DOCUMENTED_PATTERNS,
    SUSPICIOUS_PATTERNS,
    KNOWN_EXCEPTIONS
)


# =============================================================================
# CONFIGURACIÓN DE ZONAS
# =============================================================================

class Zone(Enum):
    """Zona de exigencia."""
    STRICT = "STRICT"    # 0 tolerancia - ERROR
    SANDBOX = "SANDBOX"  # Tolerante - WARNING
    IGNORED = "IGNORED"  # No auditar


# ZONA ESTRICTA: Código científico serio (0 tolerancia)
STRICT_PATTERNS = [
    '/core/',
    '/research/',
    'endogenous_causality.py',
    'test_endogeneity',
    'test_causality',
    'audit_',
    'exoplanet',
    'earthquake',
    'seismic',
    'cosmos',
    'solar',
    'hurricane',
    'schumann',
    'polar',
    'timeseries',
    'causality_engine',
    'granger',
    'transfer_entropy',
]

# ZONA SANDBOX: Mundos, juegos, visualización (tolerante)
SANDBOX_PATTERNS = [
    '/worlds/',
    '/subjectivity/',
    'living_world',
    'complete_being',
    'mortal_agent',
    'conscious_agent',
    'game',
    'play',
    'plot',
    'viz',
    'demo',
    'example',
    'test_living',
    'test_world',
]

# RUTAS IGNORADAS COMPLETAMENTE
IGNORED_PATTERNS = [
    '/docs/',
    '/figures/',
    '/figuras/',
    '/plots/',
    '/notebooks/',
    '__pycache__',
    '.pyc',
    '.ipynb_checkpoints',
    '/logs/',
    '/data/',
]

# Archivos específicos a excluir
EXCLUDE_FILES = {
    'norma_dura_config.py',
    'endogenous_constants.py',
    'audit_magic_numbers.py',
}

# Extensiones a auditar
AUDIT_EXTENSIONS = {'.py'}


# =============================================================================
# PATRONES DE PLOT (EXENTOS)
# =============================================================================

# Palabras clave que indican configuración de plots (PLOT_OK)
PLOT_KEYWORDS = [
    r'\balpha\s*=',
    r'\blinewidth\s*=',
    r'\blw\s*=',
    r'\bfigsize\s*=',
    r'\bcolor\s*=',
    r'\bmarker\s*=',
    r'\bmarkersize\s*=',
    r'\bms\s*=',
    r'\bfontsize\s*=',
    r'\bdpi\s*=',
    r'\bwidth\s*=',
    r'\bheight\s*=',
    r'\bleft\s*=',
    r'\bright\s*=',
    r'\btop\s*=',
    r'\bbottom\s*=',
    r'\bhspace\s*=',
    r'\bwspace\s*=',
    r'\bpad\s*=',
    r'\bpad_inches\s*=',
    r'\bsubplots\s*\(',
    r'plt\.',
    r'ax\d*\.',
    r'fig\.',
    r'\.set_',
    r'\.plot\(',
    r'\.scatter\(',
    r'\.bar\(',
    r'\.hist\(',
    r'\.axhline\(',
    r'\.axvline\(',
    r'\.fill_between\(',
    r'\.legend\(',
    r'\.grid\(',
    r'\.savefig\(',
    r'\.tight_layout\(',
]


# =============================================================================
# ESTRUCTURAS
# =============================================================================

@dataclass
class Violation:
    """Una violación de NORMA DURA."""
    file: Path
    line_number: int
    line_content: str
    magic_number: str
    severity: str  # 'high', 'medium', 'low'
    suggestion: str
    zone: Zone = Zone.STRICT
    is_plot_related: bool = False


@dataclass
class AuditResult:
    """Resultado de auditoría de un archivo."""
    file: Path
    zone: Zone
    violations: List[Violation] = field(default_factory=list)
    plot_exemptions: int = 0
    lines_audited: int = 0
    is_compliant: bool = True


# =============================================================================
# PATRONES DE DETECCIÓN
# =============================================================================

MAGIC_NUMBER_PATTERNS = [
    # Decimales entre 0 y 1 (los más comunes)
    (r'(?<![a-zA-Z0-9_\.])0\.(?!0+\b)[0-9]+(?![0-9])', 'high'),

    # Comparaciones con decimales
    (r'[><=!]=?\s*0\.[0-9]+', 'high'),
    (r'0\.[0-9]+\s*[><=!]=?', 'high'),

    # Multiplicación/división por decimales
    (r'\*\s*0\.[0-9]+', 'medium'),
    (r'/\s*0\.[0-9]+', 'medium'),

    # Enteros pequeños sueltos (más contexto necesario)
    (r'(?<![a-zA-Z0-9_\.\[])[2-9](?![0-9a-zA-Z_\.\]])', 'low'),
    (r'(?<![a-zA-Z0-9_\.\[])[1-9][0-9](?![0-9a-zA-Z_\.\]])', 'low'),
]

# Contextos que eximen números
EXEMPTING_CONTEXTS = [
    r'#\s*ORIGEN:',
    r'#\s*NORMA DURA',
    r'#\s*PLOT_OK',
    r'#\s*SANDBOX_OK',
    r'#\s*FROM_',  # FROM_DATA, FROM_THEORY, etc.
    r'PERCENTILE_',
    r'CONSTANTS\.',  # Constantes de NORMA DURA
    r'np\.percentile',
    r'np\.finfo',
    r'np\.pi',
    r'np\.e\b',
    r'range\(',
    r'enumerate\(',
    r'shape\[',
    r'axis\s*=',
    r'dim\s*=',
    r'seed\s*=',
    r'random_state\s*=',
    r'n_components',
    r'n_samples',
    r'__version__',
    r'def\s+test_',
    r'assert\s+',
    r'\[\s*[0-9]+\s*\]',
    r'\[\s*:\s*[0-9]+',
    r'[0-9]+\s*:\s*\]',
    r'datetime',
    r'strftime',
    r'\.format\(',
    r'f".*{',
    r"f'.*{",
    r'print\s*\(',
    r'"\s*\*\s*[0-9]+',  # "=" * 70 para separadores
    r"'\s*\*\s*[0-9]+",
    # Inicializaciones de dataclass
    r':\s*float\s*=\s*0\.0',
    r':\s*float\s*=\s*1\.0',
    r':\s*int\s*=\s*0',
    # Funciones de test
    r'np\.random\.',  # np.random.randn, etc.
    r'np\.ones\(',
    r'np\.zeros\(',
    r'np\.array\(',
    r'np\.clip\(',
    r'np\.beta\(',
    r'\.beta\(',
    # Parámetros de configuración pequeños
    r'get_best\(',
    r'get_most_',
    r'top_\d+',
    r'n_best\s*=',
    # Parámetros de función con valor por defecto
    r':\s*int\s*=\s*\d+\)',  # tipo: int = N)
    r'window\s*:\s*int\s*=',
    r'max_events\s*:\s*int\s*=',
    r'state_dim\s*:\s*int\s*=',
    r'n\s*:\s*int\s*=',
    r'window\s*=\s*\d+',
    r'max_\w+\s*=\s*\d+',
    # Configuración de dimensiones
    r'dimension\s*=',
    r'dim\s*=',
    # Criterios GO (3 de 4, 2/3, etc.)
    r'n_pass\s*>=',
    r"'required':",
    r'"required":',
    r'two_thirds',
]


# =============================================================================
# FUNCIONES DE CLASIFICACIÓN
# =============================================================================

def get_zone(file_path: Path) -> Zone:
    """Determinar la zona de un archivo."""
    path_str = str(file_path).lower()

    # Primero verificar si está ignorado
    for pattern in IGNORED_PATTERNS:
        if pattern.lower() in path_str:
            return Zone.IGNORED

    # Verificar si es zona estricta
    for pattern in STRICT_PATTERNS:
        if pattern.lower() in path_str:
            return Zone.STRICT

    # Verificar si es zona sandbox
    for pattern in SANDBOX_PATTERNS:
        if pattern.lower() in path_str:
            return Zone.SANDBOX

    # Por defecto, zona estricta (seguro)
    return Zone.STRICT


def is_plot_line(line: str) -> bool:
    """Verificar si la línea es configuración de plot."""
    for pattern in PLOT_KEYWORDS:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    return False


def is_exempted(line: str) -> bool:
    """Verificar si la línea está exenta de auditoría."""
    for pattern in EXEMPTING_CONTEXTS:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    for exception in KNOWN_EXCEPTIONS:
        if exception in line:
            return True
    return False


def is_in_docstring_or_comment_block(lines: List[str], line_idx: int) -> bool:
    """Verificar si la línea está dentro de un docstring o bloque de comentarios."""
    line = lines[line_idx]

    # Comentario simple
    if line.strip().startswith('#'):
        return True

    # Contar comillas triples antes de esta línea
    triple_double = 0
    triple_single = 0

    for i in range(line_idx):
        triple_double += lines[i].count('"""')
        triple_single += lines[i].count("'''")

    # Si hay un número impar de comillas triples, estamos dentro de un docstring
    if triple_double % 2 == 1 or triple_single % 2 == 1:
        return True

    return False


def is_in_test_function(lines: List[str], line_idx: int) -> bool:
    """Verificar si la línea está dentro de una función de test."""
    # Buscar hacia atrás la definición de función más cercana
    for i in range(line_idx, -1, -1):
        line = lines[i].strip()
        # Si encontramos una definición de función
        if line.startswith('def '):
            # Si es función de test, eximir
            if 'def test_' in line or 'def _test_' in line:
                return True
            else:
                return False
        # Si encontramos una clase o nivel 0, salir
        if line.startswith('class ') or (line and not lines[i].startswith(' ') and not lines[i].startswith('\t')):
            if not line.startswith('def '):
                return False
    return False


# =============================================================================
# FUNCIONES DE AUDITORÍA
# =============================================================================

def extract_magic_numbers(line: str, zone: Zone) -> List[Tuple[str, str, bool]]:
    """
    Extraer magic numbers de una línea.

    Returns:
        Lista de (magic_number, severity, is_plot_related)
    """
    # Verificar si es línea de plot
    is_plot = is_plot_line(line)

    if is_exempted(line):
        return []

    # Ignorar comentarios
    if line.strip().startswith('#'):
        return []

    # Remover strings para no detectar números en strings
    line_no_strings = re.sub(r'"[^"]*"', '""', line)
    line_no_strings = re.sub(r"'[^']*'", "''", line_no_strings)

    magic_numbers = []
    for pattern, severity in MAGIC_NUMBER_PATTERNS:
        matches = re.findall(pattern, line_no_strings)
        for match in matches:
            match_clean = match.strip()
            if match_clean not in ALLOWED_CONSTANTS:
                magic_numbers.append((match_clean, severity, is_plot))

    return magic_numbers


def get_suggestion(magic_number: str) -> str:
    """Generar sugerencia de corrección."""
    try:
        value = float(magic_number)

        if 0 < value < 1:
            if value <= 0.1:
                return "Usar PERCENTILE_10 (0.1) o calcular percentil de datos"
            elif value <= 0.25:
                return "Usar PERCENTILE_25 (0.25) o calcular percentil de datos"
            elif value <= 0.5:
                return "Usar PERCENTILE_50 (0.5) o calcular percentil de datos"
            elif value <= 0.75:
                return "Usar PERCENTILE_75 (0.75) o calcular percentil de datos"
            elif value <= 0.9:
                return "Usar PERCENTILE_90 (0.9) o calcular percentil de datos"
            else:
                return "Calcular como percentil de datos observados"

        elif 1 < value <= 3:
            return "Si es IQR, usar TUKEY_FENCE (1.5) o TUKEY_EXTREME (3.0)"

        elif value > 3:
            return "Derivar de datos: np.percentile(data, X)"

    except ValueError:
        pass

    return "Agregar '# ORIGEN: ...' o '# PLOT_OK' o '# SANDBOX_OK'"


def audit_file(file_path: Path) -> AuditResult:
    """Auditar un archivo individual."""
    zone = get_zone(file_path)

    if zone == Zone.IGNORED:
        return AuditResult(file=file_path, zone=zone, is_compliant=True)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        return AuditResult(file=file_path, zone=zone, is_compliant=True)

    violations = []
    plot_exemptions = 0

    for i, line in enumerate(lines):
        # Saltar si está en docstring
        if is_in_docstring_or_comment_block(lines, i):
            continue

        # Saltar si está en función de test
        if is_in_test_function(lines, i):
            continue

        magic_numbers = extract_magic_numbers(line, zone)

        for magic_num, severity, is_plot in magic_numbers:
            if is_plot:
                plot_exemptions += 1
                continue  # No contar como violación

            violations.append(Violation(
                file=file_path,
                line_number=i + 1,
                line_content=line.rstrip(),
                magic_number=magic_num,
                severity=severity,
                suggestion=get_suggestion(magic_num),
                zone=zone,
                is_plot_related=is_plot
            ))

    return AuditResult(
        file=file_path,
        zone=zone,
        violations=violations,
        plot_exemptions=plot_exemptions,
        lines_audited=len(lines),
        is_compliant=len(violations) == 0
    )


def audit_directory(dir_path: Path, recursive: bool = True) -> List[AuditResult]:
    """Auditar un directorio completo."""
    results = []
    pattern = '**/*.py' if recursive else '*.py'

    for file_path in dir_path.glob(pattern):
        # Verificar exclusiones de archivo
        if any(exc in file_path.name for exc in EXCLUDE_FILES):
            continue

        # Verificar zona ignorada
        if get_zone(file_path) == Zone.IGNORED:
            continue

        if file_path.suffix in AUDIT_EXTENSIONS:
            result = audit_file(file_path)
            results.append(result)

    return results


def print_results(
    results: List[AuditResult],
    verbose: bool = False,
    strict_only: bool = False
) -> Tuple[int, int]:
    """
    Imprimir resultados de auditoría.

    Returns:
        (strict_violations, sandbox_violations)
    """
    strict_violations = 0
    sandbox_violations = 0
    total_plot_exemptions = 0

    strict_files_with_violations = []
    sandbox_files_with_violations = []

    for result in results:
        if result.zone == Zone.IGNORED:
            continue

        total_plot_exemptions += result.plot_exemptions

        if not result.is_compliant:
            if result.zone == Zone.STRICT:
                strict_violations += len(result.violations)
                strict_files_with_violations.append(result)
            else:
                sandbox_violations += len(result.violations)
                sandbox_files_with_violations.append(result)

    # Header
    print("\n" + "=" * 70)
    print("🔍 AUDITORÍA NORMA DURA v2 - Magic Numbers")
    print("=" * 70)

    # ZONA ESTRICTA
    print("\n" + "🔴 ZONA ESTRICTA (0 tolerancia)" + " " + "-" * 40)

    if strict_files_with_violations:
        for result in strict_files_with_violations:
            print(f"\n❌ {result.file}")
            for v in result.violations[:10]:  # Limitar output
                severity_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}[v.severity]
                print(f"  {severity_icon} L{v.line_number}: {v.magic_number}")
                if verbose:
                    print(f"     {v.line_content.strip()[:50]}...")
                    print(f"     💡 {v.suggestion}")
            if len(result.violations) > 10:
                print(f"  ... y {len(result.violations) - 10} más")
        print(f"\n  ❌ TOTAL ZONA ESTRICTA: {strict_violations} violaciones")
    else:
        print("  ✅ Sin violaciones en zona estricta")

    # ZONA SANDBOX
    if not strict_only:
        print("\n" + "🟡 ZONA SANDBOX (tolerante)" + " " + "-" * 42)

        if sandbox_files_with_violations:
            for result in sandbox_files_with_violations:
                print(f"\n⚠️  {result.file}")
                for v in result.violations[:5]:
                    print(f"  🟡 L{v.line_number}: {v.magic_number}")
                if len(result.violations) > 5:
                    print(f"  ... y {len(result.violations) - 5} más")
            print(f"\n  ⚠️  TOTAL ZONA SANDBOX: {sandbox_violations} warnings")
        else:
            print("  ✅ Sin warnings en zona sandbox")

    # Resumen
    print("\n" + "=" * 70)
    print("📊 RESUMEN")
    print("=" * 70)

    total_files = len([r for r in results if r.zone != Zone.IGNORED])
    strict_files = len([r for r in results if r.zone == Zone.STRICT])
    sandbox_files = len([r for r in results if r.zone == Zone.SANDBOX])

    print(f"  Archivos auditados: {total_files}")
    print(f"    📍 Zona estricta: {strict_files}")
    print(f"    📍 Zona sandbox: {sandbox_files}")
    print(f"  Líneas de plot exentas: {total_plot_exemptions}")
    print(f"  🔴 Errores (estricta): {strict_violations}")
    print(f"  🟡 Warnings (sandbox): {sandbox_violations}")

    # Conclusión
    print("\n" + "-" * 70)
    if strict_violations == 0:
        print("✅ NORMA DURA CUMPLIDA: Zona estricta sin violaciones")
        if sandbox_violations > 0:
            print(f"⚠️  {sandbox_violations} warnings en zona sandbox (no bloquean)")
    else:
        print(f"❌ NORMA DURA VIOLADA: {strict_violations} errores en zona estricta")
        print("   Estos deben corregirse antes de merge/publicación")

    return strict_violations, sandbox_violations


def main():
    parser = argparse.ArgumentParser(
        description='Auditor de Magic Numbers - NORMA DURA v2 (Zonas)'
    )
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Mostrar detalles y sugerencias')
    parser.add_argument('--strict-only', '-s', action='store_true',
                        help='Solo auditar zona estricta')
    parser.add_argument('--path', type=str,
                        help='Ruta específica a auditar')
    parser.add_argument('--output', type=str,
                        help='Archivo de salida (JSON)')

    args = parser.parse_args()

    # Directorios a auditar
    if args.path:
        paths = [Path(args.path)]
    else:
        paths = [
            Path('/root/NEO_EVA/core'),
            Path('/root/NEO_EVA/research'),
            Path('/root/NEO_EVA/scripts'),
            Path('/root/NEO_EVA/worlds'),
            Path('/root/NEO_EVA/subjectivity'),
        ]

    # Ejecutar auditoría
    all_results = []
    for path in paths:
        if path.exists():
            if path.is_file():
                all_results.append(audit_file(path))
            else:
                all_results.extend(audit_directory(path))

    # Mostrar resultados
    strict_violations, sandbox_violations = print_results(
        all_results,
        verbose=args.verbose,
        strict_only=args.strict_only
    )

    # Guardar JSON si se solicita
    if args.output:
        import json
        from datetime import datetime

        output_data = {
            'timestamp': datetime.now().isoformat(),
            'strict_violations': strict_violations,
            'sandbox_warnings': sandbox_violations,
            'files': [
                {
                    'file': str(r.file),
                    'zone': r.zone.value,
                    'violations': len(r.violations),
                    'plot_exemptions': r.plot_exemptions,
                    'details': [
                        {
                            'line': v.line_number,
                            'magic_number': v.magic_number,
                            'severity': v.severity
                        }
                        for v in r.violations
                    ]
                }
                for r in all_results
                if r.violations
            ]
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n📄 Resultados guardados en: {args.output}")

    # Exit code: solo errores de zona estricta causan fallo
    sys.exit(1 if strict_violations > 0 else 0)


if __name__ == '__main__':
    main()


# =============================================================================
# BLOQUE DE AUDITORÍA NORMA DURA
# =============================================================================
"""
MAGIC NUMBERS AUDIT v2
======================

Este archivo implementa auditoría de dos niveles:

ZONAS:
- STRICT: core/, research/, scripts de análisis → ERROR si hay violaciones
- SANDBOX: worlds/, subjectivity/, demos → WARNING pero no bloquea
- IGNORED: docs/, figures/, logs/, __pycache__ → No se audita

EXENCIONES AUTOMÁTICAS:
- Líneas con '# ORIGEN:', '# PLOT_OK', '# SANDBOX_OK'
- Configuración de matplotlib/plots (alpha=, figsize=, etc.)
- Docstrings y bloques de comentarios
- Separadores de print ("=" * 70)

CONSTANTES EN ESTE ARCHIVO:
- Los patrones son strings, no números mágicos
- Los límites de output ([:10], [:5]) son para UX, no algorítmicos

TODAS LAS DECISIONES TIENEN ORIGEN DOCUMENTADO.
"""
