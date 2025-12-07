#!/usr/bin/env python3
"""
Auditor de Independencia de Recompensas v2
==========================================

Verifica que el código científico no depende de sistemas de recompensa.

NORMA DURA para Reward Independence:

1) ZONA ESTRICTA (core/, research/, módulos del paper):
   - NO puede aparecer: "reward", "compute_reward", "maximize_reward",
     "discount", "policy_gradient" en código ejecutable.
   - Si aparecen → ERROR (exit 1)

2) ZONA SANDBOX (worlds/, subjectivity/, juegos):
   - Se permite reward, pero DEBE tener comentario de procedencia:
     # FROM_THEORY: ... o # FROM_CALIB: ...
   - Sin comentario → WARNING (no bloquea)

Uso:
    python scripts/audit_reward_independence.py [--verbose] [--strict-only]
"""

import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, '/root/NEO_EVA')

from core.norma_dura_config import PROVENANCE_TAGS


# =============================================================================
# CONFIGURACIÓN DE ZONAS
# =============================================================================

class Zone(Enum):
    """Zona de exigencia."""
    STRICT = "STRICT"    # 0 tolerancia a reward
    SANDBOX = "SANDBOX"  # Permite reward documentado
    IGNORED = "IGNORED"  # No auditar


# ZONA ESTRICTA: Código científico (0 tolerancia a reward)
STRICT_PATTERNS = [
    '/core/',
    '/research/',
    '/cognition/',
    '/tools/',
    'endogenous_causality',
    'internal_causality',
    'test_endogeneity',
    'test_causality',
    'audit_',
    'exoplanet',
    'earthquake',
    'seismic',
    'cosmos',
    'solar',
    'timeseries',
    'transfer_entropy',
    'granger',
    'phase_coherence',
    '_spec',
    'WORLD-1',
]

# ZONA SANDBOX: Juegos, mundos, simulaciones
SANDBOX_PATTERNS = [
    '/worlds/',
    '/subjectivity/',
    'living_world',
    'complete_being',
    'mortal_agent',
    'conscious_agent',
    'game',
    'play',
    'demo',
    'example',
    'test_living',
    'test_world',
    'test_5agents',
    'benchmark',
]

# RUTAS IGNORADAS
IGNORED_PATTERNS = [
    '/docs/',
    '/figures/',
    '/figuras/',
    '/plots/',
    '/notebooks/',
    '__pycache__',
    '.pyc',
    '/logs/',
    '/data/',
    'audit_reward',  # Este mismo archivo
]


# =============================================================================
# PALABRAS PROHIBIDAS EN ZONA ESTRICTA
# =============================================================================

FORBIDDEN_REWARD_PATTERNS = [
    r'\breward\b',
    r'\bcompute_reward\b',
    r'\bcalculate_reward\b',
    r'\bget_reward\b',
    r'\bmaximize_reward\b',
    r'\bminimize_reward\b',
    r'\breward_function\b',
    r'\bdiscount\b',
    r'\bdiscount_factor\b',
    r'\bgamma\b',  # Común para discount
    r'\bpolicy_gradient\b',
    r'\bvalue_function\b',
    r'\bq_value\b',
    r'\bq_learning\b',
    r'\bactor_critic\b',
    r'\bppo\b',
    r'\breinforce\b',
    r'\btd_error\b',
    r'\btemporal_difference\b',
]

# Patrones que indican documentación válida
VALID_PROVENANCE_PATTERNS = [
    r'#\s*FROM_THEORY:',
    r'#\s*FROM_CALIB:',
    r'#\s*FROM_DATA:',
    r'#\s*ORIGEN:',
    r'#\s*SANDBOX_OK',
]


# =============================================================================
# ESTRUCTURAS
# =============================================================================

@dataclass
class RewardViolation:
    """Una violación de independencia de reward."""
    file: Path
    line_number: int
    line_content: str
    matched_pattern: str
    zone: Zone
    has_provenance: bool
    severity: str  # 'error' o 'warning'


@dataclass
class AuditResult:
    """Resultado de auditoría de un archivo."""
    file: Path
    zone: Zone
    errors: List[RewardViolation] = field(default_factory=list)
    warnings: List[RewardViolation] = field(default_factory=list)
    is_compliant: bool = True


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

    # Por defecto, zona estricta
    return Zone.STRICT


def has_valid_provenance(line: str, prev_lines: List[str]) -> bool:
    """
    Verificar si la línea o las anteriores tienen documentación de procedencia.
    """
    all_lines = prev_lines[-3:] + [line]  # Últimas 3 líneas + actual
    for l in all_lines:
        for pattern in VALID_PROVENANCE_PATTERNS:
            if re.search(pattern, l, re.IGNORECASE):
                return True
    return False


def is_in_string_or_comment(line: str) -> bool:
    """Verificar si el contenido relevante está en string o comentario."""
    stripped = line.strip()

    # Comentario completo
    if stripped.startswith('#'):
        return True

    # Docstring
    if stripped.startswith('"""') or stripped.startswith("'''"):
        return True

    return False


def is_in_docstring_block(lines: List[str], line_idx: int) -> bool:
    """Verificar si estamos dentro de un docstring."""
    triple_double = 0
    triple_single = 0

    for i in range(line_idx):
        triple_double += lines[i].count('"""')
        triple_single += lines[i].count("'''")

    if triple_double % 2 == 1 or triple_single % 2 == 1:
        return True

    return False


# =============================================================================
# FUNCIONES DE AUDITORÍA
# =============================================================================

def find_reward_violations(
    lines: List[str],
    file_path: Path,
    zone: Zone
) -> Tuple[List[RewardViolation], List[RewardViolation]]:
    """
    Buscar violaciones de reward independence.

    Returns:
        (errors, warnings)
    """
    errors = []
    warnings = []

    for i, line in enumerate(lines):
        # Saltar comentarios, docstrings, strings
        if is_in_string_or_comment(line):
            continue
        if is_in_docstring_block(lines, i):
            continue

        # Buscar patrones de reward
        for pattern in FORBIDDEN_REWARD_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                # Verificar si tiene documentación de procedencia
                prev_lines = lines[max(0, i-3):i]
                has_prov = has_valid_provenance(line, prev_lines)

                violation = RewardViolation(
                    file=file_path,
                    line_number=i + 1,
                    line_content=line.rstrip(),
                    matched_pattern=pattern,
                    zone=zone,
                    has_provenance=has_prov,
                    severity='error' if zone == Zone.STRICT else 'warning'
                )

                if zone == Zone.STRICT:
                    # En zona estricta, CUALQUIER reward es error
                    errors.append(violation)
                else:
                    # En zona sandbox, solo warning si no tiene documentación
                    if not has_prov:
                        warnings.append(violation)

                break  # Solo una violación por línea

    return errors, warnings


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

    errors, warnings = find_reward_violations(lines, file_path, zone)

    return AuditResult(
        file=file_path,
        zone=zone,
        errors=errors,
        warnings=warnings,
        is_compliant=len(errors) == 0
    )


def audit_directory(dir_path: Path) -> List[AuditResult]:
    """Auditar un directorio."""
    results = []

    for file_path in dir_path.glob('**/*.py'):
        if get_zone(file_path) == Zone.IGNORED:
            continue
        if '__pycache__' in str(file_path):
            continue

        result = audit_file(file_path)
        results.append(result)

    return results


def print_results(
    results: List[AuditResult],
    verbose: bool = False,
    strict_only: bool = False
) -> Tuple[int, int]:
    """
    Imprimir resultados.

    Returns:
        (total_errors, total_warnings)
    """
    total_errors = 0
    total_warnings = 0

    strict_files_with_errors = []
    sandbox_files_with_warnings = []

    for result in results:
        if result.zone == Zone.IGNORED:
            continue

        if result.errors:
            total_errors += len(result.errors)
            strict_files_with_errors.append(result)

        if result.warnings:
            total_warnings += len(result.warnings)
            sandbox_files_with_warnings.append(result)

    # Header
    print("\n" + "=" * 70)
    print("🎯 AUDITORÍA DE INDEPENDENCIA DE RECOMPENSAS v2")
    print("=" * 70)

    # ZONA ESTRICTA - ERRORES
    print("\n" + "🔴 ZONA ESTRICTA (reward prohibido)" + " " + "-" * 30)

    if strict_files_with_errors:
        for result in strict_files_with_errors:
            print(f"\n❌ {result.file}")
            for v in result.errors[:10]:
                print(f"  🔴 L{v.line_number}: '{v.matched_pattern}'")
                if verbose:
                    print(f"     {v.line_content.strip()[:50]}...")
            if len(result.errors) > 10:
                print(f"  ... y {len(result.errors) - 10} más")
        print(f"\n  ❌ TOTAL ERRORES: {total_errors}")
    else:
        print("  ✅ Sin violaciones en zona estricta")

    # ZONA SANDBOX - WARNINGS
    if not strict_only:
        print("\n" + "🟡 ZONA SANDBOX (reward sin documentar)" + " " + "-" * 25)

        if sandbox_files_with_warnings:
            for result in sandbox_files_with_warnings:
                print(f"\n⚠️  {result.file}")
                for v in result.warnings[:5]:
                    print(f"  🟡 L{v.line_number}: '{v.matched_pattern}'")
                    if verbose:
                        print(f"     Agregar: # FROM_CALIB: ... o # FROM_THEORY: ...")
                if len(result.warnings) > 5:
                    print(f"  ... y {len(result.warnings) - 5} más")
            print(f"\n  ⚠️  TOTAL WARNINGS: {total_warnings}")
        else:
            print("  ✅ Todos los rewards tienen documentación")

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
    print(f"  🔴 Errores (estricta): {total_errors}")
    print(f"  🟡 Warnings (sandbox): {total_warnings}")

    # Conclusión
    print("\n" + "-" * 70)
    if total_errors == 0:
        print("✅ INDEPENDENCIA DE RECOMPENSAS VERIFICADA")
        if total_warnings > 0:
            print(f"⚠️  {total_warnings} warnings en sandbox (agregar FROM_CALIB o FROM_THEORY)")
    else:
        print(f"❌ VIOLACIÓN DE INDEPENDENCIA: {total_errors} errores en zona estricta")
        print("   El código científico NO debe usar sistemas de reward")
        print("   Mover a zona sandbox o eliminar")

    return total_errors, total_warnings


def main():
    parser = argparse.ArgumentParser(
        description='Auditor de Independencia de Recompensas v2'
    )
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Mostrar detalles')
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
            Path('/root/NEO_EVA/cognition'),
            Path('/root/NEO_EVA/tools'),
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
    total_errors, total_warnings = print_results(
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
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'errors': [
                {
                    'file': str(v.file),
                    'line': v.line_number,
                    'pattern': v.matched_pattern,
                    'content': v.line_content[:80]
                }
                for r in all_results
                for v in r.errors
            ],
            'warnings': [
                {
                    'file': str(v.file),
                    'line': v.line_number,
                    'pattern': v.matched_pattern,
                }
                for r in all_results
                for v in r.warnings
            ]
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n📄 Resultados guardados en: {args.output}")

    # Exit code: solo errores de zona estricta causan fallo
    sys.exit(1 if total_errors > 0 else 0)


if __name__ == '__main__':
    main()


# =============================================================================
# BLOQUE DE AUDITORÍA NORMA DURA
# =============================================================================
"""
MAGIC NUMBERS AUDIT v2
======================

PALABRAS PROHIBIDAS EN ZONA ESTRICTA:
- reward, compute_reward, maximize_reward
- discount, discount_factor, gamma
- policy_gradient, value_function, q_value
- ppo, reinforce, actor_critic, td_error

DOCUMENTACIÓN VÁLIDA PARA SANDBOX:
- # FROM_THEORY: justificación teórica
- # FROM_CALIB: tuning empírico documentado
- # ORIGEN: descripción del origen

ZONAS:
- STRICT: core/, research/, cognition/ → ERROR si hay reward
- SANDBOX: worlds/, subjectivity/ → WARNING sin documentación

TODAS LAS DECISIONES TIENEN ORIGEN DOCUMENTADO.
"""
