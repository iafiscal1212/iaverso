#!/usr/bin/env python3
"""
Generador de Reporte NORMA DURA
===============================

Agrupa violaciones de audit_magic_numbers por archivo,
genera ranking de archivos con más violaciones,
y guarda reporte en docs/NORMA_DURA_REPORT.md

Uso:
    python scripts/norma_dura_report.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, '/root/NEO_EVA')

# Importar el auditor para reutilizar funciones
from scripts.audit_magic_numbers import (
    audit_file, audit_directory, get_zone, Zone,
    MAGIC_NUMBER_PATTERNS, EXEMPTING_CONTEXTS
)


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

OUTPUT_FILE = Path('/root/NEO_EVA/docs/NORMA_DURA_REPORT.md')

AUDIT_DIRS = [
    Path('/root/NEO_EVA/core'),
    Path('/root/NEO_EVA/research'),
    Path('/root/NEO_EVA/scripts'),
    Path('/root/NEO_EVA/cognition'),
    Path('/root/NEO_EVA/tools'),
    Path('/root/NEO_EVA/worlds'),
    Path('/root/NEO_EVA/subjectivity'),
]


# =============================================================================
# FUNCIONES DE ANÁLISIS
# =============================================================================

def collect_all_violations() -> Dict:
    """
    Recopilar todas las violaciones de todos los archivos.

    Returns:
        Dict con estadísticas completas
    """
    all_results = []

    for path in AUDIT_DIRS:
        if path.exists():
            all_results.extend(audit_directory(path))

    # Agrupar por zona
    strict_violations = []
    sandbox_violations = []

    for result in all_results:
        if result.zone == Zone.IGNORED:
            continue

        if result.violations:
            entry = {
                'file': str(result.file),
                'file_name': result.file.name,
                'zone': result.zone.value,
                'n_violations': len(result.violations),
                'by_severity': {
                    'high': len([v for v in result.violations if v.severity == 'high']),
                    'medium': len([v for v in result.violations if v.severity == 'medium']),
                    'low': len([v for v in result.violations if v.severity == 'low']),
                },
                'violations': [
                    {
                        'line': v.line_number,
                        'magic_number': v.magic_number,
                        'severity': v.severity,
                    }
                    for v in result.violations
                ]
            }

            if result.zone == Zone.STRICT:
                strict_violations.append(entry)
            else:
                sandbox_violations.append(entry)

    # Ordenar por número de violaciones (descendente)
    strict_violations.sort(key=lambda x: -x['n_violations'])
    sandbox_violations.sort(key=lambda x: -x['n_violations'])

    # Estadísticas globales
    total_strict = sum(e['n_violations'] for e in strict_violations)
    total_sandbox = sum(e['n_violations'] for e in sandbox_violations)

    return {
        'timestamp': datetime.now().isoformat(),
        'strict': {
            'files': strict_violations,
            'total_violations': total_strict,
            'n_files_with_violations': len(strict_violations),
        },
        'sandbox': {
            'files': sandbox_violations,
            'total_violations': total_sandbox,
            'n_files_with_violations': len(sandbox_violations),
        },
        'totals': {
            'total_violations': total_strict + total_sandbox,
            'strict_violations': total_strict,
            'sandbox_violations': total_sandbox,
        }
    }


def generate_markdown_report(data: Dict) -> str:
    """
    Generar reporte en formato Markdown.
    """
    lines = []

    # Header
    lines.append("# NORMA DURA - Reporte de Violaciones")
    lines.append("")
    lines.append(f"**Generado:** {data['timestamp']}")
    lines.append("")
    lines.append("> \"Ningún número entra al código sin poder explicar de qué distribución sale\"")
    lines.append("")

    # Resumen
    lines.append("## Resumen Ejecutivo")
    lines.append("")
    lines.append("| Métrica | Valor |")
    lines.append("|---------|-------|")
    lines.append(f"| Total violaciones | {data['totals']['total_violations']} |")
    lines.append(f"| 🔴 Zona Estricta | {data['totals']['strict_violations']} |")
    lines.append(f"| 🟡 Zona Sandbox | {data['totals']['sandbox_violations']} |")
    lines.append(f"| Archivos afectados (estricta) | {data['strict']['n_files_with_violations']} |")
    lines.append(f"| Archivos afectados (sandbox) | {data['sandbox']['n_files_with_violations']} |")
    lines.append("")

    # TOP 20 ZONA ESTRICTA
    lines.append("## 🔴 Top 20 Archivos - ZONA ESTRICTA")
    lines.append("")
    lines.append("Estos archivos **DEBEN** ser corregidos antes de publicación.")
    lines.append("")
    lines.append("| # | Archivo | Violaciones | Alta | Media | Baja |")
    lines.append("|---|---------|-------------|------|-------|------|")

    for i, entry in enumerate(data['strict']['files'][:20], 1):
        file_name = entry['file_name']
        n = entry['n_violations']
        high = entry['by_severity']['high']
        med = entry['by_severity']['medium']
        low = entry['by_severity']['low']
        lines.append(f"| {i} | `{file_name}` | {n} | {high} | {med} | {low} |")

    if len(data['strict']['files']) > 20:
        lines.append(f"| ... | _{len(data['strict']['files']) - 20} archivos más_ | | | | |")

    lines.append("")

    # Detalles de Top 5
    lines.append("### Detalles de Top 5")
    lines.append("")

    for i, entry in enumerate(data['strict']['files'][:5], 1):
        lines.append(f"#### {i}. `{entry['file_name']}`")
        lines.append("")
        lines.append(f"**Ruta:** `{entry['file']}`")
        lines.append(f"**Violaciones:** {entry['n_violations']}")
        lines.append("")

        # Mostrar primeras 10 violaciones
        lines.append("| Línea | Magic Number | Severidad |")
        lines.append("|-------|--------------|-----------|")
        for v in entry['violations'][:10]:
            sev_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}[v['severity']]
            lines.append(f"| {v['line']} | `{v['magic_number']}` | {sev_icon} {v['severity']} |")
        if len(entry['violations']) > 10:
            lines.append(f"| ... | _{len(entry['violations']) - 10} más_ | |")
        lines.append("")

    # TOP 10 ZONA SANDBOX
    lines.append("## 🟡 Top 10 Archivos - ZONA SANDBOX")
    lines.append("")
    lines.append("Estos archivos tienen warnings pero **no bloquean** el test.")
    lines.append("")
    lines.append("| # | Archivo | Warnings |")
    lines.append("|---|---------|----------|")

    for i, entry in enumerate(data['sandbox']['files'][:10], 1):
        lines.append(f"| {i} | `{entry['file_name']}` | {entry['n_violations']} |")

    lines.append("")

    # Guía de corrección
    lines.append("## Guía de Corrección")
    lines.append("")
    lines.append("### Para números decimales (0.3, 0.7, etc.):")
    lines.append("```python")
    lines.append("# ANTES (prohibido):")
    lines.append("if confidence > 0.7:")
    lines.append("")
    lines.append("# DESPUÉS (correcto):")
    lines.append("from core.norma_dura_config import CONSTANTS")
    lines.append("if confidence > CONSTANTS.PERCENTILE_75:  # ORIGEN: percentil 75 de U(0,1)")
    lines.append("```")
    lines.append("")
    lines.append("### Para umbrales derivados de datos:")
    lines.append("```python")
    lines.append("# ANTES (prohibido):")
    lines.append("threshold = 1.5")
    lines.append("")
    lines.append("# DESPUÉS (correcto):")
    lines.append("threshold = np.percentile(data, 75)  # ORIGEN: percentil 75 de datos observados")
    lines.append("```")
    lines.append("")
    lines.append("### Para valores iniciales:")
    lines.append("```python")
    lines.append("# ANTES (prohibido):")
    lines.append("initial_value = 0.5")
    lines.append("")
    lines.append("# DESPUÉS (correcto):")
    lines.append("initial_value = 0.5  # ORIGEN: máxima incertidumbre en escala [0,1]")
    lines.append("```")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Generado automáticamente por `scripts/norma_dura_report.py`*")

    return '\n'.join(lines)


def main():
    print("=" * 70)
    print("📊 GENERADOR DE REPORTE NORMA DURA")
    print("=" * 70)

    # Recopilar violaciones
    print("\n📋 Recopilando violaciones...")
    data = collect_all_violations()

    print(f"   Total violaciones: {data['totals']['total_violations']}")
    print(f"   🔴 Zona estricta: {data['totals']['strict_violations']}")
    print(f"   🟡 Zona sandbox: {data['totals']['sandbox_violations']}")

    # Generar Markdown
    print("\n📝 Generando reporte Markdown...")
    markdown = generate_markdown_report(data)

    # Guardar
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        f.write(markdown)

    print(f"   Guardado en: {OUTPUT_FILE}")

    # Mostrar top 5
    print("\n" + "=" * 70)
    print("🔴 TOP 5 ARCHIVOS - ZONA ESTRICTA (a corregir)")
    print("=" * 70)

    for i, entry in enumerate(data['strict']['files'][:5], 1):
        print(f"\n{i}. {entry['file_name']}: {entry['n_violations']} violaciones")
        print(f"   Ruta: {entry['file']}")
        print(f"   Alta: {entry['by_severity']['high']}, Media: {entry['by_severity']['medium']}, Baja: {entry['by_severity']['low']}")

    print("\n" + "=" * 70)
    print("✅ Reporte generado exitosamente")
    print("=" * 70)

    # Guardar JSON también
    json_file = Path('/root/NEO_EVA/logs/norma_dura/violations_report.json')
    json_file.parent.mkdir(parents=True, exist_ok=True)
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n📄 JSON guardado en: {json_file}")

    return data


if __name__ == '__main__':
    data = main()
