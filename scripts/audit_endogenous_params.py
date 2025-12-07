#!/usr/bin/env python3
"""
Auditor de Parámetros Endógenos
===============================

Analiza el log de parámetros endógenos para verificar
que todos los parámetros tienen origen documentado.

NORMA DURA: Todo parámetro debe poder explicar su origen.

Uso:
    python scripts/audit_endogenous_params.py [--summary] [--by-module]
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

sys.path.insert(0, '/root/NEO_EVA')

from core.norma_dura_config import PROVENANCE_TAGS


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

LOG_FILE = Path('/root/NEO_EVA/logs/endogenous/endogenous_params.jsonl')


# =============================================================================
# FUNCIONES DE ANÁLISIS
# =============================================================================

def load_params(log_file: Path) -> List[Dict]:
    """Cargar parámetros del log."""
    params = []
    if log_file.exists():
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        params.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return params


def validate_param(param: Dict) -> List[str]:
    """Validar un parámetro individual."""
    issues = []

    # Campos requeridos
    required = ['name', 'value', 'provenance', 'source_description']
    for field in required:
        if field not in param or param[field] is None:
            issues.append(f"Falta campo requerido: {field}")

    # Validar provenance
    if 'provenance' in param:
        if param['provenance'] not in PROVENANCE_TAGS:
            issues.append(f"Provenance inválido: {param['provenance']}")

    # Validar que FROM_DATA tiene estadísticas
    if param.get('provenance') == 'FROM_DATA':
        if not param.get('source_data_stats'):
            issues.append("FROM_DATA sin source_data_stats")

    # Validar que source_description no está vacía
    if param.get('source_description', '').strip() == '':
        issues.append("source_description vacío")

    return issues


def analyze_by_provenance(params: List[Dict]) -> Dict[str, Dict]:
    """Analizar parámetros por tipo de provenance."""
    by_prov = defaultdict(list)

    for param in params:
        prov = param.get('provenance', 'UNKNOWN')
        by_prov[prov].append(param)

    analysis = {}
    for prov, plist in by_prov.items():
        analysis[prov] = {
            'count': len(plist),
            'params': [p['name'] for p in plist],
            'modules': list(set(p.get('module', 'unknown') for p in plist))
        }

    return analysis


def analyze_by_module(params: List[Dict]) -> Dict[str, Dict]:
    """Analizar parámetros por módulo de origen."""
    by_module = defaultdict(list)

    for param in params:
        module = param.get('module', 'unknown')
        by_module[module].append(param)

    analysis = {}
    for module, plist in by_module.items():
        prov_counts = defaultdict(int)
        for p in plist:
            prov_counts[p.get('provenance', 'UNKNOWN')] += 1

        analysis[module] = {
            'count': len(plist),
            'by_provenance': dict(prov_counts),
            'params': [p['name'] for p in plist]
        }

    return analysis


def check_data_coverage(params: List[Dict]) -> Dict:
    """Verificar cobertura de datos en parámetros FROM_DATA."""
    from_data = [p for p in params if p.get('provenance') == 'FROM_DATA']

    stats = {
        'total_from_data': len(from_data),
        'with_stats': 0,
        'without_stats': 0,
        'sample_sizes': [],
        'min_samples': None,
        'max_samples': None,
        'median_samples': None,
    }

    for p in from_data:
        if p.get('source_data_stats'):
            stats['with_stats'] += 1
            n = p['source_data_stats'].get('n', 0)
            if n > 0:
                stats['sample_sizes'].append(n)
        else:
            stats['without_stats'] += 1

    if stats['sample_sizes']:
        import numpy as np
        sizes = np.array(stats['sample_sizes'])
        stats['min_samples'] = int(np.min(sizes))
        stats['max_samples'] = int(np.max(sizes))
        stats['median_samples'] = int(np.median(sizes))

    return stats


def generate_report(params: List[Dict], verbose: bool = False) -> Dict:
    """Generar reporte completo de auditoría."""
    # Validar todos los parámetros
    all_issues = []
    for param in params:
        issues = validate_param(param)
        if issues:
            all_issues.append({
                'param': param.get('name', 'unknown'),
                'issues': issues
            })

    # Análisis por provenance
    by_prov = analyze_by_provenance(params)

    # Análisis por módulo
    by_module = analyze_by_module(params)

    # Cobertura de datos
    data_coverage = check_data_coverage(params)

    report = {
        'timestamp': datetime.now().isoformat(),
        'total_params': len(params),
        'valid_params': len(params) - len(all_issues),
        'invalid_params': len(all_issues),
        'issues': all_issues if verbose else len(all_issues),
        'by_provenance': by_prov,
        'by_module': by_module,
        'data_coverage': data_coverage,
        'compliance_rate': (len(params) - len(all_issues)) / max(len(params), 1)
    }

    return report


def print_report(report: Dict, by_module: bool = False, verbose: bool = False):
    """Imprimir reporte de auditoría."""
    print("\n" + "=" * 70)
    print("📊 AUDITORÍA DE PARÁMETROS ENDÓGENOS")
    print("=" * 70)

    print(f"\n📅 Fecha: {report['timestamp']}")
    print(f"📁 Total parámetros: {report['total_params']}")
    print(f"✅ Válidos: {report['valid_params']}")
    print(f"❌ Inválidos: {report['invalid_params']}")
    print(f"📈 Tasa de cumplimiento: {report['compliance_rate']:.1%}")

    # Por provenance
    print("\n" + "-" * 50)
    print("📋 Por Tipo de Procedencia:")
    print("-" * 50)
    for prov, data in report['by_provenance'].items():
        icon = {
            'FROM_DATA': '📊',
            'FROM_DIST': '📈',
            'FROM_CALIB': '🔧',
            'FROM_THEORY': '📐'
        }.get(prov, '❓')
        print(f"  {icon} {prov}: {data['count']} parámetros")
        if verbose:
            for name in data['params'][:5]:
                print(f"      • {name}")
            if len(data['params']) > 5:
                print(f"      ... y {len(data['params']) - 5} más")

    # Por módulo (si se solicita)
    if by_module:
        print("\n" + "-" * 50)
        print("📦 Por Módulo:")
        print("-" * 50)
        for module, data in sorted(report['by_module'].items()):
            print(f"\n  📁 {module}: {data['count']} parámetros")
            for prov, count in data['by_provenance'].items():
                print(f"      {prov}: {count}")

    # Cobertura de datos
    dc = report['data_coverage']
    print("\n" + "-" * 50)
    print("📊 Cobertura de Datos (FROM_DATA):")
    print("-" * 50)
    print(f"  Total FROM_DATA: {dc['total_from_data']}")
    print(f"  Con estadísticas: {dc['with_stats']}")
    print(f"  Sin estadísticas: {dc['without_stats']}")
    if dc['median_samples']:
        print(f"  Tamaño muestral (min/med/max): {dc['min_samples']}/{dc['median_samples']}/{dc['max_samples']}")

    # Issues (si hay)
    if report['invalid_params'] > 0:
        print("\n" + "-" * 50)
        print("⚠️  Problemas Encontrados:")
        print("-" * 50)
        if isinstance(report['issues'], list):
            for issue_item in report['issues'][:10]:
                print(f"  ❌ {issue_item['param']}:")
                for issue in issue_item['issues']:
                    print(f"      • {issue}")
            if len(report['issues']) > 10:
                print(f"  ... y {len(report['issues']) - 10} más")
        else:
            print(f"  {report['issues']} parámetros con problemas")
            print("  Ejecutar con --verbose para ver detalles")

    # Conclusión
    print("\n" + "=" * 70)
    if report['compliance_rate'] >= 1.0:
        print("✅ ¡TODOS LOS PARÁMETROS CUMPLEN NORMA DURA!")
    elif report['compliance_rate'] >= 0.9:
        print("🟡 CASI COMPLETO - Algunos parámetros necesitan corrección")
    else:
        print("❌ SE REQUIERE CORRECCIÓN - Muchos parámetros sin documentar")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Auditor de Parámetros Endógenos')
    parser.add_argument('--summary', '-s', action='store_true', help='Solo resumen')
    parser.add_argument('--by-module', '-m', action='store_true', help='Desglose por módulo')
    parser.add_argument('--verbose', '-v', action='store_true', help='Mostrar detalles')
    parser.add_argument('--log-file', type=str, help='Archivo de log alternativo')
    parser.add_argument('--output', type=str, help='Guardar reporte en JSON')

    args = parser.parse_args()

    # Cargar log
    log_file = Path(args.log_file) if args.log_file else LOG_FILE

    if not log_file.exists():
        print(f"⚠️  Log no encontrado: {log_file}")
        print("   No hay parámetros endógenos registrados aún.")
        print("   Los parámetros se registran durante la ejecución de los scripts.")
        sys.exit(0)

    params = load_params(log_file)

    if not params:
        print("⚠️  El log está vacío.")
        sys.exit(0)

    # Generar reporte
    report = generate_report(params, verbose=args.verbose)

    # Mostrar
    print_report(report, by_module=args.by_module, verbose=args.verbose)

    # Guardar si se solicita
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📄 Reporte guardado en: {args.output}")

    # Exit code
    sys.exit(0 if report['compliance_rate'] >= 1.0 else 1)


if __name__ == '__main__':
    main()


# =============================================================================
# BLOQUE DE AUDITORÍA NORMA DURA
# =============================================================================
"""
MAGIC NUMBERS AUDIT
==================

Este archivo es parte de la infraestructura de NORMA DURA.

CONSTANTES:
- compliance_rate >= 1.0: 100% es el umbral de cumplimiento total
- compliance_rate >= 0.9: 90% es "casi completo" (ORIGEN: convención)
- [:5], [:10]: Límites de preview (ORIGEN: UX, mostrar suficiente sin abrumar)

TODAS LAS DECISIONES TIENEN ORIGEN DOCUMENTADO.
"""
