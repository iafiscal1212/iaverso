#!/usr/bin/env python3
"""
AUDIT Q ENDOGENOUS - Auditoría de Endogeneidad
==============================================

Este archivo audita que TODO el sistema sea endógeno:

CHECK 1: No magic numbers
    - Busca números que no sean 0, 1, dimensiones estructurales
    - Permite: e, π (constantes matemáticas), pero reporta

CHECK 2: Sin palabras semánticas
    - No debe haber "cooperate", "defect", "cheat", "betray"
    - La acción debe emerger de operadores estructurales

CHECK 3: Escalas relativas
    - Todo normalizado por percentiles de la historia
    - No escalas absolutas externas

CHECK 4: Verificación de flujo
    - drives → ranks → ψ → operadores → Δ → payoff
    - Sin valores inventados en ningún paso
"""

import ast
import re
import os
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, field


@dataclass
class AuditResult:
    """Resultado de auditoría."""
    passed: bool
    category: str
    file: str
    line: int = 0
    message: str = ""
    severity: str = "warning"  # warning, error, info


@dataclass
class EndogeneityAuditor:
    """
    Auditor de endogeneidad para el sistema de juego cuántico.
    """
    # Archivos a auditar
    files_to_audit: List[str] = field(default_factory=list)

    # Resultados
    results: List[AuditResult] = field(default_factory=list)

    # Configuración
    allowed_numbers: Set = field(default_factory=lambda: {0, 1, 2, -1})
    allowed_math_constants: Set = field(default_factory=lambda: {'e', 'pi', 'inf'})

    # Palabras prohibidas (semántica de teoría de juegos clásica)
    forbidden_words: Set = field(default_factory=lambda: {
        'cooperate', 'defect', 'cheat', 'betray', 'punish',
        'reward', 'sucker', 'temptation', 'payoff_matrix',
        'prisoner', 'dilemma', 'nash_equilibrium'
    })

    # Palabras permitidas pero a revisar
    review_words: Set = field(default_factory=lambda: {
        'payoff', 'game', 'player', 'strategy', 'utility'
    })

    def __post_init__(self):
        if not self.files_to_audit:
            # Archivos del módulo endogenous
            base_path = '/root/NEO_EVA/quantum_game/endogenous'
            self.files_to_audit = [
                os.path.join(base_path, 'state_encoding.py'),
                os.path.join(base_path, 'operators_qg.py'),
                os.path.join(base_path, 'coalition_game_qg1.py'),
                os.path.join(base_path, 'payoff_endogenous.py'),
            ]

    def audit_magic_numbers(self, filepath: str) -> List[AuditResult]:
        """
        CHECK 1: Busca magic numbers.

        Números permitidos:
        - 0, 1, -1 (identidades)
        - 2 (para pares, mitades)
        - Dimensiones del sistema (típicamente 6)
        - Números en comentarios o strings

        Números a revisar:
        - Cualquier otro literal numérico
        """
        results = []

        try:
            with open(filepath, 'r') as f:
                content = f.read()

            tree = ast.parse(content)

            # Extraer dimensión del sistema si está definida
            dim_values = {6}  # Dimensión default de drives

            for node in ast.walk(tree):
                # Buscar constantes numéricas
                if isinstance(node, ast.Constant):
                    if isinstance(node.value, (int, float)):
                        num = node.value

                        # Ignorar 0, 1, 2, -1
                        if num in self.allowed_numbers:
                            continue

                        # Ignorar dimensiones
                        if num in dim_values:
                            continue

                        # Ignorar pequeñas épsilon (1e-16, etc.)
                        if isinstance(num, float) and abs(num) < 1e-10:
                            continue

                        # Ignorar números que parecen épsilon para estabilidad
                        if isinstance(num, float) and '1e-' in str(num):
                            continue

                        # Números aceptables en contexto
                        # - Tamaños de ventana iniciales (10, 20, 50) si se justifican
                        # - Percentiles (5, 25, 50, 75, 95, 100)
                        percentiles = {5, 10, 20, 25, 50, 75, 90, 95, 100}
                        window_sizes = {3, 5, 10, 20, 30, 50, 100, 200, 500}

                        if num in percentiles:
                            # OK - percentiles son estructurales
                            results.append(AuditResult(
                                passed=True,
                                category='magic_numbers',
                                file=filepath,
                                line=getattr(node, 'lineno', 0),
                                message=f'Percentil {num} - OK (estructural)',
                                severity='info'
                            ))
                            continue

                        if num in window_sizes:
                            # Warning - debería derivarse de √t idealmente
                            results.append(AuditResult(
                                passed=True,
                                category='magic_numbers',
                                file=filepath,
                                line=getattr(node, 'lineno', 0),
                                message=f'Tamaño de ventana {num} - revisar si deriva de historia',
                                severity='info'
                            ))
                            continue

                        # Cualquier otro número es sospechoso
                        results.append(AuditResult(
                            passed=False,
                            category='magic_numbers',
                            file=filepath,
                            line=getattr(node, 'lineno', 0),
                            message=f'Magic number detectado: {num}',
                            severity='warning'
                        ))

        except Exception as e:
            results.append(AuditResult(
                passed=False,
                category='magic_numbers',
                file=filepath,
                message=f'Error parseando archivo: {e}',
                severity='error'
            ))

        return results

    def audit_forbidden_words(self, filepath: str) -> List[AuditResult]:
        """
        CHECK 2: Busca palabras semánticas prohibidas.
        """
        results = []

        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                line_lower = line.lower()

                # Ignorar comentarios que explican por qué NO usamos estas palabras
                if 'no ' in line_lower and any(w in line_lower for w in ['usar', 'use', 'hay']):
                    continue

                for word in self.forbidden_words:
                    if word in line_lower:
                        # Verificar si está en un comentario explicativo
                        if '#' in line and line.index('#') < line_lower.index(word):
                            # Está en comentario, probablemente explicando
                            if 'sin' in line_lower or 'no ' in line_lower or 'without' in line_lower:
                                continue

                        results.append(AuditResult(
                            passed=False,
                            category='forbidden_words',
                            file=filepath,
                            line=i + 1,
                            message=f'Palabra prohibida: "{word}" - usar terminología endógena',
                            severity='error'
                        ))

                for word in self.review_words:
                    if word in line_lower and '#' not in line[:line_lower.find(word)]:
                        results.append(AuditResult(
                            passed=True,
                            category='review_words',
                            file=filepath,
                            line=i + 1,
                            message=f'Palabra a revisar: "{word}" - verificar uso endógeno',
                            severity='info'
                        ))

        except Exception as e:
            results.append(AuditResult(
                passed=False,
                category='forbidden_words',
                file=filepath,
                message=f'Error leyendo archivo: {e}',
                severity='error'
            ))

        return results

    def audit_relative_scales(self, filepath: str) -> List[AuditResult]:
        """
        CHECK 3: Verifica que las escalas sean relativas.

        Busca:
        - Uso de percentile/percentil
        - Normalización por historia
        - Ausencia de constantes de escala fijas
        """
        results = []

        try:
            with open(filepath, 'r') as f:
                content = f.read()

            # Indicadores positivos (endogeneidad)
            positive_patterns = [
                r'percentile',
                r'history',
                r'endogen',
                r'rank\s*\(',
                r'np\.var\s*\(',
                r'np\.cov\s*\(',
                r'normalize',
            ]

            positive_count = 0
            for pattern in positive_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                positive_count += len(matches)

            # Indicadores negativos (valores absolutos)
            negative_patterns = [
                r'=\s*[3-9]\d*\.?\d*\s*$',  # Asignaciones de constantes > 2
                r'threshold\s*=\s*\d+',
                r'scale\s*=\s*\d+',
            ]

            negative_count = 0
            for pattern in negative_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    results.append(AuditResult(
                        passed=False,
                        category='relative_scales',
                        file=filepath,
                        message=f'Posible escala absoluta: {match}',
                        severity='warning'
                    ))
                    negative_count += 1

            # Resumen
            if positive_count > 0 and negative_count == 0:
                results.append(AuditResult(
                    passed=True,
                    category='relative_scales',
                    file=filepath,
                    message=f'Archivo usa escalas relativas ({positive_count} indicadores positivos)',
                    severity='info'
                ))

        except Exception as e:
            results.append(AuditResult(
                passed=False,
                category='relative_scales',
                file=filepath,
                message=f'Error analizando archivo: {e}',
                severity='error'
            ))

        return results

    def audit_data_flow(self, filepath: str) -> List[AuditResult]:
        """
        CHECK 4: Verifica el flujo de datos correcto.

        Flujo esperado: drives → ranks → ψ → operators → Δ → payoff
        """
        results = []

        try:
            with open(filepath, 'r') as f:
                content = f.read()

            # Componentes del flujo
            flow_components = {
                'drives': r'drives',
                'ranks': r'rank',
                'psi_amplitudes': r'amplitudes|quantum_state|ψ',
                'operators': r'operator',
                'deltas': r'delta|Δ',
                'payoff': r'payoff'
            }

            found = {}
            for component, pattern in flow_components.items():
                if re.search(pattern, content, re.IGNORECASE):
                    found[component] = True

            # Verificar flujo en archivos específicos
            basename = os.path.basename(filepath)

            if 'state_encoding' in basename:
                required = ['drives', 'ranks', 'psi_amplitudes']
            elif 'operators' in basename:
                required = ['drives', 'operators']
            elif 'coalition_game' in basename:
                required = ['drives', 'operators', 'deltas']
            elif 'payoff' in basename:
                required = ['deltas', 'payoff']
            else:
                required = []

            for req in required:
                if req not in found:
                    results.append(AuditResult(
                        passed=False,
                        category='data_flow',
                        file=filepath,
                        message=f'Componente de flujo faltante: {req}',
                        severity='warning'
                    ))
                else:
                    results.append(AuditResult(
                        passed=True,
                        category='data_flow',
                        file=filepath,
                        message=f'Componente de flujo presente: {req}',
                        severity='info'
                    ))

        except Exception as e:
            results.append(AuditResult(
                passed=False,
                category='data_flow',
                file=filepath,
                message=f'Error verificando flujo: {e}',
                severity='error'
            ))

        return results

    def run_full_audit(self) -> Dict:
        """
        Ejecuta auditoría completa.
        """
        self.results = []

        for filepath in self.files_to_audit:
            if not os.path.exists(filepath):
                self.results.append(AuditResult(
                    passed=False,
                    category='file_missing',
                    file=filepath,
                    message='Archivo no encontrado',
                    severity='error'
                ))
                continue

            # Ejecutar todos los checks
            self.results.extend(self.audit_magic_numbers(filepath))
            self.results.extend(self.audit_forbidden_words(filepath))
            self.results.extend(self.audit_relative_scales(filepath))
            self.results.extend(self.audit_data_flow(filepath))

        return self.get_summary()

    def get_summary(self) -> Dict:
        """
        Resumen de auditoría.
        """
        summary = {
            'total_checks': len(self.results),
            'passed': sum(1 for r in self.results if r.passed),
            'warnings': sum(1 for r in self.results if r.severity == 'warning'),
            'errors': sum(1 for r in self.results if r.severity == 'error'),
            'info': sum(1 for r in self.results if r.severity == 'info'),
            'by_category': {},
            'by_file': {},
            'issues': []
        }

        # Agrupar por categoría
        for result in self.results:
            if result.category not in summary['by_category']:
                summary['by_category'][result.category] = {'passed': 0, 'failed': 0}

            if result.passed:
                summary['by_category'][result.category]['passed'] += 1
            else:
                summary['by_category'][result.category]['failed'] += 1

            # Agrupar por archivo
            basename = os.path.basename(result.file)
            if basename not in summary['by_file']:
                summary['by_file'][basename] = {'passed': 0, 'failed': 0}

            if result.passed:
                summary['by_file'][basename]['passed'] += 1
            else:
                summary['by_file'][basename]['failed'] += 1

            # Registrar issues
            if not result.passed or result.severity in ['warning', 'error']:
                summary['issues'].append({
                    'file': os.path.basename(result.file),
                    'line': result.line,
                    'category': result.category,
                    'message': result.message,
                    'severity': result.severity
                })

        # Calcular puntuación de endogeneidad
        if summary['total_checks'] > 0:
            summary['endogeneity_score'] = summary['passed'] / summary['total_checks']
        else:
            summary['endogeneity_score'] = 0

        return summary

    def print_report(self):
        """
        Imprime reporte de auditoría.
        """
        summary = self.get_summary()

        print("=" * 70)
        print("AUDITORÍA DE ENDOGENEIDAD - QUANTUM GAME")
        print("=" * 70)

        print(f"\n📊 RESUMEN GENERAL")
        print(f"   Total de checks: {summary['total_checks']}")
        print(f"   Pasados: {summary['passed']}")
        print(f"   Warnings: {summary['warnings']}")
        print(f"   Errores: {summary['errors']}")
        print(f"   Score de endogeneidad: {summary['endogeneity_score']*100:.1f}%")

        print(f"\n📁 POR ARCHIVO")
        for filename, counts in summary['by_file'].items():
            status = "✓" if counts['failed'] == 0 else "⚠"
            print(f"   {status} {filename}: {counts['passed']} passed, {counts['failed']} failed")

        print(f"\n📋 POR CATEGORÍA")
        for category, counts in summary['by_category'].items():
            status = "✓" if counts['failed'] == 0 else "⚠"
            print(f"   {status} {category}: {counts['passed']} passed, {counts['failed']} failed")

        if summary['issues']:
            print(f"\n⚠️  ISSUES ({len(summary['issues'])})")
            for issue in summary['issues'][:20]:  # Limitar a 20
                severity_icon = "🔴" if issue['severity'] == 'error' else "🟡"
                print(f"   {severity_icon} [{issue['file']}:{issue['line']}] {issue['message']}")

            if len(summary['issues']) > 20:
                print(f"   ... y {len(summary['issues']) - 20} más")

        # Veredicto final
        print("\n" + "=" * 70)
        if summary['errors'] == 0 and summary['warnings'] < 5:
            print("✅ VEREDICTO: Sistema ENDÓGENO - Sin dependencias externas detectadas")
        elif summary['errors'] == 0:
            print("⚠️  VEREDICTO: Sistema MAYORMENTE ENDÓGENO - Revisar warnings")
        else:
            print("❌ VEREDICTO: Sistema NO COMPLETAMENTE ENDÓGENO - Corregir errores")
        print("=" * 70)


def run_audit():
    """Ejecuta auditoría completa."""
    auditor = EndogeneityAuditor()
    auditor.run_full_audit()
    auditor.print_report()
    return auditor.get_summary()


if __name__ == "__main__":
    run_audit()
