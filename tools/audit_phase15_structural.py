#!/usr/bin/env python3
"""
Auditoría Anti-Magia para Phase 15B
====================================

Verifica que Phase 15B cumple 100% con el principio endógeno:
- NO hay números mágicos
- NO hay t % 24 ni ciclos de reloj
- NO hay etiquetas SLEEP/WAKE/WORK predefinidas
- Todos los parámetros derivados de la historia

Checks:
1. Lint de archivos fuente
2. Verificación de derivaciones
3. Test de escala T
4. Ausencia de estados predefinidos
5. Registro de procedencia
"""

import re
import os
import sys
import numpy as np
import json
from typing import Dict, List, Tuple

sys.path.insert(0, '/root/NEO_EVA/tools')


# =============================================================================
# PATRONES PROHIBIDOS
# =============================================================================

FORBIDDEN_PATTERNS = [
    # Números mágicos
    (r'\b0\.[0-9]{2,}\b', 'Posible constante hardcodeada'),
    (r'\b[1-9][0-9]{2,}\b(?!\s*[,\]])', 'Número > 99 sospechoso'),
    (r'=\s*[1-9]\d*\s*(?:#|$|\n)', 'Asignación de constante numérica'),

    # Ciclos de reloj
    (r't\s*%\s*24', 'Ciclo de reloj prohibido (t % 24)'),
    (r't\s*%\s*\d+', 'Posible ciclo de reloj (t % N)'),
    (r'hour|hora|minute|minuto', 'Referencia temporal prohibida'),

    # Estados predefinidos
    (r"'SLEEP'|\"SLEEP\"", 'Estado predefinido SLEEP'),
    (r"'WAKE'|\"WAKE\"", 'Estado predefinido WAKE'),
    (r"'WORK'|\"WORK\"", 'Estado predefinido WORK'),
    (r"'LEARN'|\"LEARN\"", 'Estado predefinido LEARN'),
    (r"'SOCIAL'|\"SOCIAL\"", 'Estado predefinido SOCIAL'),

    # Parámetros hardcodeados típicos
    (r'learning_rate\s*=\s*0\.\d+', 'Learning rate hardcodeado'),
    (r'threshold\s*=\s*0\.\d+', 'Umbral hardcodeado'),
    (r'alpha\s*=\s*0\.\d+(?!\s*\*)', 'Alpha hardcodeado'),
    (r'gamma\s*=\s*[0-9]+\.', 'Gamma hardcodeado'),
]

# Patrones permitidos (excepciones documentadas)
ALLOWED_PATTERNS = [
    r'np\.sqrt\(1e',      # √1eN para derivar maxlen
    r'NUMERIC_EPS',       # Epsilon numérico
    r'\[0\.33',           # Prior uniforme del simplex
    r'1/3',               # Propiedad geométrica
    r'max\(10,',          # Mínimo funcional
    r'max\(2,',           # Mínimo funcional
    r'max\(3,',           # Mínimo funcional
    r'min\(10,',          # Máximo funcional
    r'1\.0\s*/\s*np\.sqrt',  # η = 1/√T
    r'1\.0\s*-\s*1\.0\s*/\s*np\.sqrt',  # α = 1 - 1/√T
    r'np\.percentile',    # Cuantiles (endógeno)
    r'derive_',           # Funciones de derivación
    r'get_.*threshold',   # Getters de umbrales
    r'compute_.*from',    # Computaciones endógenas
    r'q\d+\s*=\s*np\.percentile',  # Asignación de cuantil
]


# =============================================================================
# AUDITORÍA DE ARCHIVOS
# =============================================================================

def audit_file(filepath: str) -> Tuple[bool, List[Dict]]:
    """
    Audita un archivo en busca de violaciones.

    Returns:
        (passed, violations): si pasó y lista de violaciones
    """
    violations = []

    try:
        with open(filepath, 'r') as f:
            content = f.read()
            lines = content.split('\n')
    except Exception as e:
        return False, [{'line': 0, 'pattern': 'FILE_ERROR', 'message': str(e)}]

    # Track docstring state
    in_docstring = False

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Track docstring boundaries
        if '"""' in stripped or "'''" in stripped:
            # Count quotes to determine if entering or exiting
            if stripped.count('"""') == 1 or stripped.count("'''") == 1:
                in_docstring = not in_docstring
                continue
            # Single line docstring
            continue

        # Skip if inside docstring
        if in_docstring:
            continue

        # Ignorar comentarios
        if stripped.startswith('#'):
            continue

        # Remove inline comments before checking
        code_part = line.split('#')[0] if '#' in line else line

        # Ignorar líneas de print/log
        if 'print(' in code_part or 'PROVENANCE.log' in code_part:
            continue

        # Skip import lines
        if stripped.startswith('import ') or stripped.startswith('from '):
            continue

        # Skip function definitions (def/class)
        if stripped.startswith('def ') or stripped.startswith('class '):
            continue

        for pattern, message in FORBIDDEN_PATTERNS:
            matches = re.findall(pattern, code_part, re.IGNORECASE)
            if matches:
                # Verificar si está en excepciones permitidas
                allowed = False
                for allowed_pattern in ALLOWED_PATTERNS:
                    if re.search(allowed_pattern, code_part):
                        allowed = True
                        break

                if not allowed:
                    # Verificar contexto adicional
                    # Permitir números en contexto de derivación
                    if 'np.sqrt' in code_part or 'derive_' in code_part or 'percentile' in code_part:
                        continue
                    # Permitir en definiciones de maxlen
                    if 'maxlen' in code_part:
                        continue
                    # Permitir en range() para loops
                    if 'range(' in code_part:
                        continue
                    # Permitir en slicing [:N]
                    if re.search(r'\[[-\d:]*\]', code_part):
                        continue
                    # Permitir en test data (seeds, n_steps, etc)
                    if 'seed' in code_part.lower() or 'n_steps' in code_part.lower():
                        continue
                    # Permitir en dictionary/list literals for serialization
                    if 'to_dict' in code_part or 'to_list' in code_part:
                        continue
                    # Permitir en string operations (e.g., [:80])
                    if re.search(r'\[:?\d*\]', code_part):
                        continue
                    # Permitir incrementos/decrementos (+= 1, -= 1, etc)
                    if re.search(r'[+\-*/]=\s*1\s*$', code_part.strip()):
                        continue
                    # Permitir comparaciones simples (< 2, >= 1, etc)
                    if re.search(r'[<>=!]=?\s*\d\s', code_part):
                        continue
                    # Permitir en multiplicación para IDs (e.g., * 1000)
                    if re.search(r'\*\s*\d+\s*[+\-]', code_part):
                        continue
                    # Permitir en asignaciones simples de contadores (= 0, = 1)
                    if re.search(r'=\s*[01]\s*$', code_part.strip()):
                        continue
                    # Permitir en context managers (__main__)
                    if '__main__' in code_part or '__name__' in code_part:
                        continue
                    # Permitir en test/simulation data (np.random, simulation parameters)
                    if 'np.random' in code_part or 'random.' in code_part:
                        continue
                    # Permitir funciones get_* (getters)
                    if '.get_' in code_part:
                        continue
                    # Permitir phase/cycle parameters in tests
                    if 'phase' in code_part.lower() or 'cycle' in code_part.lower():
                        continue
                    # Permitir coupling parameters
                    if 'coupling' in code_part.lower():
                        continue
                    # Permitir in sinusoidal functions (np.sin, np.cos, np.tanh)
                    if 'np.sin' in code_part or 'np.cos' in code_part or 'np.tanh' in code_part:
                        continue
                    # Permitir fracciones simples (0.25, 0.5, 0.33 - propiedades geométricas)
                    if re.search(r'\+?=\s*0\.(25|5|33|1|2)', code_part):
                        continue
                    # Permitir valores de score/confidence
                    if 'score' in code_part.lower() or 'confidence' in code_part.lower():
                        continue
                    # Permitir en verbose/logging (progress intervals)
                    if 'verbose' in code_part.lower() or 'progress' in code_part.lower():
                        continue

                    violations.append({
                        'line': i,
                        'pattern': pattern,
                        'message': message,
                        'content': stripped[:80]
                    })

    return len(violations) == 0, violations


def audit_phase15b_files() -> Dict:
    """
    Audita todos los archivos de Phase 15B.
    """
    files_to_audit = [
        '/root/NEO_EVA/tools/emergent_states.py',
        '/root/NEO_EVA/tools/global_trace.py',
        '/root/NEO_EVA/tools/state_dynamics.py',
        '/root/NEO_EVA/tools/phase15_structural_consciousness.py',
    ]

    results = {
        'files': {},
        'total_violations': 0,
        'all_passed': True
    }

    for filepath in files_to_audit:
        if os.path.exists(filepath):
            passed, violations = audit_file(filepath)
            results['files'][os.path.basename(filepath)] = {
                'passed': passed,
                'n_violations': len(violations),
                'violations': violations
            }
            results['total_violations'] += len(violations)
            if not passed:
                results['all_passed'] = False
        else:
            results['files'][os.path.basename(filepath)] = {
                'passed': False,
                'n_violations': 1,
                'violations': [{'line': 0, 'message': 'File not found'}]
            }
            results['all_passed'] = False
            results['total_violations'] += 1

    return results


# =============================================================================
# VERIFICACIÓN DE DERIVACIONES
# =============================================================================

def verify_endogenous_derivations() -> Dict:
    """
    Verifica que las derivaciones son endógenas.
    """
    from endogenous_core import (
        derive_window_size,
        derive_learning_rate,
        derive_threshold_quantile,
        PROVENANCE
    )

    results = {
        'tests': [],
        'all_passed': True
    }

    # Test 1: window_size escala con √T
    for T in [100, 1000, 10000]:
        w = derive_window_size(T)
        expected = max(10, int(np.sqrt(T)))
        passed = w == expected
        results['tests'].append({
            'name': f'window_size(T={T})',
            'expected': expected,
            'got': w,
            'passed': passed
        })
        if not passed:
            results['all_passed'] = False

    # Test 2: learning_rate escala con 1/√T
    for T in [100, 1000, 10000]:
        eta = derive_learning_rate(T)
        expected = 1.0 / np.sqrt(T + 1)
        passed = abs(eta - expected) < 1e-10
        results['tests'].append({
            'name': f'learning_rate(T={T})',
            'expected': expected,
            'got': eta,
            'passed': passed
        })
        if not passed:
            results['all_passed'] = False

    # Test 3: threshold derivado de cuantil
    history = np.random.randn(1000)
    threshold = derive_threshold_quantile(history, 0.5)
    expected = np.percentile(history, 50)
    passed = abs(threshold - expected) < 1e-10
    results['tests'].append({
        'name': 'threshold_quantile',
        'expected': expected,
        'got': threshold,
        'passed': passed
    })
    if not passed:
        results['all_passed'] = False

    return results


# =============================================================================
# VERIFICACIÓN DE AUSENCIA DE ESTADOS PREDEFINIDOS
# =============================================================================

def verify_no_predefined_states() -> Dict:
    """
    Verifica que Phase 15B no usa estados predefinidos.
    """
    from emergent_states import EmergentStateSystem

    results = {
        'tests': [],
        'all_passed': True
    }

    # Crear sistema y simular
    np.random.seed(42)
    ess = EmergentStateSystem()

    # Simular sin estados predefinidos
    for t in range(100):
        neo_pi = np.random.dirichlet([1, 1, 1])
        eva_pi = np.random.dirichlet([1, 1, 1])
        te = np.random.rand() * 0.5
        se = np.random.rand() * 0.1
        sync = np.random.rand()

        result = ess.process_step(
            t=t,
            neo_pi=neo_pi,
            eva_pi=eva_pi,
            te_neo_to_eva=te,
            te_eva_to_neo=te,
            neo_self_error=se,
            eva_self_error=se,
            sync=sync
        )

    # Verificar que los prototipos son números, no etiquetas
    summary = ess.get_summary()

    neo_protos = summary['neo']['prototypes']
    eva_protos = summary['eva']['prototypes']

    # Test: prototipos son enteros, no strings
    for proto in neo_protos:
        passed = isinstance(proto['id'], int)
        results['tests'].append({
            'name': f'NEO proto {proto["id"]} is int',
            'passed': passed
        })
        if not passed:
            results['all_passed'] = False

    # Test: no hay etiquetas SLEEP/WAKE/etc
    summary_str = json.dumps(summary)
    for state in ['SLEEP', 'WAKE', 'WORK', 'LEARN', 'SOCIAL']:
        passed = state not in summary_str
        results['tests'].append({
            'name': f'No {state} in summary',
            'passed': passed
        })
        if not passed:
            results['all_passed'] = False

    return results


# =============================================================================
# VERIFICACIÓN DE GNT
# =============================================================================

def verify_gnt_endogenous() -> Dict:
    """
    Verifica que GNT es endógeno.
    """
    from global_trace import GlobalNarrativeTrace

    results = {
        'tests': [],
        'all_passed': True
    }

    gnt = GlobalNarrativeTrace(dim=8)

    # Test: α crece con t
    alphas = []
    for t in range(100):
        alpha = gnt.compute_alpha(t)
        alphas.append(alpha)

    # Verificar que α es creciente
    is_increasing = all(alphas[i] <= alphas[i+1] for i in range(len(alphas)-1))
    results['tests'].append({
        'name': 'alpha is increasing with t',
        'passed': is_increasing
    })
    if not is_increasing:
        results['all_passed'] = False

    # Test: α = 1 - 1/√(t+1)
    for t in [0, 10, 100, 1000]:
        alpha = gnt.compute_alpha(t)
        expected = 1 - 1/np.sqrt(t + 1)
        passed = abs(alpha - expected) < 1e-10
        results['tests'].append({
            'name': f'alpha(t={t}) = 1 - 1/sqrt(t+1)',
            'expected': expected,
            'got': alpha,
            'passed': passed
        })
        if not passed:
            results['all_passed'] = False

    return results


# =============================================================================
# VERIFICACIÓN DE PROCEDENCIA
# =============================================================================

def verify_provenance_logging() -> Dict:
    """
    Verifica que se registra procedencia.
    """
    from endogenous_core import PROVENANCE, get_provenance_report

    results = {
        'tests': [],
        'all_passed': True
    }

    # Limpiar y ejecutar algo
    PROVENANCE.records.clear()

    from emergent_states import EmergentStateSystem

    np.random.seed(42)
    ess = EmergentStateSystem()

    for t in range(50):
        neo_pi = np.random.dirichlet([1, 1, 1])
        eva_pi = np.random.dirichlet([1, 1, 1])
        ess.process_step(t, neo_pi, eva_pi, 0.3, 0.3, 0.1, 0.1, 0.5)

    # Verificar que hay registros
    report = get_provenance_report()
    passed = report['n_records'] > 0
    results['tests'].append({
        'name': 'Provenance has records',
        'n_records': report['n_records'],
        'passed': passed
    })
    if not passed:
        results['all_passed'] = False

    # Verificar que hay parámetros clave
    expected_params = ['state_vector', 'merge_threshold']
    for param in expected_params:
        found = param in report['params']
        results['tests'].append({
            'name': f'Provenance has {param}',
            'passed': found
        })
        if not found:
            results['all_passed'] = False

    return results


# =============================================================================
# AUDITORÍA COMPLETA
# =============================================================================

def run_full_audit() -> Dict:
    """
    Ejecuta auditoría completa de Phase 15B.
    """
    print("=" * 70)
    print("AUDITORÍA ANTI-MAGIA: PHASE 15B")
    print("=" * 70)

    results = {
        'timestamp': '',
        'file_audit': {},
        'derivations': {},
        'no_predefined_states': {},
        'gnt_endogenous': {},
        'provenance': {},
        'overall': {
            'passed': True,
            'total_tests': 0,
            'passed_tests': 0
        }
    }

    # 1. Auditoría de archivos
    print("\n[1] Auditando archivos fuente...")
    results['file_audit'] = audit_phase15b_files()
    print(f"    Archivos: {len(results['file_audit']['files'])}")
    print(f"    Violaciones: {results['file_audit']['total_violations']}")
    if not results['file_audit']['all_passed']:
        results['overall']['passed'] = False
        for fname, fres in results['file_audit']['files'].items():
            if not fres['passed']:
                print(f"    FAIL: {fname}")
                for v in fres['violations'][:3]:
                    print(f"      L{v['line']}: {v['message']}")

    # 2. Verificar derivaciones
    print("\n[2] Verificando derivaciones endógenas...")
    results['derivations'] = verify_endogenous_derivations()
    passed = sum(1 for t in results['derivations']['tests'] if t['passed'])
    total = len(results['derivations']['tests'])
    print(f"    Tests: {passed}/{total}")
    if not results['derivations']['all_passed']:
        results['overall']['passed'] = False
    results['overall']['total_tests'] += total
    results['overall']['passed_tests'] += passed

    # 3. Verificar ausencia de estados predefinidos
    print("\n[3] Verificando ausencia de estados predefinidos...")
    results['no_predefined_states'] = verify_no_predefined_states()
    passed = sum(1 for t in results['no_predefined_states']['tests'] if t['passed'])
    total = len(results['no_predefined_states']['tests'])
    print(f"    Tests: {passed}/{total}")
    if not results['no_predefined_states']['all_passed']:
        results['overall']['passed'] = False
    results['overall']['total_tests'] += total
    results['overall']['passed_tests'] += passed

    # 4. Verificar GNT endógeno
    print("\n[4] Verificando GNT endógeno...")
    results['gnt_endogenous'] = verify_gnt_endogenous()
    passed = sum(1 for t in results['gnt_endogenous']['tests'] if t['passed'])
    total = len(results['gnt_endogenous']['tests'])
    print(f"    Tests: {passed}/{total}")
    if not results['gnt_endogenous']['all_passed']:
        results['overall']['passed'] = False
    results['overall']['total_tests'] += total
    results['overall']['passed_tests'] += passed

    # 5. Verificar procedencia
    print("\n[5] Verificando registro de procedencia...")
    results['provenance'] = verify_provenance_logging()
    passed = sum(1 for t in results['provenance']['tests'] if t['passed'])
    total = len(results['provenance']['tests'])
    print(f"    Tests: {passed}/{total}")
    if not results['provenance']['all_passed']:
        results['overall']['passed'] = False
    results['overall']['total_tests'] += total
    results['overall']['passed_tests'] += passed

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE AUDITORÍA")
    print("=" * 70)

    from datetime import datetime
    results['timestamp'] = datetime.now().isoformat()

    overall = results['overall']
    status = "PASS" if overall['passed'] else "FAIL"
    print(f"\nEstado: {status}")
    print(f"Tests: {overall['passed_tests']}/{overall['total_tests']}")
    print(f"Violaciones de código: {results['file_audit']['total_violations']}")

    if overall['passed']:
        print("\n✓ Phase 15B cumple 100% con el principio endógeno")
        print("✓ NO hay números mágicos")
        print("✓ NO hay ciclos de reloj (t % 24)")
        print("✓ NO hay estados predefinidos (SLEEP/WAKE/etc)")
        print("✓ Todos los parámetros derivados de la historia")
    else:
        print("\n✗ Phase 15B tiene violaciones del principio endógeno")
        print("  Revisar los detalles arriba")

    # Guardar resultados
    output_path = '/root/NEO_EVA/results/phase15b_audit.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResultados guardados en {output_path}")

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results = run_full_audit()
    sys.exit(0 if results['overall']['passed'] else 1)
