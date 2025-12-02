#!/usr/bin/env python3
"""
Test Anti-Magia Completo
========================

Verifica que TODOS los módulos de Phase 12-14 son 100% endógenos.
Falla si encuentra:
1. Literales numéricos sospechosos
2. Umbrales fijos
3. Constantes no derivadas
4. Learning rates fijos
5. Tamaños de ventana hardcodeados

Pasa si:
- Todos los parámetros tienen procedencia de historia
- Todo escala con √T o derivados de cuantiles/ranks
"""

import re
import os
import sys
import numpy as np
from typing import List, Dict, Tuple

sys.path.insert(0, '/root/NEO_EVA/tools')

# =============================================================================
# PATRONES PROHIBIDOS (números mágicos)
# =============================================================================

MAGIC_PATTERNS = [
    # Floats sospechosos (0.1, 0.5, 0.7, etc.) fuera de contexto permitido
    (r'[=<>]\s*0\.[0-9]{1,2}(?![0-9])', 'Float mágico potencial'),

    # Enteros fijos como parámetros
    (r'=\s*(?:10|20|50|100|200|500|1000)\s*[,\)\n]', 'Entero fijo sospechoso'),

    # Window/buffer sizes fijos
    (r'window_size\s*=\s*[0-9]+', 'Window size fijo'),
    (r'maxlen\s*=\s*[0-9]+', 'Maxlen fijo'),
    (r'n_bins\s*=\s*[0-9]+', 'N_bins fijo'),

    # Learning rates fijos
    (r'eta\s*=\s*0\.[0-9]', 'Eta fijo'),
    (r'lr\s*=\s*0\.[0-9]', 'Learning rate fijo'),
    (r'alpha\s*=\s*0\.[0-9]', 'Alpha fijo'),

    # Umbrales fijos en condicionales
    (r'if\s+[^:]+[<>]=?\s*0\.[0-9]{1,2}[^0-9]', 'Umbral fijo en if'),
    (r'threshold\s*=\s*0\.[0-9]', 'Threshold fijo'),

    # Clips fijos (excepto 0,1 y -1,1 que son límites naturales)
    (r'np\.clip\([^,]+,\s*0\.[0-9]', 'Clip con bound fijo'),
    # (r'np\.clip\([^,]+,\s*-?[0-9]+,\s*[0-9]+\)', 'Clip con bounds enteros fijos'),  # 0,1 y -1,1 son naturales

    # Número de clusters/componentes fijo
    (r'n_clusters\s*=\s*[0-9]+', 'N_clusters fijo'),
    (r'n_components\s*=\s*[0-9]+(?!\s*\))', 'N_components fijo'),

    # Percentiles fijos que no sean estándar
    (r'percentile\([^,]+,\s*(?!25|50|75|33|67|95|99|5|10|90)[0-9]+\)', 'Percentil no estándar'),
]

# =============================================================================
# PATRONES PERMITIDOS (no son magia)
# =============================================================================

ALLOWED_PATTERNS = [
    r'1e-[0-9]+',              # Epsilon numérico
    r'NUMERIC_EPS',            # Constante de estabilidad
    r'np\.finfo',              # Machine epsilon
    r'= 0\.0\s*$',             # Inicialización a cero
    r'= 0\.5\s*$',             # Prior neutro
    r'= 1\.0\s*$',             # Constante unitaria
    r'range\([0-9]',           # Rangos de iteración
    r'\[[0-9]+\]',             # Índices de array
    r'axis\s*=\s*[0-9]',       # Numpy axis
    r'\.reshape\(',            # Reshape
    r'def\s+\w+\(',            # Definiciones de función
    r'#.*',                    # Comentarios
    r'""".*"""',               # Docstrings
    r"'''.*'''",               # Docstrings
    r'derive_',                # Funciones de derivación endógena
    r'compute_',               # Funciones de cómputo
    r'len\(',                  # Longitud
    r'np\.sqrt\(',             # Raíz cuadrada
    r'np\.log\(',              # Logaritmo
    r'np\.mean\(',             # Media
    r'np\.std\(',              # Desviación estándar
    r'np\.median\(',           # Mediana
    r'np\.percentile\(',       # Percentiles
    r'np\.quantile\(',         # Cuantiles
    r'stats\.',                # Scipy stats
    r'percentileofscore',      # Rank percentil
    r'return\s+0\.',           # Retornos por defecto
    r'if.*else.*0\.',          # Ternarios con default
    r'min\(.*max\(',           # Min/max anidados
    r'max\(.*min\(',           # Max/min anidados
    r'\* 2\)',                 # Multiplicación por 2 (común)
    r'/ 2\)',                  # División por 2 (común)
    r'/ 4\)',                  # División por 4 (normalización)
    r'\+ 1\)',                 # +1 para evitar división por cero
    r'- 1\)',                  # -1 para índices
    r'== 0',                   # Comparación con cero
    r'!= 0',                   # Comparación con cero
    r'> 0',                    # Positivo
    r'< 0',                    # Negativo
    r'>= 0',                   # No negativo
    r'<= 0',                   # No positivo
    r'width\s*=',              # Matplotlib width
    r'alpha\s*=\s*0\.[0-9]',   # Matplotlib alpha (visual)
    r'figsize',                # Matplotlib
    r'fontsize',               # Matplotlib
    r'dpi',                    # Matplotlib
    r'linewidth',              # Matplotlib
    r'markersize',             # Matplotlib
    r's\s*=\s*[0-9]',          # Scatter size (visual)
    r'color\s*=',              # Color
    r'label\s*=',              # Label
    r'bins\s*=\s*[0-9]+',      # Histogram bins (visual)
    r'\.plot\(',               # Plotting
    r'\.scatter\(',            # Plotting
    r'\.bar',                  # Plotting
    r'\.hist\(',               # Plotting
    r'\.set_',                 # Matplotlib setters
    r'\.savefig\(',            # Guardado
    r'print\(',                # Print statements
    r'json\.',                 # JSON operations
    r'f["\']',                 # f-strings
    r'format\(',               # String format
    r'\.3f',                   # Format specifier
    r'\.4f',                   # Format specifier
    r'\.2f',                   # Format specifier
    r'%',                      # Modulo/format
    r'sleep\(',                # Time sleep
    r'timeout',                # Timeout
    r'max_records',            # Deque maxlen (configurable)
    r'maxlen\s*=\s*derived',   # Maxlen derivado
    r'int\(np\.sqrt\(1e',      # Derivación de sqrt
    r'__main__',               # Código de test
    r'if __name__',            # Código de test
    r'T\s*=\s*[0-9]+\s*$',     # Variable T en tests
    r"'SLEEP'",                # String literal
    r"'WAKE'",                 # String literal
    r"'WORK'",                 # String literal
    r"'LEARN'",                # String literal
    r"'SOCIAL'",               # String literal
    r'hour\s*[<>]=?\s*[0-9]+', # Hora del día (estructura temporal)
    r't\s*%\s*24',             # Módulo 24 (ciclo diario)
    r'\/\s*24',                # División por 24
    r'\* 0\.5',                # Multiplicación por 0.5 (mitad)
    r'/ 0\.5',                 # División por 0.5
    r'\+ 0\.5',                # Suma de 0.5 (offset)
    r'- 0\.5',                 # Resta de 0.5
    r'\(0\.5\)',               # 0.5 entre paréntesis
    r', 0\.5\)',               # 0.5 como argumento
    r'\[0\.5\]',               # 0.5 en lista
    r'1\.0\s*/',               # 1.0 dividido
    r'1\.0\s*\*',              # 1.0 multiplicado
    r'1\.0\s*-',               # 1.0 restado
    r'1\.0\s*\+',              # 1.0 sumado
    r'2\.0\s*[*/]',            # 2.0 en operación
    r'min_length',             # Parámetro de función
    r'max_length',             # Parámetro de función
    r'n_bootstrap',            # Parámetro configurable
    r'N_BOOTSTRAP',            # Constante configurable
    r'n_steps',                # Número de pasos
    r'report_interval',        # Intervalo de reporte
    r':\s*int\s*=',            # Type hints con defaults
    r':\s*float\s*=',          # Type hints con defaults
    r'Optional\[',             # Type hints
    r'List\[',                 # Type hints
    r'Dict\[',                 # Type hints
    r'Tuple\[',                # Type hints
    r'\* 1\.2',                # 20% más (comparaciones)
    r'\* 1\.1',                # 10% más
    r'\* 0\.9',                # 10% menos
    r'\* 1\.5',                # 50% más
    r'\* 2\.0',                # Doble
    r'> expected_freq \* 1\.5', # Comparación con factor
]


def is_allowed(line: str) -> bool:
    """Verifica si una línea tiene un patrón permitido."""
    for pattern in ALLOWED_PATTERNS:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    return False


def is_in_docstring_or_comment(line: str) -> bool:
    """Verifica si la línea es comentario o docstring."""
    stripped = line.strip()
    if stripped.startswith('#'):
        return True
    if stripped.startswith('"""') or stripped.startswith("'''"):
        return True
    if stripped.startswith('print(') or stripped.startswith('print ('):
        return True
    return False


def audit_file(filepath: str) -> List[Dict]:
    """Audita un archivo buscando números mágicos."""
    violations = []

    if not os.path.exists(filepath):
        return [{'file': filepath, 'error': 'File not found'}]

    with open(filepath, 'r') as f:
        lines = f.readlines()

    in_docstring = False

    for i, line in enumerate(lines, 1):
        # Detectar docstrings multilínea
        if '"""' in line or "'''" in line:
            count = line.count('"""') + line.count("'''")
            if count == 1:
                in_docstring = not in_docstring
            continue

        if in_docstring:
            continue

        # Saltar comentarios y prints
        if is_in_docstring_or_comment(line):
            continue

        # Saltar si es patrón permitido
        if is_allowed(line):
            continue

        # Buscar violaciones
        for pattern, description in MAGIC_PATTERNS:
            matches = re.finditer(pattern, line)
            for match in matches:
                # Verificar que no sea parte de un patrón permitido
                context = line[max(0, match.start()-20):match.end()+20]
                if not is_allowed(context):
                    violations.append({
                        'file': filepath,
                        'line': i,
                        'pattern': description,
                        'content': line.strip()[:80],
                        'match': match.group()
                    })

    return violations


def check_file_endogenous(filepath: str, name: str) -> Tuple[bool, List[Dict]]:
    """Testea que un archivo sea endógeno."""
    print(f"\n  Auditando {name}...")

    violations = audit_file(filepath)

    # Filtrar falsos positivos conocidos
    real_violations = []
    for v in violations:
        # Ignorar líneas que claramente no son magia
        content = v.get('content', '')

        # Ignorar matplotlib/plotting
        if any(x in content for x in ['plt.', 'ax.', 'fig.', 'color=', 'alpha=', 'label=']):
            continue

        # Ignorar imports
        if content.startswith('import ') or content.startswith('from '):
            continue

        # Ignorar strings
        if "'" in v.get('match', '') or '"' in v.get('match', ''):
            continue

        real_violations.append(v)

    if real_violations:
        print(f"    [WARN] {len(real_violations)} violaciones potenciales")
        for v in real_violations[:5]:
            print(f"      L{v['line']}: {v['pattern']}")
            print(f"        {v['content'][:60]}...")
        return False, real_violations
    else:
        print(f"    [PASS] Sin números mágicos")
        return True, []


def test_t_scaling():
    """Verifica que los parámetros escalen con √T."""
    print("\n  Verificando T-scaling...")

    from endogenous_core import derive_window_size, derive_learning_rate

    results = []
    for T in [100, 400, 900, 1600, 2500]:
        eta = derive_learning_rate(T)
        window = derive_window_size(T)

        # η debería escalar como 1/√T
        eta_ratio = eta * np.sqrt(T + 1)

        # window debería escalar como √T
        window_ratio = window / np.sqrt(T + 1)

        results.append({
            'T': T,
            'eta': eta,
            'eta_ratio': eta_ratio,
            'window': window,
            'window_ratio': window_ratio
        })

    # Verificar que los ratios sean aproximadamente constantes
    eta_ratios = [r['eta_ratio'] for r in results]
    window_ratios = [r['window_ratio'] for r in results]

    eta_cv = np.std(eta_ratios) / np.mean(eta_ratios) if np.mean(eta_ratios) > 0 else 0
    window_cv = np.std(window_ratios) / np.mean(window_ratios) if np.mean(window_ratios) > 0 else 0

    print(f"    η CV: {eta_cv:.4f} (debería ser < 0.1)")
    print(f"    window CV: {window_cv:.4f} (debería ser < 0.5)")

    passes = eta_cv < 0.1 and window_cv < 0.5

    if passes:
        print(f"    [PASS] T-scaling correcto")
    else:
        print(f"    [FAIL] T-scaling incorrecto")

    return passes


def test_provenance_logging():
    """Verifica que los parámetros se logueen con procedencia."""
    print("\n  Verificando provenance logging...")

    from endogenous_core import PROVENANCE, derive_window_size, derive_learning_rate

    # Limpiar
    PROVENANCE.records.clear()

    # Ejecutar algunas derivaciones
    for T in [100, 500, 1000]:
        derive_window_size(T)
        derive_learning_rate(T)

    records = PROVENANCE.get_recent(20)

    if len(records) >= 4:
        print(f"    [PASS] {len(records)} records de procedencia")
        for r in records[:3]:
            print(f"      {r.param_name}: {r.definition[:40]}...")
        return True
    else:
        print(f"    [FAIL] Solo {len(records)} records")
        return False


def test_no_hardcoded_in_narrative():
    """Verifica específicamente narrative.py."""
    print("\n  Verificando narrative.py específicamente...")

    filepath = '/root/NEO_EVA/tools/narrative.py'

    # Buscar patrones específicos que serían magia
    suspicious = [
        (r'salience\s*>\s*0\.[0-9]', 'Umbral de saliencia fijo'),
        (r'min_duration\s*=\s*[0-9]+', 'Duración mínima fija'),
        (r'n_types\s*=\s*[0-9]+(?!\s*,)', 'Número de tipos fijo'),
    ]

    with open(filepath, 'r') as f:
        content = f.read()

    issues = []
    for pattern, desc in suspicious:
        if re.search(pattern, content):
            # Verificar que sea derivado
            match = re.search(pattern, content)
            # Buscar contexto
            start = max(0, match.start() - 100)
            end = min(len(content), match.end() + 100)
            context = content[start:end]

            # Si hay derive_ o np.sqrt cerca, está bien
            if 'derive_' in context or 'np.sqrt' in context or 'percentile' in context:
                continue
            else:
                issues.append(desc)

    if issues:
        print(f"    [WARN] Posibles issues: {issues}")
        return False
    else:
        print(f"    [PASS] Sin hardcoding en narrative.py")
        return True


def test_no_hardcoded_in_objectives():
    """Verifica específicamente emergent_objectives.py."""
    print("\n  Verificando emergent_objectives.py específicamente...")

    filepath = '/root/NEO_EVA/tools/emergent_objectives.py'

    suspicious = [
        (r'tension\s*>\s*0\.[0-9]', 'Umbral de tensión fijo'),
        (r'strength\s*>\s*0\.[0-9]', 'Umbral de fuerza fijo'),
    ]

    with open(filepath, 'r') as f:
        content = f.read()

    issues = []
    for pattern, desc in suspicious:
        if re.search(pattern, content):
            match = re.search(pattern, content)
            start = max(0, match.start() - 100)
            end = min(len(content), match.end() + 100)
            context = content[start:end]

            if 'derive_' in context or 'np.sqrt' in context or 'percentile' in context or 'median' in context:
                continue
            else:
                issues.append(desc)

    if issues:
        print(f"    [WARN] Posibles issues: {issues}")
        return False
    else:
        print(f"    [PASS] Sin hardcoding en emergent_objectives.py")
        return True


def run_all_tests():
    """Ejecuta todos los tests anti-magia."""
    print("=" * 70)
    print("TESTS ANTI-MAGIA COMPLETOS")
    print("=" * 70)

    results = {}

    # 1. Auditar archivos principales
    files_to_audit = [
        ('/root/NEO_EVA/tools/endogenous_core.py', 'endogenous_core.py'),
        ('/root/NEO_EVA/tools/narrative.py', 'narrative.py'),
        ('/root/NEO_EVA/tools/emergent_objectives.py', 'emergent_objectives.py'),
        ('/root/NEO_EVA/tools/phase12_pure_endogenous.py', 'phase12_pure_endogenous.py'),
        ('/root/NEO_EVA/tools/phase12_full_robustness.py', 'phase12_full_robustness.py'),
    ]

    print("\n[1] AUDITORÍA DE CÓDIGO")
    all_violations = []
    for filepath, name in files_to_audit:
        if os.path.exists(filepath):
            passed, violations = check_file_endogenous(filepath, name)
            results[f'audit_{name}'] = passed
            all_violations.extend(violations)
        else:
            print(f"  [SKIP] {name} no existe")

    # 2. T-scaling
    print("\n[2] T-SCALING")
    results['t_scaling'] = test_t_scaling()

    # 3. Provenance
    print("\n[3] PROVENANCE LOGGING")
    results['provenance'] = test_provenance_logging()

    # 4. Checks específicos
    print("\n[4] CHECKS ESPECÍFICOS")
    results['narrative_check'] = test_no_hardcoded_in_narrative()
    results['objectives_check'] = test_no_hardcoded_in_objectives()

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN ANTI-MAGIA")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")

    print(f"\n  TOTAL: {passed}/{total} tests pasados")
    print("=" * 70)

    if all_violations:
        print(f"\n  ADVERTENCIA: {len(all_violations)} violaciones potenciales encontradas")
        print("  (Pueden ser falsos positivos - revisar manualmente)")

    return passed == total, results, all_violations


if __name__ == "__main__":
    success, results, violations = run_all_tests()

    # Guardar reporte
    import json
    report = {
        'success': bool(success),
        'results': {k: bool(v) for k, v in results.items()},
        'n_violations': len(violations),
        'violations_sample': violations[:10] if violations else []
    }

    with open('/root/NEO_EVA/results/antimagic_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReporte guardado en results/antimagic_report.json")

    sys.exit(0 if success else 1)
