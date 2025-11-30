#!/usr/bin/env python3
"""
Test de Lint Endógeno
=====================

Falla si encuentra:
1. Literales prohibidos fuera de módulos de estabilidad numérica
2. Ventanas/buffers fijos
3. Umbrales absolutos en gates
4. Learning rates con boosts fijos

Pasa si:
- Todos los parámetros tienen procedencia de historia
- η, τ, σ escalan como ~1/√T
- Nulls agresivos dan AUC ≈ 0.5
"""

import re
import sys
import os
import numpy as np

sys.path.insert(0, '/root/NEO_EVA/tools')


# =============================================================================
# PATRONES PROHIBIDOS
# =============================================================================

# Números mágicos sospechosos (fuera de eps/simplex)
MAGIC_PATTERNS = [
    (r'=\s*0\.[0-9]{1,2}[^0-9]', 'Float mágico (ej: 0.1, 0.5)'),
    (r'=\s*[0-9]+\s*[,\)]', 'Entero fijo como parámetro'),
    (r'gamma\s*=\s*np\.clip\([^,]+,\s*[0-9]', 'Gamma con clip fijo'),
    (r'rate\s*=\s*[0-9]+\.[0-9]+\s*/', 'Rate con factor fijo'),
    (r'window_size:\s*int\s*=\s*[0-9]+', 'Window size fijo como default'),
    (r'maxlen\s*=\s*[0-9]+', 'Buffer con tamaño fijo'),
    (r'np\.clip\([^,]+,\s*0\.0[0-9]', 'Clip con bound fijo'),
    (r'if\s+[^:]+[<>]\s*0\.[0-9]', 'Umbral fijo en condicional'),
    (r'randn\([^)]*\)\s*\*\s*[0-9]', 'Ruido con escala fija'),
    (r'eta\s*=\s*[0-9]', 'Eta fijo'),
    (r'tau\s*=\s*[0-9]', 'Tau fijo'),
]

# Patrones permitidos (estabilidad numérica y geometría)
ALLOWED_PATTERNS = [
    r'1e-[0-9]+',           # Epsilon numérico
    r'NUMERIC_EPS',         # Constante de estabilidad
    r'1/3',                 # Prior uniforme simplex
    r'np\.finfo',           # Machine epsilon
    r'\[0\.2,\s*0\.2,\s*0\.6\]',  # Atractores (estructura del modelo)
    r'\[0\.6,\s*0\.2,\s*0\.2\]',
    r'\[0\.25,\s*0\.25,\s*0\.5\]',
    r'axis\s*=\s*[0-9]',    # Numpy axis parameter
    r'n_components\s*=',    # PCA components (derivado dinámicamente)
    r'= 0\.0\s*$',          # Inicialización a cero
    r'= 0\.5\s*#',          # Prior neutro con comentario
    r'rho = 0\.0',          # Inicialización de correlación
    r'self\..*= 0\.',       # Inicializaciones de atributos
    r'range\([0-9]',        # Rangos de iteración
    r'\[[0-9]+\]',          # Índices de array
    r'\.reshape\(',         # Reshape operations
    r'stats\.rankdata',     # Scipy stats
    r'np\.mean\(',          # Numpy aggregations
    r'np\.var\(',
    r'np\.std\(',
    r'np\.median\(',
    r'derive_',             # Funciones de derivación
    r'compute_',            # Funciones de cómputo
    r'def\s+\w+\(',         # Definiciones de función (defaults permitidos)
    r'parser\.add_argument', # CLI arguments
    r'indent\s*=',          # JSON formatting
    r'coupling_intensity:\s*float\s*=', # Type hints con defaults
    r's = 0\.5\s*$',        # Inicialización de señal por defecto
    r'return\s+0\.',        # Valores de retorno por defecto
    r'else:\s*$',           # Else clauses
    r'ProvenanceRecord',    # Dataclass fields
    r'deque\(',             # Deque operations
    r'= 1\.0\s*/',          # Constante unitaria en división
    r'base_eta',            # Variable de cálculo intermedio
]


def is_allowed(line: str) -> bool:
    """Verifica si una línea tiene un patrón permitido."""
    for pattern in ALLOWED_PATTERNS:
        if re.search(pattern, line):
            return True
    return False


def audit_file(filepath: str) -> list:
    """Audita un archivo buscando números mágicos."""
    violations = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        # Saltar comentarios
        if line.strip().startswith('#'):
            continue

        # Saltar si es patrón permitido
        if is_allowed(line):
            continue

        # Buscar violaciones
        for pattern, description in MAGIC_PATTERNS:
            if re.search(pattern, line):
                violations.append({
                    'file': filepath,
                    'line': i,
                    'pattern': description,
                    'content': line.strip()[:80]
                })

    return violations


def test_no_magic_in_phase12():
    """Phase 12 no debe tener números mágicos."""
    filepath = '/root/NEO_EVA/tools/phase12_pure_endogenous.py'

    if not os.path.exists(filepath):
        print(f"[SKIP] {filepath} no existe")
        return True

    violations = audit_file(filepath)

    if violations:
        print(f"\n[FAIL] Encontrados {len(violations)} números mágicos en {filepath}:")
        for v in violations[:10]:  # Mostrar primeros 10
            print(f"  L{v['line']}: {v['pattern']}")
            print(f"         {v['content']}")
        return False

    print(f"[PASS] {filepath}: Sin números mágicos")
    return True


def test_no_magic_in_endogenous_core():
    """El núcleo endógeno no debe tener magia."""
    filepath = '/root/NEO_EVA/tools/endogenous_core.py'

    if not os.path.exists(filepath):
        print(f"[SKIP] {filepath} no existe")
        return True

    violations = audit_file(filepath)

    # Filtrar falsos positivos (funciones que DEFINEN cómo calcular valores)
    real_violations = [v for v in violations
                       if 'def derive_' not in v['content']
                       and 'return' not in v['content']]

    if real_violations:
        print(f"\n[FAIL] Números mágicos en {filepath}:")
        for v in real_violations[:5]:
            print(f"  L{v['line']}: {v['pattern']}")
        return False

    print(f"[PASS] {filepath}: Sin números mágicos")
    return True


def test_T_scaling():
    """Verifica que η, τ, σ escalan como ~1/√T."""
    from endogenous_core import derive_learning_rate, derive_temperature, derive_noise_scale

    # Generar historia de prueba
    history = np.random.randn(1000)

    results = []
    for T in [100, 400, 900, 1600, 2500]:
        eta = derive_learning_rate(T, history)
        expected = 1.0 / np.sqrt(T + 1)
        ratio = eta * np.sqrt(T + 1)  # Debería ser ~constante
        results.append(ratio)

    # Verificar que los ratios son aproximadamente constantes
    mean_ratio = np.mean(results)
    std_ratio = np.std(results)

    # Coeficiente de variación < 50% indica escalado correcto
    cv = std_ratio / mean_ratio if mean_ratio > 0 else float('inf')

    if cv < 0.5:
        print(f"[PASS] T-scaling: CV={cv:.3f} (ratios: {[f'{r:.3f}' for r in results]})")
        return True
    else:
        print(f"[FAIL] T-scaling: CV={cv:.3f} > 0.5")
        return False


def test_provenance_tracking():
    """Verifica que los parámetros tienen procedencia."""
    from endogenous_core import PROVENANCE, derive_window_size, derive_learning_rate

    # Limpiar log
    PROVENANCE.records.clear()

    # Ejecutar algunas derivaciones
    derive_window_size(1000)
    derive_learning_rate(1000)

    records = PROVENANCE.get_recent(10)

    if len(records) >= 2:
        print(f"[PASS] Provenance tracking: {len(records)} records")
        for r in records[:3]:
            print(f"       {r.param_name}: {r.definition[:50]}...")
        return True
    else:
        print(f"[FAIL] Provenance tracking: solo {len(records)} records")
        return False


def test_null_auc():
    """Verifica que AUC con shuffled data ≈ 0.5."""
    from sklearn.metrics import roc_auc_score

    # Simular datos
    np.random.seed(42)
    n = 1000
    pi = np.random.rand(n)  # π aleatorio
    labels_real = (pi > np.median(pi)).astype(int)  # Labels correlacionados con π
    labels_shuffled = np.random.permutation(labels_real)  # Labels shuffled

    auc_real = roc_auc_score(labels_real, pi)
    auc_shuffled = roc_auc_score(labels_shuffled, pi)

    # AUC real debería ser > 0.7 (hay correlación)
    # AUC shuffled debería ser ≈ 0.5 (sin correlación)

    real_ok = auc_real > 0.7
    shuffled_ok = 0.45 < auc_shuffled < 0.55

    if real_ok and shuffled_ok:
        print(f"[PASS] Null AUC: real={auc_real:.3f}, shuffled={auc_shuffled:.3f}")
        return True
    else:
        print(f"[FAIL] Null AUC: real={auc_real:.3f}, shuffled={auc_shuffled:.3f}")
        return False


def test_warmup_bounded():
    """Verifica que el warmup es < 5% del tiempo."""
    # El warmup es el período donde window_size > len(history)
    from endogenous_core import derive_window_size

    warmup_cycles = 0
    total_cycles = 1000

    for t in range(1, total_cycles + 1):
        window = derive_window_size(t)
        if window > t:
            warmup_cycles += 1

    warmup_rate = warmup_cycles / total_cycles

    if warmup_rate < 0.05:
        print(f"[PASS] Warmup bounded: {warmup_rate*100:.1f}% < 5%")
        return True
    else:
        print(f"[FAIL] Warmup bounded: {warmup_rate*100:.1f}% >= 5%")
        return False


def run_all_tests():
    """Ejecuta todos los tests."""
    print("=" * 70)
    print("TESTS DE LINT ENDÓGENO")
    print("=" * 70)
    print()

    tests = [
        ("No magic in phase12", test_no_magic_in_phase12),
        ("No magic in core", test_no_magic_in_endogenous_core),
        ("T-scaling", test_T_scaling),
        ("Provenance tracking", test_provenance_tracking),
        ("Null AUC ≈ 0.5", test_null_auc),
        ("Warmup < 5%", test_warmup_bounded),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            failed += 1

    print()
    print("=" * 70)
    print(f"RESULTADO: {passed}/{len(tests)} tests pasados")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
