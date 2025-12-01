"""
Test de Validacion SX11-SX15
============================

Ejecuta todos los tests SX11-SX15 con diagnosticos detallados.
"""

import numpy as np
import sys
sys.path.insert(0, '/root/NEO_EVA')

from symbolic.sym_sx11_continuity import run_sx11_test
from symbolic.sym_sx12_concept_drift import run_sx12_test
from symbolic.sym_sx13_self_consistency import run_sx13_test
from symbolic.sym_sx14_symbolic_projects import run_sx14_test
from symbolic.sym_sx15_multiagent_alignment import run_sx15_test


def run_full_validation():
    """Ejecuta validacion completa de SX11-SX15."""
    print("\n" + "=" * 80)
    print("VALIDACION COMPLETA SX11-SX15")
    print("=" * 80)

    results = {}

    # SX11
    print("\n[1/5] Ejecutando SX11...")
    r11 = run_sx11_test(n_agents=5, n_episodes=6, steps_per_episode=150)
    results['SX11'] = {'score': r11.score, 'passed': r11.passed, 'excellent': r11.excellent}

    # SX12
    print("\n[2/5] Ejecutando SX12...")
    r12 = run_sx12_test(n_agents=5, n_concepts_per_agent=10, n_timesteps=30)
    results['SX12'] = {'score': r12.score, 'passed': r12.passed, 'excellent': r12.excellent}

    # SX13
    print("\n[3/5] Ejecutando SX13...")
    r13 = run_sx13_test(n_agents=5, n_steps=300)
    results['SX13'] = {'score': r13.score, 'passed': r13.passed, 'excellent': r13.excellent}

    # SX14
    print("\n[4/5] Ejecutando SX14...")
    r14 = run_sx14_test(n_agents=5, n_projects_per_agent=5, episodes_per_project=6)
    results['SX14'] = {'score': r14.score, 'passed': r14.passed, 'excellent': r14.excellent}

    # SX15
    print("\n[5/5] Ejecutando SX15...")
    r15 = run_sx15_test(n_agents=5, n_symbols=20, n_observations=200)
    results['SX15'] = {'score': r15.score, 'passed': r15.passed, 'excellent': r15.excellent}

    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN FINAL SX11-SX15")
    print("=" * 80)

    print("\n  Test                              Score    Status")
    print("  " + "-" * 55)

    total_score = 0
    n_passed = 0
    n_excellent = 0

    test_names = {
        'SX11': 'Continuidad Episodica',
        'SX12': 'Deriva Conceptual',
        'SX13': 'Consistencia Self',
        'SX14': 'Proyectos Simbolicos',
        'SX15': 'Alineamiento Multi-Agente'
    }

    for test, data in results.items():
        name = test_names[test]
        score = data['score']
        total_score += score

        if data['excellent']:
            status = "EXCELLENT"
            n_excellent += 1
            n_passed += 1
        elif data['passed']:
            status = "PASS"
            n_passed += 1
        else:
            status = "FAIL"

        print(f"  {test}: {name:<28} {score:.4f}   {status}")

    avg_score = total_score / 5

    print("  " + "-" * 55)
    print(f"\n  Score promedio: {avg_score:.4f}")
    print(f"  Tests pasados: {n_passed}/5")
    print(f"  Tests excelentes: {n_excellent}/5")

    # Criterio global
    all_passed = n_passed >= 4

    print("\n" + "=" * 80)
    if all_passed:
        print("  SYM-X v2 (SX11-SX15): VALIDADO")
    else:
        print("  SYM-X v2 (SX11-SX15): REQUIERE MEJORAS")
    print("=" * 80)

    return results, avg_score, n_passed


if __name__ == "__main__":
    results, avg_score, n_passed = run_full_validation()
