#!/usr/bin/env python3
"""
SYM-X Benchmark Runner
======================

Ejecuta los 10 tests del benchmark SYM-X y genera reporte.
"""

import sys
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/root/NEO_EVA')

from benchmark.symx import TESTS


def run_all_tests(verbose: bool = True) -> dict:
    """
    Ejecuta todos los tests SYM-X.

    Returns:
        Dict con resultados por test y métricas globales.
    """
    results = {}
    scores = []
    passed_count = 0

    print("=" * 70)
    print("SYM-X BENCHMARK SUITE")
    print("Symbolic AI Capabilities Assessment")
    print("=" * 70)
    print()

    for test_id, (test_name, run_func) in TESTS.items():
        if verbose:
            print(f"Running {test_id}: {test_name}...", end=" ", flush=True)

        start_time = time.time()
        try:
            result = run_func()
            elapsed = time.time() - start_time

            results[test_id] = {
                'name': test_name,
                'score': result['score'],
                'passed': result['passed'],
                'details': result['details'],
                'elapsed_seconds': elapsed
            }

            scores.append(result['score'])
            if result['passed']:
                passed_count += 1

            if verbose:
                status = "PASS" if result['passed'] else "FAIL"
                print(f"{status} (score: {result['score']:.3f}, {elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - start_time
            results[test_id] = {
                'name': test_name,
                'score': 0.0,
                'passed': False,
                'error': str(e),
                'elapsed_seconds': elapsed
            }
            scores.append(0.0)

            if verbose:
                print(f"ERROR: {e}")

    # Calcular métricas globales
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    n_tests = len(TESTS)
    pass_rate = passed_count / n_tests

    # Calcular SYM_X score (ponderado por varianza inversa)
    if len(scores) > 1:
        variances = []
        for result in results.values():
            if 'details' in result:
                # Usar variabilidad de métricas internas como proxy
                detail_values = [v for v in result['details'].values() if isinstance(v, (int, float))]
                if detail_values:
                    variances.append(np.var(detail_values) + 0.01)
                else:
                    variances.append(0.1)
            else:
                variances.append(0.1)

        weights = [1.0 / v for v in variances]
        weight_sum = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / weight_sum
    else:
        weighted_score = mean_score

    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_tests': n_tests,
        'passed': passed_count,
        'failed': n_tests - passed_count,
        'pass_rate': pass_rate,
        'mean_score': mean_score,
        'std_score': std_score,
        'weighted_score': weighted_score,
        'sym_x_score': weighted_score,
        'individual_results': results
    }

    return summary


def print_summary(summary: dict) -> None:
    """Imprime resumen del benchmark."""
    print()
    print("=" * 70)
    print("SYM-X BENCHMARK RESULTS")
    print("=" * 70)
    print()

    print(f"Tests Run:     {summary['n_tests']}")
    print(f"Tests Passed:  {summary['passed']}")
    print(f"Tests Failed:  {summary['failed']}")
    print(f"Pass Rate:     {summary['pass_rate']*100:.1f}%")
    print()
    print(f"Mean Score:    {summary['mean_score']:.4f}")
    print(f"Std Score:     {summary['std_score']:.4f}")
    print(f"SYM_X Score:   {summary['sym_x_score']:.4f}")
    print()

    print("-" * 70)
    print(f"{'Test':<6} {'Name':<30} {'Score':>8} {'Status':>8}")
    print("-" * 70)

    for test_id, result in summary['individual_results'].items():
        name = result['name'][:28]
        score = result['score']
        status = "PASS" if result['passed'] else "FAIL"
        print(f"{test_id:<6} {name:<30} {score:>8.3f} {status:>8}")

    print("-" * 70)
    print()

    # Interpretación
    sym_x = summary['sym_x_score']
    if sym_x >= 0.8:
        level = "EXCELLENT - Mature symbolic AI system"
    elif sym_x >= 0.6:
        level = "GOOD - Functional symbolic capabilities"
    elif sym_x >= 0.4:
        level = "FAIR - Basic symbolic emergence"
    else:
        level = "POOR - Limited symbolic capabilities"

    print(f"Interpretation: {level}")
    print()


def save_results(summary: dict, output_dir: str = "/root/NEO_EVA/benchmark/results") -> str:
    """Guarda resultados en JSON y CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON completo
    json_path = f"{output_dir}/symx_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # CSV resumido
    csv_path = f"{output_dir}/symx_results_{timestamp}.csv"
    with open(csv_path, 'w') as f:
        f.write("test_id,name,score,passed\n")
        for test_id, result in summary['individual_results'].items():
            f.write(f"{test_id},{result['name']},{result['score']:.4f},{result['passed']}\n")
        f.write(f"\nSYM_X_SCORE,{summary['sym_x_score']:.4f}\n")
        f.write(f"PASS_RATE,{summary['pass_rate']:.4f}\n")

    return json_path


def main():
    """Ejecuta el benchmark SYM-X completo."""
    print(f"\nStarting SYM-X Benchmark at {datetime.now().isoformat()}\n")

    summary = run_all_tests(verbose=True)
    print_summary(summary)

    json_path = save_results(summary)
    print(f"Results saved to: {json_path}")

    return summary


if __name__ == "__main__":
    main()
