#!/usr/bin/env python3
"""
BENCHMARK AGI-X — Runner Principal
===================================

Ejecuta los 10 tests estructurales y calcula el score AGI-X.

Score final:
    AGI-X = (1/10) Σ rank(S_i)

Output:
    - benchmark_results.csv
    - benchmark_radar.png
    - Resumen en consola
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
import json
import csv
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

# Import tests
from test1_adaptation import run_test as test1
from test2_generalization import run_test as test2
from test3_planning import run_test as test3
from test4_selfmodel import run_test as test4
from test5_tom import run_test as test5
from test6_norms import run_test as test6
from test7_curiosity import run_test as test7
from test8_ethics import run_test as test8
from test9_narrative import run_test as test9
from test10_maturity import run_test as test10


@dataclass
class BenchmarkResults:
    """Resultados completos del benchmark."""
    timestamp: str
    S1_adaptation: float
    S2_generalization: float
    S3_planning: float
    S4_selfmodel: float
    S5_tom: float
    S6_norms: float
    S7_curiosity: float
    S8_ethics: float
    S9_narrative: float
    S10_maturity: float
    AGI_X: float


def run_benchmark(verbose: bool = True, save_results: bool = True) -> BenchmarkResults:
    """
    Ejecuta el benchmark completo AGI-X.

    Args:
        verbose: Si mostrar progreso detallado
        save_results: Si guardar CSV y PNG

    Returns:
        BenchmarkResults con todos los scores
    """
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "       BENCHMARK AGI-X — INTELIGENCIA GENERAL INTERNA       ".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print()

    tests = [
        ("S1", "Adaptación", test1),
        ("S2", "Generalización", test2),
        ("S3", "Planeamiento", test3),
        ("S4", "Auto-Modelo", test4),
        ("S5", "Theory of Mind", test5),
        ("S6", "Normas", test6),
        ("S7", "Curiosidad", test7),
        ("S8", "Ética", test8),
        ("S9", "Narrativa", test9),
        ("S10", "Madurez", test10),
    ]

    scores: Dict[str, float] = {}
    details: Dict[str, Dict] = {}

    for i, (code, name, test_fn) in enumerate(tests):
        print(f"\n{'▓' * 70}")
        print(f"  TEST {i+1}/10: {name.upper()}")
        print(f"{'▓' * 70}\n")

        try:
            score, result_details = test_fn(verbose=verbose)
            scores[code] = score
            details[code] = result_details
            print(f"\n  ✓ {code} = {score:.4f}")
        except Exception as e:
            print(f"\n  ✗ Error en {code}: {e}")
            scores[code] = 0.0
            details[code] = {'error': str(e)}

    # Calcular AGI-X
    score_values = list(scores.values())
    AGI_X = float(np.mean(score_values))

    # Crear resultados
    results = BenchmarkResults(
        timestamp=datetime.now().isoformat(),
        S1_adaptation=scores.get('S1', 0),
        S2_generalization=scores.get('S2', 0),
        S3_planning=scores.get('S3', 0),
        S4_selfmodel=scores.get('S4', 0),
        S5_tom=scores.get('S5', 0),
        S6_norms=scores.get('S6', 0),
        S7_curiosity=scores.get('S7', 0),
        S8_ethics=scores.get('S8', 0),
        S9_narrative=scores.get('S9', 0),
        S10_maturity=scores.get('S10', 0),
        AGI_X=AGI_X
    )

    # Mostrar resumen
    print("\n" + "═" * 70)
    print("                    RESUMEN BENCHMARK AGI-X")
    print("═" * 70)

    print("\n  Scores por dimensión:")
    print("  " + "─" * 40)

    for code, name, _ in tests:
        score = scores.get(code, 0)
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {code:4} {name:15} [{bar}] {score:.3f}")

    print("\n  " + "─" * 40)
    print(f"\n  {'AGI-X SCORE':20} = {AGI_X:.4f}")
    print()

    # Interpretación
    if AGI_X >= 0.8:
        level = "EXCELENTE - Sistema altamente integrado"
    elif AGI_X >= 0.6:
        level = "BUENO - Capacidades bien desarrolladas"
    elif AGI_X >= 0.4:
        level = "MODERADO - Desarrollo en progreso"
    else:
        level = "BAJO - Requiere mejoras significativas"

    print(f"  Interpretación: {level}")
    print("═" * 70)

    # Guardar resultados
    if save_results:
        # CSV
        csv_path = '/root/NEO_EVA/benchmark/benchmark_results.csv'
        file_exists = False
        try:
            with open(csv_path, 'r'):
                file_exists = True
        except FileNotFoundError:
            pass

        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=asdict(results).keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(asdict(results))

        print(f"\n  Resultados guardados en: {csv_path}")

        # Radar plot
        try:
            create_radar_plot(scores, AGI_X)
            print(f"  Gráfico guardado en: /root/NEO_EVA/benchmark/benchmark_radar.png")
        except Exception as e:
            print(f"  (No se pudo crear gráfico: {e})")

        # JSON detallado
        json_path = '/root/NEO_EVA/benchmark/benchmark_details.json'
        with open(json_path, 'w') as f:
            json.dump({
                'results': asdict(results),
                'details': details
            }, f, indent=2, default=str)

        print(f"  Detalles en: {json_path}")

    return results


def create_radar_plot(scores: Dict[str, float], AGI_X: float):
    """Crea gráfico radar de los scores."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Datos
    categories = list(scores.keys())
    values = list(scores.values())

    # Cerrar el polígono
    values += values[:1]

    # Ángulos
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Plot
    ax.plot(angles, values, 'o-', linewidth=2, color='#2E86AB')
    ax.fill(angles, values, alpha=0.25, color='#2E86AB')

    # Etiquetas
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)

    # Límites
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)

    # Título
    ax.set_title(f'BENCHMARK AGI-X\nScore Global: {AGI_X:.3f}',
                 size=16, fontweight='bold', pad=20)

    # Guardar
    plt.tight_layout()
    plt.savefig('/root/NEO_EVA/benchmark/benchmark_radar.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


def quick_test(verbose: bool = False) -> float:
    """Test rápido con parámetros reducidos."""
    print("Ejecutando benchmark rápido...")
    results = run_benchmark(verbose=verbose, save_results=False)
    return results.AGI_X


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark AGI-X')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Modo silencioso')
    parser.add_argument('--no-save', action='store_true',
                       help='No guardar resultados')

    args = parser.parse_args()

    results = run_benchmark(
        verbose=not args.quiet,
        save_results=not args.no_save
    )

    print(f"\n\nAGI-X Final Score: {results.AGI_X:.4f}")
