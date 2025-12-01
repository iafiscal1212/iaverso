"""
Visualización de resultados del benchmark AGI-X v2.0
"""

import numpy as np
import matplotlib.pyplot as plt
from benchmark_suite import AGIXBenchmarkSuite


def create_radar_chart(categories, scores, title="AGI-X v2.0 Benchmark Results"):
    """Crea gráfico radar de scores por categoría."""
    # Número de categorías
    N = len(categories)

    # Ángulos para cada categoría
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Cerrar el círculo

    # Scores (cerrar el círculo)
    values = list(scores.values())
    values += values[:1]

    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Dibujar el radar
    ax.plot(angles, values, 'o-', linewidth=2, color='#2E86AB')
    ax.fill(angles, values, alpha=0.25, color='#2E86AB')

    # Etiquetas
    short_labels = [cat.replace('_', '\n') for cat in categories]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short_labels, size=9)

    # Límites
    ax.set_ylim(0, 1)

    # Título
    plt.title(title, size=14, y=1.08)

    return fig


def create_bar_chart(results, title="AGI-X v2.0 - All 40 Tests"):
    """Crea gráfico de barras de todos los tests."""
    # Extraer datos
    test_ids = [r.test_id for r in results]
    scores = [r.score for r in results]
    categories = [r.category for r in results]

    # Colores por categoría
    unique_cats = list(set(categories))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cats)))
    cat_colors = {cat: colors[i] for i, cat in enumerate(unique_cats)}
    bar_colors = [cat_colors[cat] for cat in categories]

    # Crear figura
    fig, ax = plt.subplots(figsize=(16, 8))

    # Barras
    bars = ax.bar(range(len(test_ids)), scores, color=bar_colors)

    # Línea de umbral
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold (0.5)')
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Excellent (0.8)')

    # Etiquetas
    ax.set_xticks(range(len(test_ids)))
    ax.set_xticklabels(test_ids, rotation=45, ha='right', size=8)
    ax.set_xlabel('Test ID')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend()

    # Límites
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    return fig


def create_summary_table(results):
    """Crea tabla resumen por categoría."""
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = {
                'scores': [],
                'passed': 0,
                'excellent': 0,
                'total': 0
            }
        categories[r.category]['scores'].append(r.score)
        categories[r.category]['total'] += 1
        if r.score >= 0.5:
            categories[r.category]['passed'] += 1
        if r.score >= 0.8:
            categories[r.category]['excellent'] += 1

    # Crear figura con tabla
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    # Datos de tabla
    cell_text = []
    for cat, data in categories.items():
        avg = np.mean(data['scores'])
        cell_text.append([
            cat.replace('_', ' '),
            f"{avg:.3f}",
            f"{data['passed']}/{data['total']}",
            f"{data['excellent']}/{data['total']}",
            "PASS" if avg >= 0.5 else "FAIL"
        ])

    # Tabla
    table = ax.table(
        cellText=cell_text,
        colLabels=['Category', 'Avg Score', 'Passed', 'Excellent', 'Status'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Colorear celdas de status
    for i, row in enumerate(cell_text):
        cell = table[(i + 1, 4)]
        if row[4] == "PASS":
            cell.set_facecolor('#90EE90')
        else:
            cell.set_facecolor('#FFB6C1')

    plt.title('AGI-X v2.0 Benchmark Summary by Category', fontsize=14, y=0.95)
    return fig


def main():
    """Ejecuta benchmark y genera visualizaciones."""
    print("Ejecutando benchmark...")
    benchmark = AGIXBenchmarkSuite(n_agents=5, n_steps=100)
    category_scores = benchmark.run_all()

    print("\nGenerando visualizaciones...")

    # 1. Radar chart
    fig_radar = create_radar_chart(
        list(category_scores.keys()),
        category_scores
    )
    fig_radar.savefig('/root/NEO_EVA/agi_v2/benchmark_radar.png', dpi=150, bbox_inches='tight')
    print("  - benchmark_radar.png guardado")

    # 2. Bar chart
    fig_bar = create_bar_chart(benchmark.results)
    fig_bar.savefig('/root/NEO_EVA/agi_v2/benchmark_bars.png', dpi=150, bbox_inches='tight')
    print("  - benchmark_bars.png guardado")

    # 3. Summary table
    fig_table = create_summary_table(benchmark.results)
    fig_table.savefig('/root/NEO_EVA/agi_v2/benchmark_summary.png', dpi=150, bbox_inches='tight')
    print("  - benchmark_summary.png guardado")

    # Print report
    print("\n" + benchmark.get_report())

    # Save JSON results
    import json
    results_dict = {
        'category_scores': category_scores,
        'all_results': [
            {
                'test_id': r.test_id,
                'category': r.category,
                'description': r.description,
                'score': r.score
            }
            for r in benchmark.results
        ],
        'summary': {
            'total_tests': len(benchmark.results),
            'passed': sum(1 for r in benchmark.results if r.score >= 0.5),
            'excellent': sum(1 for r in benchmark.results if r.score >= 0.8),
            'average_score': np.mean([r.score for r in benchmark.results])
        }
    }

    with open('/root/NEO_EVA/agi_v2/benchmark_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    print("  - benchmark_results.json guardado")

    plt.close('all')
    print("\nVisualizaciones completadas!")


if __name__ == "__main__":
    main()
