#!/usr/bin/env python3
"""
Runner para Phases R1-R5
=========================

Ejecuta todas las fases de razonamiento, goals, tareas,
proto-lenguaje y fenomenología.

Genera:
- Reportes JSON por fase
- Resumen consolidado
- Figuras de visualización
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Importar módulos
sys.path.insert(0, '/root/NEO_EVA/tools')

from phaseR1_structural_reasoning import run_phaseR1_test
from phaseR2_goal_manifold import run_phaseR2_test
from phaseR3_task_acquisition import run_phaseR3_test
from phaseR4_proto_language import run_phaseR4_test
from phaseR5_phenomenology import run_phaseR5_test


def generate_summary_figure(results: dict, output_path: str):
    """Genera figura resumen de todas las fases R."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # R1: Razonamiento
    ax = axes[0, 0]
    if 'R1' in results and 'reasons' in results['R1']:
        ax.plot(results['R1']['reasons'], alpha=0.7)
        ax.axhline(y=np.mean(results['R1']['reasons']), color='r', linestyle='--', label='mean')
        ax.set_title(f"R1: Structural Reasoning\n{'GO' if results['R1']['go'] else 'NO-GO'}")
        ax.set_xlabel('Step')
        ax.set_ylabel('Structural Reason R(ẑ)')
        ax.legend()

    # R2: Goal Manifold
    ax = axes[0, 1]
    if 'R2' in results and 'G_values' in results['R2']:
        ax.plot(results['R2']['G_values'], alpha=0.7, label='G(z)')
        ax.plot(results['R2']['S_values'], alpha=0.5, label='S')
        ax.set_title(f"R2: Goal Manifold\n{'GO' if results['R2']['go'] else 'NO-GO'}")
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.legend()

    # R3: Task Acquisition
    ax = axes[0, 2]
    if 'R3' in results and 'tasks_over_time' in results['R3']:
        ax.plot(results['R3']['tasks_over_time'], alpha=0.7, label='Valid Tasks')
        ax.set_title(f"R3: Task Acquisition\n{'GO' if results['R3']['go'] else 'NO-GO'}")
        ax.set_xlabel('Step')
        ax.set_ylabel('Number of Tasks')
        ax.legend()

    # R4: Proto-Language
    ax = axes[1, 0]
    if 'R4' in results and 'coordination_values' in results['R4']:
        # Moving average
        coords = results['R4']['coordination_values']
        window = 50
        if len(coords) > window:
            ma = np.convolve(coords, np.ones(window)/window, mode='valid')
            ax.plot(ma, alpha=0.7)
        ax.set_title(f"R4: Proto-Language\n{'GO' if results['R4']['go'] else 'NO-GO'}")
        ax.set_xlabel('Step')
        ax.set_ylabel('Coordination (MA)')

    # R5: Phenomenology
    ax = axes[1, 1]
    if 'R5' in results and 'PSI_values' in results['R5']:
        ax.plot(results['R5']['PSI_values'], alpha=0.7, label='PSI')
        ax.plot(results['R5']['CF_values'], alpha=0.5, label='CF')
        ax.set_title(f"R5: Phenomenology (Ψ²)\n{'GO' if results['R5']['go'] else 'NO-GO'}")
        ax.set_xlabel('Step')
        ax.set_ylabel('Index')
        ax.legend()

    # Summary
    ax = axes[1, 2]
    phases = ['R1', 'R2', 'R3', 'R4', 'R5']
    go_status = [results.get(p, {}).get('go', False) for p in phases]
    colors = ['green' if g else 'red' for g in go_status]
    ax.barh(phases, [1]*5, color=colors, alpha=0.7)
    ax.set_xlim(0, 1.5)
    ax.set_title('Summary: GO/NO-GO')
    for i, (phase, status) in enumerate(zip(phases, go_status)):
        ax.text(1.1, i, 'GO' if status else 'NO-GO', va='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Figura guardada: {output_path}")


def generate_criteria_figure(results: dict, output_path: str):
    """Genera figura de criterios por fase."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, phase in enumerate(['R1', 'R2', 'R3', 'R4', 'R5']):
        ax = axes[idx // 3, idx % 3]

        if phase in results and 'criteria' in results[phase]:
            criteria = results[phase]['criteria']
            names = list(criteria.keys())
            values = [1 if v else 0 for v in criteria.values()]
            colors = ['green' if v else 'red' for v in criteria.values()]

            y_pos = np.arange(len(names))
            ax.barh(y_pos, values, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([n[:20] + '...' if len(n) > 20 else n for n in names], fontsize=8)
            ax.set_xlim(0, 1.5)
            ax.set_title(f'Phase {phase} Criteria')

    # Último subplot: resumen numérico
    ax = axes[1, 2]
    summary_text = []
    for phase in ['R1', 'R2', 'R3', 'R4', 'R5']:
        if phase in results:
            n_pass = sum(results[phase].get('criteria', {}).values())
            n_total = len(results[phase].get('criteria', {}))
            go = results[phase].get('go', False)
            summary_text.append(f"{phase}: {n_pass}/{n_total} {'GO' if go else 'NO-GO'}")

    ax.text(0.1, 0.5, '\n'.join(summary_text), fontsize=12, family='monospace',
            verticalalignment='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Summary')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Figura guardada: {output_path}")


def main():
    """Ejecuta todas las fases R."""
    print("=" * 70)
    print("PHASES R1-R5: ADVANCED STRUCTURAL COGNITION")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}")
    print()

    # Crear directorios
    results_dir = '/root/NEO_EVA/results'
    figures_dir = '/root/NEO_EVA/figures'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    results = {}
    total_start = time.time()

    # Phase R1: Structural Reasoning
    print("\n" + "=" * 70)
    print("EJECUTANDO PHASE R1...")
    print("=" * 70)
    start = time.time()
    results['R1'] = run_phaseR1_test(n_steps=500)
    print(f"R1 completado en {time.time() - start:.1f}s")

    # Phase R2: Goal Manifold
    print("\n" + "=" * 70)
    print("EJECUTANDO PHASE R2...")
    print("=" * 70)
    start = time.time()
    results['R2'] = run_phaseR2_test(n_steps=1000)
    print(f"R2 completado en {time.time() - start:.1f}s")

    # Phase R3: Task Acquisition
    print("\n" + "=" * 70)
    print("EJECUTANDO PHASE R3...")
    print("=" * 70)
    start = time.time()
    results['R3'] = run_phaseR3_test(n_steps=1500)
    print(f"R3 completado en {time.time() - start:.1f}s")

    # Phase R4: Proto-Language
    print("\n" + "=" * 70)
    print("EJECUTANDO PHASE R4...")
    print("=" * 70)
    start = time.time()
    results['R4'] = run_phaseR4_test(n_steps=1500)
    print(f"R4 completado en {time.time() - start:.1f}s")

    # Phase R5: Phenomenology
    print("\n" + "=" * 70)
    print("EJECUTANDO PHASE R5...")
    print("=" * 70)
    start = time.time()
    results['R5'] = run_phaseR5_test(n_steps=1500)
    print(f"R5 completado en {time.time() - start:.1f}s")

    total_time = time.time() - total_start

    # Generar figuras
    print("\n" + "=" * 70)
    print("GENERANDO FIGURAS...")
    print("=" * 70)

    generate_summary_figure(results, f'{figures_dir}/phasesR_summary.png')
    generate_criteria_figure(results, f'{figures_dir}/phasesR_criteria.png')

    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN FINAL PHASES R1-R5")
    print("=" * 70)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_time_seconds': total_time,
        'phases': {}
    }

    all_go = True
    for phase in ['R1', 'R2', 'R3', 'R4', 'R5']:
        if phase in results:
            go = results[phase].get('go', False)
            n_pass = sum(results[phase].get('criteria', {}).values())
            n_total = len(results[phase].get('criteria', {}))

            summary['phases'][phase] = {
                'go': go,
                'criteria_passed': n_pass,
                'criteria_total': n_total
            }

            status = "✅ GO" if go else "❌ NO-GO"
            print(f"  {phase}: {status} ({n_pass}/{n_total} criteria)")

            if not go:
                all_go = False

    summary['all_go'] = all_go

    print()
    if all_go:
        print("🎉 TODAS LAS FASES R PASARON - SISTEMA COGNITIVO ESTRUCTURAL VALIDADO")
    else:
        print("⚠️  Algunas fases no pasaron - revisar criterios")

    print(f"\nTiempo total: {total_time:.1f}s")

    # Convertir numpy bools a Python bools
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        return obj

    # Guardar resumen
    with open(f'{results_dir}/phasesR_summary.json', 'w') as f:
        json.dump(convert_types(summary), f, indent=2)

    print(f"\nResultados guardados en:")
    print(f"  - {results_dir}/phasesR_summary.json")
    print(f"  - {figures_dir}/phasesR_summary.png")
    print(f"  - {figures_dir}/phasesR_criteria.png")

    return summary


if __name__ == "__main__":
    summary = main()
