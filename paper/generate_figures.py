#!/usr/bin/env python3
"""
Generador de figuras para el paper NEO↔EVA v2.0-endogenous
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 150

FIGURES_DIR = Path('/root/NEO_EVA/paper/figures')
FIGURES_DIR.mkdir(exist_ok=True)

def load_data():
    """Cargar todos los datos necesarios."""
    data = {}

    # Corrida larga
    with open('/root/NEO_EVA/repro/longrun_20k.json') as f:
        data['longrun'] = json.load(f)

    # v2 coupled
    with open('/root/NEO_EVA/results/phase6_v2_neo.json') as f:
        data['v2_neo'] = json.load(f)
    with open('/root/NEO_EVA/results/phase6_v2_eva.json') as f:
        data['v2_eva'] = json.load(f)
    with open('/root/NEO_EVA/results/phase6_v2_results.json') as f:
        data['v2_results'] = json.load(f)

    # v2 ablation
    with open('/root/NEO_EVA/results/phase6_v2_ablation_results.json') as f:
        data['v2_ablation'] = json.load(f)

    # v1 (si existe)
    try:
        with open('/root/NEO_EVA/results/phase6_coupled_results.json') as f:
            data['v1_results'] = json.load(f)
    except:
        data['v1_results'] = None

    return data


def fig1_longrun_correlation(data):
    """Figura 1: Evolución de correlación en corrida larga."""
    snapshots = data['longrun']['snapshots']

    cycles = [s['cycle'] for s in snapshots]
    mean_corr = [s['mean_corr'] for s in snapshots]
    corr_S = [s['corr_S'] for s in snapshots]
    corr_N = [s['corr_N'] for s in snapshots]
    corr_C = [s['corr_C'] for s in snapshots]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(cycles, mean_corr, 'k-', linewidth=2, label='Media', marker='o', markersize=6)
    ax.plot(cycles, corr_S, 'r--', linewidth=1.5, alpha=0.7, label='S (Sintaxis)')
    ax.plot(cycles, corr_N, 'g--', linewidth=1.5, alpha=0.7, label='N (Novedad)')
    ax.plot(cycles, corr_C, 'b--', linewidth=1.5, alpha=0.7, label='C (Coherencia)')

    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    ax.fill_between(cycles, -0.1, 0.1, alpha=0.1, color='gray', label='Zona ~0')

    ax.set_xlabel('Ciclo')
    ax.set_ylabel('Correlación Pearson (NEO↔EVA)')
    ax.set_title('Figura 1: Evolución de Correlación en Corrida Larga (20k ciclos)')
    ax.legend(loc='upper right')
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, 20000)

    # Anotaciones
    ax.annotate('Correlación inicial\n(warmup)', xy=(2000, 0.57), xytext=(4000, 0.75),
                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                fontsize=9, ha='center')
    ax.annotate('Convergencia a ~0\n(equilibrio)', xy=(18000, -0.07), xytext=(15000, -0.4),
                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                fontsize=9, ha='center')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig1_longrun_correlation.png', dpi=150, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig1_longrun_correlation.pdf', bbox_inches='tight')
    plt.close()
    print("[OK] Figura 1: Corrida larga")


def fig2_coupling_activations(data):
    """Figura 2: Activaciones de acoplamiento."""
    snapshots = data['longrun']['snapshots']

    cycles = [s['cycle'] for s in snapshots]
    neo_acts = [s['neo_coupling_acts'] for s in snapshots]
    eva_acts = [s['eva_coupling_acts'] for s in snapshots]

    # Calcular tasas incrementales
    neo_rates = [neo_acts[0]/cycles[0]*100]
    eva_rates = [eva_acts[0]/cycles[0]*100]
    for i in range(1, len(cycles)):
        neo_rates.append((neo_acts[i]-neo_acts[i-1])/(cycles[i]-cycles[i-1])*100)
        eva_rates.append((eva_acts[i]-eva_acts[i-1])/(cycles[i]-cycles[i-1])*100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Panel izquierdo: acumulado
    ax1.plot(cycles, neo_acts, 'b-', linewidth=2, label='NEO', marker='s', markersize=5)
    ax1.plot(cycles, eva_acts, 'r-', linewidth=2, label='EVA', marker='o', markersize=5)
    ax1.set_xlabel('Ciclo')
    ax1.set_ylabel('Activaciones Acumuladas')
    ax1.set_title('(a) Activaciones Acumuladas')
    ax1.legend()
    ax1.set_xlim(0, 20000)

    # Panel derecho: tasa por ventana
    ax2.bar(np.array(cycles)-250, neo_rates, width=400, alpha=0.7, label='NEO', color='blue')
    ax2.bar(np.array(cycles)+250, eva_rates, width=400, alpha=0.7, label='EVA', color='red')
    ax2.set_xlabel('Ciclo')
    ax2.set_ylabel('Tasa de Activación (%)')
    ax2.set_title('(b) Tasa por Ventana de 1k')
    ax2.legend()
    ax2.set_xlim(0, 20000)

    plt.suptitle('Figura 2: Activaciones de Acoplamiento κ', fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig2_coupling_activations.png', dpi=150, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig2_coupling_activations.pdf', bbox_inches='tight')
    plt.close()
    print("[OK] Figura 2: Activaciones")


def fig3_v1_vs_v2_comparison(data):
    """Figura 3: Comparación v1 vs v2."""

    # Datos
    categories = ['Correlación\nMedia', 'Activaciones\nNEO (%)', 'Activaciones\nEVA (%)', 'Varianza\nNEO', 'Varianza\nEVA']

    v1_values = [0.35, 27.6, 29.0, 0.267, 0.210]  # De resultados anteriores
    v2_coupled = [-0.07, 3.8, 31.9, 0.282, 0.609]  # De corrida larga final
    v2_ablation = [0.01, 0, 0, 0.356, 0.289]  # De ablación

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))

    bars1 = ax.bar(x - width, v1_values, width, label='v1 (hardcoded)', color='#ff7f0e', alpha=0.8)
    bars2 = ax.bar(x, v2_coupled, width, label='v2 (endógeno)', color='#2ca02c', alpha=0.8)
    bars3 = ax.bar(x + width, v2_ablation, width, label='v2 ablación', color='#d62728', alpha=0.8)

    ax.set_ylabel('Valor')
    ax.set_title('Figura 3: Comparación v1 (hardcoded) vs v2 (100% endógeno)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)

    # Añadir valores sobre barras
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig3_v1_vs_v2.png', dpi=150, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig3_v1_vs_v2.pdf', bbox_inches='tight')
    plt.close()
    print("[OK] Figura 3: v1 vs v2")


def fig4_endogenous_parameters(data):
    """Figura 4: Parámetros endógenos (τ, η, límites OU)."""
    snapshots = data['longrun']['snapshots']

    cycles = [s['cycle'] for s in snapshots]
    tau_last = [s['tau_last'] for s in snapshots]

    # Valores teóricos de escalado 1/√T
    theoretical = [1.0 / np.sqrt(c) * 0.01 for c in cycles]  # Escala aproximada

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Panel izquierdo: τ observado
    ax1.semilogy(cycles, tau_last, 'b-', linewidth=2, label='τ observado', marker='o', markersize=5)
    ax1.semilogy(cycles, theoretical, 'r--', linewidth=1.5, alpha=0.7, label='∝ 1/√T (teórico)')
    ax1.set_xlabel('Ciclo')
    ax1.set_ylabel('τ (escala log)')
    ax1.set_title('(a) Tasa de Aprendizaje τ')
    ax1.legend()
    ax1.set_xlim(0, 20000)

    # Panel derecho: Diagrama de fórmulas
    ax2.axis('off')
    formulas = [
        r'$w = \max\{10, \lfloor\sqrt{T}\rfloor\}$',
        r'$\sigma_{med} = \mathrm{median}(\sigma_S, \sigma_N, \sigma_C)$',
        r'$\tau = \frac{\mathrm{IQR}(r)}{\sqrt{T}} \cdot \frac{\sigma_{med}}{\mathrm{IQR}_{hist} + \epsilon}$',
        r'$\tau_{floor} = \frac{\sigma_{med}}{T}$',
        r'$\eta = \tau$ (sin boost)',
        r'Gate: $\rho \geq \rho_{p95}$ AND $\mathrm{IQR} \geq \mathrm{IQR}_{p75}$',
    ]
    y_pos = 0.9
    for formula in formulas:
        ax2.text(0.1, y_pos, formula, fontsize=12, transform=ax2.transAxes,
                verticalalignment='top', fontfamily='serif')
        y_pos -= 0.15
    ax2.set_title('(b) Fórmulas Endógenas', fontsize=12)

    plt.suptitle('Figura 4: Parámetros Endógenos', fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig4_endogenous_params.png', dpi=150, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig4_endogenous_params.pdf', bbox_inches='tight')
    plt.close()
    print("[OK] Figura 4: Parámetros endógenos")


def fig5_audit_summary():
    """Figura 5: Resumen de auditoría."""

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # Tabla de auditoría
    table_data = [
        ['Módulo', 'Estado', 'Detalles'],
        ['Auditoría Estática', '✅ PASS', '0 violaciones de 252 hallazgos'],
        ['Auditoría Dinámica', '✅ PASS', '2/2 tests de invariancia'],
        ['Auditoría κ', '✅ PASS', '5 ejemplos, sin magia'],
        ['Estado Global', '✅ GO', 'Listo para publicación'],
    ]

    colors = [['#e6e6e6']*3,
              ['#d4edda', '#d4edda', '#d4edda'],
              ['#d4edda', '#d4edda', '#d4edda'],
              ['#d4edda', '#d4edda', '#d4edda'],
              ['#c3e6cb', '#c3e6cb', '#c3e6cb']]

    table = ax.table(cellText=table_data, cellColours=colors,
                     loc='center', cellLoc='center',
                     colWidths=[0.3, 0.2, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Título
    ax.set_title('Figura 5: Resumen de Auditoría de Endogeneidad', fontsize=14, pad=20)

    # Principio
    ax.text(0.5, 0.05, '"Si no sale de la historia, no entra en la dinámica"',
            ha='center', va='center', fontsize=12, style='italic',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig5_audit_summary.png', dpi=150, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig5_audit_summary.pdf', bbox_inches='tight')
    plt.close()
    print("[OK] Figura 5: Auditoría")


def fig6_simplex_trajectory(data):
    """Figura 6: Trayectoria en el simplex."""

    # Cargar series
    neo_series = data['v2_neo']['series'][-100:]  # Últimos 100 pasos
    eva_series = data['v2_eva']['series'][-100:]

    neo_S = [s['S_new'] for s in neo_series]
    neo_N = [s['N_new'] for s in neo_series]
    neo_C = [s['C_new'] for s in neo_series]

    eva_S = [s['S_new'] for s in eva_series]
    eva_N = [s['N_new'] for s in eva_series]
    eva_C = [s['C_new'] for s in eva_series]

    # Proyección 2D del simplex (coordenadas baricéntricas)
    def to_2d(s, n, c):
        x = 0.5 * (2*n + c)
        y = (np.sqrt(3)/2) * c
        return x, y

    neo_x, neo_y = [], []
    eva_x, eva_y = [], []

    for s, n, c in zip(neo_S, neo_N, neo_C):
        x, y = to_2d(s, n, c)
        neo_x.append(x)
        neo_y.append(y)

    for s, n, c in zip(eva_S, eva_N, eva_C):
        x, y = to_2d(s, n, c)
        eva_x.append(x)
        eva_y.append(y)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Dibujar triángulo del simplex
    triangle = plt.Polygon([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]],
                           fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(triangle)

    # Etiquetas de vértices
    ax.text(-0.05, -0.05, 'S (Sintaxis)', fontsize=10, ha='center')
    ax.text(1.05, -0.05, 'N (Novedad)', fontsize=10, ha='center')
    ax.text(0.5, np.sqrt(3)/2 + 0.05, 'C (Coherencia)', fontsize=10, ha='center')

    # Trayectorias
    ax.plot(neo_x, neo_y, 'b-', alpha=0.5, linewidth=1, label='NEO')
    ax.scatter(neo_x[-1], neo_y[-1], c='blue', s=100, marker='o', zorder=5, edgecolors='black')
    ax.scatter(neo_x[0], neo_y[0], c='blue', s=50, marker='s', zorder=5, alpha=0.5)

    ax.plot(eva_x, eva_y, 'r-', alpha=0.5, linewidth=1, label='EVA')
    ax.scatter(eva_x[-1], eva_y[-1], c='red', s=100, marker='o', zorder=5, edgecolors='black')
    ax.scatter(eva_x[0], eva_y[0], c='red', s=50, marker='s', zorder=5, alpha=0.5)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right')
    ax.set_title('Figura 6: Trayectorias en el Simplex (últimos 100 pasos)', fontsize=12)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig6_simplex_trajectory.png', dpi=150, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig6_simplex_trajectory.pdf', bbox_inches='tight')
    plt.close()
    print("[OK] Figura 6: Simplex")


def main():
    """Generar todas las figuras."""
    print("=" * 60)
    print("Generando figuras para el paper")
    print("=" * 60)

    data = load_data()

    fig1_longrun_correlation(data)
    fig2_coupling_activations(data)
    fig3_v1_vs_v2_comparison(data)
    fig4_endogenous_parameters(data)
    fig5_audit_summary()
    fig6_simplex_trajectory(data)

    print("=" * 60)
    print(f"Figuras guardadas en: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
