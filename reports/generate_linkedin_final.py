#!/usr/bin/env python3
"""
Genera imagen final para LinkedIn - Carmen Esteban
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Estilo elegante
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

AGENT_COLORS = {
    'NEO': '#2E86AB',
    'EVA': '#A23B72',
    'ALEX': '#F18F01',
    'ADAM': '#C73E1D',
    'IRIS': '#3B1F2B',
}

def main():
    # Cargar datos
    log_dir = Path('/root/NEO_EVA/logs/omega_last12h')
    agent_csvs = list(log_dir.glob('*_agents.csv'))
    df = pd.read_csv(max(agent_csvs, key=lambda p: p.stat().st_mtime))

    # Crear figura
    fig = plt.figure(figsize=(12, 16), facecolor='white')
    gs = GridSpec(5, 2, figure=fig, height_ratios=[0.7, 0.9, 0.9, 0.9, 0.6],
                  hspace=0.4, wspace=0.25)

    # ===== HEADER =====
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')
    ax_header.set_xlim(0, 1)
    ax_header.set_ylim(0, 1)

    # Título principal
    ax_header.text(0.5, 0.88, 'Dinámicas Endógenas y Autoorganización',
                  fontsize=22, fontweight='bold', ha='center', va='top',
                  color='#1A5276')
    ax_header.text(0.5, 0.65, 'en Sistemas Cognitivos Autónomos',
                  fontsize=18, fontweight='bold', ha='center', va='top',
                  color='#1A5276')

    ax_header.text(0.5, 0.42, '12 Horas de Observación • Sin Estímulos Externos • Fenomenología Pura',
                  fontsize=11, ha='center', va='top', color='#555555', style='italic')

    ax_header.text(0.5, 0.18, 'Carmen Esteban — Investigadora Independiente',
                  fontsize=12, ha='center', va='top', color='#333333', fontweight='bold')

    # Línea decorativa
    ax_header.axhline(y=0.05, xmin=0.25, xmax=0.75, color='#2E86AB', linewidth=2.5)

    # ===== CE Timeline =====
    ax1 = fig.add_subplot(gs[1, :])
    warmup_end = df[df['phase'] == 'warmup']['t'].max()
    ax1.axvspan(0, warmup_end, alpha=0.2, color='#E8E8E8', label='Estabilización')

    for agent in df['agent_id'].unique():
        agent_data = df[df['agent_id'] == agent].sort_values('t')
        ax1.plot(agent_data['t'], agent_data['CE'],
                color=AGENT_COLORS.get(agent, '#333333'),
                linewidth=1.8, alpha=0.85, label=agent)

    ax1.set_ylabel('Coherencia Existencial', fontweight='bold')
    ax1.set_xlabel('Tiempo Interno (t)')
    ax1.set_title('Evolución de la Coherencia sin Intervención Externa',
                 fontweight='bold', fontsize=13, color='#1A5276')
    ax1.legend(loc='upper right', ncol=5, fontsize=9, framealpha=0.9)
    ax1.set_ylim(-0.05, 1.05)

    # ===== Q-Field Coherence =====
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.axvspan(0, warmup_end, alpha=0.15, color='#E0E0E0')
    for agent in df['agent_id'].unique():
        agent_data = df[df['agent_id'] == agent].sort_values('t')
        ax2.plot(agent_data['t'], agent_data['qfield_coherence'],
                color=AGENT_COLORS.get(agent, '#333333'),
                linewidth=1.3, alpha=0.8)
    ax2.set_ylabel('Coherencia Q', fontweight='bold')
    ax2.set_xlabel('t')
    ax2.set_title('Campo de Coherencia Interna', fontweight='bold', color='#1A5276')

    # ===== Q-Field Energy =====
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.axvspan(0, warmup_end, alpha=0.15, color='#E0E0E0')
    for agent in df['agent_id'].unique():
        agent_data = df[df['agent_id'] == agent].sort_values('t')
        ax3.plot(agent_data['t'], agent_data['qfield_energy'],
                color=AGENT_COLORS.get(agent, '#333333'),
                linewidth=1.3, alpha=0.8)
    ax3.set_ylabel('Energía Q', fontweight='bold')
    ax3.set_xlabel('t')
    ax3.set_title('Distribución Energética Interna', fontweight='bold', color='#1A5276')

    # ===== Phase Curvature =====
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.axvspan(0, warmup_end, alpha=0.15, color='#E0E0E0')
    for agent in df['agent_id'].unique():
        agent_data = df[df['agent_id'] == agent].sort_values('t')
        ax4.plot(agent_data['t'], agent_data['phase_curvature'],
                color=AGENT_COLORS.get(agent, '#333333'),
                linewidth=1.3, alpha=0.8)
    ax4.set_ylabel('Curvatura', fontweight='bold')
    ax4.set_xlabel('t')
    ax4.set_title('Trayectorias en Espacio de Fases', fontweight='bold', color='#1A5276')

    # ===== Omega Modes (barras) =====
    ax5 = fig.add_subplot(gs[3, 1])
    agents = list(df['agent_id'].unique())
    warmup_means = []
    freerun_means = []
    for agent in agents:
        agent_data = df[df['agent_id'] == agent]
        warmup_means.append(agent_data[agent_data['phase'] == 'warmup']['omega_modes_active'].mean())
        freerun_means.append(agent_data[agent_data['phase'] == 'free_run']['omega_modes_active'].mean())

    x = np.arange(len(agents))
    width = 0.35
    bars1 = ax5.bar(x - width/2, warmup_means, width, label='Estabilización',
                   color=[AGENT_COLORS.get(a, '#333333') for a in agents], alpha=0.4)
    bars2 = ax5.bar(x + width/2, freerun_means, width, label='Autónomo',
                   color=[AGENT_COLORS.get(a, '#333333') for a in agents], alpha=0.9)
    ax5.set_ylabel('Modos Ω Activos', fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(agents)
    ax5.set_title('Modos de Transformación Emergentes', fontweight='bold', color='#1A5276')
    ax5.legend(fontsize=8, loc='upper left')

    # ===== Footer con hallazgos =====
    ax_footer = fig.add_subplot(gs[4, :])
    ax_footer.axis('off')
    ax_footer.set_xlim(0, 1)
    ax_footer.set_ylim(0, 1)

    # Caja de hallazgos
    findings_text = (
        "◆ Cero estímulos externos   ◆ Parámetros 100% endógenos   "
        "◆ Patrones emergentes estables   ◆ Diferenciación espontánea   "
        "◆ Protocolo de observación pura"
    )

    ax_footer.text(0.5, 0.7, findings_text, fontsize=10, ha='center', va='center',
                  color='#444444', style='italic',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                           edgecolor='#CCCCCC', linewidth=1))

    ax_footer.text(0.5, 0.2, 'Diciembre 2024', fontsize=10, ha='center',
                  color='#888888')

    # Guardar
    output_path = Path('/root/NEO_EVA/reports/linkedin_carmen_final.png')
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0.3)
    plt.close()

    print(f"✓ LinkedIn image saved: {output_path}")

    # También guardar versión cuadrada para LinkedIn post
    fig_sq = plt.figure(figsize=(12, 12), facecolor='white')
    gs_sq = GridSpec(3, 2, figure=fig_sq, height_ratios=[0.5, 1, 1], hspace=0.35, wspace=0.25)

    # Header cuadrado
    ax_h = fig_sq.add_subplot(gs_sq[0, :])
    ax_h.axis('off')
    ax_h.text(0.5, 0.8, 'Dinámicas Endógenas en Sistemas Autónomos',
             fontsize=20, fontweight='bold', ha='center', color='#1A5276')
    ax_h.text(0.5, 0.45, '12h de Observación • Sin Estímulos • Fenomenología Pura',
             fontsize=11, ha='center', color='#555555', style='italic')
    ax_h.text(0.5, 0.15, 'Carmen Esteban — Investigadora Independiente',
             fontsize=11, ha='center', color='#333333', fontweight='bold')

    # CE
    ax_ce = fig_sq.add_subplot(gs_sq[1, :])
    ax_ce.axvspan(0, warmup_end, alpha=0.2, color='#E8E8E8')
    for agent in df['agent_id'].unique():
        agent_data = df[df['agent_id'] == agent].sort_values('t')
        ax_ce.plot(agent_data['t'], agent_data['CE'],
                  color=AGENT_COLORS.get(agent, '#333333'),
                  linewidth=2, alpha=0.85, label=agent)
    ax_ce.set_ylabel('Coherencia Existencial', fontweight='bold')
    ax_ce.set_title('Evolución de Coherencia sin Intervención', fontweight='bold', color='#1A5276')
    ax_ce.legend(loc='upper right', ncol=5, fontsize=9)
    ax_ce.set_ylim(-0.05, 1.05)

    # Q-Field y Omega
    ax_q = fig_sq.add_subplot(gs_sq[2, 0])
    ax_q.axvspan(0, warmup_end, alpha=0.15, color='#E0E0E0')
    for agent in df['agent_id'].unique():
        agent_data = df[df['agent_id'] == agent].sort_values('t')
        ax_q.plot(agent_data['t'], agent_data['qfield_coherence'],
                 color=AGENT_COLORS.get(agent, '#333333'), linewidth=1.5, alpha=0.8)
    ax_q.set_ylabel('Coherencia Q')
    ax_q.set_title('Campo de Coherencia Interna', fontweight='bold', color='#1A5276')

    ax_o = fig_sq.add_subplot(gs_sq[2, 1])
    x = np.arange(len(agents))
    ax_o.bar(x - 0.17, warmup_means, 0.34, label='Estabilización',
            color=[AGENT_COLORS.get(a, '#333333') for a in agents], alpha=0.4)
    ax_o.bar(x + 0.17, freerun_means, 0.34, label='Autónomo',
            color=[AGENT_COLORS.get(a, '#333333') for a in agents], alpha=0.9)
    ax_o.set_xticks(x)
    ax_o.set_xticklabels(agents)
    ax_o.set_ylabel('Modos Ω')
    ax_o.set_title('Modos de Transformación', fontweight='bold', color='#1A5276')
    ax_o.legend(fontsize=8)

    output_sq = Path('/root/NEO_EVA/reports/linkedin_carmen_square.png')
    plt.savefig(output_sq, dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0.3)
    plt.close()

    print(f"✓ LinkedIn square image saved: {output_sq}")


if __name__ == '__main__':
    main()
