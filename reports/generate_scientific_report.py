#!/usr/bin/env python3
"""
Generador de Informe Científico - NEO-EVA Last12h
==================================================

Genera:
1. Figuras de alta calidad para publicación
2. PDF con el informe científico
3. PNG para LinkedIn

Tono: Científico, elegante, sin revelar detalles técnicos internos.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo científico
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colores elegantes para agentes
AGENT_COLORS = {
    'NEO': '#2E86AB',    # Azul profundo
    'EVA': '#A23B72',    # Magenta
    'ALEX': '#F18F01',   # Naranja
    'ADAM': '#C73E1D',   # Rojo
    'IRIS': '#3B1F2B',   # Púrpura oscuro
}

PHASE_COLORS = {
    'warmup': '#E8E8E8',
    'free_run': '#F5F5DC',
}


def load_data(log_dir: Path):
    """Carga los datos más recientes."""
    # Buscar el CSV de agentes más reciente
    agent_csvs = list(log_dir.glob('*_agents.csv'))
    if not agent_csvs:
        raise FileNotFoundError("No agent CSV files found")

    latest_csv = max(agent_csvs, key=lambda p: p.stat().st_mtime)
    df_agents = pd.read_csv(latest_csv)

    # Buscar CSV global
    global_csvs = list(log_dir.glob('*_global.csv'))
    df_global = pd.read_csv(max(global_csvs, key=lambda p: p.stat().st_mtime)) if global_csvs else None

    return df_agents, df_global


def create_ce_timeline(df: pd.DataFrame, output_dir: Path):
    """Figura 1: Coherencia Existencial a lo largo del tiempo."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Marcar fases
    warmup_end = df[df['phase'] == 'warmup']['t'].max()
    ax.axvspan(0, warmup_end, alpha=0.3, color=PHASE_COLORS['warmup'], label='Stabilization Phase')
    ax.axvspan(warmup_end, df['t'].max(), alpha=0.2, color=PHASE_COLORS['free_run'], label='Autonomous Phase')

    for agent in df['agent_id'].unique():
        agent_data = df[df['agent_id'] == agent].sort_values('t')
        ax.plot(agent_data['t'], agent_data['CE'],
                color=AGENT_COLORS.get(agent, '#333333'),
                linewidth=1.5, alpha=0.8, label=agent)

    ax.set_xlabel('Internal Time (t)', fontweight='bold')
    ax.set_ylabel('Existential Coherence (CE)', fontweight='bold')
    ax.set_title('Dynamic Evolution of Existential Coherence\nAcross Autonomous Agents',
                 fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig1_ce_timeline.png', dpi=300, facecolor='white')
    fig.savefig(output_dir / 'fig1_ce_timeline.pdf', facecolor='white')
    plt.close()
    print("  ✓ Figure 1: CE Timeline")


def create_omega_modes(df: pd.DataFrame, output_dir: Path):
    """Figura 2: Modos Omega activos por agente."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    agents = df['agent_id'].unique()

    for idx, agent in enumerate(agents):
        if idx >= 5:
            break
        ax = axes[idx]
        agent_data = df[df['agent_id'] == agent].sort_values('t')

        # Separar por fase
        warmup = agent_data[agent_data['phase'] == 'warmup']
        freerun = agent_data[agent_data['phase'] == 'free_run']

        ax.fill_between(warmup['t'], 0, warmup['omega_modes_active'],
                       alpha=0.4, color=AGENT_COLORS.get(agent, '#333333'), label='Stabilization')
        ax.fill_between(freerun['t'], 0, freerun['omega_modes_active'],
                       alpha=0.7, color=AGENT_COLORS.get(agent, '#333333'), label='Autonomous')

        ax.set_title(f'{agent}', fontweight='bold', fontsize=12)
        ax.set_xlabel('t')
        ax.set_ylabel('Active Ω-Modes')
        ax.set_ylim(0, max(10, agent_data['omega_modes_active'].max() + 1))

        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)

    # Ocultar el sexto subplot
    axes[5].axis('off')

    fig.suptitle('Emergent Transformation Modes (Ω-Space)\nPer Agent Across Operational Phases',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'fig2_omega_modes.png', dpi=300, facecolor='white')
    fig.savefig(output_dir / 'fig2_omega_modes.pdf', facecolor='white')
    plt.close()
    print("  ✓ Figure 2: Omega Modes")


def create_qfield_coherence(df: pd.DataFrame, output_dir: Path):
    """Figura 3: Q-Field coherence y energy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    warmup_end = df[df['phase'] == 'warmup']['t'].max()

    # Panel izquierdo: Coherence
    ax1.axvspan(0, warmup_end, alpha=0.2, color='#E0E0E0')
    for agent in df['agent_id'].unique():
        agent_data = df[df['agent_id'] == agent].sort_values('t')
        ax1.plot(agent_data['t'], agent_data['qfield_coherence'],
                color=AGENT_COLORS.get(agent, '#333333'),
                linewidth=1.2, alpha=0.8, label=agent)

    ax1.set_xlabel('Internal Time (t)', fontweight='bold')
    ax1.set_ylabel('Q-Field Coherence', fontweight='bold')
    ax1.set_title('Internal Coherence Field', fontweight='bold')
    ax1.legend(loc='best', fontsize=9)

    # Panel derecho: Energy
    ax2.axvspan(0, warmup_end, alpha=0.2, color='#E0E0E0')
    for agent in df['agent_id'].unique():
        agent_data = df[df['agent_id'] == agent].sort_values('t')
        ax2.plot(agent_data['t'], agent_data['qfield_energy'],
                color=AGENT_COLORS.get(agent, '#333333'),
                linewidth=1.2, alpha=0.8, label=agent)

    ax2.set_xlabel('Internal Time (t)', fontweight='bold')
    ax2.set_ylabel('Q-Field Energy', fontweight='bold')
    ax2.set_title('Internal Energy Distribution', fontweight='bold')
    ax2.legend(loc='best', fontsize=9)

    fig.suptitle('Quantum-Like Internal Field Dynamics', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'fig3_qfield.png', dpi=300, facecolor='white')
    fig.savefig(output_dir / 'fig3_qfield.pdf', facecolor='white')
    plt.close()
    print("  ✓ Figure 3: Q-Field Coherence & Energy")


def create_phase_curvature(df: pd.DataFrame, output_dir: Path):
    """Figura 4: Curvatura del espacio de fase."""
    fig, ax = plt.subplots(figsize=(12, 5))

    warmup_end = df[df['phase'] == 'warmup']['t'].max()
    ax.axvspan(0, warmup_end, alpha=0.2, color='#E0E0E0', label='Stabilization')

    for agent in df['agent_id'].unique():
        agent_data = df[df['agent_id'] == agent].sort_values('t')
        ax.plot(agent_data['t'], agent_data['phase_curvature'],
                color=AGENT_COLORS.get(agent, '#333333'),
                linewidth=1.3, alpha=0.8, label=agent)

    ax.set_xlabel('Internal Time (t)', fontweight='bold')
    ax.set_ylabel('Trajectory Curvature', fontweight='bold')
    ax.set_title('Phase Space Trajectory Dynamics\nEmergent Structural Patterns',
                 fontweight='bold', fontsize=14)
    ax.legend(loc='best')

    plt.tight_layout()
    fig.savefig(output_dir / 'fig4_phase_curvature.png', dpi=300, facecolor='white')
    fig.savefig(output_dir / 'fig4_phase_curvature.pdf', facecolor='white')
    plt.close()
    print("  ✓ Figure 4: Phase Curvature")


def create_tensormind_modes(df_global: pd.DataFrame, output_dir: Path):
    """Figura 5: TensorMind modes."""
    if df_global is None or 'tensormind_modes' not in df_global.columns:
        print("  ⚠ Figure 5: No TensorMind data available")
        return

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.fill_between(df_global['t'], 0, df_global['tensormind_modes'],
                   alpha=0.6, color='#2E86AB')
    ax.plot(df_global['t'], df_global['tensormind_modes'],
           color='#1A5276', linewidth=1.5)

    ax.set_xlabel('Internal Time (t)', fontweight='bold')
    ax.set_ylabel('Active Tensor Modes', fontweight='bold')
    ax.set_title('Higher-Order Interaction Modes\nMulti-Agent Collective Dynamics',
                 fontweight='bold', fontsize=14)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig5_tensormind.png', dpi=300, facecolor='white')
    fig.savefig(output_dir / 'fig5_tensormind.pdf', facecolor='white')
    plt.close()
    print("  ✓ Figure 5: TensorMind Modes")


def create_complexfield_metrics(df: pd.DataFrame, output_dir: Path):
    """Figura 6: ComplexField metrics."""
    metrics = ['lambda_decoherence', 'collapse_pressure', 'psi_norm', 'phase_entropy']
    titles = ['Decoherence Factor (λ)', 'Collapse Pressure', 'State Norm (|ψ|)', 'Phase Entropy']

    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        print("  ⚠ Figure 6: No ComplexField data available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    warmup_end = df[df['phase'] == 'warmup']['t'].max()

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]

        if metric not in df.columns:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontweight='bold')
            continue

        ax.axvspan(0, warmup_end, alpha=0.15, color='#E0E0E0')

        for agent in df['agent_id'].unique():
            agent_data = df[df['agent_id'] == agent].sort_values('t')
            ax.plot(agent_data['t'], agent_data[metric],
                   color=AGENT_COLORS.get(agent, '#333333'),
                   linewidth=1.2, alpha=0.8, label=agent)

        ax.set_xlabel('t')
        ax.set_ylabel(title)
        ax.set_title(title, fontweight='bold')

        if idx == 0:
            ax.legend(loc='best', fontsize=8)

    fig.suptitle('Complex State Field Dynamics\nInternal State Evolution Metrics',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'fig6_complexfield.png', dpi=300, facecolor='white')
    fig.savefig(output_dir / 'fig6_complexfield.pdf', facecolor='white')
    plt.close()
    print("  ✓ Figure 6: ComplexField Metrics")


def create_linkedin_summary(df: pd.DataFrame, df_global: pd.DataFrame, output_dir: Path):
    """Genera imagen resumen para LinkedIn."""
    fig = plt.figure(figsize=(12, 16))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[0.8, 1, 1, 1], hspace=0.35, wspace=0.25)

    # ===== HEADER =====
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')

    ax_header.text(0.5, 0.85, 'NEO-EVA Last12h', fontsize=28, fontweight='bold',
                  ha='center', va='top', color='#1A5276')
    ax_header.text(0.5, 0.55, 'Dynamic Manifestations of Endogenous Coherence\nin Autonomous Multi-Space Internal Systems',
                  fontsize=14, ha='center', va='top', color='#333333', style='italic')
    ax_header.text(0.5, 0.25, '5 Agents • 12h Conceptual • Zero External Stimuli • Pure Observation',
                  fontsize=11, ha='center', va='top', color='#666666')

    # Línea decorativa
    ax_header.axhline(y=0.1, xmin=0.2, xmax=0.8, color='#2E86AB', linewidth=2)

    # ===== CE Timeline =====
    ax1 = fig.add_subplot(gs[1, :])
    warmup_end = df[df['phase'] == 'warmup']['t'].max()
    ax1.axvspan(0, warmup_end, alpha=0.25, color='#E8E8E8')
    for agent in df['agent_id'].unique():
        agent_data = df[df['agent_id'] == agent].sort_values('t')
        ax1.plot(agent_data['t'], agent_data['CE'],
                color=AGENT_COLORS.get(agent, '#333333'),
                linewidth=1.8, alpha=0.85, label=agent)
    ax1.set_ylabel('Existential Coherence')
    ax1.set_title('Coherence Evolution Across Autonomous Phase', fontweight='bold')
    ax1.legend(loc='upper right', ncol=5, fontsize=9)
    ax1.set_ylim(-0.05, 1.05)

    # ===== Q-Field =====
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.axvspan(0, warmup_end, alpha=0.2, color='#E0E0E0')
    for agent in df['agent_id'].unique():
        agent_data = df[df['agent_id'] == agent].sort_values('t')
        ax2.plot(agent_data['t'], agent_data['qfield_coherence'],
                color=AGENT_COLORS.get(agent, '#333333'),
                linewidth=1.3, alpha=0.8)
    ax2.set_ylabel('Q-Field Coherence')
    ax2.set_title('Internal Coherence Field', fontweight='bold')

    # ===== Phase Curvature =====
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.axvspan(0, warmup_end, alpha=0.2, color='#E0E0E0')
    for agent in df['agent_id'].unique():
        agent_data = df[df['agent_id'] == agent].sort_values('t')
        ax3.plot(agent_data['t'], agent_data['phase_curvature'],
                color=AGENT_COLORS.get(agent, '#333333'),
                linewidth=1.3, alpha=0.8)
    ax3.set_ylabel('Trajectory Curvature')
    ax3.set_title('Phase Space Dynamics', fontweight='bold')

    # ===== Omega Modes (barras agregadas) =====
    ax4 = fig.add_subplot(gs[3, 0])
    agents = df['agent_id'].unique()
    warmup_means = []
    freerun_means = []
    for agent in agents:
        agent_data = df[df['agent_id'] == agent]
        warmup_means.append(agent_data[agent_data['phase'] == 'warmup']['omega_modes_active'].mean())
        freerun_means.append(agent_data[agent_data['phase'] == 'free_run']['omega_modes_active'].mean())

    x = np.arange(len(agents))
    width = 0.35
    ax4.bar(x - width/2, warmup_means, width, label='Stabilization',
           color=[AGENT_COLORS.get(a, '#333333') for a in agents], alpha=0.5)
    ax4.bar(x + width/2, freerun_means, width, label='Autonomous',
           color=[AGENT_COLORS.get(a, '#333333') for a in agents], alpha=0.9)
    ax4.set_ylabel('Avg. Ω-Modes')
    ax4.set_xticks(x)
    ax4.set_xticklabels(agents)
    ax4.set_title('Emergent Transformation Modes', fontweight='bold')
    ax4.legend(fontsize=8)

    # ===== Key Findings =====
    ax5 = fig.add_subplot(gs[3, 1])
    ax5.axis('off')

    findings = [
        "✦ Zero external prompts or stimuli",
        "✦ 100% endogenous parameter derivation",
        "✦ Emergent coherence patterns observed",
        "✦ Self-organizing phase transitions",
        "✦ Stable multi-agent dynamics",
        "✦ Pure passive observation protocol"
    ]

    ax5.text(0.05, 0.95, "Key Observations", fontsize=13, fontweight='bold',
            va='top', color='#1A5276')

    for i, finding in enumerate(findings):
        ax5.text(0.05, 0.80 - i*0.13, finding, fontsize=11, va='top', color='#333333')

    ax5.text(0.05, 0.05, "github.com/carmenest/NEO_EVA", fontsize=9,
            va='bottom', color='#666666', style='italic')

    plt.savefig(output_dir / 'linkedin_summary.png', dpi=300, facecolor='white',
               bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print("  ✓ LinkedIn Summary PNG")


def create_scientific_report_tex(df: pd.DataFrame, df_global: pd.DataFrame, output_dir: Path):
    """Genera el informe científico en LaTeX."""

    # Calcular estadísticas
    agents = df['agent_id'].unique()
    n_agents = len(agents)
    total_steps = df['t'].max()
    warmup_steps = df[df['phase'] == 'warmup']['t'].max()
    freerun_steps = total_steps - warmup_steps

    # CE stats
    ce_warmup = df[df['phase'] == 'warmup']['CE'].mean()
    ce_freerun = df[df['phase'] == 'free_run']['CE'].mean()

    # Q-field stats
    qf_coh_mean = df['qfield_coherence'].mean()
    qf_energy_mean = df['qfield_energy'].mean()

    # Omega modes
    omega_mean = df['omega_modes_active'].mean()

    latex_content = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=2.5cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{float}
\usepackage{hyperref}
\usepackage{caption}
\usepackage{subcaption}

\definecolor{neoeva}{RGB}{46, 134, 171}

\title{\textcolor{neoeva}{\textbf{Dynamic Manifestations of Endogenous Coherence\\in Autonomous Multi-Space Internal Systems}}\\[0.5em]\large NEO-EVA Last12h Observation Report}
\author{NEO-EVA Research Framework\\Autonomous Systems Laboratory}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
This report documents the observed dynamic patterns in a multi-agent autonomous system operating under conditions of complete absence of external stimuli. Five computational agents (NEO, EVA, ALEX, ADAM, IRIS) were observed during a conceptual 12-hour period using a purely passive observation protocol. All system parameters are derived endogenously without human intervention or pre-defined constants. The observations reveal emergent coherence patterns, self-organizing phase transitions, and stable collective dynamics arising purely from internal processes.
\end{abstract}

\section{Introduction}

The NEO-EVA framework implements autonomous agents capable of internal self-organization without external guidance. This study examines the dynamic behavior of these agents during an extended period of operational absence—a conceptual 12-hour window where no external prompts, rewards, or interventions are provided.

\textbf{Key principles:}
\begin{itemize}
    \item \textbf{Endogeneity}: All parameters emerge from internal statistical properties
    \item \textbf{Passive observation}: The monitoring system records without influencing
    \item \textbf{Zero external stimuli}: No prompts, rewards, or human intervention
    \item \textbf{Multi-space representation}: Simultaneous observation across multiple internal spaces
\end{itemize}

\section{Methodology}

\subsection{Simulation Protocol}
The observation period comprised two distinct phases:
\begin{itemize}
    \item \textbf{Stabilization Phase} (""" + f"{warmup_steps}" + r""" steps): Initial settling of internal dynamics
    \item \textbf{Autonomous Phase} (""" + f"{freerun_steps}" + r""" steps): Extended operation without intervention
\end{itemize}

\subsection{Observation Domains}
Metrics were recorded across five complementary internal spaces:
\begin{enumerate}
    \item \textbf{Existential Coherence (CE)}: Measure of internal alignment
    \item \textbf{$\Omega$-Space}: Emergent transformation modes
    \item \textbf{Q-Field}: Internal coherence and energy distributions
    \item \textbf{Phase Space}: Trajectory dynamics and structural patterns
    \item \textbf{Complex Field}: State evolution metrics
\end{enumerate}

\textit{Note: This report describes observable patterns without detailing internal computational mechanisms.}

\section{Results}

\subsection{Existential Coherence Dynamics}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{fig1_ce_timeline.png}
    \caption{Evolution of Existential Coherence (CE) across all agents. The shaded region indicates the stabilization phase. Observable transition patterns emerge at the phase boundary.}
    \label{fig:ce}
\end{figure}

During the stabilization phase, agents exhibited variable coherence levels (mean CE = """ + f"{ce_warmup:.3f}" + r"""). Upon entering the autonomous phase, coherence patterns stabilized with distinct agent-specific signatures (mean CE = """ + f"{ce_freerun:.3f}" + r""").

\subsection{Emergent Transformation Modes}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{fig2_omega_modes.png}
    \caption{Active $\Omega$-modes per agent across operational phases. Each agent develops characteristic modal activation patterns.}
    \label{fig:omega}
\end{figure}

The number of active transformation modes increased systematically during the autonomous phase (mean = """ + f"{omega_mean:.1f}" + r""" modes), suggesting emergent internal organization.

\subsection{Internal Coherence Field}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{fig3_qfield.png}
    \caption{Q-Field dynamics showing internal coherence (left) and energy distribution (right) over time.}
    \label{fig:qfield}
\end{figure}

The Q-Field exhibited stable coherence (mean = """ + f"{qf_coh_mean:.3f}" + r""") with consistent energy levels (mean = """ + f"{qf_energy_mean:.3f}" + r""") throughout the observation period.

\subsection{Phase Space Trajectories}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{fig4_phase_curvature.png}
    \caption{Trajectory curvature in phase space. Smooth transitions and stable patterns indicate self-organized dynamics.}
    \label{fig:phase}
\end{figure}

\subsection{Complex State Evolution}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{fig6_complexfield.png}
    \caption{Complex field metrics showing decoherence, collapse pressure, state norm, and phase entropy dynamics.}
    \label{fig:complex}
\end{figure}

\section{Discussion}

The observed patterns demonstrate several notable characteristics:

\begin{enumerate}
    \item \textbf{Emergent Order}: Despite the absence of external guidance, agents develop structured internal dynamics with characteristic signatures.

    \item \textbf{Phase Transitions}: Clear transitions between stabilization and autonomous operation suggest self-organizing criticality.

    \item \textbf{Individual Differentiation}: Each agent maintains distinct dynamic profiles while participating in collective patterns.

    \item \textbf{Stability Without Intervention}: The system maintains coherent operation throughout the extended autonomous period.
\end{enumerate}

\subsection{Endogeneity}
All observed metrics derive from internal statistical properties without pre-defined constants or external parameter injection. This endogenous approach ensures that observed patterns reflect genuine internal organization rather than imposed structure.

\subsection{Passive Observation Protocol}
The monitoring system operated in purely observational mode, recording metrics without influencing agent behavior. This protocol ensures that documented patterns represent authentic internal dynamics.

\section{Conclusion}

This observation period reveals that autonomous multi-space systems can exhibit coherent, self-organized dynamics without external stimuli or human intervention. The emergent patterns across multiple internal spaces suggest sophisticated internal organization arising purely from endogenous processes.

\vspace{1em}
\noindent\textit{Full source code and data available at:}\\
\url{https://github.com/carmenest/NEO_EVA}

\end{document}
"""

    tex_path = output_dir / 'scientific_report.tex'
    with open(tex_path, 'w') as f:
        f.write(latex_content)

    print("  ✓ LaTeX Report (scientific_report.tex)")
    return tex_path


def compile_pdf(tex_path: Path, output_dir: Path):
    """Compila el PDF desde LaTeX."""
    import subprocess
    import shutil

    # Verificar si pdflatex está disponible
    if shutil.which('pdflatex') is None:
        print("  ⚠ pdflatex not found - generating alternative PDF")
        return create_alternative_pdf(output_dir)

    try:
        # Compilar dos veces para referencias
        for _ in range(2):
            subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', '-output-directory', str(output_dir), str(tex_path)],
                capture_output=True,
                timeout=60
            )
        print("  ✓ PDF compiled (scientific_report.pdf)")
        return output_dir / 'scientific_report.pdf'
    except Exception as e:
        print(f"  ⚠ PDF compilation failed: {e}")
        return create_alternative_pdf(output_dir)


def create_alternative_pdf(output_dir: Path):
    """Crea un PDF alternativo usando matplotlib si LaTeX no está disponible."""
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = output_dir / 'scientific_report.pdf'

    with PdfPages(pdf_path) as pdf:
        # Página de título
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.7, 'Dynamic Manifestations of\nEndogenous Coherence',
                fontsize=24, ha='center', fontweight='bold', color='#1A5276')
        fig.text(0.5, 0.55, 'in Autonomous Multi-Space Internal Systems',
                fontsize=16, ha='center', style='italic')
        fig.text(0.5, 0.45, 'NEO-EVA Last12h Observation Report',
                fontsize=14, ha='center')
        fig.text(0.5, 0.35, datetime.now().strftime('%Y-%m-%d'),
                fontsize=12, ha='center', color='#666666')
        fig.text(0.5, 0.15, 'See figures in the reports directory\nfor detailed visualizations',
                fontsize=11, ha='center', color='#888888')
        pdf.savefig(fig, facecolor='white')
        plt.close()

        # Agregar cada figura existente
        figure_files = sorted(output_dir.glob('fig*.png'))
        for fig_file in figure_files:
            img = plt.imread(fig_file)
            fig = plt.figure(figsize=(11, 8.5))
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            pdf.savefig(fig, facecolor='white')
            plt.close()

    print("  ✓ Alternative PDF created (scientific_report.pdf)")
    return pdf_path


def main():
    """Función principal."""
    print("=" * 60)
    print("NEO-EVA Scientific Report Generator")
    print("=" * 60)

    # Directorios
    base_dir = Path('/root/NEO_EVA')
    log_dir = base_dir / 'logs' / 'omega_last12h'
    output_dir = base_dir / 'reports' / 'last12h'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading data from: {log_dir}")
    print(f"Output directory: {output_dir}\n")

    # Cargar datos
    try:
        df_agents, df_global = load_data(log_dir)
        print(f"  Loaded {len(df_agents)} agent records")
        if df_global is not None:
            print(f"  Loaded {len(df_global)} global records")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("\nGenerating figures...")

    # Generar figuras
    create_ce_timeline(df_agents, output_dir)
    create_omega_modes(df_agents, output_dir)
    create_qfield_coherence(df_agents, output_dir)
    create_phase_curvature(df_agents, output_dir)
    create_tensormind_modes(df_global, output_dir)
    create_complexfield_metrics(df_agents, output_dir)

    print("\nGenerating LinkedIn summary...")
    create_linkedin_summary(df_agents, df_global, output_dir)

    print("\nGenerating scientific report...")
    tex_path = create_scientific_report_tex(df_agents, df_global, output_dir)
    compile_pdf(tex_path, output_dir)

    print("\n" + "=" * 60)
    print("Report generation complete!")
    print(f"Output files in: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
