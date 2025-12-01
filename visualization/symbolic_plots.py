#!/usr/bin/env python3
"""
Symbolic Evolution Plots
========================

Genera visualizaciones automáticas de la evolución simbólica:
- Evolución temporal de métricas
- Comparación entre agentes
- Radar charts de capacidades
- Heatmaps de correlaciones

Para paper Nature ML / NEJM-AI.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

import sys
sys.path.insert(0, '/root/NEO_EVA')


class SymbolicPlotter:
    """Genera plots de evolución simbólica."""

    COLORS = {
        'NEO': '#2E86AB',
        'EVA': '#A23B72',
        'ALEX': '#F18F01',
        'ADAM': '#C73E1D',
        'IRIS': '#3B1F2B'
    }

    def __init__(self, output_dir: str = '/root/NEO_EVA/visualization/figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_temporal_evolution(
        self,
        step_metrics: Dict[str, List[float]],
        agent_ids: List[str],
        title: str = "Symbolic Evolution Over Time",
        save_name: str = "temporal_evolution.png"
    ) -> str:
        """
        Plot de evolución temporal de métricas globales y por agente.
        """
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

        # 1. Métricas globales
        ax1 = fig.add_subplot(gs[0, :])
        steps = range(1, len(step_metrics['global_sym_x']) + 1)

        ax1.plot(steps, step_metrics['global_sym_x'], 'b-', linewidth=2, label='SYM-X')
        ax1.plot(steps, step_metrics['global_cf'], 'g-', linewidth=2, label='CF (Counterfactual)')
        ax1.plot(steps, step_metrics['global_ci'], 'r-', linewidth=2, label='CI (Causality)')

        ax1.axhline(y=0.62, color='g', linestyle='--', alpha=0.5, label='CF Target (0.62)')
        ax1.axhline(y=0.60, color='r', linestyle='--', alpha=0.5, label='CI Target (0.60)')

        ax1.set_xlabel('Training Step', fontsize=11)
        ax1.set_ylabel('Score', fontsize=11)
        ax1.set_title('Global Metrics Evolution', fontsize=12, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # 2. SymScore por agente
        ax2 = fig.add_subplot(gs[1, 0])
        for aid in agent_ids:
            key = f'{aid}_sym_score'
            if key in step_metrics:
                ax2.plot(steps, step_metrics[key], color=self.COLORS.get(aid, 'gray'),
                        linewidth=1.5, label=aid, alpha=0.8)

        ax2.set_xlabel('Training Step', fontsize=11)
        ax2.set_ylabel('SymScore', fontsize=11)
        ax2.set_title('Symbol Quality by Agent', fontsize=12, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(True, alpha=0.3)

        # 3. Richness por agente
        ax3 = fig.add_subplot(gs[1, 1])
        for aid in agent_ids:
            key = f'{aid}_richness'
            if key in step_metrics:
                ax3.plot(steps, step_metrics[key], color=self.COLORS.get(aid, 'gray'),
                        linewidth=1.5, label=aid, alpha=0.8)

        ax3.set_xlabel('Training Step', fontsize=11)
        ax3.set_ylabel('Richness (|Σ|/√t)', fontsize=11)
        ax3.set_title('Symbolic Richness by Agent', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3)

        # 4. CF por agente
        ax4 = fig.add_subplot(gs[2, 0])
        for aid in agent_ids:
            key = f'{aid}_cf_score'
            if key in step_metrics:
                ax4.plot(steps, step_metrics[key], color=self.COLORS.get(aid, 'gray'),
                        linewidth=1.5, label=aid, alpha=0.8)

        ax4.axhline(y=0.62, color='black', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Training Step', fontsize=11)
        ax4.set_ylabel('CF Score', fontsize=11)
        ax4.set_title('Counterfactual Strength by Agent', fontsize=12, fontweight='bold')
        ax4.legend(loc='lower right', fontsize=9)
        ax4.grid(True, alpha=0.3)

        # 5. CI por agente
        ax5 = fig.add_subplot(gs[2, 1])
        for aid in agent_ids:
            key = f'{aid}_ci_score'
            if key in step_metrics:
                ax5.plot(steps, step_metrics[key], color=self.COLORS.get(aid, 'gray'),
                        linewidth=1.5, label=aid, alpha=0.8)

        ax5.axhline(y=0.60, color='black', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Training Step', fontsize=11)
        ax5.set_ylabel('CI Score', fontsize=11)
        ax5.set_title('Internal Causality by Agent', fontsize=12, fontweight='bold')
        ax5.legend(loc='lower right', fontsize=9)
        ax5.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        return str(save_path)

    def plot_radar_comparison(
        self,
        agent_metrics: Dict[str, Dict[str, float]],
        title: str = "Multi-Agent Capability Comparison",
        save_name: str = "radar_comparison.png"
    ) -> str:
        """
        Radar chart comparando capacidades de agentes.
        """
        categories = ['SymScore', 'CF', 'CI', 'Richness', 'Grounding\nWorld', 'Grounding\nSocial']
        n_cats = len(categories)

        # Ángulos para el radar
        angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
        angles += angles[:1]  # Cerrar el polígono

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        for aid, metrics in agent_metrics.items():
            values = [
                metrics.get('sym_score', 0),
                metrics.get('cf_score', 0),
                metrics.get('ci_score', 0),
                min(metrics.get('richness', 0), 1.0),
                metrics.get('grounding_world', 0),
                metrics.get('grounding_social', 0)
            ]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2, label=aid,
                   color=self.COLORS.get(aid, 'gray'), alpha=0.8)
            ax.fill(angles, values, alpha=0.15, color=self.COLORS.get(aid, 'gray'))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)

        plt.title(title, fontsize=14, fontweight='bold', pad=20)

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        return str(save_path)

    def plot_symbol_emergence(
        self,
        step_metrics: Dict[str, List[float]],
        agent_ids: List[str],
        title: str = "Symbol Emergence Dynamics",
        save_name: str = "symbol_emergence.png"
    ) -> str:
        """
        Plot de emergencia simbólica: número de símbolos vs tiempo.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        steps = range(1, len(step_metrics.get('NEO_n_symbols', [])) + 1)

        # 1. Número de símbolos
        ax1 = axes[0, 0]
        for aid in agent_ids:
            key = f'{aid}_n_symbols'
            if key in step_metrics:
                ax1.plot(steps, step_metrics[key], color=self.COLORS.get(aid, 'gray'),
                        linewidth=1.5, label=aid, alpha=0.8)

        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Number of Symbols')
        ax1.set_title('Symbol Count Over Time', fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # 2. Barplot de símbolos finales
        ax2 = axes[0, 1]
        final_symbols = {aid: step_metrics.get(f'{aid}_n_symbols', [0])[-1] for aid in agent_ids}
        bars = ax2.bar(final_symbols.keys(), final_symbols.values(),
                      color=[self.COLORS.get(aid, 'gray') for aid in agent_ids])
        ax2.set_ylabel('Final Symbol Count')
        ax2.set_title('Final Symbols per Agent', fontweight='bold')
        for bar, val in zip(bars, final_symbols.values()):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=10)

        # 3. Correlación SymScore vs Richness
        ax3 = axes[1, 0]
        for aid in agent_ids:
            sym_key = f'{aid}_sym_score'
            rich_key = f'{aid}_richness'
            if sym_key in step_metrics and rich_key in step_metrics:
                ax3.scatter(step_metrics[rich_key], step_metrics[sym_key],
                           c=self.COLORS.get(aid, 'gray'), alpha=0.5, s=10, label=aid)

        ax3.set_xlabel('Richness')
        ax3.set_ylabel('SymScore')
        ax3.set_title('SymScore vs Richness Correlation', fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        # 4. Boxplot de scores finales
        ax4 = axes[1, 1]
        data = []
        labels = []
        for aid in agent_ids:
            for metric in ['sym_score', 'cf_score', 'ci_score']:
                key = f'{aid}_{metric}'
                if key in step_metrics:
                    data.append(step_metrics[key][-100:])  # Últimos 100 steps
                    labels.append(f'{aid[:3]}\n{metric[:3].upper()}')

        if data:
            bp = ax4.boxplot(data, labels=labels, patch_artist=True)
            colors = [self.COLORS.get(aid, 'gray') for aid in agent_ids for _ in range(3)]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

        ax4.set_ylabel('Score')
        ax4.set_title('Score Distributions (Last 100 Steps)', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        return str(save_path)

    def plot_cf_ci_analysis(
        self,
        step_metrics: Dict[str, List[float]],
        title: str = "Counterfactual & Causality Analysis",
        save_name: str = "cf_ci_analysis.png"
    ) -> str:
        """
        Análisis detallado de CF y CI.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        steps = range(1, len(step_metrics['global_cf']) + 1)

        # 1. CF vs CI scatter
        ax1 = axes[0, 0]
        ax1.scatter(step_metrics['global_cf'], step_metrics['global_ci'],
                   c=list(steps), cmap='viridis', alpha=0.6, s=20)
        ax1.axvline(x=0.62, color='green', linestyle='--', alpha=0.5, label='CF target')
        ax1.axhline(y=0.60, color='red', linestyle='--', alpha=0.5, label='CI target')
        ax1.set_xlabel('CF Score')
        ax1.set_ylabel('CI Score')
        ax1.set_title('CF vs CI Trajectory', fontweight='bold')
        ax1.legend()
        cbar = plt.colorbar(ax1.collections[0], ax=ax1)
        cbar.set_label('Training Step')

        # 2. Rolling average CF
        ax2 = axes[0, 1]
        window = 50
        cf_smooth = np.convolve(step_metrics['global_cf'],
                                np.ones(window)/window, mode='valid')
        ci_smooth = np.convolve(step_metrics['global_ci'],
                                np.ones(window)/window, mode='valid')
        smooth_steps = range(window, len(step_metrics['global_cf']) + 1)

        ax2.plot(smooth_steps, cf_smooth, 'g-', linewidth=2, label='CF (smoothed)')
        ax2.plot(smooth_steps, ci_smooth, 'r-', linewidth=2, label='CI (smoothed)')
        ax2.fill_between(smooth_steps, cf_smooth, alpha=0.2, color='green')
        ax2.fill_between(smooth_steps, ci_smooth, alpha=0.2, color='red')
        ax2.axhline(y=0.62, color='green', linestyle='--', alpha=0.5)
        ax2.axhline(y=0.60, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Score')
        ax2.set_title(f'Smoothed Metrics (window={window})', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Improvement rate
        ax3 = axes[1, 0]
        cf_diff = np.diff(step_metrics['global_cf'])
        ci_diff = np.diff(step_metrics['global_ci'])
        ax3.plot(cf_diff, 'g-', alpha=0.5, linewidth=0.5, label='CF δ')
        ax3.plot(ci_diff, 'r-', alpha=0.5, linewidth=0.5, label='CI δ')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Score Change')
        ax3.set_title('Improvement Rate', fontweight='bold')
        ax3.legend()

        # 4. Final comparison bar
        ax4 = axes[1, 1]
        final_cf = step_metrics['global_cf'][-1]
        final_ci = step_metrics['global_ci'][-1]
        bars = ax4.bar(['CF', 'CI'], [final_cf, final_ci],
                      color=['green', 'red'], alpha=0.7)
        ax4.axhline(y=0.62, color='green', linestyle='--', alpha=0.5, label='CF target')
        ax4.axhline(y=0.60, color='red', linestyle='--', alpha=0.5, label='CI target')
        ax4.set_ylabel('Final Score')
        ax4.set_title('Final CF & CI Scores', fontweight='bold')
        ax4.set_ylim(0, 1)
        for bar, val in zip(bars, [final_cf, final_ci]):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax4.legend()

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        return str(save_path)

    def generate_all_plots(
        self,
        results_path: str
    ) -> Dict[str, str]:
        """
        Genera todos los plots desde un archivo de resultados.
        """
        with open(results_path, 'r') as f:
            results = json.load(f)

        step_metrics = results['step_metrics']
        agent_ids = results['agent_ids']

        # Construir agent_metrics para radar
        agent_metrics = {}
        for aid in agent_ids:
            agent_metrics[aid] = {
                'sym_score': step_metrics.get(f'{aid}_sym_score', [0])[-1],
                'cf_score': step_metrics.get(f'{aid}_cf_score', [0])[-1],
                'ci_score': step_metrics.get(f'{aid}_ci_score', [0])[-1],
                'richness': step_metrics.get(f'{aid}_richness', [0])[-1],
                'grounding_world': 0.5,  # Placeholder
                'grounding_social': 0.5,  # Placeholder
            }

        plots = {}

        plots['temporal'] = self.plot_temporal_evolution(step_metrics, agent_ids)
        plots['radar'] = self.plot_radar_comparison(agent_metrics)
        plots['emergence'] = self.plot_symbol_emergence(step_metrics, agent_ids)
        plots['cf_ci'] = self.plot_cf_ci_analysis(step_metrics)

        return plots


def main():
    """Genera plots de demostración."""
    np.random.seed(42)

    plotter = SymbolicPlotter()

    # Generar datos de ejemplo
    n_steps = 1000
    agent_ids = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']

    step_metrics = {
        'global_sym_x': list(0.3 + 0.4 * (1 - np.exp(-np.arange(n_steps) / 300)) + np.random.randn(n_steps) * 0.05),
        'global_cf': list(0.4 + 0.25 * (1 - np.exp(-np.arange(n_steps) / 400)) + np.random.randn(n_steps) * 0.05),
        'global_ci': list(0.35 + 0.3 * (1 - np.exp(-np.arange(n_steps) / 350)) + np.random.randn(n_steps) * 0.05),
    }

    for aid in agent_ids:
        offset = agent_ids.index(aid) * 0.05
        step_metrics[f'{aid}_sym_score'] = list(0.3 + offset + 0.4 * (1 - np.exp(-np.arange(n_steps) / 300)) + np.random.randn(n_steps) * 0.03)
        step_metrics[f'{aid}_cf_score'] = list(0.4 + offset + 0.25 * (1 - np.exp(-np.arange(n_steps) / 400)) + np.random.randn(n_steps) * 0.03)
        step_metrics[f'{aid}_ci_score'] = list(0.35 + offset + 0.3 * (1 - np.exp(-np.arange(n_steps) / 350)) + np.random.randn(n_steps) * 0.03)
        step_metrics[f'{aid}_n_symbols'] = list(np.cumsum(np.random.poisson(0.1, n_steps)))
        step_metrics[f'{aid}_richness'] = list(np.array(step_metrics[f'{aid}_n_symbols']) / np.sqrt(np.arange(1, n_steps + 1)))

    agent_metrics = {
        aid: {
            'sym_score': step_metrics[f'{aid}_sym_score'][-1],
            'cf_score': step_metrics[f'{aid}_cf_score'][-1],
            'ci_score': step_metrics[f'{aid}_ci_score'][-1],
            'richness': step_metrics[f'{aid}_richness'][-1],
            'grounding_world': 0.5 + np.random.rand() * 0.3,
            'grounding_social': 0.4 + np.random.rand() * 0.3,
        }
        for aid in agent_ids
    }

    print("Generating plots...")

    p1 = plotter.plot_temporal_evolution(step_metrics, agent_ids)
    print(f"  Temporal evolution: {p1}")

    p2 = plotter.plot_radar_comparison(agent_metrics)
    print(f"  Radar comparison: {p2}")

    p3 = plotter.plot_symbol_emergence(step_metrics, agent_ids)
    print(f"  Symbol emergence: {p3}")

    p4 = plotter.plot_cf_ci_analysis(step_metrics)
    print(f"  CF/CI analysis: {p4}")

    print("\nAll plots generated!")


if __name__ == "__main__":
    main()
