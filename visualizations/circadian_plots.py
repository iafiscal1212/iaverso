"""
Circadian Plots: Visualizaciones del ciclo circadiano
======================================================

Genera gráficos a partir de los logs JSON/CSV del sistema circadiano.
"""

import numpy as np
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import sys
sys.path.insert(0, '/root/NEO_EVA')

# Intentar importar matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Backend sin display
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib no disponible. Instalando...")


def load_json_log(filepath: str) -> Dict[str, Any]:
    """Carga log JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_csv_log(filepath: str) -> List[Dict]:
    """Carga log CSV."""
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


class CircadianVisualizer:
    """Visualizador de datos circadianos."""

    # Colores por fase
    PHASE_COLORS = {
        'wake': '#FFD700',      # Dorado - activo
        'rest': '#4169E1',      # Azul real - descanso
        'dream': '#9370DB',     # Púrpura medio - sueño
        'liminal': '#20B2AA'    # Verde azulado - transición
    }

    # Colores por agente
    AGENT_COLORS = {
        'NEO': '#FF6B6B',       # Rojo coral
        'EVA': '#4ECDC4',       # Turquesa
        'ALEX': '#45B7D1',      # Azul cielo
        'ADAM': '#96CEB4',      # Verde menta
        'IRIS': '#DDA0DD'       # Ciruela
    }

    def __init__(self, data: Dict[str, Any]):
        """
        Args:
            data: Datos del log JSON
        """
        self.data = data
        self.snapshots = data['snapshots']
        self.events = data.get('events', [])
        self.metadata = data['metadata']
        self.agents = self.metadata['agents']

        # Organizar por agente
        self.agent_data = {agent: [] for agent in self.agents}
        for s in self.snapshots:
            self.agent_data[s['agent_id']].append(s)

    def plot_energy_stress_timeline(self, output_path: str = None):
        """Gráfico de energía y estrés en el tiempo."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        ax_energy = axes[0]
        ax_stress = axes[1]

        for agent_id in self.agents:
            agent_snaps = self.agent_data[agent_id]
            t = [s['t'] for s in agent_snaps]
            energy = [float(s['energy']) for s in agent_snaps]
            stress = [float(s['stress']) for s in agent_snaps]

            color = self.AGENT_COLORS.get(agent_id, '#888888')

            ax_energy.plot(t, energy, label=agent_id, color=color, alpha=0.8, linewidth=1.5)
            ax_stress.plot(t, stress, label=agent_id, color=color, alpha=0.8, linewidth=1.5)

        ax_energy.set_ylabel('Energía', fontsize=12)
        ax_energy.set_ylim(0, 1.05)
        ax_energy.legend(loc='upper right', ncol=5)
        ax_energy.grid(True, alpha=0.3)
        ax_energy.set_title('Energía y Estrés de Agentes en el Tiempo', fontsize=14, fontweight='bold')

        ax_stress.set_ylabel('Estrés', fontsize=12)
        ax_stress.set_xlabel('Tiempo (pasos)', fontsize=12)
        ax_stress.set_ylim(0, 1.05)
        ax_stress.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f'  Guardado: {output_path}')

        plt.close()

    def plot_phase_distribution(self, output_path: str = None):
        """Gráfico de distribución de fases por agente."""
        fig, ax = plt.subplots(figsize=(12, 6))

        phases = ['wake', 'rest', 'dream', 'liminal']
        x = np.arange(len(self.agents))
        width = 0.2

        phase_counts = {agent: {p: 0 for p in phases} for agent in self.agents}

        for s in self.snapshots:
            phase_counts[s['agent_id']][s['phase']] += 1

        for i, phase in enumerate(phases):
            counts = [phase_counts[agent][phase] for agent in self.agents]
            total = [sum(phase_counts[agent].values()) for agent in self.agents]
            percentages = [c / t * 100 if t > 0 else 0 for c, t in zip(counts, total)]

            bars = ax.bar(x + i * width, percentages, width,
                         label=phase.upper(), color=self.PHASE_COLORS[phase], alpha=0.85)

        ax.set_ylabel('% del Tiempo', fontsize=12)
        ax.set_xlabel('Agente', fontsize=12)
        ax.set_title('Distribución de Tiempo por Fase Circadiana', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(self.agents)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f'  Guardado: {output_path}')

        plt.close()

    def plot_cycles_comparison(self, output_path: str = None):
        """Gráfico de ciclos completados por agente."""
        fig, ax = plt.subplots(figsize=(10, 6))

        final_cycles = []
        colors = []

        for agent_id in self.agents:
            agent_snaps = self.agent_data[agent_id]
            if agent_snaps:
                final_cycles.append(agent_snaps[-1]['cycles_completed'])
            else:
                final_cycles.append(0)
            colors.append(self.AGENT_COLORS.get(agent_id, '#888888'))

        bars = ax.bar(self.agents, final_cycles, color=colors, alpha=0.85, edgecolor='black')

        # Añadir valores encima de las barras
        for bar, cycles in zip(bars, final_cycles):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   str(cycles), ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_ylabel('Ciclos Completados', fontsize=12)
        ax.set_xlabel('Agente', fontsize=12)
        ax.set_title('Ciclos Circadianos Completados', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f'  Guardado: {output_path}')

        plt.close()

    def plot_symbolic_life(self, output_path: str = None):
        """Gráfico de vida simbólica en el tiempo."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        ax_index = axes[0]
        ax_symbols = axes[1]

        for agent_id in self.agents:
            agent_snaps = self.agent_data[agent_id]
            t = [s['t'] for s in agent_snaps]
            sli = [float(s['symbolic_life_index']) for s in agent_snaps]
            symbols = [int(s['total_symbols']) for s in agent_snaps]

            color = self.AGENT_COLORS.get(agent_id, '#888888')

            ax_index.plot(t, sli, label=agent_id, color=color, alpha=0.8, linewidth=1.5)
            ax_symbols.plot(t, symbols, label=agent_id, color=color, alpha=0.8, linewidth=1.5)

        ax_index.set_ylabel('Índice Vida Simbólica', fontsize=12)
        ax_index.set_ylim(0, 1)
        ax_index.legend(loc='upper left', ncol=5)
        ax_index.grid(True, alpha=0.3)
        ax_index.set_title('Evolución de la Vida Simbólica', fontsize=14, fontweight='bold')

        ax_symbols.set_ylabel('Total Símbolos', fontsize=12)
        ax_symbols.set_xlabel('Tiempo (pasos)', fontsize=12)
        ax_symbols.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f'  Guardado: {output_path}')

        plt.close()

    def plot_phase_timeline(self, output_path: str = None):
        """Gráfico de fases en el tiempo (tipo Gantt)."""
        fig, ax = plt.subplots(figsize=(14, 6))

        for i, agent_id in enumerate(self.agents):
            agent_snaps = self.agent_data[agent_id]

            current_phase = None
            phase_start = 0

            for s in agent_snaps:
                if s['phase'] != current_phase:
                    if current_phase is not None:
                        # Dibujar fase anterior
                        color = self.PHASE_COLORS.get(current_phase, '#888888')
                        ax.barh(i, s['t'] - phase_start, left=phase_start,
                               color=color, height=0.6, alpha=0.8)
                    current_phase = s['phase']
                    phase_start = s['t']

            # Última fase
            if current_phase and agent_snaps:
                color = self.PHASE_COLORS.get(current_phase, '#888888')
                ax.barh(i, agent_snaps[-1]['t'] - phase_start + 1, left=phase_start,
                       color=color, height=0.6, alpha=0.8)

        ax.set_yticks(range(len(self.agents)))
        ax.set_yticklabels(self.agents)
        ax.set_xlabel('Tiempo (pasos)', fontsize=12)
        ax.set_title('Timeline de Fases Circadianas', fontsize=14, fontweight='bold')

        # Leyenda
        legend_patches = [
            mpatches.Patch(color=color, label=phase.upper())
            for phase, color in self.PHASE_COLORS.items()
        ]
        ax.legend(handles=legend_patches, loc='upper right', ncol=4)

        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f'  Guardado: {output_path}')

        plt.close()

    def plot_quality_metrics(self, output_path: str = None):
        """Gráfico de métricas de calidad (wake, rest, dream)."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metrics = ['wake_quality', 'rest_depth', 'dream_vividness']
        titles = ['Calidad WAKE', 'Profundidad REST', 'Viveza DREAM']

        for ax, metric, title in zip(axes, metrics, titles):
            final_values = []
            colors = []

            for agent_id in self.agents:
                agent_snaps = self.agent_data[agent_id]
                if agent_snaps:
                    final_values.append(float(agent_snaps[-1][metric]))
                else:
                    final_values.append(0)
                colors.append(self.AGENT_COLORS.get(agent_id, '#888888'))

            bars = ax.bar(self.agents, final_values, color=colors, alpha=0.85, edgecolor='black')

            ax.set_ylabel(title, fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')

            # Valores encima
            for bar, val in zip(bars, final_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f'  Guardado: {output_path}')

        plt.close()

    def plot_archetypes(self, output_path: str = None):
        """Gráfico de arquetipos por agente."""
        fig, ax = plt.subplots(figsize=(10, 6))

        archetypes = {}
        for agent_id in self.agents:
            agent_snaps = self.agent_data[agent_id]
            if agent_snaps:
                arch = agent_snaps[-1]['dominant_archetype']
                strength = float(agent_snaps[-1]['archetype_strength'])
                archetypes[agent_id] = (arch, strength)

        # Colores por arquetipo
        archetype_colors = {
            'hero': '#FF4444',
            'shadow': '#444444',
            'anima': '#FF69B4',
            'self': '#FFD700',
            'trickster': '#00CED1',
            'none': '#CCCCCC'
        }

        agents_list = list(archetypes.keys())
        strengths = [archetypes[a][1] for a in agents_list]
        colors = [archetype_colors.get(archetypes[a][0], '#888888') for a in agents_list]
        labels = [archetypes[a][0].upper() for a in agents_list]

        bars = ax.bar(agents_list, strengths, color=colors, alpha=0.85, edgecolor='black')

        # Etiquetas
        for bar, label, strength in zip(bars, labels, strengths):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{label}\n({strength:.2f})', ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('Fuerza del Arquetipo', fontsize=12)
        ax.set_xlabel('Agente', fontsize=12)
        ax.set_title('Arquetipos Dominantes por Agente', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(strengths) * 1.3 if strengths else 1)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f'  Guardado: {output_path}')

        plt.close()

    def plot_events_distribution(self, output_path: str = None):
        """Gráfico de distribución de eventos."""
        if not self.events:
            print("  No hay eventos para graficar")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Por tipo de evento
        event_types = {}
        for e in self.events:
            et = e['event_type']
            event_types[et] = event_types.get(et, 0) + 1

        ax1 = axes[0]
        types = list(event_types.keys())
        counts = list(event_types.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(types)))

        ax1.pie(counts, labels=types, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Distribución por Tipo de Evento', fontsize=12, fontweight='bold')

        # Por agente
        agent_events = {agent: 0 for agent in self.agents}
        for e in self.events:
            agent_events[e['agent_id']] += 1

        ax2 = axes[1]
        agents = list(agent_events.keys())
        counts = list(agent_events.values())
        colors = [self.AGENT_COLORS.get(a, '#888888') for a in agents]

        bars = ax2.bar(agents, counts, color=colors, alpha=0.85, edgecolor='black')
        ax2.set_ylabel('Número de Eventos', fontsize=12)
        ax2.set_xlabel('Agente', fontsize=12)
        ax2.set_title('Eventos por Agente', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, count in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f'  Guardado: {output_path}')

        plt.close()

    def generate_all_plots(self, output_dir: str = '/root/NEO_EVA/visualizations'):
        """Genera todos los gráficos."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print('\n' + '=' * 70)
        print('GENERANDO VISUALIZACIONES')
        print('=' * 70 + '\n')

        self.plot_energy_stress_timeline(output_dir / 'energy_stress_timeline.png')
        self.plot_phase_distribution(output_dir / 'phase_distribution.png')
        self.plot_cycles_comparison(output_dir / 'cycles_comparison.png')
        self.plot_symbolic_life(output_dir / 'symbolic_life.png')
        self.plot_phase_timeline(output_dir / 'phase_timeline.png')
        self.plot_quality_metrics(output_dir / 'quality_metrics.png')
        self.plot_archetypes(output_dir / 'archetypes.png')
        self.plot_events_distribution(output_dir / 'events_distribution.png')

        print('\n' + '=' * 70)
        print('VISUALIZACIONES COMPLETADAS')
        print('=' * 70)


def main():
    """Función principal."""
    # Buscar el log más reciente
    log_dir = Path('/root/NEO_EVA/logs')
    json_files = list(log_dir.glob('circadian_log_*.json'))

    if not json_files:
        print("No se encontraron logs JSON. Ejecutando simulación primero...")
        from logs.circadian_logger import run_simulation_and_export
        agents = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
        json_path, _, _, _ = run_simulation_and_export(agents)
    else:
        # Usar el más reciente
        json_path = max(json_files, key=lambda p: p.stat().st_mtime)
        print(f'Usando log: {json_path}')

    # Cargar datos
    data = load_json_log(str(json_path))

    # Crear visualizador y generar gráficos
    viz = CircadianVisualizer(data)
    viz.generate_all_plots()


if __name__ == '__main__':
    if not HAS_MATPLOTLIB:
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'matplotlib', '-q'])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

    main()
