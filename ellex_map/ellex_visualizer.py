"""
ELLEX Visualizer: Visualizacion del Mapa Existencial
=====================================================

Genera visualizaciones ASCII y datos para graficos del estado ELLEX.

Visualizaciones:
    1. Mapa Existencial (radar de 9 capas)
    2. Tension-Identidad Plot
    3. Ciclo de Vida
    4. Red Simbolica

100% endogeno.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

import sys
sys.path.insert(0, '/root/NEO_EVA')

from ellex_map.ellex_map import ELLEXMap, ELLEXMapState


@dataclass
class VisualizationData:
    """Datos para visualizacion."""
    radar_data: Dict[str, float]        # Datos para radar chart
    tension_identity: Tuple[float, float]  # (tension, identity)
    lifecycle_phase: str                # Fase del ciclo
    lifecycle_progress: float           # Progreso en el ciclo
    symbolic_network: Dict[str, Any]    # Red de conexiones
    ascii_art: str                      # Representacion ASCII


class ELLEXVisualizer:
    """
    Visualizador del mapa ELLEX.

    Genera representaciones visuales del estado existencial.
    """

    # Caracteres para barras ASCII
    BAR_CHARS = " ▁▂▃▄▅▆▇█"

    # Nombres cortos de capas
    LAYER_SHORT = {
        'L1_cognitive': 'COG',
        'L2_symbolic': 'SYM',
        'L3_narrative': 'NAR',
        'L4_life': 'LIF',
        'L5_health': 'HEA',
        'L6_social': 'SOC',
        'L7_tension': 'TEN',
        'L8_identity': 'IDE',
        'L9_phase': 'PHA'
    }

    LAYER_NAMES = {
        'L1_cognitive': 'Coherencia Cognitiva',
        'L2_symbolic': 'Coherencia Simbolica',
        'L3_narrative': 'Coherencia Narrativa',
        'L4_life': 'Coherencia de Vida',
        'L5_health': 'Salud Interior',
        'L6_social': 'Coherencia Social',
        'L7_tension': 'Tension Existencial',
        'L8_identity': 'Identidad Persistente',
        'L9_phase': 'Equilibrio de Fases'
    }

    def __init__(self, ellex_map: ELLEXMap):
        self.ellex_map = ellex_map

    def _value_to_bar(self, value: float, width: int = 20) -> str:
        """Convierte un valor [0,1] a una barra ASCII."""
        filled = int(value * width)
        bar = "█" * filled + "░" * (width - filled)
        return bar

    def _value_to_char(self, value: float) -> str:
        """Convierte un valor [0,1] a un caracter de altura."""
        idx = int(value * (len(self.BAR_CHARS) - 1))
        idx = max(0, min(idx, len(self.BAR_CHARS) - 1))
        return self.BAR_CHARS[idx]

    def _get_zone_symbol(self, zone: str) -> str:
        """Obtiene simbolo para una zona."""
        symbols = {
            'stagnant': '○',
            'healthy': '●',
            'crisis': '◉',
            'struggling': '▼',
            'balanced': '■',
            'flourishing': '▲'
        }
        return symbols.get(zone, '?')

    def generate_ascii_radar(self, state: ELLEXMapState) -> str:
        """
        Genera un radar chart ASCII de las 9 capas.
        """
        lines = []
        lines.append("╔══════════════════════════════════════════════════╗")
        lines.append("║          ELLEX EXISTENTIAL MAP                   ║")
        lines.append("╠══════════════════════════════════════════════════╣")

        # Valores de capas
        layer_values = {
            'L1_cognitive': state.L1_cognitive,
            'L2_symbolic': state.L2_symbolic,
            'L3_narrative': state.L3_narrative,
            'L4_life': state.L4_life,
            'L5_health': state.L5_health,
            'L6_social': state.L6_social,
            'L7_tension': state.L7_tension,
            'L8_identity': state.L8_identity,
            'L9_phase': state.L9_phase
        }

        for layer_key, value in layer_values.items():
            short = self.LAYER_SHORT[layer_key]
            name = self.LAYER_NAMES[layer_key]
            bar = self._value_to_bar(value, 20)
            pct = int(value * 100)

            # Colorizar tension (no es coherencia)
            if layer_key == 'L7_tension':
                zone = state.tension_zone
                zone_sym = self._get_zone_symbol(zone)
                line = f"║ {short} {bar} {pct:3d}% {zone_sym} {name[:15]:<15} ║"
            else:
                line = f"║ {short} {bar} {pct:3d}%   {name[:15]:<15} ║"

            lines.append(line)

        lines.append("╠══════════════════════════════════════════════════╣")

        # ELLEX total
        ellex_bar = self._value_to_bar(state.ellex, 20)
        ellex_pct = int(state.ellex * 100)
        zone_sym = self._get_zone_symbol(state.existential_zone)
        lines.append(f"║ ELLEX {ellex_bar} {ellex_pct:3d}% {zone_sym}              ║")

        # Tendencia
        if state.trend > 0.1:
            trend_sym = "↑"
        elif state.trend < -0.1:
            trend_sym = "↓"
        else:
            trend_sym = "→"

        lines.append(f"║ Zone: {state.existential_zone:<12} Trend: {trend_sym}  Stability: {int(state.stability*100):3d}% ║")
        lines.append("╚══════════════════════════════════════════════════╝")

        return "\n".join(lines)

    def generate_tension_identity_plot(self, state: ELLEXMapState) -> str:
        """
        Genera un plot ASCII de Tension vs Identidad.

        Cuadrantes:
            Alta I, Baja T = Estable pero Estancado
            Alta I, Alta T = Crecimiento
            Baja I, Baja T = Perdido
            Baja I, Alta T = Crisis de Identidad
        """
        width = 40
        height = 15

        # Crear grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]

        # Dibujar ejes
        for i in range(height):
            grid[i][width//2] = '│'
        for j in range(width):
            grid[height//2][j] = '─'
        grid[height//2][width//2] = '┼'

        # Posicion del agente
        t = state.L7_tension
        i = state.L8_identity

        x = int(t * (width - 1))
        y = int((1 - i) * (height - 1))  # Invertir Y

        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))

        grid[y][x] = '●'

        # Labels
        lines = []
        lines.append("         TENSION vs IDENTIDAD")
        lines.append("    Identity")
        lines.append("       ↑")
        lines.append("  1.0  │  Estable    │   Crecimiento")
        lines.append("       │  Estancado  │")

        for row in grid:
            lines.append("       " + "".join(row))

        lines.append("       │             │")
        lines.append("  0.0  │   Perdido   │ Crisis Identidad")
        lines.append("       └─────────────┼──────────────→ Tension")
        lines.append("                    0.5              1.0")

        return "\n".join(lines)

    def generate_lifecycle_view(self, state: ELLEXMapState) -> str:
        """
        Genera visualizacion del ciclo de vida (fases circadianas).
        """
        phases = ['WAKE', 'LIMINAL', 'REST', 'DREAM']
        phase_symbols = {
            'wake': '☀',
            'rest': '☾',
            'dream': '★',
            'liminal': '◐'
        }

        # Obtener fase actual del estado de fases
        current_phase = 'wake'  # Default
        if hasattr(self.ellex_map.L9, '_transition_history'):
            if self.ellex_map.L9._transition_history:
                current_phase = self.ellex_map.L9._transition_history[-1]

        lines = []
        lines.append("╔══════════════════════════════════════╗")
        lines.append("║       LIFECYCLE PHASE VIEW           ║")
        lines.append("╠══════════════════════════════════════╣")

        # Dibujar ciclo
        cycle_line = "║  "
        for phase in phases:
            sym = phase_symbols.get(phase.lower(), '?')
            if phase.lower() == current_phase:
                cycle_line += f"[{sym} {phase}] → "
            else:
                cycle_line += f" {sym} {phase}  → "

        cycle_line = cycle_line[:-3]  # Quitar ultima flecha
        cycle_line += "║"
        lines.append(cycle_line[:40] + "║")

        # Proporciones
        if hasattr(self.ellex_map.L9, '_phase_counts'):
            counts = self.ellex_map.L9._phase_counts
            total = sum(counts.values()) or 1

            lines.append("╠══════════════════════════════════════╣")
            lines.append("║ Phase Distribution:                  ║")

            for phase in ['wake', 'rest', 'dream', 'liminal']:
                prop = counts.get(phase, 0) / total
                bar = self._value_to_bar(prop, 15)
                pct = int(prop * 100)
                lines.append(f"║   {phase.upper():<8} {bar} {pct:3d}%   ║")

        lines.append("╠══════════════════════════════════════╣")
        lines.append(f"║ Phase Equilibrium: {int(state.L9_phase*100):3d}%              ║")
        lines.append("╚══════════════════════════════════════╝")

        return "\n".join(lines)

    def generate_symbolic_network(self) -> str:
        """
        Genera visualizacion de la red simbolica.
        """
        lines = []
        lines.append("╔══════════════════════════════════════╗")
        lines.append("║      SYMBOLIC NETWORK VIEW           ║")
        lines.append("╠══════════════════════════════════════╣")

        # Obtener conceptos activos
        if hasattr(self.ellex_map, '_symbolic_cohesion'):
            sc = self.ellex_map._symbolic_cohesion
            if sc._concept_history:
                # Mostrar top conceptos
                concept_means = {}
                for concept, history in sc._concept_history.items():
                    if history:
                        concept_means[concept] = np.mean(history[-5:])

                sorted_concepts = sorted(
                    concept_means.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]

                lines.append("║ Active Concepts:                     ║")
                for concept, activation in sorted_concepts:
                    bar = self._value_to_bar(activation, 10)
                    concept_short = concept[:15]
                    lines.append(f"║   {concept_short:<15} {bar}     ║")

                # Conexiones
                if sc._connection_history:
                    recent_conns = sc._connection_history[-1] if sc._connection_history else set()
                    lines.append("╠══════════════════════════════════════╣")
                    lines.append(f"║ Connections: {len(recent_conns):<3}                     ║")
            else:
                lines.append("║ No concepts observed yet             ║")
        else:
            lines.append("║ Symbolic cohesion not available      ║")

        lines.append("╚══════════════════════════════════════╝")
        return "\n".join(lines)

    def generate_health_dashboard(self, state: ELLEXMapState) -> str:
        """
        Genera dashboard de salud.
        """
        lines = []
        lines.append("╔══════════════════════════════════════╗")
        lines.append("║        HEALTH DASHBOARD              ║")
        lines.append("╠══════════════════════════════════════╣")

        # Status general
        status_sym = {
            'healthy': '✓',
            'recovering': '~',
            'unhealthy': '✗'
        }
        sym = status_sym.get(state.health_status, '?')
        lines.append(f"║ Status: {sym} {state.health_status.upper():<20}     ║")

        # Barra de salud
        health_bar = self._value_to_bar(state.L5_health, 25)
        lines.append(f"║ Health: {health_bar} {int(state.L5_health*100):3d}% ║")

        # Componentes si estan disponibles
        if hasattr(self.ellex_map.L5, '_current_components'):
            comps = self.ellex_map.L5._current_components
            lines.append("╠══════════════════════════════════════╣")

            for key in ['diagnosis_quality', 'treatment_efficacy',
                       'iatrogenesis_free', 'rotation_health', 'resilience']:
                if key in comps:
                    value = comps[key]
                    short_name = key.replace('_', ' ').title()[:15]
                    bar = self._value_to_bar(value, 12)
                    lines.append(f"║   {short_name:<15} {bar} {int(value*100):3d}%║")

        lines.append("╚══════════════════════════════════════╝")
        return "\n".join(lines)

    def generate_full_report(self, state: ELLEXMapState) -> str:
        """
        Genera reporte completo del estado ELLEX.
        """
        sections = []

        # Header
        sections.append("=" * 60)
        sections.append("           ELLEX FULL EXISTENTIAL REPORT")
        sections.append(f"           Agent: {self.ellex_map.agent_id}")
        sections.append(f"           Time: t={state.t}")
        sections.append("=" * 60)
        sections.append("")

        # Radar principal
        sections.append(self.generate_ascii_radar(state))
        sections.append("")

        # Tension-Identidad
        sections.append(self.generate_tension_identity_plot(state))
        sections.append("")

        # Lifecycle
        sections.append(self.generate_lifecycle_view(state))
        sections.append("")

        # Health
        sections.append(self.generate_health_dashboard(state))
        sections.append("")

        # Symbolic Network
        sections.append(self.generate_symbolic_network())
        sections.append("")

        # Summary
        sections.append("=" * 60)
        sections.append("                    SUMMARY")
        sections.append("=" * 60)

        # Weakest and strongest
        weakest = self.ellex_map.get_weakest_areas(3)
        strongest = self.ellex_map.get_strongest_areas(3)

        sections.append("\nWeakest Areas (need attention):")
        for name, value in weakest:
            sections.append(f"  - {self.LAYER_NAMES.get(name, name)}: {int(value*100)}%")

        sections.append("\nStrongest Areas:")
        for name, value in strongest:
            sections.append(f"  + {self.LAYER_NAMES.get(name, name)}: {int(value*100)}%")

        sections.append("")
        sections.append("=" * 60)

        return "\n".join(sections)

    def get_visualization_data(self, state: ELLEXMapState) -> VisualizationData:
        """
        Obtiene datos estructurados para visualizacion externa.
        """
        # Radar data
        radar_data = {
            'L1_cognitive': state.L1_cognitive,
            'L2_symbolic': state.L2_symbolic,
            'L3_narrative': state.L3_narrative,
            'L4_life': state.L4_life,
            'L5_health': state.L5_health,
            'L6_social': state.L6_social,
            'L7_tension': state.L7_tension,
            'L8_identity': state.L8_identity,
            'L9_phase': state.L9_phase
        }

        # Tension-Identity
        tension_identity = (state.L7_tension, state.L8_identity)

        # Lifecycle
        current_phase = 'wake'
        if hasattr(self.ellex_map.L9, '_transition_history'):
            if self.ellex_map.L9._transition_history:
                current_phase = self.ellex_map.L9._transition_history[-1]

        # Progress (simplified)
        phase_progress = state.L9_phase

        # Symbolic network
        symbolic_network = {'concepts': {}, 'connections': []}
        if hasattr(self.ellex_map, '_symbolic_cohesion'):
            sc = self.ellex_map._symbolic_cohesion
            if sc._concept_history:
                for concept, history in sc._concept_history.items():
                    if history:
                        symbolic_network['concepts'][concept] = np.mean(history[-5:])
            if sc._connection_history:
                symbolic_network['connections'] = list(
                    sc._connection_history[-1] if sc._connection_history else []
                )

        # ASCII art
        ascii_art = self.generate_ascii_radar(state)

        return VisualizationData(
            radar_data=radar_data,
            tension_identity=tension_identity,
            lifecycle_phase=current_phase,
            lifecycle_progress=phase_progress,
            symbolic_network=symbolic_network,
            ascii_art=ascii_art
        )
