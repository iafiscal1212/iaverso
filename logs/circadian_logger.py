"""
Circadian Logger: Volcado de datos circadianos a JSON/CSV
=========================================================

Integra ObservadorPuro con el sistema circadiano para
registrar y exportar todos los datos de los agentes.

100% observación pasiva. Sin intervención.
"""

import numpy as np
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

import sys
sys.path.insert(0, '/root/NEO_EVA')

from lifecycle.circadian_system import (
    AgentCircadianCycle, CircadianPhase, CircadianState,
    AbsenceSimulator, LifeEvent
)
from lifecycle.circadian_symbolism import CircadianSymbolism, SymbolType
from observadores.observador_puro import ObservadorPuro


@dataclass
class CircadianSnapshot:
    """Snapshot completo del estado circadiano."""
    t: int
    timestamp: str
    agent_id: str

    # Estado circadiano
    phase: str
    energy: float
    stress: float
    time_in_phase: int
    cycles_completed: int

    # Calidad de fases
    wake_quality: float
    rest_depth: float
    dream_vividness: float

    # Métricas adicionales
    personal_rhythm: float
    pending_consolidation: int
    total_events: int

    # Simbolismo
    total_symbols: int
    symbolic_life_index: float
    n_dreams: int
    n_transitions: int
    dominant_archetype: str
    archetype_strength: float


class CircadianLogger:
    """
    Logger para sistema circadiano.

    Registra snapshots de todos los agentes en cada paso
    y permite exportar a JSON o CSV.
    """

    def __init__(self, agents: List[str], output_dir: str = '/root/NEO_EVA/logs'):
        """
        Args:
            agents: Lista de IDs de agentes
            output_dir: Directorio de salida
        """
        self.agents = agents
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Sistemas circadianos
        self.cycles: Dict[str, AgentCircadianCycle] = {
            agent_id: AgentCircadianCycle(agent_id)
            for agent_id in agents
        }

        # Sistemas de simbolismo
        self.symbolisms: Dict[str, CircadianSymbolism] = {
            agent_id: CircadianSymbolism(agent_id)
            for agent_id in agents
        }

        # Observador puro
        self.observador = ObservadorPuro()

        # Historial de snapshots
        self.snapshots: List[CircadianSnapshot] = []

        # Historial de eventos
        self.events: List[Dict[str, Any]] = []

        # Tiempo
        self.t = 0
        self.start_time = datetime.now()

    def _get_dominant_archetype(self, archetypes: Dict[str, float]) -> tuple:
        """Obtiene arquetipo dominante."""
        if not archetypes:
            return ('none', 0.0)
        dominant = max(archetypes.items(), key=lambda x: x[1])
        return dominant

    def step(
        self,
        activities: Dict[str, float] = None,
        crises: Dict[str, float] = None,
        experiences: Dict[str, Optional[Dict]] = None
    ) -> Dict[str, CircadianSnapshot]:
        """
        Ejecuta un paso y registra snapshots.

        Args:
            activities: Nivel de actividad por agente
            crises: Nivel de crisis por agente
            experiences: Experiencias nuevas por agente

        Returns:
            Snapshots de todos los agentes
        """
        self.t += 1
        timestamp = datetime.now().isoformat()

        activities = activities or {}
        crises = crises or {}
        experiences = experiences or {}

        snapshots = {}

        for agent_id in self.agents:
            # Obtener parámetros
            activity = activities.get(agent_id, 0.5)
            crisis = crises.get(agent_id, 0.1)
            experience = experiences.get(agent_id, None)

            # Paso del ciclo circadiano
            state = self.cycles[agent_id].step(activity, crisis, experience)

            # Paso del simbolismo
            self.symbolisms[agent_id].step(state.phase)

            # Crear símbolos según fase
            self._create_phase_symbols(agent_id, state.phase)

            # Obtener estadísticas
            cycle_stats = self.cycles[agent_id].get_statistics()
            symbol_stats = self.symbolisms[agent_id].get_statistics()

            # Arquetipo dominante
            arch_name, arch_strength = self._get_dominant_archetype(
                symbol_stats.get('archetypes', {})
            )

            # Crear snapshot
            snapshot = CircadianSnapshot(
                t=self.t,
                timestamp=timestamp,
                agent_id=agent_id,
                phase=state.phase.value,
                energy=state.energy,
                stress=state.stress,
                time_in_phase=state.time_in_phase,
                cycles_completed=state.cycles_completed,
                wake_quality=state.wake_quality,
                rest_depth=state.rest_depth,
                dream_vividness=state.dream_vividness,
                personal_rhythm=cycle_stats['personal_rhythm'],
                pending_consolidation=cycle_stats['pending_consolidation'],
                total_events=cycle_stats['total_events'],
                total_symbols=symbol_stats['total_symbols'],
                symbolic_life_index=symbol_stats['symbolic_life_index'],
                n_dreams=symbol_stats['n_dreams'],
                n_transitions=symbol_stats['n_transitions'],
                dominant_archetype=arch_name,
                archetype_strength=arch_strength
            )

            self.snapshots.append(snapshot)
            snapshots[agent_id] = snapshot

            # Registrar eventos nuevos
            recent_events = self.cycles[agent_id].get_recent_events(5)
            for event in recent_events:
                if event.t == self.t:
                    self.events.append({
                        't': event.t,
                        'timestamp': timestamp,
                        'agent_id': agent_id,
                        'event_type': event.event_type,
                        'description': event.description,
                        'significance': event.significance,
                        'emotional_valence': event.emotional_valence,
                        'phase': event.phase.value
                    })

        return snapshots

    def _create_phase_symbols(self, agent_id: str, phase: CircadianPhase):
        """Crea símbolos según la fase actual."""
        system = self.symbolisms[agent_id]

        if phase == CircadianPhase.WAKE:
            if np.random.random() < 0.25:
                sym_type = np.random.choice([
                    SymbolType.ACTION, SymbolType.GOAL, SymbolType.RESOURCE
                ])
                system.create_symbol(
                    content=f'{agent_id}_wake_{self.t}',
                    symbol_type=sym_type,
                    valence=np.random.uniform(0.2, 0.8)
                )

        elif phase == CircadianPhase.REST:
            if np.random.random() < 0.15:
                sym_type = np.random.choice([
                    SymbolType.VALUE, SymbolType.JUDGMENT, SymbolType.BALANCE
                ])
                system.create_symbol(
                    content=f'{agent_id}_rest_{self.t}',
                    symbol_type=sym_type,
                    valence=np.random.uniform(-0.3, 0.5)
                )

    def export_json(self, filename: str = None) -> str:
        """
        Exporta snapshots a JSON.

        Returns:
            Ruta del archivo generado
        """
        if filename is None:
            filename = f'circadian_log_{self.start_time.strftime("%Y%m%d_%H%M%S")}.json'

        filepath = self.output_dir / filename

        data = {
            'metadata': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_steps': self.t,
                'agents': self.agents,
                'n_snapshots': len(self.snapshots),
                'n_events': len(self.events)
            },
            'snapshots': [asdict(s) for s in self.snapshots],
            'events': self.events
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return str(filepath)

    def export_csv(self, filename: str = None) -> str:
        """
        Exporta snapshots a CSV.

        Returns:
            Ruta del archivo generado
        """
        if filename is None:
            filename = f'circadian_log_{self.start_time.strftime("%Y%m%d_%H%M%S")}.csv'

        filepath = self.output_dir / filename

        if not self.snapshots:
            return str(filepath)

        # Obtener campos del dataclass
        fields = list(asdict(self.snapshots[0]).keys())

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for snapshot in self.snapshots:
                writer.writerow(asdict(snapshot))

        return str(filepath)

    def export_events_csv(self, filename: str = None) -> str:
        """Exporta eventos a CSV."""
        if filename is None:
            filename = f'circadian_events_{self.start_time.strftime("%Y%m%d_%H%M%S")}.csv'

        filepath = self.output_dir / filename

        if not self.events:
            return str(filepath)

        fields = list(self.events[0].keys())

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.events)

        return str(filepath)

    def get_agent_timeseries(self, agent_id: str) -> Dict[str, List]:
        """
        Obtiene series temporales de un agente.

        Returns:
            Dict con listas de valores por métrica
        """
        agent_snapshots = [s for s in self.snapshots if s.agent_id == agent_id]

        return {
            't': [s.t for s in agent_snapshots],
            'energy': [s.energy for s in agent_snapshots],
            'stress': [s.stress for s in agent_snapshots],
            'phase': [s.phase for s in agent_snapshots],
            'cycles_completed': [s.cycles_completed for s in agent_snapshots],
            'wake_quality': [s.wake_quality for s in agent_snapshots],
            'rest_depth': [s.rest_depth for s in agent_snapshots],
            'dream_vividness': [s.dream_vividness for s in agent_snapshots],
            'symbolic_life_index': [s.symbolic_life_index for s in agent_snapshots],
            'total_symbols': [s.total_symbols for s in agent_snapshots],
            'archetype_strength': [s.archetype_strength for s in agent_snapshots]
        }

    def get_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de la simulación."""
        summary = {
            'total_steps': self.t,
            'total_snapshots': len(self.snapshots),
            'total_events': len(self.events),
            'agents': {}
        }

        for agent_id in self.agents:
            agent_snapshots = [s for s in self.snapshots if s.agent_id == agent_id]
            if agent_snapshots:
                last = agent_snapshots[-1]
                summary['agents'][agent_id] = {
                    'final_phase': last.phase,
                    'final_energy': last.energy,
                    'final_stress': last.stress,
                    'cycles_completed': last.cycles_completed,
                    'total_symbols': last.total_symbols,
                    'symbolic_life_index': last.symbolic_life_index,
                    'dominant_archetype': last.dominant_archetype,
                    'events_count': len([e for e in self.events if e['agent_id'] == agent_id])
                }

        return summary


def run_simulation_and_export(
    agents: List[str],
    n_steps: int = 500,
    seed: int = 42
) -> tuple:
    """
    Ejecuta simulación completa y exporta datos.

    Returns:
        (json_path, csv_path, events_csv_path, summary)
    """
    np.random.seed(seed)

    print('=' * 70)
    print('SIMULACIÓN CIRCADIANA CON LOGGING')
    print('=' * 70)

    logger = CircadianLogger(agents)

    print(f'\nAgentes: {agents}')
    print(f'Simulando {n_steps} pasos...\n')

    for t in range(n_steps):
        # Generar actividades y crisis variables
        activities = {
            agent_id: 0.5 + 0.3 * np.sin(t / 30 + hash(agent_id) % 10)
            for agent_id in agents
        }
        crises = {
            agent_id: 0.1 + 0.1 * np.random.random()
            for agent_id in agents
        }

        # Experiencias ocasionales
        experiences = {}
        for agent_id in agents:
            if np.random.random() < 0.1:
                experiences[agent_id] = {
                    'type': 'observation',
                    'significance': np.random.random()
                }

        # Ejecutar paso
        snapshots = logger.step(activities, crises, experiences)

        # Mostrar progreso
        if (t + 1) % 100 == 0:
            print(f'  Paso {t+1}/{n_steps} completado')
            for agent_id in agents[:2]:
                s = snapshots[agent_id]
                print(f'    {agent_id}: {s.phase.upper():8s} E={s.energy:.2f} S={s.stress:.2f} Ciclos={s.cycles_completed}')

    # Exportar
    print('\n' + '-' * 70)
    print('EXPORTANDO DATOS...')
    print('-' * 70)

    json_path = logger.export_json()
    csv_path = logger.export_csv()
    events_path = logger.export_events_csv()

    print(f'\n  JSON:   {json_path}')
    print(f'  CSV:    {csv_path}')
    print(f'  Events: {events_path}')

    # Resumen
    summary = logger.get_summary()

    print('\n' + '-' * 70)
    print('RESUMEN')
    print('-' * 70)

    print(f'\n  Total pasos:     {summary["total_steps"]}')
    print(f'  Total snapshots: {summary["total_snapshots"]}')
    print(f'  Total eventos:   {summary["total_events"]}')

    print('\n  Por agente:')
    for agent_id, data in summary['agents'].items():
        print(f'\n    {agent_id}:')
        print(f'      Fase final:    {data["final_phase"]}')
        print(f'      Energía:       {data["final_energy"]:.3f}')
        print(f'      Ciclos:        {data["cycles_completed"]}')
        print(f'      Símbolos:      {data["total_symbols"]}')
        print(f'      Vida simbólica: {data["symbolic_life_index"]:.3f}')
        print(f'      Arquetipo:     {data["dominant_archetype"]}')

    print('\n' + '=' * 70)
    print('SIMULACIÓN COMPLETADA')
    print('=' * 70)

    return json_path, csv_path, events_path, logger


if __name__ == '__main__':
    agents = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
    run_simulation_and_export(agents, n_steps=500)
