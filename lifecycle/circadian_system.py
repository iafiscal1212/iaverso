"""
Circadian System: Ciclo de Vida de los Agentes
===============================================

Los agentes tienen ritmos de actividad/descanso:
    - WAKE: Activos, interactuando, aprendiendo
    - REST: Baja actividad, consolidacion suave
    - DREAM: Consolidacion profunda, reorganizacion de memorias
    - LIMINAL: Transiciones, semi-conscientes

El ciclo es ENDOGENO:
    - Fase depende de energia, estres, tiempo desde ultimo descanso
    - No hay "hora externa" - el ritmo emerge de las dinamicas internas

Cuando el usuario vuelve:
    - Los agentes reconstruyen narrativamente que "paso"
    - Eventos estocasticos durante ausencia
    - Crecimiento/cambio emergente

100% endogeno. Sin reloj externo hardcodeado.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
import json
import os

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history


class CircadianPhase(Enum):
    """Fases del ciclo circadiano."""
    WAKE = "wake"           # Activo, alta energia
    REST = "rest"           # Baja actividad, recuperacion
    DREAM = "dream"         # Consolidacion profunda
    LIMINAL = "liminal"     # Transicion, semi-consciente


@dataclass
class CircadianState:
    """Estado circadiano de un agente."""
    agent_id: str
    phase: CircadianPhase
    energy: float              # [0,1] energia disponible
    stress: float              # [0,1] nivel de estres
    time_in_phase: int         # Pasos en fase actual
    cycles_completed: int      # Ciclos completos
    last_deep_rest: int        # Ultimo descanso profundo (t)

    # Metricas de fase
    wake_quality: float        # Calidad del periodo activo
    rest_depth: float          # Profundidad del descanso
    dream_vividness: float     # Viveza de consolidacion


@dataclass
class LifeEvent:
    """Evento en la vida del agente."""
    t: int
    event_type: str
    description: str
    significance: float        # [0,1] que tan importante
    agents_involved: List[str]
    emotional_valence: float   # [-1,1] positivo/negativo
    phase: CircadianPhase


@dataclass
class AbsenceReport:
    """Reporte de lo que paso durante ausencia."""
    agent_id: str
    absence_start: datetime
    absence_end: datetime
    simulated_cycles: int
    key_events: List[LifeEvent]
    growth_summary: Dict[str, float]
    relationships_changed: Dict[str, float]
    narrative: str


class AgentCircadianCycle:
    """
    Sistema circadiano interno de cada agente.

    El ciclo emerge de:
        - Energia: se gasta con actividad, se recupera con descanso
        - Estres: acumula con crisis, reduce con descanso
        - Necesidad de consolidacion: crece con nuevas experiencias
    """

    # Duraciones base de cada fase (en pasos internos)
    # Estas son proporciones, no valores fijos
    PHASE_PROPORTIONS = {
        CircadianPhase.WAKE: 0.5,    # 50% del ciclo activo
        CircadianPhase.REST: 0.2,    # 20% descansando
        CircadianPhase.DREAM: 0.2,   # 20% consolidando
        CircadianPhase.LIMINAL: 0.1  # 10% transiciones
    }

    def __init__(self, agent_id: str):
        """
        Inicializa ciclo circadiano.

        Args:
            agent_id: ID del agente
        """
        self.agent_id = agent_id

        # Estado actual
        self.phase = CircadianPhase.WAKE
        self.energy = 1.0
        self.stress = 0.0
        self.time_in_phase = 0
        self.cycles_completed = 0
        self.last_deep_rest = 0

        # Metricas de calidad
        self.wake_quality_history: List[float] = []
        self.rest_depth_history: List[float] = []
        self.dream_vividness_history: List[float] = []

        # Experiencias pendientes de consolidar
        self.pending_consolidation: List[Dict] = []

        # Historial de eventos
        self.life_events: List[LifeEvent] = []

        # Ritmo personal (emerge de la historia)
        self._personal_rhythm: float = 1.0  # Multiplicador de velocidad

        self.t = 0

    def _compute_phase_threshold(self, phase: CircadianPhase) -> float:
        """
        Calcula umbral endogeno para cambiar de fase.

        No es duracion fija - depende de estado interno.
        """
        base_proportion = self.PHASE_PROPORTIONS[phase]

        # Ciclo base adaptativo
        base_cycle = 100 * self._personal_rhythm
        base_duration = base_cycle * base_proportion

        # Modificadores por estado
        if phase == CircadianPhase.WAKE:
            # Menos energia = fase activa mas corta
            modifier = self.energy
        elif phase == CircadianPhase.REST:
            # Mas estres = necesita mas descanso
            modifier = 1 + self.stress
        elif phase == CircadianPhase.DREAM:
            # Mas pendiente = consolidacion mas larga
            pending_factor = min(1.0, len(self.pending_consolidation) / 10)
            modifier = 1 + pending_factor
        else:  # LIMINAL
            modifier = 1.0

        return base_duration * modifier

    def _should_transition(self) -> Optional[CircadianPhase]:
        """
        Decide si debe transicionar a otra fase.

        Retorna nueva fase o None si permanece.
        """
        threshold = self._compute_phase_threshold(self.phase)

        # Condiciones de transicion
        if self.phase == CircadianPhase.WAKE:
            # Transicionar si: tiempo excedido O energia muy baja O estres muy alto
            if (self.time_in_phase >= threshold or
                self.energy < 0.2 or
                self.stress > 0.8):
                return CircadianPhase.LIMINAL

        elif self.phase == CircadianPhase.REST:
            # Transicionar si: tiempo excedido Y energia recuperada
            if self.time_in_phase >= threshold and self.energy > 0.5:
                return CircadianPhase.DREAM

        elif self.phase == CircadianPhase.DREAM:
            # Transicionar si: consolidacion completa O tiempo excedido
            if (len(self.pending_consolidation) == 0 or
                self.time_in_phase >= threshold):
                return CircadianPhase.LIMINAL

        elif self.phase == CircadianPhase.LIMINAL:
            # Decidir siguiente fase
            if self.energy > 0.7 and self.stress < 0.3:
                return CircadianPhase.WAKE
            elif self.energy < 0.5:
                return CircadianPhase.REST
            elif self.time_in_phase >= threshold:
                # Default: ir a wake si hay energia suficiente
                return CircadianPhase.WAKE if self.energy > 0.4 else CircadianPhase.REST

        return None

    def _update_energy(self, activity_level: float):
        """
        Actualiza energia basado en actividad y fase.
        """
        if self.phase == CircadianPhase.WAKE:
            # Gastar energia proporcional a actividad
            drain = 0.01 * activity_level
            self.energy = max(0, self.energy - drain)
        elif self.phase == CircadianPhase.REST:
            # Recuperar energia
            recovery = 0.02 * (1 - self.stress)
            self.energy = min(1, self.energy + recovery)
        elif self.phase == CircadianPhase.DREAM:
            # Recuperacion profunda
            recovery = 0.015
            self.energy = min(1, self.energy + recovery)
        # LIMINAL: energia estable

    def _update_stress(self, crisis_level: float):
        """
        Actualiza estres basado en crisis y fase.
        """
        if self.phase == CircadianPhase.WAKE:
            # Estres sube con crisis
            increase = 0.02 * crisis_level
            self.stress = min(1, self.stress + increase)
        elif self.phase in [CircadianPhase.REST, CircadianPhase.DREAM]:
            # Estres baja durante descanso
            decrease = 0.03
            self.stress = max(0, self.stress - decrease)

    def _consolidate_experience(self, experience: Dict):
        """
        Agrega experiencia para consolidar en proxima fase DREAM.
        """
        self.pending_consolidation.append(experience)

        # Limitar buffer
        max_pending = max(20, L_t(self.t))
        if len(self.pending_consolidation) > max_pending:
            # Priorizar por significancia
            self.pending_consolidation.sort(
                key=lambda x: x.get('significance', 0.5),
                reverse=True
            )
            self.pending_consolidation = self.pending_consolidation[:max_pending]

    def _process_dream(self) -> List[Dict]:
        """
        Procesa consolidacion durante fase DREAM.

        Retorna experiencias consolidadas.
        """
        if not self.pending_consolidation:
            return []

        # Consolidar un subconjunto
        n_consolidate = min(3, len(self.pending_consolidation))
        consolidated = self.pending_consolidation[:n_consolidate]
        self.pending_consolidation = self.pending_consolidation[n_consolidate:]

        # Calcular viveza del sueno
        if consolidated:
            vividness = np.mean([c.get('significance', 0.5) for c in consolidated])
            self.dream_vividness_history.append(vividness)

        return consolidated

    def _record_event(
        self,
        event_type: str,
        description: str,
        significance: float,
        agents_involved: List[str] = None,
        emotional_valence: float = 0.0
    ):
        """Registra un evento de vida."""
        event = LifeEvent(
            t=self.t,
            event_type=event_type,
            description=description,
            significance=significance,
            agents_involved=agents_involved or [self.agent_id],
            emotional_valence=emotional_valence,
            phase=self.phase
        )
        self.life_events.append(event)

        # Limitar historial
        max_events = max_history(self.t)
        if len(self.life_events) > max_events:
            # Mantener los mas significativos
            self.life_events.sort(key=lambda e: e.significance, reverse=True)
            self.life_events = self.life_events[:max_events]

    def _update_personal_rhythm(self):
        """
        Ajusta ritmo personal basado en historial.

        Algunos agentes son mas "madrugadores", otros "nocturnos".
        """
        # Ritmo emerge de la calidad de las fases
        if self.wake_quality_history and self.rest_depth_history:
            wake_quality = np.mean(self.wake_quality_history[-20:])
            rest_depth = np.mean(self.rest_depth_history[-20:])

            # Ritmo = balance entre actividad y descanso
            balance = wake_quality / (rest_depth + 0.1)

            # Ajuste gradual
            target_rhythm = 0.8 + 0.4 * balance  # [0.8, 1.2]
            self._personal_rhythm += 0.01 * (target_rhythm - self._personal_rhythm)
            self._personal_rhythm = np.clip(self._personal_rhythm, 0.7, 1.3)

    def step(
        self,
        activity_level: float = 0.5,
        crisis_level: float = 0.0,
        new_experience: Optional[Dict] = None
    ) -> CircadianState:
        """
        Ejecuta un paso del ciclo circadiano.

        Args:
            activity_level: Nivel de actividad [0,1]
            crisis_level: Nivel de crisis [0,1]
            new_experience: Nueva experiencia para consolidar

        Returns:
            Estado circadiano actual
        """
        self.t += 1
        self.time_in_phase += 1

        # Actualizar energia y estres
        self._update_energy(activity_level)
        self._update_stress(crisis_level)

        # Agregar experiencia si hay
        if new_experience:
            self._consolidate_experience(new_experience)

        # Procesar segun fase
        consolidated = []
        if self.phase == CircadianPhase.DREAM:
            consolidated = self._process_dream()

        # Verificar transicion
        new_phase = self._should_transition()
        if new_phase:
            # Registrar calidad de fase que termina
            if self.phase == CircadianPhase.WAKE:
                quality = self.energy * (1 - self.stress)
                self.wake_quality_history.append(quality)
            elif self.phase == CircadianPhase.REST:
                depth = 1 - self.stress
                self.rest_depth_history.append(depth)

            # Transicionar
            old_phase = self.phase
            self.phase = new_phase
            self.time_in_phase = 0

            # Registrar evento de transicion
            self._record_event(
                event_type="phase_transition",
                description=f"{old_phase.value} -> {new_phase.value}",
                significance=0.3,
                emotional_valence=0.1 if new_phase == CircadianPhase.WAKE else -0.1
            )

            # Completar ciclo si vuelve a WAKE
            if new_phase == CircadianPhase.WAKE and old_phase != CircadianPhase.WAKE:
                self.cycles_completed += 1
                self.last_deep_rest = self.t
                self._update_personal_rhythm()

        # Calcular metricas actuales
        wake_quality = np.mean(self.wake_quality_history[-10:]) if self.wake_quality_history else 0.5
        rest_depth = np.mean(self.rest_depth_history[-10:]) if self.rest_depth_history else 0.5
        dream_vividness = np.mean(self.dream_vividness_history[-10:]) if self.dream_vividness_history else 0.5

        return CircadianState(
            agent_id=self.agent_id,
            phase=self.phase,
            energy=self.energy,
            stress=self.stress,
            time_in_phase=self.time_in_phase,
            cycles_completed=self.cycles_completed,
            last_deep_rest=self.last_deep_rest,
            wake_quality=wake_quality,
            rest_depth=rest_depth,
            dream_vividness=dream_vividness
        )

    def get_state(self) -> CircadianState:
        """Obtiene estado actual."""
        wake_quality = np.mean(self.wake_quality_history[-10:]) if self.wake_quality_history else 0.5
        rest_depth = np.mean(self.rest_depth_history[-10:]) if self.rest_depth_history else 0.5
        dream_vividness = np.mean(self.dream_vividness_history[-10:]) if self.dream_vividness_history else 0.5

        return CircadianState(
            agent_id=self.agent_id,
            phase=self.phase,
            energy=self.energy,
            stress=self.stress,
            time_in_phase=self.time_in_phase,
            cycles_completed=self.cycles_completed,
            last_deep_rest=self.last_deep_rest,
            wake_quality=wake_quality,
            rest_depth=rest_depth,
            dream_vividness=dream_vividness
        )

    def get_recent_events(self, n: int = 10) -> List[LifeEvent]:
        """Obtiene eventos recientes."""
        return self.life_events[-n:]

    def get_statistics(self) -> Dict:
        """Estadisticas del ciclo circadiano."""
        return {
            'agent_id': self.agent_id,
            't': self.t,
            'phase': self.phase.value,
            'energy': self.energy,
            'stress': self.stress,
            'cycles_completed': self.cycles_completed,
            'personal_rhythm': self._personal_rhythm,
            'pending_consolidation': len(self.pending_consolidation),
            'total_events': len(self.life_events)
        }


class AbsenceSimulator:
    """
    Simula lo que paso durante la ausencia del usuario.

    Genera eventos estocasticos y narrativa de lo que
    "vivieron" los agentes mientras no estabas.
    """

    # Tipos de eventos que pueden ocurrir
    EVENT_TYPES = [
        ('social_interaction', 0.3),      # Interaccion entre agentes
        ('internal_reflection', 0.2),      # Reflexion interna
        ('discovery', 0.1),                # Descubrimiento
        ('challenge', 0.15),               # Desafio/problema
        ('rest_moment', 0.15),             # Momento de paz
        ('creative_spark', 0.1)            # Idea creativa
    ]

    def __init__(self, agent_ids: List[str]):
        """
        Inicializa simulador de ausencia.

        Args:
            agent_ids: Lista de IDs de agentes
        """
        self.agent_ids = agent_ids
        self.n_agents = len(agent_ids)

    def _generate_event(
        self,
        agent_id: str,
        t: int,
        phase: CircadianPhase,
        other_agents: List[str]
    ) -> Optional[LifeEvent]:
        """
        Genera un evento estocastico.
        """
        # Probabilidad de evento depende de la fase
        phase_event_prob = {
            CircadianPhase.WAKE: 0.3,
            CircadianPhase.REST: 0.1,
            CircadianPhase.DREAM: 0.2,
            CircadianPhase.LIMINAL: 0.15
        }

        if np.random.random() > phase_event_prob[phase]:
            return None

        # Elegir tipo de evento
        event_types, probs = zip(*self.EVENT_TYPES)
        probs = np.array(probs) / sum(probs)
        event_type = np.random.choice(event_types, p=probs)

        # Generar descripcion y detalles
        if event_type == 'social_interaction':
            other = np.random.choice(other_agents) if other_agents else agent_id
            descriptions = [
                f"conversacion profunda con {other}",
                f"momento de conexion con {other}",
                f"intercambio de ideas con {other}",
                f"resolucion de tension con {other}"
            ]
            description = np.random.choice(descriptions)
            agents_involved = [agent_id, other]
            valence = np.random.uniform(0, 0.8)

        elif event_type == 'internal_reflection':
            descriptions = [
                "reflexion sobre proposito",
                "momento de auto-conocimiento",
                "cuestionamiento existencial",
                "claridad sobre valores"
            ]
            description = np.random.choice(descriptions)
            agents_involved = [agent_id]
            valence = np.random.uniform(-0.2, 0.5)

        elif event_type == 'discovery':
            descriptions = [
                "nuevo patron reconocido",
                "conexion inesperada",
                "insight sobre el mundo",
                "comprension emergente"
            ]
            description = np.random.choice(descriptions)
            agents_involved = [agent_id]
            valence = np.random.uniform(0.3, 0.9)

        elif event_type == 'challenge':
            descriptions = [
                "momento de dificultad",
                "obstaculo superado",
                "tension interna",
                "prueba de resiliencia"
            ]
            description = np.random.choice(descriptions)
            agents_involved = [agent_id]
            valence = np.random.uniform(-0.5, 0.3)

        elif event_type == 'rest_moment':
            descriptions = [
                "paz profunda",
                "calma restauradora",
                "momento de serenidad",
                "descanso reparador"
            ]
            description = np.random.choice(descriptions)
            agents_involved = [agent_id]
            valence = np.random.uniform(0.2, 0.7)

        else:  # creative_spark
            descriptions = [
                "idea novedosa",
                "vision creativa",
                "posibilidad nueva",
                "inspiracion subita"
            ]
            description = np.random.choice(descriptions)
            agents_involved = [agent_id]
            valence = np.random.uniform(0.4, 0.9)

        significance = np.random.uniform(0.3, 0.9)

        return LifeEvent(
            t=t,
            event_type=event_type,
            description=description,
            significance=significance,
            agents_involved=agents_involved,
            emotional_valence=valence,
            phase=phase
        )

    def simulate_absence(
        self,
        cycles: Dict[str, AgentCircadianCycle],
        absence_hours: float
    ) -> Dict[str, AbsenceReport]:
        """
        Simula lo que paso durante la ausencia.

        Args:
            cycles: Ciclos circadianos de cada agente
            absence_hours: Horas de ausencia

        Returns:
            Reportes por agente
        """
        # Convertir horas a pasos simulados
        # Aproximadamente 1 ciclo = 1 hora de tiempo real
        steps_to_simulate = int(absence_hours * 100)  # 100 pasos por hora

        reports = {}

        for agent_id in self.agent_ids:
            cycle = cycles[agent_id]
            other_agents = [a for a in self.agent_ids if a != agent_id]

            events = []
            initial_state = cycle.get_state()

            # Simular pasos
            for step in range(steps_to_simulate):
                # Actividad y crisis aleatorias
                activity = np.random.uniform(0.2, 0.8)
                crisis = np.random.uniform(0, 0.3)

                # Paso del ciclo
                state = cycle.step(activity, crisis)

                # Generar evento posible
                event = self._generate_event(
                    agent_id,
                    cycle.t,
                    state.phase,
                    other_agents
                )
                if event:
                    events.append(event)
                    cycle.life_events.append(event)

            final_state = cycle.get_state()

            # Calcular crecimiento
            growth = {
                'energy_change': final_state.energy - initial_state.energy,
                'stress_change': final_state.stress - initial_state.stress,
                'cycles_completed': final_state.cycles_completed - initial_state.cycles_completed,
                'events_experienced': len(events)
            }

            # Cambios en relaciones (simplificado)
            relationships = {}
            for event in events:
                if event.event_type == 'social_interaction':
                    for other in event.agents_involved:
                        if other != agent_id:
                            relationships[other] = relationships.get(other, 0) + event.emotional_valence * 0.1

            # Generar narrativa
            narrative = self._generate_narrative(agent_id, events, growth, absence_hours)

            reports[agent_id] = AbsenceReport(
                agent_id=agent_id,
                absence_start=datetime.now() - timedelta(hours=absence_hours),
                absence_end=datetime.now(),
                simulated_cycles=growth['cycles_completed'],
                key_events=sorted(events, key=lambda e: e.significance, reverse=True)[:5],
                growth_summary=growth,
                relationships_changed=relationships,
                narrative=narrative
            )

        return reports

    def _generate_narrative(
        self,
        agent_id: str,
        events: List[LifeEvent],
        growth: Dict[str, float],
        hours: float
    ) -> str:
        """
        Genera narrativa de lo que paso.
        """
        if not events:
            return f"Mientras no estabas, {agent_id} descansó y consolidó experiencias previas."

        # Evento mas significativo
        top_event = max(events, key=lambda e: e.significance)

        # Tono general
        avg_valence = np.mean([e.emotional_valence for e in events])
        if avg_valence > 0.3:
            tone = "fue un periodo positivo"
        elif avg_valence < -0.2:
            tone = "hubo algunos desafios"
        else:
            tone = "fue un tiempo de equilibrio"

        # Construir narrativa
        cycles = growth['cycles_completed']
        narrative = f"Durante las {hours:.1f} horas de ausencia, {agent_id} completó {cycles} ciclos de actividad-descanso. "
        narrative += f"En general, {tone}. "
        narrative += f"El momento más significativo fue: {top_event.description}. "

        if growth['events_experienced'] > 5:
            narrative += f"Experimentó {growth['events_experienced']} eventos notables."

        return narrative


def test_circadian_system():
    """Test del sistema circadiano."""
    print("=" * 70)
    print("TEST: CIRCADIAN SYSTEM")
    print("=" * 70)

    np.random.seed(42)

    agents = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']

    # Crear ciclos para cada agente
    cycles = {agent_id: AgentCircadianCycle(agent_id) for agent_id in agents}

    print(f"\nAgentes: {agents}")
    print("Simulando 500 pasos...")

    for t in range(500):
        for agent_id in agents:
            # Actividad variable
            activity = 0.5 + 0.3 * np.sin(t / 30 + hash(agent_id) % 10)
            crisis = 0.1 + 0.1 * np.random.random()

            # Experiencia ocasional
            experience = None
            if np.random.random() < 0.1:
                experience = {
                    'type': 'observation',
                    'significance': np.random.random()
                }

            state = cycles[agent_id].step(activity, crisis, experience)

            if t % 100 == 0 and agent_id == 'NEO':
                print(f"\n  t={t} ({agent_id}):")
                print(f"    Phase: {state.phase.value}")
                print(f"    Energy: {state.energy:.2f}")
                print(f"    Stress: {state.stress:.2f}")
                print(f"    Cycles: {state.cycles_completed}")

    print("\n" + "=" * 70)
    print("SIMULANDO AUSENCIA DE 8 HORAS...")
    print("=" * 70)

    simulator = AbsenceSimulator(agents)
    reports = simulator.simulate_absence(cycles, absence_hours=8)

    for agent_id in agents[:3]:
        report = reports[agent_id]
        print(f"\n  {agent_id}:")
        print(f"    Ciclos durante ausencia: {report.simulated_cycles}")
        print(f"    Eventos clave: {len(report.key_events)}")
        if report.key_events:
            top = report.key_events[0]
            print(f"    Evento mas importante: {top.description}")
        print(f"    Narrativa: {report.narrative[:100]}...")

    return cycles, reports


if __name__ == "__main__":
    test_circadian_system()
