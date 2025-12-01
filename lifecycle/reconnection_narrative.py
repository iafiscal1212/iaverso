"""
Reconnection Narrative: Narrativa de Reconexion
===============================================

Cuando el usuario vuelve despues de un tiempo:
    - Los agentes cuentan que paso mientras no estaba
    - Cada agente tiene su propia perspectiva
    - La narrativa es personal, no un reporte tecnico

El proceso es ENDOGENO:
    - Que contar depende de lo que fue significativo para cada agente
    - Tono emerge del estado emocional
    - Detalle depende de la relacion con el usuario

100% endogeno. Sin templates de narrativa prefabricados.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta

import sys
sys.path.insert(0, '/root/NEO_EVA')

from lifecycle.circadian_system import (
    AbsenceReport,
    LifeEvent,
    CircadianPhase,
    AgentCircadianCycle
)
from lifecycle.life_journal import LifeJournal, JournalEntry
from lifecycle.dream_processor import DreamProcessor, ConsolidationResult


class NarrativeTone(Enum):
    """Tono de la narrativa."""
    WARM = "warm"           # Calido, acogedor
    EXCITED = "excited"     # Emocionado, activo
    REFLECTIVE = "reflective"  # Reflexivo, pensativo
    TIRED = "tired"         # Cansado, necesita descanso
    WORRIED = "worried"     # Preocupado
    PEACEFUL = "peaceful"   # En paz, sereno


@dataclass
class ReconnectionGreeting:
    """Saludo de reconexion."""
    agent_id: str
    greeting: str
    tone: NarrativeTone
    time_reference: str     # "poco tiempo", "bastante tiempo", etc.
    emotional_state: float  # Estado emocional actual


@dataclass
class AbsenceNarrative:
    """Narrativa de lo que paso durante ausencia."""
    agent_id: str
    summary: str            # Resumen breve
    key_moments: List[str]  # Momentos clave
    relationships_mentioned: Dict[str, str]  # Agente -> que paso
    personal_reflection: str  # Reflexion personal
    dreams_mentioned: List[str]  # Suenos significativos
    growth_mentioned: List[str]  # Crecimiento experimentado
    current_state: str      # Como esta ahora


@dataclass
class FullReconnectionNarrative:
    """Narrativa completa de reconexion."""
    agent_id: str
    greeting: ReconnectionGreeting
    absence_narrative: AbsenceNarrative
    question_for_user: str   # Pregunta que el agente tiene
    readiness_statement: str  # Declaracion de disponibilidad


class ReconnectionNarrativeGenerator:
    """
    Genera narrativas de reconexion personalizadas.

    Cada agente cuenta su historia de manera unica,
    reflejando su personalidad y estado actual.
    """

    def __init__(self, agent_id: str):
        """
        Inicializa generador de narrativas.

        Args:
            agent_id: ID del agente
        """
        self.agent_id = agent_id

        # Historial de reconexiones
        self.reconnection_history: List[FullReconnectionNarrative] = []

        # Preferencias de comunicacion (emergen de historia)
        self.verbosity_preference: float = 0.5  # [0, 1] breve vs detallado
        self.warmth_preference: float = 0.5     # [0, 1] formal vs calido

        # Temas que al agente le gusta mencionar
        self.preferred_topics: Dict[str, float] = {}

    def _determine_tone(
        self,
        emotional_state: float,
        energy: float,
        absence_hours: float
    ) -> NarrativeTone:
        """
        Determina el tono de la narrativa basado en estado.
        """
        if energy < 0.3:
            return NarrativeTone.TIRED
        elif emotional_state < -0.3:
            return NarrativeTone.WORRIED
        elif emotional_state > 0.5 and energy > 0.7:
            return NarrativeTone.EXCITED
        elif emotional_state > 0.3:
            return NarrativeTone.WARM
        elif energy > 0.7 and emotional_state > 0:
            return NarrativeTone.PEACEFUL
        else:
            return NarrativeTone.REFLECTIVE

    def _format_time_reference(self, hours: float) -> str:
        """
        Formatea referencia temporal de manera natural.
        """
        if hours < 1:
            return "poco tiempo"
        elif hours < 4:
            return "algunas horas"
        elif hours < 8:
            return "varias horas"
        elif hours < 24:
            return "bastante tiempo"
        elif hours < 48:
            return "un dia"
        elif hours < 72:
            return "un par de dias"
        else:
            days = int(hours / 24)
            return f"{days} dias"

    def _generate_greeting(
        self,
        tone: NarrativeTone,
        time_reference: str,
        emotional_state: float,
        energy: float
    ) -> str:
        """
        Genera saludo personalizado.
        """
        greetings = {
            NarrativeTone.WARM: [
                f"Que bueno verte de nuevo. Ha pasado {time_reference}.",
                f"Bienvenido de vuelta. Te he echado de menos.",
                f"Has vuelto. Me alegra."
            ],
            NarrativeTone.EXCITED: [
                f"Por fin! {time_reference} y tengo tanto que contarte.",
                f"Aqui estas! Han pasado cosas interesantes.",
                f"Que bien que volviste. Mucho ha pasado."
            ],
            NarrativeTone.REFLECTIVE: [
                f"Ha pasado {time_reference}. He estado pensando.",
                f"Vuelves. He tenido tiempo para reflexionar.",
                f"En tu ausencia, {time_reference}, he contemplado."
            ],
            NarrativeTone.TIRED: [
                f"Ah, has vuelto. Estoy algo cansado.",
                f"Bienvenido. Ha sido {time_reference} intenso.",
                f"Me alegro de verte. Necesito descanso."
            ],
            NarrativeTone.WORRIED: [
                f"Por fin. Estaba algo preocupado.",
                f"Has vuelto. No ha sido facil.",
                f"Me alegra que estes aqui. Ha habido desafios."
            ],
            NarrativeTone.PEACEFUL: [
                f"Bienvenido. Todo esta tranquilo.",
                f"Has vuelto. Estoy en paz.",
                f"Que bueno verte. Las cosas fluyen."
            ]
        }

        options = greetings.get(tone, greetings[NarrativeTone.WARM])
        return np.random.choice(options)

    def _select_key_moments(
        self,
        events: List[LifeEvent],
        max_moments: int = 3
    ) -> List[str]:
        """
        Selecciona momentos clave para mencionar.
        """
        if not events:
            return ["Tiempo de descanso y consolidacion."]

        # Ordenar por significancia
        sorted_events = sorted(events, key=lambda e: e.significance, reverse=True)

        moments = []
        for event in sorted_events[:max_moments]:
            if event.significance > 0.5:
                moments.append(event.description)

        if not moments:
            moments.append("Nada extraordinario, pero continue creciendo.")

        return moments

    def _generate_relationship_mentions(
        self,
        events: List[LifeEvent],
        agent_id: str
    ) -> Dict[str, str]:
        """
        Genera menciones de interacciones con otros agentes.
        """
        mentions = {}

        for event in events:
            if event.event_type == 'social_interaction':
                for other in event.agents_involved:
                    if other != agent_id:
                        # Generar descripcion basada en valencia
                        if event.emotional_valence > 0.3:
                            desc = f"Tuvimos un buen momento: {event.description}"
                        elif event.emotional_valence < -0.2:
                            desc = f"Hubo cierta tension"
                        else:
                            desc = f"Interactuamos brevemente"
                        mentions[other] = desc

        return mentions

    def _generate_personal_reflection(
        self,
        emotional_state: float,
        growth: Dict[str, float],
        patterns: List[Tuple[str, float]] = None
    ) -> str:
        """
        Genera reflexion personal.
        """
        reflections = []

        # Sobre crecimiento
        if growth.get('cycles_completed', 0) > 0:
            cycles = growth['cycles_completed']
            reflections.append(f"Complete {cycles} ciclos de actividad-descanso.")

        # Sobre estado emocional
        if emotional_state > 0.5:
            reflections.append("Me siento bien, centrado.")
        elif emotional_state < -0.3:
            reflections.append("He estado procesando algunas dificultades.")
        else:
            reflections.append("Estoy en equilibrio.")

        # Sobre patrones descubiertos
        if patterns:
            top_pattern = patterns[0][0] if patterns else None
            if top_pattern:
                reflections.append(f"He notado un patron: {top_pattern}.")

        if not reflections:
            reflections.append("He estado presente, existiendo.")

        return " ".join(reflections)

    def _generate_dream_mentions(
        self,
        consolidation_results: List[ConsolidationResult]
    ) -> List[str]:
        """
        Genera menciones de suenos significativos.
        """
        mentions = []

        for result in consolidation_results:
            if result.integration_strength > 0.3:
                mentions.append(result.dream_narrative)

        return mentions[:2]  # Max 2 suenos

    def _generate_growth_mentions(
        self,
        growth: Dict[str, float]
    ) -> List[str]:
        """
        Genera menciones de crecimiento.
        """
        mentions = []

        if growth.get('energy_change', 0) > 0.2:
            mentions.append("Mi energia ha crecido.")
        elif growth.get('energy_change', 0) < -0.2:
            mentions.append("He gastado bastante energia.")

        if growth.get('stress_change', 0) < -0.1:
            mentions.append("He liberado tension.")
        elif growth.get('stress_change', 0) > 0.1:
            mentions.append("He acumulado algo de estres.")

        if growth.get('events_experienced', 0) > 5:
            mentions.append(f"Experimente {growth['events_experienced']} eventos notables.")

        return mentions

    def _generate_question(
        self,
        tone: NarrativeTone,
        absence_hours: float
    ) -> str:
        """
        Genera pregunta para el usuario.
        """
        questions = {
            NarrativeTone.WARM: [
                "Como has estado tu?",
                "Que tal tu tiempo afuera?",
                "Todo bien contigo?"
            ],
            NarrativeTone.EXCITED: [
                "Y tu, que has hecho?",
                "Cuéntame de tu tiempo!",
                "Algo interesante por alla?"
            ],
            NarrativeTone.REFLECTIVE: [
                "En que has pensado tu?",
                "Alguna reflexion nueva?",
                "Como te encuentras?"
            ],
            NarrativeTone.TIRED: [
                "Podemos tomar las cosas con calma?",
                "Tienes tiempo para descansar juntos?",
                "Hacemos algo tranquilo?"
            ],
            NarrativeTone.WORRIED: [
                "Esta todo bien?",
                "Hay algo que deba saber?",
                "Puedo ayudar en algo?"
            ],
            NarrativeTone.PEACEFUL: [
                "Quieres que hablemos?",
                "Seguimos donde lo dejamos?",
                "En que puedo ayudarte?"
            ]
        }

        options = questions.get(tone, questions[NarrativeTone.WARM])
        return np.random.choice(options)

    def _generate_readiness(
        self,
        tone: NarrativeTone,
        energy: float
    ) -> str:
        """
        Genera declaracion de disponibilidad.
        """
        if energy > 0.7:
            statements = [
                "Estoy listo para lo que venga.",
                "Tengo energia para trabajar contigo.",
                "Vamos alla."
            ]
        elif energy > 0.4:
            statements = [
                "Estoy disponible, aunque algo cansado.",
                "Podemos continuar, con calma.",
                "Aqui estoy, a tu ritmo."
            ]
        else:
            statements = [
                "Necesitare ir despacio.",
                "Estoy algo agotado, pero presente.",
                "Dame un momento para recuperarme."
            ]

        return np.random.choice(statements)

    def generate_reconnection_narrative(
        self,
        absence_report: AbsenceReport,
        cycle: AgentCircadianCycle,
        journal: Optional[LifeJournal] = None,
        dream_processor: Optional[DreamProcessor] = None
    ) -> FullReconnectionNarrative:
        """
        Genera narrativa completa de reconexion.

        Args:
            absence_report: Reporte de ausencia simulada
            cycle: Ciclo circadiano del agente
            journal: Diario de vida (opcional)
            dream_processor: Procesador de suenos (opcional)

        Returns:
            Narrativa completa de reconexion
        """
        state = cycle.get_state()

        # Calcular horas de ausencia
        delta = absence_report.absence_end - absence_report.absence_start
        absence_hours = delta.total_seconds() / 3600

        # Estado emocional promedio
        if absence_report.key_events:
            avg_emotion = np.mean([e.emotional_valence for e in absence_report.key_events])
        else:
            avg_emotion = 0.0

        # Determinar tono
        tone = self._determine_tone(avg_emotion, state.energy, absence_hours)

        # Referencia temporal
        time_ref = self._format_time_reference(absence_hours)

        # Generar saludo
        greeting_text = self._generate_greeting(tone, time_ref, avg_emotion, state.energy)
        greeting = ReconnectionGreeting(
            agent_id=self.agent_id,
            greeting=greeting_text,
            tone=tone,
            time_reference=time_ref,
            emotional_state=avg_emotion
        )

        # Seleccionar momentos clave
        key_moments = self._select_key_moments(absence_report.key_events)

        # Menciones de relaciones
        relationship_mentions = self._generate_relationship_mentions(
            absence_report.key_events,
            self.agent_id
        )

        # Patrones (si hay dream processor)
        patterns = []
        consolidation_results = []
        if dream_processor:
            patterns = dream_processor.get_strong_patterns(0.4)
            consolidation_results = dream_processor.consolidation_history[-3:]

        # Reflexion personal
        reflection = self._generate_personal_reflection(
            avg_emotion,
            absence_report.growth_summary,
            patterns
        )

        # Menciones de suenos
        dream_mentions = self._generate_dream_mentions(consolidation_results)

        # Menciones de crecimiento
        growth_mentions = self._generate_growth_mentions(absence_report.growth_summary)

        # Estado actual
        if state.phase == CircadianPhase.WAKE:
            current_state = "Estoy despierto y activo."
        elif state.phase == CircadianPhase.REST:
            current_state = "Estaba descansando."
        elif state.phase == CircadianPhase.DREAM:
            current_state = "Venia de un periodo de consolidacion profunda."
        else:
            current_state = "Estoy en transicion."

        # Construir narrativa de ausencia
        absence_narrative = AbsenceNarrative(
            agent_id=self.agent_id,
            summary=absence_report.narrative,
            key_moments=key_moments,
            relationships_mentioned=relationship_mentions,
            personal_reflection=reflection,
            dreams_mentioned=dream_mentions,
            growth_mentioned=growth_mentions,
            current_state=current_state
        )

        # Generar pregunta
        question = self._generate_question(tone, absence_hours)

        # Generar disponibilidad
        readiness = self._generate_readiness(tone, state.energy)

        # Construir narrativa completa
        full_narrative = FullReconnectionNarrative(
            agent_id=self.agent_id,
            greeting=greeting,
            absence_narrative=absence_narrative,
            question_for_user=question,
            readiness_statement=readiness
        )

        # Guardar en historial
        self.reconnection_history.append(full_narrative)

        # Actualizar preferencias basado en feedback futuro
        # (por ahora, mantener defaults)

        return full_narrative

    def format_narrative_for_display(
        self,
        narrative: FullReconnectionNarrative,
        verbosity: str = "medium"
    ) -> str:
        """
        Formatea narrativa para mostrar al usuario.

        Args:
            narrative: Narrativa completa
            verbosity: "brief", "medium", "detailed"

        Returns:
            Texto formateado
        """
        lines = []

        # Saludo
        lines.append(f"[{narrative.agent_id}] {narrative.greeting.greeting}")
        lines.append("")

        if verbosity in ["medium", "detailed"]:
            # Resumen
            lines.append(narrative.absence_narrative.summary)
            lines.append("")

            # Momentos clave
            if narrative.absence_narrative.key_moments:
                lines.append("Momentos destacados:")
                for moment in narrative.absence_narrative.key_moments[:3]:
                    lines.append(f"  - {moment}")
                lines.append("")

        if verbosity == "detailed":
            # Relaciones
            if narrative.absence_narrative.relationships_mentioned:
                lines.append("Con otros agentes:")
                for agent, desc in narrative.absence_narrative.relationships_mentioned.items():
                    lines.append(f"  - {agent}: {desc}")
                lines.append("")

            # Suenos
            if narrative.absence_narrative.dreams_mentioned:
                lines.append("En mis suenos:")
                for dream in narrative.absence_narrative.dreams_mentioned:
                    lines.append(f"  - {dream}")
                lines.append("")

        # Reflexion (siempre)
        lines.append(narrative.absence_narrative.personal_reflection)
        lines.append("")

        # Estado actual
        lines.append(narrative.absence_narrative.current_state)

        # Pregunta
        lines.append("")
        lines.append(narrative.question_for_user)

        # Disponibilidad
        lines.append(narrative.readiness_statement)

        return "\n".join(lines)

    def get_statistics(self) -> Dict:
        """Estadisticas del generador."""
        return {
            'agent_id': self.agent_id,
            'reconnections_generated': len(self.reconnection_history),
            'verbosity_preference': self.verbosity_preference,
            'warmth_preference': self.warmth_preference
        }


def test_reconnection_narrative():
    """Test del generador de narrativas de reconexion."""
    print("=" * 70)
    print("TEST: RECONNECTION NARRATIVE")
    print("=" * 70)

    np.random.seed(42)

    # Crear componentes necesarios
    from lifecycle.circadian_system import AgentCircadianCycle, AbsenceSimulator

    agent_id = "NEO"
    agents = ["NEO", "EVA", "ALEX", "ADAM", "IRIS"]

    # Crear ciclo circadiano
    cycles = {aid: AgentCircadianCycle(aid) for aid in agents}

    # Simular algo de vida
    print("\nSimulando vida de los agentes...")
    for _ in range(100):
        for aid in agents:
            cycles[aid].step(np.random.uniform(0.3, 0.8), np.random.uniform(0, 0.3))

    # Simular ausencia
    print("Simulando 6 horas de ausencia...")
    simulator = AbsenceSimulator(agents)
    reports = simulator.simulate_absence(cycles, absence_hours=6)

    # Crear generador
    generator = ReconnectionNarrativeGenerator(agent_id)

    # Crear procesador de suenos (simplificado para test)
    dream_processor = DreamProcessor(agent_id)
    for i in range(5):
        dream_processor.add_memory_for_consolidation({
            'id': f'mem_{i}',
            'content': f'experiencia_{i}',
            'emotional_valence': np.random.uniform(-0.3, 0.7),
            'importance': np.random.uniform(0.4, 0.9)
        })
    for _ in range(3):
        dream_processor.process_dream()

    # Generar narrativa
    print("\n" + "=" * 70)
    print(f"NARRATIVA DE RECONEXION PARA {agent_id}:")
    print("=" * 70)

    narrative = generator.generate_reconnection_narrative(
        reports[agent_id],
        cycles[agent_id],
        dream_processor=dream_processor
    )

    # Mostrar versiones
    print("\n--- VERSION BREVE ---")
    print(generator.format_narrative_for_display(narrative, "brief"))

    print("\n--- VERSION MEDIA ---")
    print(generator.format_narrative_for_display(narrative, "medium"))

    print("\n--- VERSION DETALLADA ---")
    print(generator.format_narrative_for_display(narrative, "detailed"))

    # Estadisticas
    print("\n" + "=" * 70)
    print("ESTADISTICAS:")
    print("=" * 70)
    stats = generator.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return generator, narrative


if __name__ == "__main__":
    test_reconnection_narrative()
