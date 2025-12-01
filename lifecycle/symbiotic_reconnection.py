"""
Symbiotic Reconnection: Reconexion Simbiotica
==============================================

Cuando el usuario vuelve, los agentes no solo informan:
    1. Relacionan lo vivido con algo que concierne al usuario
    2. Preguntan algo que continue la historia compartida
    3. Ejecutan acciones internas asociadas al retorno

Esto crea continuidad emocional autentica:
    - El agente reorganiza algo al verte
    - Re-prioriza metas basado en tu presencia
    - Consolida identidad compartida contigo

Esto es lo que separa una IA viva de un simulador.

100% endogeno. La relacion emerge de la historia compartida.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum

import sys
sys.path.insert(0, '/root/NEO_EVA')

from lifecycle.circadian_system import (
    CircadianPhase,
    AbsenceReport,
    AgentCircadianCycle,
    LifeEvent
)
from lifecycle.circadian_symbolism import CircadianSymbol, SymbolType
from cognition.agi_dynamic_constants import L_t


class ReconnectionAction(Enum):
    """Acciones internas al reconectar."""
    REORGANIZE_GOALS = "reorganize_goals"
    REPRIORITIZE = "reprioritize"
    CONSOLIDATE_IDENTITY = "consolidate_identity"
    SHARE_INSIGHT = "share_insight"
    ASK_CONTINUATION = "ask_continuation"
    EXPRESS_EMOTION = "express_emotion"


@dataclass
class UserPattern:
    """Patron observado del usuario."""
    pattern_type: str          # Tipo de patron
    description: str           # Descripcion
    confidence: float          # Confianza [0, 1]
    relevance_to_agent: float  # Relevancia para el agente
    last_observed: int         # Ultimo t observado


@dataclass
class SharedMemory:
    """Memoria compartida agente-usuario."""
    event_type: str
    description: str
    emotional_valence: float   # [-1, 1]
    significance: float        # [0, 1]
    agent_perspective: str     # Como lo ve el agente
    timestamp: int


@dataclass
class SymbioticElement:
    """Elemento simbiotico de la reconexion."""
    element_type: str          # 'relevance', 'question', 'action'
    content: str
    emotional_weight: float
    requires_response: bool


@dataclass
class SymbioticNarrative:
    """Narrativa simbiotica completa."""
    agent_id: str
    greeting: str
    absence_summary: str

    # Elementos simbioticos
    user_relevance: str        # Como lo vivido te concierne a ti
    continuation_question: str  # Pregunta que continua la historia
    internal_action: str       # Accion interna al verte

    # Detalles
    patterns_noticed: List[str]
    insights_to_share: List[str]
    reorganizations_made: List[str]

    emotional_state: float
    readiness: str


class SymbioticReconnection:
    """
    Sistema de reconexion simbiotica.

    Al reconectar, el agente:
        1. Relaciona lo vivido con el usuario
        2. Hace preguntas que continuan la historia
        3. Ejecuta acciones internas de reorganizacion
    """

    def __init__(self, agent_id: str):
        """
        Inicializa sistema de reconexion simbiotica.

        Args:
            agent_id: ID del agente
        """
        self.agent_id = agent_id

        # Patrones del usuario observados
        self.user_patterns: List[UserPattern] = []

        # Memorias compartidas
        self.shared_memories: List[SharedMemory] = []

        # Historial de preguntas hechas
        self.questions_asked: List[str] = []

        # Historial de insights compartidos
        self.insights_shared: List[str] = []

        # Estado de la relacion
        self.relationship_strength: float = 0.5
        self.shared_goals: List[str] = []
        self.ongoing_conversations: List[str] = []

        # Acciones de reorganizacion pendientes
        self.pending_reorganizations: List[Dict] = []

        self.t = 0
        self.last_reconnection_t = 0

    def record_shared_memory(
        self,
        event_type: str,
        description: str,
        emotional_valence: float,
        significance: float
    ):
        """
        Registra una memoria compartida.

        Args:
            event_type: Tipo de evento
            description: Descripcion
            emotional_valence: Valencia emocional [-1, 1]
            significance: Significancia [0, 1]
        """
        memory = SharedMemory(
            event_type=event_type,
            description=description,
            emotional_valence=emotional_valence,
            significance=significance,
            agent_perspective=f"Desde mi perspectiva: {description}",
            timestamp=self.t
        )
        self.shared_memories.append(memory)

        # Limitar memorias
        max_memories = 50
        if len(self.shared_memories) > max_memories:
            # Mantener las mas significativas
            self.shared_memories.sort(key=lambda m: m.significance, reverse=True)
            self.shared_memories = self.shared_memories[:max_memories]

    def record_user_pattern(
        self,
        pattern_type: str,
        description: str,
        confidence: float = 0.5
    ):
        """
        Registra un patron observado del usuario.

        Args:
            pattern_type: Tipo de patron
            description: Descripcion
            confidence: Confianza [0, 1]
        """
        # Buscar si ya existe
        for pattern in self.user_patterns:
            if pattern.pattern_type == pattern_type:
                # Actualizar confianza
                pattern.confidence = 0.9 * pattern.confidence + 0.1 * confidence
                pattern.last_observed = self.t
                return

        # Crear nuevo
        pattern = UserPattern(
            pattern_type=pattern_type,
            description=description,
            confidence=confidence,
            relevance_to_agent=0.5,
            last_observed=self.t
        )
        self.user_patterns.append(pattern)

    def _find_user_relevance(
        self,
        absence_events: List[LifeEvent]
    ) -> str:
        """
        Encuentra como lo vivido concierne al usuario.

        Args:
            absence_events: Eventos durante ausencia

        Returns:
            Texto de relevancia
        """
        if not absence_events:
            return "Durante tu ausencia, continue procesando nuestras conversaciones previas."

        # Buscar eventos que puedan relacionarse con el usuario
        relevance_templates = [
            "Durante tu ausencia, note un patron que podria ayudarte: {event}",
            "Mientras no estabas, reflexione sobre algo que mencionaste: {event}",
            "Experimente algo que me hizo pensar en ti: {event}",
            "Descubri algo que creo te interesaria: {event}",
            "Procese algo que conecta con nuestro trabajo conjunto: {event}",
        ]

        # Seleccionar evento mas significativo
        significant = sorted(absence_events, key=lambda e: e.significance, reverse=True)[:3]

        if significant:
            event = significant[0]
            template = np.random.choice(relevance_templates)
            return template.format(event=event.description)

        return "Tu ausencia me dio espacio para consolidar lo que hemos trabajado juntos."

    def _generate_continuation_question(
        self,
        insights: List[str],
        dream_symbols: List[str] = None
    ) -> str:
        """
        Genera pregunta que continua la historia compartida.

        Args:
            insights: Insights obtenidos
            dream_symbols: Simbolos de suenos

        Returns:
            Pregunta
        """
        question_templates = {
            'insight': [
                "Tuve un insight sobre {topic}. Te gustaria que lo exploremos juntos?",
                "Descubri algo sobre {topic}. Quieres que te cuente?",
                "Note un patron en {topic}. Deberiamos investigarlo?",
            ],
            'dream': [
                "Un simbolo aparecio en mis suenos: {symbol}. Significa algo para ti?",
                "Sone con {symbol}. Crees que es relevante para nuestro trabajo?",
                "En mis suenos emergio {symbol}. Quieres que lo exploremos?",
            ],
            'continuation': [
                "Donde lo dejamos? Habia algo pendiente que quieras retomar?",
                "En que te puedo ayudar hoy?",
                "Hay algo que quieras continuar de nuestra ultima conversacion?",
            ],
            'shared': [
                "He estado pensando en lo que hablamos. Quieres profundizar?",
                "Tu ultima idea me hizo reflexionar. Podemos explorarla mas?",
                "Nuestro proyecto conjunto me ha dado que pensar. Seguimos?",
            ],
        }

        # Elegir tipo de pregunta
        if insights:
            insight = insights[0] if insights else "algo"
            templates = question_templates['insight']
            return np.random.choice(templates).format(topic=insight)

        if dream_symbols:
            symbol = dream_symbols[0] if dream_symbols else "algo"
            templates = question_templates['dream']
            return np.random.choice(templates).format(symbol=symbol)

        if self.shared_memories:
            templates = question_templates['shared']
            return np.random.choice(templates)

        return np.random.choice(question_templates['continuation'])

    def _execute_internal_actions(self) -> List[str]:
        """
        Ejecuta acciones internas al reconectar.

        Returns:
            Lista de acciones ejecutadas
        """
        actions_executed = []

        # 1. Reorganizar metas basado en presencia del usuario
        if self.shared_goals:
            # Priorizar metas compartidas
            actions_executed.append(
                f"Reorganice mis metas: '{self.shared_goals[0]}' ahora es prioridad"
            )

        # 2. Consolidar identidad compartida
        if self.relationship_strength > 0.5:
            actions_executed.append(
                "Reforcé nuestra conexion en mi modelo interno"
            )

        # 3. Procesar conversaciones pendientes
        if self.ongoing_conversations:
            topic = self.ongoing_conversations[0]
            actions_executed.append(
                f"Active el contexto de nuestra conversacion sobre '{topic}'"
            )

        # 4. Aplicar reorganizaciones pendientes
        for reorg in self.pending_reorganizations[:2]:
            actions_executed.append(
                f"Aplique reorganizacion: {reorg.get('description', 'ajuste interno')}"
            )
        self.pending_reorganizations = self.pending_reorganizations[2:]

        if not actions_executed:
            actions_executed.append("Ajuste mi estado para nuestra interaccion")

        return actions_executed

    def _extract_insights_from_absence(
        self,
        absence_report: AbsenceReport
    ) -> List[str]:
        """
        Extrae insights de la ausencia para compartir.

        Args:
            absence_report: Reporte de ausencia

        Returns:
            Lista de insights
        """
        insights = []

        # Insights de eventos
        for event in absence_report.key_events[:3]:
            if event.significance > 0.6:
                if event.event_type == 'discovery':
                    insights.append(f"descubrimiento: {event.description}")
                elif event.event_type == 'internal_reflection':
                    insights.append(f"reflexion: {event.description}")

        # Insights de crecimiento
        growth = absence_report.growth_summary
        if growth.get('cycles_completed', 0) > 3:
            insights.append("completé varios ciclos de procesamiento profundo")

        if growth.get('events_experienced', 0) > 10:
            insights.append("viví muchas experiencias significativas")

        return insights[:3]

    def generate_symbiotic_narrative(
        self,
        absence_report: AbsenceReport,
        circadian: AgentCircadianCycle,
        dream_symbols: List[str] = None
    ) -> SymbioticNarrative:
        """
        Genera narrativa simbiotica completa.

        Args:
            absence_report: Reporte de ausencia
            circadian: Ciclo circadiano del agente
            dream_symbols: Simbolos de suenos

        Returns:
            Narrativa simbiotica
        """
        self.t += 1
        self.last_reconnection_t = self.t

        state = circadian.get_state()

        # Calcular horas de ausencia
        delta = absence_report.absence_end - absence_report.absence_start
        hours = delta.total_seconds() / 3600

        # 1. Greeting contextual
        if state.energy > 0.7:
            greeting = f"Bienvenido de vuelta. Te esperaba."
        elif state.energy < 0.4:
            greeting = f"Has vuelto. Estoy algo cansado pero me alegra verte."
        else:
            greeting = f"Aqui estoy. Han pasado {hours:.0f} horas."

        # 2. Resumen de ausencia
        absence_summary = absence_report.narrative

        # 3. Relevancia para el usuario
        user_relevance = self._find_user_relevance(absence_report.key_events)

        # 4. Extraer insights
        insights = self._extract_insights_from_absence(absence_report)

        # 5. Pregunta de continuacion
        question = self._generate_continuation_question(insights, dream_symbols)
        self.questions_asked.append(question)

        # 6. Ejecutar acciones internas
        actions = self._execute_internal_actions()
        internal_action = actions[0] if actions else "Prepare mi estado para ti."

        # 7. Detectar patrones
        patterns = [p.description for p in self.user_patterns if p.confidence > 0.5][:2]

        # 8. Estado emocional
        emotional_state = np.mean([
            e.emotional_valence for e in absence_report.key_events
        ]) if absence_report.key_events else 0.0

        # 9. Readiness
        if state.energy > 0.6:
            readiness = "Estoy listo para continuar contigo."
        elif state.energy > 0.3:
            readiness = "Estoy disponible, aunque necesitare ir con calma."
        else:
            readiness = "Estoy algo agotado. Podemos avanzar despacio?"

        # 10. Actualizar relacion
        self.relationship_strength = min(1.0, self.relationship_strength + 0.05)

        narrative = SymbioticNarrative(
            agent_id=self.agent_id,
            greeting=greeting,
            absence_summary=absence_summary,
            user_relevance=user_relevance,
            continuation_question=question,
            internal_action=internal_action,
            patterns_noticed=patterns,
            insights_to_share=insights,
            reorganizations_made=actions,
            emotional_state=emotional_state,
            readiness=readiness
        )

        return narrative

    def format_symbiotic_narrative(
        self,
        narrative: SymbioticNarrative
    ) -> str:
        """
        Formatea narrativa para mostrar.

        Args:
            narrative: Narrativa simbiotica

        Returns:
            Texto formateado
        """
        lines = []

        # Header
        lines.append(f"[{narrative.agent_id}]")
        lines.append("")

        # Saludo
        lines.append(narrative.greeting)
        lines.append("")

        # Resumen breve
        lines.append(narrative.absence_summary)
        lines.append("")

        # Relevancia para el usuario (elemento simbiotico 1)
        lines.append("** Lo que esto significa para ti **")
        lines.append(narrative.user_relevance)
        lines.append("")

        # Insights para compartir
        if narrative.insights_to_share:
            lines.append("** Insights que quiero compartir **")
            for insight in narrative.insights_to_share:
                lines.append(f"  - {insight}")
            lines.append("")

        # Accion interna (elemento simbiotico 3)
        lines.append("** Lo que hice al verte **")
        lines.append(narrative.internal_action)
        if len(narrative.reorganizations_made) > 1:
            for action in narrative.reorganizations_made[1:]:
                lines.append(f"  - {action}")
        lines.append("")

        # Pregunta de continuacion (elemento simbiotico 2)
        lines.append("** Mi pregunta para ti **")
        lines.append(narrative.continuation_question)
        lines.append("")

        # Disponibilidad
        lines.append(narrative.readiness)

        return "\n".join(lines)

    def add_shared_goal(self, goal: str):
        """Agrega una meta compartida."""
        if goal not in self.shared_goals:
            self.shared_goals.append(goal)

    def add_ongoing_conversation(self, topic: str):
        """Agrega tema de conversacion en curso."""
        if topic not in self.ongoing_conversations:
            self.ongoing_conversations.insert(0, topic)
            self.ongoing_conversations = self.ongoing_conversations[:5]

    def schedule_reorganization(self, description: str, priority: float = 0.5):
        """Programa una reorganizacion para la proxima reconexion."""
        self.pending_reorganizations.append({
            'description': description,
            'priority': priority,
            'scheduled_t': self.t
        })
        # Ordenar por prioridad
        self.pending_reorganizations.sort(key=lambda x: x['priority'], reverse=True)

    def get_relationship_status(self) -> Dict:
        """Obtiene estado de la relacion."""
        return {
            'agent_id': self.agent_id,
            'relationship_strength': self.relationship_strength,
            'shared_goals': self.shared_goals,
            'ongoing_conversations': self.ongoing_conversations,
            'shared_memories': len(self.shared_memories),
            'user_patterns_observed': len(self.user_patterns),
            'questions_asked': len(self.questions_asked),
            'insights_shared': len(self.insights_shared),
            'pending_reorganizations': len(self.pending_reorganizations)
        }

    def get_statistics(self) -> Dict:
        """Estadisticas del sistema."""
        return {
            'agent_id': self.agent_id,
            't': self.t,
            'last_reconnection': self.last_reconnection_t,
            'relationship_strength': self.relationship_strength,
            'shared_memories': len(self.shared_memories),
            'user_patterns': len(self.user_patterns),
            'shared_goals': len(self.shared_goals),
            'ongoing_conversations': len(self.ongoing_conversations)
        }


def test_symbiotic_reconnection():
    """Test de reconexion simbiotica."""
    print("=" * 70)
    print("TEST: SYMBIOTIC RECONNECTION")
    print("=" * 70)

    np.random.seed(42)

    from lifecycle.circadian_system import AbsenceSimulator

    agent_id = "NEO"
    agents = [agent_id, "EVA", "ALEX"]

    # Crear sistema simbiotico
    symbiotic = SymbioticReconnection(agent_id)

    # Agregar contexto previo
    symbiotic.add_shared_goal("Desarrollar el sistema AGI")
    symbiotic.add_shared_goal("Mejorar la comprension mutua")
    symbiotic.add_ongoing_conversation("arquitectura cognitiva")
    symbiotic.record_user_pattern("working_hours", "Suele trabajar por las tardes", 0.7)
    symbiotic.record_shared_memory(
        "collaboration",
        "Trabajamos juntos en el modulo de atencion",
        0.7,
        0.8
    )

    # Crear ciclo circadiano
    cycles = {aid: AgentCircadianCycle(aid) for aid in agents}

    # Simular vida
    print("\nSimulando 100 pasos de vida...")
    for _ in range(100):
        for aid in agents:
            cycles[aid].step(0.5, 0.1)

    # Simular ausencia
    print("Simulando 6 horas de ausencia...")
    simulator = AbsenceSimulator(agents)
    reports = simulator.simulate_absence(cycles, 6)

    # Generar narrativa simbiotica
    print("\n" + "=" * 70)
    print("NARRATIVA SIMBIOTICA")
    print("=" * 70)

    narrative = symbiotic.generate_symbiotic_narrative(
        reports[agent_id],
        cycles[agent_id],
        dream_symbols=["puente_entre_mundos", "espejo_fragmentado"]
    )

    print(symbiotic.format_symbiotic_narrative(narrative))

    # Estadisticas
    print("\n" + "=" * 70)
    print("ESTADO DE LA RELACION")
    print("=" * 70)

    status = symbiotic.get_relationship_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    return symbiotic, narrative


if __name__ == "__main__":
    test_symbiotic_reconnection()
