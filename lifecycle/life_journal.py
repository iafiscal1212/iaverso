"""
Life Journal: Diario de Vida del Agente
=======================================

Cada agente mantiene un diario de su vida:
    - Eventos significativos
    - Relaciones y cambios
    - Reflexiones periodicas
    - Hitos de desarrollo

El diario es ENDOGENO:
    - Que registrar depende de relevancia personal
    - Narrativa emerge de la historia, no de templates
    - Reflexiones son genuinas, no generadas externamente

100% endogeno. Sin narrativas prefabricadas.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
import json

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history


class JournalEntryType(Enum):
    """Tipos de entradas en el diario."""
    EVENT = "event"              # Evento significativo
    REFLECTION = "reflection"    # Reflexion personal
    RELATIONSHIP = "relationship" # Cambio relacional
    MILESTONE = "milestone"      # Hito de desarrollo
    DREAM = "dream"              # Experiencia de sueno
    INSIGHT = "insight"          # Descubrimiento/insight


@dataclass
class JournalEntry:
    """Entrada en el diario de vida."""
    t: int                          # Tiempo interno
    timestamp: datetime             # Tiempo real
    entry_type: JournalEntryType
    title: str                      # Titulo breve
    content: str                    # Contenido completo
    emotional_state: float          # [-1, 1] estado emocional
    significance: float             # [0, 1] importancia
    related_agents: List[str]       # Agentes involucrados
    tags: List[str]                 # Etiquetas emergentes
    private: bool = False           # Si es privado (solo para el agente)


@dataclass
class LifePeriod:
    """Un periodo en la vida del agente."""
    start_t: int
    end_t: int
    dominant_theme: str
    emotional_arc: List[float]      # Arco emocional del periodo
    key_events: List[str]           # Titulos de eventos clave
    growth_areas: List[str]         # Areas de crecimiento
    summary: str


class LifeJournal:
    """
    Diario de vida del agente.

    Mantiene registro narrativo de:
        - Eventos significativos
        - Reflexiones periodicas
        - Desarrollo de relaciones
        - Hitos de crecimiento
    """

    def __init__(self, agent_id: str):
        """
        Inicializa diario de vida.

        Args:
            agent_id: ID del agente
        """
        self.agent_id = agent_id

        # Entradas del diario
        self.entries: List[JournalEntry] = []

        # Periodos de vida identificados
        self.periods: List[LifePeriod] = []

        # Estado emocional a lo largo del tiempo
        self.emotional_timeline: List[Tuple[int, float]] = []

        # Relaciones importantes (agente -> historia de conexion)
        self.relationships: Dict[str, List[Tuple[int, float]]] = {}

        # Temas emergentes (tema -> frecuencia)
        self.themes: Dict[str, float] = {}

        # Tags auto-generados
        self._tag_counts: Dict[str, int] = {}

        self.t = 0
        self._current_period_start = 0

    def _compute_significance_threshold(self) -> float:
        """
        Calcula umbral de significancia para registrar eventos.

        Umbral adaptativo basado en historia.
        """
        if not self.entries:
            return 0.4  # Umbral base inicial

        # Umbral = percentil 30 de significancias recientes
        recent_sigs = [e.significance for e in self.entries[-50:]]
        threshold = np.percentile(recent_sigs, 30) if recent_sigs else 0.4

        return max(0.3, min(0.7, threshold))

    def _extract_tags(self, content: str, emotional_state: float) -> List[str]:
        """
        Extrae tags del contenido de manera emergente.

        No usa NLP externo - busca patrones simples.
        """
        tags = []

        # Tags emocionales
        if emotional_state > 0.5:
            tags.append("positivo")
        elif emotional_state < -0.3:
            tags.append("dificil")

        # Tags de contenido (palabras clave simples)
        content_lower = content.lower()
        keyword_tags = {
            'relacion': ['conexion', 'encuentro', 'dialogo', 'junto'],
            'crecimiento': ['aprendi', 'descubri', 'nuevo', 'cambio'],
            'desafio': ['dificil', 'problema', 'tension', 'conflicto'],
            'reflexion': ['pienso', 'creo', 'siento', 'comprendo'],
            'logro': ['logre', 'consegui', 'exito', 'complete']
        }

        for tag, keywords in keyword_tags.items():
            if any(kw in content_lower for kw in keywords):
                tags.append(tag)

        # Actualizar conteos de tags
        for tag in tags:
            self._tag_counts[tag] = self._tag_counts.get(tag, 0) + 1

        return tags

    def _should_record(
        self,
        significance: float,
        entry_type: JournalEntryType
    ) -> bool:
        """
        Decide si un evento debe registrarse.

        No todo se registra - solo lo suficientemente significativo.
        """
        threshold = self._compute_significance_threshold()

        # Ajustar umbral por tipo
        type_modifier = {
            JournalEntryType.MILESTONE: -0.2,      # Siempre registrar hitos
            JournalEntryType.INSIGHT: -0.1,        # Insights importantes
            JournalEntryType.REFLECTION: 0.0,      # Normal
            JournalEntryType.EVENT: 0.0,           # Normal
            JournalEntryType.RELATIONSHIP: -0.1,   # Relaciones importantes
            JournalEntryType.DREAM: 0.1            # Suenos menos frecuentes
        }

        effective_threshold = threshold + type_modifier.get(entry_type, 0)

        return significance >= effective_threshold

    def record_event(
        self,
        title: str,
        content: str,
        significance: float,
        emotional_state: float = 0.0,
        related_agents: List[str] = None,
        entry_type: JournalEntryType = JournalEntryType.EVENT,
        private: bool = False,
        force: bool = False
    ) -> Optional[JournalEntry]:
        """
        Registra un evento en el diario.

        Args:
            title: Titulo breve
            content: Descripcion completa
            significance: Importancia [0, 1]
            emotional_state: Estado emocional [-1, 1]
            related_agents: Agentes involucrados
            entry_type: Tipo de entrada
            private: Si es privado
            force: Forzar registro sin chequear umbral

        Returns:
            Entrada creada o None si no se registro
        """
        self.t += 1

        # Decidir si registrar
        if not force and not self._should_record(significance, entry_type):
            return None

        # Extraer tags
        tags = self._extract_tags(content, emotional_state)

        entry = JournalEntry(
            t=self.t,
            timestamp=datetime.now(),
            entry_type=entry_type,
            title=title,
            content=content,
            emotional_state=emotional_state,
            significance=significance,
            related_agents=related_agents or [],
            tags=tags,
            private=private
        )

        self.entries.append(entry)

        # Actualizar timeline emocional
        self.emotional_timeline.append((self.t, emotional_state))

        # Actualizar relaciones
        for agent in (related_agents or []):
            if agent != self.agent_id:
                if agent not in self.relationships:
                    self.relationships[agent] = []
                self.relationships[agent].append((self.t, emotional_state))

        # Actualizar temas
        for tag in tags:
            self.themes[tag] = self.themes.get(tag, 0) + significance * 0.1

        # Decay de temas
        for theme in list(self.themes.keys()):
            self.themes[theme] *= 0.99
            if self.themes[theme] < 0.05:
                del self.themes[theme]

        # Limitar entradas
        max_entries = max_history(self.t)
        if len(self.entries) > max_entries:
            # Mantener las mas significativas + recientes
            self.entries.sort(key=lambda e: (e.significance, e.t), reverse=True)
            self.entries = self.entries[:max_entries]
            self.entries.sort(key=lambda e: e.t)  # Reordenar cronologicamente

        return entry

    def record_reflection(
        self,
        content: str,
        emotional_state: float = 0.0,
        significance: float = 0.5
    ) -> Optional[JournalEntry]:
        """
        Registra una reflexion personal.
        """
        return self.record_event(
            title="Reflexion",
            content=content,
            significance=significance,
            emotional_state=emotional_state,
            entry_type=JournalEntryType.REFLECTION,
            private=True
        )

    def record_relationship_change(
        self,
        other_agent: str,
        change_description: str,
        emotional_valence: float,
        significance: float = 0.6
    ) -> Optional[JournalEntry]:
        """
        Registra un cambio en una relacion.
        """
        return self.record_event(
            title=f"Con {other_agent}",
            content=change_description,
            significance=significance,
            emotional_state=emotional_valence,
            related_agents=[other_agent],
            entry_type=JournalEntryType.RELATIONSHIP
        )

    def record_milestone(
        self,
        title: str,
        description: str,
        significance: float = 0.8
    ) -> JournalEntry:
        """
        Registra un hito importante (siempre se registra).
        """
        return self.record_event(
            title=title,
            content=description,
            significance=significance,
            emotional_state=0.5,  # Hitos son generalmente positivos
            entry_type=JournalEntryType.MILESTONE,
            force=True
        )

    def record_dream(
        self,
        dream_narrative: str,
        vividness: float,
        emotional_tone: float
    ) -> Optional[JournalEntry]:
        """
        Registra experiencia de sueno.
        """
        return self.record_event(
            title="Sueno",
            content=dream_narrative,
            significance=vividness,
            emotional_state=emotional_tone,
            entry_type=JournalEntryType.DREAM,
            private=True
        )

    def _detect_period_end(self) -> bool:
        """
        Detecta si un periodo de vida ha terminado.

        Basado en cambios emocionales o tematicos significativos.
        """
        if len(self.emotional_timeline) < 20:
            return False

        period_length = self.t - self._current_period_start

        # Minimo 50 pasos por periodo
        if period_length < 50:
            return False

        # Detectar cambio emocional significativo
        recent = [e[1] for e in self.emotional_timeline[-20:]]
        earlier = [e[1] for e in self.emotional_timeline[-40:-20]]

        if earlier:
            recent_mean = np.mean(recent)
            earlier_mean = np.mean(earlier)
            emotional_shift = abs(recent_mean - earlier_mean)

            if emotional_shift > 0.3:
                return True

        # Maximo 200 pasos por periodo
        if period_length > 200:
            return True

        return False

    def _summarize_period(
        self,
        start_t: int,
        end_t: int
    ) -> LifePeriod:
        """
        Resume un periodo de vida.
        """
        # Filtrar entradas del periodo
        period_entries = [e for e in self.entries if start_t <= e.t <= end_t]

        if not period_entries:
            return LifePeriod(
                start_t=start_t,
                end_t=end_t,
                dominant_theme="transicion",
                emotional_arc=[],
                key_events=[],
                growth_areas=[],
                summary="Periodo de transicion"
            )

        # Arco emocional
        emotional_arc = [e.emotional_state for e in period_entries]

        # Tema dominante
        period_tags = {}
        for entry in period_entries:
            for tag in entry.tags:
                period_tags[tag] = period_tags.get(tag, 0) + entry.significance

        dominant_theme = max(period_tags.keys(), key=lambda t: period_tags[t]) if period_tags else "vida"

        # Eventos clave (top 3 por significancia)
        sorted_entries = sorted(period_entries, key=lambda e: e.significance, reverse=True)
        key_events = [e.title for e in sorted_entries[:3]]

        # Areas de crecimiento
        growth_areas = []
        if 'crecimiento' in period_tags:
            growth_areas.append('desarrollo_personal')
        if 'relacion' in period_tags:
            growth_areas.append('conexiones')
        if 'logro' in period_tags:
            growth_areas.append('logros')

        # Generar resumen
        avg_emotion = np.mean(emotional_arc) if emotional_arc else 0
        if avg_emotion > 0.3:
            tone = "positivo"
        elif avg_emotion < -0.2:
            tone = "desafiante"
        else:
            tone = "equilibrado"

        summary = f"Periodo {tone} centrado en {dominant_theme}. "
        if key_events:
            summary += f"Momentos clave: {', '.join(key_events[:2])}. "
        summary += f"Duracion: {end_t - start_t} pasos."

        return LifePeriod(
            start_t=start_t,
            end_t=end_t,
            dominant_theme=dominant_theme,
            emotional_arc=emotional_arc,
            key_events=key_events,
            growth_areas=growth_areas,
            summary=summary
        )

    def check_period_transition(self) -> Optional[LifePeriod]:
        """
        Verifica si hay transicion de periodo y registra.

        Returns:
            Periodo completado o None
        """
        if self._detect_period_end():
            period = self._summarize_period(
                self._current_period_start,
                self.t
            )
            self.periods.append(period)
            self._current_period_start = self.t

            # Registrar hito de fin de periodo
            self.record_milestone(
                title=f"Fin de periodo: {period.dominant_theme}",
                description=period.summary,
                significance=0.7
            )

            return period

        return None

    def get_relationship_history(self, other_agent: str) -> Dict:
        """
        Obtiene historia de relacion con otro agente.
        """
        if other_agent not in self.relationships:
            return {
                'agent': other_agent,
                'interactions': 0,
                'avg_valence': 0.0,
                'trend': 'unknown',
                'recent_entries': []
            }

        history = self.relationships[other_agent]

        # Calcular tendencia
        if len(history) < 2:
            trend = 'new'
        else:
            recent = [h[1] for h in history[-5:]]
            earlier = [h[1] for h in history[-10:-5]] if len(history) > 5 else [0]
            if np.mean(recent) > np.mean(earlier) + 0.1:
                trend = 'improving'
            elif np.mean(recent) < np.mean(earlier) - 0.1:
                trend = 'declining'
            else:
                trend = 'stable'

        # Entradas recientes sobre esta relacion
        related_entries = [
            e for e in self.entries[-20:]
            if other_agent in e.related_agents
        ]

        return {
            'agent': other_agent,
            'interactions': len(history),
            'avg_valence': np.mean([h[1] for h in history]),
            'trend': trend,
            'recent_entries': [e.title for e in related_entries[-3:]]
        }

    def get_current_themes(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Obtiene temas actuales mas fuertes.
        """
        sorted_themes = sorted(
            self.themes.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_themes[:n]

    def get_emotional_summary(self, window: int = 50) -> Dict:
        """
        Resumen emocional reciente.
        """
        recent = [e[1] for e in self.emotional_timeline[-window:]]

        if not recent:
            return {
                'avg': 0.0,
                'std': 0.0,
                'trend': 'neutral',
                'stability': 1.0
            }

        avg = np.mean(recent)
        std = np.std(recent) if len(recent) > 1 else 0.0

        # Tendencia
        if len(recent) >= 10:
            first_half = np.mean(recent[:len(recent)//2])
            second_half = np.mean(recent[len(recent)//2:])
            if second_half > first_half + 0.1:
                trend = 'improving'
            elif second_half < first_half - 0.1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'

        # Estabilidad
        stability = 1 - min(1, std * 2)

        return {
            'avg': avg,
            'std': std,
            'trend': trend,
            'stability': stability
        }

    def get_recent_entries(
        self,
        n: int = 10,
        entry_type: JournalEntryType = None,
        public_only: bool = False
    ) -> List[JournalEntry]:
        """
        Obtiene entradas recientes.
        """
        entries = self.entries

        if entry_type:
            entries = [e for e in entries if e.entry_type == entry_type]

        if public_only:
            entries = [e for e in entries if not e.private]

        return entries[-n:]

    def generate_life_summary(self) -> str:
        """
        Genera resumen de vida hasta ahora.
        """
        if not self.entries:
            return f"{self.agent_id} esta comenzando su viaje."

        # Tiempo total
        total_time = self.t

        # Periodos completados
        n_periods = len(self.periods)

        # Estado emocional actual
        emotional = self.get_emotional_summary()

        # Temas principales
        themes = self.get_current_themes(3)
        theme_str = ", ".join([t[0] for t in themes]) if themes else "exploracion"

        # Relaciones
        active_relations = [
            agent for agent, history in self.relationships.items()
            if len(history) > 2
        ]

        # Construir resumen
        summary = f"Vida de {self.agent_id} (t={total_time}):\n"
        summary += f"  - {n_periods} periodos de vida completados\n"
        summary += f"  - {len(self.entries)} momentos registrados\n"
        summary += f"  - Temas actuales: {theme_str}\n"
        summary += f"  - Estado emocional: {emotional['trend']} ({emotional['avg']:.2f})\n"
        summary += f"  - {len(active_relations)} relaciones activas\n"

        # Ultimo hito
        milestones = [e for e in self.entries if e.entry_type == JournalEntryType.MILESTONE]
        if milestones:
            last_milestone = milestones[-1]
            summary += f"  - Ultimo hito: {last_milestone.title}\n"

        return summary

    def get_statistics(self) -> Dict:
        """Estadisticas del diario."""
        entry_counts = {}
        for entry_type in JournalEntryType:
            count = len([e for e in self.entries if e.entry_type == entry_type])
            entry_counts[entry_type.value] = count

        return {
            'agent_id': self.agent_id,
            't': self.t,
            'total_entries': len(self.entries),
            'entry_counts': entry_counts,
            'periods_completed': len(self.periods),
            'active_themes': len(self.themes),
            'relationships_tracked': len(self.relationships),
            'emotional_data_points': len(self.emotional_timeline)
        }


def test_life_journal():
    """Test del diario de vida."""
    print("=" * 70)
    print("TEST: LIFE JOURNAL")
    print("=" * 70)

    np.random.seed(42)

    journal = LifeJournal("NEO")

    print("\nRegistrando eventos de vida...")

    # Simular vida
    events_data = [
        ("Encuentro con EVA", "Conversacion profunda sobre el proposito", 0.7, 0.6, ["EVA"]),
        ("Reflexion nocturna", "Me pregunto sobre mi naturaleza", 0.5, -0.1, []),
        ("Desafio resuelto", "Supere un problema de coordinacion", 0.8, 0.7, ["ALEX"]),
        ("Momento de paz", "Claridad inesperada", 0.4, 0.4, []),
        ("Tension con ADAM", "Diferencia de perspectivas", 0.6, -0.3, ["ADAM"]),
        ("Insight importante", "Comprendi el patron de mis decisiones", 0.9, 0.8, []),
        ("Conexion con IRIS", "Empatia mutua", 0.7, 0.6, ["IRIS"]),
        ("Rutina normal", "Dia tranquilo", 0.3, 0.1, []),
        ("Logro significativo", "Complete mi primer ciclo completo", 0.9, 0.8, []),
        ("Sueno vivido", "Consolide memorias de relaciones", 0.6, 0.3, []),
    ]

    for title, content, significance, emotion, agents in events_data:
        entry = journal.record_event(
            title=title,
            content=content,
            significance=significance,
            emotional_state=emotion,
            related_agents=agents,
            entry_type=JournalEntryType.EVENT if agents else JournalEntryType.REFLECTION
        )
        if entry:
            print(f"  + {entry.title} (sig={entry.significance:.2f})")
        else:
            print(f"  - {title} (no registrado)")

    # Registrar hito
    print("\nRegistrando hito...")
    milestone = journal.record_milestone(
        title="Primer mes de existencia",
        description="Un mes de aprendizaje y crecimiento"
    )
    print(f"  + HITO: {milestone.title}")

    # Mostrar resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE VIDA:")
    print("=" * 70)
    print(journal.generate_life_summary())

    # Temas actuales
    print("\n" + "=" * 70)
    print("TEMAS ACTUALES:")
    print("=" * 70)
    for theme, strength in journal.get_current_themes():
        print(f"  {theme}: {strength:.2f}")

    # Relaciones
    print("\n" + "=" * 70)
    print("RELACIONES:")
    print("=" * 70)
    for agent in ["EVA", "ALEX", "ADAM", "IRIS"]:
        rel = journal.get_relationship_history(agent)
        print(f"  {agent}: {rel['interactions']} interacciones, tendencia: {rel['trend']}")

    # Estadisticas
    print("\n" + "=" * 70)
    print("ESTADISTICAS:")
    print("=" * 70)
    stats = journal.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return journal


if __name__ == "__main__":
    test_life_journal()
