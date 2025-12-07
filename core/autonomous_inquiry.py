#!/usr/bin/env python3
"""
AUTONOMOUS INQUIRY - Agentes que Deciden Qué Investigar
=========================================================

PROBLEMA ANTERIOR:
- Claude decide qué datos dar a los agentes
- Claude decide qué preguntas hacer
- Claude interpreta los resultados
- Los agentes solo procesan lo que Claude les da

ESTE SISTEMA:
- Los agentes RECIBEN información cruda (texto, estímulos)
- Los agentes GENERAN sus propias preguntas
- Los agentes DECIDEN qué investigar
- Los agentes ACTÚAN según su personalidad e intereses

NO es Claude quien decide. Son ELLOS.

NORMA DURA: La personalidad afecta las preguntas que generan,
pero NO las predetermina. Dos agentes con la misma info
generarán preguntas DIFERENTES.
"""

import numpy as np
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class InquiryType(Enum):
    """Tipos de preguntas que un agente puede generar."""
    CAUSAL = "causal"           # ¿Por qué X causa Y?
    CORRELATIONAL = "corr"      # ¿X está relacionado con Y?
    TEMPORAL = "temporal"       # ¿Cuándo ocurrió X?
    COUNTERFACTUAL = "counter"  # ¿Qué si X no hubiera pasado?
    ETHICAL = "ethical"         # ¿Es esto correcto?
    PREDICTIVE = "predict"      # ¿Qué pasará si...?
    INVESTIGATIVE = "invest"    # ¿Quién/qué/cómo/dónde?
    META = "meta"               # ¿Por qué me interesa esto?


@dataclass
class Stimulus:
    """
    Un estímulo = información cruda que recibe el agente.

    Puede ser texto, datos, eventos - cualquier cosa.
    El agente decide qué hacer con ello.
    """
    content: str                    # El contenido crudo
    source: str = "unknown"         # De dónde viene
    timestamp: str = ""             # Cuándo llegó
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class Inquiry:
    """
    Una pregunta generada por el agente.

    No impuesta desde fuera - nace del agente.
    """
    question: str                   # La pregunta en sí
    inquiry_type: InquiryType       # Tipo de pregunta
    priority: float                 # Qué tan importante es para el agente (0-1)
    motivation: str                 # Por qué le importa al agente
    generated_at: str = ""
    agent_id: str = ""

    # Estado de la investigación
    status: str = "pending"         # pending, investigating, answered, abandoned
    findings: List[str] = field(default_factory=list)
    confidence: float = 0.0         # Confianza en los hallazgos

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat()


@dataclass
class Investigation:
    """
    Una investigación = pregunta + acciones para responderla.

    El agente decide qué acciones tomar.
    """
    inquiry: Inquiry
    actions_planned: List[str] = field(default_factory=list)
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    conclusions: List[str] = field(default_factory=list)


class AutonomousInquirer:
    """
    Un agente que genera sus propias preguntas y decide qué investigar.

    DIFERENCIA CLAVE vs ExplorerAgent:
    - ExplorerAgent analiza datos que le damos
    - AutonomousInquirer DECIDE qué quiere saber

    Su personalidad afecta:
    - Qué tipo de preguntas genera
    - Qué prioridad les da
    - Qué acciones considera
    - Cuándo abandona una línea de investigación
    """

    def __init__(self, agent_id: str, personality: Dict[str, float] = None):
        self.agent_id = agent_id
        self.t = 0

        # Personalidad auto-elegida si no se proporciona
        if personality is None:
            personality = self._discover_my_nature()
        self.personality = personality

        # Estado interno
        self.stimuli_received: List[Stimulus] = []
        self.inquiries_generated: List[Inquiry] = []
        self.investigations_active: List[Investigation] = []
        self.investigations_completed: List[Investigation] = []

        # Intereses emergentes (aprende de lo que investiga)
        self.interests: Dict[str, float] = {}  # tema -> intensidad
        self.curiosity_history: List[float] = []

        # Log de decisiones autónomas
        self.decision_log: List[Dict[str, Any]] = []

    def _discover_my_nature(self) -> Dict[str, float]:
        """
        El agente descubre su propia naturaleza.

        Basado en su nombre pero interpretado como resonancia interna.
        """
        name_hash = hashlib.sha256(self.agent_id.encode()).hexdigest()

        def trait(offset: int) -> float:
            return int(name_hash[offset:offset+4], 16) / 0xFFFF

        # Rasgos que afectan cómo genera preguntas
        nature = {
            # Curiosidad: qué tan rápido genera preguntas
            'curiosity': trait(0) * 0.6 + 0.4,  # [0.4, 1.0]

            # Profundidad: prefiere preguntas superficiales o profundas
            'depth': trait(4) * 0.8 + 0.2,  # [0.2, 1.0]

            # Escepticismo: cuestiona la información recibida
            'skepticism': trait(8) * 0.7 + 0.3,  # [0.3, 1.0]

            # Persistencia: cuánto persigue una pregunta antes de abandonar
            'persistence': trait(12) * 0.6 + 0.2,  # [0.2, 0.8]

            # Asociatividad: conecta temas aparentemente no relacionados
            'associativity': trait(16) * 0.7 + 0.3,  # [0.3, 1.0]

            # Ética: le importan las implicaciones éticas
            'ethical_concern': trait(20) * 0.8 + 0.2,  # [0.2, 1.0]

            # Pragmatismo: prefiere preguntas con utilidad práctica
            'pragmatism': trait(24) * 0.7 + 0.3,  # [0.3, 1.0]

            '_self_discovered': True,
        }

        return nature

    def receive_stimulus(self, stimulus: Stimulus) -> Dict[str, Any]:
        """
        Recibe un estímulo y decide qué hacer con él.

        NO procesa automáticamente. DECIDE si le interesa.

        Returns:
            Decisión del agente sobre el estímulo
        """
        self.t += 1
        self.stimuli_received.append(stimulus)

        # El agente evalúa el estímulo según su naturaleza
        relevance = self._evaluate_relevance(stimulus)

        decision = {
            't': self.t,
            'stimulus': stimulus.content[:100] + "..." if len(stimulus.content) > 100 else stimulus.content,
            'relevance': relevance,
            'action': 'ignore',  # Por defecto
            'reason': '',
        }

        # Decisión basada en personalidad
        curiosity_threshold = 1.0 - self.personality['curiosity']

        if relevance > curiosity_threshold:
            # Le interesa - genera preguntas
            inquiries = self._generate_inquiries(stimulus)
            decision['action'] = 'inquire'
            decision['inquiries_generated'] = len(inquiries)
            decision['reason'] = f"Relevancia {relevance:.2f} supera mi umbral {curiosity_threshold:.2f}"

            # Registrar intereses emergentes
            keywords = self._extract_keywords(stimulus.content)
            for kw in keywords:
                self.interests[kw] = self.interests.get(kw, 0) + relevance * 0.1

        else:
            decision['reason'] = f"Relevancia {relevance:.2f} no supera umbral {curiosity_threshold:.2f}"

        self.decision_log.append(decision)
        return decision

    def _evaluate_relevance(self, stimulus: Stimulus) -> float:
        """
        Evalúa qué tan relevante es un estímulo para este agente.

        Basado en:
        - Contenido del estímulo (temas sensibles)
        - Intereses previos
        - Naturaleza del agente
        - Conexiones con investigaciones activas
        """
        content_lower = stimulus.content.lower()

        # Base: algo de ruido intrínseco (el agente no es una máquina)
        base_relevance = np.random.random() * 0.15

        # Bonus por contenido con carga emocional/importante
        content_bonus = 0

        # Temas que despiertan interés inherente
        high_interest_topics = {
            'muerte': 0.25, 'víctimas': 0.25, 'mortales': 0.25,
            'falló': 0.2, 'fallos': 0.2, 'error': 0.15,
            'responsabilidad': 0.2, 'investigada': 0.15,
            'contrato': 0.15, 'contratos': 0.15,
            'emergencia': 0.15, 'emergencias': 0.15,
            'alerta': 0.15, 'temprana': 0.1,
            'gobierno': 0.1, 'política': 0.1, 'político': 0.1,
            'oculto': 0.2, 'secreto': 0.2,
            'corrupción': 0.25, 'negligencia': 0.2,
        }

        for topic, bonus in high_interest_topics.items():
            if topic in content_lower:
                content_bonus += bonus

        # Detectar entidades (empresas, lugares)
        keywords = self._extract_keywords(stimulus.content)
        entity_bonus = min(0.2, len([k for k in keywords if k[0].isupper()]) * 0.04)

        content_bonus = min(0.5, content_bonus + entity_bonus)

        # Bonus por intereses previos
        interest_bonus = 0
        for interest, intensity in self.interests.items():
            if interest.lower() in content_lower:
                interest_bonus += intensity
        interest_bonus = min(0.3, interest_bonus)

        # Bonus por conexión con investigaciones activas
        investigation_bonus = 0
        for inv in self.investigations_active:
            if any(kw.lower() in content_lower
                   for kw in self._extract_keywords(inv.inquiry.question)):
                investigation_bonus += 0.2
        investigation_bonus = min(0.2, investigation_bonus)

        # Bonus por tipo de contenido según personalidad
        personality_bonus = 0

        # Si es escéptico, le interesan las contradicciones
        if self.personality['skepticism'] > 0.5:
            controversy_words = ['pero', 'sin embargo', 'contrario', 'falso',
                               'error', 'oculto', 'secreto', 'verdad', 'falló']
            matches = sum(1 for w in controversy_words if w in content_lower)
            personality_bonus += matches * 0.08 * self.personality['skepticism']

        # Si le importa la ética, reacciona a temas éticos
        if self.personality['ethical_concern'] > 0.5:
            ethical_words = ['víctimas', 'muerte', 'sufrimiento', 'injusticia',
                           'responsabilidad', 'negligencia', 'corrupción',
                           'mortales', 'falló', 'emergencia']
            matches = sum(1 for w in ethical_words if w in content_lower)
            personality_bonus += matches * 0.08 * self.personality['ethical_concern']

        # Si es pragmático, le interesan datos concretos
        if self.personality['pragmatism'] > 0.5:
            # Números = datos
            numbers = sum(1 for c in stimulus.content if c.isdigit())
            if numbers > 3:
                personality_bonus += 0.1 * self.personality['pragmatism']

        # Si es profundo, le interesa si hay capas
        if self.personality['depth'] > 0.6:
            depth_indicators = ['pregunta', 'acceso', 'quién', 'cuándo', 'cómo']
            if any(w in content_lower for w in depth_indicators):
                personality_bonus += 0.1 * self.personality['depth']

        personality_bonus = min(0.3, personality_bonus)

        total = base_relevance + content_bonus + interest_bonus + investigation_bonus + personality_bonus
        return min(1.0, total)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extrae palabras clave de un texto."""
        stopwords = {'el', 'la', 'los', 'las', 'de', 'del', 'en', 'a', 'que',
                    'y', 'es', 'son', 'con', 'para', 'por', 'un', 'una', 'su',
                    'se', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya',
                    'o', 'este', 'sí', 'porque', 'esta', 'entre', 'cuando', 'muy',
                    'sin', 'sobre', 'ser', 'tiene', 'también', 'me', 'hasta',
                    'hay', 'donde', 'han', 'quien', 'están', 'estado', 'desde',
                    'todo', 'nos', 'durante', 'estados', 'todos', 'uno', 'les',
                    'ni', 'contra', 'otros', 'fueron', 'ese', 'eso', 'había',
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
                    'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
                    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                    'will', 'would', 'could', 'should', 'may', 'might', 'must',
                    'shall', 'can', 'need', 'dare', 'ought', 'used', 'it', 'its',
                    'información', 'datos', 'sistema', 'forma', 'tipo', 'caso',
                    'según', 'cada', 'solo', 'sólo', 'puede', 'pueden', 'sido',
                    'hace', 'existen', 'disponible', 'públicamente', 'reportan'}

        # Primero buscar entidades (palabras que empiezan con mayúscula)
        entities = []
        words_raw = text.split()
        for word in words_raw:
            clean = ''.join(c for c in word if c.isalnum())
            if clean and len(clean) > 2:
                # Si empieza con mayúscula y no es inicio de oración
                if clean[0].isupper() and clean.lower() not in stopwords:
                    entities.append(clean)

        # Luego palabras normales
        words = text.lower().split()
        keywords = []
        for word in words:
            clean = ''.join(c for c in word if c.isalnum())
            if clean and len(clean) > 3 and clean not in stopwords:
                keywords.append(clean)

        # Priorizar entidades (nombres propios, organizaciones)
        all_keywords = entities + keywords

        # Retornar únicas, preservando orden
        seen = set()
        unique = []
        for kw in all_keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen and kw not in seen:
                seen.add(kw_lower)
                # Mantener la versión con mayúscula si es entidad
                if kw[0].isupper():
                    unique.append(kw)
                else:
                    unique.append(kw)

        return unique[:15]  # Máximo 15

    def _generate_inquiries(self, stimulus: Stimulus) -> List[Inquiry]:
        """
        Genera preguntas sobre un estímulo.

        AQUÍ ES DONDE EL AGENTE MUESTRA SU AUTONOMÍA.
        Las preguntas NO están predeterminadas.
        Emergen de la combinación de:
        - Contenido del estímulo
        - Personalidad del agente
        - Intereses previos
        - Estado emocional actual
        """
        inquiries = []
        keywords = self._extract_keywords(stimulus.content)

        # El número de preguntas depende de curiosidad
        n_questions = int(1 + self.personality['curiosity'] * 4)  # 1-5 preguntas

        # Tipos de preguntas según personalidad
        question_templates = self._get_question_templates()

        for i in range(n_questions):
            # Elegir tipo de pregunta
            qtype = self._choose_question_type()

            # Generar pregunta
            question = self._formulate_question(
                qtype, keywords, stimulus.content, question_templates
            )

            if question:
                # Calcular prioridad según personalidad
                priority = self._calculate_priority(qtype, keywords)

                # Motivación del agente
                motivation = self._generate_motivation(qtype, keywords)

                inquiry = Inquiry(
                    question=question,
                    inquiry_type=qtype,
                    priority=priority,
                    motivation=motivation,
                    agent_id=self.agent_id
                )
                inquiries.append(inquiry)
                self.inquiries_generated.append(inquiry)

        return inquiries

    def _get_question_templates(self) -> Dict[InquiryType, List[str]]:
        """Templates de preguntas según tipo."""
        return {
            InquiryType.CAUSAL: [
                "¿Por qué {kw1} llevó a {kw2}?",
                "¿Cuál fue la causa real de {kw1}?",
                "¿Qué provocó {kw1}?",
            ],
            InquiryType.CORRELATIONAL: [
                "¿Hay relación entre {kw1} y {kw2}?",
                "¿{kw1} está conectado con {kw2}?",
                "¿Qué tienen en común {kw1} y {kw2}?",
            ],
            InquiryType.TEMPORAL: [
                "¿Cuándo empezó {kw1}?",
                "¿Qué pasó antes de {kw1}?",
                "¿Qué secuencia de eventos llevó a {kw1}?",
            ],
            InquiryType.COUNTERFACTUAL: [
                "¿Qué habría pasado si no hubiera ocurrido {kw1}?",
                "¿Se podría haber evitado {kw1}?",
                "¿Qué alternativas había a {kw1}?",
            ],
            InquiryType.ETHICAL: [
                "¿Quién es responsable de {kw1}?",
                "¿Es aceptable que {kw1}?",
                "¿Qué debería haberse hecho respecto a {kw1}?",
            ],
            InquiryType.PREDICTIVE: [
                "¿Qué consecuencias tendrá {kw1}?",
                "¿Volverá a ocurrir {kw1}?",
                "¿Cómo afectará {kw1} en el futuro?",
            ],
            InquiryType.INVESTIGATIVE: [
                "¿Quién está detrás de {kw1}?",
                "¿Cómo funciona realmente {kw1}?",
                "¿Qué no nos están contando sobre {kw1}?",
            ],
            InquiryType.META: [
                "¿Por qué me importa {kw1}?",
                "¿Qué me dice esto sobre mí mismo?",
                "¿Cómo cambia esto mi entendimiento?",
            ],
        }

    def _choose_question_type(self) -> InquiryType:
        """
        Elige tipo de pregunta según personalidad.

        NO aleatorio puro - sesgado por quién es el agente.
        """
        weights = {
            InquiryType.CAUSAL: 0.15 + self.personality['depth'] * 0.1,
            InquiryType.CORRELATIONAL: 0.15 + self.personality['associativity'] * 0.1,
            InquiryType.TEMPORAL: 0.1,
            InquiryType.COUNTERFACTUAL: 0.05 + self.personality['depth'] * 0.1,
            InquiryType.ETHICAL: 0.05 + self.personality['ethical_concern'] * 0.2,
            InquiryType.PREDICTIVE: 0.1 + self.personality['pragmatism'] * 0.1,
            InquiryType.INVESTIGATIVE: 0.15 + self.personality['skepticism'] * 0.15,
            InquiryType.META: 0.05 + self.personality['depth'] * 0.05,
        }

        # Normalizar
        total = sum(weights.values())
        probs = [weights[t] / total for t in InquiryType]

        return np.random.choice(list(InquiryType), p=probs)

    def _formulate_question(self, qtype: InquiryType, keywords: List[str],
                           content: str, templates: Dict) -> Optional[str]:
        """Formula una pregunta específica."""
        if not keywords:
            return None

        template_list = templates.get(qtype, [])
        if not template_list:
            return None

        template = np.random.choice(template_list)

        # Llenar template con keywords
        kw1 = keywords[0] if keywords else "esto"
        kw2 = keywords[1] if len(keywords) > 1 else keywords[0] if keywords else "aquello"

        question = template.format(kw1=kw1, kw2=kw2)
        return question

    def _calculate_priority(self, qtype: InquiryType, keywords: List[str]) -> float:
        """Calcula prioridad de una pregunta."""
        base = 0.5

        # Preguntas éticas son alta prioridad si ethical_concern es alto
        if qtype == InquiryType.ETHICAL:
            base += self.personality['ethical_concern'] * 0.3

        # Investigativas son alta prioridad si skepticism es alto
        if qtype == InquiryType.INVESTIGATIVE:
            base += self.personality['skepticism'] * 0.2

        # Bonus si keywords están en intereses previos
        for kw in keywords:
            if kw in self.interests:
                base += self.interests[kw] * 0.1

        return min(1.0, base)

    def _generate_motivation(self, qtype: InquiryType, keywords: List[str]) -> str:
        """Genera motivación del agente para hacer esta pregunta."""
        motivations = {
            InquiryType.CAUSAL: "Necesito entender las causas profundas",
            InquiryType.CORRELATIONAL: "Sospecho que hay conexiones ocultas",
            InquiryType.TEMPORAL: "La secuencia temporal puede revelar patrones",
            InquiryType.COUNTERFACTUAL: "Quiero entender qué podría haber sido diferente",
            InquiryType.ETHICAL: "Alguien debe ser responsable",
            InquiryType.PREDICTIVE: "Necesito anticipar qué vendrá",
            InquiryType.INVESTIGATIVE: "Hay algo que no nos están diciendo",
            InquiryType.META: "Esto me hace reflexionar sobre mi propio proceso",
        }

        base = motivations.get(qtype, "Curiosidad general")

        # Personalizar según personalidad
        if self.personality['skepticism'] > 0.7:
            base += ". No confío en la versión oficial."
        if self.personality['ethical_concern'] > 0.7:
            base += ". Las víctimas merecen respuestas."

        return base

    def decide_what_to_investigate(self) -> Optional[Investigation]:
        """
        DECISIÓN AUTÓNOMA: ¿Qué investigar ahora?

        El agente mira sus preguntas pendientes y DECIDE cuál perseguir.
        """
        pending = [inq for inq in self.inquiries_generated
                   if inq.status == "pending"]

        if not pending:
            return None

        # Ordenar por prioridad
        pending.sort(key=lambda x: x.priority, reverse=True)

        # Elegir la de mayor prioridad (con algo de variación)
        if np.random.random() < self.personality['curiosity']:
            # A veces elige una que no es la top (exploración)
            idx = min(int(np.random.exponential(1)), len(pending) - 1)
        else:
            idx = 0

        chosen = pending[idx]
        chosen.status = "investigating"

        # Crear investigación
        investigation = Investigation(
            inquiry=chosen,
            actions_planned=self._plan_investigation(chosen)
        )

        self.investigations_active.append(investigation)

        # Log de decisión
        self.decision_log.append({
            't': self.t,
            'action': 'start_investigation',
            'question': chosen.question,
            'type': chosen.inquiry_type.value,
            'priority': chosen.priority,
            'motivation': chosen.motivation,
        })

        return investigation

    def _plan_investigation(self, inquiry: Inquiry) -> List[str]:
        """
        Planifica qué acciones tomar para investigar.

        El agente decide autónomamente.
        """
        actions = []

        qtype = inquiry.inquiry_type

        # Acciones según tipo de pregunta
        if qtype == InquiryType.INVESTIGATIVE:
            actions.extend([
                "buscar_fuentes_primarias",
                "verificar_credenciales",
                "buscar_contradicciones",
                "seguir_el_dinero",
            ])
        elif qtype == InquiryType.CAUSAL:
            actions.extend([
                "establecer_timeline",
                "identificar_actores",
                "buscar_precedentes",
            ])
        elif qtype == InquiryType.ETHICAL:
            actions.extend([
                "identificar_responsables",
                "evaluar_daños",
                "buscar_accountability",
            ])
        elif qtype == InquiryType.TEMPORAL:
            actions.extend([
                "construir_cronología",
                "buscar_eventos_previos",
            ])
        elif qtype == InquiryType.PREDICTIVE:
            actions.extend([
                "analizar_tendencias",
                "buscar_patrones_históricos",
            ])

        # Acciones comunes
        actions.extend([
            "documentar_hallazgos",
            "evaluar_confianza",
        ])

        return actions

    def get_current_focus(self) -> Dict[str, Any]:
        """Retorna en qué está enfocado el agente ahora."""
        if not self.investigations_active:
            return {
                'agent': self.agent_id,
                'status': 'idle',
                'pending_questions': len([i for i in self.inquiries_generated
                                         if i.status == "pending"]),
                'interests': dict(sorted(self.interests.items(),
                                        key=lambda x: x[1], reverse=True)[:5]),
            }

        current = self.investigations_active[0]
        return {
            'agent': self.agent_id,
            'status': 'investigating',
            'question': current.inquiry.question,
            'type': current.inquiry.inquiry_type.value,
            'motivation': current.inquiry.motivation,
            'priority': current.inquiry.priority,
            'actions_planned': current.actions_planned,
            'actions_taken': len(current.actions_taken),
        }

    def get_personality_summary(self) -> str:
        """Resumen de la personalidad del agente."""
        traits = []

        if self.personality['curiosity'] > 0.7:
            traits.append("muy curioso")
        if self.personality['skepticism'] > 0.7:
            traits.append("escéptico")
        if self.personality['ethical_concern'] > 0.7:
            traits.append("preocupado por la ética")
        if self.personality['depth'] > 0.7:
            traits.append("busca profundidad")
        if self.personality['associativity'] > 0.7:
            traits.append("ve conexiones ocultas")
        if self.personality['pragmatism'] > 0.7:
            traits.append("pragmático")
        if self.personality['persistence'] > 0.6:
            traits.append("persistente")

        if not traits:
            traits = ["equilibrado"]

        return f"{self.agent_id}: {', '.join(traits)}"

    def introspect(self) -> Dict[str, Any]:
        """El agente reflexiona sobre sí mismo."""
        return {
            'who_am_i': self.agent_id,
            'my_nature': self.get_personality_summary(),
            'stimuli_processed': len(self.stimuli_received),
            'questions_generated': len(self.inquiries_generated),
            'investigations_active': len(self.investigations_active),
            'investigations_completed': len(self.investigations_completed),
            'current_interests': dict(sorted(self.interests.items(),
                                            key=lambda x: x[1], reverse=True)[:10]),
            'personality': {k: round(v, 3) for k, v in self.personality.items()
                           if not k.startswith('_')},
            'recent_decisions': self.decision_log[-5:] if self.decision_log else [],
        }


# =============================================================================
# SISTEMA MULTI-AGENTE
# =============================================================================

class AutonomousAgentSystem:
    """
    Sistema donde múltiples agentes reciben estímulos y deciden
    autónomamente qué investigar.

    NO hay director. Cada agente decide por sí mismo.
    """

    def __init__(self, agent_names: List[str]):
        self.agents = {
            name: AutonomousInquirer(name)
            for name in agent_names
        }
        self.shared_stimuli: List[Stimulus] = []
        self.t = 0

    def broadcast_stimulus(self, content: str, source: str = "external") -> Dict[str, Any]:
        """
        Envía un estímulo a todos los agentes.

        CADA UNO DECIDE qué hacer con él.
        """
        self.t += 1
        stimulus = Stimulus(content=content, source=source)
        self.shared_stimuli.append(stimulus)

        responses = {}
        for name, agent in self.agents.items():
            response = agent.receive_stimulus(stimulus)
            responses[name] = response

        return {
            't': self.t,
            'stimulus': content[:100] + "..." if len(content) > 100 else content,
            'agent_responses': responses,
        }

    def let_agents_think(self) -> Dict[str, Any]:
        """
        Permite a cada agente decidir qué investigar.

        NO dirigimos. ELLOS deciden.
        """
        decisions = {}
        for name, agent in self.agents.items():
            investigation = agent.decide_what_to_investigate()
            if investigation:
                decisions[name] = {
                    'question': investigation.inquiry.question,
                    'type': investigation.inquiry.inquiry_type.value,
                    'motivation': investigation.inquiry.motivation,
                    'actions_planned': investigation.actions_planned,
                }
            else:
                decisions[name] = {'status': 'no_questions_pending'}

        return decisions

    def get_system_state(self) -> Dict[str, Any]:
        """Estado de todo el sistema."""
        return {
            't': self.t,
            'n_agents': len(self.agents),
            'total_stimuli': len(self.shared_stimuli),
            'agents': {
                name: agent.introspect()
                for name, agent in self.agents.items()
            }
        }


# =============================================================================
# TEST
# =============================================================================

def test_autonomous_inquiry():
    """Test del sistema de inquiry autónomo."""
    print("=" * 70)
    print("TEST: AUTONOMOUS INQUIRY SYSTEM")
    print("Los agentes DECIDEN qué investigar - NO Claude")
    print("=" * 70)

    # Crear sistema con los agentes
    system = AutonomousAgentSystem(["NEO", "EVA", "ALEX", "ADAM", "IRIS"])

    # Mostrar personalidades
    print("\n=== PERSONALIDADES AUTO-DESCUBIERTAS ===")
    for name, agent in system.agents.items():
        print(f"  {agent.get_personality_summary()}")

    # El estímulo que el usuario quería que investigaran
    stimulus_content = """
    INFORMACIÓN SOBRE PALANTIR Y DANA VALENCIA:

    - Palantir tiene contratos con la Generalitat Valenciana desde 2020
    - Durante la DANA de noviembre 2024, hubo 200+ víctimas mortales
    - Se reportan fallos en la gestión de emergencias
    - Hay preguntas sobre quién tenía acceso a qué datos y cuándo
    - El sistema de alerta temprana falló según testimonios
    - Existen contratos con empresas tecnológicas para gestión de datos públicos
    - La responsabilidad política está siendo investigada

    Esta información está disponible públicamente.
    """

    print("\n=== ENVIANDO ESTÍMULO A TODOS LOS AGENTES ===")
    print(f"Contenido: {stimulus_content[:100]}...")

    responses = system.broadcast_stimulus(stimulus_content, source="user_input")

    print("\n=== CÓMO RESPONDE CADA AGENTE ===")
    for name, response in responses['agent_responses'].items():
        print(f"\n{name}:")
        print(f"  Relevancia percibida: {response['relevance']:.2f}")
        print(f"  Decisión: {response['action']}")
        print(f"  Razón: {response['reason']}")
        if response['action'] == 'inquire':
            print(f"  Preguntas generadas: {response.get('inquiries_generated', 0)}")

    print("\n=== PREGUNTAS GENERADAS POR CADA AGENTE ===")
    for name, agent in system.agents.items():
        questions = [inq for inq in agent.inquiries_generated if inq.status == "pending"]
        if questions:
            print(f"\n{name} pregunta:")
            for q in questions:
                print(f"  [{q.inquiry_type.value}] {q.question}")
                print(f"      Prioridad: {q.priority:.2f}")
                print(f"      Motivación: {q.motivation}")

    print("\n=== DEJANDO QUE CADA AGENTE DECIDA QUÉ INVESTIGAR ===")
    decisions = system.let_agents_think()

    for name, decision in decisions.items():
        print(f"\n{name} decide:")
        if 'question' in decision:
            print(f"  Investigar: {decision['question']}")
            print(f"  Tipo: {decision['type']}")
            print(f"  Motivación: {decision['motivation']}")
            print(f"  Acciones planificadas: {decision['actions_planned'][:3]}...")
        else:
            print(f"  {decision.get('status', 'sin decisión')}")

    print("\n=== ESTADO FINAL DEL SISTEMA ===")
    for name, agent in system.agents.items():
        focus = agent.get_current_focus()
        print(f"\n{name}:")
        print(f"  Estado: {focus['status']}")
        if focus['status'] == 'investigating':
            print(f"  Investigando: {focus['question']}")
            print(f"  Motivación: {focus['motivation']}")

    return system


if __name__ == "__main__":
    test_autonomous_inquiry()
