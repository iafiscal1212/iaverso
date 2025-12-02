"""
Percepción del Mundo Creativo (Perception)
==========================================

Ver lo que otros han creado.

Percibir no es solo "tener datos de".
Percibir es:
1. Detectar que algo existe
2. Reconocer qué tipo de cosa es
3. Evaluar si es relevante para mí
4. Sentir algo respecto a ello (resonancia)
5. Decidir si interactuar

La percepción es selectiva y personal:
- Un agente no ve "todo" - ve lo que puede ver desde su posición
- Un agente no ve "igual" - interpreta según su identidad
- Un agente no ve "neutral" - siente atracción/indiferencia/rechazo

La percepción de creaciones ajenas puede:
- Inspirar nuevas ideas
- Modificar el estado interno
- Provocar respuestas creativas
- Generar conexiones entre agentes

100% endógeno. Sin números mágicos.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import sys

sys.path.insert(0, '/root/NEO_EVA')

from genesis.materialization import MaterializedObject, ObjectType
from genesis.shared_world import SharedWorld


class PerceptionType(Enum):
    """Tipos de percepción según la relación con el objeto."""
    ATTRACTION = "attraction"    # Me atrae, quiero acercarme
    CURIOSITY = "curiosity"      # Me intriga, quiero entender
    RESONANCE = "resonance"      # Resuena conmigo, me reconozco
    INDIFFERENCE = "indifference"  # No me dice nada
    TENSION = "tension"          # Me genera tensión, conflicto
    INSPIRATION = "inspiration"  # Me inspira a crear


@dataclass
class PerceivedObject:
    """Un objeto tal como lo percibe un agente específico."""
    # El objeto real
    obj: MaterializedObject

    # Percepción espacial
    distance: float             # Distancia al agente
    direction: np.ndarray       # Dirección hacia el objeto
    in_focus: bool              # Si está en el foco de atención

    # Percepción semántica
    perceived_form: np.ndarray  # Cómo lo "ve" (puede diferir del original)
    similarity_to_self: float   # Cuánto se parece a mi identidad
    novelty_for_me: float       # Cuán nuevo es para mí

    # Respuesta emocional/evaluativa
    perception_type: PerceptionType
    intensity: float            # Intensidad de la percepción
    valence: float              # Positivo/negativo [-1, 1]

    # Potencial
    inspiration_potential: float  # Probabilidad de que inspire
    interaction_urge: float     # Urgencia de interactuar


@dataclass
class PerceptionField:
    """Estado del campo perceptual de un agente."""
    agent_id: str
    t: int
    position: np.ndarray
    focus_direction: np.ndarray
    focus_radius: float
    n_perceived: int
    dominant_perception: PerceptionType
    total_inspiration: float


class CreativePerception:
    """
    Sistema de percepción para el mundo creativo.

    Permite a los agentes:
    1. Ver objetos en el mundo
    2. Interpretarlos según su identidad
    3. Sentir algo respecto a ellos
    4. Potencialmente inspirarse
    """

    def __init__(self, world: SharedWorld):
        """
        Inicializa el sistema de percepción.

        Args:
            world: El mundo compartido que se percibe
        """
        self.world = world
        self.t = 0
        self.eps = np.finfo(float).eps

        # Historial de percepciones por agente
        self._perception_history: Dict[str, List[PerceivedObject]] = {}

        # Objetos ya vistos por cada agente
        self._seen_objects: Dict[str, set] = {}

        # Estado del foco de atención
        self._focus_direction: Dict[str, np.ndarray] = {}
        self._focus_radius: Dict[str, float] = {}

    def _register_agent(self, agent_id: str):
        """Registra un agente nuevo."""
        if agent_id not in self._perception_history:
            self._perception_history[agent_id] = []
            self._seen_objects[agent_id] = set()
            self._focus_direction[agent_id] = np.zeros(self.world.dim)
            self._focus_radius[agent_id] = 1.0

    def _compute_perceived_form(
        self,
        obj: MaterializedObject,
        identity: np.ndarray,
        agent_id: str
    ) -> np.ndarray:
        """
        Calcula cómo el agente "ve" el objeto.

        La percepción no es objetiva. Está filtrada por
        la identidad del perceptor.

        El agente "proyecta" un poco de sí mismo en lo que ve.
        """
        original = obj.form

        # Asegurar dimensiones compatibles
        if len(identity) != len(original):
            # Proyectar a dimensión común
            min_dim = min(len(identity), len(original))
            identity = identity[:min_dim]
            original = original[:min_dim]

        # Filtro perceptual: mezcla de original con identidad
        # Cuánto se mezcla depende de la similitud
        similarity = np.dot(original, identity) / (
            np.linalg.norm(original) * np.linalg.norm(identity) + self.eps
        )
        similarity = (similarity + 1) / 2  # Normalizar a [0, 1]

        # Más similar = veo más el original
        # Menos similar = proyecto más mi identidad
        blend = similarity * original + (1 - similarity) * 0.1 * identity

        # Normalizar
        blend = blend / (np.linalg.norm(blend) + self.eps) * np.linalg.norm(original)

        return blend

    def _compute_similarity_to_self(
        self,
        perceived_form: np.ndarray,
        identity: np.ndarray
    ) -> float:
        """Calcula similitud entre lo percibido y la identidad."""
        # Asegurar dimensiones
        min_dim = min(len(perceived_form), len(identity))

        sim = np.dot(perceived_form[:min_dim], identity[:min_dim]) / (
            np.linalg.norm(perceived_form[:min_dim]) *
            np.linalg.norm(identity[:min_dim]) + self.eps
        )

        return float((sim + 1) / 2)  # Normalizar a [0, 1]

    def _compute_novelty_for_agent(
        self,
        obj: MaterializedObject,
        agent_id: str
    ) -> float:
        """
        Calcula cuán novel es este objeto para el agente.

        Basado en:
        - Si ya lo ha visto antes
        - Similitud con objetos que ya ha visto
        """
        seen = self._seen_objects.get(agent_id, set())

        # Si ya lo vio, novedad = 0
        if obj.id in seen:
            return 0.0

        # Si no ha visto nada, todo es novel
        if not seen:
            return 1.0

        # Calcular similitud con objetos vistos
        seen_objects = [
            self.world._objects[oid]
            for oid in seen
            if oid in self.world._objects
        ]

        if not seen_objects:
            return 1.0

        similarities = []
        for seen_obj in seen_objects:
            sim = np.dot(obj.form, seen_obj.form) / (
                np.linalg.norm(obj.form) *
                np.linalg.norm(seen_obj.form) + self.eps
            )
            similarities.append((sim + 1) / 2)

        # Novedad = 1 - máxima similitud
        max_sim = max(similarities)
        return float(1 - max_sim)

    def _determine_perception_type(
        self,
        similarity_to_self: float,
        novelty: float,
        distance: float,
        obj: MaterializedObject,
        agent_id: str
    ) -> Tuple[PerceptionType, float, float]:
        """
        Determina el tipo de percepción, intensidad y valencia.

        Basado en la combinación de similitud, novedad y distancia.
        """
        # Cercanía normalizada (más cerca = más intenso)
        closeness = 1.0 / (1.0 + distance)

        # Combinaciones que determinan el tipo
        if similarity_to_self > 0.8 and novelty < 0.3:
            # Muy similar y no novel: resonancia
            ptype = PerceptionType.RESONANCE
            intensity = similarity_to_self * closeness
            valence = 0.7  # Positivo

        elif novelty > 0.7 and similarity_to_self > 0.4:
            # Muy novel pero algo similar: inspiración
            ptype = PerceptionType.INSPIRATION
            intensity = novelty * closeness
            valence = 0.9  # Muy positivo

        elif novelty > 0.7 and similarity_to_self < 0.3:
            # Muy novel y muy diferente: curiosidad o tensión
            if closeness > 0.5:
                ptype = PerceptionType.TENSION
                intensity = novelty * closeness
                valence = -0.3  # Ligeramente negativo
            else:
                ptype = PerceptionType.CURIOSITY
                intensity = novelty * 0.5
                valence = 0.3  # Ligeramente positivo

        elif similarity_to_self > 0.5:
            # Similar: atracción
            ptype = PerceptionType.ATTRACTION
            intensity = similarity_to_self * closeness
            valence = 0.5

        elif similarity_to_self < 0.3 and novelty < 0.3:
            # Ni similar ni novel: indiferencia
            ptype = PerceptionType.INDIFFERENCE
            intensity = 0.1
            valence = 0.0

        else:
            # Caso general: curiosidad
            ptype = PerceptionType.CURIOSITY
            intensity = (novelty + closeness) / 2
            valence = 0.2

        return ptype, float(intensity), float(valence)

    def _compute_inspiration_potential(
        self,
        perceived: PerceivedObject,
        agent_state: np.ndarray
    ) -> float:
        """
        Calcula el potencial de inspiración.

        Un objeto inspira si:
        - Es novel pero no alienante
        - Resuena algo con la identidad
        - Activa algo en el estado actual
        """
        if perceived.perception_type == PerceptionType.INDIFFERENCE:
            return 0.0

        # Base: novedad × similitud (máximo en el punto medio)
        base = 4 * perceived.novelty_for_me * perceived.similarity_to_self

        # Bonus por tipo de percepción
        type_bonus = {
            PerceptionType.INSPIRATION: 0.5,
            PerceptionType.CURIOSITY: 0.3,
            PerceptionType.RESONANCE: 0.2,
            PerceptionType.ATTRACTION: 0.1,
            PerceptionType.TENSION: 0.15,  # La tensión también puede inspirar
            PerceptionType.INDIFFERENCE: 0.0
        }

        bonus = type_bonus.get(perceived.perception_type, 0)

        # Activación: correlación con estado actual
        min_dim = min(len(perceived.perceived_form), len(agent_state))
        activation = np.corrcoef(
            perceived.perceived_form[:min_dim],
            agent_state[:min_dim]
        )[0, 1]
        if np.isnan(activation):
            activation = 0

        activation = (activation + 1) / 2  # Normalizar

        potential = base + bonus + 0.2 * activation

        return float(np.clip(potential, 0, 1))

    def _compute_interaction_urge(
        self,
        perceived: PerceivedObject
    ) -> float:
        """
        Calcula la urgencia de interactuar con el objeto.

        Basada en intensidad, valencia y distancia.
        """
        # Intensidad alta + valencia no neutral = urgencia
        urge = perceived.intensity * abs(perceived.valence)

        # Cercanía aumenta urgencia
        closeness = 1.0 / (1.0 + perceived.distance)
        urge *= (1 + closeness)

        # Tipos que generan más urgencia
        if perceived.perception_type in [
            PerceptionType.INSPIRATION,
            PerceptionType.TENSION,
            PerceptionType.ATTRACTION
        ]:
            urge *= 1.3

        return float(np.clip(urge, 0, 1))

    def perceive(
        self,
        agent_id: str,
        position: np.ndarray,
        identity: np.ndarray,
        state: np.ndarray,
        max_objects: int = 10
    ) -> List[PerceivedObject]:
        """
        Percibe el mundo desde la perspectiva de un agente.

        Args:
            agent_id: Identificador del agente
            position: Posición del agente en el mundo
            identity: Identidad del agente
            state: Estado actual del agente
            max_objects: Máximo de objetos a percibir

        Returns:
            Lista de objetos percibidos, ordenados por relevancia
        """
        self.t += 1
        self._register_agent(agent_id)

        # Obtener objetos visibles
        visible_objects = self.world.get_visible_objects()

        if not visible_objects:
            return []

        # Calcular percepciones
        perceptions = []

        for obj in visible_objects:
            # No percibir propias creaciones de la misma manera
            is_own = obj.creator_id == agent_id

            # Distancia
            distance = np.linalg.norm(obj.position - position)

            # Dirección
            direction = (obj.position - position)
            direction = direction / (np.linalg.norm(direction) + self.eps)

            # ¿En foco?
            focus_dir = self._focus_direction[agent_id]
            if np.linalg.norm(focus_dir) > self.eps:
                alignment = np.dot(direction, focus_dir)
                in_focus = alignment > 0.5  # Cono de ~60 grados
            else:
                in_focus = True  # Sin foco definido, todo está en foco

            # Forma percibida
            perceived_form = self._compute_perceived_form(obj, identity, agent_id)

            # Similitud
            similarity = self._compute_similarity_to_self(perceived_form, identity)

            # Novedad
            novelty = self._compute_novelty_for_agent(obj, agent_id)

            # Tipo de percepción
            ptype, intensity, valence = self._determine_perception_type(
                similarity, novelty, distance, obj, agent_id
            )

            # Si es propio, modificar percepción
            if is_own:
                ptype = PerceptionType.RESONANCE
                intensity *= 0.5  # Menos intenso
                novelty = 0.0  # No es novel

            # Crear objeto percibido
            perceived = PerceivedObject(
                obj=obj,
                distance=distance,
                direction=direction,
                in_focus=in_focus,
                perceived_form=perceived_form,
                similarity_to_self=similarity,
                novelty_for_me=novelty,
                perception_type=ptype,
                intensity=intensity,
                valence=valence,
                inspiration_potential=0.0,  # Se calcula después
                interaction_urge=0.0  # Se calcula después
            )

            # Calcular potenciales
            perceived.inspiration_potential = self._compute_inspiration_potential(
                perceived, state
            )
            perceived.interaction_urge = self._compute_interaction_urge(perceived)

            perceptions.append(perceived)

        # Ordenar por relevancia (intensidad × en_foco)
        perceptions.sort(
            key=lambda p: p.intensity * (1.5 if p.in_focus else 1.0),
            reverse=True
        )

        # Limitar
        perceptions = perceptions[:max_objects]

        # Registrar como vistos
        for p in perceptions:
            self._seen_objects[agent_id].add(p.obj.id)

        # Guardar en historial
        self._perception_history[agent_id].extend(perceptions)

        # Limitar historial
        max_history = 100
        if len(self._perception_history[agent_id]) > max_history:
            self._perception_history[agent_id] = \
                self._perception_history[agent_id][-max_history:]

        return perceptions

    def set_focus(
        self,
        agent_id: str,
        direction: np.ndarray,
        radius: float = 1.0
    ):
        """Establece el foco de atención de un agente."""
        self._register_agent(agent_id)
        norm = np.linalg.norm(direction)
        if norm > self.eps:
            self._focus_direction[agent_id] = direction / norm
        self._focus_radius[agent_id] = radius

    def get_most_inspiring(
        self,
        agent_id: str,
        n: int = 3
    ) -> List[PerceivedObject]:
        """Obtiene los objetos más inspiradores percibidos recientemente."""
        history = self._perception_history.get(agent_id, [])
        if not history:
            return []

        # Filtrar por potencial de inspiración
        sorted_by_inspiration = sorted(
            history,
            key=lambda p: p.inspiration_potential,
            reverse=True
        )

        return sorted_by_inspiration[:n]

    def get_perception_field(self, agent_id: str) -> Optional[PerceptionField]:
        """Obtiene el estado del campo perceptual de un agente."""
        self._register_agent(agent_id)

        history = self._perception_history.get(agent_id, [])
        position = self.world.get_agent_position(agent_id)

        if position is None:
            position = np.zeros(self.world.dim)

        # Tipo dominante
        if history:
            type_counts = {}
            for p in history[-20:]:  # Últimas 20 percepciones
                t = p.perception_type
                type_counts[t] = type_counts.get(t, 0) + 1
            dominant = max(type_counts.keys(), key=lambda k: type_counts[k])

            total_inspiration = sum(p.inspiration_potential for p in history[-20:])
        else:
            dominant = PerceptionType.INDIFFERENCE
            total_inspiration = 0.0

        return PerceptionField(
            agent_id=agent_id,
            t=self.t,
            position=position,
            focus_direction=self._focus_direction[agent_id],
            focus_radius=self._focus_radius[agent_id],
            n_perceived=len(self._seen_objects.get(agent_id, set())),
            dominant_perception=dominant,
            total_inspiration=total_inspiration
        )

    def get_statistics(self, agent_id: str) -> Dict[str, Any]:
        """Retorna estadísticas de percepción para un agente."""
        history = self._perception_history.get(agent_id, [])
        seen = self._seen_objects.get(agent_id, set())

        return {
            'objects_seen': len(seen),
            'perceptions_total': len(history),
            'mean_intensity': float(np.mean([p.intensity for p in history])) if history else 0,
            'mean_inspiration': float(np.mean([p.inspiration_potential for p in history])) if history else 0,
            'perception_types': {
                t.value: sum(1 for p in history if p.perception_type == t)
                for t in PerceptionType
            } if history else {}
        }
