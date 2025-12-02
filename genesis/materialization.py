"""
Materialización (Materialization)
==================================

El momento donde una idea se convierte en ALGO.

No basta tener la idea. No basta que resuene.
Para que exista en el mundo, debe materializarse.

Materializar es:
1. Invertir energía (la idea "cuesta" algo)
2. Darle forma persistente (no desaparece)
3. Colocarla en el espacio compartido (otros pueden verla)

Un objeto materializado tiene:
- Forma: estructura heredada de la idea
- Posición: dónde está en el mundo compartido
- Firma: quién lo creó
- Energía: cuánto costó crearlo
- Edad: cuánto tiempo lleva existiendo

El costo de materializar es endógeno:
- Depende de la complejidad de la idea
- Depende de la energía disponible del agente
- Depende de qué tan "lejos" está de ideas anteriores

100% endógeno. Sin números mágicos.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import sys

sys.path.insert(0, '/root/NEO_EVA')

from genesis.idea_field import Idea


class ObjectType(Enum):
    """Tipos de objetos materializados."""
    STRUCTURE = "structure"      # Estructura estática
    PATTERN = "pattern"          # Patrón dinámico
    LINK = "link"               # Conexión entre cosas
    FIELD = "field"             # Campo que afecta el espacio
    ARTIFACT = "artifact"       # Objeto compuesto


@dataclass
class MaterializedObject:
    """
    Un objeto que existe en el mundo compartido.

    Nació de una idea, pero ahora tiene existencia propia.
    """
    # Identidad
    id: str                     # Identificador único
    creator_id: str             # Quién lo creó
    source_idea: Idea           # Idea de la que nació

    # Forma
    form: np.ndarray            # Representación vectorial
    structure: np.ndarray       # Estructura relacional
    object_type: ObjectType     # Tipo de objeto

    # Posición en el mundo
    position: np.ndarray        # Coordenadas en espacio compartido

    # Energía
    creation_cost: float        # Cuánto costó crearlo
    current_energy: float       # Energía actual (puede decaer)

    # Tiempo
    t_created: int              # Cuándo se creó
    t_last_interaction: int     # Última interacción

    # Estado
    visible: bool = True        # ¿Se puede ver?
    stable: bool = True         # ¿Es estable o está decayendo?

    # Interacciones
    interactions: int = 0       # Veces que otros interactuaron
    observers: List[str] = field(default_factory=list)  # Quién lo ha visto


@dataclass
class MaterializationResult:
    """Resultado de un intento de materialización."""
    success: bool
    obj: Optional[MaterializedObject]
    cost_paid: float
    energy_remaining: float
    reason: str = ""


class Materializer:
    """
    Sistema de materialización de ideas.

    Convierte ideas adoptadas en objetos que existen
    en el espacio compartido.

    La materialización NO es automática.
    Requiere:
    - Energía suficiente
    - Decisión de materializar (implícita si hay resonancia)
    - Espacio disponible
    """

    def __init__(self, world_dim: int = 3):
        """
        Inicializa el materializador.

        Args:
            world_dim: Dimensiones del espacio compartido
        """
        self.world_dim = world_dim
        self.t = 0
        self.eps = np.finfo(float).eps

        # Objetos creados
        self._objects: Dict[str, MaterializedObject] = {}

        # Energía por agente
        self._agent_energy: Dict[str, float] = {}

        # Historial de materializaciones
        self._history: Dict[str, List[str]] = {}  # agent_id -> [object_ids]

        # Contador de IDs
        self._id_counter = 0

    def _generate_id(self) -> str:
        """Genera un ID único para un objeto."""
        self._id_counter += 1
        return f"obj_{self._id_counter:06d}"

    def _register_agent(self, agent_id: str, initial_energy: float = 1.0):
        """Registra un agente nuevo."""
        if agent_id not in self._agent_energy:
            self._agent_energy[agent_id] = initial_energy
            self._history[agent_id] = []

    def set_agent_energy(self, agent_id: str, energy: float):
        """Establece la energía de un agente."""
        self._register_agent(agent_id)
        self._agent_energy[agent_id] = max(0, energy)

    def add_agent_energy(self, agent_id: str, delta: float):
        """Añade (o resta) energía a un agente."""
        self._register_agent(agent_id)
        self._agent_energy[agent_id] = max(0, self._agent_energy[agent_id] + delta)

    def get_agent_energy(self, agent_id: str) -> float:
        """Retorna la energía actual de un agente."""
        return self._agent_energy.get(agent_id, 0.0)

    def _compute_cost(
        self,
        idea: Idea,
        agent_id: str
    ) -> float:
        """
        Calcula el costo de materializar una idea.

        Costo endógeno basado en:
        1. Complejidad de la idea (entropía de estructura)
        2. Novedad (ideas muy nuevas cuestan más)
        3. Tamaño (norma del vector)
        4. Distancia a materializaciones previas

        C = base * (1 + complejidad) * (1 + novedad) * factor_distancia
        """
        # Complejidad: entropía de la estructura
        struct_flat = np.abs(idea.structure.flatten())
        struct_norm = struct_flat / (np.sum(struct_flat) + self.eps)
        entropy = -np.sum(struct_norm * np.log(struct_norm + self.eps))
        max_entropy = np.log(len(struct_flat))
        complexity = entropy / (max_entropy + self.eps)

        # Base: proporcional a la norma del vector
        base = np.linalg.norm(idea.vector) / np.sqrt(len(idea.vector))

        # Factor de novedad
        novelty_factor = 1 + idea.novelty / 2  # Novedad 2σ → factor 2

        # Factor de distancia a creaciones previas
        prev_objects = [
            self._objects[oid]
            for oid in self._history.get(agent_id, [])
            if oid in self._objects
        ]

        if prev_objects:
            # Distancia mínima a objetos previos
            distances = [
                np.linalg.norm(idea.vector - o.form)
                for o in prev_objects
            ]
            min_dist = min(distances)
            mean_norm = np.mean([np.linalg.norm(o.form) for o in prev_objects])
            dist_factor = 1 + min_dist / (mean_norm + self.eps)
        else:
            # Primera materialización: factor = 1 + 1/2 (entre 1 y 2)
            # Endógeno: punto medio del rango [1, 2]
            dist_factor = 1 + 1 / 2

        cost = base * (1 + complexity) * novelty_factor * dist_factor

        # Normalizar a rango razonable [0.1, 1.0]
        cost = np.clip(cost / 10, 0.1, 1.0)

        return float(cost)

    def _compute_position(
        self,
        idea: Idea,
        agent_id: str
    ) -> np.ndarray:
        """
        Calcula la posición del objeto en el mundo compartido.

        La posición depende de:
        1. El contenido de la idea (primeras componentes)
        2. Las posiciones de objetos previos del agente
        3. Un poco de variación basada en el tiempo

        NO es aleatorio, es determinista dado el estado.
        """
        # Base: proyección de la idea al espacio del mundo
        idea_vec = idea.vector
        if len(idea_vec) >= self.world_dim:
            base_pos = idea_vec[:self.world_dim]
        else:
            base_pos = np.concatenate([
                idea_vec,
                np.zeros(self.world_dim - len(idea_vec))
            ])

        # Normalizar al rango del mundo
        base_pos = base_pos / (np.linalg.norm(base_pos) + self.eps)

        # Offset basado en creaciones previas
        prev_objects = [
            self._objects[oid]
            for oid in self._history.get(agent_id, [])
            if oid in self._objects
        ]

        if prev_objects:
            # Centro de creaciones previas
            prev_positions = np.array([o.position for o in prev_objects])
            center = np.mean(prev_positions, axis=0)

            # Nueva posición: entre base y cerca del centro
            position = 0.7 * base_pos + 0.3 * (center / (np.linalg.norm(center) + self.eps))
        else:
            position = base_pos

        return position

    def _determine_type(self, idea: Idea) -> ObjectType:
        """
        Determina el tipo de objeto basado en la idea.

        Basado en las características de la estructura.
        Umbrales endógenos: tercios (1/3, 2/3) y mitad (1/2).
        """
        struct = idea.structure

        # Diagonal dominante → estructura
        diag_strength = np.sum(np.abs(np.diag(struct)))
        total_strength = np.sum(np.abs(struct)) + self.eps
        diag_ratio = diag_strength / total_strength

        # Simetría → patrón
        symmetry = 1 - np.linalg.norm(struct - struct.T) / (np.linalg.norm(struct) + self.eps)

        # Sparsity → link
        # Umbral endógeno: percentil basado en estadística de la estructura
        abs_struct = np.abs(struct)
        sparsity_threshold = np.percentile(abs_struct, 50)  # Mediana
        nonzero_ratio = np.count_nonzero(abs_struct > sparsity_threshold) / struct.size

        # Umbrales endógenos basados en tercios
        LOW = 1 / 3    # ~0.33
        HIGH = 2 / 3   # ~0.67

        if nonzero_ratio < LOW:
            return ObjectType.LINK
        elif diag_ratio > HIGH:
            return ObjectType.STRUCTURE
        elif symmetry > HIGH:
            return ObjectType.PATTERN
        elif idea.coherence > HIGH:
            return ObjectType.FIELD
        else:
            return ObjectType.ARTIFACT

    def can_materialize(
        self,
        idea: Idea,
        agent_id: str
    ) -> Tuple[bool, float, str]:
        """
        Verifica si un agente puede materializar una idea.

        Returns:
            (puede, costo, razón)
        """
        self._register_agent(agent_id)

        # Calcular costo
        cost = self._compute_cost(idea, agent_id)

        # Verificar energía
        energy = self._agent_energy[agent_id]

        if energy < cost:
            return False, cost, f"Insufficient energy: {energy:.3f} < {cost:.3f}"

        # Verificar que la idea no esté ya materializada
        if idea.materialized:
            return False, cost, "Idea already materialized"

        return True, cost, "OK"

    def materialize(
        self,
        idea: Idea,
        agent_id: str
    ) -> MaterializationResult:
        """
        Materializa una idea en un objeto.

        Args:
            idea: La idea a materializar
            agent_id: El agente que materializa

        Returns:
            MaterializationResult con el objeto creado (o None si falla)
        """
        self.t += 1
        self._register_agent(agent_id)

        # Verificar si puede materializar
        can, cost, reason = self.can_materialize(idea, agent_id)

        if not can:
            return MaterializationResult(
                success=False,
                obj=None,
                cost_paid=0,
                energy_remaining=self._agent_energy[agent_id],
                reason=reason
            )

        # Pagar costo
        self._agent_energy[agent_id] -= cost

        # Crear objeto
        obj_id = self._generate_id()
        position = self._compute_position(idea, agent_id)
        obj_type = self._determine_type(idea)

        obj = MaterializedObject(
            id=obj_id,
            creator_id=agent_id,
            source_idea=idea,
            form=idea.vector.copy(),
            structure=idea.structure.copy(),
            object_type=obj_type,
            position=position,
            creation_cost=cost,
            current_energy=cost,  # Energía inicial = costo
            t_created=self.t,
            t_last_interaction=self.t
        )

        # Registrar
        self._objects[obj_id] = obj
        self._history[agent_id].append(obj_id)

        # Marcar idea como materializada
        idea.materialized = True
        idea.energy = cost

        return MaterializationResult(
            success=True,
            obj=obj,
            cost_paid=cost,
            energy_remaining=self._agent_energy[agent_id],
            reason="OK"
        )

    def get_object(self, obj_id: str) -> Optional[MaterializedObject]:
        """Obtiene un objeto por ID."""
        return self._objects.get(obj_id)

    def get_objects_by_creator(self, agent_id: str) -> List[MaterializedObject]:
        """Obtiene todos los objetos de un creador."""
        return [
            self._objects[oid]
            for oid in self._history.get(agent_id, [])
            if oid in self._objects
        ]

    def get_all_objects(self) -> List[MaterializedObject]:
        """Obtiene todos los objetos existentes."""
        return list(self._objects.values())

    def get_visible_objects(self) -> List[MaterializedObject]:
        """Obtiene todos los objetos visibles."""
        return [o for o in self._objects.values() if o.visible]

    def decay_objects(self, decay_rate: float = 0.01):
        """
        Aplica decaimiento a todos los objetos.

        Los objetos pierden energía con el tiempo.
        Cuando la energía llega a 0, desaparecen.
        """
        to_remove = []

        for obj_id, obj in self._objects.items():
            # Decaimiento proporcional a la edad
            age = self.t - obj.t_created
            decay = decay_rate * (1 + age / 100)

            obj.current_energy -= decay

            if obj.current_energy <= 0:
                obj.visible = False
                obj.stable = False
                to_remove.append(obj_id)

        # No eliminamos los objetos, solo los marcamos
        # Esto permite que queden en la historia

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas del materializador."""
        visible_objects = self.get_visible_objects()

        return {
            't': self.t,
            'total_objects': len(self._objects),
            'visible_objects': len(visible_objects),
            'total_agents': len(self._agent_energy),
            'mean_energy': float(np.mean(list(self._agent_energy.values()))) if self._agent_energy else 0,
            'objects_by_type': {
                t.value: sum(1 for o in visible_objects if o.object_type == t)
                for t in ObjectType
            }
        }
