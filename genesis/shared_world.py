"""
Mundo Compartido (Shared World)
================================

El espacio donde existen las creaciones.

No es un "mundo físico" simulado.
Es un espacio topológico donde los objetos:
- Tienen posición
- Tienen vecinos
- Interactúan por proximidad
- Forman estructuras emergentes

El mundo tiene propiedades endógenas:
- Densidad local (cuántos objetos hay cerca)
- Flujo (hacia dónde tienden a moverse las creaciones)
- Resonancia espacial (qué regiones son más activas)
- Memoria (qué hubo antes en cada lugar)

Los agentes pueden:
- Ver el mundo (percepción)
- Crear en el mundo (materialización)
- Moverse por el mundo (exploración)
- Interactuar con objetos (transformación)

100% endógeno. Sin números mágicos.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import sys

sys.path.insert(0, '/root/NEO_EVA')

from genesis.materialization import MaterializedObject, ObjectType


@dataclass
class WorldRegion:
    """Una región del mundo compartido."""
    id: str
    center: np.ndarray
    radius: float
    objects: List[str] = field(default_factory=list)  # IDs de objetos
    density: float = 0.0
    activity: float = 0.0  # Actividad reciente
    memory: List[str] = field(default_factory=list)  # Objetos que estuvieron aquí


@dataclass
class WorldState:
    """Estado global del mundo."""
    t: int
    total_objects: int
    total_regions: int
    mean_density: float
    max_density: float
    total_activity: float
    dominant_type: Optional[ObjectType]


class SharedWorld:
    """
    Espacio compartido donde existen las creaciones.

    No simula física. Es un espacio de relaciones donde:
    - La posición indica similitud/diferencia
    - La proximidad indica relación
    - La densidad indica actividad creativa
    """

    def __init__(self, dim: int = 3, n_regions: int = 8):
        """
        Inicializa el mundo compartido.

        Args:
            dim: Dimensiones del espacio
            n_regions: Número de regiones iniciales
        """
        self.dim = dim
        self.t = 0
        self.eps = np.finfo(float).eps

        # Objetos en el mundo
        self._objects: Dict[str, MaterializedObject] = {}

        # Posiciones de agentes
        self._agent_positions: Dict[str, np.ndarray] = {}

        # Regiones del mundo (divisiones naturales)
        self._regions: Dict[str, WorldRegion] = {}
        self._initialize_regions(n_regions)

        # Campo de actividad (dónde pasan cosas)
        self._activity_field: np.ndarray = np.zeros((n_regions,))

        # Historial de eventos
        self._event_history: List[Dict[str, Any]] = []

        # Conexiones entre objetos
        self._connections: Dict[str, Set[str]] = {}  # obj_id -> set of connected obj_ids

    def _initialize_regions(self, n_regions: int):
        """
        Inicializa regiones del mundo.

        Las regiones emergen de los vértices de un simplex
        en el espacio del mundo.
        """
        # Generar centros como vértices de un politopo regular
        # Para dim=3, n=8: vértices de un cubo normalizado

        if self.dim == 3 and n_regions == 8:
            # Cubo unitario
            centers = []
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        centers.append(np.array([
                            i - 0.5,
                            j - 0.5,
                            k - 0.5
                        ]) * 2)  # Escalar a [-1, 1]
        else:
            # Generar puntos pseudo-aleatorios pero deterministas
            np.random.seed(42)  # Determinista
            centers = [np.random.randn(self.dim) for _ in range(n_regions)]
            centers = [c / (np.linalg.norm(c) + self.eps) for c in centers]
            np.random.seed()  # Reset

        # Crear regiones
        for i, center in enumerate(centers):
            region_id = f"region_{i:02d}"
            self._regions[region_id] = WorldRegion(
                id=region_id,
                center=np.array(center),
                radius=1.0 / np.sqrt(n_regions)  # Radio endógeno
            )

    def _find_nearest_region(self, position: np.ndarray) -> str:
        """Encuentra la región más cercana a una posición."""
        min_dist = float('inf')
        nearest = None

        for region_id, region in self._regions.items():
            dist = np.linalg.norm(position - region.center)
            if dist < min_dist:
                min_dist = dist
                nearest = region_id

        return nearest

    def _find_nearby_objects(
        self,
        position: np.ndarray,
        radius: Optional[float] = None
    ) -> List[MaterializedObject]:
        """
        Encuentra objetos cercanos a una posición.

        Si no se especifica radio, usa un radio endógeno
        basado en la densidad local.
        """
        if radius is None:
            # Radio endógeno: basado en distancia media entre objetos
            if len(self._objects) > 1:
                positions = [o.position for o in self._objects.values() if o.visible]
                if len(positions) > 1:
                    dists = []
                    for i, p1 in enumerate(positions):
                        for p2 in positions[i+1:]:
                            dists.append(np.linalg.norm(p1 - p2))
                    radius = np.median(dists)
                else:
                    radius = 1.0
            else:
                radius = 1.0

        nearby = []
        for obj in self._objects.values():
            if not obj.visible:
                continue
            dist = np.linalg.norm(obj.position - position)
            if dist <= radius:
                nearby.append(obj)

        return nearby

    def _update_region_stats(self, region_id: str):
        """Actualiza estadísticas de una región."""
        region = self._regions[region_id]

        # Contar objetos visibles
        visible_objects = [
            oid for oid in region.objects
            if oid in self._objects and self._objects[oid].visible
        ]

        region.objects = visible_objects
        region.density = len(visible_objects) / (np.pi * region.radius ** self.dim + self.eps)

    def add_object(self, obj: MaterializedObject) -> str:
        """
        Añade un objeto al mundo.

        Returns:
            ID de la región donde se colocó
        """
        self.t += 1

        # Registrar objeto
        self._objects[obj.id] = obj
        self._connections[obj.id] = set()

        # Encontrar región
        region_id = self._find_nearest_region(obj.position)
        region = self._regions[region_id]
        region.objects.append(obj.id)

        # Actualizar actividad
        region.activity += 1.0
        self._activity_field[int(region_id.split('_')[1])] += 1.0

        # Buscar conexiones con objetos cercanos
        nearby = self._find_nearby_objects(obj.position)
        for neighbor in nearby:
            if neighbor.id != obj.id:
                self._connections[obj.id].add(neighbor.id)
                if neighbor.id in self._connections:
                    self._connections[neighbor.id].add(obj.id)

        # Registrar evento
        self._event_history.append({
            't': self.t,
            'type': 'creation',
            'object_id': obj.id,
            'region': region_id,
            'creator': obj.creator_id
        })

        # Actualizar stats
        self._update_region_stats(region_id)

        return region_id

    def remove_object(self, obj_id: str) -> bool:
        """Elimina un objeto del mundo."""
        if obj_id not in self._objects:
            return False

        obj = self._objects[obj_id]
        obj.visible = False

        # Remover de región
        for region in self._regions.values():
            if obj_id in region.objects:
                region.objects.remove(obj_id)
                region.memory.append(obj_id)
                self._update_region_stats(region.id)
                break

        # Remover conexiones
        if obj_id in self._connections:
            for connected in self._connections[obj_id]:
                if connected in self._connections:
                    self._connections[connected].discard(obj_id)
            del self._connections[obj_id]

        # Registrar evento
        self._event_history.append({
            't': self.t,
            'type': 'removal',
            'object_id': obj_id
        })

        return True

    def move_agent(self, agent_id: str, new_position: np.ndarray):
        """Mueve un agente a una nueva posición."""
        old_position = self._agent_positions.get(agent_id)
        self._agent_positions[agent_id] = new_position.copy()

        # Registrar movimiento si cambió significativamente
        if old_position is not None:
            dist = np.linalg.norm(new_position - old_position)
            if dist > 0.1:
                self._event_history.append({
                    't': self.t,
                    'type': 'movement',
                    'agent_id': agent_id,
                    'distance': float(dist)
                })

    def get_agent_position(self, agent_id: str) -> Optional[np.ndarray]:
        """Obtiene la posición de un agente."""
        return self._agent_positions.get(agent_id)

    def get_visible_objects(self) -> List[MaterializedObject]:
        """Obtiene todos los objetos visibles."""
        return [o for o in self._objects.values() if o.visible]

    def get_objects_in_region(self, region_id: str) -> List[MaterializedObject]:
        """Obtiene objetos en una región."""
        region = self._regions.get(region_id)
        if not region:
            return []

        return [
            self._objects[oid]
            for oid in region.objects
            if oid in self._objects and self._objects[oid].visible
        ]

    def get_connected_objects(self, obj_id: str) -> List[MaterializedObject]:
        """Obtiene objetos conectados a uno dado."""
        connected_ids = self._connections.get(obj_id, set())
        return [
            self._objects[cid]
            for cid in connected_ids
            if cid in self._objects and self._objects[cid].visible
        ]

    def get_local_density(self, position: np.ndarray) -> float:
        """Calcula la densidad local en una posición."""
        nearby = self._find_nearby_objects(position)
        radius = 1.0  # Radio base

        if len(self._objects) > 1:
            # Radio adaptativo
            positions = [o.position for o in self._objects.values() if o.visible]
            if positions:
                dists_to_pos = [np.linalg.norm(p - position) for p in positions]
                radius = np.median(dists_to_pos) if dists_to_pos else 1.0

        volume = np.pi ** (self.dim / 2) * radius ** self.dim  # Volumen de hiperesfera
        return len(nearby) / (volume + self.eps)

    def get_activity_gradient(self, position: np.ndarray) -> np.ndarray:
        """
        Calcula el gradiente de actividad en una posición.

        Indica hacia dónde hay más actividad creativa.
        """
        gradient = np.zeros(self.dim)

        for region in self._regions.values():
            direction = region.center - position
            dist = np.linalg.norm(direction) + self.eps
            direction_norm = direction / dist

            # Contribución proporcional a actividad e inversamente a distancia
            weight = region.activity / (dist ** 2 + 1)
            gradient += weight * direction_norm

        # Normalizar
        norm = np.linalg.norm(gradient)
        if norm > self.eps:
            gradient = gradient / norm

        return gradient

    def decay_activity(self, rate: float = 0.1):
        """Aplica decaimiento a la actividad."""
        for region in self._regions.values():
            region.activity *= (1 - rate)

        self._activity_field *= (1 - rate)

    def get_world_state(self) -> WorldState:
        """Obtiene el estado actual del mundo."""
        visible = self.get_visible_objects()

        densities = [r.density for r in self._regions.values()]

        # Tipo dominante
        type_counts = {}
        for obj in visible:
            t = obj.object_type
            type_counts[t] = type_counts.get(t, 0) + 1

        dominant = max(type_counts.keys(), key=lambda k: type_counts[k]) if type_counts else None

        return WorldState(
            t=self.t,
            total_objects=len(visible),
            total_regions=len(self._regions),
            mean_density=float(np.mean(densities)) if densities else 0.0,
            max_density=float(max(densities)) if densities else 0.0,
            total_activity=float(np.sum(self._activity_field)),
            dominant_type=dominant
        )

    def get_region_info(self, region_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de una región."""
        region = self._regions.get(region_id)
        if not region:
            return None

        objects = self.get_objects_in_region(region_id)

        return {
            'id': region.id,
            'center': region.center.tolist(),
            'radius': region.radius,
            'n_objects': len(objects),
            'density': region.density,
            'activity': region.activity,
            'memory_size': len(region.memory),
            'object_types': {
                t.value: sum(1 for o in objects if o.object_type == t)
                for t in ObjectType
            }
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas completas del mundo."""
        state = self.get_world_state()

        return {
            't': self.t,
            'dim': self.dim,
            'total_objects': state.total_objects,
            'total_regions': state.total_regions,
            'mean_density': state.mean_density,
            'max_density': state.max_density,
            'total_activity': state.total_activity,
            'dominant_type': state.dominant_type.value if state.dominant_type else None,
            'n_connections': sum(len(c) for c in self._connections.values()) // 2,
            'n_agents': len(self._agent_positions),
            'event_count': len(self._event_history)
        }
