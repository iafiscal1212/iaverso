"""
PhaseSpace-X: Espacio de Fase Estructural
==========================================

PhaseSpace-X es un espacio donde se registran trayectorias (S, dS/dt)
sin saber qué es "física" o "dinámica". Solo observa:
- "Así es mi estado ahora"
- "Así está cambiando mi estado"

Principios:
- NO introduce conocimiento externo (física, teoría de sistemas)
- NO añade objetivos a los agentes
- NO emite instrucciones de comportamiento
- NO crea recompensas ni penalizaciones
- NO usa números mágicos

Todos los umbrales y pesos se derivan de:
- medias, varianzas, covarianzas
- percentiles
- tamaños de dimensión (1/K, 1/√d)
- eps de máquina

Este módulo es NEUTRAL: calcula estructuras y métricas internas, nada más.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class PhasePoint:
    """Punto en el espacio de fase (S, V)."""
    t: int
    agent_id: str
    position: np.ndarray      # S(t) - estado
    velocity: np.ndarray      # V(t) = dS/dt - velocidad de cambio
    speed: float              # |V(t)| - rapidez
    acceleration: float       # |dV/dt| - aceleración


@dataclass
class Trajectory:
    """Trayectoria en espacio de fase."""
    agent_id: str
    points: List[PhasePoint]
    total_length: float       # Longitud total de la trayectoria
    mean_speed: float         # Velocidad media
    curvature: float          # Curvatura media


@dataclass
class Attractor:
    """Atractor emergente en el espacio de fase."""
    center: np.ndarray        # Centro del atractor
    radius: float             # Radio característico
    strength: float           # Fuerza de atracción (densidad de puntos)
    n_points: int             # Número de puntos cercanos


class PhaseSpaceX:
    """
    Espacio de fase estructural.

    Registra pares (S(t), V(t)) donde:
    - S(t) es el estado del agente
    - V(t) = S(t) - S(t-1) es la "velocidad" de cambio

    Métricas emergentes:
    - Trayectorias: cómo se mueve cada agente en el espacio
    - Atractores: regiones donde tienden a converger
    - Divergencia: qué tan dispersas son las trayectorias

    NO dice a los agentes qué hacer.
    Solo mapea estas estructuras.
    """

    def __init__(self):
        """Inicializa PhaseSpace-X."""
        self.t = 0

        # Estados y velocidades por agente
        self._states: Dict[str, List[np.ndarray]] = {}
        self._velocities: Dict[str, List[np.ndarray]] = {}

        # Puntos en espacio de fase
        self._phase_points: Dict[str, List[PhasePoint]] = {}

        # Todos los puntos para análisis global
        self._all_points: List[Tuple[np.ndarray, np.ndarray]] = []

        # Atractores detectados
        self._attractors: List[Attractor] = []

        # Estadísticas para umbrales endógenos
        self._speed_history: List[float] = []
        self._acceleration_history: List[float] = []

        # Dimensión del espacio
        self._dim: Optional[int] = None

    def register_state(
        self,
        agent_id: str,
        S_t: np.ndarray
    ) -> Optional[PhasePoint]:
        """
        Registra un estado y calcula punto en espacio de fase.

        Args:
            agent_id: Identificador del agente
            S_t: Estado actual

        Returns:
            PhasePoint si hay estado previo, None si es el primero
        """
        self.t += 1
        S_t = np.array(S_t, dtype=float)

        # Establecer dimensión
        if self._dim is None:
            self._dim = len(S_t)

        # Inicializar si necesario
        if agent_id not in self._states:
            self._states[agent_id] = []
            self._velocities[agent_id] = []
            self._phase_points[agent_id] = []

        # Guardar estado
        self._states[agent_id].append(S_t.copy())

        # Si es el primer estado, no hay velocidad
        if len(self._states[agent_id]) < 2:
            return None

        # Calcular velocidad V(t) = S(t) - S(t-1)
        S_prev = self._states[agent_id][-2]

        # Alinear dimensiones
        min_dim = min(len(S_t), len(S_prev))
        V_t = S_t[:min_dim] - S_prev[:min_dim]

        self._velocities[agent_id].append(V_t.copy())

        # Calcular rapidez
        speed = float(np.linalg.norm(V_t))
        self._speed_history.append(speed)

        # Calcular aceleración si hay velocidad previa
        acceleration = 0.0
        if len(self._velocities[agent_id]) >= 2:
            V_prev = self._velocities[agent_id][-2]
            min_dim_v = min(len(V_t), len(V_prev))
            dV = V_t[:min_dim_v] - V_prev[:min_dim_v]
            acceleration = float(np.linalg.norm(dV))
            self._acceleration_history.append(acceleration)

        # Crear punto de fase
        phase_point = PhasePoint(
            t=self.t,
            agent_id=agent_id,
            position=S_t[:min_dim].copy(),
            velocity=V_t.copy(),
            speed=speed,
            acceleration=acceleration
        )

        self._phase_points[agent_id].append(phase_point)

        # Guardar para análisis global
        self._all_points.append((S_t[:min_dim].copy(), V_t.copy()))

        # Limitar historial endógenamente
        max_hist = self._get_max_history()
        if len(self._states[agent_id]) > max_hist:
            self._states[agent_id] = self._states[agent_id][-max_hist:]
            self._velocities[agent_id] = self._velocities[agent_id][-(max_hist-1):]
            self._phase_points[agent_id] = self._phase_points[agent_id][-(max_hist-1):]

        return phase_point

    def _get_max_history(self) -> int:
        """Calcula tamaño máximo de historial endógenamente."""
        total_points = len(self._all_points)
        if total_points < 100:
            return 100
        return max(100, int(np.sqrt(total_points) * 10))

    def get_trajectory(self, agent_id: str) -> Optional[Trajectory]:
        """
        Calcula trayectoria de un agente.

        Returns:
            Trajectory con métricas, o None si no hay suficientes puntos
        """
        if agent_id not in self._phase_points:
            return None

        points = self._phase_points[agent_id]
        if len(points) < 2:
            return None

        # Longitud total de la trayectoria
        total_length = sum(p.speed for p in points)

        # Velocidad media
        mean_speed = total_length / len(points)

        # Curvatura media (cambio de dirección)
        curvature = self._compute_curvature(points)

        return Trajectory(
            agent_id=agent_id,
            points=points,
            total_length=total_length,
            mean_speed=mean_speed,
            curvature=curvature
        )

    def _compute_curvature(self, points: List[PhasePoint]) -> float:
        """
        Calcula curvatura media de una trayectoria.

        Curvatura = cambio de dirección / distancia recorrida
        """
        if len(points) < 3:
            return 0.0

        total_angle_change = 0.0
        n_segments = 0

        for i in range(1, len(points) - 1):
            v1 = points[i].velocity
            v2 = points[i + 1].velocity

            # Normalizar
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)

            if n1 > np.finfo(float).eps and n2 > np.finfo(float).eps:
                # Ángulo entre vectores
                cos_angle = np.dot(v1, v2) / (n1 * n2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                total_angle_change += angle
                n_segments += 1

        if n_segments == 0:
            return 0.0

        return float(total_angle_change / n_segments)

    def detect_attractors(self) -> List[Attractor]:
        """
        Detecta atractores en el espacio de fase.

        Usa clustering basado en densidad con parámetros endógenos.

        Returns:
            Lista de Attractor detectados
        """
        if len(self._all_points) < 10:
            self._attractors = []
            return []

        # Extraer posiciones
        positions = np.array([p[0] for p in self._all_points])

        # Parámetros endógenos
        # Radio: desviación estándar / √K donde K = número de clusters estimado
        std_positions = np.std(positions, axis=0)
        mean_std = np.mean(std_positions)

        # Estimar número de clusters como √n
        n = len(positions)
        K_est = max(1, int(np.sqrt(n)))

        radius = mean_std / np.sqrt(K_est) if K_est > 0 else mean_std

        # Encontrar centros de alta densidad
        attractors = []
        used_points = set()

        # Calcular densidad para cada punto
        densities = []
        for i, pos in enumerate(positions):
            distances = np.linalg.norm(positions - pos, axis=1)
            density = np.sum(distances < radius)
            densities.append((i, density, pos))

        # Ordenar por densidad descendente
        densities.sort(key=lambda x: x[1], reverse=True)

        # Umbral de densidad: percentil 75
        density_values = [d[1] for d in densities]
        if len(density_values) > 0:
            density_threshold = np.percentile(density_values, 75)
        else:
            density_threshold = 1

        # Extraer atractores
        for idx, density, center in densities:
            if idx in used_points:
                continue

            if density < density_threshold:
                break

            # Marcar puntos cercanos como usados
            distances = np.linalg.norm(positions - center, axis=1)
            nearby = np.where(distances < radius)[0]

            for j in nearby:
                used_points.add(j)

            # Calcular centro real (media de puntos cercanos)
            actual_center = np.mean(positions[nearby], axis=0)

            # Calcular radio real
            actual_radius = np.mean(distances[nearby]) if len(nearby) > 0 else radius

            # Fuerza = densidad normalizada
            strength = density / n

            attractor = Attractor(
                center=actual_center,
                radius=float(actual_radius),
                strength=float(strength),
                n_points=len(nearby)
            )
            attractors.append(attractor)

        self._attractors = attractors
        return attractors

    def compute_divergence(self, agent_id_1: str, agent_id_2: str) -> Optional[float]:
        """
        Calcula divergencia entre dos trayectorias.

        Divergencia = diferencia media de velocidades para estados similares.

        Returns:
            Float con divergencia, o None si no hay datos
        """
        if agent_id_1 not in self._phase_points or agent_id_2 not in self._phase_points:
            return None

        points_1 = self._phase_points[agent_id_1]
        points_2 = self._phase_points[agent_id_2]

        if not points_1 or not points_2:
            return None

        # Calcular divergencia: diferencia de velocidades en posiciones cercanas
        divergences = []

        for p1 in points_1:
            # Encontrar punto más cercano en trayectoria 2
            min_dist = float('inf')
            closest_p2 = None

            for p2 in points_2:
                # Alinear dimensiones
                min_dim = min(len(p1.position), len(p2.position))
                dist = np.linalg.norm(p1.position[:min_dim] - p2.position[:min_dim])
                if dist < min_dist:
                    min_dist = dist
                    closest_p2 = p2

            if closest_p2 is not None:
                # Diferencia de velocidades
                min_dim_v = min(len(p1.velocity), len(closest_p2.velocity))
                vel_diff = np.linalg.norm(
                    p1.velocity[:min_dim_v] - closest_p2.velocity[:min_dim_v]
                )
                divergences.append(vel_diff)

        if not divergences:
            return None

        return float(np.mean(divergences))

    def get_phase_portrait(self) -> Dict[str, Any]:
        """
        Retorna retrato de fase completo del sistema.

        Incluye estadísticas globales y atractores.
        """
        if not self._all_points:
            return {
                't': self.t,
                'n_agents': 0,
                'n_points': 0,
                'attractors': [],
                'mean_speed': 0.0,
                'mean_acceleration': 0.0
            }

        # Detectar atractores si no lo hemos hecho recientemente
        if not self._attractors or len(self._all_points) % 50 == 0:
            self.detect_attractors()

        return {
            't': self.t,
            'n_agents': len(self._phase_points),
            'n_points': len(self._all_points),
            'dim': self._dim,
            'attractors': [
                {
                    'center': a.center.tolist(),
                    'radius': a.radius,
                    'strength': a.strength,
                    'n_points': a.n_points
                }
                for a in self._attractors
            ],
            'n_attractors': len(self._attractors),
            'mean_speed': float(np.mean(self._speed_history)) if self._speed_history else 0.0,
            'std_speed': float(np.std(self._speed_history)) if len(self._speed_history) > 1 else 0.0,
            'mean_acceleration': float(np.mean(self._acceleration_history)) if self._acceleration_history else 0.0,
            'speed_threshold': self._get_speed_threshold(),
            'acceleration_threshold': self._get_acceleration_threshold()
        }

    def _get_speed_threshold(self) -> float:
        """Calcula umbral de velocidad endógenamente."""
        if len(self._speed_history) < 10:
            return float('inf')  # Sin umbral si no hay historial
        return float(np.percentile(self._speed_history, 90))

    def _get_acceleration_threshold(self) -> float:
        """Calcula umbral de aceleración endógenamente."""
        if len(self._acceleration_history) < 10:
            return float('inf')
        return float(np.percentile(self._acceleration_history, 90))

    def is_near_attractor(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Verifica si un agente está cerca de un atractor.

        Returns:
            Dict con info del atractor más cercano, o None
        """
        if agent_id not in self._phase_points or not self._phase_points[agent_id]:
            return None

        if not self._attractors:
            return None

        # Posición actual
        current_pos = self._phase_points[agent_id][-1].position

        # Encontrar atractor más cercano
        min_dist = float('inf')
        nearest = None

        for i, attractor in enumerate(self._attractors):
            # Alinear dimensiones
            min_dim = min(len(current_pos), len(attractor.center))
            dist = np.linalg.norm(current_pos[:min_dim] - attractor.center[:min_dim])

            if dist < min_dist:
                min_dist = dist
                nearest = (i, attractor, dist)

        if nearest is None:
            return None

        idx, attractor, dist = nearest

        return {
            'attractor_index': idx,
            'distance': float(dist),
            'within_radius': dist <= attractor.radius,
            'attractor_strength': attractor.strength
        }

    def get_agent_dynamics(self, agent_id: str) -> Dict[str, Any]:
        """Retorna dinámica de un agente específico."""
        trajectory = self.get_trajectory(agent_id)
        attractor_info = self.is_near_attractor(agent_id)

        if trajectory is None:
            return {
                'agent_id': agent_id,
                'n_points': 0,
                'trajectory': None,
                'attractor_info': None
            }

        return {
            'agent_id': agent_id,
            'n_points': len(trajectory.points),
            'total_length': trajectory.total_length,
            'mean_speed': trajectory.mean_speed,
            'curvature': trajectory.curvature,
            'attractor_info': attractor_info,
            'current_speed': trajectory.points[-1].speed if trajectory.points else 0.0,
            'current_acceleration': trajectory.points[-1].acceleration if trajectory.points else 0.0
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas completas del sistema."""
        portrait = self.get_phase_portrait()

        # Estadísticas por agente
        agent_stats = {}
        for agent_id in self._phase_points:
            agent_stats[agent_id] = self.get_agent_dynamics(agent_id)

        return {
            **portrait,
            'agent_dynamics': agent_stats
        }
