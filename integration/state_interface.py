"""
State Interface: Puente entre WORLD-1 y AGI-X
=============================================

Conecta el estado del mundo con los módulos cognitivos:
- Self-Model recibe estado real del agente en el mundo
- ToM recibe observaciones de otros agentes
- Los módulos AGI reciben percepciones estructuradas

Sin números mágicos. Todo endógeno.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import (
    L_t, max_history, adaptive_momentum, to_simplex, normalized_entropy
)


@dataclass
class AgentWorldState:
    """Estado de un agente en WORLD-1."""
    agent_name: str
    position: np.ndarray          # Posición en el mundo
    internal_state: np.ndarray    # Estado interno (drives, phi, etc.)
    resources: float              # Recursos que posee
    activity: float               # Nivel de actividad
    velocity: np.ndarray          # Velocidad/cambio reciente
    t: int                        # Tiempo actual


@dataclass
class WorldPerception:
    """Percepción estructurada del mundo para un agente."""
    own_state: AgentWorldState
    other_agents: Dict[str, AgentWorldState]
    world_fields: np.ndarray      # Campos del mundo (temperatura, presión, etc.)
    world_resources: np.ndarray   # Recursos disponibles en el mundo
    world_mode: np.ndarray        # Régimen actual del mundo
    regime_name: str              # Nombre del régimen dominante
    t: int


class StateInterface:
    """
    Interface entre WORLD-1 y el sistema cognitivo AGI-X.

    Traduce estados del mundo a percepciones cognitivas y
    decisiones cognitivas a acciones en el mundo.
    """

    REGIME_NAMES = ['stable', 'volatile', 'transitional']

    def __init__(self, agent_names: List[str], world_dim: int = 15):
        """
        Inicializa la interface.

        Args:
            agent_names: Nombres de los agentes cognitivos
            world_dim: Dimensión total del mundo
        """
        self.agent_names = agent_names
        self.world_dim = world_dim
        self.n_agents = len(agent_names)

        # Mapeo agente → índice en entities del mundo
        self.agent_to_entity: Dict[str, int] = {
            name: i for i, name in enumerate(agent_names)
        }

        # Historial de percepciones por agente
        self.perception_history: Dict[str, List[WorldPerception]] = {
            name: [] for name in agent_names
        }

        # Historial de acciones por agente
        self.action_history: Dict[str, List[np.ndarray]] = {
            name: [] for name in agent_names
        }

        # Estadísticas para normalización endógena
        self.field_stats: Dict[str, List[float]] = {
            'mean': [], 'std': []
        }

        self.t = 0

    def extract_agent_state(self, world_state: Any, agent_name: str,
                           entity_population: Any) -> AgentWorldState:
        """
        Extrae el estado de un agente desde WORLD-1.

        Args:
            world_state: WorldState de world1_core
            agent_name: Nombre del agente
            entity_population: EntityPopulation de world1_entities

        Returns:
            AgentWorldState del agente
        """
        entity_idx = self.agent_to_entity.get(agent_name, 0)

        if entity_population and entity_idx in entity_population.entities:
            entity = entity_population.entities[entity_idx]
            position = entity.position.copy()
            internal_state = entity.internal_state.copy()
            activity = entity.activity
            velocity = entity.get_velocity()
        else:
            # Fallback: extraer de world_state.entities
            n_per_entity = len(world_state.entities) // self.n_agents
            start_idx = entity_idx * n_per_entity
            end_idx = start_idx + n_per_entity

            entity_slice = world_state.entities[start_idx:end_idx]
            position = entity_slice[:3] if len(entity_slice) >= 3 else entity_slice
            internal_state = entity_slice[3:] if len(entity_slice) > 3 else np.zeros(4)
            activity = float(np.mean(np.abs(internal_state)))
            velocity = np.zeros_like(position)

        # Recursos: proporción de recursos del mundo asignados al agente
        resources = float(world_state.resources.mean())

        return AgentWorldState(
            agent_name=agent_name,
            position=position,
            internal_state=internal_state,
            resources=resources,
            activity=activity,
            velocity=velocity,
            t=self.t
        )

    def perceive(self, world_state: Any, agent_name: str,
                entity_population: Any = None) -> WorldPerception:
        """
        Genera percepción del mundo para un agente.

        Args:
            world_state: WorldState actual
            agent_name: Agente que percibe
            entity_population: Población de entidades (opcional)

        Returns:
            WorldPerception estructurada
        """
        self.t += 1

        # Estado propio
        own_state = self.extract_agent_state(world_state, agent_name, entity_population)

        # Estados de otros agentes
        other_agents = {}
        for other_name in self.agent_names:
            if other_name != agent_name:
                other_agents[other_name] = self.extract_agent_state(
                    world_state, other_name, entity_population
                )

        # Campos del mundo
        world_fields = world_state.fields.copy()

        # Actualizar estadísticas para normalización
        self.field_stats['mean'].append(float(np.mean(world_fields)))
        self.field_stats['std'].append(float(np.std(world_fields)))
        max_hist = max_history(self.t)
        for key in self.field_stats:
            if len(self.field_stats[key]) > max_hist:
                self.field_stats[key] = self.field_stats[key][-max_hist:]

        # Recursos del mundo
        world_resources = world_state.resources.copy()

        # Modo/régimen del mundo
        world_mode = world_state.modes.copy()
        dominant_mode = int(np.argmax(world_mode))
        regime_name = self.REGIME_NAMES[dominant_mode] if dominant_mode < len(self.REGIME_NAMES) else 'unknown'

        perception = WorldPerception(
            own_state=own_state,
            other_agents=other_agents,
            world_fields=world_fields,
            world_resources=world_resources,
            world_mode=world_mode,
            regime_name=regime_name,
            t=self.t
        )

        # Guardar en historial
        self.perception_history[agent_name].append(perception)
        if len(self.perception_history[agent_name]) > max_hist:
            self.perception_history[agent_name] = self.perception_history[agent_name][-max_hist:]

        return perception

    def perception_to_cognitive_state(self, perception: WorldPerception) -> Dict[str, np.ndarray]:
        """
        Convierte percepción del mundo a estados para módulos cognitivos.

        Returns:
            Dict con:
            - 'z': estado estructural para self-model
            - 'phi': estado fenomenológico
            - 'drives': vector de drives
            - 'context': contexto del mundo
        """
        own = perception.own_state

        # z: combina posición + internal_state normalizado
        z_raw = np.concatenate([own.position, own.internal_state[:3]])
        z = to_simplex(np.abs(z_raw) + 0.01)

        # phi: actividad + velocidad + recursos
        phi_components = [
            own.activity,
            float(np.linalg.norm(own.velocity)),
            own.resources,
            normalized_entropy(perception.world_mode),
            float(np.mean(perception.world_fields))
        ]
        phi = np.array(phi_components)

        # drives: derivados de internal_state
        drives_raw = own.internal_state.copy()
        if len(drives_raw) < 6:
            drives_raw = np.concatenate([drives_raw, np.zeros(6 - len(drives_raw))])
        drives = to_simplex(np.abs(drives_raw[:6]) + 0.01)

        # context: información del mundo
        context = np.concatenate([
            perception.world_fields,
            perception.world_resources,
            perception.world_mode
        ])

        return {
            'z': z,
            'phi': phi,
            'drives': drives,
            'context': context,
            'regime': perception.regime_name,
            't': perception.t
        }

    def get_other_agents_for_tom(self, perception: WorldPerception,
                                  observer: str) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extrae estados de otros agentes para Theory of Mind.

        Returns:
            Dict[target_name, {z, phi, drives}]
        """
        others = {}
        for target_name, target_state in perception.other_agents.items():
            # Construir pseudo-percepción del otro
            target_z_raw = np.concatenate([target_state.position, target_state.internal_state[:3]])
            target_z = to_simplex(np.abs(target_z_raw) + 0.01)

            target_phi = np.array([
                target_state.activity,
                float(np.linalg.norm(target_state.velocity)),
                target_state.resources,
                0.5,  # No conocemos su percepción del modo
                0.5
            ])

            target_drives_raw = target_state.internal_state.copy()
            if len(target_drives_raw) < 6:
                target_drives_raw = np.concatenate([target_drives_raw, np.zeros(6 - len(target_drives_raw))])
            target_drives = to_simplex(np.abs(target_drives_raw[:6]) + 0.01)

            others[target_name] = {
                'z': target_z,
                'phi': target_phi,
                'drives': target_drives
            }

        return others

    def cognitive_decision_to_action(self, decision: Dict[str, Any],
                                     agent_name: str,
                                     world_dim: int) -> np.ndarray:
        """
        Convierte decisión cognitiva a perturbación del mundo.

        Args:
            decision: Dict con 'direction', 'magnitude', 'target_resource', etc.
            agent_name: Agente que actúa
            world_dim: Dimensión del vector de perturbación

        Returns:
            Vector de perturbación para WORLD-1
        """
        perturbation = np.zeros(world_dim)

        entity_idx = self.agent_to_entity.get(agent_name, 0)

        # Extraer componentes de la decisión
        direction = decision.get('direction', np.zeros(3))
        magnitude = decision.get('magnitude', 0.1)
        confidence = decision.get('confidence', 0.5)

        # Escalar magnitud por confianza
        effective_magnitude = magnitude * confidence

        # Aplicar dirección a la posición del agente
        # Asumiendo que entities ocupa índices [n_fields : n_fields + n_entities]
        n_fields = 4  # De World1Core
        entity_start = n_fields + entity_idx

        if entity_start < world_dim:
            # Mover en la dirección decidida
            for i, d in enumerate(direction[:min(3, world_dim - entity_start)]):
                perturbation[entity_start + i] = d * effective_magnitude

        # Efecto en recursos si hay target_resource
        if 'target_resource' in decision:
            resource_idx = n_fields + 5  # Después de entities
            if resource_idx < world_dim:
                perturbation[resource_idx] = decision['target_resource'] * effective_magnitude * 0.1

        # Guardar en historial
        self.action_history[agent_name].append(perturbation.copy())
        max_hist = max_history(self.t)
        if len(self.action_history[agent_name]) > max_hist:
            self.action_history[agent_name] = self.action_history[agent_name][-max_hist:]

        return perturbation

    def get_perception_delta(self, agent_name: str) -> Optional[Dict[str, float]]:
        """
        Calcula cambio en percepción reciente (para recompensa/sorpresa).
        """
        history = self.perception_history[agent_name]
        if len(history) < 2:
            return None

        prev = history[-2]
        curr = history[-1]

        # Cambio en posición
        pos_delta = float(np.linalg.norm(
            curr.own_state.position - prev.own_state.position
        ))

        # Cambio en recursos
        resource_delta = curr.own_state.resources - prev.own_state.resources

        # Cambio en campos del mundo
        field_delta = float(np.linalg.norm(curr.world_fields - prev.world_fields))

        # Cambio de régimen
        regime_changed = curr.regime_name != prev.regime_name

        return {
            'position_delta': pos_delta,
            'resource_delta': resource_delta,
            'field_delta': field_delta,
            'regime_changed': float(regime_changed)
        }

    def get_statistics(self) -> Dict:
        """Estadísticas de la interface."""
        return {
            't': self.t,
            'n_agents': self.n_agents,
            'agent_names': self.agent_names,
            'total_perceptions': sum(len(h) for h in self.perception_history.values()),
            'total_actions': sum(len(h) for h in self.action_history.values()),
            'field_mean_avg': np.mean(self.field_stats['mean']) if self.field_stats['mean'] else 0,
            'field_std_avg': np.mean(self.field_stats['std']) if self.field_stats['std'] else 0
        }


def test_state_interface():
    """Test de StateInterface."""
    print("=" * 60)
    print("TEST: STATE INTERFACE")
    print("=" * 60)

    from world1.world1_core import World1Core, WorldState
    from world1.world1_entities import EntityPopulation

    # Crear mundo y entidades
    world = World1Core(n_fields=4, n_entities=5, n_resources=3, n_modes=3)
    entities = EntityPopulation(n_entities=5, position_dim=3, state_dim=4)

    # Crear interface
    agents = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
    interface = StateInterface(agents, world_dim=world.D)

    print(f"\nAgentes: {agents}")
    print(f"Dimensión del mundo: {world.D}")

    # Simular pasos
    for t in range(100):
        # Avanzar mundo
        world.step()
        entities.step(world.get_state().fields)

        # Cada agente percibe
        for agent in agents:
            perception = interface.perceive(world.get_state(), agent, entities)

            # Convertir a estado cognitivo
            cog_state = interface.perception_to_cognitive_state(perception)

            # Obtener otros para ToM
            others = interface.get_other_agents_for_tom(perception, agent)

            # Simular decisión cognitiva simple
            decision = {
                'direction': np.random.randn(3) * 0.1,
                'magnitude': 0.1,
                'confidence': cog_state['phi'][0]  # Usar actividad como confianza
            }

            # Convertir a acción
            action = interface.cognitive_decision_to_action(decision, agent, world.D)

        if (t + 1) % 25 == 0:
            stats = interface.get_statistics()
            print(f"\n  t={t+1}:")
            print(f"    Percepciones totales: {stats['total_perceptions']}")
            print(f"    Acciones totales: {stats['total_actions']}")

            # Mostrar percepción de NEO
            neo_perception = interface.perception_history['NEO'][-1]
            print(f"    NEO régimen: {neo_perception.regime_name}")
            print(f"    NEO actividad: {neo_perception.own_state.activity:.3f}")

    print("\n" + "=" * 60)
    print("STATE INTERFACE TEST COMPLETADO")
    print("=" * 60)

    return interface


if __name__ == "__main__":
    test_state_interface()
