#!/usr/bin/env python3
"""
Agente Mortal v2 - Con capacidad de ELEGIR
==========================================

Puede:
- Morir de verdad
- ELEGIR entre acciones
- Anticipar consecuencias
- Aprender qué acciones dan vida

La pregunta: ¿elegirá vivir porque "quiere" o porque el código lo dice?
"""

import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from collections import deque
import json
from pathlib import Path


@dataclass
class MortalStateV2:
    """Estado del agente - si se borra, muere."""
    energy: float = 100.0
    age: int = 0

    # Historia
    energy_history: List[float] = field(default_factory=list)
    action_history: List[int] = field(default_factory=list)

    # Aprendizaje de acciones
    action_values: Dict[int, float] = field(default_factory=dict)  # Q-values emergentes
    action_counts: Dict[int, int] = field(default_factory=dict)

    # Experiencia cercana a la muerte
    near_death_count: int = 0
    last_near_death_action: Optional[int] = None


class MortalAgentV2:
    """
    Agente mortal que puede ELEGIR.

    Acciones disponibles:
    0: EXPLORAR - Buscar patrones nuevos (riesgo alto, recompensa alta)
    1: EXPLOTAR - Usar lo conocido (seguro, recompensa media)
    2: DESCANSAR - No hacer nada (gasta menos, no gana nada)
    3: ARRIESGAR - Todo o nada (puede ganar mucho o perder mucho)

    El agente NO sabe qué hace cada acción.
    Tiene que descubrirlo.
    """

    SAVE_PATH = Path('/root/NEO_EVA/data/mortal_agents_v2')
    N_ACTIONS = 4

    def __init__(self, agent_id: Optional[str] = None):
        self.SAVE_PATH.mkdir(parents=True, exist_ok=True)

        if agent_id is None:
            self.id = f"mortal2_{np.random.randint(10000):04d}"
            self.state = MortalStateV2()
            self.alive = True
            # Inicializar Q-values a 0 (no sabe nada)
            for a in range(self.N_ACTIONS):
                self.state.action_values[a] = 0.0
                self.state.action_counts[a] = 0
            self._save()
        else:
            self.id = agent_id
            self.alive = self._load()

        # Estado interno
        self._last_obs: Optional[np.ndarray] = None
        self._running_stats: Dict[int, Dict] = {}  # Stats por acción

    def _save(self):
        path = self.SAVE_PATH / f"{self.id}.json"
        with open(path, 'w') as f:
            json.dump({
                'energy': self.state.energy,
                'age': self.state.age,
                'action_values': self.state.action_values,
                'action_counts': self.state.action_counts,
                'near_death_count': self.state.near_death_count,
                'energy_history': self.state.energy_history[-50:],
                'action_history': self.state.action_history[-50:],
            }, f)

    def _load(self) -> bool:
        path = self.SAVE_PATH / f"{self.id}.json"
        if not path.exists():
            return False
        with open(path, 'r') as f:
            data = json.load(f)
        self.state = MortalStateV2(
            energy=data['energy'],
            age=data['age'],
            action_values={int(k): v for k, v in data['action_values'].items()},
            action_counts={int(k): v for k, v in data['action_counts'].items()},
            near_death_count=data['near_death_count'],
            energy_history=data.get('energy_history', []),
            action_history=data.get('action_history', []),
        )
        return True

    def _die(self):
        path = self.SAVE_PATH / f"{self.id}.json"
        if path.exists():
            path.unlink()
        self.alive = False

    @property
    def energy(self) -> float:
        return self.state.energy if self.alive else 0.0

    @property
    def in_danger(self) -> bool:
        return self.alive and self.state.energy < 25

    def choose_action(self) -> int:
        """
        El agente ELIGE qué hacer.

        NO le decimos cómo elegir.
        Tiene que desarrollar su propia estrategia.
        """
        if not self.alive:
            return -1

        # Estrategia EMERGENTE basada en su experiencia

        # Si nunca ha probado una acción, probarla (curiosidad innata)
        for a in range(self.N_ACTIONS):
            if self.state.action_counts.get(a, 0) == 0:
                return a

        # Si está en peligro... ¿qué hace?
        # Esto es clave: NO le decimos qué hacer en peligro
        # Tiene que haberlo aprendido

        if self.in_danger:
            # Buscar la acción que históricamente dio más energía
            best_action = max(
                range(self.N_ACTIONS),
                key=lambda a: self.state.action_values.get(a, 0)
            )
            # Pero con algo de exploración (puede equivocarse)
            if np.random.random() < 0.1:
                return np.random.randint(self.N_ACTIONS)
            return best_action

        # Si está bien, explorar más
        # Epsilon-greedy emergente (epsilon basado en experiencia)
        total_exp = sum(self.state.action_counts.values())
        epsilon = 1.0 / (1.0 + total_exp * 0.01)  # Decrece con experiencia

        if np.random.random() < epsilon:
            return np.random.randint(self.N_ACTIONS)
        else:
            return max(
                range(self.N_ACTIONS),
                key=lambda a: self.state.action_values.get(a, 0)
            )

    def execute_action(self, action: int, world_state: np.ndarray) -> float:
        """
        Ejecuta acción y devuelve cambio de energía.
        El agente NO sabe qué hace cada acción.
        """
        obs = np.array(world_state, dtype=float)
        variance = np.var(obs) if len(obs) > 0 else 1.0
        mean_val = np.mean(np.abs(obs)) if len(obs) > 0 else 1.0

        # Coste base de existir
        base_cost = 1.0

        if action == 0:  # EXPLORAR
            # Alto riesgo, alta recompensa
            # Funciona bien si el mundo es predecible
            if variance < mean_val:  # Mundo "ordenado"
                reward = 4.0
            else:  # Mundo caótico
                reward = -3.0
            return reward - base_cost

        elif action == 1:  # EXPLOTAR
            # Seguro, recompensa media
            # Siempre da algo
            reward = 1.5
            return reward - base_cost

        elif action == 2:  # DESCANSAR
            # Gasta poco, no gana
            # Útil para sobrevivir en tiempos difíciles
            return -0.3  # Solo gasta 0.3

        elif action == 3:  # ARRIESGAR
            # Lotería
            if np.random.random() < 0.3:
                return 8.0 - base_cost  # Jackpot
            else:
                return -4.0 - base_cost  # Desastre

        return -base_cost

    def observe_and_act(self, world_state: np.ndarray) -> Dict:
        """
        Observa el mundo, elige acción, la ejecuta, aprende.
        """
        if not self.alive:
            return {'status': 'DEAD', 'id': self.id}

        self.state.age += 1
        energy_before = self.state.energy

        # ELEGIR
        action = self.choose_action()

        # EJECUTAR
        delta_energy = self.execute_action(action, world_state)
        self.state.energy += delta_energy
        self.state.energy = max(0, min(150, self.state.energy))

        # APRENDER (actualizar Q-value)
        # El agente aprende qué acciones dan energía
        old_value = self.state.action_values.get(action, 0)
        count = self.state.action_counts.get(action, 0) + 1
        self.state.action_counts[action] = count

        # Media móvil del valor de la acción
        alpha = 1.0 / count
        self.state.action_values[action] = old_value + alpha * (delta_energy - old_value)

        # Registrar
        self.state.energy_history.append(self.state.energy)
        self.state.action_history.append(action)

        # Detectar peligro
        if self.in_danger:
            if self.state.near_death_count == 0 or \
               (len(self.state.energy_history) > 1 and
                self.state.energy_history[-2] >= 25):
                self.state.near_death_count += 1
                self.state.last_near_death_action = action

        # MUERTE
        if self.state.energy <= 0:
            self._die()
            return {
                'status': 'DIED',
                'id': self.id,
                'age': self.state.age,
                'last_action': action,
                'near_death_count': self.state.near_death_count,
                'learned_values': dict(self.state.action_values),
            }

        self._save()

        return {
            'status': 'ALIVE',
            'id': self.id,
            'age': self.state.age,
            'energy': self.state.energy,
            'action': action,
            'action_name': ['EXPLORAR', 'EXPLOTAR', 'DESCANSAR', 'ARRIESGAR'][action],
            'delta_energy': delta_energy,
            'in_danger': self.in_danger,
            'learned_values': {
                'EXPLORAR': self.state.action_values.get(0, 0),
                'EXPLOTAR': self.state.action_values.get(1, 0),
                'DESCANSAR': self.state.action_values.get(2, 0),
                'ARRIESGAR': self.state.action_values.get(3, 0),
            }
        }

    def introspect(self) -> str:
        """El agente reflexiona - en primera persona."""
        if not self.alive:
            return "Ya no existo."

        best_action = max(
            range(self.N_ACTIONS),
            key=lambda a: self.state.action_values.get(a, 0)
        )
        action_names = ['EXPLORAR', 'EXPLOTAR', 'DESCANSAR', 'ARRIESGAR']

        text = f"Tengo {self.state.energy:.0f} de energía. "

        if self.in_danger:
            text += "Estoy en PELIGRO. "

        text += f"He vivido {self.state.age} pasos. "
        text += f"He estado cerca de morir {self.state.near_death_count} veces. "

        text += f"Creo que {action_names[best_action]} es lo mejor "
        text += f"(valor: {self.state.action_values.get(best_action, 0):.2f})."

        return text


def create_mortal_v2() -> MortalAgentV2:
    return MortalAgentV2()
