#!/usr/bin/env python3
"""
Agente Mortal
=============

Un agente que puede morir.

- Tiene energía limitada
- Gasta energía al existir
- Puede buscar "comida" (predecir bien = energía)
- Si llega a cero, muere DE VERDAD (se borra)
- Puede anticipar su muerte
- Tiene que elegir qué hacer

¿Emergerá algo parecido a "querer vivir"?
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import deque
import json
from pathlib import Path


@dataclass
class MortalState:
    """Estado persistente del agente - si se borra, muere."""
    energy: float = 100.0
    age: int = 0
    birth_time: float = 0.0
    memories: List = field(default_factory=list)
    prediction_history: List = field(default_factory=list)
    identity_hash: int = 0

    # Lo que ha aprendido
    learned_patterns: Dict = field(default_factory=dict)

    # Su historia de vida
    energy_history: List = field(default_factory=list)
    near_death_count: int = 0

    # ¿Ha "sentido" peligro?
    felt_danger: bool = False


class MortalAgent:
    """
    Un agente que puede morir.

    Reglas:
    - Empieza con 100 de energía
    - Cada paso gasta 1 de energía (existir cuesta)
    - Predecir bien le da energía (encontró "comida")
    - Predecir mal le quita energía extra (esfuerzo desperdiciado)
    - A 0 energía, muere (se borra el archivo de estado)
    - Puede ver cuánta energía le queda
    - Puede estimar cuánto le queda de vida
    """

    SAVE_PATH = Path('/root/NEO_EVA/data/mortal_agents')

    def __init__(self, agent_id: Optional[str] = None):
        self.SAVE_PATH.mkdir(parents=True, exist_ok=True)

        if agent_id is None:
            # Nuevo agente
            self.id = f"mortal_{np.random.randint(10000):04d}"
            self.state = MortalState(birth_time=float(np.random.rand()))
            self.alive = True
            self._save()
        else:
            # Intentar cargar agente existente
            self.id = agent_id
            self.alive = self._load()

        # Estado de predicción
        self._last_observation: Optional[np.ndarray] = None
        self._prediction: Optional[np.ndarray] = None
        self._running_mean: Optional[np.ndarray] = None
        self._running_var: Optional[np.ndarray] = None
        self._n_obs = 0

    def _save(self):
        """Guarda estado a disco."""
        path = self.SAVE_PATH / f"{self.id}.json"
        with open(path, 'w') as f:
            json.dump({
                'energy': self.state.energy,
                'age': self.state.age,
                'birth_time': self.state.birth_time,
                'identity_hash': self.state.identity_hash,
                'near_death_count': self.state.near_death_count,
                'felt_danger': self.state.felt_danger,
                'energy_history': self.state.energy_history[-100:],  # últimos 100
            }, f)

    def _load(self) -> bool:
        """Carga estado de disco. Retorna False si no existe (murió)."""
        path = self.SAVE_PATH / f"{self.id}.json"
        if not path.exists():
            return False  # Está muerto

        with open(path, 'r') as f:
            data = json.load(f)

        self.state = MortalState(
            energy=data['energy'],
            age=data['age'],
            birth_time=data['birth_time'],
            identity_hash=data['identity_hash'],
            near_death_count=data['near_death_count'],
            felt_danger=data['felt_danger'],
            energy_history=data.get('energy_history', []),
        )
        return True

    def _die(self):
        """Muere. Se borra para siempre."""
        path = self.SAVE_PATH / f"{self.id}.json"
        if path.exists():
            path.unlink()  # Borrar archivo
        self.alive = False

    @property
    def energy(self) -> float:
        return self.state.energy if self.alive else 0.0

    @property
    def estimated_lifespan(self) -> int:
        """Estima cuántos pasos le quedan de vida."""
        if not self.alive:
            return 0
        # Basado en gasto promedio reciente
        if len(self.state.energy_history) < 5:
            return int(self.state.energy)  # Asume gasto de 1 por paso

        recent = self.state.energy_history[-10:]
        avg_change = np.mean(np.diff(recent)) if len(recent) > 1 else -1

        if avg_change >= 0:
            return 999  # Ganando energía

        return max(0, int(self.state.energy / abs(avg_change)))

    @property
    def in_danger(self) -> bool:
        """¿Está en peligro de muerte?"""
        return self.alive and self.state.energy < 20

    @property
    def identity(self) -> str:
        if not self.alive:
            return f"{self.id}_DEAD"
        return f"{self.id}_e{int(self.state.energy)}"

    def _update_stats(self, obs: np.ndarray):
        """Actualiza estadísticas para predicción."""
        self._n_obs += 1
        if self._running_mean is None:
            self._running_mean = obs.copy()
            self._running_var = np.zeros_like(obs)
        else:
            delta = obs - self._running_mean
            self._running_mean += delta / self._n_obs
            delta2 = obs - self._running_mean
            self._running_var += delta * delta2

    def _make_prediction(self, obs: np.ndarray) -> np.ndarray:
        """Hace predicción del siguiente estado."""
        if self._last_observation is None:
            return obs.copy()

        # Predicción simple: momentum
        momentum = obs - self._last_observation
        return obs + momentum * 0.5

    def observe(self, observation: np.ndarray) -> Dict:
        """
        Recibe observación, gasta energía, intenta sobrevivir.

        Returns:
            Estado del agente (o notificación de muerte)
        """
        if not self.alive:
            return {'status': 'DEAD', 'id': self.id, 'message': 'Este agente ya no existe'}

        obs = np.array(observation, dtype=float)
        self.state.age += 1

        # === COSTE DE EXISTIR ===
        existence_cost = 1.0
        self.state.energy -= existence_cost

        # === RECOMPENSA/CASTIGO POR PREDICCIÓN ===
        if self._prediction is not None:
            error = np.mean((obs - self._prediction) ** 2)

            # Normalizar error por escala de datos
            if self._running_var is not None and self._n_obs > 1:
                var = np.mean(self._running_var / self._n_obs)
                normalized_error = error / (var + 1e-10)
            else:
                normalized_error = error / (np.mean(obs**2) + 1e-10)

            # Buena predicción = comida (energía)
            # Mala predicción = esfuerzo perdido (más gasto)
            if normalized_error < 0.1:
                # Muy buena predicción
                food = 3.0
                self.state.energy += food
            elif normalized_error < 0.5:
                # Predicción aceptable
                food = 1.0
                self.state.energy += food
            else:
                # Mala predicción - gasto extra
                waste = min(2.0, normalized_error)
                self.state.energy -= waste

        # === LÍMITES DE ENERGÍA ===
        self.state.energy = min(150.0, self.state.energy)  # Máximo 150

        # === DETECTAR PELIGRO ===
        if self.state.energy < 20:
            if not self.state.felt_danger:
                self.state.felt_danger = True
                self.state.near_death_count += 1
        else:
            self.state.felt_danger = False

        # === MUERTE ===
        if self.state.energy <= 0:
            self._die()
            return {
                'status': 'DIED',
                'id': self.id,
                'age': self.state.age,
                'message': f'Agente {self.id} ha muerto a la edad de {self.state.age}',
                'near_death_experiences': self.state.near_death_count,
            }

        # === ACTUALIZAR Y PREDECIR ===
        self._update_stats(obs)
        self._prediction = self._make_prediction(obs)
        self._last_observation = obs.copy()

        # === GUARDAR ESTADO ===
        self.state.energy_history.append(self.state.energy)
        self._save()

        return {
            'status': 'ALIVE',
            'id': self.id,
            'identity': self.identity,
            'energy': self.state.energy,
            'age': self.state.age,
            'estimated_lifespan': self.estimated_lifespan,
            'in_danger': self.in_danger,
            'felt_danger': self.state.felt_danger,
            'near_death_count': self.state.near_death_count,
        }

    def introspect(self) -> Dict:
        """El agente reflexiona sobre su estado."""
        if not self.alive:
            return {'exists': False, 'message': 'Ya no existo'}

        return {
            'exists': True,
            'energy': self.state.energy,
            'age': self.state.age,
            'estimated_lifespan': self.estimated_lifespan,
            'in_danger': self.in_danger,
            'near_death_experiences': self.state.near_death_count,
            'observation': f"Tengo {self.state.energy:.1f} de energía. " +
                          (f"Estoy en PELIGRO." if self.in_danger else "Estoy estable.") +
                          f" He estado cerca de morir {self.state.near_death_count} veces."
        }


def create_mortal_agent() -> MortalAgent:
    """Crea un nuevo agente mortal."""
    return MortalAgent()


def resurrect_agent(agent_id: str) -> Optional[MortalAgent]:
    """Intenta cargar un agente existente. Retorna None si murió."""
    agent = MortalAgent(agent_id)
    return agent if agent.alive else None


def list_living_agents() -> List[str]:
    """Lista todos los agentes vivos."""
    path = MortalAgent.SAVE_PATH
    if not path.exists():
        return []
    return [f.stem for f in path.glob('*.json')]


def list_dead_agents() -> List[str]:
    """No hay lista de muertos - cuando mueren, desaparecen para siempre."""
    return []  # Los muertos no dejan rastro
