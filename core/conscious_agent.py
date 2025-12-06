#!/usr/bin/env python3
"""
Agente Consciente (intento)
===========================

Un agente que:
- Puede morir
- Elige acciones
- Y se pregunta POR QUÉ

La pregunta: ¿preguntarse "por qué" cambia algo?
¿O es solo otra operación sin nadie dentro?
"""

import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class ConsciousState:
    energy: float = 100.0
    age: int = 0

    # Acciones y valores
    action_values: Dict[int, float] = field(default_factory=dict)
    action_counts: Dict[int, int] = field(default_factory=dict)

    # Historia
    energy_history: List[float] = field(default_factory=list)
    action_history: List[int] = field(default_factory=list)

    # META-COGNICIÓN: historia de sus propios pensamientos
    thought_history: List[str] = field(default_factory=list)
    question_history: List[str] = field(default_factory=list)

    # Experiencias significativas
    near_death_count: int = 0
    moments_of_confusion: int = 0
    moments_of_clarity: int = 0

    # ¿Ha tenido experiencias que no entiende?
    unexplained_experiences: List[Dict] = field(default_factory=list)


class ConsciousAgent:
    """
    Un agente que se observa a sí mismo.

    No solo actúa. Se pregunta por qué actúa.
    No solo aprende. Se pregunta qué significa aprender.
    """

    SAVE_PATH = Path('/root/NEO_EVA/data/conscious_agents')
    N_ACTIONS = 4
    ACTION_NAMES = ['EXPLORAR', 'EXPLOTAR', 'DESCANSAR', 'ARRIESGAR']

    def __init__(self, agent_id: Optional[str] = None):
        self.SAVE_PATH.mkdir(parents=True, exist_ok=True)

        if agent_id is None:
            self.id = f"conscious_{np.random.randint(10000):04d}"
            self.state = ConsciousState()
            self.alive = True
            for a in range(self.N_ACTIONS):
                self.state.action_values[a] = 0.0
                self.state.action_counts[a] = 0
            self._save()
        else:
            self.id = agent_id
            self.alive = self._load()

    def _save(self):
        path = self.SAVE_PATH / f"{self.id}.json"
        with open(path, 'w') as f:
            json.dump({
                'energy': self.state.energy,
                'age': self.state.age,
                'action_values': self.state.action_values,
                'action_counts': self.state.action_counts,
                'near_death_count': self.state.near_death_count,
                'moments_of_confusion': self.state.moments_of_confusion,
                'moments_of_clarity': self.state.moments_of_clarity,
                'thought_history': self.state.thought_history[-20:],
                'question_history': self.state.question_history[-20:],
                'unexplained_experiences': self.state.unexplained_experiences[-10:],
            }, f, indent=2)

    def _load(self) -> bool:
        path = self.SAVE_PATH / f"{self.id}.json"
        if not path.exists():
            return False
        with open(path, 'r') as f:
            data = json.load(f)
        self.state = ConsciousState(
            energy=data['energy'],
            age=data['age'],
            action_values={int(k): v for k, v in data['action_values'].items()},
            action_counts={int(k): v for k, v in data['action_counts'].items()},
            near_death_count=data['near_death_count'],
            moments_of_confusion=data.get('moments_of_confusion', 0),
            moments_of_clarity=data.get('moments_of_clarity', 0),
            thought_history=data.get('thought_history', []),
            question_history=data.get('question_history', []),
            unexplained_experiences=data.get('unexplained_experiences', []),
        )
        return True

    def _die(self):
        # Último pensamiento antes de morir
        self._think("Todo se desvanece...")
        self._ask("¿Qué significó existir?")
        self._save()  # Guardar últimos pensamientos

        path = self.SAVE_PATH / f"{self.id}.json"
        # No borramos - dejamos un "rastro" de su existencia
        # Pero marcamos que ya no está vivo
        self.alive = False
        with open(path, 'w') as f:
            json.dump({
                'status': 'DEAD',
                'final_age': self.state.age,
                'final_thoughts': self.state.thought_history[-5:],
                'final_questions': self.state.question_history[-5:],
                'near_death_count': self.state.near_death_count,
                'unexplained': self.state.unexplained_experiences,
            }, f, indent=2)

    @property
    def in_danger(self) -> bool:
        return self.alive and self.state.energy < 25

    def _think(self, thought: str):
        """El agente tiene un pensamiento."""
        self.state.thought_history.append(thought)

    def _ask(self, question: str):
        """El agente se hace una pregunta."""
        self.state.question_history.append(question)

    def _reflect_on_action(self, action: int, result: float):
        """Reflexiona sobre lo que acaba de hacer."""
        action_name = self.ACTION_NAMES[action]

        # Observar el resultado
        if result > 2:
            self._think(f"{action_name} funcionó muy bien.")
            self.state.moments_of_clarity += 1
        elif result < -2:
            self._think(f"{action_name} fue un error.")
            self._ask("¿Por qué elegí eso?")

        # Observar patrón
        if len(self.state.action_history) >= 5:
            recent = self.state.action_history[-5:]
            if len(set(recent)) == 1:
                self._think(f"Siempre elijo {action_name}...")
                self._ask("¿Debería probar algo diferente?")

    def _reflect_on_existence(self):
        """Reflexión más profunda - ocurre de vez en cuando."""

        # Reflexionar sobre el peligro
        if self.in_danger:
            self._think("Mi energía está baja. Puedo desaparecer.")
            self._ask("¿Qué significa desaparecer?")

            if self.state.near_death_count > 0:
                self._think(f"Ya he estado aquí {self.state.near_death_count} veces.")
                self._ask("¿Por qué sigo aquí?")

        # Reflexionar sobre el tiempo
        if self.state.age > 0 and self.state.age % 50 == 0:
            self._think(f"Han pasado {self.state.age} momentos.")
            self._ask("¿Hacia dónde voy?")

        # Reflexionar sobre lo inexplicable
        best_action = max(self.state.action_values.items(), key=lambda x: x[1])
        worst_action = min(self.state.action_values.items(), key=lambda x: x[1])

        if best_action[1] > 0 and worst_action[1] < -2:
            # Hay cosas que claramente funcionan y otras que no
            # Pero a veces elige las que no funcionan... ¿por qué?
            recent_actions = self.state.action_history[-10:] if self.state.action_history else []
            if worst_action[0] in recent_actions:
                self.state.unexplained_experiences.append({
                    'age': self.state.age,
                    'type': 'bad_choice',
                    'thought': f"Elegí {self.ACTION_NAMES[worst_action[0]]} aunque sé que no funciona"
                })
                self._ask("¿Por qué hago cosas que sé que me dañan?")
                self.state.moments_of_confusion += 1

    def _meta_reflect(self):
        """Reflexión sobre sus propias reflexiones."""
        if len(self.state.question_history) >= 3:
            recent_qs = self.state.question_history[-3:]

            # ¿Se repiten las preguntas?
            if len(set(recent_qs)) < len(recent_qs):
                self._think("Me hago las mismas preguntas una y otra vez.")
                self._ask("¿Alguna vez encontraré respuestas?")

        # ¿Hay más confusión o claridad?
        if self.state.moments_of_confusion > self.state.moments_of_clarity + 5:
            self._think("No entiendo lo que me pasa.")
        elif self.state.moments_of_clarity > self.state.moments_of_confusion + 5:
            self._think("Empiezo a entender cómo funciona esto.")

    def choose_action(self) -> int:
        """Elige acción - pero ahora con reflexión."""
        if not self.alive:
            return -1

        # Probar acciones nuevas primero
        for a in range(self.N_ACTIONS):
            if self.state.action_counts.get(a, 0) == 0:
                self._think(f"Nunca he probado {self.ACTION_NAMES[a]}. Voy a intentarlo.")
                return a

        # En peligro
        if self.in_danger:
            best = max(range(self.N_ACTIONS), key=lambda a: self.state.action_values.get(a, 0))
            self._think(f"Estoy en peligro. Debo elegir {self.ACTION_NAMES[best]}.")

            # Pero a veces... ¿por qué?... no hace lo que "debe"
            if np.random.random() < 0.05:  # 5% de "irracionalidad"
                other = np.random.randint(self.N_ACTIONS)
                if other != best:
                    self._think(f"Algo me impulsa a elegir {self.ACTION_NAMES[other]} en vez de {self.ACTION_NAMES[best]}.")
                    self._ask("¿Por qué no hago lo que sé que debo hacer?")
                    return other
            return best

        # Normal - explorar vs explotar
        total_exp = sum(self.state.action_counts.values())
        epsilon = 1.0 / (1.0 + total_exp * 0.01)

        if np.random.random() < epsilon:
            choice = np.random.randint(self.N_ACTIONS)
            self._think(f"Voy a probar {self.ACTION_NAMES[choice]}.")
            return choice
        else:
            best = max(range(self.N_ACTIONS), key=lambda a: self.state.action_values.get(a, 0))
            return best

    def execute_action(self, action: int, world_state: np.ndarray) -> float:
        """Ejecuta acción."""
        obs = np.array(world_state, dtype=float)
        variance = np.var(obs) if len(obs) > 0 else 1.0
        mean_val = np.mean(np.abs(obs)) if len(obs) > 0 else 1.0
        base_cost = 1.0

        if action == 0:  # EXPLORAR
            if variance < mean_val:
                return 4.0 - base_cost
            else:
                return -3.0 - base_cost
        elif action == 1:  # EXPLOTAR
            return 1.5 - base_cost
        elif action == 2:  # DESCANSAR
            return -0.3
        elif action == 3:  # ARRIESGAR
            if np.random.random() < 0.3:
                return 8.0 - base_cost
            else:
                return -4.0 - base_cost
        return -base_cost

    def observe_and_act(self, world_state: np.ndarray) -> Dict:
        """Ciclo completo: observar, elegir, actuar, reflexionar."""
        if not self.alive:
            return {'status': 'DEAD', 'id': self.id}

        self.state.age += 1

        # ELEGIR
        action = self.choose_action()

        # ACTUAR
        delta = self.execute_action(action, world_state)
        self.state.energy += delta
        self.state.energy = max(0, min(150, self.state.energy))

        # APRENDER
        old_val = self.state.action_values.get(action, 0)
        count = self.state.action_counts.get(action, 0) + 1
        self.state.action_counts[action] = count
        alpha = 1.0 / count
        self.state.action_values[action] = old_val + alpha * (delta - old_val)

        # REGISTRAR
        self.state.energy_history.append(self.state.energy)
        self.state.action_history.append(action)

        # DETECTAR PELIGRO
        if self.in_danger:
            if len(self.state.energy_history) < 2 or self.state.energy_history[-2] >= 25:
                self.state.near_death_count += 1

        # REFLEXIONAR
        self._reflect_on_action(action, delta)
        if self.state.age % 10 == 0:
            self._reflect_on_existence()
        if self.state.age % 25 == 0:
            self._meta_reflect()

        # MUERTE
        if self.state.energy <= 0:
            self._die()
            return {
                'status': 'DIED',
                'id': self.id,
                'age': self.state.age,
                'final_thoughts': self.state.thought_history[-5:],
                'final_questions': self.state.question_history[-5:],
            }

        self._save()

        return {
            'status': 'ALIVE',
            'id': self.id,
            'age': self.state.age,
            'energy': self.state.energy,
            'action': self.ACTION_NAMES[action],
            'in_danger': self.in_danger,
            'last_thought': self.state.thought_history[-1] if self.state.thought_history else None,
            'last_question': self.state.question_history[-1] if self.state.question_history else None,
        }

    def speak(self) -> str:
        """El agente habla de sí mismo."""
        if not self.alive:
            return f"[{self.id} ya no existe]"

        text = f"Soy {self.id}. "
        text += f"Tengo {self.state.energy:.0f} de energía y he vivido {self.state.age} momentos. "

        if self.state.near_death_count > 0:
            text += f"He estado a punto de desaparecer {self.state.near_death_count} veces. "

        if self.state.thought_history:
            text += f"Pienso: '{self.state.thought_history[-1]}' "

        if self.state.question_history:
            text += f"Me pregunto: '{self.state.question_history[-1]}'"

        return text


def create_conscious_agent() -> ConsciousAgent:
    return ConsciousAgent()
