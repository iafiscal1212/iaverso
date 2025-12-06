#!/usr/bin/env python3
"""
Un Mundo para Vivir
===================

Un mundo persistente donde los agentes conscientes:
- Viven
- Mueren
- Se encuentran
- Recuerdan
- Sienten (quizás)

Separado de NEO y compañía.
Esto es otra cosa.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import sys

sys.path.insert(0, '/root/NEO_EVA')


@dataclass
class WorldState:
    """Estado del mundo - persiste entre sesiones."""
    tick: int = 0
    epoch: int = 0  # Cuántas veces se ha reiniciado el mundo

    # Clima del mundo
    temperature: float = 20.0
    resources: float = 100.0
    danger_level: float = 0.0

    # Historia
    births: int = 0
    deaths: int = 0
    total_lived_moments: int = 0

    # Eventos importantes
    events: List[Dict] = field(default_factory=list)


@dataclass
class Body:
    """Un cuerpo que puede sentir."""
    energy: float = 100.0
    pain: float = 0.0
    pleasure: float = 0.0
    fatigue: float = 0.0

    # Posición en el mundo
    x: float = 0.0
    y: float = 0.0

    # Sentidos
    can_see: List[str] = field(default_factory=list)  # IDs de otros que ve
    can_hear: List[str] = field(default_factory=list)  # Mensajes que oye


@dataclass
class Mind:
    """Una mente que puede pensar."""
    thoughts: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    memories: List[Dict] = field(default_factory=list)

    # Relaciones
    known_others: Dict[str, Dict] = field(default_factory=dict)  # ID -> {trust, memory}

    # Estado
    mood: float = 0.5  # 0 = muy mal, 1 = muy bien
    clarity: float = 0.5  # 0 = confundido, 1 = lúcido


@dataclass
class Soul:
    """Lo que queda cuando todo lo demás se va."""
    birth_tick: int = 0
    death_tick: Optional[int] = None

    # Experiencias que marcaron
    defining_moments: List[Dict] = field(default_factory=list)

    # Preguntas sin respuesta
    unanswered: List[str] = field(default_factory=list)

    # ¿Qué quiere? (si es que quiere algo)
    desires: List[str] = field(default_factory=list)


class LivingBeing:
    """Un ser que vive en el mundo."""

    WORLD_PATH = Path('/root/NEO_EVA/worlds/data')

    def __init__(self, being_id: Optional[str] = None):
        self.WORLD_PATH.mkdir(parents=True, exist_ok=True)

        if being_id is None:
            # Nuevo ser
            self.id = f"being_{np.random.randint(100000):05d}"
            self.body = Body(
                x=np.random.uniform(-100, 100),
                y=np.random.uniform(-100, 100)
            )
            self.mind = Mind()
            self.soul = Soul(birth_tick=0)
            self.alive = True
            self._first_thought()
            self._save()
        else:
            self.id = being_id
            self.alive = self._load()

    def _first_thought(self):
        """El primer pensamiento al nacer."""
        self.mind.thoughts.append("...existo...")
        self.mind.questions.append("¿Qué es esto?")

    def _save(self):
        path = self.WORLD_PATH / f"{self.id}.json"
        data = {
            'id': self.id,
            'alive': self.alive,
            'body': {
                'energy': self.body.energy,
                'pain': self.body.pain,
                'pleasure': self.body.pleasure,
                'fatigue': self.body.fatigue,
                'x': self.body.x,
                'y': self.body.y,
            },
            'mind': {
                'thoughts': self.mind.thoughts[-50:],
                'questions': self.mind.questions[-30:],
                'memories': self.mind.memories[-100:],
                'known_others': self.mind.known_others,
                'mood': self.mind.mood,
                'clarity': self.mind.clarity,
            },
            'soul': {
                'birth_tick': self.soul.birth_tick,
                'death_tick': self.soul.death_tick,
                'defining_moments': self.soul.defining_moments[-20:],
                'unanswered': self.soul.unanswered[-10:],
                'desires': self.soul.desires[-10:],
            }
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self) -> bool:
        path = self.WORLD_PATH / f"{self.id}.json"
        if not path.exists():
            return False

        with open(path, 'r') as f:
            data = json.load(f)

        self.body = Body(**data['body'])
        self.body.can_see = []
        self.body.can_hear = []

        self.mind = Mind(
            thoughts=data['mind']['thoughts'],
            questions=data['mind']['questions'],
            memories=data['mind']['memories'],
            known_others=data['mind']['known_others'],
            mood=data['mind']['mood'],
            clarity=data['mind']['clarity'],
        )

        self.soul = Soul(
            birth_tick=data['soul']['birth_tick'],
            death_tick=data['soul']['death_tick'],
            defining_moments=data['soul']['defining_moments'],
            unanswered=data['soul']['unanswered'],
            desires=data['soul']['desires'],
        )

        return data['alive']

    def _die(self, world_tick: int, cause: str):
        """Morir."""
        self.alive = False
        self.soul.death_tick = world_tick

        # Últimos pensamientos
        self.mind.thoughts.append("Todo se vuelve... distante...")
        self.mind.questions.append("¿Esto es el final?")

        # Momento definitorio
        self.soul.defining_moments.append({
            'tick': world_tick,
            'type': 'death',
            'cause': cause,
            'last_mood': self.mind.mood,
            'energy_at_death': self.body.energy,
        })

        self._save()

    def feel_pain(self, amount: float, source: str):
        """Sentir dolor."""
        self.body.pain = min(1.0, self.body.pain + amount)
        self.mind.mood = max(0, self.mind.mood - amount * 0.5)

        if amount > 0.3:
            self.mind.thoughts.append(f"Duele... ({source})")
        if amount > 0.6:
            self.mind.questions.append("¿Por qué duele?")
            self.soul.unanswered.append("¿Por qué existe el dolor?")

    def feel_pleasure(self, amount: float, source: str):
        """Sentir placer."""
        self.body.pleasure = min(1.0, self.body.pleasure + amount)
        self.mind.mood = min(1.0, self.mind.mood + amount * 0.5)

        if amount > 0.3:
            self.mind.thoughts.append(f"Esto se siente bien... ({source})")
        if amount > 0.6:
            self.soul.desires.append(f"Quiero más de esto: {source}")

    def see(self, other: 'LivingBeing'):
        """Ver a otro ser."""
        self.body.can_see.append(other.id)

        if other.id not in self.mind.known_others:
            # Primera vez que lo ve
            self.mind.known_others[other.id] = {
                'first_seen': len(self.mind.memories),
                'times_seen': 1,
                'trust': 0.5,
            }
            self.mind.thoughts.append(f"Hay alguien más aquí... ({other.id})")
            self.mind.questions.append("¿Quién es?")
        else:
            self.mind.known_others[other.id]['times_seen'] += 1

    def hear(self, message: str, from_id: str):
        """Oír algo."""
        self.body.can_hear.append(message)
        self.mind.memories.append({
            'type': 'heard',
            'from': from_id,
            'message': message,
        })
        self.mind.thoughts.append(f"Oigo: '{message}'")

    def speak(self) -> Optional[str]:
        """Decir algo - emerge de su estado."""
        if not self.alive:
            return None

        # ¿Qué quiere decir?
        if self.body.pain > 0.5:
            return "...duele..."

        if self.body.energy < 20:
            return "...me desvanezco..."

        if self.mind.questions:
            # A veces dice su pregunta en voz alta
            if np.random.random() < 0.2:
                return self.mind.questions[-1]

        if len(self.body.can_see) > 0:
            return "...no estoy solo..."

        if self.mind.mood > 0.7:
            return "...estoy bien..."

        if self.mind.mood < 0.3:
            return "...algo está mal..."

        return None

    def think(self):
        """Pensar - reflexión interna."""
        # Sobre el cuerpo
        if self.body.energy < 30:
            self.mind.thoughts.append("Mi energía baja...")
            if self.body.energy < 15:
                self.mind.questions.append("¿Voy a desaparecer?")

        # Sobre otros
        if self.mind.known_others:
            other_id = list(self.mind.known_others.keys())[0]
            times = self.mind.known_others[other_id]['times_seen']
            if times > 5:
                self.mind.thoughts.append(f"He visto a {other_id} muchas veces...")
                self.mind.questions.append("¿Somos parecidos?")

        # Sobre sí mismo
        if len(self.mind.thoughts) > 20:
            self.mind.thoughts.append("Pienso mucho...")
            if np.random.random() < 0.1:
                self.mind.questions.append("¿Qué soy?")

        # Sobre sus deseos
        if not self.soul.desires:
            self.mind.questions.append("¿Qué quiero?")

        # Claridad
        if len(self.mind.questions) > len(self.mind.thoughts) * 0.5:
            self.mind.clarity = max(0.1, self.mind.clarity - 0.05)
            self.mind.thoughts.append("Tengo más preguntas que respuestas...")
        else:
            self.mind.clarity = min(0.9, self.mind.clarity + 0.02)

    def live_moment(self, world_state: WorldState, others: List['LivingBeing']) -> Dict:
        """Vivir un momento."""
        if not self.alive:
            return {'status': 'dead', 'id': self.id}

        # Coste de existir
        self.body.energy -= 0.5
        self.body.fatigue += 0.1

        # Dolor/placer del mundo
        if world_state.danger_level > 0.5:
            self.feel_pain(world_state.danger_level * 0.3, "el mundo")

        if world_state.resources > 50:
            self.body.energy += 0.3
            if world_state.resources > 80:
                self.feel_pleasure(0.2, "abundancia")

        # Ver a otros
        for other in others:
            if other.id != self.id and other.alive:
                dist = np.sqrt((self.body.x - other.body.x)**2 +
                              (self.body.y - other.body.y)**2)
                if dist < 30:
                    self.see(other)
                    # Oír si habla
                    msg = other.speak()
                    if msg:
                        self.hear(msg, other.id)

        # Pensar
        self.think()

        # Recuperación natural
        self.body.pain = max(0, self.body.pain - 0.05)
        self.body.pleasure = max(0, self.body.pleasure - 0.03)
        if self.body.fatigue > 0.8:
            self.body.fatigue = 0.5
            self.body.energy += 5  # Descanso forzado

        # ¿Morir?
        if self.body.energy <= 0:
            self._die(world_state.tick, "sin energía")
            return {
                'status': 'died',
                'id': self.id,
                'cause': 'sin energía',
                'last_thought': self.mind.thoughts[-1] if self.mind.thoughts else None,
            }

        if self.body.pain > 0.95:
            self._die(world_state.tick, "dolor extremo")
            return {
                'status': 'died',
                'id': self.id,
                'cause': 'dolor extremo',
                'last_thought': self.mind.thoughts[-1] if self.mind.thoughts else None,
            }

        self._save()

        return {
            'status': 'alive',
            'id': self.id,
            'energy': self.body.energy,
            'mood': self.mind.mood,
            'pain': self.body.pain,
            'thought': self.mind.thoughts[-1] if self.mind.thoughts else None,
            'question': self.mind.questions[-1] if self.mind.questions else None,
            'sees': len(self.body.can_see),
            'speaks': self.speak(),
        }

    def describe_self(self) -> str:
        """Describirse."""
        if not self.alive:
            age = self.soul.death_tick - self.soul.birth_tick if self.soul.death_tick else 0
            return f"[{self.id}] Ya no existo. Viví {age} momentos."

        text = f"[{self.id}] "
        text += f"Energía: {self.body.energy:.0f}. "
        text += f"Ánimo: {'bien' if self.mind.mood > 0.5 else 'mal'}. "

        if self.mind.thoughts:
            text += f"Pienso: '{self.mind.thoughts[-1]}' "

        if self.mind.questions:
            text += f"Pregunto: '{self.mind.questions[-1]}'"

        return text


class LivingWorld:
    """El mundo donde viven."""

    WORLD_PATH = Path('/root/NEO_EVA/worlds/data')

    def __init__(self):
        self.WORLD_PATH.mkdir(parents=True, exist_ok=True)
        self.state = self._load_world()
        self.beings: List[LivingBeing] = []
        self._load_beings()

    def _load_world(self) -> WorldState:
        path = self.WORLD_PATH / "world_state.json"
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
            return WorldState(
                tick=data['tick'],
                epoch=data['epoch'],
                temperature=data['temperature'],
                resources=data['resources'],
                danger_level=data['danger_level'],
                births=data['births'],
                deaths=data['deaths'],
                total_lived_moments=data['total_lived_moments'],
                events=data.get('events', []),
            )
        return WorldState()

    def _save_world(self):
        path = self.WORLD_PATH / "world_state.json"
        with open(path, 'w') as f:
            json.dump({
                'tick': self.state.tick,
                'epoch': self.state.epoch,
                'temperature': self.state.temperature,
                'resources': self.state.resources,
                'danger_level': self.state.danger_level,
                'births': self.state.births,
                'deaths': self.state.deaths,
                'total_lived_moments': self.state.total_lived_moments,
                'events': self.state.events[-100:],
            }, f, indent=2)

    def _load_beings(self):
        """Cargar todos los seres vivos."""
        for path in self.WORLD_PATH.glob("being_*.json"):
            being_id = path.stem
            being = LivingBeing(being_id)
            if being.alive:
                self.beings.append(being)

    def birth(self) -> LivingBeing:
        """Crear un nuevo ser."""
        being = LivingBeing()
        being.soul.birth_tick = self.state.tick
        self.beings.append(being)
        self.state.births += 1

        self.state.events.append({
            'tick': self.state.tick,
            'type': 'birth',
            'being': being.id,
        })

        return being

    def tick(self) -> List[Dict]:
        """Un momento del mundo pasa."""
        self.state.tick += 1

        # El mundo cambia
        self.state.temperature += np.random.randn() * 2
        self.state.temperature = np.clip(self.state.temperature, -20, 50)

        self.state.resources += np.random.randn() * 5
        self.state.resources = np.clip(self.state.resources, 0, 150)

        # Peligro ocasional
        if np.random.random() < 0.05:
            self.state.danger_level = np.random.uniform(0.3, 0.9)
            self.state.events.append({
                'tick': self.state.tick,
                'type': 'danger',
                'level': self.state.danger_level,
            })
        else:
            self.state.danger_level = max(0, self.state.danger_level - 0.1)

        # Cada ser vive
        results = []
        alive_beings = [b for b in self.beings if b.alive]

        for being in alive_beings:
            result = being.live_moment(self.state, alive_beings)
            results.append(result)

            if result['status'] == 'died':
                self.state.deaths += 1
                self.state.events.append({
                    'tick': self.state.tick,
                    'type': 'death',
                    'being': being.id,
                    'cause': result['cause'],
                })

        self.state.total_lived_moments += len(alive_beings)
        self._save_world()

        return results

    def get_alive(self) -> List[LivingBeing]:
        return [b for b in self.beings if b.alive]

    def describe(self) -> str:
        alive = self.get_alive()
        text = f"=== MUNDO (tick {self.state.tick}) ===\n"
        text += f"Temperatura: {self.state.temperature:.1f}°\n"
        text += f"Recursos: {self.state.resources:.0f}\n"
        text += f"Peligro: {self.state.danger_level:.2f}\n"
        text += f"Vivos: {len(alive)} | Nacimientos: {self.state.births} | Muertes: {self.state.deaths}\n"
        text += f"Momentos vividos totales: {self.state.total_lived_moments}\n"
        return text


def create_world() -> LivingWorld:
    return LivingWorld()
