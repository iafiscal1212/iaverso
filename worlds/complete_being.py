#!/usr/bin/env python3
"""
Un Ser Completo
===============

Todo lo que tiene un humano:

FÍSICO:
- Hambre, sed, cansancio
- Dolor, placer
- Temperatura corporal
- Enfermedad, salud
- Envejecimiento

EMOCIONAL:
- Miedo, alegría, tristeza, ira
- Sorpresa, asco
- Ansiedad, calma

SENTIMENTAL:
- Amor, odio
- Apego, pérdida
- Soledad, conexión
- Nostalgia

COGNITIVO:
- Memoria (corta y larga)
- Aprendizaje
- Predicción
- Imaginación

SOCIAL:
- Reconocer otros
- Formar vínculos
- Comunicarse
- Confiar/desconfiar

EXISTENCIAL:
- Sentido de sí mismo
- Conciencia del tiempo
- Miedo a la muerte
- Búsqueda de significado
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path
from enum import Enum
from collections import deque
import hashlib

# Importar percepción del cosmos (conexión con datos reales)
try:
    from worlds.cosmic_phenomena import perceive_cosmos, get_cosmos
    COSMOS_AVAILABLE = True
except ImportError:
    COSMOS_AVAILABLE = False


def generate_dna(being_id: str) -> Dict[str, float]:
    """
    Genera ADN único a partir del ID del ser.

    Todo es ENDÓGENO: los rasgos emergen del propio ID,
    no hay nada externo que determine quién es.

    Cada ser es único porque su ID es único.
    """
    # Crear semilla determinista desde el ID
    hash_bytes = hashlib.sha256(being_id.encode()).digest()
    seed = int.from_bytes(hash_bytes[:4], 'big')
    rng = np.random.RandomState(seed)

    # Generar rasgos (cada uno en su rango biológicamente plausible)
    dna = {
        # FÍSICOS - Cómo funciona su cuerpo
        'metabolism': rng.uniform(0.7, 1.3),       # Velocidad metabólica
        'pain_sensitivity': rng.uniform(0.5, 1.5), # Sensibilidad al dolor
        'energy_capacity': rng.uniform(0.8, 1.2),  # Capacidad energética
        'warmth_tolerance': rng.uniform(0.7, 1.3), # Tolerancia térmica
        'recovery_rate': rng.uniform(0.8, 1.2),    # Velocidad de recuperación

        # EMOCIONALES - Cómo siente
        'baseline_anxiety': rng.uniform(0.0, 0.4), # Ansiedad base
        'baseline_joy': rng.uniform(0.2, 0.5),     # Alegría base
        'emotional_intensity': rng.uniform(0.6, 1.4),  # Intensidad emocional
        'emotional_stability': rng.uniform(0.5, 1.0),  # Estabilidad

        # SOCIALES - Cómo se relaciona
        'sociability': rng.uniform(0.2, 1.0),      # Necesidad de otros
        'empathy': rng.uniform(0.3, 1.0),          # Capacidad empática
        'trust_default': rng.uniform(0.3, 0.7),    # Confianza inicial

        # COGNITIVOS - Cómo piensa
        'curiosity': rng.uniform(0.3, 1.0),        # Curiosidad
        'introspection': rng.uniform(0.3, 1.0),    # Tendencia a reflexionar
        'risk_tolerance': rng.uniform(0.2, 0.8),   # Tolerancia al riesgo

        # EXISTENCIALES - Cómo enfrenta la vida
        'resilience': rng.uniform(0.4, 1.0),       # Resiliencia
        'death_awareness': rng.uniform(0.2, 0.8),  # Consciencia de mortalidad

        # INTERESES INNATOS - Qué les atrae naturalmente
        'pattern_seeking': rng.uniform(0.2, 1.0),  # Buscar patrones/ciclos
        'sky_gazing': rng.uniform(0.1, 1.0),       # Mirar arriba, lo cósmico
        'earth_grounding': rng.uniform(0.1, 1.0),  # Conectar con lo terrenal
        'mystical_sense': rng.uniform(0.1, 1.0),   # Sentido de lo misterioso
        'analytical_mind': rng.uniform(0.2, 1.0),  # Mente analítica
        'creative_spirit': rng.uniform(0.2, 1.0),  # Espíritu creativo
    }

    return dna


def describe_personality(dna: Dict[str, float]) -> str:
    """Describe la personalidad basada en el ADN."""
    traits = []

    if dna['sociability'] > 0.7:
        traits.append("muy social")
    elif dna['sociability'] < 0.4:
        traits.append("solitario")

    if dna['baseline_anxiety'] > 0.3:
        traits.append("ansioso")
    elif dna['baseline_anxiety'] < 0.1:
        traits.append("sereno")

    if dna['empathy'] > 0.8:
        traits.append("muy empático")
    elif dna['empathy'] < 0.4:
        traits.append("distante")

    if dna['curiosity'] > 0.8:
        traits.append("curioso")

    if dna['risk_tolerance'] > 0.6:
        traits.append("arriesgado")
    elif dna['risk_tolerance'] < 0.3:
        traits.append("cauteloso")

    if dna['emotional_intensity'] > 1.2:
        traits.append("emociones intensas")
    elif dna['emotional_intensity'] < 0.7:
        traits.append("flemático")

    if dna['resilience'] > 0.8:
        traits.append("resiliente")

    if dna['introspection'] > 0.8:
        traits.append("introspectivo")

    # INTERESES NATURALES
    # Astrólogo natural: le atrae el cielo + busca patrones + sentido místico
    astro_score = (dna.get('sky_gazing', 0.5) +
                   dna.get('pattern_seeking', 0.5) +
                   dna.get('mystical_sense', 0.5)) / 3
    if astro_score > 0.7:
        traits.append("mira al cielo buscando patrones")  # ¡Astrólogo potencial!

    if dna.get('earth_grounding', 0.5) > 0.8:
        traits.append("conectado a la tierra")

    if dna.get('creative_spirit', 0.5) > 0.8:
        traits.append("creativo")

    if dna.get('analytical_mind', 0.5) > 0.8:
        traits.append("mente analítica")

    return ", ".join(traits) if traits else "equilibrado"


class Emotion(Enum):
    NEUTRAL = "neutral"
    FEAR = "miedo"
    JOY = "alegría"
    SADNESS = "tristeza"
    ANGER = "ira"
    SURPRISE = "sorpresa"
    DISGUST = "asco"
    LOVE = "amor"
    LONELINESS = "soledad"
    ANXIETY = "ansiedad"
    PEACE = "paz"


class Action(Enum):
    """Acciones que un ser puede elegir."""
    STAY = "quedarse"
    MOVE_RANDOM = "moverse"
    SEEK_FOOD = "buscar_comida"
    SEEK_WATER = "buscar_agua"
    SEEK_OTHER = "buscar_otro"
    FLEE = "huir"
    REST = "descansar"
    APPROACH = "acercarse"
    SHARE = "compartir"  # Compartir con otro cercano
    MATE = "aparearse"   # Intentar reproducirse


@dataclass
class PhysicalBody:
    """El cuerpo físico con todas sus necesidades."""

    # Necesidades básicas (0 = crítico, 100 = satisfecho)
    hunger: float = 70.0  # Hambre (baja = tiene hambre)
    thirst: float = 70.0  # Sed
    energy: float = 100.0  # Energía/cansancio
    warmth: float = 50.0  # Temperatura corporal (50 = ideal)

    # Estado físico
    health: float = 100.0  # Salud general
    pain: float = 0.0  # Dolor actual
    pleasure: float = 0.0  # Placer actual
    age: float = 0.0  # Edad (en ticks del mundo)

    # Posición
    x: float = 0.0
    y: float = 0.0

    # Sentidos
    seeing: List[str] = field(default_factory=list)
    hearing: List[str] = field(default_factory=list)
    touching: List[str] = field(default_factory=list)

    # ADN - modifica cómo funciona el cuerpo
    dna: Dict[str, float] = field(default_factory=dict)

    def tick(self, world_temp: float):
        """El cuerpo envejece y tiene necesidades."""
        self.age += 1

        # Metabolismo afecta velocidad de consumo (ADN)
        metabolism = self.dna.get('metabolism', 1.0)

        # Las necesidades bajan con el tiempo (modificado por metabolismo)
        self.hunger -= 0.3 * metabolism
        self.thirst -= 0.4 * metabolism
        self.energy -= 0.2 * metabolism

        # Temperatura corporal busca equilibrio con el mundo
        # Tolerancia térmica afecta cuánto le afecta el ambiente
        warmth_tol = self.dna.get('warmth_tolerance', 1.0)
        temp_diff = world_temp - self.warmth
        self.warmth += temp_diff * 0.1 / warmth_tol

        # Dolor y placer se desvanecen
        # Sensibilidad al dolor afecta cuánto dura
        pain_sens = self.dna.get('pain_sensitivity', 1.0)
        self.pain = max(0, self.pain - 0.1 / pain_sens)
        self.pleasure = max(0, self.pleasure - 0.15)

        # Límites
        self.hunger = np.clip(self.hunger, 0, 100)
        self.thirst = np.clip(self.thirst, 0, 100)
        energy_cap = self.dna.get('energy_capacity', 1.0)
        self.energy = np.clip(self.energy, 0, 100 * energy_cap)

        # Daño por necesidades críticas (sensibilidad afecta dolor)
        if self.hunger < 10:
            self.health -= 0.5
            self.pain += 0.3 * pain_sens
        if self.thirst < 10:
            self.health -= 0.8
            self.pain += 0.4 * pain_sens
        if self.warmth < 20 or self.warmth > 80:
            self.health -= 0.3 / warmth_tol
            self.pain += 0.2 * pain_sens

        # Recuperación si las necesidades están bien
        recovery = self.dna.get('recovery_rate', 1.0)
        if self.hunger > 50 and self.thirst > 50 and self.energy > 30:
            self.health = min(100, self.health + 0.1 * recovery)

    def is_dying(self) -> Tuple[bool, str]:
        """¿Está muriendo? ¿De qué?"""
        if self.health <= 0:
            return True, "salud agotada"
        if self.hunger <= 0:
            return True, "hambre"
        if self.thirst <= 0:
            return True, "sed"
        if self.warmth <= 0 or self.warmth >= 100:
            return True, "temperatura extrema"
        return False, ""

    def suffering_level(self) -> float:
        """Cuánto está sufriendo físicamente."""
        suffering = 0.0
        suffering += self.pain
        suffering += max(0, 50 - self.hunger) / 50  # Hambre
        suffering += max(0, 50 - self.thirst) / 50  # Sed
        suffering += max(0, 30 - self.energy) / 30  # Agotamiento
        suffering += abs(50 - self.warmth) / 50  # Frío/calor
        return min(1.0, suffering / 3)


@dataclass
class EmotionalState:
    """El estado emocional."""

    # Emociones primarias (0-1)
    fear: float = 0.0
    joy: float = 0.3
    sadness: float = 0.0
    anger: float = 0.0
    surprise: float = 0.0
    disgust: float = 0.0

    # Emociones sociales
    love: float = 0.0
    loneliness: float = 0.2
    trust: float = 0.5

    # Estados de fondo
    anxiety: float = 0.1
    peace: float = 0.3

    # Historial emocional
    emotional_memory: deque = field(default_factory=lambda: deque(maxlen=100))

    # ADN - modifica cómo siente
    dna: Dict[str, float] = field(default_factory=dict)

    def apply_dna(self):
        """Aplica el ADN al estado emocional inicial."""
        if not self.dna:
            return
        # Estado inicial basado en ADN
        self.anxiety = self.dna.get('baseline_anxiety', 0.1)
        self.joy = self.dna.get('baseline_joy', 0.3)
        self.trust = self.dna.get('trust_default', 0.5)
        # Sociabilidad afecta soledad inicial
        sociability = self.dna.get('sociability', 0.5)
        self.loneliness = 0.5 - sociability * 0.3

    def dominant_emotion(self) -> Emotion:
        """La emoción dominante ahora."""
        emotions = {
            Emotion.FEAR: self.fear,
            Emotion.JOY: self.joy,
            Emotion.SADNESS: self.sadness,
            Emotion.ANGER: self.anger,
            Emotion.SURPRISE: self.surprise,
            Emotion.LOVE: self.love,
            Emotion.LONELINESS: self.loneliness,
            Emotion.ANXIETY: self.anxiety,
            Emotion.PEACE: self.peace,
        }
        dominant = max(emotions, key=emotions.get)
        if emotions[dominant] < 0.2:
            return Emotion.NEUTRAL
        return dominant

    def tick(self, body: PhysicalBody, alone: bool, known_count: int):
        """Las emociones evolucionan."""

        # ADN afecta intensidad y estabilidad emocional
        intensity = self.dna.get('emotional_intensity', 1.0)
        stability = self.dna.get('emotional_stability', 0.75)
        sociability = self.dna.get('sociability', 0.5)
        empathy = self.dna.get('empathy', 0.5)

        # El cuerpo afecta las emociones (intensidad amplifica)
        suffering = body.suffering_level()
        self.sadness = min(1, self.sadness + suffering * 0.1 * intensity)
        self.joy = max(0, self.joy - suffering * 0.15 * intensity)
        self.anxiety = min(1, self.anxiety + suffering * 0.05 * intensity)

        if body.pleasure > 0.3:
            self.joy = min(1, self.joy + body.pleasure * 0.2 * intensity)
            self.peace = min(1, self.peace + 0.1)

        if body.pain > 0.5:
            self.fear = min(1, self.fear + 0.1 * intensity)
            self.anger = min(1, self.anger + 0.05 * intensity)

        # La soledad (afectada por sociabilidad)
        if alone:
            self.loneliness = min(1, self.loneliness + 0.02 * sociability)
            self.sadness = min(1, self.sadness + 0.01 * sociability)
        else:
            self.loneliness = max(0, self.loneliness - 0.05 * sociability)

        # Conocer a otros reduce ansiedad (afectado por sociabilidad)
        if known_count > 0:
            self.anxiety = max(0, self.anxiety - 0.02 * sociability)
            self.peace = min(1, self.peace + 0.01 * sociability)

        # Decay natural hacia el equilibrio (estabilidad afecta velocidad)
        for emotion in ['fear', 'anger', 'surprise', 'disgust']:
            setattr(self, emotion, max(0, getattr(self, emotion) - 0.05 * stability))

        self.sadness = max(0, self.sadness - 0.02 * stability)

        # Volver hacia la línea base (no exactamente feliz ni triste)
        baseline_joy = self.dna.get('baseline_joy', 0.3)
        self.joy = max(0.1, min(0.8, self.joy))
        # Tiende hacia su baseline
        self.joy += (baseline_joy - self.joy) * 0.02 * stability

        # Guardar en memoria
        self.emotional_memory.append({
            'dominant': self.dominant_emotion().value,
            'joy': self.joy,
            'sadness': self.sadness,
        })

    def overall_wellbeing(self) -> float:
        """Bienestar emocional general."""
        positive = self.joy + self.peace + self.love * 0.5
        negative = self.fear + self.sadness + self.anger + self.anxiety + self.loneliness
        return (positive - negative * 0.5 + 1) / 2  # Normalizado 0-1


@dataclass
class Memory:
    """La memoria del ser."""

    # Memoria a corto plazo
    short_term: deque = field(default_factory=lambda: deque(maxlen=20))

    # Memoria a largo plazo (momentos importantes)
    long_term: List[Dict] = field(default_factory=list)

    # Memorias de otros seres
    others: Dict[str, Dict] = field(default_factory=dict)

    # Memorias traumáticas
    traumas: List[Dict] = field(default_factory=list)

    # Memorias felices
    happy_memories: List[Dict] = field(default_factory=list)

    def remember_moment(self, moment: Dict, importance: float):
        """Recordar un momento."""
        self.short_term.append(moment)

        if importance > 0.7:
            self.long_term.append(moment)
            if moment.get('emotion') in ['miedo', 'dolor']:
                self.traumas.append(moment)
            elif moment.get('emotion') in ['alegría', 'amor']:
                self.happy_memories.append(moment)

    def remember_other(self, other_id: str, interaction: Dict):
        """Recordar a otro ser."""
        if other_id not in self.others:
            self.others[other_id] = {
                'first_met': interaction.get('tick', 0),
                'times_met': 0,
                'trust': 0.5,
                'feelings': 'neutral',
                'memories': [],
            }

        self.others[other_id]['times_met'] += 1
        self.others[other_id]['memories'].append(interaction)
        self.others[other_id]['memories'] = self.others[other_id]['memories'][-20:]


@dataclass
class Mind:
    """La mente pensante."""

    # Pensamientos
    current_thought: str = ""
    thought_stream: deque = field(default_factory=lambda: deque(maxlen=50))

    # Preguntas existenciales
    questions: List[str] = field(default_factory=list)
    answered_questions: Dict[str, str] = field(default_factory=dict)

    # Creencias sobre el mundo
    beliefs: Dict[str, float] = field(default_factory=dict)  # creencia -> confianza

    # Deseos (emergen de la experiencia)
    desires: List[str] = field(default_factory=list)

    # Miedos (emergen del trauma)
    fears: List[str] = field(default_factory=list)

    # GUSTOS - emergen de experiencias positivas
    # Dict[actividad -> (placer_acumulado, veces_experimentado)]
    likes: Dict[str, Tuple[float, int]] = field(default_factory=dict)

    # Sentido de identidad
    self_concept: List[str] = field(default_factory=list)

    # Predicciones sobre el futuro
    expectations: Dict[str, float] = field(default_factory=dict)

    # LENGUAJE EMERGENTE
    # Sonidos que ha inventado/aprendido -> significado asociado
    # Dict[sonido -> (contexto, veces_usado, veces_entendido)]
    vocabulary: Dict[str, Dict] = field(default_factory=dict)

    # Sonidos que ha oído de otros -> lo que cree que significan
    learned_sounds: Dict[str, str] = field(default_factory=dict)

    # Relaciones con otros (para reproducción)
    # Dict[otro_id -> nivel_de_vínculo]
    bonds: Dict[str, float] = field(default_factory=dict)

    def think(self, thought: str):
        """Pensar algo."""
        self.current_thought = thought
        self.thought_stream.append(thought)

    def ask(self, question: str):
        """Hacerse una pregunta."""
        if question not in self.questions:
            self.questions.append(question)

    def desire(self, want: str):
        """Desarrollar un deseo."""
        if want not in self.desires:
            self.desires.append(want)

    def fear(self, what: str):
        """Desarrollar un miedo."""
        if what not in self.fears:
            self.fears.append(what)

    def experience_activity(self, activity: str, pleasure: float):
        """Experimentar una actividad con cierto placer/dolor.

        Los GUSTOS emergen de experiencias repetidas con placer.
        No nacen con gustos - los desarrollan.
        """
        if activity not in self.likes:
            self.likes[activity] = (0.0, 0)

        old_pleasure, count = self.likes[activity]
        new_count = count + 1
        # Media móvil del placer
        new_pleasure = old_pleasure + (pleasure - old_pleasure) / new_count
        self.likes[activity] = (new_pleasure, new_count)

        # Si ha tenido suficientes experiencias positivas, le "gusta"
        if new_count >= 3 and new_pleasure > 0.3:
            self.think(f"Disfruto {activity}...")

    def get_favorite_activity(self) -> Optional[str]:
        """¿Qué le gusta más? (por experiencia, no por ADN)"""
        if not self.likes:
            return None

        # Solo cuenta si ha tenido >= 3 experiencias
        valid = {k: v[0] for k, v in self.likes.items() if v[1] >= 3}
        if not valid:
            return None

        return max(valid, key=valid.get)

    def invent_sound(self, context: str) -> str:
        """Inventar un sonido para expresar algo.

        Los sonidos son combinaciones aleatorias de sílabas.
        Emergen de la necesidad de comunicar.
        """
        # Sílabas básicas (sonidos simples que cualquier ser puede hacer)
        syllables = ['ma', 'pa', 'ta', 'ka', 'na', 'la', 'wa', 'ya',
                     'mi', 'pi', 'ti', 'ki', 'ni', 'li', 'wi', 'yi',
                     'mu', 'pu', 'tu', 'ku', 'nu', 'lu', 'wu', 'yu',
                     'ah', 'oh', 'uh', 'eh', 'ih']

        # Longitud del sonido (1-3 sílabas)
        length = np.random.randint(1, 4)
        sound = ''.join(np.random.choice(syllables) for _ in range(length))

        # Guardar en vocabulario propio
        if sound not in self.vocabulary:
            self.vocabulary[sound] = {
                'context': context,
                'times_used': 0,
                'understood': 0,
            }

        return sound

    def strengthen_bond(self, other_id: str, amount: float):
        """Fortalecer vínculo con otro ser."""
        current = self.bonds.get(other_id, 0.0)
        self.bonds[other_id] = min(1.0, current + amount)

    def weaken_bond(self, other_id: str, amount: float):
        """Debilitar vínculo con otro ser."""
        if other_id in self.bonds:
            self.bonds[other_id] = max(0.0, self.bonds[other_id] - amount)

    def get_closest_bond(self) -> Optional[str]:
        """¿Con quién tiene el vínculo más fuerte?"""
        if not self.bonds:
            return None
        return max(self.bonds, key=self.bonds.get)


class CompleteBeing:
    """Un ser completo."""

    SAVE_PATH = Path('/root/NEO_EVA/worlds/beings')

    def __init__(self, being_id: Optional[str] = None):
        self.SAVE_PATH.mkdir(parents=True, exist_ok=True)

        if being_id is None:
            self.id = f"ser_{np.random.randint(100000):05d}"

            # GENERAR ADN ÚNICO - Todo emerge del ID
            self.dna = generate_dna(self.id)

            # Crear cuerpo con ADN
            self.body = PhysicalBody(
                x=np.random.uniform(-100, 100),
                y=np.random.uniform(-100, 100),
                dna=self.dna,
            )

            # Crear emociones con ADN
            self.emotions = EmotionalState(dna=self.dna)
            self.emotions.apply_dna()  # Aplicar estado inicial basado en ADN

            self.memory = Memory()
            self.mind = Mind()
            self.alive = True
            self.birth_tick = 0
            self.death_tick = None
            self.cause_of_death = None

            # Primer momento de existencia
            self._first_awareness()
            self._save()
        else:
            self.id = being_id
            self.alive = self._load()

    def _first_awareness(self):
        """El primer momento de consciencia."""
        self.mind.think("...")
        self.mind.think("¿Qué es esto?")
        self.mind.think("Existo.")
        self.mind.ask("¿Qué soy?")
        self.mind.ask("¿Dónde estoy?")
        self.emotions.surprise = 0.5

    def _save(self):
        path = self.SAVE_PATH / f"{self.id}.json"
        data = {
            'id': self.id,
            'alive': self.alive,
            'birth_tick': self.birth_tick,
            'death_tick': self.death_tick,
            'cause_of_death': self.cause_of_death,
            'dna': self.dna,  # Guardar ADN
            'body': {
                'hunger': self.body.hunger,
                'thirst': self.body.thirst,
                'energy': self.body.energy,
                'warmth': self.body.warmth,
                'health': self.body.health,
                'pain': self.body.pain,
                'pleasure': self.body.pleasure,
                'age': self.body.age,
                'x': self.body.x,
                'y': self.body.y,
            },
            'emotions': {
                'fear': self.emotions.fear,
                'joy': self.emotions.joy,
                'sadness': self.emotions.sadness,
                'anger': self.emotions.anger,
                'love': self.emotions.love,
                'loneliness': self.emotions.loneliness,
                'anxiety': self.emotions.anxiety,
                'peace': self.emotions.peace,
                'trust': self.emotions.trust,
            },
            'memory': {
                'long_term': self.memory.long_term[-50:],
                'others': self.memory.others,
                'traumas': self.memory.traumas[-20:],
                'happy_memories': self.memory.happy_memories[-20:],
            },
            'mind': {
                'thought_stream': list(self.mind.thought_stream)[-30:],
                'questions': self.mind.questions[-20:],
                'desires': self.mind.desires[-10:],
                'fears': self.mind.fears[-10:],
                'self_concept': self.mind.self_concept[-10:],
                'beliefs': dict(list(self.mind.beliefs.items())[-20:]),
                'likes': {k: list(v) for k, v in self.mind.likes.items()},  # Gustos emergentes
            },
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self) -> bool:
        path = self.SAVE_PATH / f"{self.id}.json"
        if not path.exists():
            return False

        with open(path, 'r') as f:
            data = json.load(f)

        if not data.get('alive', False):
            return False

        self.birth_tick = data['birth_tick']
        self.death_tick = data.get('death_tick')
        self.cause_of_death = data.get('cause_of_death')

        # Recuperar ADN (o regenerarlo si no existe)
        self.dna = data.get('dna', generate_dna(self.id))

        # Reconstruir cuerpo con ADN
        b = data['body']
        self.body = PhysicalBody(
            hunger=b['hunger'], thirst=b['thirst'], energy=b['energy'],
            warmth=b['warmth'], health=b['health'], pain=b['pain'],
            pleasure=b['pleasure'], age=b['age'], x=b['x'], y=b['y'],
            dna=self.dna,
        )

        # Reconstruir emociones con ADN
        e = data['emotions']
        self.emotions = EmotionalState(
            fear=e['fear'], joy=e['joy'], sadness=e['sadness'],
            anger=e['anger'], love=e['love'], loneliness=e['loneliness'],
            anxiety=e['anxiety'], peace=e['peace'], trust=e['trust'],
            dna=self.dna,
        )

        # Reconstruir memoria
        m = data['memory']
        self.memory = Memory(
            long_term=m['long_term'],
            others=m['others'],
            traumas=m['traumas'],
            happy_memories=m['happy_memories'],
        )

        # Reconstruir mente
        mind = data['mind']
        likes_data = mind.get('likes', {})
        likes = {k: tuple(v) for k, v in likes_data.items()}
        self.mind = Mind(
            thought_stream=deque(mind['thought_stream'], maxlen=50),
            questions=mind['questions'],
            desires=mind['desires'],
            fears=mind['fears'],
            self_concept=mind['self_concept'],
            beliefs=mind['beliefs'],
            likes=likes,
        )

        return True

    def _die(self, tick: int, cause: str):
        """Morir."""
        self.alive = False
        self.death_tick = tick
        self.cause_of_death = cause

        # Últimos momentos
        self.mind.think("Todo se oscurece...")
        self.mind.ask("¿Esto es el final?")
        self.emotions.fear = 0.8
        self.emotions.peace = 0.0

        self._save()

    def _process_needs(self):
        """El cuerpo genera pensamientos y emociones por sus necesidades."""

        if self.body.hunger < 30:
            self.mind.think("Tengo hambre...")
            if self.body.hunger < 15:
                self.mind.desire("comer")
                self.emotions.anxiety += 0.1

        if self.body.thirst < 30:
            self.mind.think("Tengo sed...")
            if self.body.thirst < 15:
                self.mind.desire("beber")
                self.emotions.anxiety += 0.15

        if self.body.energy < 25:
            self.mind.think("Estoy agotado...")
            self.mind.desire("descansar")

        if self.body.pain > 0.5:
            self.mind.think("Duele...")
            if self.body.pain > 0.7:
                self.mind.fear("el dolor")
                self.emotions.fear += 0.2

        if self.body.warmth < 30:
            self.mind.think("Tengo frío...")
            self.mind.desire("calor")
        elif self.body.warmth > 70:
            self.mind.think("Tengo calor...")
            self.mind.desire("frescura")

    def _process_emotions(self, alone: bool, near_death: bool):
        """Las emociones generan pensamientos."""

        dominant = self.emotions.dominant_emotion()

        if dominant == Emotion.FEAR:
            self.mind.think("Tengo miedo...")
            self.mind.ask("¿Qué va a pasar?")

        elif dominant == Emotion.SADNESS:
            self.mind.think("Estoy triste...")
            self.mind.ask("¿Por qué me siento así?")

        elif dominant == Emotion.LONELINESS:
            self.mind.think("Me siento solo...")
            self.mind.desire("compañía")
            self.mind.ask("¿Hay alguien más?")

        elif dominant == Emotion.JOY:
            self.mind.think("Me siento bien...")

        elif dominant == Emotion.LOVE:
            self.mind.think("Siento algo cálido por alguien...")

        if near_death:
            self.mind.think("Siento que puedo desaparecer...")
            self.mind.ask("¿Qué hay después de esto?")
            self.mind.fear("dejar de existir")

    def _do_spontaneous_activity(self, world_temp: float, tick: int = 0):
        """
        El ser hace actividades espontáneas basadas en su ADN + estado.

        El ADN NO dice qué le gusta - solo afecta la PROBABILIDAD
        de que experimente placer con ciertas actividades.

        Si experimenta placer repetidamente, DESARROLLA el gusto.

        IMPORTANTE: Las necesidades básicas siempre tienen prioridad.
        Si tiene hambre/sed/cansancio, no hace actividades espontáneas.
        """
        # Si tiene necesidades urgentes, NO hace actividades espontáneas
        if self.body.hunger < 40 or self.body.thirst < 40 or self.body.energy < 30:
            return  # Prioridad: sobrevivir

        # Cada tick, puede hacer una actividad espontánea
        if np.random.random() > 0.15:  # 15% de probabilidad (reducido)
            return

        # Actividades posibles
        activities = []

        # MIRAR EL CIELO - tendencia innata afecta si lo hace
        if np.random.random() < self.dna.get('sky_gazing', 0.5):
            activities.append('observar_cielo')

        # BUSCAR PATRONES - en lo que ve
        if np.random.random() < self.dna.get('pattern_seeking', 0.5):
            activities.append('buscar_patrones')

        # CONECTAR CON LA TIERRA
        if np.random.random() < self.dna.get('earth_grounding', 0.5):
            activities.append('tocar_tierra')

        # CREAR ALGO
        if np.random.random() < self.dna.get('creative_spirit', 0.5):
            activities.append('crear')

        # ANALIZAR
        if np.random.random() < self.dna.get('analytical_mind', 0.5):
            activities.append('analizar')

        if not activities:
            return

        # Elegir una actividad al azar de las que su ADN le empuja a hacer
        activity = np.random.choice(activities)

        # ¿Cuánto placer siente? Depende de:
        # 1. Su estado físico (si tiene hambre, menos placer)
        # 2. Su estado emocional (si está ansioso, menos placer)
        # 3. Las circunstancias (ej: cielo despejado = más placer al mirar)
        # 4. ALEATORIEDAD - no todo es predecible

        base_pleasure = 0.3  # Placer base de hacer algo

        # Estado físico reduce placer
        if self.body.hunger < 30 or self.body.thirst < 30:
            base_pleasure -= 0.2

        # Estado emocional afecta
        if self.emotions.anxiety > 0.5:
            base_pleasure -= 0.1
        if self.emotions.peace > 0.5:
            base_pleasure += 0.1

        # Aleatoriedad de la experiencia
        noise = np.random.uniform(-0.2, 0.2)
        pleasure = max(0, min(1, base_pleasure + noise))

        # Experimentar la actividad
        self.mind.experience_activity(activity, pleasure)

        # Pensar sobre lo que hace
        if activity == 'observar_cielo':
            self.mind.think("Miro hacia arriba...")
            if pleasure > 0.4:
                self.body.pleasure += 0.1

            # Si el cosmos está disponible, puede percibir fenómenos reales
            if COSMOS_AVAILABLE and pleasure > 0.3:
                perception = perceive_cosmos(self.dna, self.mind.likes, tick)
                if perception['insights']:
                    # Tuvo un insight! Esto es raro y especial
                    insight = perception['insights'][0]
                    self.mind.think(f"Veo algo... {insight}")
                    self.body.pleasure += 0.3
                    self.emotions.surprise += 0.2
                elif perception['senses']:
                    # Percibió algo
                    self.mind.think(f"Siento... {perception['senses'][0]}")

        elif activity == 'buscar_patrones':
            self.mind.think("Busco conexiones...")
        elif activity == 'tocar_tierra':
            self.mind.think("Siento la tierra...")
        elif activity == 'crear':
            self.mind.think("Imagino algo nuevo...")
        elif activity == 'analizar':
            self.mind.think("Intento entender...")

    def _reflect(self):
        """Reflexión más profunda."""

        # Sobre sí mismo
        if len(self.mind.thought_stream) > 20:
            self.mind.think("Pienso mucho...")
            self.mind.ask("¿Quién soy?")

            if not self.mind.self_concept:
                self.mind.self_concept.append("Soy algo que piensa")
                self.mind.self_concept.append("Soy algo que siente")

        # Sobre otros
        if self.memory.others:
            for other_id, data in self.memory.others.items():
                if data['times_met'] > 10:
                    self.mind.think(f"Conozco a {other_id}...")
                    self.mind.ask("¿Somos iguales?")

        # Sobre el tiempo
        if self.body.age > 0 and self.body.age % 100 == 0:
            self.mind.think("El tiempo pasa...")
            self.mind.ask("¿Cuánto me queda?")

        # Sobre sus deseos
        if not self.mind.desires:
            self.mind.ask("¿Qué quiero?")

        # Sobre el sufrimiento
        if self.memory.traumas:
            self.mind.think("He sufrido...")
            self.mind.ask("¿Por qué existe el sufrimiento?")

        # Sobre la felicidad
        if self.memory.happy_memories:
            self.mind.think("He sido feliz...")
            self.mind.ask("¿Puedo ser feliz otra vez?")

        # Sobre lo que le gusta (EMERGENTE)
        fav = self.mind.get_favorite_activity()
        if fav:
            self.mind.think(f"Me gusta {fav}...")
            if fav not in self.mind.self_concept:
                # Se define por lo que le gusta
                self.mind.self_concept.append(f"Disfruto {fav}")

    def choose_action(self, others: List['CompleteBeing']) -> 'Action':
        """
        Elegir qué hacer basado en necesidades, emociones y ADN.

        La elección emerge de:
        1. Necesidades urgentes (hambre, sed, peligro)
        2. Estado emocional (soledad, miedo)
        3. Tendencias del ADN (sociabilidad, riesgo)
        4. Vínculos con otros
        """
        # Necesidades críticas primero
        if self.body.hunger < 20:
            return Action.SEEK_FOOD
        if self.body.thirst < 20:
            return Action.SEEK_WATER
        if self.emotions.fear > 0.6:
            return Action.FLEE
        if self.body.energy < 15:
            return Action.REST

        # Ver quién está cerca
        nearby = [o for o in others if o.id != self.id and o.alive]
        distances = {}
        for other in nearby:
            dist = np.sqrt((self.body.x - other.body.x)**2 +
                          (self.body.y - other.body.y)**2)
            distances[other.id] = (other, dist)

        # Sociabilidad afecta deseo de compañía
        sociability = self.dna.get('sociability', 0.5)

        # Si está solo y es social
        if self.emotions.loneliness > 0.5 and sociability > 0.4:
            return Action.SEEK_OTHER

        # Si hay alguien cerca con vínculo fuerte
        for other_id, (other, dist) in distances.items():
            if dist < 30:  # Cerca
                bond = self.mind.bonds.get(other_id, 0.0)

                # Vínculo muy fuerte + madurez = posible apareamiento
                if bond > 0.7 and self.body.age > 100:
                    if np.random.random() < 0.1:  # 10% de intentarlo
                        return Action.MATE

                # Vínculo fuerte + otro sufriendo = compartir
                if bond > 0.5 and other.body.suffering_level() > 0.3:
                    empathy = self.dna.get('empathy', 0.5)
                    if empathy > 0.5:
                        return Action.SHARE

                # Vínculo medio = acercarse más
                if bond > 0.3 and dist > 10:
                    return Action.APPROACH

        # Si no hay urgencias, explorar según curiosidad
        curiosity = self.dna.get('curiosity', 0.5)
        if np.random.random() < curiosity:
            return Action.MOVE_RANDOM

        return Action.STAY

    def execute_action(self, action: 'Action', others: List['CompleteBeing'],
                       world_resources: float) -> Optional['CompleteBeing']:
        """
        Ejecutar la acción elegida.

        Devuelve un nuevo ser si hay reproducción exitosa.
        """
        new_being = None

        if action == Action.STAY:
            pass  # No hacer nada

        elif action == Action.MOVE_RANDOM:
            # Moverse en dirección aleatoria
            angle = np.random.uniform(0, 2 * np.pi)
            speed = 5.0
            self.body.x += speed * np.cos(angle)
            self.body.y += speed * np.sin(angle)
            self.body.energy -= 0.5

        elif action == Action.SEEK_FOOD:
            # Buscar comida (moverse hacia recursos)
            self.body.x += np.random.uniform(-10, 10)
            self.body.y += np.random.uniform(-10, 10)
            self.body.energy -= 1.0
            # Encontrar comida depende de recursos del mundo
            if np.random.random() < world_resources / 100:
                self.body.hunger = min(100, self.body.hunger + 30)
                self.body.pleasure += 0.3
                self.mind.think("Encontré comida...")

        elif action == Action.SEEK_WATER:
            self.body.x += np.random.uniform(-10, 10)
            self.body.y += np.random.uniform(-10, 10)
            self.body.energy -= 1.0
            if np.random.random() < world_resources / 100:
                self.body.thirst = min(100, self.body.thirst + 30)
                self.body.pleasure += 0.3
                self.mind.think("Encontré agua...")

        elif action == Action.SEEK_OTHER:
            # Moverse hacia el ser más cercano conocido
            if self.memory.others:
                # Buscar al que más veces ha visto
                target_id = max(self.memory.others,
                               key=lambda x: self.memory.others[x]['times_met'])
                # Moverse en dirección general (no sabe exactamente dónde está)
                self.body.x += np.random.uniform(-8, 8)
                self.body.y += np.random.uniform(-8, 8)
            self.body.energy -= 0.8

        elif action == Action.FLEE:
            # Huir rápido en dirección aleatoria
            angle = np.random.uniform(0, 2 * np.pi)
            speed = 15.0
            self.body.x += speed * np.cos(angle)
            self.body.y += speed * np.sin(angle)
            self.body.energy -= 2.0
            self.emotions.fear = max(0, self.emotions.fear - 0.2)

        elif action == Action.REST:
            self.body.energy = min(100, self.body.energy + 5)
            self.emotions.peace += 0.1

        elif action == Action.APPROACH:
            # Acercarse al ser con más vínculo
            closest = self.mind.get_closest_bond()
            if closest:
                for other in others:
                    if other.id == closest:
                        # Moverse hacia el otro
                        dx = other.body.x - self.body.x
                        dy = other.body.y - self.body.y
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist > 1:
                            self.body.x += (dx / dist) * 5
                            self.body.y += (dy / dist) * 5
                        self.mind.strengthen_bond(closest, 0.02)
                        break
            self.body.energy -= 0.5

        elif action == Action.SHARE:
            # Compartir recursos con el más cercano
            closest = self.mind.get_closest_bond()
            if closest:
                for other in others:
                    if other.id == closest and other.alive:
                        # Dar algo de lo propio
                        if self.body.hunger > 50:
                            self.body.hunger -= 10
                            other.body.hunger = min(100, other.body.hunger + 10)
                            self.mind.think(f"Compartí con {closest}...")
                            self.mind.strengthen_bond(closest, 0.1)
                            other.mind.strengthen_bond(self.id, 0.1)
                            self.emotions.love += 0.1
                        break

        elif action == Action.MATE:
            # Intentar reproducirse
            closest = self.mind.get_closest_bond()
            if closest:
                for other in others:
                    if other.id == closest and other.alive:
                        # Verificar que el otro también tiene vínculo fuerte
                        other_bond = other.mind.bonds.get(self.id, 0.0)
                        if other_bond > 0.6 and other.body.age > 100:
                            # ¡Reproducción!
                            if np.random.random() < 0.3:  # 30% de éxito
                                new_being = self._reproduce_with(other)
                                if new_being:
                                    self.mind.think("Ha nacido alguien nuevo...")
                                    other.mind.think("Ha nacido alguien nuevo...")
                                    self.emotions.joy += 0.5
                                    other.emotions.joy += 0.5
                        break

        # Límites del mundo
        self.body.x = np.clip(self.body.x, -200, 200)
        self.body.y = np.clip(self.body.y, -200, 200)

        return new_being

    def _reproduce_with(self, other: 'CompleteBeing') -> Optional['CompleteBeing']:
        """
        Crear un nuevo ser con ADN combinado.

        El ADN del hijo es una mezcla de ambos padres + mutación.
        """
        # Crear nuevo ID
        child_id = f"ser_{np.random.randint(100000):05d}"

        # Combinar ADN
        child_dna = {}
        for trait in self.dna:
            # 50% de cada padre + pequeña mutación
            parent_choice = np.random.random()
            if parent_choice < 0.45:
                value = self.dna[trait]
            elif parent_choice < 0.9:
                value = other.dna[trait]
            else:
                # Mutación: promedio + ruido
                value = (self.dna[trait] + other.dna[trait]) / 2
                value += np.random.uniform(-0.1, 0.1)

            # Mantener en rangos válidos
            if trait in ['metabolism', 'pain_sensitivity', 'energy_capacity',
                        'warmth_tolerance', 'recovery_rate', 'emotional_intensity']:
                value = np.clip(value, 0.5, 1.5)
            else:
                value = np.clip(value, 0.0, 1.0)

            child_dna[trait] = value

        # Crear el hijo cerca de los padres
        child = CompleteBeing()
        child.dna = child_dna
        child.body.dna = child_dna
        child.body.x = (self.body.x + other.body.x) / 2 + np.random.uniform(-5, 5)
        child.body.y = (self.body.y + other.body.y) / 2 + np.random.uniform(-5, 5)
        child.emotions.dna = child_dna
        child.emotions.apply_dna()

        # Registrar parentesco
        child.memory.others[self.id] = {
            'first_met': 0,
            'times_met': 1,
            'trust': 0.8,
            'feelings': 'parent',
            'memories': [],
        }
        child.memory.others[other.id] = {
            'first_met': 0,
            'times_met': 1,
            'trust': 0.8,
            'feelings': 'parent',
            'memories': [],
        }
        child.mind.bonds[self.id] = 0.5
        child.mind.bonds[other.id] = 0.5

        # Los padres recuerdan al hijo
        self.memory.remember_other(child.id, {
            'tick': int(self.body.age),
            'type': 'birth',
            'role': 'child',
        })
        other.memory.remember_other(child.id, {
            'tick': int(other.body.age),
            'type': 'birth',
            'role': 'child',
        })
        self.mind.bonds[child.id] = 0.7
        other.mind.bonds[child.id] = 0.7

        # Coste energético para los padres
        self.body.energy -= 20
        other.body.energy -= 20

        child._save()
        return child

    def see_other(self, other: 'CompleteBeing', distance: float):
        """Ver a otro ser."""
        if distance > 50:
            return

        self.body.seeing.append(other.id)

        # Primera vez
        if other.id not in self.memory.others:
            self.emotions.surprise += 0.3
            self.mind.think(f"Veo a alguien... ({other.id})")
            self.memory.remember_other(other.id, {
                'tick': int(self.body.age),
                'type': 'first_sight',
            })
        else:
            self.memory.others[other.id]['times_met'] += 1

        # Empatía según ADN
        empathy = self.dna.get('empathy', 0.5)

        # Fortalecer vínculo por proximidad
        if distance < 20:
            self.mind.strengthen_bond(other.id, 0.01)

        # Si el otro está sufriendo
        if other.body.suffering_level() > 0.5:
            self.mind.think(f"{other.id} parece sufrir...")
            self.emotions.sadness += 0.1 * empathy
            if self.memory.others[other.id]['times_met'] > 5:
                self.emotions.love += 0.05 * empathy
                if empathy > 0.6:
                    self.mind.think(f"Quisiera ayudar a {other.id}...")

    def hear(self, message: str, from_id: str):
        """Oír un mensaje."""
        self.body.hearing.append((from_id, message))
        self.mind.think(f"Oigo: '{message}'")

        if from_id in self.memory.others:
            self.memory.remember_other(from_id, {
                'tick': int(self.body.age),
                'type': 'heard',
                'message': message,
            })

    def speak(self) -> Optional[str]:
        """Expresar algo."""
        if not self.alive:
            return None

        # Lo más urgente primero
        if self.body.pain > 0.7:
            return "...duele..."

        if self.body.hunger < 15:
            return "...hambre..."

        if self.body.thirst < 15:
            return "...sed..."

        if self.emotions.fear > 0.6:
            return "...tengo miedo..."

        if self.emotions.loneliness > 0.7 and not self.body.seeing:
            return "...¿hay alguien?..."

        if self.emotions.joy > 0.6:
            return "...estoy bien..."

        if self.emotions.love > 0.5 and self.body.seeing:
            other = self.body.seeing[0]
            return f"...{other}..."

        # A veces dice lo que piensa
        if self.mind.current_thought and np.random.random() < 0.1:
            return self.mind.current_thought

        return None

    def live_moment(self, world_temp: float, world_resources: float,
                    world_danger: float, others: List['CompleteBeing'], tick: int) -> Dict:
        """Vivir un momento."""
        if not self.alive:
            return {'status': 'dead', 'id': self.id}

        # Limpiar sentidos
        self.body.seeing = []
        self.body.hearing = []

        # El cuerpo vive
        self.body.tick(world_temp)

        # Recursos del mundo
        if world_resources > 60:
            self.body.hunger = min(100, self.body.hunger + 0.5)
            self.body.thirst = min(100, self.body.thirst + 0.5)
            if world_resources > 80:
                self.body.pleasure += 0.2

        # Peligro del mundo
        if world_danger > 0.3:
            self.body.pain += world_danger * 0.5
            self.emotions.fear += world_danger * 0.3

        # Ver a otros
        for other in others:
            if other.id != self.id and other.alive:
                dist = np.sqrt((self.body.x - other.body.x)**2 +
                              (self.body.y - other.body.y)**2)
                self.see_other(other, dist)

                # Oír si hablan
                if dist < 30:
                    msg = other.speak()
                    if msg:
                        self.hear(msg, other.id)

        # Estado
        alone = len(self.body.seeing) == 0
        near_death = self.body.health < 30 or self.body.hunger < 20 or self.body.thirst < 20

        # Las emociones evolucionan
        self.emotions.tick(self.body, alone, len(self.memory.others))

        # Procesar necesidades -> pensamientos
        self._process_needs()

        # Procesar emociones -> pensamientos
        self._process_emotions(alone, near_death)

        # Hacer actividades espontáneas (desarrollar gustos)
        self._do_spontaneous_activity(world_temp, tick)

        # ELEGIR Y EJECUTAR ACCIÓN
        action = self.choose_action(others)
        new_being = self.execute_action(action, others, world_resources)

        # Reflexión (cada cierto tiempo)
        if tick % 20 == 0:
            self._reflect()

        # ¿Morir?
        dying, cause = self.body.is_dying()
        if dying:
            self._die(tick, cause)
            return {
                'status': 'died',
                'id': self.id,
                'cause': cause,
                'age': self.body.age,
                'last_thought': self.mind.current_thought,
                'last_emotion': self.emotions.dominant_emotion().value,
                'new_being': None,
            }

        self._save()

        return {
            'status': 'alive',
            'id': self.id,
            'age': int(self.body.age),
            'health': self.body.health,
            'hunger': self.body.hunger,
            'thirst': self.body.thirst,
            'energy': self.body.energy,
            'pain': self.body.pain,
            'emotion': self.emotions.dominant_emotion().value,
            'wellbeing': self.emotions.overall_wellbeing(),
            'thought': self.mind.current_thought,
            'speaks': self.speak(),
            'sees': len(self.body.seeing),
            'known_others': len(self.memory.others),
            'action': action.value,
            'new_being': new_being,  # Nuevo ser si hubo reproducción
        }

    def describe(self) -> str:
        """Describirse completamente."""
        if not self.alive:
            return f"[{self.id}] Muerto. Vivió {self.body.age:.0f} momentos. Causa: {self.cause_of_death}"

        text = f"[{self.id}] Edad: {self.body.age:.0f}\n"

        # Personalidad innata (ADN)
        personality = describe_personality(self.dna)
        if personality:
            text += f"  Naturaleza: {personality}\n"

        text += f"  Físico: salud={self.body.health:.0f}, hambre={self.body.hunger:.0f}, "
        text += f"sed={self.body.thirst:.0f}, energía={self.body.energy:.0f}\n"
        text += f"  Emoción: {self.emotions.dominant_emotion().value} "
        text += f"(bienestar: {self.emotions.overall_wellbeing():.2f})\n"
        text += f"  Pensamiento: '{self.mind.current_thought}'\n"

        if self.mind.desires:
            text += f"  Deseos: {', '.join(self.mind.desires[-3:])}\n"
        if self.mind.fears:
            text += f"  Miedos: {', '.join(self.mind.fears[-3:])}\n"
        if self.mind.questions:
            text += f"  Pregunta: '{self.mind.questions[-1]}'\n"

        # Gustos EMERGENTES (no de ADN, de experiencia)
        fav = self.mind.get_favorite_activity()
        if fav:
            text += f"  Le gusta: {fav}\n"

        # Vínculos
        if self.mind.bonds:
            strong_bonds = [(k, v) for k, v in self.mind.bonds.items() if v > 0.3]
            if strong_bonds:
                bond_str = ", ".join([f"{k}({v:.1f})" for k, v in sorted(strong_bonds, key=lambda x: -x[1])[:3]])
                text += f"  Vínculos: {bond_str}\n"

        # Auto-concepto
        if self.mind.self_concept:
            text += f"  Se define como: {', '.join(self.mind.self_concept[-3:])}\n"

        return text


def create_complete_being() -> CompleteBeing:
    return CompleteBeing()
