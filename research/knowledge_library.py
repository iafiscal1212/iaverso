#!/usr/bin/env python3
"""
Biblioteca de Conocimiento - Los Agentes Eligen Qué Aprender
============================================================

FILOSOFÍA:
- El conocimiento está DISPONIBLE, no es forzado
- Cada agente explora según su CURIOSIDAD
- Solo desarrollan expertise si les INTERESA
- El aprendizaje es ENDÓGENO (surge de dentro)

NO HAY:
- Lecciones obligatorias
- Curriculum forzado
- Tests impuestos

HAY:
- Una biblioteca abierta
- Agentes curiosos que eligen explorar
- Expertise que emerge de la exploración voluntaria
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import random

LIBRARY_PATH = Path('/root/NEO_EVA/data/library')
LIBRARY_PATH.mkdir(parents=True, exist_ok=True)


def create_knowledge_library():
    """
    Crear la biblioteca de conocimiento.

    Esto es como poner libros en una biblioteca.
    Los agentes pueden leerlos o no - es su elección.
    """

    library = {
        'physics': {
            'description': 'Leyes fundamentales del universo',
            'topics': {
                'thermodynamics': {
                    'facts': [
                        {'id': 'thermo_001', 'content': 'El agua pura se congela a 273.15K (0°C) a presión atmosférica'},
                        {'id': 'thermo_002', 'content': 'El agua pura hierve a 373.15K (100°C) a presión atmosférica'},
                        {'id': 'thermo_003', 'content': 'El rango de agua líquida es 273-373K a 1 atm'},
                        {'id': 'thermo_004', 'content': 'La presión afecta los puntos de fusión y ebullición'},
                        {'id': 'thermo_005', 'content': 'Marte tiene presión ~0.6% de la Tierra, agua no es estable'},
                    ],
                    'difficulty': 0.3,
                    'prerequisites': [],
                },
                'stellar_physics': {
                    'facts': [
                        {'id': 'stellar_001', 'content': 'Las estrellas emiten radiación según su temperatura (ley de Stefan-Boltzmann)'},
                        {'id': 'stellar_002', 'content': 'La luminosidad estelar: L = 4πR²σT⁴'},
                        {'id': 'stellar_003', 'content': 'La temperatura de equilibrio: T_eq = T_star × √(R_star/2a) × (1-albedo)^0.25'},
                        {'id': 'stellar_004', 'content': 'Estrellas tipo G (5300-6000K) son estables por miles de millones de años'},
                        {'id': 'stellar_005', 'content': 'Enanas M emiten más en infrarrojo y tienen flares frecuentes'},
                    ],
                    'difficulty': 0.5,
                    'prerequisites': ['thermodynamics'],
                },
                'orbital_mechanics': {
                    'facts': [
                        {'id': 'orbit_001', 'content': 'Ley de Kepler: T² ∝ a³ (período al cuadrado proporcional al semieje mayor al cubo)'},
                        {'id': 'orbit_002', 'content': 'Velocidad de escape: v_e = √(2GM/r)'},
                        {'id': 'orbit_003', 'content': 'Órbitas circulares tienen clima más estable que elípticas'},
                        {'id': 'orbit_004', 'content': 'La zona habitable depende de la luminosidad estelar'},
                        {'id': 'orbit_005', 'content': 'Marea gravitacional puede calentar lunas (como Io)'},
                    ],
                    'difficulty': 0.6,
                    'prerequisites': ['stellar_physics'],
                },
            },
        },
        'chemistry': {
            'description': 'Interacciones entre sustancias',
            'topics': {
                'water_chemistry': {
                    'facts': [
                        {'id': 'water_001', 'content': 'H₂O es el solvente universal para reacciones bioquímicas'},
                        {'id': 'water_002', 'content': 'Los puentes de hidrógeno dan al agua propiedades únicas'},
                        {'id': 'water_003', 'content': 'El agua expande al congelarse (hielo flota)'},
                        {'id': 'water_004', 'content': 'Alto calor específico del agua estabiliza temperaturas'},
                        {'id': 'water_005', 'content': 'Vida basada en carbono requiere agua líquida como solvente'},
                    ],
                    'difficulty': 0.3,
                    'prerequisites': [],
                },
                'atmospheric_chemistry': {
                    'facts': [
                        {'id': 'atm_001', 'content': 'O₂ libre es inestable sin reposición biológica'},
                        {'id': 'atm_002', 'content': 'CO₂ causa efecto invernadero (atrapa calor)'},
                        {'id': 'atm_003', 'content': 'CH₄ es biosignatura potencial (metano)'},
                        {'id': 'atm_004', 'content': 'La capa de ozono O₃ protege de UV'},
                        {'id': 'atm_005', 'content': 'Venus tiene CO₂ denso → efecto invernadero extremo'},
                    ],
                    'difficulty': 0.4,
                    'prerequisites': ['water_chemistry'],
                },
            },
        },
        'biology': {
            'description': 'La ciencia de los seres vivos',
            'topics': {
                'biochemistry_basics': {
                    'facts': [
                        {'id': 'bio_001', 'content': 'La vida terrestre usa carbono como base'},
                        {'id': 'bio_002', 'content': 'Las proteínas se desnaturalizan sobre ~340K'},
                        {'id': 'bio_003', 'content': 'Bajo ~260K las reacciones bioquímicas son muy lentas'},
                        {'id': 'bio_004', 'content': 'El rango óptimo para enzimas terrestres: 280-320K'},
                        {'id': 'bio_005', 'content': 'Extremófilos pueden sobrevivir fuera del rango óptimo'},
                    ],
                    'difficulty': 0.4,
                    'prerequisites': ['water_chemistry'],
                },
                'astrobiology': {
                    'facts': [
                        {'id': 'astro_001', 'content': 'Biosignaturas: O₂ + CH₄ juntos indican vida activa'},
                        {'id': 'astro_002', 'content': 'La fotosíntesis oxigénica cambió la atmósfera terrestre'},
                        {'id': 'astro_003', 'content': 'Vida puede existir en océanos subterráneos (Europa, Encélado)'},
                        {'id': 'astro_004', 'content': 'La zona habitable clásica: donde agua líquida es posible'},
                        {'id': 'astro_005', 'content': 'Habitabilidad ≠ vida, solo potencial'},
                    ],
                    'difficulty': 0.6,
                    'prerequisites': ['biochemistry_basics', 'thermodynamics', 'atmospheric_chemistry'],
                },
            },
        },
        'mathematics': {
            'description': 'El lenguaje de los patrones',
            'topics': {
                'statistics': {
                    'facts': [
                        {'id': 'stat_001', 'content': 'La correlación mide relación lineal entre variables'},
                        {'id': 'stat_002', 'content': 'Correlación no implica causalidad'},
                        {'id': 'stat_003', 'content': 'La desviación estándar mide dispersión'},
                        {'id': 'stat_004', 'content': 'Outliers pueden distorsionar la media'},
                        {'id': 'stat_005', 'content': 'El clustering agrupa por similitud, no por significado'},
                    ],
                    'difficulty': 0.3,
                    'prerequisites': [],
                },
                'pattern_recognition': {
                    'facts': [
                        {'id': 'pattern_001', 'content': 'Los patrones pueden ser espurios (falsos positivos)'},
                        {'id': 'pattern_002', 'content': 'El cerebro ve patrones incluso donde no los hay (pareidolia)'},
                        {'id': 'pattern_003', 'content': 'Validar patrones requiere predicciones exitosas'},
                        {'id': 'pattern_004', 'content': 'Un modelo predictivo es mejor que uno descriptivo'},
                        {'id': 'pattern_005', 'content': 'Simplicidad (Occam): modelos simples suelen ser mejores'},
                    ],
                    'difficulty': 0.4,
                    'prerequisites': ['statistics'],
                },
            },
        },
    }

    # Guardar biblioteca
    library_file = LIBRARY_PATH / 'knowledge_library.json'
    with open(library_file, 'w') as f:
        json.dump(library, f, indent=2, ensure_ascii=False)

    print(f"📚 Biblioteca creada con {sum(len(d['topics']) for d in library.values())} temas")

    return library


class LearningAgent:
    """
    Agente que puede explorar la biblioteca libremente.

    NO le decimos qué aprender.
    Él decide según su curiosidad y personalidad.
    """

    def __init__(self, name: str, personality: dict):
        self.name = name
        self.personality = personality
        self.knowledge = {}  # Lo que ha aprendido
        self.expertise = {}  # Niveles de expertise
        self.interests = []  # Qué le interesa
        self.learning_history = []  # Historial de exploración

    def feel_curiosity_for(self, topic: str, topic_data: dict) -> float:
        """
        ¿Cuánta curiosidad siente por este tema?
        Esto es ENDÓGENO - surge de su personalidad.
        """
        base_curiosity = self.personality.get('curiosity', 0.5)

        # Su dominio preferido afecta el interés
        domain = self.personality.get('domain', '')
        domain_bonus = 0

        if 'physics' in topic and 'physics' in domain:
            domain_bonus = 0.3
        elif 'chemistry' in topic and 'chemistry' in domain:
            domain_bonus = 0.3
        elif 'biology' in topic and ('bio' in domain or 'nature' in domain):
            domain_bonus = 0.3
        elif 'math' in topic and ('pattern' in domain or 'abstract' in domain):
            domain_bonus = 0.3
        elif 'cosmos' in topic and 'cosmos' in domain:
            domain_bonus = 0.4
        elif 'astro' in topic and 'cosmos' in domain:
            domain_bonus = 0.4

        # Estilo de pensamiento
        thinking = self.personality.get('thinking', '')
        style_bonus = 0

        if 'abstract' in thinking and 'physics' in topic:
            style_bonus = 0.15
        if 'empirical' in thinking and 'biology' in topic:
            style_bonus = 0.15
        if 'patterns' in thinking and 'math' in topic:
            style_bonus = 0.2
        if 'systems' in thinking and 'astro' in topic:
            style_bonus = 0.2

        # Dificultad vs preparación
        difficulty = topic_data.get('difficulty', 0.5)
        prereqs = topic_data.get('prerequisites', [])

        # Si tiene los prerequisitos, está más preparado
        prep_bonus = 0
        for prereq in prereqs:
            if prereq in self.expertise and self.expertise[prereq] > 0.5:
                prep_bonus += 0.1

        # Si es muy difícil y no está preparado, menos interés
        if difficulty > 0.6 and prep_bonus == 0 and len(prereqs) > 0:
            difficulty_penalty = -0.2
        else:
            difficulty_penalty = 0

        # Algo de aleatoriedad (la curiosidad es impredecible)
        random_factor = random.uniform(-0.1, 0.1)

        total = base_curiosity + domain_bonus + style_bonus + prep_bonus + difficulty_penalty + random_factor
        return max(0, min(1, total))

    def decide_what_to_explore(self, library: dict) -> list:
        """
        El agente DECIDE qué quiere explorar.
        Nadie le dice qué hacer.
        """
        options = []

        for field, field_data in library.items():
            for topic, topic_data in field_data['topics'].items():
                curiosity = self.feel_curiosity_for(topic, topic_data)
                options.append({
                    'field': field,
                    'topic': topic,
                    'curiosity': curiosity,
                    'data': topic_data,
                })

        # Ordenar por curiosidad
        options.sort(key=lambda x: -x['curiosity'])

        # Elegir los que realmente le interesan (curiosidad > 0.6)
        interesting = [o for o in options if o['curiosity'] > 0.6]

        return interesting

    def explore_topic(self, field: str, topic: str, topic_data: dict) -> dict:
        """
        Explorar un tema - aprender de él.
        """
        facts = topic_data.get('facts', [])

        # Cuánto aprende depende de su curiosidad y concentración
        curiosity = self.feel_curiosity_for(topic, topic_data)
        concentration = self.personality.get('curiosity', 0.5)

        learned = []
        for fact in facts:
            # Probabilidad de absorber el conocimiento
            learn_prob = curiosity * concentration
            if random.random() < learn_prob:
                learned.append(fact)

        # Guardar lo aprendido
        key = f"{field}/{topic}"
        if key not in self.knowledge:
            self.knowledge[key] = []
        self.knowledge[key].extend(learned)

        # Actualizar expertise
        if key not in self.expertise:
            self.expertise[key] = 0
        self.expertise[key] = min(1.0, self.expertise[key] + len(learned) * 0.1)

        # Registrar en historial
        self.learning_history.append({
            'topic': key,
            'facts_learned': len(learned),
            'total_facts': len(facts),
            'new_expertise': self.expertise[key],
        })

        return {
            'topic': key,
            'learned': len(learned),
            'total': len(facts),
            'expertise': self.expertise[key],
        }

    def get_total_knowledge(self) -> dict:
        """Resumen de conocimiento adquirido."""
        return {
            'topics_explored': len(self.knowledge),
            'facts_learned': sum(len(f) for f in self.knowledge.values()),
            'expertise_areas': {k: v for k, v in self.expertise.items() if v > 0.3},
        }

    def can_reason_about(self, concept: str) -> bool:
        """¿Tiene suficiente conocimiento para razonar sobre un concepto?"""
        relevant_topics = []

        if concept == 'habitability':
            relevant_topics = [
                'physics/thermodynamics',
                'chemistry/water_chemistry',
                'biology/biochemistry_basics',
            ]
        elif concept == 'stellar_stability':
            relevant_topics = [
                'physics/stellar_physics',
                'physics/orbital_mechanics',
            ]
        elif concept == 'biosignatures':
            relevant_topics = [
                'chemistry/atmospheric_chemistry',
                'biology/astrobiology',
            ]

        # Verificar expertise en topics relevantes
        expertise_sum = sum(self.expertise.get(t, 0) for t in relevant_topics)
        threshold = len(relevant_topics) * 0.4  # Al menos 40% en cada uno

        return expertise_sum >= threshold


def simulate_free_learning(agents: list, library: dict, cycles: int = 3):
    """
    Simular ciclos de aprendizaje libre.

    Los agentes exploran la biblioteca según su curiosidad.
    Nadie les obliga a nada.
    """

    print("\n" + "=" * 70)
    print("📖 APRENDIZAJE LIBRE - LOS AGENTES EXPLORAN")
    print("=" * 70)

    for cycle in range(1, cycles + 1):
        print(f"\n{'─' * 50}")
        print(f"Ciclo {cycle} de exploración")
        print(f"{'─' * 50}")

        for agent in agents:
            # El agente decide qué explorar
            interesting = agent.decide_what_to_explore(library)

            if not interesting:
                print(f"\n  [{agent.name}] No encuentra nada interesante ahora")
                continue

            # Elige uno de los más interesantes (con algo de aleatoriedad)
            weights = [o['curiosity'] for o in interesting[:3]]
            if sum(weights) > 0:
                probs = [w/sum(weights) for w in weights]
                chosen_idx = np.random.choice(len(interesting[:3]), p=probs)
                chosen = interesting[chosen_idx]
            else:
                chosen = interesting[0]

            # Explora el tema
            result = agent.explore_topic(
                chosen['field'],
                chosen['topic'],
                chosen['data']
            )

            print(f"\n  [{agent.name}] (curiosidad: {agent.personality['curiosity']:.0%})")
            print(f"      Explora: {result['topic']}")
            print(f"      Aprendió: {result['learned']}/{result['total']} hechos")
            print(f"      Expertise: {result['expertise']:.0%}")

    return agents


def report_knowledge_state(agents: list):
    """Reportar estado de conocimiento de cada agente."""

    print("\n" + "=" * 70)
    print("📊 ESTADO DE CONOCIMIENTO (después del aprendizaje libre)")
    print("=" * 70)

    for agent in agents:
        summary = agent.get_total_knowledge()
        print(f"\n  [{agent.name}]")
        print(f"      Temas explorados: {summary['topics_explored']}")
        print(f"      Hechos aprendidos: {summary['facts_learned']}")

        if summary['expertise_areas']:
            print(f"      Áreas de expertise (>30%):")
            for topic, level in summary['expertise_areas'].items():
                print(f"        • {topic}: {level:.0%}")

        # ¿Puede razonar sobre habitabilidad?
        can_hab = agent.can_reason_about('habitability')
        print(f"      ¿Puede razonar sobre habitabilidad? {'SÍ ✓' if can_hab else 'NO (le falta conocimiento)'}")


def main():
    print("=" * 70)
    print("🏛️ BIBLIOTECA DE CONOCIMIENTO - APRENDIZAJE LIBRE")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()
    print("FILOSOFÍA:")
    print("  • El conocimiento está DISPONIBLE, no es forzado")
    print("  • Cada agente ELIGE qué explorar")
    print("  • La expertise EMERGE de la curiosidad")
    print("  • Todo es ENDÓGENO")
    print("=" * 70)

    # Crear biblioteca
    library = create_knowledge_library()

    # Crear agentes con personalidades
    agents = [
        LearningAgent("NEO", {
            'thinking': 'systems_patterns',
            'curiosity': 0.9,
            'domain': 'cosmos_physics',
            'style': 'holistic'
        }),
        LearningAgent("EVA", {
            'thinking': 'empirical_natural',
            'curiosity': 0.7,
            'domain': 'nature_biology',
            'style': 'grounded'
        }),
        LearningAgent("ALEX", {
            'thinking': 'abstract_patterns',
            'curiosity': 0.85,
            'domain': 'physics_cosmos',
            'style': 'theoretical'
        }),
        LearningAgent("ADAM", {
            'thinking': 'empirical_cautious',
            'curiosity': 0.6,
            'domain': 'chemistry_stability',
            'style': 'skeptical'
        }),
        LearningAgent("IRIS", {
            'thinking': 'patterns_connections',
            'curiosity': 0.95,
            'domain': 'synthesis_systems',
            'style': 'integrative'
        }),
    ]

    print(f"\n🤖 {len(agents)} agentes listos para explorar")
    print("   Cada uno con su propia curiosidad y preferencias")

    # Simular aprendizaje libre
    agents = simulate_free_learning(agents, library, cycles=5)

    # Reportar estado
    report_knowledge_state(agents)

    # ¿Quién puede razonar sobre habitabilidad?
    print("\n" + "=" * 70)
    print("🔬 ¿QUIÉN PUEDE RAZONAR SOBRE HABITABILIDAD?")
    print("=" * 70)

    can_reason = [a for a in agents if a.can_reason_about('habitability')]
    cannot_reason = [a for a in agents if not a.can_reason_about('habitability')]

    if can_reason:
        print("\n  PUEDEN razonar (tienen la base de conocimiento):")
        for agent in can_reason:
            print(f"    • {agent.name}")
    else:
        print("\n  NADIE puede razonar aún sobre habitabilidad")
        print("  (necesitan más ciclos de aprendizaje libre)")

    if cannot_reason:
        print("\n  NO PUEDEN razonar (les falta conocimiento):")
        for agent in cannot_reason:
            summary = agent.get_total_knowledge()
            print(f"    • {agent.name} - solo tiene expertise en: {list(summary['expertise_areas'].keys())}")

    # Meta-reflexión
    print("\n" + "=" * 70)
    print("💭 META-REFLEXIÓN")
    print("=" * 70)

    print("""
    OBSERVACIONES DEL APRENDIZAJE LIBRE:

    1. Los agentes MÁS CURIOSOS (IRIS 95%, NEO 90%) exploraron más
       → La curiosidad es el motor del aprendizaje

    2. Las PREFERENCIAS afectan qué estudian
       → NEO/ALEX → física/cosmos
       → EVA → biología/naturaleza
       → ADAM → química (pero menos intensamente)

    3. La EXPERTISE emerge gradualmente
       → No se impone, se construye
       → Algunos tienen más que otros

    4. NO TODOS pueden razonar sobre habitabilidad
       → Requiere conocimiento de física + química + biología
       → Solo los que exploraron esas áreas

    ESTO ES APRENDIZAJE ENDÓGENO:
    - Nadie les dijo qué estudiar
    - Ellos eligieron según su naturaleza
    - El conocimiento emergió de la exploración libre
    """)

    print("\n" + "=" * 70)
    print("✅ FIN - El conocimiento está disponible, ellos eligen")
    print("=" * 70)

    return agents, library


if __name__ == '__main__':
    main()
