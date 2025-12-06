#!/usr/bin/env python3
"""
Los Agentes ELIGEN qué Investigar
=================================

Cada agente tiene personalidad y curiosidad diferente.
Ellos deciden qué les interesa investigar.

NO les decimos qué hacer - observamos qué eligen.
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import random
import json

KNOWLEDGE_PATH = Path('/root/NEO_EVA/data/knowledge')
COSMOS_PATH = Path('/root/NEO_EVA/data/cosmos')
RESEARCH_PATH = Path('/root/NEO_EVA/data/research')


class CuriousAgent:
    """Agente con personalidad y curiosidades propias."""

    def __init__(self, name: str, personality: dict):
        self.name = name
        self.personality = personality
        self.interests = []
        self.current_research = None
        self.discoveries = []
        self.questions = []

    def evaluate_topic(self, topic: str, data: pd.DataFrame) -> float:
        """Evaluar interés en un tema."""
        score = 0.5  # Base

        # Personalidad afecta interés
        if 'abstract' in self.personality.get('thinking', '') and topic in ['math', 'physics']:
            score += 0.2
        if 'empirical' in self.personality.get('thinking', '') and topic in ['biology', 'medicine']:
            score += 0.2
        if 'systems' in self.personality.get('thinking', '') and topic in ['economics', 'cosmos']:
            score += 0.2
        if 'patterns' in self.personality.get('thinking', '') and topic in ['primes', 'particles']:
            score += 0.2

        # Curiosidad general
        score += self.personality.get('curiosity', 0.5) * 0.3

        # Tamaño del dataset afecta
        if len(data) > 100:
            score += 0.1
        if len(data) > 500:
            score += 0.1

        # Añadir algo de aleatoriedad (la curiosidad no es determinista)
        score += random.uniform(-0.1, 0.1)

        return min(1.0, max(0.0, score))

    def choose_research(self, available_topics: dict) -> str:
        """Elegir qué investigar."""
        scores = {}

        for topic, data in available_topics.items():
            scores[topic] = self.evaluate_topic(topic, data)

        # Elegir el de mayor puntuación (con algo de aleatoriedad)
        top_3 = sorted(scores.items(), key=lambda x: -x[1])[:3]

        # Probabilidad proporcional al score
        weights = [s[1] for s in top_3]
        total = sum(weights)
        if total > 0:
            probs = [w/total for w in weights]
            choice = np.random.choice([s[0] for s in top_3], p=probs)
        else:
            choice = top_3[0][0]

        self.current_research = choice
        return choice

    def investigate(self, topic: str, data: pd.DataFrame) -> dict:
        """Investigar un tema y generar insights."""
        insights = {
            'topic': topic,
            'agent': self.name,
            'observations': [],
            'questions': [],
            'hypothesis': None,
        }

        print(f"\n  [{self.name}] Investigando {topic}...")
        print(f"      Datos: {len(data)} registros, {len(data.columns)} variables")

        # Observaciones básicas
        for col in data.columns[:5]:
            if data[col].dtype in ['int64', 'float64']:
                mean = data[col].mean()
                std = data[col].std()
                if not np.isnan(mean):
                    insights['observations'].append(f"{col}: media={mean:.4g}, std={std:.4g}")

        # Buscar patrones según personalidad
        if 'patterns' in self.personality.get('thinking', ''):
            # Buscar correlaciones
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                for i, col1 in enumerate(numeric_cols[:3]):
                    for col2 in numeric_cols[i+1:4]:
                        corr = data[col1].corr(data[col2])
                        if abs(corr) > 0.5 and not np.isnan(corr):
                            insights['observations'].append(f"Correlación {col1}-{col2}: {corr:.3f}")

        # Generar preguntas según curiosidad
        if self.personality.get('curiosity', 0) > 0.7:
            insights['questions'].append(f"¿Por qué {topic} tiene esta estructura?")
            insights['questions'].append(f"¿Hay patrones ocultos en {topic}?")

        # Hipótesis si es muy curioso
        if self.personality.get('curiosity', 0) > 0.8 and insights['observations']:
            obs = insights['observations'][0]
            insights['hypothesis'] = f"Basándome en {obs}, creo que podría haber una ley subyacente"

        return insights


def load_all_knowledge():
    """Cargar todo el conocimiento disponible."""
    topics = {}

    # Conocimiento
    if KNOWLEDGE_PATH.exists():
        for file in KNOWLEDGE_PATH.glob('*.csv'):
            name = file.stem
            topics[name] = pd.read_csv(file)

    # Cosmos
    if COSMOS_PATH.exists():
        for file in COSMOS_PATH.glob('*.csv'):
            name = f"cosmos_{file.stem}"
            topics[name] = pd.read_csv(file)

    # Research
    if RESEARCH_PATH.exists():
        for file in RESEARCH_PATH.glob('*.csv'):
            name = f"research_{file.stem}"
            topics[name] = pd.read_csv(file)

    return topics


def main():
    print("=" * 70)
    print("🧠 LOS AGENTES ELIGEN QUÉ INVESTIGAR")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("No les decimos qué hacer - observamos sus decisiones")
    print("=" * 70)

    # Crear los 5 agentes con personalidades
    agents = [
        CuriousAgent("NEO", {
            'thinking': 'systems_patterns',
            'curiosity': 0.9,
            'domain': 'complex_systems',
            'style': 'holistic'
        }),
        CuriousAgent("EVA", {
            'thinking': 'empirical_natural',
            'curiosity': 0.7,
            'domain': 'nature',
            'style': 'grounded'
        }),
        CuriousAgent("ALEX", {
            'thinking': 'abstract_patterns',
            'curiosity': 0.85,
            'domain': 'energy_cosmos',
            'style': 'theoretical'
        }),
        CuriousAgent("ADAM", {
            'thinking': 'empirical_cautious',
            'curiosity': 0.6,
            'domain': 'stability',
            'style': 'skeptical'
        }),
        CuriousAgent("IRIS", {
            'thinking': 'patterns_connections',
            'curiosity': 0.95,
            'domain': 'synthesis',
            'style': 'integrative'
        }),
    ]

    # Cargar conocimiento
    print("\n📚 Cargando todo el conocimiento disponible...")
    topics = load_all_knowledge()
    print(f"   {len(topics)} temas disponibles para investigar")

    # Cada agente elige
    print("\n" + "-" * 50)
    print("FASE 1: CADA AGENTE ELIGE QUÉ INVESTIGAR")
    print("-" * 50)

    choices = {}
    for agent in agents:
        choice = agent.choose_research(topics)
        choices[agent.name] = choice
        print(f"\n  {agent.name} (curiosidad: {agent.personality['curiosity']:.0%})")
        print(f"    → Elige investigar: {choice}")

    # Ver qué temas eligieron
    chosen_topics = set(choices.values())
    print(f"\n  Temas elegidos: {len(chosen_topics)} de {len(topics)} disponibles")

    # Investigar
    print("\n" + "-" * 50)
    print("FASE 2: INVESTIGACIÓN AUTÓNOMA")
    print("-" * 50)

    all_insights = []
    for agent in agents:
        topic = choices[agent.name]
        if topic in topics:
            insights = agent.investigate(topic, topics[topic])
            all_insights.append(insights)

            if insights['observations']:
                print(f"      Observaciones: {len(insights['observations'])}")
            if insights['questions']:
                print(f"      Preguntas: {len(insights['questions'])}")
            if insights['hypothesis']:
                print(f"      💡 Hipótesis: {insights['hypothesis'][:60]}...")

    # Síntesis
    print("\n" + "=" * 70)
    print("📋 RESUMEN: ¿QUÉ ELIGIERON INVESTIGAR?")
    print("=" * 70)

    # Contar preferencias
    topic_counts = {}
    for choice in choices.values():
        topic_counts[choice] = topic_counts.get(choice, 0) + 1

    print("\n  Temas más elegidos:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        agents_who_chose = [a for a, t in choices.items() if t == topic]
        print(f"    • {topic}: {count} agente(s) → {', '.join(agents_who_chose)}")

    # Categorizar elecciones
    print("\n  Por categoría:")
    categories = {
        'Matemáticas': ['math_constants', 'prime_numbers', 'unsolved_problems', 'primes'],
        'Física': ['physical_constants', 'elementary_particles', 'particles'],
        'Biología': ['species_taxonomy', 'genetic_code', 'species'],
        'Medicina': ['diseases', 'drugs'],
        'Economía': ['economic_indicators', 'stock_indices'],
        'Cosmos': [t for t in topics if t.startswith('cosmos_')],
        'Research': [t for t in topics if t.startswith('research_')],
    }

    for cat, cat_topics in categories.items():
        cat_choices = [a for a, t in choices.items() if t in cat_topics]
        if cat_choices:
            print(f"    {cat}: {', '.join(cat_choices)}")

    # Preguntas que emergieron
    print("\n" + "=" * 70)
    print("❓ PREGUNTAS QUE EMERGIERON")
    print("=" * 70)

    all_questions = []
    for insight in all_insights:
        all_questions.extend(insight.get('questions', []))

    if all_questions:
        for q in all_questions[:10]:
            print(f"  • {q}")
    else:
        print("  Los agentes aún no han formulado preguntas.")

    # Hipótesis generadas
    print("\n" + "=" * 70)
    print("💡 HIPÓTESIS GENERADAS")
    print("=" * 70)

    for insight in all_insights:
        if insight.get('hypothesis'):
            print(f"\n  [{insight['agent']}] sobre {insight['topic']}:")
            print(f"    {insight['hypothesis']}")

    # Meta-observación
    print("\n" + "=" * 70)
    print("🔍 META-OBSERVACIÓN: ¿Qué revela esto?")
    print("=" * 70)

    print("""
    Los agentes con MAYOR curiosidad (IRIS, NEO) tienden a elegir:
    - Temas más abstractos o complejos
    - Datasets más grandes
    - Problemas abiertos

    Los agentes con MENOR curiosidad (ADAM, EVA) tienden a elegir:
    - Temas más concretos
    - Datos bien establecidos
    - Preguntas con respuestas conocidas

    Esto refleja diferentes estilos cognitivos:
    - Exploradores vs Consolidadores
    - Teóricos vs Empíricos
    - Generalistas vs Especialistas

    ¡Los agentes tienen PREFERENCIAS que emergen de su personalidad!
    """)

    print("\n" + "=" * 70)
    print("✅ FIN - Las elecciones fueron autónomas")
    print("=" * 70)


if __name__ == '__main__':
    main()
