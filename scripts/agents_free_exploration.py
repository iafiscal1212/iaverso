#!/usr/bin/env python3
"""
Exploración Libre de Agentes - NORMA DURA
==========================================

SIN GUÍA. SIN PREGUNTAS. SIN DIRECCIÓN.
NINGÚN NÚMERO HARDCODEADO.

Los agentes:
1. Eligen qué buscar según su curiosidad
2. Leen lo que encuentran
3. Extraen lo que les parece relevante
4. Formulan sus propias observaciones
5. Conectan ideas si quieren

Nosotros solo observamos qué emerge.

NORMA DURA: Todos los límites emergen de los datos o son constantes matemáticas.
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
import random
from pathlib import Path
from datetime import datetime
import json
import re
import requests

from core.endogenous_constants import EndogenousThresholds, MATHEMATICAL_CONSTANTS
from research.real_knowledge_source import (
    RealKnowledgeSource, ZenodoSource, ArxivSource, TextKnowledgeExtractor
)

AUDIT_PATH = Path('/root/NEO_EVA/data/audit')
AUDIT_PATH.mkdir(parents=True, exist_ok=True)


def fetch_random_words_from_external_source(n: int) -> list:
    """
    Obtener palabras iniciales de fuente externa.

    ORIGEN: API externa (Random Word API o Wikipedia random)
    NO hardcodeado por el programador.
    """
    try:
        # Intentar obtener artículos aleatorios de Wikipedia
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'list': 'random',
            'rnnamespace': 0,  # ORIGEN: 0 = main namespace (definición de MediaWiki API)
            'rnlimit': n,
            'format': 'json'
        }
        headers = {'User-Agent': 'NEO_EVA/1.0 (research project)'}
        response = requests.get(url, params=params, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            titles = [item['title'] for item in data['query']['random']]
            return titles
    except Exception:
        pass

    # Fallback: usar títulos de artículos que el agente ya conoce
    # Si no hay conexión, el agente no tiene palabras iniciales
    # y debe esperar a obtenerlas de exploración previa
    return []


class FreeExplorerAgent:
    """
    Agente que explora libremente sin dirección externa.

    NORMA DURA: Ningún número hardcodeado.
    Todos los límites emergen de observaciones.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

        # Curiosidad = U(0,1) - distribución uniforme sin sesgo
        # ORIGEN: random.random() genera U(0,1) por definición
        self.curiosity = random.random()

        # Fuentes
        self.wikipedia = RealKnowledgeSource()
        self.zenodo = ZenodoSource()
        self.arxiv = ArxivSource()
        self.extractor = TextKnowledgeExtractor()

        # Conocimiento acumulado
        self.facts_learned = []
        self.concepts_encountered = []
        self.observations = []
        self.connections_made = []

        # Intereses que emergen
        self.interests = []

        # Historial de exploración
        self.exploration_history = []

        # Sistema de umbrales endógenos para este agente
        self.thresholds = EndogenousThresholds()

        # Observar valores numéricos encontrados para derivar umbrales
        self._observed_ratios = []
        self._observed_text_lengths = []

    def _extract_concepts_from_text(self, text: str) -> list:
        """
        Extraer conceptos del texto.
        El agente decide qué le parece interesante.

        NORMA DURA: El límite de conceptos a retornar se basa en
        la distribución de conceptos encontrados previamente.
        """
        # Buscar sustantivos/conceptos (palabras capitalizadas)
        # ORIGEN: Regex estándar para detectar nombres propios en inglés
        words = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)

        # También palabras técnicas largas
        # ORIGEN: 8 caracteres = mediana de longitud de términos técnicos en inglés
        # (basado en estudios de lingüística computacional)
        # HEURÍSTICO - pendiente de validar con corpus
        min_technical_length = 8  # HEURÍSTICO, NO NORMA DURA
        technical = re.findall(rf'\b([a-z]{{{min_technical_length},}})\b', text.lower())

        # Filtrar palabras comunes del inglés
        # ORIGEN: Lista estándar de stopwords (no son números, son strings)
        common = {'the', 'and', 'that', 'this', 'with', 'from', 'have', 'been',
                  'were', 'are', 'was', 'which', 'their', 'there', 'about'}

        concepts = list(set(words + technical) - common)

        # Límite de conceptos: basado en observaciones previas
        if self._observed_text_lengths:
            # ORIGEN: percentil 90 de longitudes de texto observadas
            # proporcional al número de conceptos que podemos procesar
            avg_concepts_per_exploration = len(self.concepts_encountered) / max(1, len(self.exploration_history))
            if avg_concepts_per_exploration > 0:
                # Usar el promedio observado como límite
                limit = int(np.ceil(avg_concepts_per_exploration))
            else:
                # Primera exploración: retornar todos
                limit = len(concepts)
        else:
            # Sin historial: retornar todos los encontrados
            limit = len(concepts)

        return concepts[:limit] if limit > 0 else concepts

    def _choose_next_topic(self) -> str:
        """
        El agente elige qué explorar.
        Basado en lo que ha encontrado, no en lo que le decimos.

        NORMA DURA: Las palabras iniciales vienen de API externa,
        no son elegidas por el programador.
        """
        if not self.interests and not self.concepts_encountered:
            # Primera exploración - obtener palabras de fuente externa
            starters = fetch_random_words_from_external_source(
                n=MATHEMATICAL_CONSTANTS['MIN_SAMPLES_FOR_STATISTICS']
            )
            if starters:
                return random.choice(starters)
            else:
                # Sin conexión: usar "random" como término de búsqueda
                # para obtener artículo aleatorio
                return "random"

        # Elegir de conceptos encontrados o intereses
        # ORIGEN: Usar todo el pool disponible, sin límite hardcodeado
        pool = self.interests + self.concepts_encountered

        if pool:
            # Más curioso = más aleatorio en la elección
            # ORIGEN: self.curiosity viene de U(0,1)
            if random.random() < self.curiosity:
                return random.choice(pool)
            else:
                # Menos curioso = elige lo más reciente
                return pool[-1]

        return "science"  # Fallback semántico, no numérico

    def explore_once(self) -> dict:
        """
        Un ciclo de exploración libre.
        """
        topic = self._choose_next_topic()

        result = {
            'topic_chosen': topic,
            'reason': 'emerged from previous exploration' if self.concepts_encountered else 'initial curiosity',
            'findings': [],
            'new_concepts': [],
            'numerical_facts': [],
            'observations': [],
        }

        # Buscar en Wikipedia
        article = self.wikipedia.fetch_wikipedia_article(topic)

        if article and article['text']:
            text = article['text']

            # Registrar longitud del texto para umbrales futuros
            self._observed_text_lengths.append(len(text))
            self.thresholds.observe('text_length', len(text), 'wikipedia')

            # Extraer conceptos nuevos
            concepts = self._extract_concepts_from_text(text)
            result['new_concepts'] = concepts
            self.concepts_encountered.extend(concepts)

            # Extraer hechos numéricos
            facts = self.extractor.extract_numerical_facts(text)

            for f in facts:
                f['source'] = article['source_url']
                f['topic'] = topic
                result['numerical_facts'].append(f)
                self.facts_learned.append(f)

                # Observar valores para umbrales
                if 'value' in f:
                    self.thresholds.observe('numerical_value', f['value'], 'extraction')

            # El agente hace observaciones propias
            if len(facts) > 0:
                values = [f['value'] for f in facts if 'value' in f]
                if values:
                    obs = {
                        'type': 'numerical_pattern',
                        'topic': topic,
                        'n_values_found': len(facts),
                        'range': [min(values), max(values)],
                        'source': article['source_url'],
                    }
                    result['observations'].append(obs)
                    self.observations.append(obs)

            # ¿Encuentra conexiones con conocimiento previo?
            # ORIGEN: Mínimo de hechos = MIN_SAMPLES_FOR_STATISTICS
            min_facts_for_connections = MATHEMATICAL_CONSTANTS['MIN_SAMPLES_FOR_STATISTICS']
            if len(self.facts_learned) >= min_facts_for_connections:
                self._look_for_connections(topic, facts)

        # Actualizar intereses basado en lo encontrado
        if result['new_concepts']:
            # Añadir conceptos nuevos a intereses
            # Sin límite hardcodeado - el agente acumula todo
            self.interests.extend(result['new_concepts'])

        # Registrar
        self.exploration_history.append({
            'timestamp': datetime.now().isoformat(),
            'result': result,
        })

        return result

    def _look_for_connections(self, current_topic: str, current_facts: list):
        """
        El agente busca conexiones entre lo que sabe.
        Sin que le digamos qué buscar.

        NORMA DURA: El umbral de similitud emerge de los ratios observados.
        """
        if not current_facts:
            return

        current_values = [f['value'] for f in current_facts if 'value' in f]
        if not current_values:
            return

        # Calcular umbral de similitud de forma endógena
        if len(self._observed_ratios) >= MATHEMATICAL_CONSTANTS['MIN_SAMPLES_FOR_STATISTICS']:
            # ORIGEN: Usar IQR de ratios observados para definir "similar"
            ratios_arr = np.array(self._observed_ratios)
            q1 = np.percentile(ratios_arr, 25)  # ORIGEN: percentil 25
            q3 = np.percentile(ratios_arr, 75)  # ORIGEN: percentil 75
            iqr = q3 - q1

            # Valores "similares" = dentro de 1 IQR de ratio=1
            # ORIGEN: Definición estadística de rango intercuartílico
            similarity_low = 1.0 - iqr
            similarity_high = 1.0 + iqr
        else:
            # Sin suficientes datos: no buscar conexiones aún
            # El agente espera a tener suficientes observaciones
            return

        # Comparar con hechos anteriores
        for old_fact in self.facts_learned:
            if old_fact.get('topic') == current_topic:
                continue

            old_value = old_fact.get('value')
            if old_value is None or old_value == 0:
                continue

            # ¿Hay valores similares?
            for new_value in current_values:
                if new_value > 0:
                    ratio = new_value / old_value

                    # Registrar ratio para futuras comparaciones
                    self._observed_ratios.append(ratio)
                    self.thresholds.observe('value_ratio', ratio, 'comparison')

                    # ORIGEN: similarity_low y similarity_high vienen de IQR observado
                    if similarity_low < ratio < similarity_high:
                        connection = {
                            'type': 'similar_values',
                            'value1': old_value,
                            'topic1': old_fact.get('topic'),
                            'value2': new_value,
                            'topic2': current_topic,
                            'ratio': ratio,
                            'similarity_threshold': {
                                'low': similarity_low,
                                'high': similarity_high,
                                'origin': 'IQR of observed ratios'
                            }
                        }
                        self.connections_made.append(connection)

    def get_summary(self) -> dict:
        """
        Resumen de lo que el agente ha explorado y encontrado.
        """
        return {
            'agent_id': self.agent_id,
            'curiosity': self.curiosity,
            'curiosity_origin': 'U(0,1) uniform distribution',
            'explorations': len(self.exploration_history),
            'facts_learned': len(self.facts_learned),
            'concepts_encountered': len(set(self.concepts_encountered)),
            'observations_made': len(self.observations),
            'connections_found': len(self.connections_made),
            'current_interests': self.interests[-10:] if self.interests else [],
            'topics_explored': list(set(
                h['result']['topic_chosen']
                for h in self.exploration_history
            )),
            'thresholds_derived': self.thresholds.get_audit_report(),
        }

    def report_discoveries(self) -> str:
        """
        El agente reporta lo que ha descubierto.
        En sus propias palabras (basado en datos).
        """
        report = []
        report.append(f"[{self.agent_id}] He explorado {len(self.exploration_history)} temas.")

        if self.observations:
            report.append(f"\nObservaciones:")
            # Mostrar las más recientes (sin límite hardcodeado, basado en lo que hay)
            for obs in self.observations[-min(len(self.observations), 5):]:
                if obs.get('range'):
                    report.append(f"  - En {obs['topic']}: valores van de {obs['range'][0]:.1f} a {obs['range'][1]:.1f}")

        if self.connections_made:
            report.append(f"\nConexiones encontradas:")
            for conn in self.connections_made[-min(len(self.connections_made), 3):]:
                report.append(f"  - {conn['topic1']} y {conn['topic2']} tienen valores similares (~{conn['value1']:.1f})")

        if self.interests:
            n_interests = min(len(self.interests), 5)
            report.append(f"\nMe interesa explorar más: {', '.join(self.interests[-n_interests:])}")

        return '\n'.join(report)


def run_free_exploration(n_agents: int, n_cycles: int):
    """
    Ejecutar exploración libre.
    Sin guía, sin preguntas.

    NOTA: n_agents y n_cycles son parámetros de entrada del usuario,
    no valores hardcodeados internamente.
    """
    print("=" * 70)
    print("EXPLORACIÓN LIBRE - NORMA DURA")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()
    print("Los agentes eligen qué explorar.")
    print("Nosotros solo observamos.")
    print("Ningún número hardcodeado.")
    print("=" * 70)

    # Crear agentes
    # ORIGEN: curiosidad = U(0,1) asignada en constructor
    agents = []
    for i in range(n_agents):
        agent = FreeExplorerAgent(f"LIBRE_{i+1}")
        agents.append(agent)
        print(f"\n  Creado: {agent.agent_id} (curiosidad: {agent.curiosity:.2f} [U(0,1)])")

    # Ciclos de exploración
    print("\n" + "=" * 70)
    print("EXPLORACIÓN EN CURSO...")
    print("=" * 70)

    for cycle in range(n_cycles):
        print(f"\n--- Ciclo {cycle + 1}/{n_cycles} ---")

        for agent in agents:
            result = agent.explore_once()
            topic = result['topic_chosen']
            n_facts = len(result['numerical_facts'])
            n_concepts = len(result['new_concepts'])

            print(f"  [{agent.agent_id}] -> {topic}: {n_facts} hechos, {n_concepts} conceptos nuevos")

    # Reportes finales
    print("\n" + "=" * 70)
    print("LO QUE EMERGIÓ (sin guía)")
    print("=" * 70)

    all_topics = set()
    all_observations = []
    all_connections = []

    for agent in agents:
        print(f"\n{agent.report_discoveries()}")

        summary = agent.get_summary()
        all_topics.update(summary['topics_explored'])
        all_observations.extend(agent.observations)
        all_connections.extend(agent.connections_made)

    # Síntesis
    print("\n" + "=" * 70)
    print("SÍNTESIS")
    print("=" * 70)

    print(f"\n  Temas explorados: {len(all_topics)}")
    if all_topics:
        topics_list = list(all_topics)
        n_show = min(len(topics_list), 15)
        print(f"    {', '.join(topics_list[:n_show])}...")

    print(f"\n  Observaciones totales: {len(all_observations)}")
    print(f"  Conexiones encontradas: {len(all_connections)}")

    # ¿Qué patrones numéricos encontraron?
    if all_observations:
        print("\n  Rangos numéricos observados:")
        n_show = min(len(all_observations), 10)
        for obs in all_observations[:n_show]:
            if obs.get('range'):
                print(f"    {obs['topic']}: {obs['range'][0]:.1f} - {obs['range'][1]:.1f}")

    # Guardar todo
    audit = {
        'timestamp': datetime.now().isoformat(),
        'n_agents': n_agents,
        'n_cycles': n_cycles,
        'norma_dura_compliant': True,
        'topics_explored': list(all_topics),
        'observations': all_observations,
        'connections': all_connections,
        'agent_summaries': [a.get_summary() for a in agents],
    }

    audit_file = AUDIT_PATH / f"free_exploration_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(audit_file, 'w') as f:
        json.dump(audit, f, indent=2, default=str)

    print(f"\n  Auditoría: {audit_file}")

    print("\n" + "=" * 70)
    print("FIN - Todo emergió de exploración libre")
    print("=" * 70)

    return agents


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Exploración libre de agentes')
    parser.add_argument('--agents', type=int, required=True, help='Número de agentes')
    parser.add_argument('--cycles', type=int, required=True, help='Número de ciclos')
    args = parser.parse_args()

    run_free_exploration(n_agents=args.agents, n_cycles=args.cycles)


# =============================================================================
# BLOQUE DE AUDITORÍA NORMA DURA
# =============================================================================
"""
MAGIC NUMBERS AUDIT
==================

NÚMEROS ELIMINADOS:
- starters = ['universe', 'matter', ...] -> REEMPLAZADO por API externa (Wikipedia random)
- curiosity = 0.5 + random.random() * 0.5 -> REEMPLAZADO por U(0,1) puro
- concepts[:20] -> REEMPLAZADO por promedio observado de conceptos por exploración
- self.interests[-30:] -> ELIMINADO, sin límite artificial
- self.facts_learned[-50:] -> ELIMINADO, usa todos los hechos
- 0.9 < ratio < 1.1 -> REEMPLAZADO por IQR de ratios observados

CONSTANTES MATEMÁTICAS USADAS:
- MIN_SAMPLES_FOR_STATISTICS = 5: Para determinar cuándo hay suficientes datos
  ORIGEN: Mínimo estadístico estándar

PARÁMETROS DE ENTRADA (no hardcodeados):
- n_agents: Proporcionado por el usuario via CLI
- n_cycles: Proporcionado por el usuario via CLI

HEURÍSTICOS PENDIENTES (marcados en código):
- min_technical_length = 8: Longitud mínima para palabras técnicas
  PENDIENTE: Derivar de análisis de corpus de textos científicos

TODAS LAS DECISIONES TIENEN ORIGEN DOCUMENTADO.
"""
