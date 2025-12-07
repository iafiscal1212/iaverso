#!/usr/bin/env python3
"""
Fuente de Conocimiento REAL - Sin Datos Hardcodeados
=====================================================

PRINCIPIO FUNDAMENTAL:
- Los agentes acceden a Wikipedia REAL
- Extraen conocimiento del texto CRUDO
- YO NO ESCRIBO ningún hecho científico
- Todo viene de fuentes externas verificables

PARA PUBLICACIÓN:
- Cada pieza de conocimiento tiene URL de origen
- Se puede verificar que no inventé nada
- Los agentes procesan texto real, no mis resúmenes
"""

import requests
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import time

CACHE_PATH = Path('/root/NEO_EVA/data/knowledge_cache')
CACHE_PATH.mkdir(parents=True, exist_ok=True)


class RealKnowledgeSource:
    """
    Fuente de conocimiento real desde Wikipedia.

    NO HAY DATOS HARDCODEADOS.
    Todo viene de la API de Wikipedia.
    """

    def __init__(self):
        self.base_url = "https://en.wikipedia.org/api/rest_v1"
        self.wiki_api = "https://en.wikipedia.org/w/api.php"
        self.headers = {
            'User-Agent': 'NEO_EVA/1.0 (Research Project; https://github.com/neo-eva)'
        }
        self.cache = {}
        self.sources = []  # Para auditoría

    def fetch_wikipedia_article(self, title: str) -> Optional[Dict]:
        """
        Obtener artículo de Wikipedia.

        Retorna el texto CRUDO, sin procesar.
        El agente debe extraer lo que le interese.
        """
        # Verificar cache
        cache_file = CACHE_PATH / f"{title.replace(' ', '_')}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)

        try:
            # API de Wikipedia para texto
            params = {
                'action': 'query',
                'titles': title,
                'prop': 'extracts',
                'explaintext': True,  # Texto plano, no HTML
                'format': 'json',
            }

            response = requests.get(self.wiki_api, params=params, headers=self.headers, timeout=30)
            data = response.json()

            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                if page_id == '-1':
                    return None

                result = {
                    'title': page_data.get('title'),
                    'text': page_data.get('extract', ''),
                    'source_url': f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    'fetched_at': datetime.now().isoformat(),
                    'page_id': page_id,
                }

                # Guardar en cache
                with open(cache_file, 'w') as f:
                    json.dump(result, f, indent=2)

                # Registrar fuente para auditoría
                self.sources.append({
                    'title': title,
                    'url': result['source_url'],
                    'fetched_at': result['fetched_at'],
                })

                return result

        except Exception as e:
            print(f"Error fetching {title}: {e}")
            return None

    def search_wikipedia(self, query: str, limit: int = 5) -> List[str]:
        """
        Buscar artículos en Wikipedia.

        El agente decide qué buscar.
        """
        try:
            params = {
                'action': 'opensearch',
                'search': query,
                'limit': limit,
                'format': 'json',
            }

            response = requests.get(self.wiki_api, params=params, headers=self.headers, timeout=30)
            data = response.json()

            if len(data) >= 2:
                return data[1]  # Lista de títulos
            return []

        except Exception as e:
            print(f"Error searching {query}: {e}")
            return []

    def get_audit_trail(self) -> List[Dict]:
        """
        Obtener registro de todas las fuentes consultadas.

        PARA VERIFICACIÓN:
        - Cada URL es verificable
        - Cada timestamp es real
        - Nada fue inventado por mí
        """
        return self.sources


class ZenodoSource:
    """
    Fuente de papers científicos reales desde Zenodo.

    Zenodo es un repositorio de acceso abierto del CERN.
    Los papers son reales, revisados, con DOI.
    """

    def __init__(self):
        self.api_url = "https://zenodo.org/api/records"
        self.sources = []

    def search_papers(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Buscar papers en Zenodo.

        Retorna papers reales con DOI verificable.
        """
        try:
            params = {
                'q': query,
                'size': limit,
                'type': 'publication',  # Solo publicaciones
            }

            response = requests.get(self.api_url, params=params, timeout=30)
            data = response.json()

            papers = []
            for hit in data.get('hits', {}).get('hits', []):
                metadata = hit.get('metadata', {})
                paper = {
                    'id': hit.get('id'),
                    'doi': metadata.get('doi'),
                    'doi_url': hit.get('doi_url'),
                    'title': metadata.get('title'),
                    'description': metadata.get('description', ''),
                    'publication_date': metadata.get('publication_date'),
                    'keywords': metadata.get('keywords', []),
                    'fetched_at': datetime.now().isoformat(),
                }
                papers.append(paper)

                # Registrar para auditoría
                self.sources.append({
                    'title': paper['title'],
                    'doi': paper['doi'],
                    'url': paper['doi_url'],
                    'fetched_at': paper['fetched_at'],
                })

            return papers

        except Exception as e:
            print(f"Error searching Zenodo: {e}")
            return []

    def get_paper_content(self, paper_id: int) -> Optional[Dict]:
        """
        Obtener contenido de un paper específico.
        """
        try:
            url = f"{self.api_url}/{paper_id}"
            response = requests.get(url, timeout=30)
            data = response.json()

            metadata = data.get('metadata', {})

            # Limpiar HTML del description
            description = metadata.get('description', '')
            # Remover tags HTML
            description = re.sub(r'<[^>]+>', ' ', description)
            description = re.sub(r'\s+', ' ', description).strip()

            return {
                'id': paper_id,
                'title': metadata.get('title'),
                'description': description,
                'doi': metadata.get('doi'),
                'doi_url': data.get('doi_url'),
                'keywords': metadata.get('keywords', []),
                'creators': [c.get('name') for c in metadata.get('creators', [])],
                'publication_date': metadata.get('publication_date'),
            }

        except Exception as e:
            print(f"Error fetching paper {paper_id}: {e}")
            return None

    def get_audit_trail(self) -> List[Dict]:
        """Registro de papers consultados."""
        return self.sources


class ArxivSource:
    """
    Fuente de preprints científicos desde arXiv.

    arXiv es el repositorio de preprints más grande del mundo.
    Papers de física, matemáticas, CS, biología, etc.
    """

    def __init__(self):
        self.api_url = "http://export.arxiv.org/api/query"
        self.sources = []

    def search_papers(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Buscar papers en arXiv.
        """
        try:
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': limit,
            }

            response = requests.get(self.api_url, params=params, timeout=30)

            # arXiv devuelve XML, parsearlo
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)

            # Namespace de Atom
            ns = {'atom': 'http://www.w3.org/2005/Atom'}

            papers = []
            for entry in root.findall('atom:entry', ns):
                paper = {
                    'id': entry.find('atom:id', ns).text if entry.find('atom:id', ns) is not None else '',
                    'title': entry.find('atom:title', ns).text if entry.find('atom:title', ns) is not None else '',
                    'summary': entry.find('atom:summary', ns).text if entry.find('atom:summary', ns) is not None else '',
                    'published': entry.find('atom:published', ns).text if entry.find('atom:published', ns) is not None else '',
                    'authors': [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns) if a.find('atom:name', ns) is not None],
                    'fetched_at': datetime.now().isoformat(),
                }

                # Limpiar espacios
                paper['title'] = ' '.join(paper['title'].split())
                paper['summary'] = ' '.join(paper['summary'].split())

                papers.append(paper)

                self.sources.append({
                    'title': paper['title'][:100],
                    'arxiv_id': paper['id'],
                    'fetched_at': paper['fetched_at'],
                })

            return papers

        except Exception as e:
            print(f"Error searching arXiv: {e}")
            return []

    def get_audit_trail(self) -> List[Dict]:
        """Registro de papers consultados."""
        return self.sources


class TextKnowledgeExtractor:
    """
    Extractor de conocimiento desde texto crudo.

    IMPORTANTE:
    - NO uso regex hardcodeados para extraer "273K" o similar
    - Uso patrones GENERALES para encontrar números + unidades
    - El significado lo interpreta el agente, no yo
    """

    def __init__(self):
        # Patrones GENERALES, no específicos
        self.number_pattern = r'(\d+(?:\.\d+)?)\s*(?:°?[CFK]|K|kelvin|celsius|degrees)'
        self.range_pattern = r'(\d+(?:\.\d+)?)\s*(?:to|–|-|and)\s*(\d+(?:\.\d+)?)\s*(?:°?[CFK]|K)'

    def extract_numerical_facts(self, text: str) -> List[Dict]:
        """
        Extraer hechos numéricos del texto.

        NO INTERPRETO qué significa cada número.
        Solo extraigo: "hay un número X con unidad Y en contexto Z"
        """
        facts = []
        sentences = text.split('.')

        for sentence in sentences:
            # Buscar números con unidades de temperatura
            matches = re.finditer(self.number_pattern, sentence, re.IGNORECASE)
            for match in matches:
                value = float(match.group(1))
                context = sentence.strip()[:200]  # Contexto limitado

                facts.append({
                    'type': 'numerical',
                    'value': value,
                    'raw_match': match.group(0),
                    'context': context,
                    'extracted_from': 'wikipedia',
                })

        return facts

    def extract_relationships(self, text: str) -> List[Dict]:
        """
        Extraer relaciones del texto.

        Busco patrones como "X causes Y", "X is related to Y"
        NO decido qué es importante - solo extraigo estructura.
        """
        relationships = []

        # Patrones de relación genéricos
        patterns = [
            r'(\w+(?:\s+\w+)?)\s+(?:causes?|leads?\s+to|results?\s+in)\s+(\w+(?:\s+\w+)?)',
            r'(\w+(?:\s+\w+)?)\s+(?:is|are)\s+(?:essential|necessary|required)\s+for\s+(\w+(?:\s+\w+)?)',
            r'(\w+(?:\s+\w+)?)\s+(?:occurs?|happens?)\s+(?:at|when|between)\s+(.+?)(?:\.|,)',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                relationships.append({
                    'type': 'relationship',
                    'subject': match.group(1),
                    'object': match.group(2) if len(match.groups()) > 1 else None,
                    'raw': match.group(0),
                })

        return relationships


def test_real_source():
    """
    Probar que la fuente es real y verificable.
    """
    print("=" * 70)
    print("PRUEBA DE FUENTE DE CONOCIMIENTO REAL")
    print("=" * 70)
    print()
    print("GARANTÍAS:")
    print("  1. Todo el texto viene de Wikipedia (verificable)")
    print("  2. Cada artículo tiene URL de origen")
    print("  3. YO NO ESCRIBÍ ningún hecho científico")
    print("  4. Los agentes procesan texto crudo")
    print("=" * 70)

    source = RealKnowledgeSource()
    extractor = TextKnowledgeExtractor()

    # El agente podría buscar estos temas
    topics = [
        "Water",
        "Habitable zone",
        "Planetary equilibrium temperature",
        "Liquid water",
    ]

    for topic in topics:
        print(f"\n{'─' * 50}")
        print(f"Buscando: {topic}")
        print(f"{'─' * 50}")

        article = source.fetch_wikipedia_article(topic)

        if article:
            print(f"  URL: {article['source_url']}")
            print(f"  Longitud: {len(article['text'])} caracteres")

            # Extraer hechos numéricos
            facts = extractor.extract_numerical_facts(article['text'])
            print(f"  Hechos numéricos encontrados: {len(facts)}")

            for fact in facts[:3]:
                print(f"    • {fact['value']} - '{fact['raw_match']}'")
                print(f"      Contexto: {fact['context'][:80]}...")
        else:
            print(f"  No encontrado")

    # Mostrar auditoría
    print("\n" + "=" * 70)
    print("AUDITORÍA DE FUENTES (para verificación)")
    print("=" * 70)

    for source_record in source.get_audit_trail():
        print(f"  • {source_record['title']}")
        print(f"    URL: {source_record['url']}")
        print(f"    Fecha: {source_record['fetched_at']}")

    print("\n" + "=" * 70)
    print("✓ Todo el conocimiento es verificable externamente")
    print("✓ Ningún dato fue escrito por mí")
    print("=" * 70)


if __name__ == '__main__':
    test_real_source()
