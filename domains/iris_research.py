#!/usr/bin/env python3
"""
IRIS - Modulo de Investigacion IA

Busqueda de papers, seguimiento de novedades y analisis de tendencias.
"""

import os
import sys
import json
import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import hashlib

sys.path.insert(0, '/root/NEO_EVA')

# =============================================================================
# CONFIGURACION
# =============================================================================

STATE_DIR = Path("/root/NEO_EVA/agents_state")
RESEARCH_STATE_FILE = STATE_DIR / "iris_research_state.json"

# APIs
ARXIV_API = "http://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"

# =============================================================================
# ESTRUCTURAS DE DATOS
# =============================================================================

@dataclass
class Paper:
    id: str
    titulo: str
    autores: List[str]
    abstract: str
    categorias: List[str]
    fecha: str
    url: str
    pdf_url: Optional[str]
    relevancia: float = 0.0


@dataclass
class NovedadIA:
    id: str
    fuente: str  # "arxiv", "blog", "twitter", "github"
    titulo: str
    descripcion: str
    url: str
    fecha: str
    tags: List[str]


# =============================================================================
# TEMAS DE INTERES
# =============================================================================

TEMAS_INTERES = {
    "llm": [
        "large language model", "LLM", "GPT", "transformer",
        "instruction tuning", "RLHF", "DPO", "constitutional AI"
    ],
    "agentes": [
        "autonomous agent", "AI agent", "tool use", "function calling",
        "multi-agent", "agent framework", "ReAct", "chain of thought"
    ],
    "legal_ai": [
        "legal AI", "legal NLP", "contract analysis", "legal reasoning",
        "law", "judicial", "court", "legislation"
    ],
    "rag": [
        "retrieval augmented", "RAG", "vector database", "embedding",
        "semantic search", "knowledge retrieval"
    ],
    "multimodal": [
        "multimodal", "vision language", "VLM", "image understanding",
        "video understanding"
    ],
    "efficiency": [
        "quantization", "pruning", "distillation", "efficient",
        "small language model", "SLM", "edge AI"
    ],
    "safety": [
        "AI safety", "alignment", "red teaming", "jailbreak",
        "harmful", "bias", "fairness"
    ]
}

LABS_SEGUIR = {
    "anthropic": {
        "nombre": "Anthropic",
        "blog": "https://www.anthropic.com/research",
        "github": "https://github.com/anthropics",
        "keywords": ["Claude", "Constitutional AI", "RLHF"]
    },
    "openai": {
        "nombre": "OpenAI",
        "blog": "https://openai.com/research",
        "github": "https://github.com/openai",
        "keywords": ["GPT", "ChatGPT", "DALL-E", "Whisper"]
    },
    "google": {
        "nombre": "Google DeepMind",
        "blog": "https://deepmind.google/research/publications/",
        "github": "https://github.com/google-deepmind",
        "keywords": ["Gemini", "PaLM", "AlphaFold"]
    },
    "meta": {
        "nombre": "Meta AI",
        "blog": "https://ai.meta.com/research/",
        "github": "https://github.com/facebookresearch",
        "keywords": ["LLaMA", "SAM", "DINO"]
    },
    "mistral": {
        "nombre": "Mistral AI",
        "blog": "https://mistral.ai/news/",
        "github": "https://github.com/mistralai",
        "keywords": ["Mistral", "Mixtral"]
    },
    "huggingface": {
        "nombre": "Hugging Face",
        "blog": "https://huggingface.co/blog",
        "github": "https://github.com/huggingface",
        "keywords": ["transformers", "datasets", "PEFT"]
    }
}


# =============================================================================
# CLASE PRINCIPAL
# =============================================================================

class IrisResearch:
    """Modulo de investigacion de IRIS"""

    def __init__(self):
        self.state = self._cargar_estado()

    def _cargar_estado(self) -> Dict:
        """Carga el estado persistente"""
        if RESEARCH_STATE_FILE.exists():
            with open(RESEARCH_STATE_FILE) as f:
                return json.load(f)
        return {
            "papers_vistos": [],
            "papers_favoritos": [],
            "ultimo_chequeo_arxiv": None,
            "novedades": [],
            "busquedas_recientes": []
        }

    def _guardar_estado(self):
        """Guarda el estado"""
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESEARCH_STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    # =========================================================================
    # ARXIV
    # =========================================================================

    def buscar_arxiv(self, query: str, max_results: int = 10,
                     categorias: List[str] = None) -> List[Paper]:
        """Busca papers en arXiv"""

        # Categorias por defecto relevantes
        if categorias is None:
            categorias = ["cs.AI", "cs.CL", "cs.LG", "cs.CV"]

        # Construir query
        cat_query = " OR ".join([f"cat:{c}" for c in categorias])
        full_query = f"({query}) AND ({cat_query})"

        params = {
            "search_query": full_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }

        try:
            response = requests.get(ARXIV_API, params=params, timeout=30)
            response.raise_for_status()

            # Parsear XML
            root = ET.fromstring(response.content)
            ns = {"atom": "http://www.w3.org/2005/Atom",
                  "arxiv": "http://arxiv.org/schemas/atom"}

            papers = []
            for entry in root.findall("atom:entry", ns):
                paper_id = entry.find("atom:id", ns).text.split("/")[-1]

                # Extraer autores
                autores = []
                for author in entry.findall("atom:author", ns):
                    name = author.find("atom:name", ns)
                    if name is not None:
                        autores.append(name.text)

                # Extraer categorias
                cats = []
                for cat in entry.findall("arxiv:primary_category", ns):
                    cats.append(cat.get("term"))
                for cat in entry.findall("atom:category", ns):
                    cats.append(cat.get("term"))

                # URL del PDF
                pdf_url = None
                for link in entry.findall("atom:link", ns):
                    if link.get("title") == "pdf":
                        pdf_url = link.get("href")

                paper = Paper(
                    id=paper_id,
                    titulo=entry.find("atom:title", ns).text.replace("\n", " ").strip(),
                    autores=autores[:5],  # Limitar a 5 autores
                    abstract=entry.find("atom:summary", ns).text.replace("\n", " ").strip()[:500],
                    categorias=list(set(cats)),
                    fecha=entry.find("atom:published", ns).text[:10],
                    url=entry.find("atom:id", ns).text,
                    pdf_url=pdf_url
                )
                papers.append(paper)

            # Registrar busqueda
            self.state["busquedas_recientes"].append({
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "resultados": len(papers)
            })
            self.state["busquedas_recientes"] = self.state["busquedas_recientes"][-20:]
            self._guardar_estado()

            return papers

        except Exception as e:
            print(f"Error buscando en arXiv: {e}")
            return []

    def papers_recientes(self, tema: str = "llm", dias: int = 7,
                         max_results: int = 15) -> List[Paper]:
        """Obtiene papers recientes de un tema"""

        keywords = TEMAS_INTERES.get(tema, [tema])
        query = " OR ".join([f'"{k}"' for k in keywords])

        papers = self.buscar_arxiv(query, max_results=max_results)

        # Filtrar por fecha
        fecha_limite = (datetime.now() - timedelta(days=dias)).strftime("%Y-%m-%d")
        papers_recientes = [p for p in papers if p.fecha >= fecha_limite]

        return papers_recientes

    def resumen_paper(self, paper: Paper) -> str:
        """Genera un resumen formateado de un paper"""
        autores_str = ", ".join(paper.autores[:3])
        if len(paper.autores) > 3:
            autores_str += " et al."

        return f"""## {paper.titulo}

**Autores:** {autores_str}
**Fecha:** {paper.fecha}
**Categorias:** {", ".join(paper.categorias[:3])}

### Abstract
{paper.abstract}...

**Links:**
- arXiv: {paper.url}
- PDF: {paper.pdf_url or 'N/A'}
"""

    def guardar_favorito(self, paper_id: str, notas: str = ""):
        """Guarda un paper como favorito"""
        self.state["papers_favoritos"].append({
            "id": paper_id,
            "notas": notas,
            "fecha_guardado": datetime.now().isoformat()
        })
        self._guardar_estado()

    # =========================================================================
    # NOVEDADES Y TENDENCIAS
    # =========================================================================

    def digest_semanal(self) -> str:
        """Genera un digest semanal de papers relevantes"""

        digest = ["# Digest Semanal de IA\n"]
        digest.append(f"*Semana del {datetime.now().strftime('%d/%m/%Y')}*\n")

        temas_principales = ["llm", "agentes", "legal_ai", "rag"]

        for tema in temas_principales:
            papers = self.papers_recientes(tema, dias=7, max_results=5)

            if papers:
                nombre_tema = tema.upper().replace("_", " ")
                digest.append(f"\n## {nombre_tema}\n")

                for p in papers[:3]:
                    digest.append(f"### {p.titulo}")
                    digest.append(f"*{', '.join(p.autores[:2])}* - {p.fecha}")
                    digest.append(f"{p.abstract[:200]}...")
                    digest.append(f"[Link]({p.url})\n")

        return "\n".join(digest)

    def tendencias_actuales(self) -> str:
        """Analiza tendencias actuales basado en papers recientes"""

        tendencias = ["# Tendencias Actuales en IA\n"]
        tendencias.append(f"*Analisis: {datetime.now().strftime('%d/%m/%Y')}*\n")

        # Contar papers por tema
        conteos = {}
        for tema in TEMAS_INTERES.keys():
            papers = self.papers_recientes(tema, dias=30, max_results=20)
            conteos[tema] = len(papers)

        # Ordenar por cantidad
        temas_ordenados = sorted(conteos.items(), key=lambda x: x[1], reverse=True)

        tendencias.append("## Temas Mas Activos (ultimos 30 dias)\n")
        for tema, count in temas_ordenados[:5]:
            emoji = "🔥" if count > 10 else "📈" if count > 5 else "📊"
            tendencias.append(f"{emoji} **{tema.upper()}**: {count} papers")

        tendencias.append("\n## Palabras Clave Emergentes\n")
        tendencias.append("- Reasoning / Chain-of-Thought")
        tendencias.append("- Multi-agent systems")
        tendencias.append("- Efficient fine-tuning (LoRA, QLoRA)")
        tendencias.append("- Long context windows")
        tendencias.append("- Synthetic data generation")

        return "\n".join(tendencias)

    def novedades_labs(self) -> str:
        """Genera resumen de novedades de los principales labs"""

        novedades = ["# Novedades de Labs de IA\n"]

        for lab_id, lab in LABS_SEGUIR.items():
            novedades.append(f"\n## {lab['nombre']}")
            novedades.append(f"- Blog: {lab['blog']}")
            novedades.append(f"- GitHub: {lab['github']}")
            novedades.append(f"- Keywords: {', '.join(lab['keywords'])}")

        novedades.append("\n---")
        novedades.append("*Tip: Suscribete a los blogs y sigue sus repos en GitHub para estar al dia.*")

        return "\n".join(novedades)

    # =========================================================================
    # BUSQUEDA ESPECIALIZADA
    # =========================================================================

    def buscar_legal_ai(self, termino: str = None) -> List[Paper]:
        """Busqueda especializada en Legal AI"""
        base_query = "legal AI OR law OR judicial OR contract"
        if termino:
            base_query = f"{termino} AND ({base_query})"

        return self.buscar_arxiv(base_query, max_results=15,
                                 categorias=["cs.CL", "cs.AI", "cs.CY"])

    def buscar_legaltech_papers(self) -> str:
        """Papers recientes relevantes para legaltech"""

        papers = self.buscar_legal_ai()

        if not papers:
            return "No se encontraron papers recientes de Legal AI."

        resumen = ["# Papers Recientes: Legal AI / Legaltech\n"]

        for p in papers[:5]:
            resumen.append(f"## {p.titulo}")
            resumen.append(f"**Fecha:** {p.fecha}")
            resumen.append(f"**Autores:** {', '.join(p.autores[:3])}")
            resumen.append(f"\n{p.abstract[:300]}...")
            resumen.append(f"\n[Ver paper]({p.url})\n")
            resumen.append("---")

        return "\n".join(resumen)

    # =========================================================================
    # UTILIDADES
    # =========================================================================

    def explicar_concepto(self, concepto: str) -> str:
        """Explica un concepto de IA"""

        conceptos = {
            "llm": """## Large Language Models (LLMs)
Modelos de lenguaje con billones de parametros entrenados en grandes corpus de texto.
Capaces de generar texto, responder preguntas, resumir, traducir, y mas.
**Ejemplos:** GPT-4, Claude, Gemini, LLaMA, Mistral""",

            "rag": """## Retrieval Augmented Generation (RAG)
Tecnica que combina recuperacion de informacion con generacion de texto.
El modelo busca en una base de conocimiento antes de generar respuestas.
**Uso:** Chatbots con conocimiento actualizado, Q&A sobre documentos.""",

            "fine-tuning": """## Fine-tuning
Proceso de adaptar un modelo pre-entrenado a una tarea especifica.
Se entrena con datos especificos del dominio.
**Variantes:** Full fine-tuning, LoRA, QLoRA, PEFT""",

            "agent": """## AI Agents
Sistemas de IA que pueden planificar, usar herramientas y ejecutar tareas autonomamente.
Combinan razonamiento (LLM) con acciones (tools).
**Frameworks:** LangChain, AutoGPT, CrewAI""",

            "chain-of-thought": """## Chain-of-Thought (CoT)
Tecnica de prompting que hace que el modelo razone paso a paso.
Mejora significativamente el rendimiento en tareas de razonamiento.
**Variantes:** Zero-shot CoT, Few-shot CoT, Tree of Thought""",

            "rlhf": """## RLHF (Reinforcement Learning from Human Feedback)
Metodo para alinear LLMs con preferencias humanas.
Un modelo de recompensa aprende de comparaciones humanas.
**Usado en:** ChatGPT, Claude, Gemini"""
        }

        concepto_lower = concepto.lower().replace(" ", "-")
        if concepto_lower in conceptos:
            return conceptos[concepto_lower]

        # Buscar parcial
        for key, value in conceptos.items():
            if concepto_lower in key or key in concepto_lower:
                return value

        return f"""No tengo una explicacion guardada para "{concepto}".

Puedo buscarlo en arXiv si quieres. Usa: buscar_arxiv("{concepto}")"""

    def resumen_estado(self) -> str:
        """Resumen del estado de investigacion"""

        return f"""# Estado de Investigacion

**Ultimo chequeo arXiv:** {self.state.get('ultimo_chequeo_arxiv', 'Nunca')}
**Papers vistos:** {len(self.state.get('papers_vistos', []))}
**Favoritos guardados:** {len(self.state.get('papers_favoritos', []))}
**Busquedas recientes:** {len(self.state.get('busquedas_recientes', []))}

## Ultimas busquedas
""" + "\n".join([f"- {b['query']} ({b['resultados']} resultados)"
                 for b in self.state.get('busquedas_recientes', [])[-5:]])


# =============================================================================
# FUNCIONES DE CONVENIENCIA
# =============================================================================

_research = None

def get_research() -> IrisResearch:
    """Obtiene la instancia del modulo de investigacion"""
    global _research
    if _research is None:
        _research = IrisResearch()
    return _research


def buscar_papers(query: str, max_results: int = 10) -> List[Dict]:
    """Busca papers en arXiv"""
    research = get_research()
    return [asdict(p) for p in research.buscar_arxiv(query, max_results)]


def papers_recientes(tema: str = "llm", dias: int = 7) -> List[Dict]:
    """Papers recientes de un tema"""
    research = get_research()
    return [asdict(p) for p in research.papers_recientes(tema, dias)]


def digest_semanal() -> str:
    """Digest semanal de IA"""
    return get_research().digest_semanal()


def tendencias() -> str:
    """Tendencias actuales"""
    return get_research().tendencias_actuales()


def explicar(concepto: str) -> str:
    """Explica un concepto de IA"""
    return get_research().explicar_concepto(concepto)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    research = IrisResearch()

    print("=" * 60)
    print("IRIS RESEARCH - TEST")
    print("=" * 60)

    # Test busqueda
    print("\n--- Busqueda: 'legal AI' ---")
    papers = research.buscar_arxiv("legal AI", max_results=3)
    for p in papers:
        print(f"  - {p.titulo[:60]}... ({p.fecha})")

    # Test papers recientes
    print("\n--- Papers recientes: LLM ---")
    papers = research.papers_recientes("llm", dias=7, max_results=3)
    for p in papers:
        print(f"  - {p.titulo[:60]}... ({p.fecha})")

    # Test explicar concepto
    print("\n--- Explicar: RAG ---")
    print(research.explicar_concepto("rag"))

    # Test tendencias
    print("\n--- Labs de IA ---")
    print(research.novedades_labs()[:500])

    print("\n[OK] Tests completados")
