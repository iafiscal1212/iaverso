#!/usr/bin/env python3
"""
IRIS Brain v2.0 - Cerebro Mejorado

Mejoras:
1. Soporte para Claude API (mejor calidad) + Ollama fallback
2. Conocimiento del codebase antes de crear
3. Memoria persistente (proyectos anteriores, rechazos)
4. Verificacion de dependencias
5. No usa APIs externas que requieran keys
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import hashlib

sys.path.insert(0, '/root/NEO_EVA')

# Intentar importar anthropic
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# Intentar importar requests para Ollama
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class IrisBrain:
    """
    Cerebro mejorado de IRIS con memoria y contexto.
    """

    MEMORY_FILE = Path("/root/NEO_EVA/agents_state/iris_memory.json")
    CODEBASE_CACHE = Path("/root/NEO_EVA/agents_state/iris_codebase_cache.json")

    # Librerias seguras que se pueden usar sin API keys
    SAFE_LIBRARIES = [
        "os", "sys", "json", "csv", "datetime", "time", "random", "math",
        "pathlib", "subprocess", "threading", "multiprocessing", "queue",
        "collections", "itertools", "functools", "re", "hashlib", "base64",
        "urllib", "http", "socket", "sqlite3", "logging", "argparse",
        "dataclasses", "typing", "abc", "contextlib", "tempfile", "shutil",
        "glob", "fnmatch", "stat", "io", "pickle", "copy", "pprint",
        # Librerias externas comunes que no requieren API keys
        "requests", "pandas", "numpy", "matplotlib", "seaborn", "scipy",
        "sklearn", "flask", "fastapi", "uvicorn", "pydantic", "aiohttp",
        "beautifulsoup4", "bs4", "lxml", "pillow", "PIL", "psutil",
        "watchdog", "schedule", "rich", "click", "typer", "pyyaml", "toml",
    ]

    # APIs que requieren keys - PROHIBIDAS
    FORBIDDEN_APIS = [
        "openai", "anthropic", "cohere", "huggingface_hub", "transformers",
        "google.generativeai", "vertexai", "azure", "aws", "boto3",
        "stripe", "twilio", "sendgrid", "firebase", "supabase",
    ]

    def __init__(self, use_claude: bool = True):
        self.use_claude = use_claude and HAS_ANTHROPIC
        self.claude_client = None
        self.memory = self._load_memory()
        self.codebase_context = ""

        if self.use_claude:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self.claude_client = anthropic.Anthropic(api_key=api_key)
                self._log("Usando Claude API (alta calidad)")
            else:
                self.use_claude = False
                self._log("ANTHROPIC_API_KEY no encontrada, usando Ollama")

        if not self.use_claude:
            self._log("Usando Ollama (calidad media)")

    def _log(self, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [Brain] {msg}")

    # =========== MEMORIA ===========

    def _load_memory(self) -> Dict:
        """Carga la memoria persistente"""
        if self.MEMORY_FILE.exists():
            try:
                with open(self.MEMORY_FILE) as f:
                    return json.load(f)
            except:
                pass
        return {
            "proyectos_creados": [],
            "proyectos_rechazados": [],
            "errores_frecuentes": [],
            "preferencias_usuario": {},
            "ultima_actividad": None
        }

    def _save_memory(self):
        """Guarda la memoria"""
        self.memory["ultima_actividad"] = datetime.now().isoformat()
        self.MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self.MEMORY_FILE, 'w') as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)

    def recordar_proyecto(self, nombre: str, descripcion: str, exito: bool, razon: str = ""):
        """Recuerda un proyecto para no repetir"""
        proyecto = {
            "nombre": nombre,
            "descripcion": descripcion,
            "fecha": datetime.now().isoformat(),
            "exito": exito,
            "razon": razon
        }
        if exito:
            self.memory["proyectos_creados"].append(proyecto)
        else:
            self.memory["proyectos_rechazados"].append(proyecto)
        self._save_memory()

    def recordar_error(self, error: str, solucion: str):
        """Recuerda errores y sus soluciones"""
        self.memory["errores_frecuentes"].append({
            "error": error[:200],
            "solucion": solucion[:500],
            "fecha": datetime.now().isoformat()
        })
        # Mantener solo ultimos 50 errores
        self.memory["errores_frecuentes"] = self.memory["errores_frecuentes"][-50:]
        self._save_memory()

    def obtener_proyectos_anteriores(self) -> str:
        """Retorna resumen de proyectos anteriores para evitar repeticion"""
        creados = self.memory.get("proyectos_creados", [])[-10:]
        rechazados = self.memory.get("proyectos_rechazados", [])[-5:]

        resumen = ""
        if creados:
            resumen += "Proyectos que YA cree (NO repetir):\n"
            for p in creados:
                resumen += f"- {p['nombre']}: {p['descripcion']}\n"
        if rechazados:
            resumen += "\nProyectos RECHAZADOS por el usuario (evitar similares):\n"
            for p in rechazados:
                resumen += f"- {p['nombre']}: {p.get('razon', 'sin razon')}\n"
        return resumen

    # =========== CONOCIMIENTO DEL CODEBASE ===========

    def analizar_codebase(self, directorio: str = "/root/NEO_EVA") -> str:
        """Analiza el codebase para entender el proyecto"""
        self._log("Analizando codebase...")

        resumen = []
        archivos_importantes = []

        for ext in ["*.py", "*.json", "*.yaml", "*.yml"]:
            for archivo in Path(directorio).rglob(ext):
                # Ignorar directorios de cache y logs
                if any(x in str(archivo) for x in ["__pycache__", ".git", "node_modules", "logs"]):
                    continue

                try:
                    contenido = archivo.read_text()[:500]
                    rel_path = archivo.relative_to(directorio)

                    # Extraer docstring o comentario inicial
                    if '"""' in contenido:
                        doc = contenido.split('"""')[1].split('"""')[0][:100]
                    elif "'''" in contenido:
                        doc = contenido.split("'''")[1].split("'''")[0][:100]
                    elif contenido.startswith("#"):
                        doc = contenido.split("\n")[0][1:].strip()[:100]
                    else:
                        doc = ""

                    if doc:
                        archivos_importantes.append(f"{rel_path}: {doc}")
                except:
                    pass

        # Limitar a 30 archivos mas relevantes
        archivos_importantes = archivos_importantes[:30]

        self.codebase_context = "\n".join(archivos_importantes)
        self._log(f"   Analizados {len(archivos_importantes)} archivos")

        return self.codebase_context

    # =========== VERIFICACION DE CODIGO ===========

    def verificar_codigo(self, codigo: str) -> Tuple[bool, str]:
        """
        Verifica que el codigo sea seguro y no use APIs prohibidas.
        Retorna (es_valido, mensaje_error)
        """
        errores = []

        # Verificar imports prohibidos
        for api in self.FORBIDDEN_APIS:
            if f"import {api}" in codigo or f"from {api}" in codigo:
                errores.append(f"Usa API prohibida que requiere key: {api}")

        # Verificar patrones peligrosos
        patrones_peligrosos = [
            ("API_KEY", "Contiene referencia a API_KEY"),
            ("SECRET", "Contiene referencia a SECRET"),
            ("password", "Contiene referencia a password hardcodeada"),
            ("rm -rf", "Contiene comando destructivo rm -rf"),
            ("sudo", "Usa sudo"),
            ("eval(", "Usa eval() - peligroso"),
            ("exec(", "Usa exec() - peligroso"),
            ("__import__", "Usa __import__ dinamico"),
        ]

        for patron, mensaje in patrones_peligrosos:
            if patron.lower() in codigo.lower():
                errores.append(mensaje)

        # Verificar que tenga estructura minima
        if "def " not in codigo and "class " not in codigo:
            if 'if __name__' not in codigo:
                errores.append("No tiene funciones/clases ni punto de entrada")

        if errores:
            return False, "; ".join(errores)
        return True, ""

    def verificar_dependencias(self, codigo: str) -> Tuple[bool, List[str]]:
        """
        Verifica que las dependencias esten instaladas.
        Retorna (todas_instaladas, lista_faltantes)
        """
        import re

        # Extraer imports
        imports = re.findall(r'^(?:from|import)\s+(\w+)', codigo, re.MULTILINE)
        imports = list(set(imports))

        faltantes = []
        for imp in imports:
            # Ignorar stdlib
            if imp in self.SAFE_LIBRARIES[:30]:  # Primeras 30 son stdlib
                continue

            # Verificar si esta instalado
            try:
                __import__(imp)
            except ImportError:
                faltantes.append(imp)

        return len(faltantes) == 0, faltantes

    # =========== PENSAMIENTO (Claude o Ollama) ===========

    def pensar(self, prompt: str, max_tokens: int = 2000, creativo: bool = False) -> Optional[str]:
        """
        Genera respuesta usando Claude API o Ollama como fallback.
        """
        if self.use_claude and self.claude_client:
            return self._pensar_claude(prompt, max_tokens, creativo)
        else:
            return self._pensar_ollama(prompt, max_tokens, creativo)

    def _pensar_claude(self, prompt: str, max_tokens: int, creativo: bool) -> Optional[str]:
        """Usa Claude API"""
        try:
            response = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                temperature=0.7 if creativo else 0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            self._log(f"Error Claude: {e}")
            # Fallback a Ollama
            return self._pensar_ollama(prompt, max_tokens, creativo)

    def _pensar_ollama(self, prompt: str, max_tokens: int, creativo: bool) -> Optional[str]:
        """Usa Ollama local"""
        if not HAS_REQUESTS:
            return None

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5-coder:7b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7 if creativo else 0.3,
                        "num_predict": max_tokens
                    }
                },
                timeout=120
            )
            if response.status_code == 200:
                return response.json().get("response", "")
        except Exception as e:
            self._log(f"Error Ollama: {e}")

        return None

    # =========== GENERACION DE PROYECTOS ===========

    def generar_idea_proyecto(self) -> Optional[Dict]:
        """
        Genera una idea de proyecto ORIGINAL basada en:
        - Conocimiento del codebase
        - Memoria de proyectos anteriores
        - Restricciones de seguridad
        """
        # Analizar codebase si no lo hemos hecho
        if not self.codebase_context:
            self.analizar_codebase()

        proyectos_anteriores = self.obtener_proyectos_anteriores()

        prompt = f"""Eres IRIS, una IA autonoma que crea herramientas utiles.

CONTEXTO DEL PROYECTO (NEO_EVA):
{self.codebase_context[:2000]}

{proyectos_anteriores}

REGLAS ESTRICTAS:
1. NO uses APIs que requieran keys (OpenAI, Anthropic, etc)
2. Solo usa librerias estandar de Python o muy comunes (requests, pandas, etc)
3. El proyecto debe ser UTIL y DIFERENTE a los anteriores
4. Debe poder ejecutarse sin argumentos para demostrar que funciona
5. Ideas: automatizacion, analisis de datos, utilidades CLI, monitores, scrapers, etc

Responde SOLO JSON valido:
{{
    "nombre": "nombre_snake_case",
    "descripcion": "que hace en una linea",
    "proposito": "por que es util y como ayuda",
    "tipo": "cli|web|script|monitor|utilidad",
    "dependencias": ["lista", "de", "imports"]
}}"""

        respuesta = self.pensar(prompt, max_tokens=500, creativo=True)

        if not respuesta:
            return None

        try:
            # Extraer JSON
            json_start = respuesta.find('{')
            json_end = respuesta.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                proyecto = json.loads(respuesta[json_start:json_end])

                # Verificar que no sea repetido
                nombres_anteriores = [p['nombre'].lower() for p in self.memory.get("proyectos_creados", [])]
                if proyecto['nombre'].lower() in nombres_anteriores:
                    self._log(f"   Proyecto repetido: {proyecto['nombre']}, generando otro...")
                    return None

                return proyecto
        except json.JSONDecodeError:
            pass

        return None

    def generar_codigo(self, proyecto: Dict) -> Optional[str]:
        """
        Genera codigo de alta calidad para un proyecto.
        """
        prompt = f"""Escribe codigo Python COMPLETO y FUNCIONAL para:

PROYECTO: {proyecto['nombre']}
DESCRIPCION: {proyecto['descripcion']}
TIPO: {proyecto.get('tipo', 'script')}

REGLAS OBLIGATORIAS:
1. Codigo completo, no fragmentos
2. Incluye if __name__ == "__main__": con demo funcional
3. Maneja errores con try/except
4. NO uses openai, anthropic ni APIs que requieran keys
5. Solo librerias estandar o comunes: {', '.join(self.SAFE_LIBRARIES[:20])}
6. Incluye docstrings en funciones principales
7. El codigo debe ejecutarse SIN argumentos y mostrar algo util

Responde SOLO con el codigo Python, sin explicaciones ni markdown."""

        codigo = self.pensar(prompt, max_tokens=3000, creativo=False)

        if not codigo:
            return None

        # Limpiar codigo
        codigo = codigo.strip()
        if codigo.startswith("```python"):
            codigo = codigo[9:]
        if codigo.startswith("```"):
            codigo = codigo[3:]
        if codigo.endswith("```"):
            codigo = codigo[:-3]
        codigo = codigo.strip()

        # Verificar codigo
        es_valido, error = self.verificar_codigo(codigo)
        if not es_valido:
            self._log(f"   Codigo invalido: {error}")
            # Intentar corregir
            codigo = self._corregir_codigo_automatico(codigo, error)
            if not codigo:
                return None

        return codigo

    def _corregir_codigo_automatico(self, codigo: str, error: str) -> Optional[str]:
        """Corrige codigo automaticamente basado en el error"""
        prompt = f"""El siguiente codigo tiene problemas: {error}

CODIGO:
```python
{codigo[:2000]}
```

Corrige el codigo para que:
1. NO use APIs que requieran keys (openai, anthropic, etc)
2. Use solo librerias estandar o comunes
3. Sea seguro y funcional

Responde SOLO con el codigo corregido, sin explicaciones."""

        return self.pensar(prompt, max_tokens=3000, creativo=False)


# Singleton
_brain_instance = None

def get_brain(use_claude: bool = True) -> IrisBrain:
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = IrisBrain(use_claude=use_claude)
    return _brain_instance


if __name__ == "__main__":
    brain = get_brain()

    print("\n=== Test de IrisBrain ===\n")

    # Test analisis codebase
    print("1. Analizando codebase...")
    ctx = brain.analizar_codebase()
    print(f"   Contexto: {len(ctx)} caracteres\n")

    # Test memoria
    print("2. Proyectos anteriores:")
    print(brain.obtener_proyectos_anteriores() or "   (ninguno)\n")

    # Test generacion de idea
    print("3. Generando idea de proyecto...")
    idea = brain.generar_idea_proyecto()
    if idea:
        print(f"   Idea: {idea['nombre']} - {idea['descripcion']}")
    else:
        print("   No se pudo generar idea")
