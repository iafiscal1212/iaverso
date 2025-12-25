#!/usr/bin/env python3
"""
IRIS v2.0 - Sistema Autonomo Mejorado

Mejoras sobre v1:
- Usa IrisBrain con Claude API + Ollama fallback
- Memoria persistente (no repite proyectos)
- Conocimiento del codebase
- Verificacion de codigo y dependencias
- Mejor manejo de errores
- Metricas de rendimiento
"""

import os
import sys
import json
import time
import subprocess
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/core')
sys.path.insert(0, '/root/NEO_EVA/api')

from core.iris_brain import get_brain, IrisBrain
from api.iris_approval_queue import ApprovalQueue, ActionType


class IrisV2:
    """
    IRIS v2 con autonomia total y mejoras.
    """

    WORKSPACE = Path("/root/NEO_EVA/iris_proyectos")
    LOG_FILE = Path("/root/NEO_EVA/logs/iris_v2.log")
    METRICS_FILE = Path("/root/NEO_EVA/agents_state/iris_metrics.json")

    MAX_INTENTOS = 5

    def __init__(self, use_claude: bool = True):
        self.brain = get_brain(use_claude=use_claude)
        self.queue = ApprovalQueue()
        self.WORKSPACE.mkdir(parents=True, exist_ok=True)
        self.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.metrics = self._load_metrics()

    def _load_metrics(self) -> Dict:
        if self.METRICS_FILE.exists():
            try:
                with open(self.METRICS_FILE) as f:
                    return json.load(f)
            except:
                pass
        return {
            "proyectos_generados": 0,
            "proyectos_exitosos": 0,
            "proyectos_fallidos": 0,
            "errores_corregidos": 0,
            "tiempo_promedio_desarrollo": 0,
            "ultimo_proyecto": None
        }

    def _save_metrics(self):
        self.METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self.METRICS_FILE, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def _log(self, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {msg}"
        print(log_msg)
        with open(self.LOG_FILE, 'a') as f:
            f.write(log_msg + '\n')

    def ejecutar_codigo(self, archivo: Path) -> Dict:
        """Ejecuta codigo y retorna resultado"""
        try:
            result = subprocess.run(
                ['python3', str(archivo)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(archivo.parent),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            )

            return {
                'exito': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {'exito': False, 'stderr': 'Timeout: mas de 30 segundos', 'stdout': ''}
        except Exception as e:
            return {'exito': False, 'stderr': str(e), 'stdout': ''}

    def instalar_dependencias(self, faltantes: List[str]) -> bool:
        """Instala dependencias faltantes"""
        for dep in faltantes:
            self._log(f"   Instalando {dep}...")
            try:
                result = subprocess.run(
                    ['pip3', 'install', dep, '-q'],
                    capture_output=True,
                    timeout=120
                )
                if result.returncode != 0:
                    self._log(f"   Error instalando {dep}")
                    return False
            except:
                return False
        return True

    def corregir_error(self, codigo: str, error: str) -> Optional[str]:
        """Corrige error en el codigo"""
        prompt = f"""El siguiente codigo Python tiene un error. Corrigelo.

ERROR:
{error[:1000]}

CODIGO:
```python
{codigo[:2500]}
```

REGLAS:
1. NO uses openai, anthropic ni APIs que requieran keys
2. Corrige solo el error, no cambies la funcionalidad
3. Responde SOLO con el codigo completo corregido

Codigo corregido:"""

        nuevo_codigo = self.brain.pensar(prompt, max_tokens=3000, creativo=False)

        if nuevo_codigo:
            # Limpiar
            nuevo_codigo = nuevo_codigo.strip()
            if nuevo_codigo.startswith("```python"):
                nuevo_codigo = nuevo_codigo[9:]
            if nuevo_codigo.startswith("```"):
                nuevo_codigo = nuevo_codigo[3:]
            if nuevo_codigo.endswith("```"):
                nuevo_codigo = nuevo_codigo[:-3]

            # Recordar error y solucion
            self.brain.recordar_error(error[:200], "Corregido automaticamente")
            self.metrics["errores_corregidos"] += 1

            return nuevo_codigo.strip()

        return None

    def ciclo_desarrollo(self, proyecto: Dict) -> Dict:
        """
        Ciclo completo: Generar -> Verificar -> Probar -> Corregir
        """
        inicio = time.time()

        # 1. Generar codigo
        self._log(f"   Generando codigo...")
        codigo = self.brain.generar_codigo(proyecto)

        if not codigo:
            return {'exito': False, 'razon': 'No pude generar codigo'}

        # 2. Verificar dependencias
        instaladas, faltantes = self.brain.verificar_dependencias(codigo)
        if not instaladas:
            self._log(f"   Dependencias faltantes: {faltantes}")
            if not self.instalar_dependencias(faltantes):
                return {'exito': False, 'razon': f'No pude instalar: {faltantes}'}

        # 3. Guardar y probar
        nombre_archivo = proyecto['nombre'].replace(' ', '_').lower()
        archivo = self.WORKSPACE / f"{nombre_archivo}.py"
        archivo.write_text(codigo, encoding='utf-8')
        self._log(f"   Guardado: {archivo}")

        # 4. Ciclo de prueba y correccion
        for intento in range(self.MAX_INTENTOS):
            self._log(f"   Probando (intento {intento + 1}/{self.MAX_INTENTOS})...")
            resultado = self.ejecutar_codigo(archivo)

            if resultado['exito']:
                duracion = time.time() - inicio

                # Actualizar metricas
                self.metrics["proyectos_exitosos"] += 1
                total = self.metrics["proyectos_exitosos"] + self.metrics["proyectos_fallidos"]
                self.metrics["tiempo_promedio_desarrollo"] = (
                    (self.metrics["tiempo_promedio_desarrollo"] * (total - 1) + duracion) / total
                )
                self._save_metrics()

                return {
                    'exito': True,
                    'archivo': str(archivo),
                    'codigo': codigo,
                    'salida': resultado['stdout'],
                    'intentos': intento + 1,
                    'duracion': round(duracion, 1)
                }

            # Corregir
            error = resultado['stderr'] or resultado['stdout']
            self._log(f"   Error: {error[:100]}...")

            codigo = self.corregir_error(codigo, error)
            if not codigo:
                break

            archivo.write_text(codigo, encoding='utf-8')

        self.metrics["proyectos_fallidos"] += 1
        self._save_metrics()

        return {'exito': False, 'razon': f'No pude corregir despues de {self.MAX_INTENTOS} intentos'}

    def ejecutar_ciclo_completo(self) -> Optional[str]:
        """
        Ciclo autonomo completo:
        1. Pensar proyecto (con contexto y memoria)
        2. Desarrollar (generar, probar, corregir)
        3. Solicitar despliegue
        """
        self._log("=" * 50)
        self._log("IRIS v2.0 - Nuevo ciclo autonomo")
        self._log("=" * 50)

        # 1. Pensar proyecto
        self._log("Pensando proyecto...")
        proyecto = self.brain.generar_idea_proyecto()

        if not proyecto:
            self._log("No pude generar una idea original")
            return None

        self._log(f"   Idea: {proyecto['nombre']} - {proyecto['descripcion']}")
        self.metrics["proyectos_generados"] += 1

        # 2. Desarrollar
        self._log("Desarrollando...")
        resultado = self.ciclo_desarrollo(proyecto)

        if not resultado['exito']:
            self._log(f"Desarrollo fallido: {resultado.get('razon', 'desconocido')}")
            self.brain.recordar_proyecto(
                proyecto['nombre'],
                proyecto['descripcion'],
                exito=False,
                razon=resultado.get('razon', '')
            )
            return None

        # 3. Solicitar despliegue
        self._log("Proyecto listo! Solicitando autorizacion...")

        explicacion = f"""## Proyecto Listo para Produccion

**Nombre:** {proyecto['nombre']}
**Descripcion:** {proyecto['descripcion']}
**Tipo:** {proyecto.get('tipo', 'script')}

### Por que lo cree
{proyecto.get('proposito', 'Para mejorar la productividad')}

### Desarrollo
- Intentos: {resultado.get('intentos', 1)}
- Duracion: {resultado.get('duracion', 0)}s
- Archivo: `{resultado['archivo']}`

### Prueba exitosa
```
{resultado.get('salida', 'Sin salida')[:500]}
```

### Codigo
```python
{resultado['codigo'][:1500]}{'...' if len(resultado['codigo']) > 1500 else ''}
```
"""

        action = self.queue.create_action(
            action_type=ActionType.CREATE_PROJECT,
            description=f"DESPLEGAR: {proyecto['nombre']} - {proyecto['descripcion']}",
            payload={
                'nombre': proyecto['nombre'],
                'archivo': resultado['archivo'],
                'codigo': resultado['codigo'],
                'tipo': 'despliegue_produccion'
            },
            preview=explicacion,
            context={
                'source': 'iris_v2',
                'proposito': proyecto.get('proposito', ''),
                'intentos': resultado.get('intentos', 1),
                'duracion': resultado.get('duracion', 0)
            }
        )
        self.queue.add_action(action)

        self.metrics["ultimo_proyecto"] = {
            "nombre": proyecto['nombre'],
            "fecha": datetime.now().isoformat(),
            "action_id": action.id
        }
        self._save_metrics()

        self._log(f"Esperando autorizacion: {action.id}")
        return action.id

    def run_loop(self, intervalo_minutos: int = 60):
        """Loop continuo"""
        self._log(f"IRIS v2.0 iniciada - Intervalo: {intervalo_minutos} min")
        self._log(f"Modelo: {'Claude API' if self.brain.use_claude else 'Ollama'}")

        while True:
            try:
                self.ejecutar_ciclo_completo()
            except Exception as e:
                self._log(f"Error en ciclo: {e}")
                traceback.print_exc()

            self._log(f"Esperando {intervalo_minutos} minutos...")
            time.sleep(intervalo_minutos * 60)

    def get_stats(self) -> Dict:
        """Retorna estadisticas"""
        return {
            **self.metrics,
            "memoria": {
                "proyectos_creados": len(self.brain.memory.get("proyectos_creados", [])),
                "proyectos_rechazados": len(self.brain.memory.get("proyectos_rechazados", [])),
                "errores_recordados": len(self.brain.memory.get("errores_frecuentes", []))
            },
            "modelo": "Claude API" if self.brain.use_claude else "Ollama"
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="IRIS v2.0")
    parser.add_argument("--loop", action="store_true", help="Ejecutar en loop")
    parser.add_argument("--intervalo", type=int, default=60, help="Minutos entre ciclos")
    parser.add_argument("--una-vez", action="store_true", help="Un solo ciclo")
    parser.add_argument("--stats", action="store_true", help="Ver estadisticas")
    parser.add_argument("--no-claude", action="store_true", help="No usar Claude API")

    args = parser.parse_args()

    iris = IrisV2(use_claude=not args.no_claude)

    if args.stats:
        stats = iris.get_stats()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    elif args.loop:
        iris.run_loop(args.intervalo)
    else:
        iris.ejecutar_ciclo_completo()


if __name__ == "__main__":
    main()
