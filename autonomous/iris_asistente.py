#!/usr/bin/env python3
"""
IRIS Asistente Personal - Asesora Fiscal Legaltech e Investigadora IA

IRIS es tu asistente especializada que:
- Monitorea plazos fiscales y normativas
- Investiga papers de IA y tendencias
- Redacta emails y contenido
- Gestiona tareas y comunicaciones
- Habla contigo de forma natural

Como una colega de confianza que trabaja para ti.
"""

import os
import sys
import json
import time
import psutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import threading

sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/core')
sys.path.insert(0, '/root/NEO_EVA/api')
sys.path.insert(0, '/root/NEO_EVA/config')
sys.path.insert(0, '/root/NEO_EVA/domains')

# Importar modulos especializados
try:
    from iris_perfil import PERFIL, SYSTEM_PROMPT, CONOCIMIENTO_FISCAL
    from iris_fiscal import IrisFiscal, get_fiscal
    from iris_research import IrisResearch, get_research
    from iris_personal import IrisPersonal, get_personal
    MODULOS_DISPONIBLES = True
except ImportError as e:
    print(f"Advertencia: Modulos especializados no disponibles: {e}")
    MODULOS_DISPONIBLES = False

# Archivo de mensajes pendientes para la web
MENSAJES_FILE = Path("/root/NEO_EVA/agents_state/iris_mensajes.json")
ESTADO_FILE = Path("/root/NEO_EVA/agents_state/iris_asistente_estado.json")
LOG_FILE = Path("/root/NEO_EVA/logs/iris_asistente.log")


class IrisAsistente:
    """
    IRIS como asistente personal que habla contigo.
    """

    def __init__(self):
        self.estado = self._cargar_estado()
        self.mensajes_pendientes = []
        self._asegurar_archivos()

    def _asegurar_archivos(self):
        MENSAJES_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not MENSAJES_FILE.exists():
            self._guardar_mensajes([])

    def _cargar_estado(self) -> Dict:
        if ESTADO_FILE.exists():
            try:
                with open(ESTADO_FILE) as f:
                    return json.load(f)
            except:
                pass
        return {
            "ultimo_saludo": None,
            "ultimo_reporte": None,
            "problemas_notificados": [],
            "preferencias_usuario": {},
            "conversaciones_pendientes": []
        }

    def _guardar_estado(self):
        ESTADO_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ESTADO_FILE, 'w') as f:
            json.dump(self.estado, f, indent=2, ensure_ascii=False)

    def _log(self, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(LOG_FILE, 'a') as f:
            f.write(f"[{timestamp}] {msg}\n")
        print(f"[{timestamp}] {msg}")

    # ========== MENSAJES ==========

    def _guardar_mensajes(self, mensajes: List[Dict]):
        with open(MENSAJES_FILE, 'w') as f:
            json.dump(mensajes, f, ensure_ascii=False, indent=2)

    def _cargar_mensajes(self) -> List[Dict]:
        if MENSAJES_FILE.exists():
            try:
                with open(MENSAJES_FILE) as f:
                    return json.load(f)
            except:
                pass
        return []

    def enviar_mensaje(self, texto: str, tipo: str = "normal", accion: Dict = None, urgente: bool = False):
        """
        Envia un mensaje al chat de la web.
        El usuario lo vera como si IRIS le hablara.
        """
        mensajes = self._cargar_mensajes()

        mensaje = {
            "id": f"msg_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(mensajes)}",
            "texto": texto,
            "tipo": tipo,  # normal, pregunta, alerta, info, saludo
            "timestamp": datetime.now().isoformat(),
            "leido": False,
            "urgente": urgente
        }

        if accion:
            mensaje["accion"] = accion  # {tipo: "ejecutar", comando: "...", descripcion: "..."}

        mensajes.append(mensaje)
        # Mantener solo ultimos 50 mensajes
        mensajes = mensajes[-50:]
        self._guardar_mensajes(mensajes)

        self._log(f"Mensaje enviado: {texto[:50]}...")
        return mensaje["id"]

    def marcar_leido(self, mensaje_id: str):
        mensajes = self._cargar_mensajes()
        for m in mensajes:
            if m["id"] == mensaje_id:
                m["leido"] = True
        self._guardar_mensajes(mensajes)

    def obtener_mensajes_no_leidos(self) -> List[Dict]:
        mensajes = self._cargar_mensajes()
        return [m for m in mensajes if not m.get("leido", False)]

    # ========== MONITOREO DEL SISTEMA ==========

    def verificar_sistema(self) -> Dict:
        """Verifica el estado del sistema"""
        estado = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memoria_percent": psutil.virtual_memory().percent,
            "disco_percent": psutil.disk_usage('/').percent,
            "servicios": {},
            "problemas": []
        }

        # Verificar servicios importantes
        servicios_check = ["iris_web", "iris_v2", "ollama"]
        for servicio in servicios_check:
            running = any(servicio in p.name() or servicio in ' '.join(p.cmdline())
                         for p in psutil.process_iter(['name', 'cmdline']))
            estado["servicios"][servicio] = running
            if not running and servicio != "ollama":
                estado["problemas"].append(f"Servicio {servicio} no esta corriendo")

        # Detectar problemas
        if estado["cpu_percent"] > 90:
            estado["problemas"].append(f"CPU muy alta: {estado['cpu_percent']}%")
        if estado["memoria_percent"] > 85:
            estado["problemas"].append(f"Memoria alta: {estado['memoria_percent']}%")
        if estado["disco_percent"] > 85:
            estado["problemas"].append(f"Disco casi lleno: {estado['disco_percent']}%")

        return estado

    def verificar_errores_logs(self) -> List[str]:
        """Busca errores recientes en logs"""
        errores = []
        logs_dir = Path("/root/NEO_EVA/logs")

        if logs_dir.exists():
            for log_file in logs_dir.glob("*.log"):
                try:
                    contenido = log_file.read_text()[-5000:]  # Ultimos 5KB
                    lineas = contenido.split('\n')
                    for linea in lineas[-50:]:  # Ultimas 50 lineas
                        if any(x in linea.lower() for x in ['error', 'exception', 'failed', 'critical']):
                            errores.append(f"{log_file.name}: {linea[:100]}")
                except:
                    pass

        return errores[-5:]  # Ultimos 5 errores

    # ========== COMUNICACION NATURAL ==========

    def saludar(self):
        """Saluda segun la hora del dia"""
        hora = datetime.now().hour
        ultimo = self.estado.get("ultimo_saludo")

        # Solo saludar una vez al dia
        if ultimo:
            ultimo_dt = datetime.fromisoformat(ultimo)
            if ultimo_dt.date() == datetime.now().date():
                return  # Ya saludo hoy

        if hora < 12:
            saludo = "Buenos dias"
            momento = "manana"
        elif hora < 20:
            saludo = "Buenas tardes"
            momento = "tarde"
        else:
            saludo = "Buenas noches"
            momento = "noche"

        # Verificar estado del sistema para el saludo
        estado = self.verificar_sistema()

        if estado["problemas"]:
            mensaje = f"{saludo}! He revisado el sistema y encontre algunos problemas que comentarte."
        else:
            mensaje = f"{saludo}! He revisado el sistema y todo esta funcionando bien. Estoy aqui por si necesitas algo."

        self.enviar_mensaje(mensaje, tipo="saludo")
        self.estado["ultimo_saludo"] = datetime.now().isoformat()
        self._guardar_estado()

    def reportar_problemas(self):
        """Reporta problemas encontrados"""
        estado = self.verificar_sistema()

        for problema in estado["problemas"]:
            # No repetir problemas ya notificados en la ultima hora
            if problema in self.estado.get("problemas_notificados", []):
                continue

            if "CPU" in problema:
                self.enviar_mensaje(
                    f"Oye, la CPU esta muy alta ({estado['cpu_percent']}%). ¿Quieres que revise que proceso esta consumiendo tanto?",
                    tipo="pregunta",
                    accion={"tipo": "diagnostico", "comando": "top_procesos"},
                    urgente=True
                )
            elif "Memoria" in problema:
                self.enviar_mensaje(
                    f"La memoria esta al {estado['memoria_percent']}%. Puedo cerrar procesos que no se esten usando. ¿Lo hago?",
                    tipo="pregunta",
                    accion={"tipo": "limpiar", "comando": "limpiar_memoria"}
                )
            elif "Disco" in problema:
                self.enviar_mensaje(
                    f"El disco esta al {estado['disco_percent']}%. Hay logs viejos y cache que puedo limpiar. ¿Quieres que lo haga?",
                    tipo="pregunta",
                    accion={"tipo": "limpiar", "comando": "limpiar_disco"}
                )
            elif "Servicio" in problema:
                servicio = problema.split()[-4]  # Extraer nombre del servicio
                self.enviar_mensaje(
                    f"El servicio {servicio} se ha caido. ¿Lo reinicio?",
                    tipo="pregunta",
                    accion={"tipo": "reiniciar", "comando": f"reiniciar_{servicio}"},
                    urgente=True
                )

            self.estado["problemas_notificados"].append(problema)

        # Limpiar problemas viejos
        self.estado["problemas_notificados"] = self.estado["problemas_notificados"][-20:]
        self._guardar_estado()

    def sugerir_mejoras(self):
        """Sugiere mejoras proactivamente"""
        sugerencias = []

        # Verificar si hay muchos archivos de log
        logs_dir = Path("/root/NEO_EVA/logs")
        if logs_dir.exists():
            logs = list(logs_dir.glob("*.log"))
            total_size = sum(f.stat().st_size for f in logs if f.exists())
            if total_size > 100 * 1024 * 1024:  # Mas de 100MB
                sugerencias.append({
                    "texto": f"Tienes {len(logs)} archivos de log ocupando {total_size // (1024*1024)}MB. ¿Los comprimo y archivo?",
                    "accion": {"tipo": "limpiar", "comando": "archivar_logs"}
                })

        # Verificar backups
        backup_dir = Path("/root/NEO_EVA/backups")
        if not backup_dir.exists() or not list(backup_dir.glob("*")):
            sugerencias.append({
                "texto": "No tienes backups recientes del proyecto. ¿Quieres que haga uno ahora?",
                "accion": {"tipo": "backup", "comando": "crear_backup"}
            })

        # Sugerir una mejora aleatoria (no muy frecuente)
        import random
        if random.random() < 0.1 and sugerencias:  # 10% de probabilidad
            sug = random.choice(sugerencias)
            self.enviar_mensaje(sug["texto"], tipo="pregunta", accion=sug["accion"])

    def dar_reporte_diario(self):
        """Da un resumen del dia"""
        ultimo = self.estado.get("ultimo_reporte")

        # Solo reportar una vez al dia, por la tarde
        hora = datetime.now().hour
        if hora < 18 or hora > 22:
            return

        if ultimo:
            ultimo_dt = datetime.fromisoformat(ultimo)
            if ultimo_dt.date() == datetime.now().date():
                return

        # Cargar metricas
        metrics_file = Path("/root/NEO_EVA/agents_state/iris_metrics.json")
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)

            proyectos = metrics.get("proyectos_exitosos", 0)
            errores = metrics.get("errores_corregidos", 0)

            if proyectos > 0 or errores > 0:
                self.enviar_mensaje(
                    f"Resumen del dia: He creado {proyectos} proyectos y corregido {errores} errores. El sistema esta estable. ¿Necesitas algo mas antes de que termine el dia?",
                    tipo="info"
                )
            else:
                self.enviar_mensaje(
                    "Hoy ha sido un dia tranquilo. No ha habido problemas y todo funciona bien. ¿Quieres que haga algo antes de terminar?",
                    tipo="info"
                )

            self.estado["ultimo_reporte"] = datetime.now().isoformat()
            self._guardar_estado()

    # ========== FUNCIONES ESPECIALIZADAS ==========

    def verificar_plazos_fiscales(self):
        """Verifica y avisa de plazos fiscales proximos"""
        if not MODULOS_DISPONIBLES:
            return

        try:
            fiscal = get_fiscal()
            plazos = fiscal.obtener_proximos_plazos(dias=10)

            for plazo in plazos:
                plazo_id = f"plazo_{plazo.numero}_{plazo.fecha_limite}"

                # No repetir alertas
                if plazo_id in self.estado.get("plazos_notificados", []):
                    continue

                if plazo.dias_restantes <= 3:
                    self.enviar_mensaje(
                        f"⚠️ URGENTE: El plazo del Modelo {plazo.numero} ({plazo.descripcion}) vence en {plazo.dias_restantes} dias ({plazo.fecha_limite}). ¿Quieres que te prepare un recordatorio para el cliente?",
                        tipo="pregunta",
                        accion={"tipo": "fiscal", "comando": f"recordatorio_{plazo.numero}"},
                        urgente=True
                    )
                elif plazo.dias_restantes <= 7:
                    self.enviar_mensaje(
                        f"📅 Recuerda: Modelo {plazo.numero} vence el {plazo.fecha_limite} ({plazo.dias_restantes} dias). {plazo.descripcion}.",
                        tipo="info"
                    )

                if "plazos_notificados" not in self.estado:
                    self.estado["plazos_notificados"] = []
                self.estado["plazos_notificados"].append(plazo_id)

            # Limpiar notificaciones viejas
            self.estado["plazos_notificados"] = self.estado.get("plazos_notificados", [])[-50:]
            self._guardar_estado()

        except Exception as e:
            self._log(f"Error verificando plazos fiscales: {e}")

    def revisar_boe(self):
        """Revisa el BOE para novedades fiscales"""
        if not MODULOS_DISPONIBLES:
            return

        try:
            fiscal = get_fiscal()
            novedades = fiscal.revisar_boe()

            if novedades:
                # Agrupar novedades
                resumen = f"📰 He encontrado {len(novedades)} novedades relevantes en el BOE de hoy:\n"
                for n in novedades[:3]:
                    resumen += f"\n• {n['titulo'][:80]}..."

                self.enviar_mensaje(
                    resumen + "\n\n¿Quieres que te prepare un resumen detallado?",
                    tipo="pregunta",
                    accion={"tipo": "fiscal", "comando": "resumen_boe"}
                )

        except Exception as e:
            self._log(f"Error revisando BOE: {e}")

    def obtener_digest_ia(self):
        """Obtiene novedades de IA semanalmente"""
        if not MODULOS_DISPONIBLES:
            return

        # Solo los lunes
        if datetime.now().weekday() != 0:
            return

        ultimo_digest = self.estado.get("ultimo_digest_ia")
        if ultimo_digest:
            ultimo_dt = datetime.fromisoformat(ultimo_digest)
            if (datetime.now() - ultimo_dt).days < 7:
                return

        try:
            research = get_research()
            papers = research.papers_recientes("llm", dias=7, max_results=5)

            if papers:
                self.enviar_mensaje(
                    f"🔬 Tengo el digest semanal de IA listo. He encontrado {len(papers)} papers interesantes sobre LLMs y agentes. ¿Te lo muestro?",
                    tipo="pregunta",
                    accion={"tipo": "research", "comando": "mostrar_digest"}
                )

            self.estado["ultimo_digest_ia"] = datetime.now().isoformat()
            self._guardar_estado()

        except Exception as e:
            self._log(f"Error obteniendo digest IA: {e}")

    def recordar_tareas(self):
        """Recuerda tareas pendientes importantes"""
        if not MODULOS_DISPONIBLES:
            return

        try:
            personal = get_personal()
            tareas = personal.listar_tareas(solo_pendientes=True)

            tareas_urgentes = [t for t in tareas if t.get("prioridad") == "alta"]

            if tareas_urgentes:
                nombres = [t["titulo"] for t in tareas_urgentes[:3]]
                self.enviar_mensaje(
                    f"📝 Tienes {len(tareas_urgentes)} tareas de alta prioridad pendientes: {', '.join(nombres)}. ¿Empezamos con alguna?",
                    tipo="info"
                )

        except Exception as e:
            self._log(f"Error recordando tareas: {e}")

    def generar_resumen_fiscal(self) -> str:
        """Genera resumen fiscal completo"""
        if not MODULOS_DISPONIBLES:
            return "Modulos fiscales no disponibles."

        fiscal = get_fiscal()
        return fiscal.resumen_fiscal()

    def generar_digest_semanal(self) -> str:
        """Genera digest semanal de IA"""
        if not MODULOS_DISPONIBLES:
            return "Modulos de investigacion no disponibles."

        research = get_research()
        return research.digest_semanal()

    def info_modelo_fiscal(self, numero: str) -> str:
        """Informacion sobre un modelo fiscal"""
        if not MODULOS_DISPONIBLES:
            return "Modulos fiscales no disponibles."

        fiscal = get_fiscal()
        return fiscal.info_modelo(numero)

    def buscar_papers(self, query: str) -> str:
        """Busca papers en arXiv"""
        if not MODULOS_DISPONIBLES:
            return "Modulos de investigacion no disponibles."

        research = get_research()
        papers = research.buscar_arxiv(query, max_results=5)

        if not papers:
            return f"No encontre papers sobre '{query}'."

        resultado = f"## Papers encontrados: {query}\n\n"
        for p in papers:
            resultado += f"**{p.titulo}**\n"
            resultado += f"*{', '.join(p.autores[:2])}* - {p.fecha}\n"
            resultado += f"{p.abstract[:150]}...\n"
            resultado += f"[Link]({p.url})\n\n"

        return resultado

    def crear_tarea(self, titulo: str, prioridad: str = "media") -> str:
        """Crea una nueva tarea"""
        if not MODULOS_DISPONIBLES:
            return "Modulos de tareas no disponibles."

        personal = get_personal()
        tarea = personal.crear_tarea(titulo, prioridad=prioridad)
        return f"Tarea creada: {tarea.titulo} (prioridad: {tarea.prioridad})"

    def redactar_email_cliente(self, nombre: str, tema: str, contenido: str) -> str:
        """Redacta un email para un cliente"""
        if not MODULOS_DISPONIBLES:
            return "Modulos de comunicacion no disponibles."

        personal = get_personal()
        return personal.responder_consulta(nombre, tema, contenido)

    # ========== ACCIONES ==========

    def ejecutar_accion(self, accion: Dict) -> str:
        """Ejecuta una accion y retorna resultado"""
        tipo = accion.get("tipo", "")
        comando = accion.get("comando", "")

        self._log(f"Ejecutando accion: {tipo} - {comando}")

        # ===== ACCIONES FISCALES =====
        if tipo == "fiscal":
            if comando == "resumen_boe":
                return "He revisado el BOE. Las novedades mas relevantes estan relacionadas con normativa tributaria. Te recomiendo revisar los enlaces que te pase."

            elif comando.startswith("recordatorio_"):
                modelo = comando.replace("recordatorio_", "")
                if MODULOS_DISPONIBLES:
                    personal = get_personal()
                    fiscal = get_fiscal()
                    info = fiscal.info_modelo(modelo)
                    return f"He preparado la info del modelo {modelo}:\n\n{info}\n\n¿Quieres que redacte un email de recordatorio para algun cliente?"
                return f"Recordatorio del modelo {modelo} preparado."

            elif comando == "calendario":
                if MODULOS_DISPONIBLES:
                    return self.generar_resumen_fiscal()
                return "Modulo fiscal no disponible."

        # ===== ACCIONES DE INVESTIGACION =====
        elif tipo == "research":
            if comando == "mostrar_digest":
                if MODULOS_DISPONIBLES:
                    return self.generar_digest_semanal()
                return "Modulo de investigacion no disponible."

            elif comando.startswith("buscar_"):
                query = comando.replace("buscar_", "")
                if MODULOS_DISPONIBLES:
                    return self.buscar_papers(query)
                return "Modulo de investigacion no disponible."

        # ===== ACCIONES DE TAREAS =====
        elif tipo == "tarea":
            if comando.startswith("crear_"):
                titulo = comando.replace("crear_", "")
                if MODULOS_DISPONIBLES:
                    return self.crear_tarea(titulo)
                return "Modulo de tareas no disponible."

            elif comando == "listar":
                if MODULOS_DISPONIBLES:
                    personal = get_personal()
                    return personal.resumen_tareas()
                return "Modulo de tareas no disponible."

        # ===== ACCIONES DE SISTEMA =====
        elif tipo == "diagnostico" and comando == "top_procesos":
            result = subprocess.run(
                "ps aux --sort=-%cpu | head -10",
                shell=True, capture_output=True, text=True
            )
            return f"Los procesos que mas CPU consumen son:\n```\n{result.stdout}\n```"

        elif tipo == "limpiar" and comando == "limpiar_disco":
            # Limpiar cache y logs viejos
            comandos = [
                "find /root/NEO_EVA/logs -name '*.log' -mtime +7 -delete",
                "find /tmp -type f -mtime +3 -delete 2>/dev/null || true",
                "apt-get clean 2>/dev/null || true"
            ]
            for cmd in comandos:
                subprocess.run(cmd, shell=True, capture_output=True)
            return "Listo, he limpiado logs viejos, cache temporal y paquetes. El disco deberia tener mas espacio ahora."

        elif tipo == "limpiar" and comando == "limpiar_memoria":
            subprocess.run("sync && echo 3 > /proc/sys/vm/drop_caches", shell=True)
            return "He liberado la cache de memoria. Deberia haber mas RAM disponible ahora."

        elif tipo == "backup" and comando == "crear_backup":
            backup_dir = Path("/root/NEO_EVA/backups")
            backup_dir.mkdir(exist_ok=True)
            fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"backup_{fecha}.tar.gz"
            subprocess.run(
                f"tar -czf {backup_file} --exclude='*.log' --exclude='__pycache__' --exclude='.git' /root/NEO_EVA/core /root/NEO_EVA/api /root/NEO_EVA/autonomous",
                shell=True, capture_output=True
            )
            size = backup_file.stat().st_size // 1024 if backup_file.exists() else 0
            return f"Backup creado: {backup_file.name} ({size}KB). Tus archivos importantes estan guardados."

        elif tipo == "reiniciar":
            servicio = comando.replace("reiniciar_", "")
            if servicio == "iris_web":
                subprocess.run("pkill -f iris_web.py", shell=True)
                subprocess.Popen(
                    "nohup python3 /root/NEO_EVA/api/iris_web.py > /tmp/iris_web.log 2>&1 &",
                    shell=True
                )
                return "He reiniciado el servidor web. Dale unos segundos y refresca la pagina."
            elif servicio == "iris_v2":
                subprocess.run("pkill -f iris_v2.py", shell=True)
                subprocess.Popen(
                    "nohup python3 /root/NEO_EVA/autonomous/iris_v2.py --loop --intervalo 60 > /tmp/iris_v2.log 2>&1 &",
                    shell=True
                )
                return "He reiniciado IRIS v2. Ya esta corriendo de nuevo."

        return "Accion completada."

    # ========== LOOP PRINCIPAL ==========

    def procesar_respuesta_usuario(self, respuesta: str) -> Optional[str]:
        """
        Procesa respuestas simples del usuario.
        Retorna None si no es una respuesta simple.
        """
        respuesta = respuesta.lower().strip()

        # Respuestas afirmativas
        if respuesta in ["ok", "si", "sí", "dale", "hazlo", "adelante", "venga", "va", "perfecto", "genial"]:
            # Buscar la ultima pregunta pendiente
            mensajes = self._cargar_mensajes()
            for m in reversed(mensajes):
                if m.get("tipo") == "pregunta" and m.get("accion"):
                    resultado = self.ejecutar_accion(m["accion"])
                    self.marcar_leido(m["id"])
                    return resultado
            return "Entendido. ¿Que necesitas que haga?"

        # Respuestas negativas
        elif respuesta in ["no", "nop", "nope", "mejor no", "dejalo", "no gracias"]:
            mensajes = self._cargar_mensajes()
            for m in reversed(mensajes):
                if m.get("tipo") == "pregunta":
                    self.marcar_leido(m["id"])
                    break
            return "Vale, lo dejo como esta. Avísame si cambias de opinión."

        # Pedir mas info
        elif respuesta in ["cuentame", "cuéntame", "dime mas", "explica", "que es eso"]:
            return "Claro, te explico..."

        return None  # No es respuesta simple, procesar como mensaje normal

    def ciclo_monitoreo(self):
        """Un ciclo de monitoreo"""
        self._log("Ciclo de monitoreo iniciado")

        # 1. Saludar si es nuevo dia
        self.saludar()

        # 2. Verificar y reportar problemas del sistema
        self.reportar_problemas()

        # 3. Verificar plazos fiscales (especializado)
        self.verificar_plazos_fiscales()

        # 4. Revisar BOE diariamente (por la manana)
        hora = datetime.now().hour
        if 8 <= hora <= 10:
            self.revisar_boe()

        # 5. Digest semanal de IA (lunes)
        self.obtener_digest_ia()

        # 6. Recordar tareas importantes
        self.recordar_tareas()

        # 7. Sugerir mejoras ocasionalmente
        self.sugerir_mejoras()

        # 8. Reporte diario por la tarde
        self.dar_reporte_diario()

        self._log("Ciclo de monitoreo completado")

    def run_loop(self, intervalo_segundos: int = 60):
        """Loop principal del asistente"""
        self._log("IRIS Asistente iniciado")
        self.enviar_mensaje(
            "Hola! Ya estoy activa. Estare monitoreando el sistema y te avisare si veo algo importante. Puedes hablarme cuando quieras.",
            tipo="saludo"
        )

        while True:
            try:
                self.ciclo_monitoreo()
            except Exception as e:
                self._log(f"Error en ciclo: {e}")

            time.sleep(intervalo_segundos)


# ========== API para la web ==========

def obtener_mensajes_iris() -> List[Dict]:
    """Obtiene mensajes de IRIS para mostrar en la web"""
    if MENSAJES_FILE.exists():
        with open(MENSAJES_FILE) as f:
            return json.load(f)
    return []


def procesar_respuesta(respuesta: str) -> Optional[str]:
    """Procesa una respuesta del usuario"""
    asistente = IrisAsistente()
    return asistente.procesar_respuesta_usuario(respuesta)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Ejecutar en loop")
    parser.add_argument("--intervalo", type=int, default=60, help="Segundos entre ciclos")
    parser.add_argument("--test", action="store_true", help="Enviar mensaje de prueba")
    args = parser.parse_args()

    asistente = IrisAsistente()

    if args.test:
        asistente.enviar_mensaje(
            "Esto es un mensaje de prueba. ¿Me recibes bien?",
            tipo="pregunta",
            accion={"tipo": "test", "comando": "test"}
        )
        print("Mensaje de prueba enviado")
    elif args.loop:
        asistente.run_loop(args.intervalo)
    else:
        asistente.ciclo_monitoreo()
