#!/usr/bin/env python3
"""
IRIS - Modulo Asistente Personal

Redaccion de emails, contenido, gestion de tareas y comunicacion.
"""

import os
import sys
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, '/root/NEO_EVA')

# =============================================================================
# CONFIGURACION
# =============================================================================

STATE_DIR = Path("/root/NEO_EVA/agents_state")
PERSONAL_STATE_FILE = STATE_DIR / "iris_personal_state.json"

# =============================================================================
# ESTRUCTURAS DE DATOS
# =============================================================================

@dataclass
class Tarea:
    id: str
    titulo: str
    descripcion: str
    prioridad: str  # "alta", "media", "baja"
    fecha_limite: Optional[str]
    categoria: str  # "fiscal", "cliente", "investigacion", "admin"
    completada: bool = False
    notas: str = ""


@dataclass
class Contacto:
    id: str
    nombre: str
    email: str
    empresa: Optional[str]
    tipo: str  # "cliente", "proveedor", "colega", "otro"
    notas: str = ""


@dataclass
class Borrador:
    id: str
    tipo: str  # "email", "informe", "post", "otro"
    titulo: str
    contenido: str
    destinatario: Optional[str]
    estado: str  # "borrador", "revision", "listo"
    timestamp: str


# =============================================================================
# PLANTILLAS DE COMUNICACION
# =============================================================================

PLANTILLAS_EMAIL = {
    "respuesta_consulta": """Estimado/a {nombre},

Gracias por tu consulta sobre {tema}.

{contenido}

Si necesitas mas informacion o tienes alguna duda adicional, no dudes en contactarme.

Un cordial saludo,
{firma}""",

    "recordatorio_plazo": """Estimado/a {nombre},

Te escribo para recordarte que el plazo para {obligacion} vence el {fecha}.

Es importante que {accion_requerida}.

Si necesitas ayuda con la documentacion o tienes alguna duda, estoy a tu disposicion.

Un cordial saludo,
{firma}""",

    "envio_informe": """Estimado/a {nombre},

Adjunto te envio el informe sobre {tema} que me solicitaste.

Resumen:
{resumen}

Quedo a tu disposicion para comentarlo cuando te venga bien.

Un cordial saludo,
{firma}""",

    "propuesta_servicios": """Estimado/a {nombre},

Siguiendo nuestra conversacion, te envio una propuesta de servicios para {servicio}.

{contenido}

Los honorarios propuestos son: {honorarios}

Quedo a la espera de tus comentarios.

Un cordial saludo,
{firma}""",

    "agradecimiento": """Estimado/a {nombre},

{contenido}

Gracias por confiar en mis servicios.

Un cordial saludo,
{firma}""",

    "seguimiento": """Estimado/a {nombre},

Me pongo en contacto contigo para hacer seguimiento de {tema}.

{contenido}

Quedamos a tu disposicion.

Un cordial saludo,
{firma}"""
}

PLANTILLAS_CONTENIDO = {
    "post_linkedin": """🔹 {titulo}

{intro}

{puntos_clave}

{conclusion}

#Fiscalidad #Legaltech #IA #{hashtags}""",

    "hilo_twitter": """1/ {titulo}

{intro}

🧵👇

2/ {punto1}

3/ {punto2}

4/ {punto3}

5/ {conclusion}

Si te ha sido util, RT y follow para mas contenido sobre fiscalidad, legaltech e IA.""",

    "newsletter": """# {titulo}

Hola,

{intro}

## Lo mas destacado esta semana

{contenido}

## Proximas fechas importantes

{fechas}

## Recurso recomendado

{recurso}

Hasta la proxima,
{firma}"""
}


# =============================================================================
# CLASE PRINCIPAL
# =============================================================================

class IrisPersonal:
    """Modulo de asistente personal de IRIS"""

    def __init__(self, firma: str = "Tu asesora fiscal"):
        self.firma = firma
        self.state = self._cargar_estado()

    def _cargar_estado(self) -> Dict:
        """Carga el estado persistente"""
        if PERSONAL_STATE_FILE.exists():
            with open(PERSONAL_STATE_FILE) as f:
                return json.load(f)
        return {
            "tareas": [],
            "contactos": [],
            "borradores": [],
            "plantillas_personalizadas": {},
            "preferencias": {
                "tono": "profesional_cercano",
                "idioma": "es",
                "firma_default": "Tu asesora fiscal"
            }
        }

    def _guardar_estado(self):
        """Guarda el estado"""
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(PERSONAL_STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    # =========================================================================
    # GESTION DE TAREAS
    # =========================================================================

    def crear_tarea(self, titulo: str, descripcion: str = "",
                    prioridad: str = "media", fecha_limite: str = None,
                    categoria: str = "admin") -> Tarea:
        """Crea una nueva tarea"""

        tarea = Tarea(
            id=f"tarea_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            titulo=titulo,
            descripcion=descripcion,
            prioridad=prioridad,
            fecha_limite=fecha_limite,
            categoria=categoria,
            completada=False
        )

        self.state["tareas"].append(asdict(tarea))
        self._guardar_estado()
        return tarea

    def listar_tareas(self, categoria: str = None,
                      solo_pendientes: bool = True) -> List[Dict]:
        """Lista tareas filtradas"""

        tareas = self.state.get("tareas", [])

        if solo_pendientes:
            tareas = [t for t in tareas if not t.get("completada")]

        if categoria:
            tareas = [t for t in tareas if t.get("categoria") == categoria]

        # Ordenar por prioridad y fecha
        prioridad_orden = {"alta": 0, "media": 1, "baja": 2}
        tareas.sort(key=lambda x: (
            prioridad_orden.get(x.get("prioridad", "media"), 1),
            x.get("fecha_limite") or "9999-99-99"
        ))

        return tareas

    def completar_tarea(self, tarea_id: str) -> bool:
        """Marca una tarea como completada"""
        for tarea in self.state.get("tareas", []):
            if tarea.get("id") == tarea_id:
                tarea["completada"] = True
                tarea["fecha_completada"] = datetime.now().isoformat()
                self._guardar_estado()
                return True
        return False

    def resumen_tareas(self) -> str:
        """Genera un resumen de tareas pendientes"""

        tareas = self.listar_tareas(solo_pendientes=True)

        if not tareas:
            return "No tienes tareas pendientes. Buen trabajo!"

        resumen = ["# Tareas Pendientes\n"]

        # Agrupar por prioridad
        por_prioridad = {"alta": [], "media": [], "baja": []}
        for t in tareas:
            prio = t.get("prioridad", "media")
            por_prioridad[prio].append(t)

        if por_prioridad["alta"]:
            resumen.append("## 🔴 Prioridad Alta")
            for t in por_prioridad["alta"]:
                fecha = f" (vence: {t['fecha_limite']})" if t.get('fecha_limite') else ""
                resumen.append(f"- [ ] **{t['titulo']}**{fecha}")
                if t.get('descripcion'):
                    resumen.append(f"      {t['descripcion'][:50]}...")

        if por_prioridad["media"]:
            resumen.append("\n## 🟡 Prioridad Media")
            for t in por_prioridad["media"]:
                fecha = f" (vence: {t['fecha_limite']})" if t.get('fecha_limite') else ""
                resumen.append(f"- [ ] {t['titulo']}{fecha}")

        if por_prioridad["baja"]:
            resumen.append("\n## 🟢 Prioridad Baja")
            for t in por_prioridad["baja"]:
                resumen.append(f"- [ ] {t['titulo']}")

        resumen.append(f"\n*Total: {len(tareas)} tareas pendientes*")
        return "\n".join(resumen)

    # =========================================================================
    # REDACCION DE EMAILS
    # =========================================================================

    def redactar_email(self, tipo: str, **kwargs) -> str:
        """Redacta un email usando una plantilla"""

        # Añadir firma por defecto
        if "firma" not in kwargs:
            kwargs["firma"] = self.firma

        plantilla = PLANTILLAS_EMAIL.get(tipo)
        if not plantilla:
            # Plantilla generica
            plantilla = """Estimado/a {nombre},

{contenido}

Un cordial saludo,
{firma}"""

        try:
            return plantilla.format(**kwargs)
        except KeyError as e:
            return f"Error: Falta el campo {e} para esta plantilla."

    def responder_consulta(self, nombre: str, tema: str, respuesta: str) -> str:
        """Genera respuesta a una consulta de cliente"""
        return self.redactar_email(
            "respuesta_consulta",
            nombre=nombre,
            tema=tema,
            contenido=respuesta
        )

    def recordatorio_fiscal(self, nombre: str, obligacion: str,
                            fecha: str, accion: str) -> str:
        """Genera un recordatorio de plazo fiscal"""
        return self.redactar_email(
            "recordatorio_plazo",
            nombre=nombre,
            obligacion=obligacion,
            fecha=fecha,
            accion_requerida=accion
        )

    def email_personalizado(self, destinatario: str, asunto: str,
                            cuerpo: str, tono: str = "formal") -> str:
        """Genera un email personalizado"""

        saludos = {
            "formal": f"Estimado/a {destinatario}",
            "cercano": f"Hola {destinatario}",
            "muy_formal": f"Distinguido/a Sr./Sra. {destinatario}"
        }

        despedidas = {
            "formal": "Un cordial saludo",
            "cercano": "Un abrazo",
            "muy_formal": "Atentamente"
        }

        return f"""{saludos.get(tono, saludos['formal'])},

{cuerpo}

{despedidas.get(tono, despedidas['formal'])},
{self.firma}"""

    # =========================================================================
    # CREACION DE CONTENIDO
    # =========================================================================

    def crear_post_linkedin(self, titulo: str, intro: str,
                            puntos: List[str], conclusion: str,
                            hashtags: List[str] = None) -> str:
        """Crea un post para LinkedIn"""

        puntos_formateados = "\n".join([f"✅ {p}" for p in puntos])

        if hashtags is None:
            hashtags = ["AsesorieFiscal"]

        hashtags_str = " #".join(hashtags)

        return PLANTILLAS_CONTENIDO["post_linkedin"].format(
            titulo=titulo,
            intro=intro,
            puntos_clave=puntos_formateados,
            conclusion=conclusion,
            hashtags=hashtags_str
        )

    def crear_hilo_twitter(self, titulo: str, intro: str,
                           puntos: List[str], conclusion: str) -> str:
        """Crea un hilo para Twitter/X"""

        # Asegurar al menos 3 puntos
        while len(puntos) < 3:
            puntos.append("")

        return PLANTILLAS_CONTENIDO["hilo_twitter"].format(
            titulo=titulo,
            intro=intro,
            punto1=puntos[0],
            punto2=puntos[1],
            punto3=puntos[2],
            conclusion=conclusion
        )

    def crear_newsletter(self, titulo: str, intro: str, contenido: str,
                         fechas: str = "", recurso: str = "") -> str:
        """Crea una newsletter"""

        return PLANTILLAS_CONTENIDO["newsletter"].format(
            titulo=titulo,
            intro=intro,
            contenido=contenido,
            fechas=fechas or "Sin fechas destacadas esta semana",
            recurso=recurso or "Proximamente...",
            firma=self.firma
        )

    # =========================================================================
    # GESTION DE BORRADORES
    # =========================================================================

    def guardar_borrador(self, tipo: str, titulo: str, contenido: str,
                         destinatario: str = None) -> Borrador:
        """Guarda un borrador para revision posterior"""

        borrador = Borrador(
            id=f"borrador_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            tipo=tipo,
            titulo=titulo,
            contenido=contenido,
            destinatario=destinatario,
            estado="borrador",
            timestamp=datetime.now().isoformat()
        )

        self.state["borradores"].append(asdict(borrador))
        self._guardar_estado()
        return borrador

    def listar_borradores(self, tipo: str = None) -> List[Dict]:
        """Lista borradores guardados"""
        borradores = self.state.get("borradores", [])
        if tipo:
            borradores = [b for b in borradores if b.get("tipo") == tipo]
        return borradores

    # =========================================================================
    # UTILIDADES
    # =========================================================================

    def resumir_texto(self, texto: str, max_palabras: int = 100) -> str:
        """Resume un texto largo"""
        palabras = texto.split()
        if len(palabras) <= max_palabras:
            return texto

        # Resumen simple: primeras oraciones hasta max_palabras
        resumen = []
        count = 0
        for palabra in palabras:
            resumen.append(palabra)
            count += 1
            if count >= max_palabras and palabra.endswith('.'):
                break

        return " ".join(resumen) + "..."

    def formatear_lista_puntos(self, items: List[str],
                               estilo: str = "bullet") -> str:
        """Formatea una lista con diferentes estilos"""

        estilos = {
            "bullet": "• ",
            "numero": lambda i: f"{i+1}. ",
            "check": "✅ ",
            "arrow": "→ ",
            "star": "⭐ "
        }

        resultado = []
        for i, item in enumerate(items):
            if estilo == "numero":
                prefijo = f"{i+1}. "
            else:
                prefijo = estilos.get(estilo, "• ")
            resultado.append(f"{prefijo}{item}")

        return "\n".join(resultado)

    def generar_agenda_dia(self) -> str:
        """Genera una agenda para hoy basada en tareas"""

        hoy = datetime.now().strftime("%Y-%m-%d")
        tareas_hoy = [
            t for t in self.listar_tareas()
            if t.get("fecha_limite") == hoy or t.get("prioridad") == "alta"
        ]

        agenda = [f"# Agenda para hoy ({datetime.now().strftime('%d/%m/%Y')})\n"]

        if tareas_hoy:
            agenda.append("## Tareas prioritarias")
            for t in tareas_hoy:
                emoji = "🔴" if t["prioridad"] == "alta" else "🟡"
                agenda.append(f"{emoji} {t['titulo']}")
        else:
            agenda.append("No hay tareas urgentes para hoy.")

        agenda.append("\n## Recordatorios")
        agenda.append("- Revisar emails pendientes")
        agenda.append("- Comprobar plazos fiscales proximos")
        agenda.append("- Revisar novedades BOE/arXiv")

        return "\n".join(agenda)


# =============================================================================
# FUNCIONES DE CONVENIENCIA
# =============================================================================

_personal = None

def get_personal() -> IrisPersonal:
    """Obtiene la instancia del modulo personal"""
    global _personal
    if _personal is None:
        _personal = IrisPersonal()
    return _personal


def crear_tarea(titulo: str, prioridad: str = "media", **kwargs) -> Dict:
    """Crea una tarea"""
    personal = get_personal()
    return asdict(personal.crear_tarea(titulo, prioridad=prioridad, **kwargs))


def listar_tareas(**kwargs) -> List[Dict]:
    """Lista tareas"""
    return get_personal().listar_tareas(**kwargs)


def resumen_tareas() -> str:
    """Resumen de tareas"""
    return get_personal().resumen_tareas()


def redactar_email(tipo: str, **kwargs) -> str:
    """Redacta un email"""
    return get_personal().redactar_email(tipo, **kwargs)


def crear_post_linkedin(**kwargs) -> str:
    """Crea post de LinkedIn"""
    return get_personal().crear_post_linkedin(**kwargs)


def agenda_hoy() -> str:
    """Agenda del dia"""
    return get_personal().generar_agenda_dia()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    personal = IrisPersonal(firma="Ana Garcia - Asesora Fiscal")

    print("=" * 60)
    print("IRIS PERSONAL - TEST")
    print("=" * 60)

    # Test crear tarea
    print("\n--- Crear tarea ---")
    tarea = personal.crear_tarea(
        "Revisar declaracion cliente X",
        prioridad="alta",
        fecha_limite="2025-01-15",
        categoria="cliente"
    )
    print(f"Tarea creada: {tarea.titulo}")

    # Test email
    print("\n--- Redactar email ---")
    email = personal.responder_consulta(
        nombre="Juan",
        tema="deduccion por vivienda",
        respuesta="Segun la normativa actual, puedes deducirte hasta un 15% de las cantidades pagadas..."
    )
    print(email[:200] + "...")

    # Test post LinkedIn
    print("\n--- Post LinkedIn ---")
    post = personal.crear_post_linkedin(
        titulo="5 cambios fiscales para 2025",
        intro="El nuevo ano trae novedades importantes para autonomos y pymes.",
        puntos=["Nuevo limite de facturacion electronica", "Cambios en el IVA", "Deducciones ampliadas"],
        conclusion="Preparate con antelacion!",
        hashtags=["Fiscalidad", "Autonomos", "2025"]
    )
    print(post[:300] + "...")

    # Test resumen tareas
    print("\n--- Resumen tareas ---")
    print(personal.resumen_tareas()[:300])

    # Test agenda
    print("\n--- Agenda del dia ---")
    print(personal.generar_agenda_dia())

    print("\n[OK] Tests completados")
