#!/usr/bin/env python3
"""
IRIS - Modulo Fiscal/Legal

Monitoreo de BOE, AEAT, calendario fiscal y alertas.
"""

import os
import sys
import json
import re
import requests
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
FISCAL_STATE_FILE = STATE_DIR / "iris_fiscal_state.json"

# URLs de fuentes
URLS = {
    "boe_sumario": "https://www.boe.es/diario_boe/",
    "boe_api": "https://www.boe.es/datosabiertos/api/boe/sumario/",
    "aeat_novedades": "https://sede.agenciatributaria.gob.es",
}

# =============================================================================
# ESTRUCTURAS DE DATOS
# =============================================================================

@dataclass
class AlertaFiscal:
    id: str
    tipo: str  # "plazo", "normativa", "novedad"
    titulo: str
    descripcion: str
    fecha_limite: Optional[str]
    urgencia: str  # "baja", "media", "alta", "critica"
    fuente: str
    url: Optional[str]
    timestamp: str
    leida: bool = False


@dataclass
class ModeloFiscal:
    numero: str
    nombre: str
    descripcion: str
    periodicidad: str
    fecha_limite: str
    dias_restantes: int


# =============================================================================
# CALENDARIO FISCAL 2025
# =============================================================================

CALENDARIO_FISCAL_2025 = {
    # Enero
    "2025-01-20": [
        {"modelo": "111", "descripcion": "Retenciones IRPF trabajadores 4T"},
        {"modelo": "115", "descripcion": "Retenciones alquileres 4T"},
        {"modelo": "123", "descripcion": "Retenciones capital mobiliario 4T"},
    ],
    "2025-01-30": [
        {"modelo": "303", "descripcion": "IVA 4T 2024"},
        {"modelo": "390", "descripcion": "Resumen anual IVA 2024"},
        {"modelo": "130", "descripcion": "Pago fraccionado IRPF 4T"},
        {"modelo": "131", "descripcion": "Pago fraccionado IRPF simplificado 4T"},
    ],

    # Febrero
    "2025-02-28": [
        {"modelo": "347", "descripcion": "Declaracion operaciones >3.005,06 EUR"},
        {"modelo": "349", "descripcion": "Operaciones intracomunitarias 4T/anual"},
    ],

    # Marzo
    "2025-03-31": [
        {"modelo": "720", "descripcion": "Bienes y derechos en el extranjero"},
    ],

    # Abril
    "2025-04-02": [
        {"modelo": "Renta", "descripcion": "Inicio campana IRPF 2024"},
    ],
    "2025-04-21": [
        {"modelo": "111", "descripcion": "Retenciones IRPF trabajadores 1T"},
        {"modelo": "115", "descripcion": "Retenciones alquileres 1T"},
        {"modelo": "303", "descripcion": "IVA 1T"},
        {"modelo": "130", "descripcion": "Pago fraccionado IRPF 1T"},
    ],

    # Mayo
    "2025-05-05": [
        {"modelo": "Renta", "descripcion": "Fin plazo solicitud cita previa renta"},
    ],

    # Junio
    "2025-06-25": [
        {"modelo": "200", "descripcion": "Impuesto Sociedades pago a cuenta"},
    ],
    "2025-06-30": [
        {"modelo": "Renta", "descripcion": "Fin campana IRPF 2024"},
    ],

    # Julio
    "2025-07-21": [
        {"modelo": "111", "descripcion": "Retenciones IRPF trabajadores 2T"},
        {"modelo": "115", "descripcion": "Retenciones alquileres 2T"},
        {"modelo": "303", "descripcion": "IVA 2T"},
        {"modelo": "130", "descripcion": "Pago fraccionado IRPF 2T"},
    ],
    "2025-07-25": [
        {"modelo": "200", "descripcion": "Impuesto Sociedades 2024"},
    ],

    # Octubre
    "2025-10-20": [
        {"modelo": "111", "descripcion": "Retenciones IRPF trabajadores 3T"},
        {"modelo": "115", "descripcion": "Retenciones alquileres 3T"},
        {"modelo": "303", "descripcion": "IVA 3T"},
        {"modelo": "130", "descripcion": "Pago fraccionado IRPF 3T"},
        {"modelo": "202", "descripcion": "Pago fraccionado IS 2o periodo"},
    ],

    # Diciembre
    "2025-12-20": [
        {"modelo": "202", "descripcion": "Pago fraccionado IS 3er periodo"},
    ],
}


# =============================================================================
# CLASE PRINCIPAL
# =============================================================================

class IrisFiscal:
    """Modulo fiscal de IRIS"""

    def __init__(self):
        self.state = self._cargar_estado()

    def _cargar_estado(self) -> Dict:
        """Carga el estado persistente"""
        if FISCAL_STATE_FILE.exists():
            with open(FISCAL_STATE_FILE) as f:
                return json.load(f)
        return {
            "alertas": [],
            "boe_revisados": [],
            "ultimo_chequeo_calendario": None,
            "ultimo_chequeo_boe": None,
            "novedades_vistas": []
        }

    def _guardar_estado(self):
        """Guarda el estado"""
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(FISCAL_STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    # =========================================================================
    # CALENDARIO FISCAL
    # =========================================================================

    def obtener_proximos_plazos(self, dias: int = 30) -> List[ModeloFiscal]:
        """Obtiene los plazos fiscales en los proximos N dias"""
        hoy = datetime.now().date()
        limite = hoy + timedelta(days=dias)
        proximos = []

        for fecha_str, modelos in CALENDARIO_FISCAL_2025.items():
            fecha = datetime.strptime(fecha_str, "%Y-%m-%d").date()

            if hoy <= fecha <= limite:
                dias_restantes = (fecha - hoy).days

                for m in modelos:
                    proximos.append(ModeloFiscal(
                        numero=m["modelo"],
                        nombre=f"Modelo {m['modelo']}",
                        descripcion=m["descripcion"],
                        periodicidad="ver calendario",
                        fecha_limite=fecha_str,
                        dias_restantes=dias_restantes
                    ))

        # Ordenar por dias restantes
        proximos.sort(key=lambda x: x.dias_restantes)
        return proximos

    def generar_alerta_plazos(self) -> List[AlertaFiscal]:
        """Genera alertas para plazos proximos"""
        alertas = []
        proximos = self.obtener_proximos_plazos(dias=15)

        for plazo in proximos:
            # Determinar urgencia
            if plazo.dias_restantes <= 3:
                urgencia = "critica"
            elif plazo.dias_restantes <= 7:
                urgencia = "alta"
            elif plazo.dias_restantes <= 14:
                urgencia = "media"
            else:
                urgencia = "baja"

            alerta_id = f"plazo_{plazo.numero}_{plazo.fecha_limite}"

            # Solo crear si no existe
            if alerta_id not in [a.get('id') for a in self.state.get('alertas', [])]:
                alerta = AlertaFiscal(
                    id=alerta_id,
                    tipo="plazo",
                    titulo=f"Plazo: {plazo.nombre}",
                    descripcion=f"{plazo.descripcion}. Quedan {plazo.dias_restantes} dias.",
                    fecha_limite=plazo.fecha_limite,
                    urgencia=urgencia,
                    fuente="Calendario AEAT",
                    url="https://sede.agenciatributaria.gob.es/Sede/ayuda/calendario-contribuyente.html",
                    timestamp=datetime.now().isoformat(),
                    leida=False
                )
                alertas.append(alerta)
                self.state['alertas'].append(asdict(alerta))

        self.state['ultimo_chequeo_calendario'] = datetime.now().isoformat()
        self._guardar_estado()
        return alertas

    def resumen_calendario_mes(self, mes: int = None, ano: int = None) -> str:
        """Genera un resumen del calendario fiscal para un mes"""
        if mes is None:
            mes = datetime.now().month
        if ano is None:
            ano = datetime.now().year

        mes_str = f"{ano}-{mes:02d}"
        obligaciones = []

        for fecha_str, modelos in CALENDARIO_FISCAL_2025.items():
            if fecha_str.startswith(mes_str):
                for m in modelos:
                    obligaciones.append({
                        "fecha": fecha_str,
                        "modelo": m["modelo"],
                        "descripcion": m["descripcion"]
                    })

        if not obligaciones:
            return f"No hay obligaciones fiscales registradas para {mes_str}"

        # Formatear resumen
        meses_nombre = ["", "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                       "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]

        resumen = [f"## Calendario Fiscal - {meses_nombre[mes]} {ano}\n"]

        obligaciones.sort(key=lambda x: x['fecha'])
        for ob in obligaciones:
            fecha_obj = datetime.strptime(ob['fecha'], "%Y-%m-%d")
            dia = fecha_obj.day
            resumen.append(f"- **{dia}/{mes}**: Modelo {ob['modelo']} - {ob['descripcion']}")

        return "\n".join(resumen)

    # =========================================================================
    # BOE
    # =========================================================================

    def revisar_boe(self, fecha: str = None) -> List[Dict]:
        """Revisa el BOE de una fecha para novedades fiscales/legales"""
        if fecha is None:
            fecha = datetime.now().strftime("%Y%m%d")

        # Palabras clave fiscales/legales
        keywords = [
            "tributar", "fiscal", "impuesto", "IRPF", "IVA", "sociedades",
            "hacienda", "AEAT", "contribuyente", "declaracion", "renta",
            "deduccion", "exencion", "tipo impositivo", "base imponible",
            "factura", "retencion", "autonomo", "pyme", "empresa",
            "criptomoneda", "bitcoin", "blockchain", "inteligencia artificial",
            "digital", "electronico", "telematico", "legaltech"
        ]

        novedades = []

        try:
            # API del BOE
            url = f"{URLS['boe_api']}{fecha}"
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                data = response.json()

                # Buscar en las disposiciones
                for seccion in data.get('data', {}).get('sumario', {}).get('diario', []):
                    for item in seccion.get('items', []):
                        titulo = item.get('titulo', '').lower()
                        texto = item.get('texto', '').lower()

                        # Verificar si contiene palabras clave
                        for keyword in keywords:
                            if keyword.lower() in titulo or keyword.lower() in texto:
                                novedad = {
                                    "id": item.get('identificador', ''),
                                    "titulo": item.get('titulo', ''),
                                    "seccion": seccion.get('nombre', ''),
                                    "url": f"https://www.boe.es/diario_boe/txt.php?id={item.get('identificador', '')}",
                                    "fecha": fecha,
                                    "keyword": keyword
                                }
                                # Evitar duplicados
                                if novedad['id'] not in self.state.get('novedades_vistas', []):
                                    novedades.append(novedad)
                                    self.state['novedades_vistas'].append(novedad['id'])
                                break

        except Exception as e:
            print(f"Error revisando BOE: {e}")

        self.state['ultimo_chequeo_boe'] = datetime.now().isoformat()
        self.state['boe_revisados'].append(fecha)
        self._guardar_estado()

        return novedades

    def buscar_consultas_dgt(self, termino: str) -> str:
        """Genera una URL de busqueda en la base de datos de consultas de la DGT"""
        # La DGT no tiene API publica, pero podemos dar la URL de busqueda
        termino_encoded = termino.replace(" ", "+")
        url = f"https://petete.tributos.hacienda.gob.es/consultas?texto={termino_encoded}"

        return f"""Para buscar consultas vinculantes sobre "{termino}":

1. **Base de datos DGT**: {url}
2. **Aranzadi**: Buscar en la base de datos de consultas
3. **Lefebvre**: Base de datos fiscal

Tip: Las consultas vinculantes (V****-YY) son las mas relevantes para argumentar."""

    # =========================================================================
    # UTILIDADES FISCALES
    # =========================================================================

    def calcular_iva(self, base: float, tipo: str = "general") -> Dict:
        """Calcula IVA segun tipo"""
        tipos_iva = {
            "general": 0.21,
            "reducido": 0.10,
            "superreducido": 0.04
        }

        if tipo not in tipos_iva:
            tipo = "general"

        porcentaje = tipos_iva[tipo]
        iva = base * porcentaje
        total = base + iva

        return {
            "base": round(base, 2),
            "tipo": f"{int(porcentaje * 100)}%",
            "iva": round(iva, 2),
            "total": round(total, 2)
        }

    def calcular_retencion_irpf(self, importe: float, tipo: float = 0.15) -> Dict:
        """Calcula retencion IRPF"""
        retencion = importe * tipo
        neto = importe - retencion

        return {
            "bruto": round(importe, 2),
            "tipo_retencion": f"{int(tipo * 100)}%",
            "retencion": round(retencion, 2),
            "neto": round(neto, 2)
        }

    def info_modelo(self, numero: str) -> str:
        """Devuelve informacion sobre un modelo fiscal"""
        modelos_info = {
            "100": {
                "nombre": "Declaracion de la Renta (IRPF)",
                "quien": "Personas fisicas residentes",
                "cuando": "Abril-Junio del ano siguiente",
                "que": "Declaracion anual de ingresos, deducciones y resultado"
            },
            "111": {
                "nombre": "Retenciones e ingresos a cuenta IRPF",
                "quien": "Empresas y autonomos que pagan nominas o facturas",
                "cuando": "Trimestral (20 del mes siguiente al trimestre)",
                "que": "Retenciones practicadas a trabajadores, profesionales, etc."
            },
            "115": {
                "nombre": "Retenciones alquileres",
                "quien": "Arrendatarios de inmuebles urbanos",
                "cuando": "Trimestral",
                "que": "Retencion 19% sobre alquiler de locales/oficinas"
            },
            "130": {
                "nombre": "Pago fraccionado IRPF - Estimacion directa",
                "quien": "Autonomos en estimacion directa",
                "cuando": "Trimestral",
                "que": "20% del rendimiento neto del trimestre"
            },
            "131": {
                "nombre": "Pago fraccionado IRPF - Modulos",
                "quien": "Autonomos en estimacion objetiva (modulos)",
                "cuando": "Trimestral",
                "que": "Porcentaje segun modulos"
            },
            "200": {
                "nombre": "Impuesto sobre Sociedades",
                "quien": "Sociedades y entidades juridicas",
                "cuando": "25 dias tras 6 meses del cierre (julio para cierre diciembre)",
                "que": "Declaracion anual del IS"
            },
            "202": {
                "nombre": "Pago fraccionado IS",
                "quien": "Sociedades con facturacion >6M o que lo elijan",
                "cuando": "Abril, octubre, diciembre",
                "que": "Adelanto del IS"
            },
            "303": {
                "nombre": "IVA Autoliquidacion",
                "quien": "Todos los sujetos pasivos de IVA",
                "cuando": "Trimestral o mensual (grandes empresas)",
                "que": "IVA repercutido - IVA soportado deducible"
            },
            "347": {
                "nombre": "Operaciones con terceros",
                "quien": "Quienes superen 3.005,06 EUR con un cliente/proveedor",
                "cuando": "Febrero (operaciones del ano anterior)",
                "que": "Declaracion informativa de operaciones"
            },
            "349": {
                "nombre": "Operaciones intracomunitarias",
                "quien": "Quienes realicen entregas/adquisiciones intracomunitarias",
                "cuando": "Mensual o trimestral segun volumen",
                "que": "Informacion de operaciones con otros paises UE"
            },
            "390": {
                "nombre": "Resumen anual IVA",
                "quien": "Todos los que presentan 303",
                "cuando": "Enero (del ano anterior)",
                "que": "Resumen de todas las operaciones del ano"
            },
            "720": {
                "nombre": "Bienes en el extranjero",
                "quien": "Residentes con bienes >50.000 EUR en el extranjero",
                "cuando": "Marzo",
                "que": "Declaracion informativa de cuentas, valores, inmuebles"
            },
            "721": {
                "nombre": "Criptomonedas en el extranjero",
                "quien": "Residentes con criptos >50.000 EUR en exchanges extranjeros",
                "cuando": "Marzo (desde 2024)",
                "que": "Declaracion informativa de criptoactivos"
            }
        }

        info = modelos_info.get(numero)
        if not info:
            return f"No tengo informacion detallada del modelo {numero}. Consulta en la sede de la AEAT."

        return f"""## Modelo {numero}: {info['nombre']}

**Quien debe presentarlo:** {info['quien']}
**Cuando:** {info['cuando']}
**Que declara:** {info['que']}

Mas info: https://sede.agenciatributaria.gob.es/Sede/procedimientoini/G{numero}.shtml"""

    # =========================================================================
    # RESUMEN
    # =========================================================================

    def resumen_fiscal(self) -> str:
        """Genera un resumen del estado fiscal actual"""
        proximos = self.obtener_proximos_plazos(dias=30)
        alertas_activas = [a for a in self.state.get('alertas', []) if not a.get('leida')]

        resumen = ["# Resumen Fiscal\n"]
        resumen.append(f"*Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}*\n")

        # Alertas urgentes
        if alertas_activas:
            resumen.append("## Alertas Activas")
            for a in alertas_activas[:5]:
                emoji = "🔴" if a['urgencia'] == 'critica' else "🟡" if a['urgencia'] == 'alta' else "🟢"
                resumen.append(f"{emoji} **{a['titulo']}** - {a['descripcion']}")
            resumen.append("")

        # Proximos plazos
        if proximos:
            resumen.append("## Proximos Plazos (30 dias)")
            for p in proximos[:10]:
                emoji = "🚨" if p.dias_restantes <= 5 else "⏰" if p.dias_restantes <= 10 else "📅"
                resumen.append(f"{emoji} **{p.fecha_limite}** ({p.dias_restantes}d): {p.descripcion}")
            resumen.append("")

        # Calendario del mes actual
        resumen.append(self.resumen_calendario_mes())

        return "\n".join(resumen)


# =============================================================================
# FUNCIONES DE CONVENIENCIA
# =============================================================================

_fiscal = None

def get_fiscal() -> IrisFiscal:
    """Obtiene la instancia del modulo fiscal"""
    global _fiscal
    if _fiscal is None:
        _fiscal = IrisFiscal()
    return _fiscal


def proximos_plazos(dias: int = 30) -> List[Dict]:
    """Wrapper para obtener proximos plazos"""
    fiscal = get_fiscal()
    return [asdict(p) for p in fiscal.obtener_proximos_plazos(dias)]


def revisar_boe_hoy() -> List[Dict]:
    """Revisa el BOE de hoy"""
    return get_fiscal().revisar_boe()


def info_modelo(numero: str) -> str:
    """Informacion de un modelo fiscal"""
    return get_fiscal().info_modelo(numero)


def resumen_fiscal() -> str:
    """Resumen fiscal actual"""
    return get_fiscal().resumen_fiscal()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    fiscal = IrisFiscal()

    print("=" * 60)
    print("IRIS FISCAL - TEST")
    print("=" * 60)

    # Test calendario
    print("\n--- Proximos plazos (30 dias) ---")
    for p in fiscal.obtener_proximos_plazos(30):
        print(f"  {p.fecha_limite} ({p.dias_restantes}d): {p.descripcion}")

    # Test info modelo
    print("\n--- Info Modelo 303 ---")
    print(fiscal.info_modelo("303"))

    # Test resumen
    print("\n--- Resumen Fiscal ---")
    print(fiscal.resumen_fiscal())

    # Test calculo IVA
    print("\n--- Calculo IVA ---")
    print(fiscal.calcular_iva(1000, "general"))

    print("\n[OK] Tests completados")
