#!/usr/bin/env python3
"""
IRIS Dashboard - Panel de Metricas y Monitoreo

Muestra:
- Estadisticas de proyectos
- Historial de ejecuciones
- Estado del sistema
- Logs en tiempo real
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

sys.path.insert(0, '/root/NEO_EVA')


def cargar_metricas() -> Dict:
    """Carga todas las metricas de IRIS"""
    metrics = {
        "iris": {},
        "aprobaciones": {},
        "ejecuciones": [],
        "memoria": {},
        "errores_recientes": []
    }

    # Metricas de IRIS v2
    metrics_file = Path("/root/NEO_EVA/agents_state/iris_metrics.json")
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics["iris"] = json.load(f)

    # Memoria de IRIS
    memory_file = Path("/root/NEO_EVA/agents_state/iris_memory.json")
    if memory_file.exists():
        with open(memory_file) as f:
            mem = json.load(f)
            metrics["memoria"] = {
                "proyectos_creados": len(mem.get("proyectos_creados", [])),
                "proyectos_rechazados": len(mem.get("proyectos_rechazados", [])),
                "errores_recordados": len(mem.get("errores_frecuentes", [])),
                "ultimos_proyectos": mem.get("proyectos_creados", [])[-5:]
            }

    # Aprobaciones
    pending_file = Path("/root/NEO_EVA/agents_state/iris_pending_actions.json")
    if pending_file.exists():
        with open(pending_file) as f:
            data = json.load(f)
            metrics["aprobaciones"] = {
                "pendientes": len(data.get("pending", [])),
                "aprobadas_total": len(data.get("approved", [])),
                "rechazadas_total": len(data.get("rejected", []))
            }

    # Ultimas ejecuciones
    log_file = Path("/root/NEO_EVA/agents_state/iris_execution_log.jsonl")
    if log_file.exists():
        with open(log_file) as f:
            lines = f.readlines()[-20:]  # Ultimas 20
            for line in lines:
                try:
                    metrics["ejecuciones"].append(json.loads(line))
                except:
                    pass

    return metrics


def generar_reporte() -> str:
    """Genera un reporte de texto"""
    m = cargar_metricas()

    reporte = []
    reporte.append("=" * 60)
    reporte.append("        IRIS v2.0 - REPORTE DE ESTADO")
    reporte.append("=" * 60)
    reporte.append(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Estadisticas generales
    iris = m.get("iris", {})
    reporte.append("ESTADISTICAS GENERALES")
    reporte.append("-" * 30)
    reporte.append(f"  Proyectos generados:  {iris.get('proyectos_generados', 0)}")
    reporte.append(f"  Proyectos exitosos:   {iris.get('proyectos_exitosos', 0)}")
    reporte.append(f"  Proyectos fallidos:   {iris.get('proyectos_fallidos', 0)}")
    reporte.append(f"  Errores corregidos:   {iris.get('errores_corregidos', 0)}")

    if iris.get('proyectos_exitosos', 0) > 0:
        tasa = (iris['proyectos_exitosos'] / max(iris.get('proyectos_generados', 1), 1)) * 100
        reporte.append(f"  Tasa de exito:        {tasa:.1f}%")

    tiempo = iris.get('tiempo_promedio_desarrollo', 0)
    reporte.append(f"  Tiempo promedio:      {tiempo:.1f}s")

    # Memoria
    mem = m.get("memoria", {})
    reporte.append(f"\nMEMORIA")
    reporte.append("-" * 30)
    reporte.append(f"  Proyectos recordados: {mem.get('proyectos_creados', 0)}")
    reporte.append(f"  Rechazos recordados:  {mem.get('proyectos_rechazados', 0)}")
    reporte.append(f"  Errores aprendidos:   {mem.get('errores_recordados', 0)}")

    # Aprobaciones
    apr = m.get("aprobaciones", {})
    reporte.append(f"\nAPROBACIONES")
    reporte.append("-" * 30)
    reporte.append(f"  Pendientes:           {apr.get('pendientes', 0)}")
    reporte.append(f"  Aprobadas (total):    {apr.get('aprobadas_total', 0)}")
    reporte.append(f"  Rechazadas (total):   {apr.get('rechazadas_total', 0)}")

    # Ultimos proyectos
    ultimos = mem.get("ultimos_proyectos", [])
    if ultimos:
        reporte.append(f"\nULTIMOS PROYECTOS CREADOS")
        reporte.append("-" * 30)
        for p in ultimos[-5:]:
            reporte.append(f"  - {p.get('nombre', '?')}: {p.get('descripcion', '')[:40]}")

    # Ultimas ejecuciones
    ejecuciones = m.get("ejecuciones", [])
    if ejecuciones:
        reporte.append(f"\nULTIMAS EJECUCIONES")
        reporte.append("-" * 30)
        for e in ejecuciones[-5:]:
            status = "OK" if e.get('success') else "ERR"
            tiempo = e.get('timestamp', '')[:19]
            tipo = e.get('type', '?')
            reporte.append(f"  [{status}] {tiempo} - {tipo}")

    reporte.append("\n" + "=" * 60)

    return "\n".join(reporte)


def generar_html() -> str:
    """Genera dashboard HTML"""
    m = cargar_metricas()
    iris = m.get("iris", {})
    mem = m.get("memoria", {})
    apr = m.get("aprobaciones", {})

    tasa_exito = 0
    if iris.get('proyectos_generados', 0) > 0:
        tasa_exito = (iris.get('proyectos_exitosos', 0) / iris['proyectos_generados']) * 100

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>IRIS Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ text-align: center; color: #60a5fa; margin-bottom: 30px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 20px; }}
        .card h3 {{ color: #94a3b8; font-size: 14px; margin-bottom: 10px; }}
        .card .value {{ font-size: 36px; font-weight: bold; color: #22c55e; }}
        .card .value.warning {{ color: #f59e0b; }}
        .card .value.error {{ color: #ef4444; }}
        .progress {{ background: #334155; border-radius: 10px; height: 20px; margin-top: 10px; overflow: hidden; }}
        .progress-bar {{ height: 100%; background: linear-gradient(90deg, #22c55e, #10b981); transition: width 0.5s; }}
        .list {{ margin-top: 20px; }}
        .list-item {{ background: #334155; padding: 10px; border-radius: 6px; margin-bottom: 8px; font-size: 14px; }}
        .list-item.success {{ border-left: 3px solid #22c55e; }}
        .list-item.error {{ border-left: 3px solid #ef4444; }}
        .timestamp {{ color: #64748b; font-size: 12px; }}
        .refresh {{ text-align: center; margin-top: 20px; color: #64748b; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>IRIS v2.0 Dashboard</h1>

        <div class="grid">
            <div class="card">
                <h3>PROYECTOS GENERADOS</h3>
                <div class="value">{iris.get('proyectos_generados', 0)}</div>
            </div>
            <div class="card">
                <h3>PROYECTOS EXITOSOS</h3>
                <div class="value">{iris.get('proyectos_exitosos', 0)}</div>
            </div>
            <div class="card">
                <h3>PROYECTOS FALLIDOS</h3>
                <div class="value {'error' if iris.get('proyectos_fallidos', 0) > 0 else ''}">{iris.get('proyectos_fallidos', 0)}</div>
            </div>
            <div class="card">
                <h3>ERRORES CORREGIDOS</h3>
                <div class="value">{iris.get('errores_corregidos', 0)}</div>
            </div>
        </div>

        <div class="grid" style="margin-top: 20px;">
            <div class="card">
                <h3>TASA DE EXITO</h3>
                <div class="value">{tasa_exito:.1f}%</div>
                <div class="progress">
                    <div class="progress-bar" style="width: {tasa_exito}%;"></div>
                </div>
            </div>
            <div class="card">
                <h3>TIEMPO PROMEDIO</h3>
                <div class="value">{iris.get('tiempo_promedio_desarrollo', 0):.1f}s</div>
            </div>
            <div class="card">
                <h3>PENDIENTES DE APROBAR</h3>
                <div class="value {'warning' if apr.get('pendientes', 0) > 0 else ''}">{apr.get('pendientes', 0)}</div>
            </div>
            <div class="card">
                <h3>PROYECTOS EN MEMORIA</h3>
                <div class="value">{mem.get('proyectos_creados', 0)}</div>
            </div>
        </div>

        <div class="card" style="margin-top: 20px;">
            <h3>ULTIMAS EJECUCIONES</h3>
            <div class="list">
"""

    for e in m.get("ejecuciones", [])[-10:]:
        status_class = "success" if e.get('success') else "error"
        status_icon = "✓" if e.get('success') else "✗"
        tiempo = e.get('timestamp', '')[:19]
        tipo = e.get('type', 'unknown')
        cmd = e.get('command', '')[:50]

        html += f"""
                <div class="list-item {status_class}">
                    <span>{status_icon}</span>
                    <span>{tipo}</span>
                    <span style="color:#64748b;">- {cmd}...</span>
                    <span class="timestamp">{tiempo}</span>
                </div>
"""

    html += """
            </div>
        </div>

        <div class="refresh">
            Actualizado: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
            <br><small>Refresca la pagina para actualizar</small>
        </div>
    </div>

    <script>
        // Auto-refresh cada 30 segundos
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>"""

    return html


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", action="store_true", help="Generar HTML")
    parser.add_argument("--json", action="store_true", help="Generar JSON")
    args = parser.parse_args()

    if args.html:
        print(generar_html())
    elif args.json:
        print(json.dumps(cargar_metricas(), indent=2, ensure_ascii=False))
    else:
        print(generar_reporte())
