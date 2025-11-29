#!/usr/bin/env python3
"""
NEO_EVA Dual Worlds Runner
===========================
Ejecuta NEO y EVA en paralelo con BUS de comunicación.
Genera reporte consolidado y métricas.

100% endógeno. Sin hardcodeo.
"""
import os
import sys
import json
import time
import threading
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/root/NEO_EVA/tools')
from common import (
    load_hist, triplet, sigmas, acf_window, pca_v1,
    median_alpha, sha256_file
)

# ============================================================================
# RUTAS
# ============================================================================
NEO_HIST_YAML = "/root/NEOSYNT/state/neo_state.yaml"
EVA_HIST = "/root/EVASYNT/state/history.jsonl"
BUS_LOG = "/root/NEO_EVA/logs/bus.log"
RESULTS_DIR = "/root/NEO_EVA/results"

# ============================================================================
# CARGA DE DATOS
# ============================================================================

def load_neo_history() -> List[Dict]:
    """Carga histórico de NEO."""
    import yaml
    if not os.path.exists(NEO_HIST_YAML):
        return []
    try:
        with open(NEO_HIST_YAML, 'r') as f:
            state = yaml.safe_load(f)
        raw = state.get("autonomy", {}).get("history_intention", [])
        return [{"t": i, "I": {"S": v[0], "N": v[1], "C": v[2]}}
               for i, v in enumerate(raw) if len(v) == 3]
    except:
        return []

def load_eva_history() -> List[Dict]:
    """Carga histórico de EVA."""
    return load_hist(EVA_HIST)

def load_bus_log() -> List[Dict]:
    """Carga log del BUS."""
    if not os.path.exists(BUS_LOG):
        return []
    records = []
    with open(BUS_LOG, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except:
                    pass
    return records

# ============================================================================
# EJECUCIÓN PARALELA
# ============================================================================

def run_neo_cycles(cycles: int, status: Dict) -> None:
    """Ejecuta ciclos de NEO (via HTTP API)."""
    import urllib.request
    import urllib.error

    status["neo_started"] = True
    status["neo_cycles"] = 0

    for i in range(cycles):
        try:
            req = urllib.request.Request(
                'http://127.0.0.1:7777/cycle',
                method='POST'
            )
            response = urllib.request.urlopen(req, timeout=30)

            # Publicar al BUS
            sys.path.insert(0, '/root/NEOSYNT')
            from neo_bus_listener import publish_neo_state
            publish_neo_state(i)

            status["neo_cycles"] = i + 1
        except Exception as e:
            status["neo_error"] = str(e)
            break

    status["neo_finished"] = True

def run_eva_cycles(cycles: int, status: Dict) -> None:
    """Ejecuta ciclos de EVA."""
    status["eva_started"] = True
    status["eva_cycles"] = 0

    try:
        sys.path.insert(0, '/root/EVASYNT/core')
        from evasynt import main as eva_main
        eva_main(cycles=cycles, verbose=False)
        status["eva_cycles"] = cycles
    except Exception as e:
        status["eva_error"] = str(e)

    status["eva_finished"] = True

# ============================================================================
# MÉTRICAS
# ============================================================================

def compute_cross_metrics(neo_hist: List[Dict], eva_hist: List[Dict]) -> Dict:
    """Computa métricas cruzadas entre NEO y EVA."""
    if not neo_hist or not eva_hist:
        return {"error": "Insufficient data"}

    neo_S = triplet(neo_hist, "S")
    eva_S = triplet(eva_hist, "S")

    # Truncar a longitud común
    n = min(len(neo_S), len(eva_S))
    if n < 10:
        return {"error": "Insufficient overlap"}

    neo_S = np.array(neo_S[-n:])
    eva_S = np.array(eva_S[-n:])

    # Correlación
    corr = float(np.corrcoef(neo_S, eva_S)[0, 1])

    # Diferencia media
    diff_mean = float(np.mean(neo_S - eva_S))
    diff_std = float(np.std(neo_S - eva_S))

    # Sorpresa cruzada (diferencia normalizada)
    neo_std = float(np.std(neo_S)) or 1.0
    eva_std = float(np.std(eva_S)) or 1.0
    surprise_neo_eva = float(np.mean(np.abs(neo_S - eva_S)) / neo_std)
    surprise_eva_neo = float(np.mean(np.abs(eva_S - neo_S)) / eva_std)

    return {
        "n_overlap": n,
        "correlation": corr,
        "diff_mean": diff_mean,
        "diff_std": diff_std,
        "surprise_neo_eva": surprise_neo_eva,
        "surprise_eva_neo": surprise_eva_neo
    }

def detect_explore_windows(bus_log: List[Dict]) -> List[Dict]:
    """Detecta ventanas de exploración conjunta."""
    windows = []
    current_window = None

    for record in bus_log:
        msg = record.get("message", record)
        agent = msg.get("agent")
        epoch = msg.get("epoch", 0)
        conf = msg.get("proposal", {}).get("conf", 0)

        # Threshold endógeno: conf > p95 histórico
        confs = [r.get("message", r).get("proposal", {}).get("conf", 0)
                for r in bus_log[:bus_log.index(record)+1]]
        if confs:
            p95 = float(np.percentile([c for c in confs if c > 0], 95))
        else:
            p95 = 0.5

        if conf > p95:
            if current_window is None:
                current_window = {"start": epoch, "agent": agent}
        else:
            if current_window is not None:
                current_window["end"] = epoch
                current_window["duration"] = epoch - current_window["start"]
                windows.append(current_window)
                current_window = None

    if current_window:
        current_window["end"] = epoch
        current_window["duration"] = epoch - current_window["start"]
        windows.append(current_window)

    return windows

# ============================================================================
# PLOTS
# ============================================================================

def plot_dual_timeseries(neo_hist: List[Dict], eva_hist: List[Dict],
                         output_path: str) -> None:
    """Genera plot de series temporales de ambos mundos."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    components = ["S", "N", "C"]

    for i, comp in enumerate(components):
        # NEO
        ax = axes[i, 0]
        if neo_hist:
            series = triplet(neo_hist, comp)
            ax.plot(series, 'b-', alpha=0.7, linewidth=0.5)
            ax.set_ylabel(f"NEO {comp}")
            ax.set_title(f"NEO: {comp} (T={len(series)})")

        # EVA
        ax = axes[i, 1]
        if eva_hist:
            series = triplet(eva_hist, comp)
            ax.plot(series, 'r-', alpha=0.7, linewidth=0.5)
            ax.set_ylabel(f"EVA {comp}")
            ax.set_title(f"EVA: {comp} (T={len(series)})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_explore_windows(bus_log: List[Dict], windows: List[Dict],
                         output_path: str) -> None:
    """Genera plot de ventanas de exploración."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Extraer confianzas por agente
    neo_epochs = []
    neo_confs = []
    eva_epochs = []
    eva_confs = []

    for record in bus_log:
        msg = record.get("message", record)
        agent = msg.get("agent")
        epoch = msg.get("epoch", 0)
        conf = msg.get("proposal", {}).get("conf", 0)

        if agent == "NEO":
            neo_epochs.append(epoch)
            neo_confs.append(conf)
        elif agent == "EVA":
            eva_epochs.append(epoch)
            eva_confs.append(conf)

    if neo_epochs:
        ax.plot(neo_epochs, neo_confs, 'b-', alpha=0.7, label='NEO conf')
    if eva_epochs:
        ax.plot(eva_epochs, eva_confs, 'r-', alpha=0.7, label='EVA conf')

    # Marcar ventanas de exploración
    for w in windows:
        ax.axvspan(w["start"], w["end"], alpha=0.3, color='green',
                   label='Explore' if w == windows[0] else None)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Confidence (varexp v1)")
    ax.set_title("Ventanas de Exploración Conjunta")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# ============================================================================
# REPORTE
# ============================================================================

def generate_report(neo_hist: List[Dict], eva_hist: List[Dict],
                    bus_log: List[Dict], explore_windows: List[Dict],
                    cross_metrics: Dict, runtime: float) -> str:
    """Genera reporte markdown consolidado."""
    report = []
    report.append("# NEO↔EVA Dual Worlds Report")
    report.append(f"\nGenerado: {datetime.now().isoformat()}")
    report.append(f"Tiempo de ejecución: {runtime:.1f} segundos")

    report.append("\n## Resumen")
    report.append(f"- NEO: T = {len(neo_hist)} ciclos")
    report.append(f"- EVA: T = {len(eva_hist)} ciclos")
    report.append(f"- Mensajes en BUS: {len(bus_log)}")
    report.append(f"- Ventanas de exploración: {len(explore_windows)}")

    report.append("\n## Estadísticas por Mundo")

    for name, hist in [("NEO", neo_hist), ("EVA", eva_hist)]:
        if not hist:
            continue
        T = len(hist)
        sig = sigmas(hist)
        w = acf_window(triplet(hist, "S")) if T > 3 else 1
        v1, lam1, varexp1 = pca_v1(hist) if T > 3 else ([0,0,0], 0, 0)
        alpha = median_alpha(sig, T)

        report.append(f"\n### {name}")
        report.append(f"- T = {T}")
        report.append(f"- σ = [{sig[0]:.4f}, {sig[1]:.4f}, {sig[2]:.4f}]")
        report.append(f"- α = {alpha:.6f}")
        report.append(f"- w (ACF) = {w}")
        report.append(f"- v1 = [{v1[0]:.4f}, {v1[1]:.4f}, {v1[2]:.4f}]")
        report.append(f"- varexp(v1) = {varexp1:.4f}")

    report.append("\n## Métricas Cruzadas")
    if "error" not in cross_metrics:
        report.append(f"- Correlación S: {cross_metrics['correlation']:.4f}")
        report.append(f"- Diferencia media: {cross_metrics['diff_mean']:.4f}")
        report.append(f"- Diferencia std: {cross_metrics['diff_std']:.4f}")
        report.append(f"- Sorpresa NEO←EVA: {cross_metrics['surprise_neo_eva']:.4f}")
        report.append(f"- Sorpresa EVA←NEO: {cross_metrics['surprise_eva_neo']:.4f}")
    else:
        report.append(f"- Error: {cross_metrics['error']}")

    report.append("\n## Ventanas de Exploración")
    if explore_windows:
        report.append("| Start | End | Duration | Agent |")
        report.append("|-------|-----|----------|-------|")
        for w in explore_windows[:20]:  # Limitar a 20
            report.append(f"| {w['start']} | {w['end']} | {w['duration']} | {w.get('agent', 'N/A')} |")
    else:
        report.append("No se detectaron ventanas de exploración.")

    report.append("\n## Parámetros Endógenos")
    report.append("Todos los parámetros derivan de estadísticas históricas:")
    report.append("- α = median(σ) / √T")
    report.append("- w = primer lag con ACF < mediana envolvente")
    report.append("- λ = IQR(sorpresa) / √T")
    report.append("- η = CV(sorpresa) / (1 + CV(sorpresa))")
    report.append("- k = floor(T^{1/3})")

    report.append("\n## Archivos Generados")
    files = [
        f"{RESULTS_DIR}/dual_timeseries.png",
        f"{RESULTS_DIR}/explore_windows.png",
        f"{RESULTS_DIR}/dual_metrics.csv",
        f"{RESULTS_DIR}/dual_worlds_report.md"
    ]
    for f in files:
        if os.path.exists(f):
            sha = sha256_file(f)
            report.append(f"- `{f}`: {sha[:16]}...")

    return "\n".join(report)

# ============================================================================
# MAIN
# ============================================================================

def main(cycles: int = 500, neo_enabled: bool = True, eva_enabled: bool = True):
    """Ejecuta ambos mundos en paralelo."""
    print(f"=" * 70)
    print("NEO↔EVA DUAL WORLDS RUNNER")
    print(f"=" * 70)
    print(f"Ciclos: {cycles}")
    print(f"NEO: {'habilitado' if neo_enabled else 'deshabilitado'}")
    print(f"EVA: {'habilitado' if eva_enabled else 'deshabilitado'}")
    print()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    start_time = time.time()

    # Estado compartido
    status = {
        "neo_started": False, "neo_finished": False, "neo_cycles": 0,
        "eva_started": False, "eva_finished": False, "eva_cycles": 0
    }

    # Iniciar BUS en background
    print("Iniciando BUS...")
    bus_proc = subprocess.Popen(
        ['python3', '/root/NEO_EVA/bus.py'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2)

    threads = []

    # Iniciar NEO
    if neo_enabled:
        print("Iniciando NEO...")
        # Verificar que el servidor está corriendo
        try:
            import urllib.request
            req = urllib.request.Request('http://127.0.0.1:7777/health')
            urllib.request.urlopen(req, timeout=5)
        except:
            print("  Servidor NEO no disponible. Iniciando...")
            subprocess.Popen(
                ['python3', '/root/NEOSYNT/core/neosynt_server.py'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(3)

        neo_thread = threading.Thread(target=run_neo_cycles, args=(cycles, status))
        neo_thread.start()
        threads.append(neo_thread)

    # Iniciar EVA
    if eva_enabled:
        print("Iniciando EVA...")
        eva_thread = threading.Thread(target=run_eva_cycles, args=(cycles, status))
        eva_thread.start()
        threads.append(eva_thread)

    # Monitorear progreso
    print("\nProgreso:")
    while any(t.is_alive() for t in threads):
        neo_pct = (status["neo_cycles"] / cycles * 100) if cycles > 0 else 0
        eva_pct = (status["eva_cycles"] / cycles * 100) if cycles > 0 else 0
        print(f"\r  NEO: {status['neo_cycles']:4d}/{cycles} ({neo_pct:5.1f}%)  "
              f"EVA: {status['eva_cycles']:4d}/{cycles} ({eva_pct:5.1f}%)", end="")
        time.sleep(1)

    print("\n")

    # Esperar a que terminen
    for t in threads:
        t.join()

    # Detener BUS
    bus_proc.terminate()
    bus_proc.wait()

    runtime = time.time() - start_time
    print(f"Ejecución completada en {runtime:.1f} segundos")

    # Cargar datos
    print("\nCargando datos...")
    neo_hist = load_neo_history()
    eva_hist = load_eva_history()
    bus_log = load_bus_log()

    print(f"  NEO: {len(neo_hist)} registros")
    print(f"  EVA: {len(eva_hist)} registros")
    print(f"  BUS: {len(bus_log)} mensajes")

    # Métricas
    print("\nCalculando métricas...")
    cross_metrics = compute_cross_metrics(neo_hist, eva_hist)
    explore_windows = detect_explore_windows(bus_log)

    # Plots
    print("\nGenerando plots...")
    plot_dual_timeseries(neo_hist, eva_hist,
                         f"{RESULTS_DIR}/dual_timeseries.png")
    plot_explore_windows(bus_log, explore_windows,
                         f"{RESULTS_DIR}/explore_windows.png")

    # CSV
    print("Guardando CSV...")
    import csv
    with open(f"{RESULTS_DIR}/dual_metrics.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["world", "t", "S", "N", "C"])
        for h in neo_hist:
            writer.writerow(["NEO", h["t"], h["I"]["S"], h["I"]["N"], h["I"]["C"]])
        for h in eva_hist:
            writer.writerow(["EVA", h["t"], h["I"]["S"], h["I"]["N"], h["I"]["C"]])

    # Reporte
    print("Generando reporte...")
    report = generate_report(neo_hist, eva_hist, bus_log,
                             explore_windows, cross_metrics, runtime)
    report_path = f"{RESULTS_DIR}/dual_worlds_report.md"
    with open(report_path, 'w') as f:
        f.write(report)

    # Checksums
    print("\n" + "=" * 70)
    print("ARCHIVOS GENERADOS:")
    print("=" * 70)
    for f in [
        f"{RESULTS_DIR}/dual_timeseries.png",
        f"{RESULTS_DIR}/explore_windows.png",
        f"{RESULTS_DIR}/dual_metrics.csv",
        f"{RESULTS_DIR}/dual_worlds_report.md"
    ]:
        if os.path.exists(f):
            sha = sha256_file(f)
            print(f"  {f}")
            print(f"    SHA256: {sha}")

    print("\n" + "=" * 70)
    print("COMPLETADO")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=500)
    parser.add_argument("--no-neo", action="store_true")
    parser.add_argument("--no-eva", action="store_true")
    args = parser.parse_args()

    main(cycles=args.cycles,
         neo_enabled=not args.no_neo,
         eva_enabled=not args.no_eva)
