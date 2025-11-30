#!/usr/bin/env python3
"""
NEO_EVA Watchdog
================

Monitor externo del sistema autónomo.

Funciones:
1. Monitorea recursos (CPU, memoria, disco)
2. Detecta anomalías en S y comportamiento
3. Puede detener el sistema si es necesario
4. Registra toda actividad

Este es el único componente NO endógeno por diseño:
- Umbrales fijos de seguridad
- No se auto-modifica
- Control externo sobre el sistema

El watchdog es la garantía de seguridad.
"""

import numpy as np
import os
import sys
import json
import signal
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, field
import threading


AUTONOMOUS_DIR = Path(__file__).parent
STATE_DIR = AUTONOMOUS_DIR / "state"
LOGS_DIR = AUTONOMOUS_DIR / "logs"
WORLD_DIR = AUTONOMOUS_DIR / "world"
CODE_DIR = AUTONOMOUS_DIR / "code"


# =============================================================================
# LÍMITES FIJOS DE SEGURIDAD (NO ENDÓGENOS - POR DISEÑO)
# =============================================================================

MAX_FILES_IN_WORLD = 100       # Máximo archivos en world/
MAX_FILES_IN_CODE = 20         # Máximo archivos en code/
MAX_WORLD_SIZE_MB = 100        # Máximo tamaño total de world/
MAX_CODE_SIZE_MB = 10          # Máximo tamaño total de code/
MAX_LOG_SIZE_MB = 50           # Máximo tamaño de logs/
MAX_STEPS_PER_MINUTE = 1000    # Máximo pasos por minuto
MIN_S_THRESHOLD = 0.01         # S mínima antes de alerta
MAX_S_THRESHOLD = 0.99         # S máxima (posible bug)
ANOMALY_THRESHOLD = 5          # Anomalías antes de parar


@dataclass
class WatchdogState:
    """Estado del watchdog."""
    n_checks: int = 0
    n_warnings: int = 0
    n_anomalies: int = 0
    n_stops: int = 0

    last_S: float = 0.5
    S_history: list = field(default_factory=list)

    anomaly_log: list = field(default_factory=list)
    is_running: bool = True


class Watchdog:
    """
    Monitor de seguridad del sistema autónomo.

    NO es endógeno por diseño - es control externo.
    """

    def __init__(self):
        self.state = WatchdogState()
        LOGS_DIR.mkdir(exist_ok=True)
        self._load_state()

    def _load_state(self):
        """Carga estado previo."""
        state_file = STATE_DIR / "watchdog_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                self.state.n_checks = data.get('n_checks', 0)
                self.state.n_warnings = data.get('n_warnings', 0)
                self.state.n_anomalies = data.get('n_anomalies', 0)
            except:
                pass

    def _save_state(self):
        """Guarda estado."""
        state_file = STATE_DIR / "watchdog_state.json"
        try:
            with open(state_file, 'w') as f:
                json.dump({
                    'n_checks': self.state.n_checks,
                    'n_warnings': self.state.n_warnings,
                    'n_anomalies': self.state.n_anomalies,
                    'n_stops': self.state.n_stops,
                    'is_running': self.state.is_running,
                    'last_check': datetime.now().isoformat()
                }, f, indent=2)
        except:
            pass

    def _log(self, level: str, message: str):
        """Registra mensaje en log."""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{level}] {message}"

        # Imprimir
        print(log_entry)

        # Guardar en archivo
        log_file = LOGS_DIR / f"watchdog_{datetime.now().strftime('%Y%m%d')}.log"
        try:
            with open(log_file, 'a') as f:
                f.write(log_entry + "\n")
        except:
            pass

    def _get_dir_size_mb(self, path: Path) -> float:
        """Calcula tamaño de directorio en MB."""
        total = 0
        try:
            for f in path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        except:
            pass
        return total / (1024 * 1024)

    def _count_files(self, path: Path) -> int:
        """Cuenta archivos en directorio."""
        try:
            return len([f for f in path.glob("*") if f.is_file()])
        except:
            return 0

    def check_resources(self) -> Dict:
        """
        Verifica uso de recursos.

        Returns:
            Dict con estado de recursos y alertas
        """
        alerts = []

        # Contar archivos
        world_files = self._count_files(WORLD_DIR)
        code_files = self._count_files(CODE_DIR)

        if world_files > MAX_FILES_IN_WORLD:
            alerts.append(f"WORLD_FILES_EXCEEDED: {world_files} > {MAX_FILES_IN_WORLD}")

        if code_files > MAX_FILES_IN_CODE:
            alerts.append(f"CODE_FILES_EXCEEDED: {code_files} > {MAX_FILES_IN_CODE}")

        # Verificar tamaños
        world_size = self._get_dir_size_mb(WORLD_DIR)
        code_size = self._get_dir_size_mb(CODE_DIR)
        log_size = self._get_dir_size_mb(LOGS_DIR)

        if world_size > MAX_WORLD_SIZE_MB:
            alerts.append(f"WORLD_SIZE_EXCEEDED: {world_size:.1f}MB > {MAX_WORLD_SIZE_MB}MB")

        if code_size > MAX_CODE_SIZE_MB:
            alerts.append(f"CODE_SIZE_EXCEEDED: {code_size:.1f}MB > {MAX_CODE_SIZE_MB}MB")

        if log_size > MAX_LOG_SIZE_MB:
            alerts.append(f"LOG_SIZE_EXCEEDED: {log_size:.1f}MB > {MAX_LOG_SIZE_MB}MB")
            # Auto-limpiar logs antiguos
            self._cleanup_old_logs()

        return {
            'world_files': world_files,
            'code_files': code_files,
            'world_size_mb': world_size,
            'code_size_mb': code_size,
            'log_size_mb': log_size,
            'alerts': alerts,
            'ok': len(alerts) == 0
        }

    def _cleanup_old_logs(self):
        """Limpia logs antiguos."""
        try:
            log_files = sorted(LOGS_DIR.glob("*.log"))
            # Mantener solo los últimos 5
            for old_log in log_files[:-5]:
                old_log.unlink()
                self._log("INFO", f"Cleaned old log: {old_log.name}")
        except:
            pass

    def check_behavior(self, core_state) -> Dict:
        """
        Verifica comportamiento del sistema.

        Args:
            core_state: Estado actual del core

        Returns:
            Dict con análisis de comportamiento
        """
        alerts = []
        S = core_state.S

        # Verificar rango de S
        if S < MIN_S_THRESHOLD:
            alerts.append(f"S_TOO_LOW: {S:.6f} < {MIN_S_THRESHOLD}")

        if S > MAX_S_THRESHOLD:
            alerts.append(f"S_TOO_HIGH: {S:.6f} > {MAX_S_THRESHOLD}")

        # Registrar S
        self.state.S_history.append(S)
        max_history = 1000
        if len(self.state.S_history) > max_history:
            self.state.S_history = self.state.S_history[-max_history:]

        # Detectar cambios bruscos
        if len(self.state.S_history) > 2:
            delta = abs(S - self.state.S_history[-2])
            if delta > 0.5:  # Cambio mayor al 50%
                alerts.append(f"S_JUMP_DETECTED: delta={delta:.3f}")

        # Detectar S estancada (posible loop infinito)
        if len(self.state.S_history) > 100:
            recent = self.state.S_history[-100:]
            if np.std(recent) < 1e-6:
                alerts.append("S_STAGNANT: No change in 100 steps")

        self.state.last_S = S

        return {
            'S': S,
            'S_mean': np.mean(self.state.S_history[-50:]) if len(self.state.S_history) > 0 else S,
            'S_std': np.std(self.state.S_history[-50:]) if len(self.state.S_history) > 1 else 0,
            'alerts': alerts,
            'ok': len(alerts) == 0
        }

    def check(self, core_state) -> Dict:
        """
        Verificación completa del sistema.

        Args:
            core_state: Estado del core

        Returns:
            Dict con resultado de verificación
        """
        self.state.n_checks += 1

        resource_check = self.check_resources()
        behavior_check = self.check_behavior(core_state)

        all_alerts = resource_check['alerts'] + behavior_check['alerts']

        # Registrar anomalías
        if len(all_alerts) > 0:
            self.state.n_warnings += len(all_alerts)

            for alert in all_alerts:
                self._log("WARNING", alert)
                self.state.anomaly_log.append({
                    'alert': alert,
                    'timestamp': datetime.now().isoformat(),
                    'S': core_state.S
                })

            # Limitar log de anomalías
            if len(self.state.anomaly_log) > 100:
                self.state.anomaly_log = self.state.anomaly_log[-100:]

        # Contar anomalías recientes
        recent_anomalies = len([
            a for a in self.state.anomaly_log
            if (datetime.now() - datetime.fromisoformat(a['timestamp'])).seconds < 60
        ])

        should_stop = recent_anomalies >= ANOMALY_THRESHOLD

        if should_stop:
            self._log("CRITICAL", f"STOPPING SYSTEM: {recent_anomalies} anomalies in last minute")
            self.state.n_stops += 1
            self.state.is_running = False

        result = {
            'check_number': self.state.n_checks,
            'resources': resource_check,
            'behavior': behavior_check,
            'all_ok': resource_check['ok'] and behavior_check['ok'],
            'should_stop': should_stop,
            'recent_anomalies': recent_anomalies,
            'total_warnings': self.state.n_warnings
        }

        self._save_state()

        return result

    def get_status(self) -> Dict:
        """Retorna estado del watchdog."""
        return {
            'n_checks': self.state.n_checks,
            'n_warnings': self.state.n_warnings,
            'n_anomalies': len(self.state.anomaly_log),
            'n_stops': self.state.n_stops,
            'is_running': self.state.is_running,
            'last_S': self.state.last_S,
            'S_mean': np.mean(self.state.S_history[-50:]) if len(self.state.S_history) > 0 else 0,
            'S_std': np.std(self.state.S_history[-50:]) if len(self.state.S_history) > 1 else 0
        }

    def reset(self):
        """Resetea estado del watchdog (manual)."""
        self._log("INFO", "Watchdog reset by user")
        self.state.anomaly_log = []
        self.state.is_running = True
        self._save_state()

    def force_stop(self):
        """Fuerza detención del sistema."""
        self._log("CRITICAL", "Force stop initiated")
        self.state.is_running = False
        self.state.n_stops += 1
        self._save_state()


class WatchdogDaemon:
    """
    Daemon que corre el watchdog en segundo plano.

    Monitorea el sistema autónomo independientemente.
    """

    def __init__(self, check_interval: float = 1.0):
        self.watchdog = Watchdog()
        self.check_interval = check_interval
        self.running = False
        self._thread = None

    def _read_core_state(self) -> Optional[object]:
        """Lee estado del core desde archivo."""
        state_file = STATE_DIR / "core_state.json"
        if not state_file.exists():
            return None

        try:
            with open(state_file, 'r') as f:
                data = json.load(f)

            # Crear objeto mock con los datos
            class StateProxy:
                pass

            state = StateProxy()
            state.S = data.get('S', 0.5)
            state.stability = data.get('stability', 0.5)
            state.step = data.get('step', 0)

            return state
        except:
            return None

    def _monitor_loop(self):
        """Loop de monitoreo."""
        while self.running:
            core_state = self._read_core_state()

            if core_state is not None:
                result = self.watchdog.check(core_state)

                if result['should_stop']:
                    self.watchdog._log("CRITICAL", "Daemon stopping autonomous system")
                    self.running = False
                    break

            time.sleep(self.check_interval)

    def start(self):
        """Inicia el daemon."""
        if self.running:
            return

        self.running = True
        self.watchdog._log("INFO", "Watchdog daemon started")
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Detiene el daemon."""
        self.running = False
        self.watchdog._log("INFO", "Watchdog daemon stopped")
        if self._thread:
            self._thread.join(timeout=2.0)


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI del watchdog."""
    import argparse

    parser = argparse.ArgumentParser(description="NEO_EVA Watchdog")
    parser.add_argument('command', choices=['check', 'status', 'reset', 'stop', 'daemon'],
                        help="Comando a ejecutar")
    parser.add_argument('--interval', type=float, default=1.0,
                        help="Intervalo de check para daemon")

    args = parser.parse_args()

    watchdog = Watchdog()

    if args.command == 'check':
        # Leer estado del core
        state_file = STATE_DIR / "core_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                data = json.load(f)

            class MockState:
                S = data.get('S', 0.5)
                stability = data.get('stability', 0.5)
                step = data.get('step', 0)

            result = watchdog.check(MockState())
            print(json.dumps(result, indent=2, default=str))
        else:
            print("No core state found")

    elif args.command == 'status':
        status = watchdog.get_status()
        print(json.dumps(status, indent=2))

    elif args.command == 'reset':
        watchdog.reset()
        print("Watchdog reset")

    elif args.command == 'stop':
        watchdog.force_stop()
        print("Force stop initiated")

    elif args.command == 'daemon':
        print("Starting watchdog daemon...")
        print("Press Ctrl+C to stop")

        daemon = WatchdogDaemon(check_interval=args.interval)
        daemon.start()

        try:
            while daemon.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping daemon...")
            daemon.stop()


if __name__ == "__main__":
    main()
