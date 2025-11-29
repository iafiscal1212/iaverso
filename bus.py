#!/usr/bin/env python3
"""
NEO_EVA BUS — Servicio de Intercambio Inter-Mundos
===================================================
UNIX socket para comunicación NEO ↔ EVA.
Solo resúmenes, sin datos crudos.
100% local, sin conexiones externas.
"""
import os
import sys
import json
import socket
import select
import threading
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
SOCK_PATH = "/run/neo_eva/bridge.sock"
BUS_LOG = "/root/NEO_EVA/logs/bus.log"
MAX_BUFFER = 1000  # Mensajes máximos en buffer (escala con uso)

# ============================================================================
# BUFFER DE MENSAJES
# ============================================================================
class MessageBuffer:
    """Buffer circular de mensajes por agente."""

    def __init__(self, maxlen: int = MAX_BUFFER):
        self.neo_messages: deque = deque(maxlen=maxlen)
        self.eva_messages: deque = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def add(self, msg: Dict) -> None:
        """Añade mensaje al buffer apropiado."""
        with self.lock:
            agent = msg.get("agent", "UNKNOWN")
            if agent == "NEO":
                self.neo_messages.append(msg)
            elif agent == "EVA":
                self.eva_messages.append(msg)

    def get_neo_messages(self, since_epoch: int = 0) -> List[Dict]:
        """Obtiene mensajes de NEO desde epoch."""
        with self.lock:
            return [m for m in self.neo_messages if m.get("epoch", 0) > since_epoch]

    def get_eva_messages(self, since_epoch: int = 0) -> List[Dict]:
        """Obtiene mensajes de EVA desde epoch."""
        with self.lock:
            return [m for m in self.eva_messages if m.get("epoch", 0) > since_epoch]

    def stats(self) -> Dict:
        """Estadísticas del buffer."""
        with self.lock:
            return {
                "neo_count": len(self.neo_messages),
                "eva_count": len(self.eva_messages),
                "neo_last_epoch": self.neo_messages[-1].get("epoch") if self.neo_messages else None,
                "eva_last_epoch": self.eva_messages[-1].get("epoch") if self.eva_messages else None
            }

# ============================================================================
# SERVIDOR BUS
# ============================================================================
class BusServer:
    """Servidor UNIX socket para el BUS."""

    def __init__(self):
        self.buffer = MessageBuffer()
        self.sock: Optional[socket.socket] = None
        self.running = False
        self.log_file = None

    def setup(self) -> bool:
        """Configura el socket."""
        # Crear directorio
        os.makedirs(os.path.dirname(SOCK_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(BUS_LOG), exist_ok=True)

        # Eliminar socket existente
        if os.path.exists(SOCK_PATH):
            os.unlink(SOCK_PATH)

        try:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self.sock.bind(SOCK_PATH)
            self.sock.setblocking(False)
            os.chmod(SOCK_PATH, 0o666)  # Permitir acceso

            self.log_file = open(BUS_LOG, 'a')
            return True
        except Exception as e:
            print(f"Error configurando BUS: {e}", file=sys.stderr)
            return False

    def log_message(self, msg: Dict) -> None:
        """Registra mensaje en log."""
        if self.log_file:
            record = {
                "timestamp": datetime.now().isoformat(),
                "message": msg
            }
            self.log_file.write(json.dumps(record) + '\n')
            self.log_file.flush()

    def validate_message(self, msg: Dict) -> bool:
        """Valida estructura del mensaje."""
        required = ["agent", "epoch"]
        return all(k in msg for k in required)

    def process_message(self, data: bytes) -> Optional[Dict]:
        """Procesa mensaje recibido."""
        try:
            msg = json.loads(data.decode('utf-8'))

            if not self.validate_message(msg):
                return None

            # Verificar checksum si existe
            if "checksum" in msg:
                payload = {k: v for k, v in msg.items() if k != "checksum"}
                expected = hashlib.sha256(
                    json.dumps(payload, sort_keys=True).encode()
                ).hexdigest()[:16]
                # No rechazar por checksum inválido, solo loggear
                if msg["checksum"] != expected:
                    msg["_checksum_valid"] = False

            self.buffer.add(msg)
            self.log_message(msg)
            return msg

        except json.JSONDecodeError:
            return None
        except Exception:
            return None

    def run(self, timeout: float = 0.1) -> None:
        """Bucle principal del servidor."""
        self.running = True
        print(f"BUS escuchando en {SOCK_PATH}")

        while self.running:
            try:
                ready, _, _ = select.select([self.sock], [], [], timeout)
                if ready:
                    try:
                        data, addr = self.sock.recvfrom(65536)
                        self.process_message(data)
                    except socket.error:
                        pass
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error en BUS: {e}", file=sys.stderr)
                time.sleep(0.1)

    def stop(self) -> None:
        """Detiene el servidor."""
        self.running = False
        if self.sock:
            self.sock.close()
        if self.log_file:
            self.log_file.close()
        if os.path.exists(SOCK_PATH):
            os.unlink(SOCK_PATH)

# ============================================================================
# CLIENTE BUS
# ============================================================================
class BusClient:
    """Cliente para enviar/recibir mensajes del BUS."""

    @staticmethod
    def send(msg: Dict) -> bool:
        """Envía mensaje al BUS."""
        if not os.path.exists(SOCK_PATH):
            return False

        try:
            # Añadir checksum
            payload = json.dumps(msg, sort_keys=True)
            msg["checksum"] = hashlib.sha256(payload.encode()).hexdigest()[:16]

            sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            sock.sendto(json.dumps(msg).encode('utf-8'), SOCK_PATH)
            sock.close()
            return True
        except Exception:
            return False

    @staticmethod
    def is_available() -> bool:
        """Verifica si el BUS está disponible."""
        return os.path.exists(SOCK_PATH)

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def create_neo_message(epoch: int, I: tuple, stats: Dict, proposal: Dict) -> Dict:
    """Crea mensaje estándar de NEO."""
    return {
        "agent": "NEO",
        "epoch": epoch,
        "acf_window": stats.get("w", 1),
        "stats": {
            "mu": stats.get("mu", I),
            "sigma": stats.get("sigma", {"S": 0, "N": 0, "C": 0}),
            "cov": stats.get("cov", [[0]*3]*3),
            "pca": stats.get("pca", {"v1": [1, 0, 0], "var1": 0, "varexp1": 0})
        },
        "proposal": proposal,
        "quantiles": stats.get("quantiles", {})
    }

def create_eva_message(epoch: int, I: tuple, stats: Dict, proposal: Dict) -> Dict:
    """Crea mensaje estándar de EVA."""
    return {
        "agent": "EVA",
        "epoch": epoch,
        "acf_window": stats.get("w", 1),
        "stats": {
            "mu": stats.get("mu", I),
            "sigma": stats.get("sigma", {"S": 0, "N": 0, "C": 0}),
            "cov": stats.get("cov", [[0]*3]*3),
            "pca": stats.get("pca", {"v1": [1, 0, 0], "var1": 0, "varexp1": 0})
        },
        "proposal": proposal,
        "quantiles": stats.get("quantiles", {})
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Inicia el servidor BUS."""
    server = BusServer()

    if not server.setup():
        sys.exit(1)

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        print("BUS detenido.")

if __name__ == "__main__":
    main()
