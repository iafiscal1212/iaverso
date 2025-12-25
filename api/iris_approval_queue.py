#!/usr/bin/env python3
"""
IRIS Approval Queue - Sistema de Cola de Aprobaciones

Gestiona acciones que requieren aprobacion del usuario antes de ejecutar.

Flujo:
1. IRIS crea PendingAction cuando detecta accion de riesgo medio/alto
2. Usuario ve la accion en la UI
3. Usuario aprueba/rechaza/modifica
4. Si aprueba, se ejecuta y se registra
"""

import json
import uuid
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union
from dataclasses import dataclass, asdict, field
from enum import Enum

import sys
sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/core')

from core.iris_executor import IrisExecutor, ExecutionResult, FileResult, RiskLevel


class ActionType(Enum):
    EXECUTE_PYTHON = "execute_python"
    EXECUTE_BASH = "execute_bash"
    WRITE_FILE = "write_file"
    DELETE_FILE = "delete_file"
    INSTALL_PACKAGE = "install_package"
    GIT_OPERATION = "git_operation"
    SYSTEM_CHANGE = "system_change"
    CREATE_PROJECT = "create_project"


class ActionStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class PendingAction:
    """Accion pendiente de aprobacion"""
    id: str
    action_type: str  # ActionType value
    description: str
    payload: dict
    risk_level: str  # RiskLevel value
    created_at: str
    expires_at: str
    status: str = "pending"
    context: dict = field(default_factory=dict)
    preview: str = ""
    result: dict = field(default_factory=dict)
    approved_at: str = ""
    rejected_at: str = ""
    rejection_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'PendingAction':
        return cls(**data)

    def is_expired(self) -> bool:
        """Verifica si la accion ha expirado"""
        expires = datetime.fromisoformat(self.expires_at)
        return datetime.now() > expires


class ApprovalQueue:
    """
    Cola de aprobaciones para IRIS.
    Persiste acciones en JSON.
    """

    QUEUE_FILE = Path("/root/NEO_EVA/agents_state/iris_pending_actions.json")
    DEFAULT_EXPIRY_HOURS = 24

    def __init__(self):
        self.executor = IrisExecutor()
        self._ensure_file()

    def _ensure_file(self):
        """Asegura que el archivo existe"""
        if not self.QUEUE_FILE.exists():
            self._save({
                "pending": [],
                "approved": [],
                "rejected": [],
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "version": "1.0"
                }
            })

    def _load(self) -> dict:
        """Carga el archivo de cola"""
        try:
            with open(self.QUEUE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {"pending": [], "approved": [], "rejected": []}

    def _save(self, data: dict):
        """Guarda el archivo de cola"""
        self.QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self.QUEUE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def create_action(
        self,
        action_type: ActionType,
        description: str,
        payload: dict,
        risk_level: RiskLevel = None,
        context: dict = None,
        preview: str = "",
        expiry_hours: int = None
    ) -> PendingAction:
        """
        Crea una nueva accion pendiente.

        Args:
            action_type: Tipo de accion
            description: Descripcion legible para el usuario
            payload: Datos de la accion (codigo, comando, etc)
            risk_level: Nivel de riesgo (auto-detectado si None)
            context: Contexto adicional (conversacion, etc)
            preview: Vista previa del resultado esperado
            expiry_hours: Horas hasta que expire (default: 24)
        """
        # Auto-detectar riesgo si no se especifica
        if risk_level is None:
            risk_level = self.executor.classify_risk(action_type.value, payload)

        expiry_hours = expiry_hours or self.DEFAULT_EXPIRY_HOURS
        now = datetime.now()

        action = PendingAction(
            id=f"act_{uuid.uuid4().hex[:12]}",
            action_type=action_type.value,
            description=description,
            payload=payload,
            risk_level=risk_level.value,
            created_at=now.isoformat(),
            expires_at=(now + timedelta(hours=expiry_hours)).isoformat(),
            context=context or {},
            preview=preview
        )

        return action

    def add_action(self, action: PendingAction) -> str:
        """
        Anade una accion a la cola de pendientes.
        Retorna el ID de la accion.
        """
        data = self._load()
        data["pending"].append(action.to_dict())
        self._save(data)
        return action.id

    def get_pending(self, include_expired: bool = False) -> List[PendingAction]:
        """Obtiene todas las acciones pendientes"""
        data = self._load()
        actions = []

        for item in data.get("pending", []):
            action = PendingAction.from_dict(item)
            if include_expired or not action.is_expired():
                actions.append(action)

        return actions

    def get_action(self, action_id: str) -> Optional[PendingAction]:
        """Obtiene una accion por ID"""
        data = self._load()

        # Buscar en pending
        for item in data.get("pending", []):
            if item["id"] == action_id:
                return PendingAction.from_dict(item)

        # Buscar en approved
        for item in data.get("approved", []):
            if item["id"] == action_id:
                return PendingAction.from_dict(item)

        # Buscar en rejected
        for item in data.get("rejected", []):
            if item["id"] == action_id:
                return PendingAction.from_dict(item)

        return None

    def approve(self, action_id: str) -> Union[ExecutionResult, FileResult, dict]:
        """
        Aprueba y ejecuta una accion.
        Retorna el resultado de la ejecucion.
        """
        data = self._load()

        # Buscar accion en pending
        action_dict = None
        action_index = None
        for i, item in enumerate(data.get("pending", [])):
            if item["id"] == action_id:
                action_dict = item
                action_index = i
                break

        if action_dict is None:
            return {"success": False, "error": f"Accion {action_id} no encontrada en pendientes"}

        action = PendingAction.from_dict(action_dict)

        # Verificar expiracion
        if action.is_expired():
            return {"success": False, "error": "Accion expirada"}

        # Ejecutar segun tipo
        result = self._execute_action(action)

        # Actualizar estado
        success = result.success if hasattr(result, 'success') else result.get('success', False)
        action.status = "executed" if success else "failed"
        action.approved_at = datetime.now().isoformat()
        action.result = result.to_dict() if hasattr(result, 'to_dict') else result

        # Mover a approved
        data["pending"].pop(action_index)
        data["approved"].append(action.to_dict())

        # Limitar historial
        data["approved"] = data["approved"][-100:]  # Ultimas 100

        self._save(data)

        return result

    def reject(self, action_id: str, reason: str = "") -> bool:
        """
        Rechaza una accion.
        """
        data = self._load()

        # Buscar accion en pending
        action_dict = None
        action_index = None
        for i, item in enumerate(data.get("pending", [])):
            if item["id"] == action_id:
                action_dict = item
                action_index = i
                break

        if action_dict is None:
            return False

        action = PendingAction.from_dict(action_dict)
        action.status = "rejected"
        action.rejected_at = datetime.now().isoformat()
        action.rejection_reason = reason

        # Mover a rejected
        data["pending"].pop(action_index)
        data["rejected"].append(action.to_dict())

        # Limitar historial
        data["rejected"] = data["rejected"][-50:]  # Ultimas 50

        self._save(data)

        return True

    def modify(self, action_id: str, modifications: dict) -> Optional[PendingAction]:
        """
        Modifica una accion pendiente.
        Permite cambiar payload, descripcion, etc.
        """
        data = self._load()

        for i, item in enumerate(data.get("pending", [])):
            if item["id"] == action_id:
                # Aplicar modificaciones permitidas
                if "payload" in modifications:
                    item["payload"].update(modifications["payload"])
                if "description" in modifications:
                    item["description"] = modifications["description"]
                if "preview" in modifications:
                    item["preview"] = modifications["preview"]

                # Recalcular riesgo si cambio el payload
                if "payload" in modifications:
                    new_risk = self.executor.classify_risk(
                        item["action_type"],
                        item["payload"]
                    )
                    item["risk_level"] = new_risk.value

                data["pending"][i] = item
                self._save(data)

                return PendingAction.from_dict(item)

        return None

    def _execute_action(self, action: PendingAction) -> Union[ExecutionResult, FileResult, dict]:
        """Ejecuta una accion segun su tipo"""
        payload = action.payload

        if action.action_type == ActionType.EXECUTE_PYTHON.value:
            return self.executor.execute_python(
                code=payload.get("code", ""),
                timeout=payload.get("timeout"),
                save_to=payload.get("save_to")
            )

        elif action.action_type == ActionType.EXECUTE_BASH.value:
            return self.executor.execute_bash(
                command=payload.get("command", ""),
                timeout=payload.get("timeout"),
                cwd=payload.get("cwd")
            )

        elif action.action_type == ActionType.WRITE_FILE.value:
            path = payload.get("path", "")
            content = payload.get("content", "")

            # Validar que tenemos path y content
            if not path or not content:
                return {
                    "success": False,
                    "error": f"Accion invalida: falta {'path' if not path else 'content'}. IRIS necesita generar codigo real."
                }

            # Validar que el path es un archivo, no directorio
            if not path.endswith(('.py', '.txt', '.json', '.yaml', '.yml', '.md', '.sh', '.js', '.ts')):
                return {
                    "success": False,
                    "error": f"Path debe ser un archivo con extension valida, no un directorio: {path}"
                }

            return self.executor.write_file(
                path=path,
                content=content
            )

        elif action.action_type == ActionType.DELETE_FILE.value:
            return self.executor.delete_file(
                path=payload.get("path", "")
            )

        elif action.action_type == ActionType.INSTALL_PACKAGE.value:
            package = payload.get("package", "")
            return self.executor.execute_bash(
                command=f"pip3 install {package}",
                timeout=300  # 5 minutos para instalacion
            )

        elif action.action_type == ActionType.GIT_OPERATION.value:
            return self.executor.execute_bash(
                command=payload.get("command", ""),
                cwd=payload.get("cwd", "/root/NEO_EVA")
            )

        else:
            return {"success": False, "error": f"Tipo de accion no soportado: {action.action_type}"}

    def approve_all(self, action_type: ActionType = None) -> List[dict]:
        """
        Aprueba todas las acciones pendientes (filtradas por tipo opcional).
        Retorna lista de resultados.
        """
        pending = self.get_pending()
        results = []

        for action in pending:
            if action_type is None or action.action_type == action_type.value:
                result = self.approve(action.id)
                results.append({
                    "action_id": action.id,
                    "description": action.description,
                    "result": result.to_dict() if hasattr(result, 'to_dict') else result
                })

        return results

    def cleanup_expired(self) -> int:
        """
        Limpia acciones expiradas.
        Retorna numero de acciones limpiadas.
        """
        data = self._load()
        original_count = len(data.get("pending", []))

        # Filtrar no expiradas
        data["pending"] = [
            item for item in data.get("pending", [])
            if not PendingAction.from_dict(item).is_expired()
        ]

        cleaned = original_count - len(data["pending"])

        if cleaned > 0:
            self._save(data)

        return cleaned

    def get_stats(self) -> dict:
        """Obtiene estadisticas de la cola"""
        data = self._load()

        return {
            "pending": len(data.get("pending", [])),
            "approved": len(data.get("approved", [])),
            "rejected": len(data.get("rejected", [])),
            "by_risk": self._count_by_risk(data.get("pending", [])),
            "by_type": self._count_by_type(data.get("pending", []))
        }

    def _count_by_risk(self, items: list) -> dict:
        counts = {}
        for item in items:
            risk = item.get("risk_level", "unknown")
            counts[risk] = counts.get(risk, 0) + 1
        return counts

    def _count_by_type(self, items: list) -> dict:
        counts = {}
        for item in items:
            atype = item.get("action_type", "unknown")
            counts[atype] = counts.get(atype, 0) + 1
        return counts


# Helpers para crear acciones comunes
def crear_accion_ejecutar_python(
    codigo: str,
    descripcion: str,
    context: dict = None,
    save_to: str = None
) -> PendingAction:
    """Helper para crear accion de ejecutar Python"""
    queue = ApprovalQueue()

    preview = f"```python\n{codigo[:500]}{'...' if len(codigo) > 500 else ''}\n```"

    return queue.create_action(
        action_type=ActionType.EXECUTE_PYTHON,
        description=descripcion,
        payload={
            "code": codigo,
            "save_to": save_to
        },
        context=context,
        preview=preview
    )


def crear_accion_ejecutar_bash(
    comando: str,
    descripcion: str,
    context: dict = None,
    cwd: str = None
) -> PendingAction:
    """Helper para crear accion de ejecutar Bash"""
    queue = ApprovalQueue()

    preview = f"```bash\n{comando}\n```"

    return queue.create_action(
        action_type=ActionType.EXECUTE_BASH,
        description=descripcion,
        payload={
            "command": comando,
            "cwd": cwd
        },
        context=context,
        preview=preview
    )


def crear_accion_escribir_archivo(
    ruta: str,
    contenido: str,
    descripcion: str,
    context: dict = None
) -> PendingAction:
    """Helper para crear accion de escribir archivo"""
    queue = ApprovalQueue()

    preview = f"Archivo: `{ruta}`\n\n```\n{contenido[:300]}{'...' if len(contenido) > 300 else ''}\n```"

    return queue.create_action(
        action_type=ActionType.WRITE_FILE,
        description=descripcion,
        payload={
            "path": ruta,
            "content": contenido
        },
        context=context,
        preview=preview
    )


# Instancia global
approval_queue = ApprovalQueue()


if __name__ == "__main__":
    # Test basico
    print("=== Test ApprovalQueue ===\n")

    queue = ApprovalQueue()

    # Crear accion de prueba
    print("1. Crear accion de prueba:")
    action = queue.create_action(
        action_type=ActionType.EXECUTE_PYTHON,
        description="Imprimir hola mundo",
        payload={"code": "print('Hola desde IRIS!')"},
        preview="```python\nprint('Hola desde IRIS!')\n```"
    )
    print(f"   ID: {action.id}")
    print(f"   Riesgo: {action.risk_level}")

    # Anadir a cola
    print("\n2. Anadir a cola:")
    queue.add_action(action)
    print(f"   Pendientes: {len(queue.get_pending())}")

    # Ver stats
    print("\n3. Estadisticas:")
    stats = queue.get_stats()
    print(f"   {stats}")

    # Aprobar
    print("\n4. Aprobar accion:")
    result = queue.approve(action.id)
    print(f"   Success: {result.success if hasattr(result, 'success') else result.get('success')}")
    print(f"   Output: {result.output if hasattr(result, 'output') else result.get('output', '')}")

    # Stats finales
    print("\n5. Stats finales:")
    print(f"   {queue.get_stats()}")
