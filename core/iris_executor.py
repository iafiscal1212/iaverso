#!/usr/bin/env python3
"""
IRIS Executor - Motor de Ejecucion Real

Capacidades:
- Ejecutar codigo Python con timeout
- Ejecutar comandos Bash
- Leer/escribir archivos
- Listar directorios
- Logging de todas las operaciones

Seguridad:
- Comandos bloqueados
- Rutas protegidas
- Timeout maximo 10 min
- Log de todas las operaciones
"""

import subprocess
import tempfile
import ast
import os
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum


class RiskLevel(Enum):
    LOW = "low"           # Auto-ejecutar
    MEDIUM = "medium"     # Pedir aprobacion
    HIGH = "high"         # Aprobacion + confirmacion
    CRITICAL = "critical" # Doble confirmacion


@dataclass
class ExecutionResult:
    """Resultado de una ejecucion"""
    success: bool
    output: str
    error: str = ""
    return_code: int = 0
    execution_time: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FileResult:
    """Resultado de operacion de archivo"""
    success: bool
    path: str
    content: str = ""
    error: str = ""
    size: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class IrisExecutor:
    """
    Motor de ejecucion para IRIS.
    Acceso completo al servidor con logging detallado.
    """

    WORKSPACE = Path("/root/NEO_EVA")
    TIMEOUT_DEFAULT = 120   # 2 minutos
    TIMEOUT_MAX = 600       # 10 minutos
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

    # Comandos bloqueados por seguridad
    BLOCKED_COMMANDS = [
        r"rm\s+-rf\s+/\s*$",
        r"rm\s+-rf\s+/\*",
        r"rm\s+-rf\s+~",
        r"shutdown",
        r"reboot",
        r"halt",
        r"poweroff",
        r"mkfs",
        r"dd\s+if=/dev/zero",
        r">\s*/dev/sd",
        r"chmod\s+-R\s+777\s+/",
        r":\(\)\{\s*:\|:&\s*\};:",  # Fork bomb
        r"mv\s+/\s+",
        r"cp\s+/dev/null\s+",
    ]

    # Rutas protegidas (no escribir)
    PROTECTED_PATHS = [
        "/etc/passwd",
        "/etc/shadow",
        "/etc/sudoers",
        "/root/.ssh/id_rsa",
        "/root/.ssh/authorized_keys",
    ]

    # Rutas de solo lectura
    READ_ONLY_PATHS = [
        "/var/log/",
        "/etc/",
    ]

    def __init__(self, workspace: str = None):
        self.workspace = Path(workspace) if workspace else self.WORKSPACE
        self.execution_log = self.workspace / "agents_state" / "iris_execution_log.jsonl"
        self.execution_log.parent.mkdir(parents=True, exist_ok=True)

    def _is_command_blocked(self, command: str) -> tuple[bool, str]:
        """Verifica si un comando esta bloqueado"""
        for pattern in self.BLOCKED_COMMANDS:
            if re.search(pattern, command, re.IGNORECASE):
                return True, f"Comando bloqueado por seguridad: {pattern}"
        return False, ""

    def _is_path_protected(self, path: str, write: bool = False) -> tuple[bool, str]:
        """Verifica si una ruta esta protegida"""
        path_str = str(Path(path).resolve())

        for protected in self.PROTECTED_PATHS:
            if path_str.startswith(protected) or path_str == protected:
                return True, f"Ruta protegida: {protected}"

        if write:
            for readonly in self.READ_ONLY_PATHS:
                if path_str.startswith(readonly):
                    return True, f"Ruta de solo lectura: {readonly}"

        return False, ""

    def _log_execution(self, exec_type: str, command: str, result: Union[ExecutionResult, FileResult, dict]):
        """Registra ejecucion en log JSONL"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": exec_type,
            "command": command[:2000],  # Limitar tamano
            "success": result.success if hasattr(result, 'success') else result.get('success', False),
            "error": (result.error if hasattr(result, 'error') else result.get('error', ''))[:500]
        }

        if hasattr(result, 'return_code'):
            log_entry["return_code"] = result.return_code
        if hasattr(result, 'execution_time'):
            log_entry["execution_time"] = result.execution_time

        try:
            with open(self.execution_log, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Error logging: {e}")

    def validate_python_code(self, code: str) -> tuple[bool, str]:
        """Valida que el codigo Python sea sintacticamente correcto"""
        try:
            ast.parse(code)
            return True, "OK"
        except SyntaxError as e:
            return False, f"SyntaxError linea {e.lineno}: {e.msg}"

    def classify_risk(self, action_type: str, payload: dict) -> RiskLevel:
        """Clasifica el nivel de riesgo de una accion"""

        # Acciones de solo lectura = bajo riesgo
        if action_type in ["read_file", "list_directory", "search"]:
            return RiskLevel.LOW

        # Ejecutar codigo = medio/alto
        if action_type == "execute_python":
            code = payload.get("code", "")
            # Detectar imports peligrosos
            dangerous_imports = ["os.system", "subprocess", "shutil.rmtree", "eval(", "exec("]
            for danger in dangerous_imports:
                if danger in code:
                    return RiskLevel.HIGH
            return RiskLevel.MEDIUM

        # Comandos bash = alto riesgo
        if action_type == "execute_bash":
            command = payload.get("command", "")
            # Comandos muy peligrosos
            if any(x in command for x in ["sudo", "rm -rf", "chmod", "chown"]):
                return RiskLevel.CRITICAL
            # Comandos de instalacion
            if any(x in command for x in ["pip install", "apt install", "npm install"]):
                return RiskLevel.HIGH
            # Git operations
            if "git push" in command or "git reset --hard" in command:
                return RiskLevel.HIGH
            return RiskLevel.MEDIUM

        # Escribir archivos = medio
        if action_type == "write_file":
            path = payload.get("path", "")
            # Archivos de configuracion = alto
            if any(x in path for x in [".env", "config", ".json", ".yaml"]):
                return RiskLevel.HIGH
            return RiskLevel.MEDIUM

        # Eliminar archivos = alto/critico
        if action_type == "delete_file":
            return RiskLevel.HIGH

        return RiskLevel.MEDIUM

    def execute_python(
        self,
        code: str,
        timeout: int = None,
        save_to: str = None,
        env: dict = None
    ) -> ExecutionResult:
        """
        Ejecuta codigo Python.

        Args:
            code: Codigo Python a ejecutar
            timeout: Timeout en segundos (default: 120, max: 600)
            save_to: Ruta opcional para guardar el script
            env: Variables de entorno adicionales
        """
        timeout = min(timeout or self.TIMEOUT_DEFAULT, self.TIMEOUT_MAX)

        # Validar sintaxis
        valid, error = self.validate_python_code(code)
        if not valid:
            result = ExecutionResult(
                success=False,
                output="",
                error=error,
                timestamp=datetime.now().isoformat()
            )
            self._log_execution("python", code[:500], result)
            return result

        # Guardar script si se solicita
        if save_to:
            script_path = self.workspace / save_to if not Path(save_to).is_absolute() else Path(save_to)
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(code)

        # Crear archivo temporal y ejecutar
        start_time = datetime.now()
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name

            # Preparar entorno
            exec_env = os.environ.copy()
            if env:
                exec_env.update(env)

            proc = subprocess.run(
                ['python3', temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace),
                env=exec_env
            )

            result = ExecutionResult(
                success=proc.returncode == 0,
                output=proc.stdout,
                error=proc.stderr,
                return_code=proc.returncode,
                execution_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now().isoformat()
            )

        except subprocess.TimeoutExpired:
            result = ExecutionResult(
                success=False,
                output="",
                error=f"Timeout: La ejecucion excedio {timeout} segundos",
                return_code=-1,
                execution_time=timeout,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            result = ExecutionResult(
                success=False,
                output="",
                error=str(e),
                return_code=-1,
                timestamp=datetime.now().isoformat()
            )
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

        self._log_execution("python", code[:500], result)
        return result

    def execute_bash(
        self,
        command: str,
        timeout: int = None,
        cwd: str = None
    ) -> ExecutionResult:
        """
        Ejecuta comando bash.

        Args:
            command: Comando a ejecutar
            timeout: Timeout en segundos
            cwd: Directorio de trabajo (default: WORKSPACE)
        """
        timeout = min(timeout or self.TIMEOUT_DEFAULT, self.TIMEOUT_MAX)
        cwd = cwd or str(self.workspace)

        # Verificar comandos bloqueados
        blocked, reason = self._is_command_blocked(command)
        if blocked:
            result = ExecutionResult(
                success=False,
                output="",
                error=reason,
                return_code=-1,
                timestamp=datetime.now().isoformat()
            )
            self._log_execution("bash_blocked", command, result)
            return result

        start_time = datetime.now()

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd
            )

            result = ExecutionResult(
                success=proc.returncode == 0,
                output=proc.stdout,
                error=proc.stderr,
                return_code=proc.returncode,
                execution_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now().isoformat()
            )

        except subprocess.TimeoutExpired:
            result = ExecutionResult(
                success=False,
                output="",
                error=f"Timeout: El comando excedio {timeout} segundos",
                return_code=-1,
                execution_time=timeout,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            result = ExecutionResult(
                success=False,
                output="",
                error=str(e),
                return_code=-1,
                timestamp=datetime.now().isoformat()
            )

        self._log_execution("bash", command, result)
        return result

    def read_file(self, path: str) -> FileResult:
        """Lee un archivo"""
        try:
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = self.workspace / path

            file_path = file_path.resolve()

            if not file_path.exists():
                return FileResult(
                    success=False,
                    path=str(file_path),
                    error="Archivo no existe"
                )

            if not file_path.is_file():
                return FileResult(
                    success=False,
                    path=str(file_path),
                    error="No es un archivo"
                )

            # Verificar tamano
            size = file_path.stat().st_size
            if size > self.MAX_FILE_SIZE:
                return FileResult(
                    success=False,
                    path=str(file_path),
                    error=f"Archivo muy grande: {size/1024/1024:.1f}MB (max: 10MB)"
                )

            content = file_path.read_text(encoding='utf-8')
            result = FileResult(
                success=True,
                path=str(file_path),
                content=content,
                size=len(content)
            )

        except UnicodeDecodeError:
            result = FileResult(
                success=False,
                path=path,
                error="Archivo binario, no se puede leer como texto"
            )
        except Exception as e:
            result = FileResult(
                success=False,
                path=path,
                error=str(e)
            )

        self._log_execution("read_file", path, result)
        return result

    def write_file(self, path: str, content: str, create_dirs: bool = True) -> FileResult:
        """Escribe un archivo"""
        try:
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = self.workspace / path

            file_path = file_path.resolve()

            # Verificar ruta protegida
            protected, reason = self._is_path_protected(str(file_path), write=True)
            if protected:
                return FileResult(
                    success=False,
                    path=str(file_path),
                    error=reason
                )

            # Verificar tamano
            if len(content.encode('utf-8')) > self.MAX_FILE_SIZE:
                return FileResult(
                    success=False,
                    path=str(file_path),
                    error=f"Contenido muy grande (max: 10MB)"
                )

            # Crear directorio padre si no existe
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)

            file_path.write_text(content, encoding='utf-8')

            result = FileResult(
                success=True,
                path=str(file_path),
                size=len(content),
                content=content  # Incluir contenido para mostrar en UI
            )

        except Exception as e:
            result = FileResult(
                success=False,
                path=path,
                error=str(e)
            )

        self._log_execution("write_file", f"{path} ({len(content)} bytes)", result)
        return result

    def delete_file(self, path: str) -> FileResult:
        """Elimina un archivo"""
        try:
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = self.workspace / path

            file_path = file_path.resolve()

            # Verificar ruta protegida
            protected, reason = self._is_path_protected(str(file_path), write=True)
            if protected:
                return FileResult(
                    success=False,
                    path=str(file_path),
                    error=reason
                )

            if not file_path.exists():
                return FileResult(
                    success=False,
                    path=str(file_path),
                    error="Archivo no existe"
                )

            file_path.unlink()

            result = FileResult(
                success=True,
                path=str(file_path)
            )

        except Exception as e:
            result = FileResult(
                success=False,
                path=path,
                error=str(e)
            )

        self._log_execution("delete_file", path, result)
        return result

    def list_directory(self, path: str = ".") -> dict:
        """Lista contenido de un directorio"""
        try:
            dir_path = Path(path)
            if not dir_path.is_absolute():
                dir_path = self.workspace / path

            dir_path = dir_path.resolve()

            if not dir_path.exists():
                return {"success": False, "path": str(dir_path), "error": "Directorio no existe"}

            if not dir_path.is_dir():
                return {"success": False, "path": str(dir_path), "error": "No es un directorio"}

            entries = []
            for entry in sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                try:
                    stat = entry.stat()
                    entries.append({
                        "name": entry.name,
                        "type": "directory" if entry.is_dir() else "file",
                        "size": stat.st_size if entry.is_file() else 0,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                except:
                    entries.append({
                        "name": entry.name,
                        "type": "unknown",
                        "size": 0
                    })

            result = {"success": True, "path": str(dir_path), "entries": entries, "count": len(entries)}

        except Exception as e:
            result = {"success": False, "path": path, "error": str(e)}

        self._log_execution("list_dir", path, result)
        return result

    def search_files(self, pattern: str, path: str = ".", max_results: int = 100) -> dict:
        """Busca archivos por patron glob"""
        try:
            search_path = Path(path)
            if not search_path.is_absolute():
                search_path = self.workspace / path

            search_path = search_path.resolve()

            matches = []
            for match in search_path.glob(pattern):
                if len(matches) >= max_results:
                    break
                try:
                    matches.append({
                        "path": str(match),
                        "name": match.name,
                        "type": "directory" if match.is_dir() else "file",
                        "size": match.stat().st_size if match.is_file() else 0
                    })
                except:
                    pass

            result = {
                "success": True,
                "pattern": pattern,
                "base_path": str(search_path),
                "matches": matches,
                "count": len(matches),
                "truncated": len(matches) >= max_results
            }

        except Exception as e:
            result = {"success": False, "pattern": pattern, "error": str(e)}

        self._log_execution("search", f"{pattern} in {path}", result)
        return result

    def get_execution_stats(self) -> dict:
        """Obtiene estadisticas de ejecuciones"""
        stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "by_type": {}
        }

        try:
            if self.execution_log.exists():
                with open(self.execution_log, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            stats["total"] += 1
                            if entry.get("success"):
                                stats["successful"] += 1
                            else:
                                stats["failed"] += 1

                            exec_type = entry.get("type", "unknown")
                            if exec_type not in stats["by_type"]:
                                stats["by_type"][exec_type] = 0
                            stats["by_type"][exec_type] += 1
                        except:
                            pass
        except:
            pass

        return stats


# Instancia global para uso directo
executor = IrisExecutor()


if __name__ == "__main__":
    # Test basico
    ex = IrisExecutor()

    print("=== Test IrisExecutor ===\n")

    # Test Python
    print("1. Ejecutar Python:")
    result = ex.execute_python("print('Hola desde IRIS!')\nprint(2+2)")
    print(f"   Success: {result.success}")
    print(f"   Output: {result.output.strip()}")
    print(f"   Time: {result.execution_time:.2f}s\n")

    # Test Bash
    print("2. Ejecutar Bash:")
    result = ex.execute_bash("echo 'Hola desde bash' && pwd")
    print(f"   Success: {result.success}")
    print(f"   Output: {result.output.strip()}\n")

    # Test leer archivo
    print("3. Leer archivo:")
    result = ex.read_file("core/iris_executor.py")
    print(f"   Success: {result.success}")
    print(f"   Size: {result.size} bytes\n")

    # Test listar directorio
    print("4. Listar directorio:")
    result = ex.list_directory("core")
    print(f"   Success: {result['success']}")
    print(f"   Files: {result.get('count', 0)}\n")

    # Test clasificacion de riesgo
    print("5. Clasificacion de riesgo:")
    print(f"   read_file: {ex.classify_risk('read_file', {}).value}")
    print(f"   execute_python: {ex.classify_risk('execute_python', {'code': 'print(1)'}).value}")
    print(f"   execute_bash (pip): {ex.classify_risk('execute_bash', {'command': 'pip install x'}).value}")
    print(f"   execute_bash (rm): {ex.classify_risk('execute_bash', {'command': 'rm -rf /'}).value}")

    print("\n=== Stats ===")
    print(ex.get_execution_stats())
