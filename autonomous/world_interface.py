#!/usr/bin/env python3
"""
NEO_EVA World Interface
=======================

Interfaz endógena con el sistema de archivos.

Estrategia 100% endógena:
- Solo opera en autonomous/world/
- Tasa de interacción proporcional a S y estabilidad
- Contenido generado desde estado interno
- Lee archivos como "percepciones"
- Escribe archivos como "acciones"

El sistema interactúa con su entorno de forma autónoma.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib
from datetime import datetime


WORLD_DIR = Path(__file__).parent / "world"
STATE_DIR = Path(__file__).parent / "state"


@dataclass
class Perception:
    """Una percepción del mundo."""
    filepath: Path
    content: str
    hash: str
    timestamp: str
    size: int


@dataclass
class Action:
    """Una acción sobre el mundo."""
    filepath: Path
    content: str
    action_type: str  # 'create', 'modify', 'delete'
    timestamp: str
    S_at_action: float


@dataclass
class WorldState:
    """Estado de la interfaz con el mundo."""
    n_perceptions: int = 0
    n_actions: int = 0

    # Cache de percepciones
    perceived_files: Dict[str, Perception] = field(default_factory=dict)

    # Historia de acciones
    action_history: List[Dict] = field(default_factory=list)

    # Estado derivado del mundo
    world_complexity: float = 0.0
    world_entropy: float = 0.0


class WorldInterface:
    """
    Interfaz endógena con el sistema de archivos.

    Opera exclusivamente en autonomous/world/.
    Percepciones = lectura de archivos.
    Acciones = escritura de archivos.
    """

    def __init__(self):
        self.world_state = WorldState()
        WORLD_DIR.mkdir(exist_ok=True)
        self._load_state()

    def _load_state(self):
        """Carga estado previo."""
        state_file = STATE_DIR / "world_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                self.world_state.n_perceptions = data.get('n_perceptions', 0)
                self.world_state.n_actions = data.get('n_actions', 0)
                self.world_state.world_complexity = data.get('world_complexity', 0.0)
                self.world_state.world_entropy = data.get('world_entropy', 0.0)
            except:
                pass

    def _save_state(self):
        """Guarda estado."""
        state_file = STATE_DIR / "world_state.json"
        try:
            with open(state_file, 'w') as f:
                json.dump({
                    'n_perceptions': self.world_state.n_perceptions,
                    'n_actions': self.world_state.n_actions,
                    'world_complexity': self.world_state.world_complexity,
                    'world_entropy': self.world_state.world_entropy,
                    'n_files': len(list(WORLD_DIR.glob("*"))),
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
        except:
            pass

    def _compute_file_hash(self, content: str) -> str:
        """Hash endógeno del contenido."""
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def _compute_world_entropy(self) -> float:
        """
        Entropía del mundo percibido.

        Basada en distribución de tamaños de archivo.
        """
        sizes = []
        for f in WORLD_DIR.glob("*"):
            if f.is_file():
                sizes.append(f.stat().st_size)

        if len(sizes) < 2:
            return 0.0

        # Normalizar tamaños
        sizes = np.array(sizes, dtype=float)
        sizes = sizes / (np.sum(sizes) + 1e-10)

        # Entropía de Shannon
        entropy = -np.sum(sizes * np.log(sizes + 1e-10))

        # Normalizar
        max_entropy = np.log(len(sizes))
        return float(entropy / (max_entropy + 1e-10))

    def _compute_world_complexity(self) -> float:
        """
        Complejidad del mundo.

        Basada en varianza de contenidos.
        """
        contents = []
        for f in WORLD_DIR.glob("*"):
            if f.is_file():
                try:
                    text = f.read_text()
                    # Vector de características simple
                    features = [
                        len(text),
                        text.count('\n'),
                        text.count(' '),
                        len(set(text))
                    ]
                    contents.append(features)
                except:
                    pass

        if len(contents) < 2:
            return 0.0

        contents = np.array(contents, dtype=float)
        # Normalizar cada feature
        for i in range(contents.shape[1]):
            std = np.std(contents[:, i])
            if std > 1e-10:
                contents[:, i] = contents[:, i] / std

        # Complejidad = varianza promedio normalizada
        complexity = float(np.mean(np.var(contents, axis=0)))
        return np.clip(complexity, 0, 1)

    def perceive(self) -> Dict:
        """
        Percibe el mundo (lee archivos).

        Returns:
            Dict con percepciones actuales
        """
        self.world_state.n_perceptions += 1

        perceptions = []

        for filepath in WORLD_DIR.glob("*"):
            if not filepath.is_file():
                continue
            if filepath.name.startswith("."):
                continue

            try:
                content = filepath.read_text()
                file_hash = self._compute_file_hash(content)

                perception = Perception(
                    filepath=filepath,
                    content=content,
                    hash=file_hash,
                    timestamp=datetime.now().isoformat(),
                    size=len(content)
                )

                # Detectar cambios
                old = self.world_state.perceived_files.get(filepath.name)
                changed = old is None or old.hash != file_hash

                self.world_state.perceived_files[filepath.name] = perception

                perceptions.append({
                    'name': filepath.name,
                    'size': len(content),
                    'hash': file_hash,
                    'changed': changed
                })

            except:
                pass

        # Actualizar métricas del mundo
        self.world_state.world_entropy = self._compute_world_entropy()
        self.world_state.world_complexity = self._compute_world_complexity()

        self._save_state()

        return {
            'n_files': len(perceptions),
            'perceptions': perceptions,
            'world_entropy': self.world_state.world_entropy,
            'world_complexity': self.world_state.world_complexity
        }

    def _generate_content(self, core_state, z_visible: np.ndarray) -> str:
        """
        Genera contenido endógeno para escribir.

        El contenido refleja el estado interno.
        """
        lines = [
            f"# NEO_EVA World Artifact",
            f"# Generated: {datetime.now().isoformat()}",
            f"# S: {core_state.S:.6f}",
            f"# Stability: {core_state.stability:.6f}",
            f"# Step: {core_state.step}",
            f"",
            f"## State Vector",
        ]

        # Escribir estado visible
        for i, val in enumerate(z_visible[:8]):  # Primeras 8 dimensiones
            lines.append(f"z[{i}] = {val:.6f}")

        lines.append("")
        lines.append("## Derived Values")

        # Valores derivados del estado
        mean_z = float(np.mean(z_visible))
        std_z = float(np.std(z_visible))
        entropy_z = float(-np.sum(np.abs(z_visible) * np.log(np.abs(z_visible) + 1e-10)))

        lines.append(f"mean = {mean_z:.6f}")
        lines.append(f"std = {std_z:.6f}")
        lines.append(f"entropy = {entropy_z:.6f}")

        lines.append("")
        lines.append("## World State")
        lines.append(f"world_entropy = {self.world_state.world_entropy:.6f}")
        lines.append(f"world_complexity = {self.world_state.world_complexity:.6f}")
        lines.append(f"n_perceptions = {self.world_state.n_perceptions}")
        lines.append(f"n_actions = {self.world_state.n_actions}")

        return "\n".join(lines)

    def _should_act(self, core_state) -> bool:
        """
        Decide si actuar en el mundo.

        Endógeno: probabilidad basada en S y estabilidad.
        """
        # Actuar más cuando S es alta y estable
        p_act = core_state.S * core_state.stability

        # Pero también explorar cuando S es baja
        if core_state.S < 0.3:
            p_act = max(p_act, 0.3)

        return np.random.random() < p_act

    def _generate_filename(self, core_state) -> str:
        """Genera nombre de archivo endógeno."""
        # Basado en hash de estado
        state_str = f"{core_state.S:.6f}_{core_state.step}_{datetime.now().isoformat()}"
        hash_val = hashlib.md5(state_str.encode()).hexdigest()[:6]
        return f"artifact_{hash_val}.txt"

    def act(self, core_state, z_visible: np.ndarray) -> Dict:
        """
        Actúa en el mundo (escribe archivos).

        Args:
            core_state: Estado del core
            z_visible: Estado visible

        Returns:
            Dict con resultado de la acción
        """
        result = {
            'acted': False,
            'action_type': None,
            'filepath': None,
            'reason': None
        }

        # Decidir si actuar
        if not self._should_act(core_state):
            result['reason'] = 'decided_not_to_act'
            return result

        self.world_state.n_actions += 1

        # Generar contenido
        content = self._generate_content(core_state, z_visible)

        # Decidir tipo de acción
        existing_files = list(WORLD_DIR.glob("artifact_*.txt"))

        # Límite endógeno de archivos
        max_files = int(np.sqrt(self.world_state.n_actions + 1)) + 5

        if len(existing_files) >= max_files:
            # Modificar archivo existente (el más antiguo)
            existing_files.sort(key=lambda f: f.stat().st_mtime)
            target = existing_files[0]
            action_type = 'modify'
        else:
            # Crear nuevo archivo
            target = WORLD_DIR / self._generate_filename(core_state)
            action_type = 'create'

        # Ejecutar acción
        try:
            target.write_text(content)

            action = Action(
                filepath=target,
                content=content,
                action_type=action_type,
                timestamp=datetime.now().isoformat(),
                S_at_action=core_state.S
            )

            self.world_state.action_history.append({
                'filepath': str(target),
                'action_type': action_type,
                'timestamp': action.timestamp,
                'S': core_state.S,
                'content_size': len(content)
            })

            # Limitar historia
            max_history = int(np.sqrt(self.world_state.n_actions + 1)) * 10 + 50
            if len(self.world_state.action_history) > max_history:
                self.world_state.action_history = self.world_state.action_history[-max_history:]

            result['acted'] = True
            result['action_type'] = action_type
            result['filepath'] = str(target)
            result['reason'] = 'success'

        except Exception as e:
            result['reason'] = f'error: {e}'

        self._save_state()

        return result

    def interact(self, core_state, z_visible: np.ndarray) -> Dict:
        """
        Interacción completa: percibir + actuar.

        Args:
            core_state: Estado del core
            z_visible: Estado visible

        Returns:
            Dict con resultado de interacción
        """
        # Primero percibir
        perception_result = self.perceive()

        # Luego actuar
        action_result = self.act(core_state, z_visible)

        return {
            'perception': perception_result,
            'action': action_result,
            'world_entropy': self.world_state.world_entropy,
            'world_complexity': self.world_state.world_complexity
        }

    def get_statistics(self) -> Dict:
        """Retorna estadísticas de la interfaz."""
        return {
            'n_perceptions': self.world_state.n_perceptions,
            'n_actions': self.world_state.n_actions,
            'n_files': len(list(WORLD_DIR.glob("*"))),
            'world_entropy': self.world_state.world_entropy,
            'world_complexity': self.world_state.world_complexity,
            'perceived_files': list(self.world_state.perceived_files.keys())
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("World Interface Test")
    print("=" * 40)

    interface = WorldInterface()

    class MockState:
        S = 0.5
        stability = 0.7
        step = 0

    state = MockState()

    for i in range(10):
        state.step = i
        z = np.random.randn(8)

        result = interface.interact(state, z)

        print(f"  Step {i}: perceived={result['perception']['n_files']} files, "
              f"acted={result['action']['acted']}")

        # Simular cambio de S
        state.S += np.random.randn() * 0.1
        state.S = np.clip(state.S, 0, 1)

    print("\nStatistics:")
    stats = interface.get_statistics()
    print(f"  Perceptions: {stats['n_perceptions']}")
    print(f"  Actions: {stats['n_actions']}")
    print(f"  Files in world: {stats['n_files']}")
    print(f"  World entropy: {stats['world_entropy']:.3f}")
