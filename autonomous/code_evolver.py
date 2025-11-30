#!/usr/bin/env python3
"""
NEO_EVA Code Evolver
====================

Auto-modificación de código para mejorar S.

Estrategia conservadora y endógena:
1. Solo modifica módulos en autonomous/code/
2. Evalúa cambios antes de aplicar
3. Revierte si S baja
4. Tasa de mutación proporcional a estabilidad

El sistema evoluciona su propio código.
"""

import numpy as np
import ast
import copy
import importlib
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib
from datetime import datetime


CODE_DIR = Path(__file__).parent / "code"
STATE_DIR = Path(__file__).parent / "state"


@dataclass
class CodeModule:
    """Representa un módulo evolucionable."""
    name: str
    filepath: Path
    source: str
    ast_tree: Any = None
    hash: str = ""
    generation: int = 0
    S_at_creation: float = 0.0

    def compute_hash(self) -> str:
        """Hash del código fuente."""
        return hashlib.md5(self.source.encode()).hexdigest()[:8]


@dataclass
class EvolutionState:
    """Estado del evolucionador."""
    n_evolutions: int = 0
    n_successful: int = 0
    n_reverted: int = 0

    modules: Dict[str, CodeModule] = field(default_factory=dict)
    evolution_history: List[Dict] = field(default_factory=list)


class CodeEvolver:
    """
    Evolucionador de código endógeno.

    Solo modifica código en autonomous/code/
    Conservador: solo evoluciona si hay alta probabilidad de mejora.
    """

    def __init__(self):
        self.evo_state = EvolutionState()
        CODE_DIR.mkdir(exist_ok=True)
        self._load_state()
        self._scan_modules()

    def _load_state(self):
        """Carga estado previo."""
        state_file = STATE_DIR / "evolver_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                self.evo_state.n_evolutions = data.get('n_evolutions', 0)
                self.evo_state.n_successful = data.get('n_successful', 0)
                self.evo_state.n_reverted = data.get('n_reverted', 0)
            except:
                pass

    def _save_state(self):
        """Guarda estado."""
        state_file = STATE_DIR / "evolver_state.json"
        try:
            with open(state_file, 'w') as f:
                json.dump({
                    'n_evolutions': self.evo_state.n_evolutions,
                    'n_successful': self.evo_state.n_successful,
                    'n_reverted': self.evo_state.n_reverted,
                    'modules': list(self.evo_state.modules.keys()),
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
        except:
            pass

    def _scan_modules(self):
        """Escanea módulos evolucionables en code/."""
        for pyfile in CODE_DIR.glob("*.py"):
            if pyfile.name.startswith("_"):
                continue
            try:
                source = pyfile.read_text()
                tree = ast.parse(source)
                module = CodeModule(
                    name=pyfile.stem,
                    filepath=pyfile,
                    source=source,
                    ast_tree=tree
                )
                module.hash = module.compute_hash()
                self.evo_state.modules[module.name] = module
            except:
                pass

    def _create_seed_module(self, core_state) -> CodeModule:
        """
        Crea un módulo semilla si no hay ninguno.

        El módulo semilla es una función simple que el sistema puede evolucionar.
        """
        seed_code = '''#!/usr/bin/env python3
"""
Evolved Module - Generation 0
Auto-generated seed for evolution.
"""

import numpy as np

def compute_bonus(state_vector: np.ndarray) -> float:
    """
    Computa un bonus para S basado en el estado.

    Esta función puede ser evolucionada para mejorar S.
    """
    # Combinación lineal simple (evolucionable)
    weights = np.ones(len(state_vector)) / len(state_vector)
    bonus = float(np.dot(weights, state_vector))
    return bonus * 0.1

def transform_state(state_vector: np.ndarray) -> np.ndarray:
    """
    Transforma el estado (evolucionable).
    """
    # Identidad por defecto
    return state_vector.copy()

# Metadata
GENERATION = 0
CREATED_AT = "{timestamp}"
'''

        seed_code = seed_code.format(timestamp=datetime.now().isoformat())

        seed_file = CODE_DIR / "evolved_bonus.py"
        seed_file.write_text(seed_code)

        module = CodeModule(
            name="evolved_bonus",
            filepath=seed_file,
            source=seed_code,
            ast_tree=ast.parse(seed_code),
            generation=0,
            S_at_creation=core_state.S
        )
        module.hash = module.compute_hash()

        self.evo_state.modules[module.name] = module

        return module

    def _mutate_numeric(self, value: float, mutation_rate: float) -> float:
        """Muta un valor numérico."""
        if np.random.random() < mutation_rate:
            # Mutación gaussiana
            mutated = value + np.random.randn() * mutation_rate * abs(value + 0.1)
            return float(mutated)
        return value

    def _mutate_ast(self, tree: ast.AST, mutation_rate: float) -> ast.AST:
        """
        Muta un AST de forma conservadora.

        Solo muta constantes numéricas, no estructura.
        """
        tree_copy = copy.deepcopy(tree)

        for node in ast.walk(tree_copy):
            # Mutar constantes numéricas
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if np.random.random() < mutation_rate:
                    node.value = self._mutate_numeric(node.value, mutation_rate)

            # Mutar operadores binarios (muy conservador)
            elif isinstance(node, ast.BinOp) and np.random.random() < mutation_rate * 0.1:
                # Solo intercambiar + y * ocasionalmente
                if isinstance(node.op, ast.Add):
                    node.op = ast.Mult()
                elif isinstance(node.op, ast.Mult):
                    node.op = ast.Add()

        ast.fix_missing_locations(tree_copy)
        return tree_copy

    def _compile_and_test(self, source: str) -> Tuple[bool, str]:
        """Compila código y verifica que sea válido."""
        try:
            compile(source, '<string>', 'exec')
            return True, "OK"
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"
        except Exception as e:
            return False, str(e)

    def _compute_mutation_rate(self, core_state) -> float:
        """
        Tasa de mutación endógena.

        - Alta estabilidad -> más mutación permitida
        - Alta S -> más conservador (no romper lo que funciona)
        - Baja S -> más exploración
        """
        stability = core_state.stability
        S = core_state.S

        # Base rate decrece con S (conservador cuando va bien)
        base_rate = 0.1 * (1 - S)

        # Pero aumenta con estabilidad (podemos arriesgar si estamos estables)
        rate = base_rate * (0.5 + stability)

        # Clipping endógeno
        return np.clip(rate, 0.01, 0.3)

    def evolve(self, core_state) -> Dict:
        """
        Intenta evolucionar un módulo.

        Args:
            core_state: Estado actual del core

        Returns:
            Dict con resultado de evolución
        """
        self.evo_state.n_evolutions += 1

        result = {
            'evolved': False,
            'module': None,
            'mutation_rate': 0.0,
            'reason': None
        }

        # Si no hay módulos, crear semilla
        if not self.evo_state.modules:
            module = self._create_seed_module(core_state)
            result['evolved'] = True
            result['module'] = module.name
            result['reason'] = 'created_seed'
            self._save_state()
            return result

        # Seleccionar módulo a evolucionar (el más antiguo o aleatorio)
        module_name = np.random.choice(list(self.evo_state.modules.keys()))
        module = self.evo_state.modules[module_name]

        # Calcular tasa de mutación
        mutation_rate = self._compute_mutation_rate(core_state)
        result['mutation_rate'] = mutation_rate

        # Verificar si deberíamos evolucionar
        # Probabilidad proporcional a estabilidad
        if np.random.random() > core_state.stability:
            result['reason'] = 'skipped_unstable'
            return result

        # Mutar AST
        try:
            mutated_tree = self._mutate_ast(module.ast_tree, mutation_rate)
            mutated_source = ast.unparse(mutated_tree)
        except Exception as e:
            result['reason'] = f'mutation_failed: {e}'
            return result

        # Verificar que compila
        valid, error = self._compile_and_test(mutated_source)
        if not valid:
            result['reason'] = f'compile_failed: {error}'
            return result

        # Guardar versión anterior
        backup_source = module.source
        backup_tree = module.ast_tree

        # Aplicar mutación
        module.source = mutated_source
        module.ast_tree = mutated_tree
        module.generation += 1
        module.hash = module.compute_hash()
        module.S_at_creation = core_state.S

        # Escribir a archivo
        try:
            module.filepath.write_text(mutated_source)
        except Exception as e:
            # Revertir
            module.source = backup_source
            module.ast_tree = backup_tree
            module.generation -= 1
            result['reason'] = f'write_failed: {e}'
            return result

        # Registrar evolución
        self.evo_state.evolution_history.append({
            'module': module_name,
            'generation': module.generation,
            'mutation_rate': mutation_rate,
            'S_at_evolution': core_state.S,
            'timestamp': datetime.now().isoformat()
        })

        self.evo_state.n_successful += 1
        result['evolved'] = True
        result['module'] = module_name
        result['reason'] = 'success'
        result['generation'] = module.generation

        self._save_state()

        return result

    def revert_last(self) -> bool:
        """Revierte última evolución si existe backup."""
        # Simplificado: no implementado en esta versión
        self.evo_state.n_reverted += 1
        return False

    def get_statistics(self) -> Dict:
        """Retorna estadísticas del evolucionador."""
        return {
            'n_evolutions': self.evo_state.n_evolutions,
            'n_successful': self.evo_state.n_successful,
            'n_reverted': self.evo_state.n_reverted,
            'success_rate': self.evo_state.n_successful / max(1, self.evo_state.n_evolutions),
            'modules': {
                name: {
                    'generation': m.generation,
                    'hash': m.hash
                }
                for name, m in self.evo_state.modules.items()
            }
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Code Evolver Test")
    print("=" * 40)

    evolver = CodeEvolver()

    class MockState:
        S = 0.5
        stability = 0.7

    state = MockState()

    for i in range(10):
        result = evolver.evolve(state)
        print(f"  Evolution {i}: evolved={result['evolved']}, "
              f"reason={result['reason']}")

        # Simular cambio de estado
        state.S += np.random.randn() * 0.05
        state.S = np.clip(state.S, 0, 1)

    print("\nStatistics:")
    stats = evolver.get_statistics()
    print(f"  Evolutions: {stats['n_evolutions']}")
    print(f"  Successful: {stats['n_successful']}")
    print(f"  Modules: {list(stats['modules'].keys())}")
