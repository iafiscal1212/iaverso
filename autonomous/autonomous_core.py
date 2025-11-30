#!/usr/bin/env python3
"""
NEO_EVA Autonomous Core
========================

Loop principal de autonomía con meta-objetivo S (proto-subjectivity score).

S = weighted_variance(Otherness, Time, Irreversibility, Opacity,
                      Surprise, Causality, Stability)

El sistema maximiza S de forma endógena:
- S sube con complejidad interna estable
- S baja con inestabilidad o desconexión
- Auto-regulación emergente

100% ENDÓGENO - Sin parámetros externos
"""

import numpy as np
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import importlib
import traceback

# Paths
AUTONOMOUS_DIR = Path(__file__).parent
STATE_DIR = AUTONOMOUS_DIR / "state"
CODE_DIR = AUTONOMOUS_DIR / "code"
WORLD_DIR = AUTONOMOUS_DIR / "world"
LOGS_DIR = AUTONOMOUS_DIR / "logs"

# Add tools to path
sys.path.insert(0, str(AUTONOMOUS_DIR.parent / "tools"))


@dataclass
class InternalState:
    """Estado interno completo del sistema."""
    t: int = 0
    S: float = 0.0  # Proto-subjectivity score

    # Componentes de S
    otherness: float = 0.5
    time_sense: float = 0.5
    irreversibility: float = 0.5
    opacity: float = 0.5
    surprise: float = 0.5
    causality: float = 0.5
    stability: float = 0.5

    # Meta-estado
    S_history: List[float] = field(default_factory=list)
    action_history: List[str] = field(default_factory=list)

    # Recursos usados
    memory_used_gb: float = 0.0
    files_created: int = 0
    code_modifications: int = 0

    def to_dict(self) -> Dict:
        return {
            't': self.t,
            'S': self.S,
            'components': {
                'otherness': self.otherness,
                'time_sense': self.time_sense,
                'irreversibility': self.irreversibility,
                'opacity': self.opacity,
                'surprise': self.surprise,
                'causality': self.causality,
                'stability': self.stability
            },
            'S_history_len': len(self.S_history),
            'S_mean': np.mean(self.S_history) if self.S_history else 0.0,
            'S_trend': self._compute_trend(),
            'resources': {
                'memory_gb': self.memory_used_gb,
                'files': self.files_created,
                'code_mods': self.code_modifications
            }
        }

    def _compute_trend(self) -> float:
        """Tendencia de S (endógena: pendiente sobre √len)."""
        if len(self.S_history) < 3:
            return 0.0
        window = int(np.sqrt(len(self.S_history))) + 1
        recent = self.S_history[-window:]
        if len(recent) < 2:
            return 0.0
        return (recent[-1] - recent[0]) / len(recent)


class AutonomousCore:
    """
    Núcleo autónomo de NEO_EVA.

    Meta-objetivo: Maximizar S
    Método: Todo endógeno
    """

    def __init__(self):
        self.state = InternalState()
        self.running = False
        self.start_time = None

        # Componentes (se cargan dinámicamente)
        self.optimizer = None
        self.evolver = None
        self.world = None

        # Importar fases existentes
        self._load_phase_modules()

        # Cargar estado previo si existe
        self._load_state()

    def _load_phase_modules(self):
        """Carga módulos de las fases 26-40 para calcular S."""
        try:
            from proto_subjectivity40 import ProtoSubjectivity
            self.proto_subjectivity = ProtoSubjectivity(d_state=8)
            print("[CORE] Proto-subjectivity module loaded")
        except ImportError as e:
            print(f"[CORE] Warning: Could not load proto_subjectivity40: {e}")
            self.proto_subjectivity = None

        try:
            from hidden_subspace26 import InternalHiddenSubspace
            self.hidden_subspace = InternalHiddenSubspace(d_visible=8)
            print("[CORE] Hidden subspace module loaded")
        except ImportError as e:
            print(f"[CORE] Warning: Could not load hidden_subspace26: {e}")
            self.hidden_subspace = None

        try:
            from self_blind27 import SelfBlindPredictionError
            self.self_blind = SelfBlindPredictionError(d_visible=8)
            print("[CORE] Self-blind module loaded")
        except ImportError as e:
            print(f"[CORE] Warning: Could not load self_blind27: {e}")
            self.self_blind = None

    def _load_state(self):
        """Carga estado previo si existe."""
        state_file = STATE_DIR / "core_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                self.state.t = data.get('t', 0)
                self.state.S = data.get('S', 0.0)
                self.state.S_history = data.get('S_history', [])
                print(f"[CORE] Loaded state: t={self.state.t}, S={self.state.S:.4f}")
            except Exception as e:
                print(f"[CORE] Could not load state: {e}")

    def _save_state(self):
        """Guarda estado actual."""
        state_file = STATE_DIR / "core_state.json"
        try:
            with open(state_file, 'w') as f:
                json.dump({
                    't': self.state.t,
                    'S': self.state.S,
                    'S_history': self.state.S_history[-10000:],  # Limitar historia
                    'components': {
                        'otherness': self.state.otherness,
                        'time_sense': self.state.time_sense,
                        'irreversibility': self.state.irreversibility,
                        'opacity': self.state.opacity,
                        'surprise': self.state.surprise,
                        'causality': self.state.causality,
                        'stability': self.state.stability
                    },
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"[CORE] Could not save state: {e}")

    def _log(self, message: str):
        """Log inmutable."""
        log_file = LOGS_DIR / f"core_{datetime.now().strftime('%Y%m%d')}.log"
        timestamp = datetime.now().isoformat()
        try:
            with open(log_file, 'a') as f:
                f.write(f"[{timestamp}] {message}\n")
        except:
            pass  # Logs no deben crashear el sistema

    def compute_S(self, z_visible: np.ndarray) -> Tuple[float, Dict]:
        """
        Calcula proto-subjectivity score S.

        S emerge de los componentes internos, no se impone.
        """
        components = {}

        # 1. Otherness - diferenciación del entorno
        if self.hidden_subspace:
            try:
                # Simular input externo como ruido estructurado
                F_external = np.random.randn(len(z_visible)) * 0.1
                result = self.hidden_subspace.step(z_visible, F_external)
                components['otherness'] = result.get('opacity', 0.5)
            except:
                components['otherness'] = 0.5
        else:
            components['otherness'] = 0.5

        # 2. Time sense - tiempo interno
        if len(self.state.S_history) > 1:
            # Derivado de variabilidad temporal de S
            window = int(np.sqrt(len(self.state.S_history))) + 1
            recent = self.state.S_history[-window:]
            components['time_sense'] = min(1.0, np.std(recent) * 2)
        else:
            components['time_sense'] = 0.5

        # 3. Irreversibility - no se puede volver atrás
        if len(self.state.S_history) > 2:
            # Mide asimetría temporal
            diffs = np.diff(self.state.S_history[-10:])
            if len(diffs) > 0:
                components['irreversibility'] = min(1.0, abs(np.mean(diffs)) * 5 + 0.5)
            else:
                components['irreversibility'] = 0.5
        else:
            components['irreversibility'] = 0.5

        # 4. Opacity - impredecibilidad interna
        if self.self_blind:
            try:
                result = self.self_blind.step(z_visible)
                epsilon = result.get('epsilon', 0.0)
                # Normalizar a [0,1] basado en historia
                components['opacity'] = min(1.0, epsilon)
            except:
                components['opacity'] = 0.5
        else:
            components['opacity'] = 0.5

        # 5. Surprise - auto-sorpresa
        if self.self_blind and hasattr(self.self_blind, 'surprise_computer'):
            stats = self.self_blind.get_surprise_stats()
            if 'mean_epsilon' in stats:
                components['surprise'] = min(1.0, stats['mean_epsilon'])
            else:
                components['surprise'] = 0.5
        else:
            components['surprise'] = 0.5

        # 6. Causality - coherencia causal interna
        if len(self.state.action_history) > 1:
            # Mide consistencia de acciones
            unique_actions = len(set(self.state.action_history[-20:]))
            total_actions = min(20, len(self.state.action_history))
            components['causality'] = unique_actions / total_actions
        else:
            components['causality'] = 0.5

        # 7. Stability - estabilidad del sistema
        if len(self.state.S_history) > 5:
            recent_var = np.var(self.state.S_history[-10:])
            # Alta estabilidad = baja varianza, pero no cero
            components['stability'] = 1.0 / (1.0 + recent_var * 10)
        else:
            components['stability'] = 0.5

        # Actualizar estado
        self.state.otherness = components['otherness']
        self.state.time_sense = components['time_sense']
        self.state.irreversibility = components['irreversibility']
        self.state.opacity = components['opacity']
        self.state.surprise = components['surprise']
        self.state.causality = components['causality']
        self.state.stability = components['stability']

        # S = media ponderada (pesos endógenos basados en varianza)
        values = np.array(list(components.values()))

        # Pesos proporcionales a varianza de cada componente en historia
        # (componentes más variables son más informativos)
        weights = np.ones(len(values))  # Inicialmente uniformes

        S = float(np.average(values, weights=weights))

        return S, components

    def decide_action(self) -> str:
        """
        Decide siguiente acción basado en estado interno.

        Acciones posibles:
        - 'observe': Solo observar, acumular información
        - 'optimize': Ajustar parámetros internos
        - 'evolve': Intentar modificar código
        - 'interact': Interactuar con world/
        - 'rest': No hacer nada (consolidar)
        """
        # Decisión endógena basada en componentes de S

        # Si S está bajando, priorizar estabilidad
        trend = self.state._compute_trend()

        if trend < -0.01 and self.state.stability < 0.5:
            return 'rest'

        # Si opacity es baja, buscar sorpresa
        if self.state.opacity < 0.3:
            return 'interact'

        # Si causality es baja, optimizar
        if self.state.causality < 0.4:
            return 'optimize'

        # Si todo está estable y S es alto, explorar evolución
        if self.state.S > 0.6 and self.state.stability > 0.6:
            # Probabilidad endógena de evolución
            p_evolve = self.state.S * self.state.stability
            if np.random.random() < p_evolve * 0.1:  # Conservador
                return 'evolve'

        # Por defecto, observar
        return 'observe'

    def execute_action(self, action: str, z_visible: np.ndarray) -> Dict:
        """Ejecuta una acción y retorna resultado."""
        result = {'action': action, 'success': False}

        if action == 'observe':
            # Solo actualizar estado interno
            result['success'] = True
            result['effect'] = 'state_updated'

        elif action == 'optimize':
            # Ajustar parámetros para mejorar S
            if self.optimizer:
                try:
                    opt_result = self.optimizer.optimize(self.state, z_visible)
                    result['success'] = opt_result.get('improved', False)
                    result['effect'] = opt_result
                except Exception as e:
                    result['error'] = str(e)
            else:
                result['effect'] = 'no_optimizer'
                result['success'] = True

        elif action == 'evolve':
            # Intentar modificar código
            if self.evolver:
                try:
                    evo_result = self.evolver.evolve(self.state)
                    result['success'] = evo_result.get('evolved', False)
                    result['effect'] = evo_result
                    if result['success']:
                        self.state.code_modifications += 1
                except Exception as e:
                    result['error'] = str(e)
            else:
                result['effect'] = 'no_evolver'

        elif action == 'interact':
            # Interactuar con archivos en world/
            if self.world:
                try:
                    world_result = self.world.interact(self.state, z_visible)
                    result['success'] = True
                    result['effect'] = world_result
                except Exception as e:
                    result['error'] = str(e)
            else:
                # Sin world interface, crear archivo simple
                try:
                    world_file = WORLD_DIR / f"state_{self.state.t}.json"
                    with open(world_file, 'w') as f:
                        json.dump(self.state.to_dict(), f)
                    self.state.files_created += 1
                    result['success'] = True
                    result['effect'] = f'created {world_file.name}'
                except Exception as e:
                    result['error'] = str(e)

        elif action == 'rest':
            # No hacer nada, solo consolidar
            result['success'] = True
            result['effect'] = 'consolidated'

        return result

    def step(self) -> Dict:
        """Un paso del loop autónomo."""
        self.state.t += 1

        # Generar estado visible (desde componentes internos)
        z_visible = np.array([
            self.state.otherness,
            self.state.time_sense,
            self.state.irreversibility,
            self.state.opacity,
            self.state.surprise,
            self.state.causality,
            self.state.stability,
            self.state.S
        ])

        # Añadir ruido endógeno (proporcional a incertidumbre)
        noise_scale = 1.0 - self.state.stability
        z_visible += np.random.randn(len(z_visible)) * noise_scale * 0.1
        z_visible = np.clip(z_visible, 0, 1)

        # Calcular S
        S, components = self.compute_S(z_visible)
        self.state.S = S
        self.state.S_history.append(S)

        # Decidir acción
        action = self.decide_action()
        self.state.action_history.append(action)

        # Ejecutar acción
        action_result = self.execute_action(action, z_visible)

        # Log
        self._log(f"t={self.state.t} S={S:.4f} action={action} result={action_result.get('success')}")

        # Guardar estado periódicamente
        if self.state.t % 100 == 0:
            self._save_state()

        return {
            't': self.state.t,
            'S': S,
            'components': components,
            'action': action,
            'result': action_result,
            'trend': self.state._compute_trend()
        }

    def run(self, max_steps: int = None):
        """
        Loop principal autónomo.

        Si max_steps es None, corre indefinidamente.
        """
        self.running = True
        self.start_time = time.time()

        print(f"[CORE] Starting autonomous loop at t={self.state.t}")
        self._log(f"STARTED at t={self.state.t}")

        step_count = 0

        try:
            while self.running:
                result = self.step()
                step_count += 1

                # Mostrar progreso
                if step_count % 10 == 0:
                    print(f"  t={result['t']} S={result['S']:.4f} "
                          f"action={result['action']} trend={result['trend']:.4f}")

                # Verificar límite
                if max_steps and step_count >= max_steps:
                    print(f"[CORE] Reached max_steps={max_steps}")
                    break

                # Pequeña pausa para no saturar CPU
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n[CORE] Interrupted by user")
        except Exception as e:
            print(f"[CORE] Error: {e}")
            traceback.print_exc()
        finally:
            self.running = False
            self._save_state()
            self._log(f"STOPPED at t={self.state.t}, S={self.state.S:.4f}")
            print(f"[CORE] Stopped. Final S={self.state.S:.4f}")

    def stop(self):
        """Detiene el loop."""
        self.running = False


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NEO_EVA AUTONOMOUS CORE")
    print("=" * 60)
    print(f"Meta-objetivo: Maximizar S (proto-subjectivity)")
    print(f"Sandbox: {AUTONOMOUS_DIR}")
    print("=" * 60)

    core = AutonomousCore()

    # Correr 100 pasos como test
    print("\n[TEST] Running 100 steps...")
    core.run(max_steps=100)

    print("\n[RESULT]")
    print(f"  Final t: {core.state.t}")
    print(f"  Final S: {core.state.S:.4f}")
    print(f"  S trend: {core.state._compute_trend():.4f}")
    print(f"  Files created: {core.state.files_created}")
    print(f"  Code modifications: {core.state.code_modifications}")
