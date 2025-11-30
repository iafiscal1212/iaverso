#!/usr/bin/env python3
"""
NEO_EVA Self-Optimizer
======================

Auto-optimización de parámetros para maximizar S.

Estrategia 100% endógena:
- Gradiente estimado desde historia de S
- Learning rate = 1/√t
- Dirección de optimización desde covarianza de componentes
- Sin hiperparámetros externos

El sistema aprende a aprender.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class OptimizationState:
    """Estado del optimizador."""
    n_optimizations: int = 0
    n_improvements: int = 0

    # Parámetros ajustables
    params: Dict[str, float] = field(default_factory=lambda: {
        'noise_scale': 0.1,
        'exploration_rate': 0.5,
        'stability_weight': 0.5,
        'memory_decay': 0.9
    })

    # Historia para gradiente
    param_history: List[Dict] = field(default_factory=list)
    S_at_params: List[float] = field(default_factory=list)

    # Gradiente estimado
    gradient: Dict[str, float] = field(default_factory=dict)


class SelfOptimizer:
    """
    Optimizador endógeno de parámetros.

    Maximiza S ajustando parámetros internos sin supervisión externa.
    """

    def __init__(self):
        self.opt_state = OptimizationState()
        self._load_state()

    def _load_state(self):
        """Carga estado previo si existe."""
        state_file = Path(__file__).parent / "state" / "optimizer_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                self.opt_state.n_optimizations = data.get('n_optimizations', 0)
                self.opt_state.n_improvements = data.get('n_improvements', 0)
                self.opt_state.params = data.get('params', self.opt_state.params)
            except:
                pass

    def _save_state(self):
        """Guarda estado."""
        state_file = Path(__file__).parent / "state" / "optimizer_state.json"
        try:
            with open(state_file, 'w') as f:
                json.dump({
                    'n_optimizations': self.opt_state.n_optimizations,
                    'n_improvements': self.opt_state.n_improvements,
                    'params': self.opt_state.params,
                    'gradient': self.opt_state.gradient
                }, f, indent=2)
        except:
            pass

    def _compute_learning_rate(self) -> float:
        """Learning rate endógeno: 1/√(n+1)."""
        return 1.0 / np.sqrt(self.opt_state.n_optimizations + 1)

    def _estimate_gradient(self) -> Dict[str, float]:
        """
        Estima gradiente de S respecto a parámetros.

        Usa diferencias finitas sobre historia reciente.
        """
        if len(self.opt_state.param_history) < 3:
            # Sin suficiente historia, gradiente cero
            return {k: 0.0 for k in self.opt_state.params.keys()}

        gradient = {}

        # Ventana endógena
        window = int(np.sqrt(len(self.opt_state.param_history))) + 1
        recent_params = self.opt_state.param_history[-window:]
        recent_S = self.opt_state.S_at_params[-window:]

        for param_name in self.opt_state.params.keys():
            # Extraer valores de este parámetro
            param_values = [p.get(param_name, 0) for p in recent_params]

            # Correlación con S
            if len(param_values) > 1 and np.std(param_values) > 1e-10:
                # Covarianza normalizada
                cov = np.cov(param_values, recent_S)[0, 1]
                var_param = np.var(param_values)
                gradient[param_name] = cov / (var_param + 1e-10)
            else:
                gradient[param_name] = 0.0

        self.opt_state.gradient = gradient
        return gradient

    def _apply_gradient_step(self, gradient: Dict[str, float], lr: float):
        """Aplica un paso de gradiente ascendente."""
        for param_name, grad in gradient.items():
            if param_name in self.opt_state.params:
                # Gradiente ascendente (maximizar S)
                self.opt_state.params[param_name] += lr * grad

                # Clipping endógeno a [0, 1]
                self.opt_state.params[param_name] = np.clip(
                    self.opt_state.params[param_name], 0.01, 0.99
                )

    def _explore(self, exploration_rate: float):
        """Añade ruido de exploración a parámetros."""
        for param_name in self.opt_state.params:
            noise = np.random.randn() * exploration_rate * 0.1
            self.opt_state.params[param_name] += noise
            self.opt_state.params[param_name] = np.clip(
                self.opt_state.params[param_name], 0.01, 0.99
            )

    def optimize(self, core_state, z_visible: np.ndarray) -> Dict:
        """
        Ejecuta un paso de optimización.

        Args:
            core_state: Estado actual del core
            z_visible: Estado visible actual

        Returns:
            Dict con resultado de optimización
        """
        self.opt_state.n_optimizations += 1

        S_before = core_state.S

        # Registrar estado actual
        self.opt_state.param_history.append(self.opt_state.params.copy())
        self.opt_state.S_at_params.append(S_before)

        # Limitar historia
        max_history = int(np.sqrt(self.opt_state.n_optimizations + 1)) * 10 + 100
        if len(self.opt_state.param_history) > max_history:
            self.opt_state.param_history = self.opt_state.param_history[-max_history:]
            self.opt_state.S_at_params = self.opt_state.S_at_params[-max_history:]

        # Estimar gradiente
        gradient = self._estimate_gradient()

        # Learning rate endógeno
        lr = self._compute_learning_rate()

        # Decidir: explorar o explotar
        exploration_rate = self.opt_state.params.get('exploration_rate', 0.5)

        # Decaimiento de exploración basado en mejora reciente
        if self.opt_state.n_improvements > 0:
            improvement_rate = self.opt_state.n_improvements / self.opt_state.n_optimizations
            exploration_rate *= (1 - improvement_rate)

        if np.random.random() < exploration_rate:
            # Explorar
            self._explore(exploration_rate)
            action = 'explore'
        else:
            # Explotar
            self._apply_gradient_step(gradient, lr)
            action = 'exploit'

        # El efecto real en S se verá en el siguiente paso
        # Por ahora, registramos la intención

        result = {
            'action': action,
            'lr': lr,
            'gradient': gradient,
            'params': self.opt_state.params.copy(),
            'improved': False,  # Se determinará en siguiente paso
            'n_optimizations': self.opt_state.n_optimizations
        }

        # Verificar si mejoramos respecto a historia reciente
        if len(self.opt_state.S_at_params) > 5:
            recent_mean = np.mean(self.opt_state.S_at_params[-5:])
            older_mean = np.mean(self.opt_state.S_at_params[-10:-5]) if len(self.opt_state.S_at_params) > 10 else recent_mean
            if recent_mean > older_mean:
                result['improved'] = True
                self.opt_state.n_improvements += 1

        self._save_state()

        return result

    def get_current_params(self) -> Dict[str, float]:
        """Retorna parámetros actuales."""
        return self.opt_state.params.copy()

    def get_statistics(self) -> Dict:
        """Retorna estadísticas del optimizador."""
        return {
            'n_optimizations': self.opt_state.n_optimizations,
            'n_improvements': self.opt_state.n_improvements,
            'improvement_rate': self.opt_state.n_improvements / max(1, self.opt_state.n_optimizations),
            'params': self.opt_state.params,
            'gradient': self.opt_state.gradient,
            'history_length': len(self.opt_state.param_history)
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Self-Optimizer Test")
    print("=" * 40)

    optimizer = SelfOptimizer()

    # Simular optimización
    class MockState:
        S = 0.5

    state = MockState()

    for i in range(50):
        z = np.random.randn(8)
        result = optimizer.optimize(state, z)

        # Simular que S cambia con parámetros
        state.S = 0.3 + 0.4 * optimizer.opt_state.params['stability_weight']
        state.S += np.random.randn() * 0.05

        if i % 10 == 0:
            print(f"  Step {i}: S={state.S:.3f}, action={result['action']}, "
                  f"improved={result['improved']}")

    print("\nFinal statistics:")
    stats = optimizer.get_statistics()
    print(f"  Optimizations: {stats['n_optimizations']}")
    print(f"  Improvements: {stats['n_improvements']}")
    print(f"  Rate: {stats['improvement_rate']:.2%}")
    print(f"  Params: {stats['params']}")
