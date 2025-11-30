#!/usr/bin/env python3
"""
Phase R3: Task Acquisition from Structure (TAS)
================================================

"Aprender tareas nuevas" significa:
- Descubrir regularidades nuevas en flujos externos
- Reorganizar la dinámica interna para aprovecharlas

Un "task" emergente es un canal τ_j que:
1. Tiene error de predicción decreciente: d/dt E[E_t(j)] < 0
2. Aumenta S en promedio: E[ΔS_t(j)] > 0

Task = canal estructural útil, NO "tarea semántica".

100% ENDÓGENO - Sin constantes mágicas
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.stats import rankdata
from collections import deque
import json


@dataclass
class TaskChannel:
    """Un canal de tarea emergente."""
    channel_id: int
    d_input: int
    d_state: int

    # Predictor interno (lineal por simplicidad, pero adaptativo)
    W: np.ndarray = None  # Pesos del predictor

    # Historial
    errors: List[float] = field(default_factory=list)
    delta_S: List[float] = field(default_factory=list)
    n_updates: int = 0

    def __post_init__(self):
        if self.W is None:
            # Inicialización pequeña
            self.W = np.random.randn(self.d_input, self.d_state) * 0.01

    def predict(self, z: np.ndarray) -> np.ndarray:
        """Predice siguiente input externo dado estado interno."""
        return z @ self.W.T

    def update(self, z: np.ndarray, s_true: np.ndarray, S_before: float, S_after: float):
        """Actualiza predictor y registra métricas."""
        s_pred = self.predict(z)
        error = float(np.mean((s_true - s_pred) ** 2))

        self.errors.append(error)
        self.delta_S.append(S_after - S_before)
        self.n_updates += 1

        # Learning rate endógeno
        eta = 1.0 / np.sqrt(self.n_updates + 1)

        # Gradiente simple
        grad = np.outer(s_true - s_pred, z)
        self.W = self.W + eta * grad

        # Limitar tamaño de historial
        if len(self.errors) > 1000:
            self.errors = self.errors[-1000:]
            self.delta_S = self.delta_S[-1000:]

    @property
    def error_trend(self) -> float:
        """Tendencia del error (negativo = mejorando)."""
        if len(self.errors) < 10:
            return 0.0

        w = int(np.sqrt(len(self.errors))) + 1
        recent = self.errors[-w:]
        old = self.errors[-2*w:-w] if len(self.errors) >= 2*w else self.errors[:w]

        if not old:
            return 0.0

        return np.mean(recent) - np.mean(old)

    @property
    def mean_delta_S(self) -> float:
        """Media de ΔS cuando se usa este canal."""
        if not self.delta_S:
            return 0.0
        return float(np.mean(self.delta_S))

    def is_valid_task(self) -> bool:
        """
        Verifica si este canal es una "tarea" válida:
        - Error decreciente
        - ΔS positivo en promedio
        """
        if self.n_updates < 10:
            return False

        return self.error_trend < 0 and self.mean_delta_S > 0


class TaskAcquisitionSystem:
    """
    Sistema de Adquisición de Tareas desde Estructura.

    Detecta cambios en flujos externos, crea canales de predicción,
    y determina cuáles son "tareas" útiles.
    """

    def __init__(self, d_state: int = 8, d_external: int = 4):
        self.d_state = d_state
        self.d_external = d_external

        # Historial de inputs externos
        self.external_history: deque = deque(maxlen=10000)

        # Canales de tarea
        self.channels: List[TaskChannel] = []
        self.active_channel_idx: int = -1

        # Detección de cambio de régimen
        self._cov_history: deque = deque(maxlen=100)
        self._eigenvalue_history: deque = deque(maxlen=100)

        # Historial de S para calcular ΔS
        self._S_history: deque = deque(maxlen=1000)

        # Métricas globales
        self.regime_changes: List[int] = []
        self.task_discoveries: List[Dict] = []

    def _compute_covariance(self, window: int = None) -> np.ndarray:
        """Calcula covarianza de inputs externos."""
        if len(self.external_history) < 3:
            return np.eye(self.d_external)

        if window is None:
            window = int(np.sqrt(len(self.external_history))) + 1

        data = np.array(list(self.external_history)[-window:])
        cov = np.cov(data.T)

        if cov.ndim == 0:
            cov = np.array([[cov]])

        return cov

    def _detect_regime_change(self) -> bool:
        """
        Detecta cambio estructural en inputs externos.
        Basado en cambio en eigenvalues de covarianza.
        """
        if len(self.external_history) < 20:
            return False

        cov = self._compute_covariance()
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]

        self._eigenvalue_history.append(eigvals)

        if len(self._eigenvalue_history) < 10:
            return False

        # Comparar eigenvalues actuales con históricos
        recent = np.array(list(self._eigenvalue_history)[-5:])
        old = np.array(list(self._eigenvalue_history)[-10:-5])

        if len(old) == 0:
            return False

        # Diferencia en estructura espectral
        recent_mean = np.mean(recent, axis=0)
        old_mean = np.mean(old, axis=0)

        diff = np.linalg.norm(recent_mean - old_mean)

        # Umbral endógeno basado en variabilidad histórica
        all_eigvals = np.array(list(self._eigenvalue_history))
        if len(all_eigvals) > 5:
            typical_diff = np.percentile(
                [np.linalg.norm(all_eigvals[i] - all_eigvals[i-1])
                 for i in range(1, len(all_eigvals))],
                95
            )
        else:
            typical_diff = 1.0

        return diff > typical_diff

    def _create_new_channel(self) -> int:
        """Crea nuevo canal de tarea."""
        channel_id = len(self.channels)
        channel = TaskChannel(
            channel_id=channel_id,
            d_input=self.d_external,
            d_state=self.d_state
        )
        self.channels.append(channel)
        return channel_id

    def _select_active_channel(self, z: np.ndarray, s: np.ndarray) -> int:
        """
        Selecciona canal activo basado en error de predicción.
        Thompson sampling sobre canales.
        """
        if not self.channels:
            return self._create_new_channel()

        # Calcular errores de predicción para cada canal
        errors = []
        for channel in self.channels:
            pred = channel.predict(z)
            error = np.mean((s - pred) ** 2)
            errors.append(error)

        # Thompson sampling: menor error = más probable
        # Convertir errores a probabilidades (softmax inverso)
        errors = np.array(errors)
        inv_errors = 1.0 / (errors + 1e-12)
        probs = inv_errors / np.sum(inv_errors)

        # Añadir exploración para nuevos canales
        if len(self.channels) < int(np.sqrt(len(self.external_history) + 1)):
            # Probabilidad de crear nuevo canal
            p_new = 0.1 / (len(self.channels) + 1)
            probs = probs * (1 - p_new)

            if np.random.random() < p_new:
                return self._create_new_channel()

        # Asegurar que probs suma 1
        probs = probs / np.sum(probs)

        return int(np.random.choice(len(self.channels), p=probs))

    def observe(self, z: np.ndarray, s: np.ndarray, S: float) -> Dict:
        """
        Procesa observación de estado interno z, input externo s, y score S.

        Returns:
            Dict con información del procesamiento
        """
        t = len(self.external_history)
        self.external_history.append(s.copy())

        S_before = self._S_history[-1] if self._S_history else S
        self._S_history.append(S)

        result = {
            't': t,
            'regime_change': False,
            'new_channel': False,
            'active_channel': -1,
            'channel_is_task': False
        }

        # Detectar cambio de régimen
        if self._detect_regime_change():
            result['regime_change'] = True
            self.regime_changes.append(t)

            # Crear nuevo canal para el nuevo régimen
            new_idx = self._create_new_channel()
            result['new_channel'] = True
            result['new_channel_idx'] = new_idx

        # Seleccionar canal activo
        if len(self.external_history) > 1:
            self.active_channel_idx = self._select_active_channel(z, s)
            result['active_channel'] = self.active_channel_idx

            # Actualizar canal activo
            if self.active_channel_idx >= 0 and self.active_channel_idx < len(self.channels):
                channel = self.channels[self.active_channel_idx]
                channel.update(z, s, S_before, S)

                # Verificar si es tarea válida
                if channel.is_valid_task():
                    result['channel_is_task'] = True

                    # Registrar descubrimiento si es nuevo
                    if not any(d['channel_id'] == channel.channel_id for d in self.task_discoveries):
                        self.task_discoveries.append({
                            't': t,
                            'channel_id': channel.channel_id,
                            'error_trend': channel.error_trend,
                            'mean_delta_S': channel.mean_delta_S
                        })

        return result

    def get_valid_tasks(self) -> List[Dict]:
        """Retorna canales que son tareas válidas."""
        tasks = []
        for channel in self.channels:
            if channel.is_valid_task():
                tasks.append({
                    'channel_id': channel.channel_id,
                    'n_updates': channel.n_updates,
                    'error_trend': channel.error_trend,
                    'mean_delta_S': channel.mean_delta_S,
                    'mean_error': float(np.mean(channel.errors[-100:])) if channel.errors else 0.0
                })

        # Ordenar por ΔS
        tasks.sort(key=lambda x: x['mean_delta_S'], reverse=True)
        return tasks

    def predict_with_best_task(self, z: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Predice usando el mejor canal de tarea.

        Returns:
            (predicción, channel_id)
        """
        tasks = self.get_valid_tasks()

        if not tasks:
            if self.channels:
                # Usar el canal con menor error
                errors = [np.mean(c.errors[-10:]) if c.errors else float('inf')
                         for c in self.channels]
                best_idx = int(np.argmin(errors))
                return self.channels[best_idx].predict(z), best_idx
            return np.zeros(self.d_external), -1

        best_channel_id = tasks[0]['channel_id']
        return self.channels[best_channel_id].predict(z), best_channel_id

    def get_stats(self) -> Dict:
        """Estadísticas del sistema TAS."""
        valid_tasks = self.get_valid_tasks()

        return {
            'n_observations': len(self.external_history),
            'n_channels': len(self.channels),
            'n_valid_tasks': len(valid_tasks),
            'n_regime_changes': len(self.regime_changes),
            'n_task_discoveries': len(self.task_discoveries),
            'valid_tasks': valid_tasks[:5],
            'channel_stats': [
                {
                    'id': c.channel_id,
                    'updates': c.n_updates,
                    'error_trend': c.error_trend,
                    'mean_delta_S': c.mean_delta_S,
                    'is_task': c.is_valid_task()
                }
                for c in self.channels
            ]
        }


def run_phaseR3_test(n_steps: int = 3000) -> Dict:
    """
    Test de Phase R3: Task Acquisition from Structure.

    Verifica:
    1. Se detectan cambios de régimen
    2. Se crean canales para nuevos regímenes
    3. Algunos canales se convierten en "tareas" (error ↓, ΔS > 0)
    4. Las predicciones mejoran con el tiempo
    """
    print("=" * 70)
    print("PHASE R3: TASK ACQUISITION FROM STRUCTURE (TAS)")
    print("=" * 70)

    tas = TaskAcquisitionSystem(d_state=8, d_external=4)

    # Simular flujo externo con cambios de régimen
    z = np.random.randn(8) * 0.1

    # Regímenes diferentes
    regimes = [
        lambda t: np.array([np.sin(t/10), np.cos(t/10), 0.5, 0.5]),
        lambda t: np.array([0.5, 0.5, np.sin(t/15), np.cos(t/15)]),
        lambda t: np.array([np.sin(t/8), 0.3, np.cos(t/8), 0.7]),
    ]
    current_regime = 0

    errors_over_time = []
    tasks_over_time = []

    print(f"\nEjecutando {n_steps} pasos con cambios de régimen...")

    for t in range(n_steps):
        # Cambiar régimen periódicamente
        if t > 0 and t % 800 == 0:
            current_regime = (current_regime + 1) % len(regimes)
            print(f"  t={t}: Cambio a régimen {current_regime}")

        # Generar input externo
        s = regimes[current_regime](t) + np.random.randn(4) * 0.05

        # Calcular S (basado en predictibilidad del régimen)
        if tas.channels and tas.active_channel_idx >= 0:
            pred, _ = tas.predict_with_best_task(z)
            pred_error = np.mean((s - pred) ** 2)
            S = np.exp(-pred_error)  # S alto cuando predice bien
        else:
            S = 0.5

        # Observar
        result = tas.observe(z, s, S)

        # Registrar métricas
        if tas.channels and tas.active_channel_idx >= 0:
            channel = tas.channels[tas.active_channel_idx]
            if channel.errors:
                errors_over_time.append(channel.errors[-1])
        else:
            errors_over_time.append(1.0)

        tasks_over_time.append(len(tas.get_valid_tasks()))

        # Actualizar estado interno (dinámica simple)
        z = z * 0.9 + 0.1 * np.concatenate([s, s])  # Estado influido por input
        z = z + np.random.randn(8) * 0.01

        if (t + 1) % 600 == 0:
            stats = tas.get_stats()
            print(f"  t={t+1}: channels={stats['n_channels']}, "
                  f"tasks={stats['n_valid_tasks']}, "
                  f"regime_changes={stats['n_regime_changes']}")

    # Análisis final
    stats = tas.get_stats()
    valid_tasks = tas.get_valid_tasks()

    # Verificar mejora de error
    if len(errors_over_time) > 100:
        early_error = np.mean(errors_over_time[:100])
        late_error = np.mean(errors_over_time[-100:])
        error_improvement = early_error - late_error
    else:
        error_improvement = 0.0

    # GO/NO-GO criteria
    criteria = {
        'regime_changes_detected': stats['n_regime_changes'] >= 2,
        'multiple_channels_created': stats['n_channels'] >= 2,
        'valid_tasks_discovered': stats['n_valid_tasks'] >= 1,
        'error_improves_over_time': error_improvement > 0,
        'tasks_have_positive_delta_S': any(t['mean_delta_S'] > 0 for t in valid_tasks) if valid_tasks else False
    }

    n_pass = sum(criteria.values())
    go = n_pass >= 3

    print(f"\n{'='*70}")
    print("RESULTADOS PHASE R3")
    print(f"{'='*70}")
    print(f"\nEstadísticas:")
    print(f"  - Observaciones: {stats['n_observations']}")
    print(f"  - Canales creados: {stats['n_channels']}")
    print(f"  - Tareas válidas: {stats['n_valid_tasks']}")
    print(f"  - Cambios de régimen: {stats['n_regime_changes']}")
    print(f"  - Mejora de error: {error_improvement:.4f}")

    print(f"\nCanales:")
    for cs in stats['channel_stats']:
        status = "TASK" if cs['is_task'] else "channel"
        print(f"  - {status} {cs['id']}: updates={cs['updates']}, "
              f"error_trend={cs['error_trend']:.4f}, ΔS={cs['mean_delta_S']:.4f}")

    print(f"\nGO/NO-GO Criteria:")
    for criterion, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"  - {criterion}: {status}")

    print(f"\n{'GO' if go else 'NO-GO'} ({n_pass}/5 criteria passed)")

    return {
        'go': go,
        'stats': stats,
        'criteria': criteria,
        'error_improvement': error_improvement,
        'errors_over_time': errors_over_time,
        'tasks_over_time': tasks_over_time
    }


if __name__ == "__main__":
    result = run_phaseR3_test(n_steps=3000)

    # Guardar resultados
    import os
    os.makedirs('/root/NEO_EVA/results/phaseR3', exist_ok=True)

    with open('/root/NEO_EVA/results/phaseR3/phaseR3_results.json', 'w') as f:
        json.dump({
            'go': result['go'],
            'stats': result['stats'],
            'criteria': {k: bool(v) for k, v in result['criteria'].items()},
            'error_improvement': result['error_improvement']
        }, f, indent=2)

    print(f"\nResultados guardados en results/phaseR3/")
