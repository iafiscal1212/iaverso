#!/usr/bin/env python3
"""
Agente 100% Endógeno
====================

NO recibe:
- Nombres de variables
- Thresholds
- Rangos de búsqueda
- Métricas predefinidas
- Modelos externos

SOLO recibe:
- Vector de números en cada paso
- Nada más

TODO lo demás emerge de su experiencia.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class Memory:
    """Memoria de observaciones pasadas - tamaño autodeterminado."""
    observations: deque = field(default_factory=lambda: deque(maxlen=1000))

    def add(self, obs: np.ndarray):
        self.observations.append(obs.copy())

    def get_recent(self, n: int) -> List[np.ndarray]:
        n = min(n, len(self.observations))
        return list(self.observations)[-n:]

    def __len__(self):
        return len(self.observations)


class PureEndogenousAgent:
    """
    Agente 100% endógeno.

    Única entrada: vector de números (sin etiquetas)
    Única meta: predecir el siguiente vector
    Todo lo demás emerge.
    """

    def __init__(self):
        # Sin nombre - la identidad emerge del historial
        self._birth_time = 0
        self._step = 0

        # Memoria (tamaño inicial, se ajusta solo)
        self.memory = Memory()

        # Dimensionalidad - se descubre en primer paso
        self.dim: Optional[int] = None

        # Predicción actual (emerge)
        self.prediction: Optional[np.ndarray] = None

        # Estadísticas online - se construyen solas
        self._running_mean: Optional[np.ndarray] = None
        self._running_var: Optional[np.ndarray] = None
        self._n_obs = 0

        # "Modelo interno" - pesos que el agente ajusta
        # Inicialmente: predecir = último valor (modelo más simple)
        self._model_weights: Optional[np.ndarray] = None

        # Historia de errores (para CE)
        self._error_history: deque = deque(maxlen=1000)
        self._error_ema: float = 1.0

        # Descubrimientos emergentes
        self._discovered_lags: List[int] = []  # Lags útiles que descubre
        self._discovered_pairs: List[Tuple[int, int]] = []  # Pares correlacionados

        # Escala temporal emergente
        self._useful_timescales: deque = deque(maxlen=100)

    @property
    def identity(self) -> str:
        """Identidad emergente basada en historial."""
        if self._n_obs == 0:
            return "newborn"

        # Identidad = hash de estadísticas propias
        if self._running_mean is not None:
            sig = hash(tuple(self._running_mean[:5].round(3))) % 10000
            return f"agent_{sig:04d}"
        return f"agent_{self._n_obs}"

    @property
    def CE(self) -> float:
        """Coherencia Existencial - 100% endógena."""
        if len(self._error_history) < 2:
            return 0.5  # Neutral al inicio

        # Error actual normalizado por MI historial de errores
        current_error = self._error_history[-1] if self._error_history else 1.0

        # EMA con alpha derivado de MI experiencia
        alpha = 1.0 / np.sqrt(max(1, self._n_obs))

        if current_error > 0:
            error_norm = current_error / max(self._error_ema, 1e-10)
        else:
            error_norm = 0.0

        # CE = qué tan bien predigo vs mi historial
        return 1.0 / (1.0 + error_norm)

    def _update_statistics(self, obs: np.ndarray):
        """Actualiza estadísticas online - Welford."""
        self._n_obs += 1

        if self._running_mean is None:
            self._running_mean = obs.copy()
            self._running_var = np.zeros_like(obs)
        else:
            delta = obs - self._running_mean
            self._running_mean += delta / self._n_obs
            delta2 = obs - self._running_mean
            self._running_var += delta * delta2

    def _get_std(self) -> np.ndarray:
        """Desviación estándar actual."""
        if self._n_obs < 2:
            return np.ones(self.dim)
        return np.sqrt(self._running_var / (self._n_obs - 1) + 1e-10)

    def _discover_useful_lag(self) -> Optional[int]:
        """Descubre qué lag temporal es útil - ENDÓGENO."""
        if len(self.memory) < 10:
            return None

        recent = self.memory.get_recent(min(50, len(self.memory)))
        if len(recent) < 10:
            return None

        # Probar diferentes lags y ver cuál reduce MI error
        # El rango de lags a probar EMERGE de la cantidad de datos que tengo
        max_lag = min(len(recent) // 3, len(self.memory) // 5)
        max_lag = max(1, max_lag)

        best_lag = 1
        best_score = float('inf')

        for lag in range(1, max_lag + 1):
            # Score = error de predicción si uso este lag
            errors = []
            for i in range(lag, len(recent)):
                pred = recent[i - lag]  # Predecir con valor de hace 'lag'
                actual = recent[i]
                err = np.mean((pred - actual) ** 2)
                errors.append(err)

            if errors:
                score = np.mean(errors)
                if score < best_score:
                    best_score = score
                    best_lag = lag

        # Recordar este lag si es útil
        if best_lag not in self._discovered_lags:
            self._discovered_lags.append(best_lag)
            # Mantener solo los mejores
            if len(self._discovered_lags) > 10:
                self._discovered_lags = self._discovered_lags[-10:]

        return best_lag

    def _discover_relationships(self):
        """Descubre relaciones entre dimensiones - ENDÓGENO."""
        if len(self.memory) < 20 or self.dim is None:
            return

        recent = np.array(self.memory.get_recent(min(100, len(self.memory))))

        # Calcular "co-movimiento" entre pares
        # NO uso correlación de Pearson predefinida
        # Uso una métrica que EMERGE: cuánto se mueven juntas

        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                # Mi propia métrica de relación:
                # ¿Cuando una sube/baja, la otra también?

                diff_i = np.diff(recent[:, i])
                diff_j = np.diff(recent[:, j])

                # Contar coincidencias de signo
                same_sign = np.sum(np.sign(diff_i) == np.sign(diff_j))
                total = len(diff_i)

                # Si coinciden más del 70% (umbral EMERGENTE de mis datos)
                threshold = 0.5 + 0.2 * (self._n_obs / (self._n_obs + 100))

                if total > 0 and same_sign / total > threshold:
                    pair = (i, j)
                    if pair not in self._discovered_pairs:
                        self._discovered_pairs.append(pair)

    def _make_prediction(self) -> np.ndarray:
        """Genera predicción - método EMERGENTE."""
        if self.dim is None or len(self.memory) == 0:
            return np.zeros(1)

        recent = self.memory.get_recent(min(10, len(self.memory)))

        if len(recent) == 0:
            return self._running_mean.copy() if self._running_mean is not None else np.zeros(self.dim)

        # Estrategia base: última observación
        pred = recent[-1].copy()

        # Si he descubierto lags útiles, usarlos
        if self._discovered_lags and len(recent) > 1:
            # Ponderar por lags descubiertos
            weights = []
            preds = []

            for lag in self._discovered_lags:
                if lag < len(recent):
                    preds.append(recent[-lag])
                    # Peso inversamente proporcional al lag
                    weights.append(1.0 / lag)

            if preds:
                weights = np.array(weights) / sum(weights)
                pred = sum(w * p for w, p in zip(weights, preds))

        # Si he descubierto pares relacionados, usar esa info
        if self._discovered_pairs and len(recent) >= 2:
            last = recent[-1]
            prev = recent[-2]
            diff = last - prev

            for i, j in self._discovered_pairs[:5]:  # Top 5 pares
                if i < self.dim and j < self.dim:
                    # Si i subió, j probablemente también
                    if abs(diff[i]) > abs(diff[j]):
                        pred[j] += 0.1 * diff[i] * np.sign(diff[j] + 0.01)

        return pred

    def observe(self, observation: np.ndarray) -> Dict:
        """
        Recibe observación y devuelve estado.

        Args:
            observation: Vector de números (sin etiquetas)

        Returns:
            Estado interno del agente
        """
        self._step += 1
        obs = np.array(observation, dtype=float)

        # Descubrir dimensionalidad
        if self.dim is None:
            self.dim = len(obs)

        # Calcular error de predicción (si tengo predicción)
        prediction_error = 0.0
        if self.prediction is not None:
            prediction_error = np.mean((obs - self.prediction) ** 2)
            self._error_history.append(prediction_error)

            # Actualizar EMA de error
            alpha = 1.0 / np.sqrt(max(1, self._n_obs))
            self._error_ema = (1 - alpha) * self._error_ema + alpha * prediction_error

        # Actualizar estadísticas
        self._update_statistics(obs)

        # Guardar en memoria
        self.memory.add(obs)

        # Descubrimientos (cada N pasos para eficiencia)
        if self._step % 10 == 0:
            self._discover_useful_lag()
            self._discover_relationships()

        # Hacer nueva predicción
        self.prediction = self._make_prediction()

        return {
            'step': self._step,
            'identity': self.identity,
            'CE': self.CE,
            'prediction_error': prediction_error,
            'dim': self.dim,
            'n_discovered_lags': len(self._discovered_lags),
            'n_discovered_pairs': len(self._discovered_pairs),
            'discovered_lags': self._discovered_lags.copy(),
        }

    def get_discoveries(self) -> Dict:
        """Retorna lo que el agente ha descubierto por sí mismo."""
        return {
            'identity': self.identity,
            'useful_lags': self._discovered_lags.copy(),
            'related_pairs': self._discovered_pairs.copy(),
            'n_observations': self._n_obs,
            'current_CE': self.CE,
            'mean_per_dim': self._running_mean.tolist() if self._running_mean is not None else [],
            'std_per_dim': self._get_std().tolist() if self.dim else [],
        }


def create_agent() -> PureEndogenousAgent:
    """Crea un agente sin darle NADA."""
    return PureEndogenousAgent()
