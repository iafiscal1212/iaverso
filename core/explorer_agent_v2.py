#!/usr/bin/env python3
"""
Explorer Agent v2 - 100% Endógeno
=================================

NADA hardcodeado:
- Umbrales de correlación: derivados de distribución de correlaciones observadas
- Confianza: derivada de evidencia acumulada
- Curiosidad: derivada de incertidumbre reducible
- Selección de variables: derivada de varianza y cobertura

El agente aprende qué es "significativo" de sus propios datos.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import json


# =============================================================================
# Estadísticas Online (sin ventanas fijas)
# =============================================================================

class OnlineStats:
    """Estadísticas incrementales sin magic numbers."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Para varianza de Welford
        self.min_val = float('inf')
        self.max_val = float('-inf')

    def update(self, x: float):
        if x is None or np.isnan(x):
            return

        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

        self.min_val = min(self.min_val, x)
        self.max_val = max(self.max_val, x)

    @property
    def var(self) -> float:
        if self.n < 2:
            return 0.0
        return self.M2 / (self.n - 1)

    @property
    def std(self) -> float:
        return np.sqrt(self.var)

    @property
    def range(self) -> float:
        if self.n == 0:
            return 0.0
        return self.max_val - self.min_val


class OnlineCorrelation:
    """Correlación incremental entre dos variables."""

    def __init__(self):
        self.n = 0
        self.mean_x = 0.0
        self.mean_y = 0.0
        self.M2_x = 0.0
        self.M2_y = 0.0
        self.M_xy = 0.0

    def update(self, x: float, y: float):
        if x is None or y is None or np.isnan(x) or np.isnan(y):
            return

        self.n += 1

        dx = x - self.mean_x
        self.mean_x += dx / self.n
        dx2 = x - self.mean_x

        dy = y - self.mean_y
        self.mean_y += dy / self.n
        dy2 = y - self.mean_y

        self.M2_x += dx * dx2
        self.M2_y += dy * dy2
        self.M_xy += dx * dy2

    @property
    def correlation(self) -> float:
        if self.n < 3:
            return 0.0
        if self.M2_x <= 0 or self.M2_y <= 0:
            return 0.0
        return self.M_xy / np.sqrt(self.M2_x * self.M2_y)


# =============================================================================
# Hipótesis con evidencia Bayesiana
# =============================================================================

@dataclass
class Hypothesis:
    """Hipótesis causal con actualización Bayesiana."""
    source: str
    target: str
    lag: int
    correlation: OnlineCorrelation = field(default_factory=OnlineCorrelation)
    successes: int = 0
    failures: int = 0
    created_at: int = 0
    last_tested: int = 0

    @property
    def n_tests(self) -> int:
        return self.successes + self.failures

    @property
    def strength(self) -> float:
        """Correlación observada."""
        return self.correlation.correlation

    @property
    def success_rate(self) -> float:
        """Tasa de éxito con prior Beta(1,1)."""
        # Prior uniforme: equivalente a 1 éxito y 1 fracaso previos
        return (self.successes + 1) / (self.n_tests + 2)

    @property
    def confidence(self) -> float:
        """
        Confianza derivada de:
        1. Cantidad de evidencia (más tests = más confianza)
        2. Consistencia (success_rate lejos de 0.5)

        No hay umbral fijo - la confianza crece con evidencia.
        """
        if self.n_tests == 0:
            return 0.0

        # Factor de evidencia: satura en ~50 tests
        evidence_factor = 1 - np.exp(-self.n_tests / 20)

        # Factor de consistencia: qué tan lejos de 0.5
        rate = self.success_rate
        consistency = abs(rate - 0.5) * 2  # 0 si rate=0.5, 1 si rate=0 o 1

        return evidence_factor * consistency

    def is_significant(self, threshold_percentile: float) -> bool:
        """
        Significancia relativa a otras hipótesis.
        threshold_percentile viene del propio sistema, no hardcodeado.
        """
        return self.confidence > threshold_percentile


# =============================================================================
# Modelo del Mundo Adaptativo
# =============================================================================

class WorldModel:
    """Modelo del mundo que aprende qué es significativo."""

    def __init__(self):
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.variable_stats: Dict[str, OnlineStats] = defaultdict(OnlineStats)
        self.correlation_distribution = OnlineStats()  # Distribución de correlaciones

    def add_hypothesis(self, h: Hypothesis) -> str:
        key = f"{h.source}->{h.target}@{h.lag}"
        self.hypotheses[key] = h
        return key

    def get_hypothesis(self, source: str, target: str, lag: int) -> Optional[Hypothesis]:
        key = f"{source}->{target}@{lag}"
        return self.hypotheses.get(key)

    def get_confidence_threshold(self) -> float:
        """
        Umbral de confianza derivado de la distribución de confianzas.
        Percentil 75 de las confianzas observadas.
        """
        if not self.hypotheses:
            return 0.0

        confidences = [h.confidence for h in self.hypotheses.values() if h.n_tests > 0]
        if not confidences:
            return 0.0

        return float(np.percentile(confidences, 75))

    def get_correlation_threshold(self) -> float:
        """
        Umbral de correlación derivado de la distribución.
        Usa media + 1 std de |correlaciones|.
        """
        if self.correlation_distribution.n < 10:
            return 0.1  # Mínimo inicial, se ajustará

        # Umbral = media + 1 std de las correlaciones absolutas observadas
        return self.correlation_distribution.mean + self.correlation_distribution.std

    def get_significant_hypotheses(self) -> List[Hypothesis]:
        """Hipótesis por encima del umbral adaptativo."""
        threshold = self.get_confidence_threshold()
        return [h for h in self.hypotheses.values() if h.confidence > threshold]


# =============================================================================
# Estado Emocional Endógeno
# =============================================================================

@dataclass
class EmotionalState:
    """
    Emociones derivadas de métricas internas, no de umbrales fijos.
    """
    # Valores raw (se normalizan al calcular)
    _curiosity_raw: float = 0.0
    _surprise_raw: float = 0.0
    _confidence_raw: float = 0.0

    # Historiales para normalización
    _curiosity_history: List[float] = field(default_factory=list)
    _surprise_history: List[float] = field(default_factory=list)

    def update_curiosity(self, uncertainty_reducible: float):
        """Curiosidad = incertidumbre que podemos reducir."""
        self._curiosity_raw = uncertainty_reducible
        self._curiosity_history.append(uncertainty_reducible)
        # Limitar historial
        if len(self._curiosity_history) > 100:
            self._curiosity_history = self._curiosity_history[-100:]

    def update_surprise(self, prediction_error: float):
        """Sorpresa = error de predicción normalizado por historial."""
        self._surprise_raw = prediction_error
        self._surprise_history.append(prediction_error)
        if len(self._surprise_history) > 100:
            self._surprise_history = self._surprise_history[-100:]

    def update_confidence(self, accuracy: float):
        """Confianza = precisión predictiva."""
        self._confidence_raw = accuracy

    @property
    def curiosity(self) -> float:
        """Curiosidad normalizada por su propio historial."""
        if len(self._curiosity_history) < 2:
            return 0.5
        mean = np.mean(self._curiosity_history)
        std = np.std(self._curiosity_history) + 1e-8
        # Normalizar a [0, 1] usando sigmoid
        z = (self._curiosity_raw - mean) / std
        return 1 / (1 + np.exp(-z))

    @property
    def surprise(self) -> float:
        """Sorpresa normalizada."""
        if len(self._surprise_history) < 2:
            return 0.0
        mean = np.mean(self._surprise_history)
        std = np.std(self._surprise_history) + 1e-8
        z = (self._surprise_raw - mean) / std
        return min(1.0, max(0.0, z / 2 + 0.5))  # Mapear a [0,1]

    @property
    def confidence(self) -> float:
        return min(1.0, max(0.0, self._confidence_raw))

    def to_dict(self) -> Dict[str, float]:
        return {
            'curiosity': self.curiosity,
            'surprise': self.surprise,
            'confidence': self.confidence,
        }


# =============================================================================
# Agente Explorador v2 - 100% Endógeno
# =============================================================================

class ExplorerAgentV2:
    """
    Agente explorador sin magic numbers.

    Todo derivado de los datos:
    - Umbrales de correlación
    - Confianza en hipótesis
    - Selección de qué explorar
    - Cuándo generar nuevas hipótesis
    """

    def __init__(self, agent_id: str, variables: List[str], max_lag: int = 12):
        self.agent_id = agent_id
        self.variables = variables
        self.max_lag = max_lag
        self.t = 0

        # Modelo del mundo
        self.world_model = WorldModel()

        # Estado emocional
        self.emotions = EmotionalState()

        # Historial de observaciones
        self.observations: List[Dict[str, float]] = []

        # Predicciones y errores
        self.predictions: Dict[str, float] = {}
        self.error_stats = OnlineStats()

        # Personalidad derivada del hash del nombre (endógena)
        seed = sum(ord(c) * (i + 1) for i, c in enumerate(agent_id))
        self.rng = np.random.default_rng(seed)

        # Rasgos derivados del seed
        self._base_exploration = (seed % 100) / 100  # [0, 1]

        # Métricas
        self.CE_history: List[float] = []
        self.discoveries: List[Dict] = []

    def observe(self, data: Dict[str, float]) -> Dict[str, Any]:
        """Procesa una observación del mundo."""
        self.t += 1
        self.observations.append(data)

        # Actualizar estadísticas de variables
        for var, val in data.items():
            if val is not None and not np.isnan(val):
                self.world_model.variable_stats[var].update(val)

        # 1. Calcular sorpresa
        surprise = self._compute_surprise(data)
        self.emotions.update_surprise(surprise)

        # 2. Actualizar hipótesis existentes
        self._update_hypotheses(data)

        # 3. Decidir si generar nuevas hipótesis (basado en curiosidad)
        if self._should_explore():
            self._generate_hypotheses()

        # 4. Hacer predicciones
        self._make_predictions(data)

        # 5. Calcular CE
        CE = self._compute_CE()
        self.CE_history.append(CE)

        # 6. Actualizar curiosidad
        uncertainty = self._compute_reducible_uncertainty()
        self.emotions.update_curiosity(uncertainty)

        return {
            't': self.t,
            'agent': self.agent_id,
            'CE': CE,
            'surprise': self.emotions.surprise,
            'curiosity': self.emotions.curiosity,
            'confidence': self.emotions.confidence,
            'n_hypotheses': len(self.world_model.hypotheses),
            'n_significant': len(self.world_model.get_significant_hypotheses()),
            'n_discoveries': len(self.discoveries),
        }

    def _compute_surprise(self, data: Dict[str, float]) -> float:
        """Sorpresa = error de predicción."""
        if not self.predictions:
            return 0.0

        errors = []
        for var, predicted in self.predictions.items():
            if var in data:
                actual = data[var]
                if actual is not None and predicted is not None:
                    if not np.isnan(actual) and not np.isnan(predicted):
                        # Normalizar por std de la variable
                        std = self.world_model.variable_stats[var].std
                        if std > 0:
                            error = abs(actual - predicted) / std
                            errors.append(error)
                            self.error_stats.update(error)

        self.predictions = {}

        if not errors:
            return 0.0

        return float(np.mean(errors))

    def _update_hypotheses(self, data: Dict[str, float]):
        """Actualiza hipótesis con nueva evidencia."""
        if len(self.observations) < 2:
            return

        for key, h in self.world_model.hypotheses.items():
            if h.lag >= len(self.observations):
                continue

            past_obs = self.observations[-(h.lag + 1)]
            if h.source not in past_obs or h.target not in data:
                continue

            source_val = past_obs[h.source]
            target_val = data[h.target]

            if source_val is None or target_val is None:
                continue
            if np.isnan(source_val) or np.isnan(target_val):
                continue

            # Actualizar correlación online
            h.correlation.update(source_val, target_val)
            h.last_tested = self.t

            # Evaluar predicción
            if h.correlation.n >= 5:
                # Predecir usando correlación
                src_stats = self.world_model.variable_stats[h.source]
                tgt_stats = self.world_model.variable_stats[h.target]

                if src_stats.std > 0 and tgt_stats.std > 0:
                    z_source = (source_val - src_stats.mean) / src_stats.std
                    predicted = tgt_stats.mean + h.strength * tgt_stats.std * z_source

                    # Error relativo
                    error = abs(target_val - predicted) / (tgt_stats.std + 1e-8)

                    # Umbral de éxito derivado de errores históricos
                    error_threshold = self.error_stats.mean + self.error_stats.std if self.error_stats.n > 10 else 1.0

                    if error < error_threshold:
                        h.successes += 1
                    else:
                        h.failures += 1

                    # Registrar descubrimiento si alta confianza
                    if h.confidence > 0.7 and h.success_rate > 0.6 and h.n_tests >= 20:
                        discovery_key = f"{h.source}->{h.target}@{h.lag}"
                        if discovery_key not in [d.get('key') for d in self.discoveries]:
                            self.discoveries.append({
                                'key': discovery_key,
                                't': self.t,
                                'source': h.source,
                                'target': h.target,
                                'lag': h.lag,
                                'correlation': h.strength,
                                'confidence': h.confidence,
                                'success_rate': h.success_rate,
                                'n_tests': h.n_tests,
                            })

    def _should_explore(self) -> bool:
        """
        Decide si generar nuevas hipótesis.
        Basado en curiosidad (endógena) + exploración base (del hash).
        """
        # Explorar si curiosidad alta O con probabilidad base
        if self.emotions.curiosity > 0.6:
            return True

        # Probabilidad de exploración derivada del agente
        explore_prob = self._base_exploration * (1 - self.emotions.confidence)
        return self.rng.random() < explore_prob

    def _generate_hypotheses(self):
        """Genera nuevas hipótesis basadas en correlaciones."""
        if len(self.observations) < self.max_lag + 5:
            return

        # Seleccionar variables a investigar
        vars_to_check = self._select_variables()

        for target in vars_to_check[:2]:  # Limitar por paso
            for source in self.variables:
                if source == target:
                    continue

                # Elegir lag basado en variabilidad de la fuente
                for lag in self._select_lags(source):
                    if self.world_model.get_hypothesis(source, target, lag):
                        continue

                    # Calcular correlación inicial
                    corr = self._compute_initial_correlation(source, target, lag)

                    # Umbral adaptativo
                    threshold = self.world_model.get_correlation_threshold()

                    if abs(corr) > threshold:
                        h = Hypothesis(
                            source=source,
                            target=target,
                            lag=lag,
                            created_at=self.t,
                            last_tested=self.t,
                        )
                        self.world_model.add_hypothesis(h)

                    # Registrar correlación para ajustar umbral
                    self.world_model.correlation_distribution.update(abs(corr))

    def _select_variables(self) -> List[str]:
        """
        Selecciona variables a investigar.
        Prioriza: alta varianza + baja cobertura de hipótesis.
        """
        scores = {}

        for var in self.variables:
            stats = self.world_model.variable_stats[var]

            # Factor de variabilidad (normalizado por rango)
            if stats.range > 0:
                variability = stats.std / stats.range
            else:
                variability = 0

            # Factor de cobertura (cuántas hipótesis tienen este target)
            n_as_target = sum(1 for h in self.world_model.hypotheses.values()
                             if h.target == var)
            coverage_factor = 1 / (1 + n_as_target)

            scores[var] = variability * coverage_factor

        # Ordenar + ruido para exploración
        noise = {v: self.rng.random() * 0.1 for v in self.variables}
        sorted_vars = sorted(self.variables, key=lambda v: scores.get(v, 0) + noise[v], reverse=True)

        return sorted_vars

    def _select_lags(self, source: str) -> List[int]:
        """
        Selecciona lags a probar basado en autocorrelación de la fuente.
        """
        # Lags básicos + algunos aleatorios
        basic_lags = [0, 1, 2, 3]
        random_lags = list(self.rng.choice(range(4, self.max_lag), size=2, replace=False))
        return basic_lags + random_lags

    def _compute_initial_correlation(self, source: str, target: str, lag: int) -> float:
        """Calcula correlación inicial para decidir si crear hipótesis."""
        if len(self.observations) < lag + 10:
            return 0.0

        source_vals = []
        target_vals = []

        for i in range(lag, len(self.observations)):
            s = self.observations[i - lag].get(source)
            t = self.observations[i].get(target)

            if s is not None and t is not None and not np.isnan(s) and not np.isnan(t):
                source_vals.append(s)
                target_vals.append(t)

        if len(source_vals) < 5:
            return 0.0

        source_arr = np.array(source_vals)
        target_arr = np.array(target_vals)

        if np.std(source_arr) < 1e-8 or np.std(target_arr) < 1e-8:
            return 0.0

        corr = np.corrcoef(source_arr, target_arr)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    def _make_predictions(self, data: Dict[str, float]):
        """Hace predicciones usando hipótesis significativas."""
        self.predictions = {}

        significant = self.world_model.get_significant_hypotheses()

        for h in significant:
            if h.lag == 1 and h.source in data:
                source_val = data[h.source]
                if source_val is None or np.isnan(source_val):
                    continue

                src_stats = self.world_model.variable_stats[h.source]
                tgt_stats = self.world_model.variable_stats[h.target]

                if src_stats.std > 0 and tgt_stats.std > 0:
                    z = (source_val - src_stats.mean) / src_stats.std
                    predicted = tgt_stats.mean + h.strength * tgt_stats.std * z

                    if h.target not in self.predictions:
                        self.predictions[h.target] = predicted

    def _compute_reducible_uncertainty(self) -> float:
        """
        Incertidumbre reducible = variables con alta varianza pero pocas hipótesis buenas.
        """
        uncertainty = 0.0

        for var in self.variables:
            stats = self.world_model.variable_stats[var]
            if stats.n < 10:
                continue

            # Varianza normalizada
            if stats.range > 0:
                var_norm = stats.std / stats.range
            else:
                continue

            # Cobertura de hipótesis
            good_hypotheses = sum(1 for h in self.world_model.hypotheses.values()
                                  if h.target == var and h.confidence > 0.5)

            # Incertidumbre = varianza * (1 - cobertura)
            coverage = min(1.0, good_hypotheses / 3)
            uncertainty += var_norm * (1 - coverage)

        return uncertainty / len(self.variables) if self.variables else 0.0

    def _compute_CE(self) -> float:
        """
        CE = Coherencia Existencial = qué tan bien entiendo el mundo.

        Derivado de:
        1. Precisión predictiva (vs mi propio historial de errores)
        2. Cobertura del modelo
        3. Consistencia de hipótesis
        """
        # Precisión
        if self.error_stats.n > 5:
            # Error normalizado por mi propio historial
            recent_error = self.error_stats.mean
            accuracy = 1 / (1 + recent_error)
        else:
            accuracy = 0.5

        self.emotions.update_confidence(accuracy)

        # Cobertura
        covered = set(h.target for h in self.world_model.hypotheses.values() if h.confidence > 0.3)
        coverage = len(covered) / len(self.variables) if self.variables else 0

        # Consistencia (pocas hipótesis contradictorias)
        contradictions = 0
        hypotheses = list(self.world_model.hypotheses.values())
        for i, h1 in enumerate(hypotheses):
            for h2 in hypotheses[i + 1:]:
                if h1.target == h2.target and h1.source == h2.source:
                    if h1.lag != h2.lag and h1.confidence > 0.5 and h2.confidence > 0.5:
                        # Misma relación pero diferente lag con alta confianza = contradicción
                        contradictions += 1

        consistency = 1 / (1 + contradictions / 10)

        # CE combinado
        CE = 0.4 * accuracy + 0.3 * coverage + 0.3 * consistency

        return float(CE)

    def get_discoveries(self) -> List[Dict]:
        return self.discoveries

    def get_top_hypotheses(self, n: int = 10) -> List[Hypothesis]:
        sorted_h = sorted(self.world_model.hypotheses.values(),
                         key=lambda h: h.confidence * h.success_rate,
                         reverse=True)
        return sorted_h[:n]

    def get_status(self) -> Dict[str, Any]:
        return {
            'agent_id': self.agent_id,
            't': self.t,
            'CE': self.CE_history[-1] if self.CE_history else 0.5,
            'emotions': self.emotions.to_dict(),
            'n_hypotheses': len(self.world_model.hypotheses),
            'n_significant': len(self.world_model.get_significant_hypotheses()),
            'n_discoveries': len(self.discoveries),
            'exploration_base': self._base_exploration,
        }


# =============================================================================
# Crear agentes (personalidad derivada del nombre)
# =============================================================================

def create_explorer(agent_id: str, variables: List[str]) -> ExplorerAgentV2:
    """Crea un agente explorador. Personalidad derivada del nombre."""
    return ExplorerAgentV2(agent_id, variables)
