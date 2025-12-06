#!/usr/bin/env python3
"""
Explorer Agent - Agente que Descubre Estructura en el Mundo Real
================================================================

A diferencia del agente anterior (que se predecía a sí mismo),
este agente:

1. OBSERVA el mundo externo (datos reales)
2. GENERA hipótesis ("¿A causa B con lag k?")
3. TESTEA las hipótesis contra datos
4. ACTUALIZA su modelo del mundo basado en sorpresa
5. ELIGE dónde explorar basado en curiosidad intrínseca

CE aquí significa: "¿Qué tan bien entiendo el mundo?"
No "¿Qué tan bien me predigo?"

Curiosidad = buscar donde hay máxima incertidumbre REDUCIBLE
(no donde ya sé, no donde es ruido puro)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import json


# =============================================================================
# Hipótesis Causal
# =============================================================================

@dataclass
class Hypothesis:
    """
    Una hipótesis causal: "X causa Y con lag k"

    Puede ser:
    - Correlación simple
    - Causalidad con delay
    - Relación no lineal
    """
    source: str           # Variable fuente
    target: str           # Variable objetivo
    lag: int             # Delay temporal (0 = simultáneo)
    strength: float      # Fuerza estimada de la relación
    confidence: float    # Confianza en la hipótesis (0-1)
    evidence_for: int    # Veces que se confirmó
    evidence_against: int  # Veces que se refutó
    created_at: int      # Paso en que se creó
    last_tested: int     # Último paso en que se testeó

    @property
    def total_tests(self) -> int:
        return self.evidence_for + self.evidence_against

    @property
    def success_rate(self) -> float:
        if self.total_tests == 0:
            return 0.5  # Prior uniforme
        return self.evidence_for / self.total_tests

    def update_confidence(self):
        """Actualiza confianza basado en evidencia."""
        # Bayesian update simplificado
        # Más evidencia = más certeza (en cualquier dirección)
        n = self.total_tests
        if n == 0:
            self.confidence = 0.5
        else:
            # Confianza = qué tan lejos estamos de 0.5
            rate = self.success_rate
            self.confidence = abs(rate - 0.5) * 2 * min(1, n / 10)


@dataclass
class WorldModel:
    """
    Modelo del mundo que el agente construye.

    Contiene:
    - Hipótesis activas
    - Estadísticas de variables
    - Grafo causal inferido
    """
    hypotheses: Dict[str, Hypothesis] = field(default_factory=dict)
    variable_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    causal_graph: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    def add_hypothesis(self, h: Hypothesis):
        key = f"{h.source}->{h.target}@{h.lag}"
        self.hypotheses[key] = h

    def get_hypothesis(self, source: str, target: str, lag: int) -> Optional[Hypothesis]:
        key = f"{source}->{target}@{lag}"
        return self.hypotheses.get(key)

    def get_confident_hypotheses(self, min_confidence: float = 0.7) -> List[Hypothesis]:
        """Retorna hipótesis con alta confianza."""
        return [h for h in self.hypotheses.values() if h.confidence >= min_confidence]


# =============================================================================
# Emociones como Señales de Control
# =============================================================================

class Emotion(Enum):
    """
    Emociones = señales internas que guían exploración.
    No son "sentimientos" sino moduladores de comportamiento.
    """
    CURIOSITY = "curiosity"      # Alta incertidumbre reducible
    SURPRISE = "surprise"        # Predicción violada
    CONFIDENCE = "confidence"    # Modelo funciona bien
    CONFUSION = "confusion"      # Muchas hipótesis contradictorias
    BOREDOM = "boredom"          # Todo es predecible, nada nuevo


@dataclass
class EmotionalState:
    """Estado emocional del agente."""
    curiosity: float = 0.5
    surprise: float = 0.0
    confidence: float = 0.5
    confusion: float = 0.0
    boredom: float = 0.0

    def dominant_emotion(self) -> Emotion:
        """Retorna la emoción dominante."""
        emotions = {
            Emotion.CURIOSITY: self.curiosity,
            Emotion.SURPRISE: self.surprise,
            Emotion.CONFIDENCE: self.confidence,
            Emotion.CONFUSION: self.confusion,
            Emotion.BOREDOM: self.boredom,
        }
        return max(emotions, key=emotions.get)

    def to_dict(self) -> Dict[str, float]:
        return {
            'curiosity': self.curiosity,
            'surprise': self.surprise,
            'confidence': self.confidence,
            'confusion': self.confusion,
            'boredom': self.boredom,
        }


# =============================================================================
# Agente Explorador
# =============================================================================

class ExplorerAgent:
    """
    Agente que descubre estructura causal en datos del mundo real.

    Ciclo:
    1. Observar nuevos datos
    2. Calcular sorpresa (¿predije bien?)
    3. Actualizar hipótesis existentes
    4. Generar nuevas hipótesis si hay curiosidad
    5. Elegir qué investigar (basado en curiosidad)
    6. Actualizar estado emocional
    """

    def __init__(self, agent_id: str, variables: List[str],
                 max_lag: int = 12, personality: Dict[str, float] = None):
        """
        Parameters
        ----------
        agent_id : str
            Identificador único
        variables : List[str]
            Variables del mundo que puede observar
        max_lag : int
            Máximo lag a considerar en hipótesis causales
        personality : Dict[str, float]
            Rasgos de personalidad que afectan exploración
        """
        self.agent_id = agent_id
        self.variables = variables
        self.max_lag = max_lag
        self.t = 0

        # Personalidad (afecta cómo explora)
        self.personality = personality or {
            'curiosity_base': 0.5,      # Tendencia a explorar
            'risk_tolerance': 0.5,       # Disposición a hipótesis arriesgadas
            'patience': 0.5,             # Cuánto espera antes de descartar
            'domain_preference': None,   # Dominio preferido (None = todos)
        }

        # Modelo del mundo
        self.world_model = WorldModel()

        # Estado emocional
        self.emotions = EmotionalState()

        # Historial de observaciones
        self.observation_history: List[Dict[str, float]] = []

        # Predicciones pendientes
        self.pending_predictions: Dict[str, Dict[str, float]] = {}

        # Métricas
        self.prediction_errors: List[float] = []
        self.discoveries: List[Dict] = []  # Hipótesis confirmadas

        # CE = Coherencia Existencial = qué tan bien entiendo el mundo
        self.CE_history: List[float] = []

    def observe(self, data: Dict[str, float]) -> Dict[str, Any]:
        """
        Observa nuevos datos del mundo.

        Returns
        -------
        Dict con métricas del paso
        """
        self.t += 1
        self.observation_history.append(data)

        # 1. Calcular sorpresa (si había predicciones)
        surprise = self._compute_surprise(data)

        # 2. Actualizar hipótesis existentes
        self._update_hypotheses(data)

        # 3. Generar nuevas hipótesis si hay curiosidad
        if self.emotions.curiosity > 0.5:
            self._generate_hypotheses()

        # 4. Hacer predicciones para el futuro
        self._make_predictions(data)

        # 5. Actualizar estado emocional
        self._update_emotions(surprise)

        # 6. Calcular CE
        CE = self._compute_CE()
        self.CE_history.append(CE)

        return {
            't': self.t,
            'agent': self.agent_id,
            'CE': CE,
            'surprise': surprise,
            'emotions': self.emotions.to_dict(),
            'dominant_emotion': self.emotions.dominant_emotion().value,
            'n_hypotheses': len(self.world_model.hypotheses),
            'n_confident': len(self.world_model.get_confident_hypotheses()),
        }

    def _compute_surprise(self, data: Dict[str, float]) -> float:
        """
        Calcula sorpresa = diferencia entre predicción y realidad.
        """
        if not self.pending_predictions:
            return 0.0

        errors = []
        for var, pred in self.pending_predictions.items():
            if var in data and pred.get('value') is not None:
                actual = data[var]
                predicted = pred['value']
                # Normalizar por std histórica de la variable
                std = self._get_variable_std(var)
                error = abs(actual - predicted) / (std + 1e-8)
                errors.append(error)
                self.prediction_errors.append(error)

        self.pending_predictions = {}

        if not errors:
            return 0.0

        # Sorpresa = error medio normalizado
        return float(np.mean(errors))

    def _get_variable_std(self, var: str) -> float:
        """Obtiene std histórica de una variable."""
        if var in self.world_model.variable_stats:
            return self.world_model.variable_stats[var].get('std', 1.0)

        # Calcular de historial
        if len(self.observation_history) > 10:
            values = [obs.get(var) for obs in self.observation_history if var in obs]
            if values:
                std = float(np.std(values))
                if var not in self.world_model.variable_stats:
                    self.world_model.variable_stats[var] = {}
                self.world_model.variable_stats[var]['std'] = std
                self.world_model.variable_stats[var]['mean'] = float(np.mean(values))
                return std
        return 1.0

    def _update_hypotheses(self, data: Dict[str, float]):
        """
        Actualiza hipótesis existentes con nueva evidencia.
        """
        if len(self.observation_history) < 2:
            return

        for key, h in self.world_model.hypotheses.items():
            # Obtener valor pasado de source
            if h.lag >= len(self.observation_history):
                continue

            past_obs = self.observation_history[-(h.lag + 1)]
            if h.source not in past_obs or h.target not in data:
                continue

            source_val = past_obs[h.source]
            target_val = data[h.target]

            # Predecir target basado en hipótesis
            # Predicción simple: target_pred = mean + strength * (source - mean_source)
            source_mean = self.world_model.variable_stats.get(h.source, {}).get('mean', 0)
            target_mean = self.world_model.variable_stats.get(h.target, {}).get('mean', 0)
            target_std = self._get_variable_std(h.target)

            predicted = target_mean + h.strength * (source_val - source_mean)
            error = abs(target_val - predicted) / (target_std + 1e-8)

            # Actualizar evidencia
            h.last_tested = self.t
            if error < 1.0:  # Predicción razonable
                h.evidence_for += 1
            else:
                h.evidence_against += 1

            h.update_confidence()

            # Si alta confianza y alta tasa de éxito, es un descubrimiento
            if h.confidence > 0.8 and h.success_rate > 0.7 and h.total_tests >= 10:
                if key not in [d.get('key') for d in self.discoveries]:
                    self.discoveries.append({
                        'key': key,
                        't': self.t,
                        'hypothesis': f"{h.source} -> {h.target} (lag={h.lag})",
                        'confidence': h.confidence,
                        'success_rate': h.success_rate,
                    })

    def _generate_hypotheses(self):
        """
        Genera nuevas hipótesis basadas en correlaciones observadas.
        """
        if len(self.observation_history) < self.max_lag + 5:
            return

        # Elegir variables a investigar basado en personalidad
        vars_to_check = self._select_variables_to_investigate()

        for target in vars_to_check[:3]:  # Limitar para no explotar
            for source in self.variables:
                if source == target:
                    continue

                for lag in range(0, min(self.max_lag, len(self.observation_history) - 1)):
                    # Ya existe?
                    if self.world_model.get_hypothesis(source, target, lag):
                        continue

                    # Calcular correlación con lag
                    corr = self._compute_lagged_correlation(source, target, lag)

                    if abs(corr) > 0.3:  # Umbral mínimo
                        h = Hypothesis(
                            source=source,
                            target=target,
                            lag=lag,
                            strength=corr,
                            confidence=0.3,  # Baja inicialmente
                            evidence_for=0,
                            evidence_against=0,
                            created_at=self.t,
                            last_tested=self.t,
                        )
                        self.world_model.add_hypothesis(h)

    def _compute_lagged_correlation(self, source: str, target: str, lag: int) -> float:
        """Calcula correlación entre source(t-lag) y target(t)."""
        if len(self.observation_history) < lag + 10:
            return 0.0

        source_vals = []
        target_vals = []

        for i in range(lag, len(self.observation_history)):
            obs_now = self.observation_history[i]
            obs_past = self.observation_history[i - lag]

            if source in obs_past and target in obs_now:
                s_val = obs_past[source]
                t_val = obs_now[target]
                if s_val is not None and t_val is not None and not np.isnan(s_val) and not np.isnan(t_val):
                    source_vals.append(s_val)
                    target_vals.append(t_val)

        if len(source_vals) < 5:
            return 0.0

        # Correlación de Pearson
        source_arr = np.array(source_vals)
        target_arr = np.array(target_vals)

        if np.std(source_arr) < 1e-8 or np.std(target_arr) < 1e-8:
            return 0.0

        corr = np.corrcoef(source_arr, target_arr)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    def _select_variables_to_investigate(self) -> List[str]:
        """
        Elige qué variables investigar basado en curiosidad.

        Curiosidad máxima donde:
        - Hay variabilidad (no constante)
        - No tenemos buenas hipótesis aún
        - Está en nuestro dominio de interés
        """
        scores = {}

        for var in self.variables:
            # Factor 1: Variabilidad
            std = self._get_variable_std(var)
            variability_score = min(1.0, std / (std + 0.1))

            # Factor 2: Incertidumbre (pocas hipótesis buenas sobre esta variable)
            related_hypotheses = [h for h in self.world_model.hypotheses.values()
                                  if h.target == var and h.confidence > 0.5]
            uncertainty_score = 1.0 / (1.0 + len(related_hypotheses))

            # Factor 3: Preferencia de dominio
            domain_score = 1.0
            pref = self.personality.get('domain_preference')
            if pref:
                domain_score = 1.5 if pref in var else 0.5

            # Score total
            scores[var] = variability_score * uncertainty_score * domain_score

        # Ordenar por score + algo de aleatoriedad (exploración)
        noise = np.random.random(len(scores)) * 0.2 * self.personality['curiosity_base']
        sorted_vars = sorted(scores.keys(),
                            key=lambda v: scores[v] + noise[list(scores.keys()).index(v)],
                            reverse=True)

        return sorted_vars

    def _make_predictions(self, data: Dict[str, float]):
        """
        Hace predicciones para el siguiente paso basado en hipótesis.
        """
        self.pending_predictions = {}

        confident_hypotheses = self.world_model.get_confident_hypotheses(0.5)

        for h in confident_hypotheses:
            if h.lag == 1 and h.source in data:
                source_val = data[h.source]
                source_mean = self.world_model.variable_stats.get(h.source, {}).get('mean', 0)
                target_mean = self.world_model.variable_stats.get(h.target, {}).get('mean', 0)

                predicted = target_mean + h.strength * (source_val - source_mean)

                if h.target not in self.pending_predictions:
                    self.pending_predictions[h.target] = {
                        'value': predicted,
                        'hypothesis': f"{h.source}->{h.target}",
                        'confidence': h.confidence,
                    }

    def _update_emotions(self, surprise: float):
        """
        Actualiza estado emocional basado en lo observado.
        """
        # Sorpresa directa
        self.emotions.surprise = min(1.0, surprise)

        # Confianza = inversa del error medio reciente
        if len(self.prediction_errors) > 5:
            recent_error = np.mean(self.prediction_errors[-10:])
            self.emotions.confidence = 1.0 / (1.0 + recent_error)

        # Curiosidad = alta si hay incertidumbre reducible
        n_uncertain = sum(1 for h in self.world_model.hypotheses.values()
                         if 0.3 < h.confidence < 0.7)
        n_total = len(self.world_model.hypotheses) + 1
        base_curiosity = self.personality['curiosity_base']
        self.emotions.curiosity = base_curiosity + 0.3 * (n_uncertain / n_total)
        self.emotions.curiosity = min(1.0, self.emotions.curiosity)

        # Confusión = muchas hipótesis contradictorias
        contradictions = 0
        for h1 in self.world_model.hypotheses.values():
            for h2 in self.world_model.hypotheses.values():
                if (h1.source == h2.source and h1.target == h2.target and
                    h1.lag != h2.lag and h1.confidence > 0.5 and h2.confidence > 0.5):
                    contradictions += 1
        self.emotions.confusion = min(1.0, contradictions / 10)

        # Aburrimiento = todo predecible, nada nuevo
        if self.emotions.confidence > 0.8 and self.emotions.curiosity < 0.3:
            self.emotions.boredom = 0.5 + 0.5 * self.emotions.confidence
        else:
            self.emotions.boredom = max(0, self.emotions.boredom - 0.1)

    def _compute_CE(self) -> float:
        """
        Coherencia Existencial = qué tan bien entiendo el mundo.

        CE alto = predicciones acertadas + modelo coherente
        CE bajo = sorpresas frecuentes + confusión
        """
        # Factor 1: Precisión predictiva
        if len(self.prediction_errors) > 0:
            recent_errors = self.prediction_errors[-20:] if len(self.prediction_errors) > 20 else self.prediction_errors
            accuracy = 1.0 / (1.0 + np.mean(recent_errors))
        else:
            accuracy = 0.5

        # Factor 2: Coherencia del modelo (hipótesis no contradictorias)
        coherence = 1.0 - self.emotions.confusion

        # Factor 3: Cobertura (tenemos hipótesis sobre muchas variables)
        covered_targets = set(h.target for h in self.world_model.hypotheses.values() if h.confidence > 0.5)
        coverage = len(covered_targets) / len(self.variables) if self.variables else 0

        # CE = combinación ponderada
        CE = 0.5 * accuracy + 0.3 * coherence + 0.2 * coverage

        return float(CE)

    def get_discoveries(self) -> List[Dict]:
        """Retorna hipótesis confirmadas (descubrimientos)."""
        return self.discoveries

    def get_best_hypotheses(self, n: int = 10) -> List[Hypothesis]:
        """Retorna las N mejores hipótesis."""
        sorted_h = sorted(self.world_model.hypotheses.values(),
                         key=lambda h: h.confidence * h.success_rate,
                         reverse=True)
        return sorted_h[:n]

    def get_status(self) -> Dict[str, Any]:
        """Retorna estado completo del agente."""
        return {
            'agent_id': self.agent_id,
            't': self.t,
            'CE': self.CE_history[-1] if self.CE_history else 0.5,
            'emotions': self.emotions.to_dict(),
            'dominant_emotion': self.emotions.dominant_emotion().value,
            'n_hypotheses': len(self.world_model.hypotheses),
            'n_confident': len(self.world_model.get_confident_hypotheses()),
            'n_discoveries': len(self.discoveries),
            'personality': self.personality,
        }


# =============================================================================
# Personalidades predefinidas para los 5 agentes
# =============================================================================

PERSONALITIES = {
    'NEO': {
        'curiosity_base': 0.8,      # Muy curioso
        'risk_tolerance': 0.7,       # Dispuesto a hipótesis arriesgadas
        'patience': 0.4,             # Impaciente, descarta rápido
        'domain_preference': 'crypto',  # Le interesa cripto
    },
    'EVA': {
        'curiosity_base': 0.6,
        'risk_tolerance': 0.5,
        'patience': 0.7,             # Paciente
        'domain_preference': 'climate',  # Le interesa clima
    },
    'ALEX': {
        'curiosity_base': 0.7,
        'risk_tolerance': 0.8,       # Muy arriesgado
        'patience': 0.5,
        'domain_preference': 'solar',   # Le interesa el sol
    },
    'ADAM': {
        'curiosity_base': 0.5,       # Moderado
        'risk_tolerance': 0.3,       # Conservador
        'patience': 0.9,             # Muy paciente
        'domain_preference': 'seismic', # Le interesan sismos
    },
    'IRIS': {
        'curiosity_base': 0.9,       # Muy muy curiosa
        'risk_tolerance': 0.6,
        'patience': 0.6,
        'domain_preference': None,   # Le interesa todo (cross-domain)
    },
}


def create_agent(agent_id: str, variables: List[str]) -> ExplorerAgent:
    """Crea un agente con personalidad predefinida."""
    personality = PERSONALITIES.get(agent_id, {
        'curiosity_base': 0.5,
        'risk_tolerance': 0.5,
        'patience': 0.5,
        'domain_preference': None,
    })
    return ExplorerAgent(agent_id, variables, personality=personality)
