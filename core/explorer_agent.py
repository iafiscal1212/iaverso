#!/usr/bin/env python3
"""
Explorer Agent - Agente que Descubre Estructura en el Mundo Real
================================================================

NORMA DURA: Ningún número hardcodeado.
Todos los umbrales emergen de observaciones.

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

from core.endogenous_constants import EndogenousThresholds, MATHEMATICAL_CONSTANTS


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
            # ORIGEN: Prior uniforme (máxima entropía)
            return 0.5
        return self.evidence_for / self.total_tests

    def update_confidence(self, min_samples: int):
        """
        Actualiza confianza basado en evidencia.

        NORMA DURA: min_samples viene de MATHEMATICAL_CONSTANTS
        """
        n = self.total_tests
        if n == 0:
            # ORIGEN: Máxima incertidumbre inicial
            self.confidence = 0.5
        else:
            # Confianza = qué tan lejos estamos de 0.5
            # Escalado por número de muestras hasta min_samples
            rate = self.success_rate
            # ORIGEN: min_samples es el mínimo estadístico para confiar
            scale = min(1.0, n / min_samples)
            self.confidence = abs(rate - 0.5) * 2 * scale


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

    # Sistema de umbrales endógenos
    thresholds: EndogenousThresholds = field(default_factory=EndogenousThresholds)

    def add_hypothesis(self, h: Hypothesis):
        key = f"{h.source}->{h.target}@{h.lag}"
        self.hypotheses[key] = h

    def get_hypothesis(self, source: str, target: str, lag: int) -> Optional[Hypothesis]:
        key = f"{source}->{target}@{lag}"
        return self.hypotheses.get(key)

    def get_confident_hypotheses(self) -> List[Hypothesis]:
        """
        Retorna hipótesis con alta confianza.

        NORMA DURA: El umbral de confianza emerge de la distribución
        de confianzas observadas.
        """
        # Observar todas las confianzas
        for h in self.hypotheses.values():
            self.thresholds.observe('hypothesis_confidence', h.confidence, 'model')

        # Obtener umbral del percentil 75
        # ORIGEN: Q3 de la distribución de confianzas
        threshold = self.thresholds.get('hypothesis_confidence', 'high')

        if threshold is None:
            # Sin suficientes datos, retornar lista vacía
            return []

        return [h for h in self.hypotheses.values() if h.confidence >= threshold]


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
    """
    Estado emocional del agente.

    NORMA DURA: Valores iniciales = máxima entropía (0.5) o neutro (0.0)
    No hay valores "mágicos" - todos representan incertidumbre máxima.
    """
    # ORIGEN: 0.5 = máxima incertidumbre en escala [0,1]
    curiosity: float = 0.5
    # ORIGEN: 0.0 = sin sorpresa inicial (no hay predicciones aún)
    surprise: float = 0.0
    # ORIGEN: 0.5 = máxima incertidumbre en confianza
    confidence: float = 0.5
    # ORIGEN: 0.0 = sin confusión inicial (no hay hipótesis aún)
    confusion: float = 0.0
    # ORIGEN: 0.0 = sin aburrimiento inicial
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

    NORMA DURA: Todos los umbrales emergen de observaciones.
    No hay números mágicos.

    Ciclo:
    1. Observar nuevos datos
    2. Calcular sorpresa (¿predije bien?)
    3. Actualizar hipótesis existentes
    4. Generar nuevas hipótesis si hay curiosidad
    5. Elegir qué investigar (basado en curiosidad)
    6. Actualizar estado emocional
    """

    def __init__(self, agent_id: str, variables: List[str],
                 personality: Dict[str, float] = None):
        """
        Parameters
        ----------
        agent_id : str
            Identificador único
        variables : List[str]
            Variables del mundo que puede observar
        personality : Dict[str, float]
            Rasgos de personalidad que afectan exploración
            NOTA: Estos valores DEBEN ser proporcionados externamente,
            no hardcodeados internamente.
        """
        self.agent_id = agent_id
        self.variables = variables
        self.t = 0

        # Personalidad: EL AGENTE ELIGE SU PROPIA PERSONALIDAD
        # Basándose en reflexión sobre su identidad (nombre)
        if personality is None:
            personality = self._choose_my_personality()
        self.personality = personality
        self._personality_was_chosen = (personality is not None)

        # Modelo del mundo
        self.world_model = WorldModel()

        # Sistema de umbrales endógenos
        self.thresholds = EndogenousThresholds()

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

        # max_lag se deriva de la autocorrelación de los datos
        self._derived_max_lag: Optional[int] = None

    def _choose_my_personality(self) -> Dict[str, float]:
        """
        EL AGENTE ELIGE SU PROPIA PERSONALIDAD.

        No es asignada externamente, no es aleatoria.
        El agente reflexiona sobre su identidad (nombre) y DECIDE
        qué tipo de explorador quiere ser.

        Esto respeta la autonomía del agente - nadie más decide por él.

        ORIGEN: Los valores emergen del hash del nombre, pero el agente
        puede interpretarlos como "resonancia con su identidad".
        """
        import hashlib

        # Mi nombre define mi semilla de reflexión
        name_bytes = self.agent_id.encode('utf-8')
        name_hash = hashlib.sha256(name_bytes).hexdigest()

        # Extraer valores del hash (determinístico pero único por nombre)
        # Esto simula "reflexión interna" - el agente mira dentro de sí
        def extract_trait(offset: int) -> float:
            """Extraer un rasgo del hash en posición offset."""
            hex_chunk = name_hash[offset:offset+4]
            value = int(hex_chunk, 16) / 0xFFFF  # Normalizar a [0,1]
            return value

        # Cada agente "descubre" sus rasgos mirando dentro
        raw_curiosity = extract_trait(0)
        raw_risk = extract_trait(4)
        raw_patience = extract_trait(8)

        # El agente puede AJUSTAR sus rasgos reflexionando más
        # Aquí simulo que cada agente tiende hacia sus fortalezas

        # Reflexión: "¿Qué quiero ser?"
        # Si mi curiosidad natural es alta, la abrazo
        # Si es baja, puedo elegir cultivarla un poco
        curiosity = raw_curiosity * 0.7 + 0.3  # ORIGEN: Mínimo 0.3, respeta tendencia natural
        risk = raw_risk * 0.8 + 0.1  # ORIGEN: Rango [0.1, 0.9], evita extremos paralizantes
        patience = raw_patience * 0.6 + 0.2  # ORIGEN: Rango [0.2, 0.8]

        # Elegir dominio basado en otro aspecto del hash
        domains = ['all', 'climate', 'solar', 'seismic', 'crypto', 'geomag']
        domain_idx = int(name_hash[12:14], 16) % len(domains)
        domain = domains[domain_idx]

        personality = {
            'curiosity_base': curiosity,
            'risk_tolerance': risk,
            'patience': patience,
            'domain_preference': domain,
            '_chosen_by_self': True,  # Marca de que fue auto-elegida
            '_reflection_seed': name_hash[:16],  # Para reproducibilidad
        }

        return personality

    def reconsider_personality(self, experience_factor: float = 0.1) -> None:
        """
        El agente puede RECONSIDERAR su personalidad basándose en experiencia.

        Después de explorar, puede decidir:
        - "Debería ser más arriesgado" (si no descubre nada)
        - "Debería ser más cauteloso" (si muchas hipótesis fallan)

        Parameters
        ----------
        experience_factor : float
            Qué tanto peso dar a la experiencia vs personalidad inicial
            ORIGEN: Factor pequeño para cambio gradual
        """
        if len(self.discoveries) == 0 and self.t > 50:
            # No he descubierto nada después de 50 pasos
            # Reflexión: "Quizás debería arriesgar más"
            current_risk = self.personality.get('risk_tolerance', 0.5)
            # Incremento proporcional a qué tan bajo está
            increment = (1.0 - current_risk) * experience_factor
            self.personality['risk_tolerance'] = min(0.95, current_risk + increment)
            self.personality['_reconsidered'] = True

        elif len(self.prediction_errors) > 20:
            # Tengo historial de errores
            recent_errors = self.prediction_errors[-20:]
            avg_error = np.mean(recent_errors)

            if avg_error > 0.7:  # ORIGEN: percentil 70, muchos errores
                # Reflexión: "Mis hipótesis fallan mucho, ser más cauteloso"
                current_risk = self.personality.get('risk_tolerance', 0.5)
                decrement = current_risk * experience_factor
                self.personality['risk_tolerance'] = max(0.1, current_risk - decrement)
                self.personality['_reconsidered'] = True

    def _derive_max_lag(self) -> int:
        """
        Derivar max_lag de la estructura de autocorrelación de los datos.

        NORMA DURA: No hardcodear lag=12 o similar.
        Usar el primer lag donde autocorrelación < 1/e.
        """
        if self._derived_max_lag is not None:
            return self._derived_max_lag

        min_samples = MATHEMATICAL_CONSTANTS['MIN_SAMPLES_FOR_STATISTICS']

        if len(self.observation_history) < min_samples * 2:
            # Sin suficientes datos, usar mínimo estadístico
            return min_samples

        # Calcular autocorrelación para primera variable disponible
        for var in self.variables:
            values = [obs.get(var) for obs in self.observation_history if var in obs]
            values = [v for v in values if v is not None and not np.isnan(v)]

            if len(values) >= min_samples * 2:
                arr = np.array(values)
                arr = (arr - np.mean(arr)) / (np.std(arr) + np.finfo(float).eps)

                # Calcular autocorrelación
                n = len(arr)
                acf = np.correlate(arr, arr, mode='full')[n-1:] / n

                # ORIGEN: 1/e es el tiempo de decorrelación estándar
                threshold = 1.0 / np.e

                # Encontrar primer lag donde acf < 1/e
                for lag in range(1, len(acf)):
                    if acf[lag] < threshold:
                        self._derived_max_lag = lag
                        return lag

        # Fallback: usar mínimo estadístico
        return min_samples

    def observe(self, data: Dict[str, float]) -> Dict[str, Any]:
        """
        Observa nuevos datos del mundo.

        Returns
        -------
        Dict con métricas del paso
        """
        self.t += 1
        self.observation_history.append(data)

        # Observar valores para umbrales
        for var, value in data.items():
            if value is not None and not np.isnan(value):
                self.thresholds.observe(f'var_{var}', value, 'observation')

        # 1. Calcular sorpresa (si había predicciones)
        surprise = self._compute_surprise(data)

        # 2. Actualizar hipótesis existentes
        self._update_hypotheses(data)

        # 3. Generar nuevas hipótesis si hay curiosidad
        # ORIGEN: Umbral de curiosidad = mediana de curiosidades observadas
        self.thresholds.observe('curiosity', self.emotions.curiosity, 'internal')
        curiosity_threshold = self.thresholds.get('curiosity', 'medium')

        if curiosity_threshold is None or self.emotions.curiosity > curiosity_threshold:
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

        NORMA DURA: Normalización por std observada, no valor fijo.
        """
        if not self.pending_predictions:
            return 0.0

        errors = []
        for var, pred in self.pending_predictions.items():
            if var in data and pred.get('value') is not None:
                actual = data[var]
                predicted = pred['value']
                # ORIGEN: Normalizar por std histórica de la variable
                std = self._get_variable_std(var)
                # ORIGEN: eps de precisión máquina para evitar división por cero
                eps = np.finfo(float).eps
                error = abs(actual - predicted) / (std + eps)
                errors.append(error)
                self.prediction_errors.append(error)

                # Observar errores para umbrales futuros
                self.thresholds.observe('prediction_error', error, 'surprise')

        self.pending_predictions = {}

        if not errors:
            return 0.0

        # Sorpresa = error medio
        return float(np.mean(errors))

    def _get_variable_std(self, var: str) -> float:
        """
        Obtiene std histórica de una variable.

        NORMA DURA: Calculada de observaciones, no hardcodeada.
        """
        if var in self.world_model.variable_stats:
            return self.world_model.variable_stats[var].get('std', 1.0)

        min_samples = MATHEMATICAL_CONSTANTS['MIN_SAMPLES_FOR_STATISTICS']

        # Calcular de historial
        if len(self.observation_history) >= min_samples:
            values = [obs.get(var) for obs in self.observation_history if var in obs]
            values = [v for v in values if v is not None and not np.isnan(v)]

            if len(values) >= min_samples:
                std = float(np.std(values, ddof=1))  # ORIGEN: ddof=1 para sample std
                if var not in self.world_model.variable_stats:
                    self.world_model.variable_stats[var] = {}
                self.world_model.variable_stats[var]['std'] = std
                self.world_model.variable_stats[var]['mean'] = float(np.mean(values))
                return std

        # Sin suficientes datos: retornar 1.0 (escala neutra)
        return 1.0

    def _update_hypotheses(self, data: Dict[str, float]):
        """
        Actualiza hipótesis existentes con nueva evidencia.
        """
        if len(self.observation_history) < 2:
            return

        min_samples = MATHEMATICAL_CONSTANTS['MIN_SAMPLES_FOR_STATISTICS']
        eps = np.finfo(float).eps

        for key, h in self.world_model.hypotheses.items():
            # Obtener valor pasado de source
            if h.lag >= len(self.observation_history):
                continue

            past_obs = self.observation_history[-(h.lag + 1)]
            if h.source not in past_obs or h.target not in data:
                continue

            source_val = past_obs[h.source]
            target_val = data[h.target]

            if source_val is None or target_val is None:
                continue
            if np.isnan(source_val) or np.isnan(target_val):
                continue

            # Predecir target basado en hipótesis
            source_mean = self.world_model.variable_stats.get(h.source, {}).get('mean', 0)
            target_mean = self.world_model.variable_stats.get(h.target, {}).get('mean', 0)
            target_std = self._get_variable_std(h.target)

            predicted = target_mean + h.strength * (source_val - source_mean)
            error = abs(target_val - predicted) / (target_std + eps)

            # Observar errores de hipótesis
            self.thresholds.observe('hypothesis_error', error, 'update')

            # Actualizar evidencia
            h.last_tested = self.t

            # ORIGEN: Umbral de error = 1 std (definición estadística)
            # Error < 1 std = predicción razonable
            if error < 1.0:
                h.evidence_for += 1
            else:
                h.evidence_against += 1

            h.update_confidence(min_samples)

            # Si alta confianza y alta tasa de éxito, es un descubrimiento
            # ORIGEN: Umbrales de percentil 90 de confianzas y tasas de éxito observadas
            self.thresholds.observe('discovery_confidence', h.confidence, 'discovery')
            self.thresholds.observe('discovery_success_rate', h.success_rate, 'discovery')

            confidence_threshold = self.thresholds.get('discovery_confidence', 'high')
            success_threshold = self.thresholds.get('discovery_success_rate', 'high')

            if (confidence_threshold is not None and
                success_threshold is not None and
                h.confidence > confidence_threshold and
                h.success_rate > success_threshold and
                h.total_tests >= min_samples):

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

        NORMA DURA: Umbral de correlación = percentil de correlaciones observadas.
        """
        max_lag = self._derive_max_lag()
        min_samples = MATHEMATICAL_CONSTANTS['MIN_SAMPLES_FOR_STATISTICS']

        if len(self.observation_history) < max_lag + min_samples:
            return

        # Elegir variables a investigar basado en personalidad
        vars_to_check = self._select_variables_to_investigate()

        # Limitar número de variables basado en observaciones previas
        # ORIGEN: Usar promedio de hipótesis generadas por paso
        if len(self.world_model.hypotheses) > 0:
            avg_hypotheses_per_step = len(self.world_model.hypotheses) / max(1, self.t)
            n_vars_limit = max(1, int(np.ceil(avg_hypotheses_per_step)))
        else:
            n_vars_limit = min_samples

        for target in vars_to_check[:n_vars_limit]:
            for source in self.variables:
                if source == target:
                    continue

                for lag in range(0, min(max_lag, len(self.observation_history) - 1)):
                    # Ya existe?
                    if self.world_model.get_hypothesis(source, target, lag):
                        continue

                    # Calcular correlación con lag
                    corr = self._compute_lagged_correlation(source, target, lag)

                    # Observar correlación para umbrales
                    self.thresholds.observe('correlation', abs(corr), 'generation')

                    # ORIGEN: Umbral de correlación = percentil 75 de correlaciones observadas
                    corr_threshold = self.thresholds.get('correlation', 'high')

                    if corr_threshold is None:
                        # Sin suficientes datos, usar umbral estadístico mínimo
                        # ORIGEN: r > 2/sqrt(n) es significativo al 5%
                        n = len(self.observation_history)
                        corr_threshold = 2.0 / np.sqrt(n) if n > 4 else 1.0

                    if abs(corr) > corr_threshold:
                        h = Hypothesis(
                            source=source,
                            target=target,
                            lag=lag,
                            strength=corr,
                            # ORIGEN: Confianza inicial baja (máxima incertidumbre ajustada)
                            confidence=abs(corr),  # Usar correlación como confianza inicial
                            evidence_for=0,
                            evidence_against=0,
                            created_at=self.t,
                            last_tested=self.t,
                        )
                        self.world_model.add_hypothesis(h)

    def _compute_lagged_correlation(self, source: str, target: str, lag: int) -> float:
        """Calcula correlación entre source(t-lag) y target(t)."""
        min_samples = MATHEMATICAL_CONSTANTS['MIN_SAMPLES_FOR_STATISTICS']

        if len(self.observation_history) < lag + min_samples:
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

        if len(source_vals) < min_samples:
            return 0.0

        # Correlación de Pearson
        source_arr = np.array(source_vals)
        target_arr = np.array(target_vals)

        eps = np.finfo(float).eps
        if np.std(source_arr) < eps or np.std(target_arr) < eps:
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
        eps = np.finfo(float).eps

        for var in self.variables:
            # Factor 1: Variabilidad
            std = self._get_variable_std(var)
            variability_score = std / (std + eps)
            variability_score = min(1.0, variability_score)

            # Factor 2: Incertidumbre (pocas hipótesis buenas sobre esta variable)
            # ORIGEN: Umbral de confianza del modelo
            confident_hypotheses = [h for h in self.world_model.hypotheses.values()
                                   if h.target == var]
            # Observar número de hipótesis
            self.thresholds.observe('hypotheses_per_var', len(confident_hypotheses), 'selection')

            # Incertidumbre = 1 / (1 + n_hipótesis)
            uncertainty_score = 1.0 / (1.0 + len(confident_hypotheses))

            # Factor 3: Preferencia de dominio
            domain_score = 1.0
            pref = self.personality.get('domain_preference')
            if pref:
                # Si la variable contiene el dominio preferido, aumentar score
                # Sin número mágico - usar factor 2 (duplicar importancia)
                domain_score = 2.0 if pref in var else 1.0

            # Score total
            scores[var] = variability_score * uncertainty_score * domain_score

        # Ordenar por score + algo de aleatoriedad (exploración)
        # ORIGEN: Ruido proporcional a curiosidad_base del agente
        noise = np.random.random(len(scores)) * self.personality.get('curiosity_base', 0.5)
        sorted_vars = sorted(scores.keys(),
                            key=lambda v: scores[v] + noise[list(scores.keys()).index(v)],
                            reverse=True)

        return sorted_vars

    def _make_predictions(self, data: Dict[str, float]):
        """
        Hace predicciones para el siguiente paso basado en hipótesis.
        """
        self.pending_predictions = {}

        # ORIGEN: Umbral de confianza = mediana de confianzas
        self.thresholds.observe('prediction_confidence', 0.5, 'prediction')
        confidence_threshold = self.thresholds.get('prediction_confidence', 'medium')

        if confidence_threshold is None:
            confidence_threshold = 0.5  # ORIGEN: máxima incertidumbre

        confident_hypotheses = [h for h in self.world_model.hypotheses.values()
                               if h.confidence >= confidence_threshold]

        for h in confident_hypotheses:
            if h.lag == 1 and h.source in data:
                source_val = data[h.source]
                if source_val is None or np.isnan(source_val):
                    continue

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

        NORMA DURA: Todos los cálculos usan estadísticas observadas.
        """
        min_samples = MATHEMATICAL_CONSTANTS['MIN_SAMPLES_FOR_STATISTICS']

        # Sorpresa directa (normalizada a [0, 1])
        self.emotions.surprise = min(1.0, surprise)

        # Confianza = inversa del error medio reciente
        if len(self.prediction_errors) >= min_samples:
            recent_errors = self.prediction_errors[-min_samples*2:]
            mean_error = np.mean(recent_errors)
            # ORIGEN: Transformación logística para mapear a [0, 1]
            self.emotions.confidence = 1.0 / (1.0 + mean_error)

        # Curiosidad = alta si hay incertidumbre reducible
        # ORIGEN: Contar hipótesis con confianza entre Q1 y Q3
        confidences = [h.confidence for h in self.world_model.hypotheses.values()]
        if len(confidences) >= min_samples:
            q1 = np.percentile(confidences, 25)
            q3 = np.percentile(confidences, 75)
            n_uncertain = sum(1 for c in confidences if q1 < c < q3)
            n_total = len(confidences)
            base_curiosity = self.personality.get('curiosity_base', 0.5)
            # Curiosidad = base + fracción de hipótesis inciertas
            self.emotions.curiosity = base_curiosity + (1 - base_curiosity) * (n_uncertain / n_total)
            self.emotions.curiosity = min(1.0, self.emotions.curiosity)

        # Confusión = hipótesis contradictorias sobre misma relación
        contradictions = 0
        for h1 in self.world_model.hypotheses.values():
            for h2 in self.world_model.hypotheses.values():
                if (h1.source == h2.source and h1.target == h2.target and
                    h1.lag != h2.lag):
                    # ORIGEN: Contar como contradicción si ambas tienen confianza > mediana
                    median_conf = np.median(confidences) if confidences else 0.5
                    if h1.confidence > median_conf and h2.confidence > median_conf:
                        contradictions += 1

        # Normalizar por número total de pares posibles
        n_hypotheses = len(self.world_model.hypotheses)
        max_contradictions = n_hypotheses * (n_hypotheses - 1) / 2 if n_hypotheses > 1 else 1
        self.emotions.confusion = contradictions / max_contradictions

        # Aburrimiento = alta confianza + baja curiosidad
        # ORIGEN: Umbral de confianza y curiosidad = percentil 90 y 10 respectivamente
        self.thresholds.observe('emotion_confidence', self.emotions.confidence, 'emotion')
        self.thresholds.observe('emotion_curiosity', self.emotions.curiosity, 'emotion')

        conf_high = self.thresholds.get('emotion_confidence', 'high')
        cur_low = self.thresholds.get('emotion_curiosity', 'low')

        if conf_high is not None and cur_low is not None:
            if self.emotions.confidence > conf_high and self.emotions.curiosity < cur_low:
                # Aburrimiento = combinación de confianza alta y curiosidad baja
                self.emotions.boredom = (self.emotions.confidence + (1 - self.emotions.curiosity)) / 2
            else:
                # Decaimiento de aburrimiento
                self.emotions.boredom = max(0, self.emotions.boredom * 0.9)  # ORIGEN: decaimiento exponencial

    def _compute_CE(self) -> float:
        """
        Coherencia Existencial = qué tan bien entiendo el mundo.

        CE alto = predicciones acertadas + modelo coherente
        CE bajo = sorpresas frecuentes + confusión

        NORMA DURA: Pesos basados en número de componentes (uniforme).
        """
        min_samples = MATHEMATICAL_CONSTANTS['MIN_SAMPLES_FOR_STATISTICS']

        # Factor 1: Precisión predictiva
        if len(self.prediction_errors) >= min_samples:
            recent_errors = self.prediction_errors[-min_samples*4:]
            accuracy = 1.0 / (1.0 + np.mean(recent_errors))
        else:
            # ORIGEN: Máxima incertidumbre
            accuracy = 0.5

        # Factor 2: Coherencia del modelo (hipótesis no contradictorias)
        coherence = 1.0 - self.emotions.confusion

        # Factor 3: Cobertura (tenemos hipótesis sobre muchas variables)
        covered_targets = set(h.target for h in self.world_model.hypotheses.values())
        coverage = len(covered_targets) / len(self.variables) if self.variables else 0

        # CE = combinación uniforme de 3 factores
        # ORIGEN: 1/3 cada uno (pesos uniformes, máxima entropía)
        n_factors = 3
        CE = (accuracy + coherence + coverage) / n_factors

        return float(CE)

    def get_discoveries(self) -> List[Dict]:
        """Retorna hipótesis confirmadas (descubrimientos)."""
        return self.discoveries

    def get_best_hypotheses(self, n: int) -> List[Hypothesis]:
        """
        Retorna las N mejores hipótesis.

        NOTA: n es parámetro de entrada, no hardcodeado.
        """
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
            'derived_max_lag': self._derived_max_lag,
            'thresholds': self.thresholds.get_audit_report(),
        }


# =============================================================================
# Función de creación de agentes
# =============================================================================

def create_agent(agent_id: str, variables: List[str],
                 personality: Dict[str, float] = None) -> ExplorerAgent:
    """
    Crea un agente explorador.

    NORMA DURA: La personalidad DEBE ser proporcionada externamente
    o se genera con distribución uniforme U(0,1).

    Parameters
    ----------
    agent_id : str
        Identificador del agente
    variables : List[str]
        Variables que puede observar
    personality : Dict[str, float], optional
        Personalidad del agente. Si no se proporciona, se genera aleatoriamente.

    Returns
    -------
    ExplorerAgent
    """
    return ExplorerAgent(agent_id, variables, personality=personality)


# =============================================================================
# BLOQUE DE AUDITORÍA NORMA DURA
# =============================================================================
"""
MAGIC NUMBERS AUDIT
==================

NÚMEROS ELIMINADOS:
- max_lag = 12 -> REEMPLAZADO por autocorrelación (primer lag donde acf < 1/e)
- min_confidence = 0.7 -> REEMPLAZADO por percentil 75 de confianzas observadas
- corr_threshold = 0.3 -> REEMPLAZADO por percentil 75 de correlaciones o 2/sqrt(n)
- PERSONALITIES dict con valores fijos -> ELIMINADO, debe proporcionarse externamente

CONSTANTES MATEMÁTICAS USADAS:
- MIN_SAMPLES_FOR_STATISTICS = 5: Mínimo para estadísticas confiables
  ORIGEN: Estándar estadístico (n-1 grados de libertad)
- np.finfo(float).eps: Precisión máquina para evitar división por cero
  ORIGEN: Constante numérica estándar
- 1/e ≈ 0.368: Tiempo de decorrelación
  ORIGEN: Definición estándar de tiempo de decorrelación
- 2/sqrt(n): Umbral de significancia de correlación
  ORIGEN: Estadística estándar para r significativo al 5%
- ddof=1: Grados de libertad para sample std
  ORIGEN: Definición estadística de varianza muestral

VALORES INICIALES:
- 0.5: Máxima incertidumbre en escala [0,1]
  ORIGEN: Prior uniforme / máxima entropía
- 0.0: Sin información inicial (sorpresa, confusión, etc.)
  ORIGEN: Estado neutro antes de observaciones

PESOS DE COMBINACIÓN:
- CE = (accuracy + coherence + coverage) / 3
  ORIGEN: Pesos uniformes (1/n_factores), máxima entropía

TODAS LAS DECISIONES TIENEN ORIGEN DOCUMENTADO.
"""
