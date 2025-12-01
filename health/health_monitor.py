"""
Health Monitor: Sistema de Monitorización de Salud Cognitiva
=============================================================

El módulo de monitorización que NO toca nada, solo evalúa.

Índice de salud por agente:
    H_t^A = σ(1 - Σ_i w_i · |m̃_i(t)|)

donde:
    m̃_i(t) = (m_i(t) - median_i) / MAD_i   (normalización endógena)
    w_i ∝ 1 / var_i                          (lo que más fluctúa, pesa menos)

Métricas monitorizadas:
    - crisis_rate: tasa de crisis
    - V_t: función de Lyapunov
    - CF_score: efectividad causal
    - CI_score: influencia causal
    - ethics_score: puntuación ética
    - narrative_continuity: continuidad narrativa
    - symbolic_stability: estabilidad simbólica

100% endógeno. Sin números mágicos.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history


class HealthLevel(Enum):
    """Niveles de salud - solo para logging/visualización."""
    CRITICAL = "critical"      # H < 0.2
    POOR = "poor"              # 0.2 <= H < 0.4
    MODERATE = "moderate"      # 0.4 <= H < 0.6
    GOOD = "good"              # 0.6 <= H < 0.8
    EXCELLENT = "excellent"    # H >= 0.8


@dataclass
class HealthMetrics:
    """Métricas de salud observadas."""
    crisis_rate: float = 0.0
    V_t: float = 1.0              # Lyapunov (menor es mejor)
    CF_score: float = 0.5         # Causal Effectiveness
    CI_score: float = 0.5         # Causal Influence
    ethics_score: float = 0.8
    narrative_continuity: float = 0.5
    symbolic_stability: float = 0.5
    self_coherence: float = 0.5   # AGI-20
    tom_accuracy: float = 0.5     # AGI-5
    config_entropy: float = 0.5   # AGI-18
    wellbeing: float = 0.5        # regulation
    metacognition: float = 0.5    # MC accuracy


@dataclass
class HealthAssessment:
    """Evaluación de salud de un agente."""
    agent_id: str
    t: int
    H_t: float                           # Índice de salud [0, 1]
    level: HealthLevel                   # Nivel categórico
    risk_factors: List[str]              # Métricas problemáticas
    contributing_factors: List[str]      # Métricas que ayudan
    normalized_metrics: Dict[str, float] # Métricas normalizadas
    weights: Dict[str, float]            # Pesos usados
    trend: float                         # Tendencia reciente


class HealthMonitor:
    """
    Monitor de salud cognitiva de un agente.

    Solo observa y evalúa - nunca modifica estado.

    Índice de salud:
        H_t = σ(1 - Σ w_i · |m̃_i|)

    donde σ es sigmoide suave y los pesos son endógenos.
    """

    # Métricas que se monitorizan
    METRIC_NAMES = [
        'crisis_rate', 'V_t', 'CF_score', 'CI_score',
        'ethics_score', 'narrative_continuity', 'symbolic_stability',
        'self_coherence', 'tom_accuracy', 'config_entropy',
        'wellbeing', 'metacognition'
    ]

    # Métricas donde menor es mejor (se invierten)
    INVERT_METRICS = ['crisis_rate', 'V_t']

    def __init__(self, agent_id: str):
        """
        Inicializa monitor de salud.

        Args:
            agent_id: Identificador del agente
        """
        self.agent_id = agent_id

        # Historiales por métrica
        self.history: Dict[str, List[float]] = {
            name: [] for name in self.METRIC_NAMES
        }

        # Historial de salud
        self.H_history: List[float] = []

        # Estadísticas acumuladas para normalización
        self.medians: Dict[str, float] = {name: 0.5 for name in self.METRIC_NAMES}
        self.mads: Dict[str, float] = {name: 0.1 for name in self.METRIC_NAMES}  # Median Absolute Deviation
        self.variances: Dict[str, float] = {name: 0.1 for name in self.METRIC_NAMES}

        self.t = 0

    def _compute_window(self) -> int:
        """Ventana endógena para cálculos."""
        return L_t(self.t) * 2

    def observe(self, t: int, metrics: HealthMetrics) -> None:
        """
        Registra observación de métricas.

        Args:
            t: Tiempo actual
            metrics: HealthMetrics con valores actuales
        """
        self.t = t

        # Registrar cada métrica
        metric_dict = {
            'crisis_rate': metrics.crisis_rate,
            'V_t': metrics.V_t,
            'CF_score': metrics.CF_score,
            'CI_score': metrics.CI_score,
            'ethics_score': metrics.ethics_score,
            'narrative_continuity': metrics.narrative_continuity,
            'symbolic_stability': metrics.symbolic_stability,
            'self_coherence': metrics.self_coherence,
            'tom_accuracy': metrics.tom_accuracy,
            'config_entropy': metrics.config_entropy,
            'wellbeing': metrics.wellbeing,
            'metacognition': metrics.metacognition
        }

        max_hist = max_history(t)

        for name, value in metric_dict.items():
            self.history[name].append(value)
            if len(self.history[name]) > max_hist:
                self.history[name] = self.history[name][-max_hist:]

        # Actualizar estadísticas periódicamente
        update_freq = max(5, L_t(t) // 2)
        if t % update_freq == 0:
            self._update_statistics()

    def _update_statistics(self):
        """Actualiza medianas, MAD y varianzas de forma endógena."""
        window = self._compute_window()

        for name in self.METRIC_NAMES:
            if len(self.history[name]) < window:
                continue

            recent = np.array(self.history[name][-window:])

            # Mediana
            self.medians[name] = float(np.median(recent))

            # MAD (Median Absolute Deviation)
            deviations = np.abs(recent - self.medians[name])
            self.mads[name] = float(np.median(deviations)) + 1e-8

            # Varianza
            self.variances[name] = float(np.var(recent)) + 1e-8

    def _normalize_metric(self, name: str, value: float) -> float:
        """
        Normaliza métrica usando historial.

        m̃_i(t) = (m_i(t) - median_i) / MAD_i

        Luego aplica sigmoide para mapear a [0, 1].
        """
        median = self.medians[name]
        mad = self.mads[name]

        # Normalización robusta
        normalized = (value - median) / mad

        # Invertir si menor es mejor
        if name in self.INVERT_METRICS:
            normalized = -normalized

        # Clip para evitar overflow en exp
        normalized = np.clip(normalized, -20, 20)

        # Mapear a [0, 1] con sigmoide suave
        # score = 1 si normalized >> 0, score = 0 si normalized << 0
        score = 1.0 / (1.0 + np.exp(-normalized))

        return float(score)

    def _compute_weights(self) -> Dict[str, float]:
        """
        Computa pesos endógenos para cada métrica.

        w_i ∝ 1 / var_i

        Las métricas más estables reciben más peso.
        """
        weights = {}

        # Peso inverso a la varianza
        total_inv_var = sum(1.0 / v for v in self.variances.values())

        for name in self.METRIC_NAMES:
            inv_var = 1.0 / self.variances[name]
            weights[name] = inv_var / total_inv_var

        return weights

    def _sigmoid(self, x: float, k: float = 1.0) -> float:
        """Sigmoide suave con pendiente endógena."""
        return 1.0 / (1.0 + np.exp(-k * x))

    def compute_health_index(self) -> Tuple[float, List[str]]:
        """
        Computa índice de salud.

        H_t = σ(1 - Σ w_i · |m̃_i(t)|)

        Returns:
            (H_t, risk_factors): índice y lista de métricas problemáticas
        """
        if self.t < L_t(self.t):
            # No hay suficiente historial
            return 0.5, []

        # Obtener valores actuales
        current_values = {
            name: self.history[name][-1] if self.history[name] else 0.5
            for name in self.METRIC_NAMES
        }

        # Normalizar
        normalized = {}
        for name, value in current_values.items():
            normalized[name] = self._normalize_metric(name, value)

        # Obtener pesos endógenos
        weights = self._compute_weights()

        # Calcular contribución ponderada
        # |m̃_i| mide cuánto se desvía del estado "normal"
        # Pero ya normalizamos a [0,1], así que usamos 1-score para problemas
        weighted_deviation = 0.0
        risk_factors = []
        contributing_factors = []

        # Umbral de riesgo endógeno: percentil 25 de scores históricos
        if self.H_history:
            risk_threshold = np.percentile(self.H_history[-self._compute_window():], 25)
        else:
            risk_threshold = 0.4

        for name in self.METRIC_NAMES:
            score = normalized[name]
            weight = weights[name]

            # Desviación del óptimo (1.0)
            deviation = abs(1.0 - score)
            weighted_deviation += weight * deviation

            # Identificar factores de riesgo
            if score < risk_threshold:
                risk_factors.append(name)
            elif score > 0.7:
                contributing_factors.append(name)

        # Calcular H_t
        # raw_H = 1 - weighted_deviation ya da [0,1]
        raw_H = 1.0 - weighted_deviation

        # Aplicar sigmoide suave centrada en 0.5
        # k endógeno basado en varianza de H histórico
        if len(self.H_history) > 10:
            var_H = np.var(self.H_history[-50:])
            k = 2.0 / (np.sqrt(var_H) + 0.1)  # Mayor k si H es estable
        else:
            k = 2.0

        H_t = self._sigmoid(raw_H - 0.5, k) * 2 - 0.5  # Mapear de nuevo
        H_t = np.clip(H_t + 0.5, 0, 1)  # Centrar en [0,1]

        # Guardar
        self.H_history.append(H_t)
        max_hist = max_history(self.t)
        if len(self.H_history) > max_hist:
            self.H_history = self.H_history[-max_hist:]

        return float(H_t), risk_factors

    def get_health_level(self, H_t: float) -> HealthLevel:
        """
        Determina nivel de salud basado en percentiles endógenos.
        """
        if len(self.H_history) < 20:
            # Umbrales default estructurales
            if H_t < 0.2:
                return HealthLevel.CRITICAL
            elif H_t < 0.4:
                return HealthLevel.POOR
            elif H_t < 0.6:
                return HealthLevel.MODERATE
            elif H_t < 0.8:
                return HealthLevel.GOOD
            else:
                return HealthLevel.EXCELLENT

        # Umbrales endógenos basados en historial
        p20 = np.percentile(self.H_history, 20)
        p40 = np.percentile(self.H_history, 40)
        p60 = np.percentile(self.H_history, 60)
        p80 = np.percentile(self.H_history, 80)

        if H_t < p20:
            return HealthLevel.CRITICAL
        elif H_t < p40:
            return HealthLevel.POOR
        elif H_t < p60:
            return HealthLevel.MODERATE
        elif H_t < p80:
            return HealthLevel.GOOD
        else:
            return HealthLevel.EXCELLENT

    def get_trend(self) -> float:
        """
        Calcula tendencia de salud.

        trend > 0: mejorando
        trend < 0: empeorando
        """
        window = min(L_t(self.t), len(self.H_history))
        if window < 5:
            return 0.0

        recent = np.array(self.H_history[-window:])
        x = np.arange(len(recent))

        # Regresión lineal simple
        x_mean = np.mean(x)
        y_mean = np.mean(recent)

        num = np.sum((x - x_mean) * (recent - y_mean))
        den = np.sum((x - x_mean) ** 2) + 1e-8

        slope = num / den
        return float(slope)

    def assess(self) -> HealthAssessment:
        """
        Evaluación completa de salud.

        Returns:
            HealthAssessment con todos los detalles
        """
        H_t, risk_factors = self.compute_health_index()

        # Obtener valores normalizados actuales
        normalized_metrics = {}
        current_values = {
            name: self.history[name][-1] if self.history[name] else 0.5
            for name in self.METRIC_NAMES
        }
        for name, value in current_values.items():
            normalized_metrics[name] = self._normalize_metric(name, value)

        # Factores contribuyentes
        contributing = [
            name for name, score in normalized_metrics.items()
            if score > 0.7
        ]

        return HealthAssessment(
            agent_id=self.agent_id,
            t=self.t,
            H_t=H_t,
            level=self.get_health_level(H_t),
            risk_factors=risk_factors,
            contributing_factors=contributing,
            normalized_metrics=normalized_metrics,
            weights=self._compute_weights(),
            trend=self.get_trend()
        )

    def get_health_threshold(self) -> float:
        """
        Umbral de salud endógeno para intervención.

        threshold = percentil_25(H_history)

        Si H_t < threshold, se recomienda intervención.
        """
        if len(self.H_history) < 20:
            return 0.3  # Default estructural

        return float(np.percentile(self.H_history, 25))

    def needs_intervention(self) -> bool:
        """
        Verifica si el agente necesita intervención médica.
        """
        if not self.H_history:
            return False

        H_t = self.H_history[-1]
        threshold = self.get_health_threshold()

        return H_t < threshold

    def get_statistics(self) -> Dict:
        """Estadísticas del monitor."""
        if not self.H_history:
            return {
                'agent_id': self.agent_id,
                't': self.t,
                'status': 'initializing'
            }

        return {
            'agent_id': self.agent_id,
            't': self.t,
            'H_current': self.H_history[-1],
            'H_mean': np.mean(self.H_history[-self._compute_window():]),
            'H_std': np.std(self.H_history[-self._compute_window():]),
            'trend': self.get_trend(),
            'level': self.get_health_level(self.H_history[-1]).value,
            'threshold': self.get_health_threshold(),
            'needs_intervention': self.needs_intervention(),
            'metric_weights': self._compute_weights()
        }


def test_health_monitor():
    """Test del Health Monitor."""
    print("=" * 70)
    print("TEST: HEALTH MONITOR")
    print("=" * 70)

    np.random.seed(42)

    monitor = HealthMonitor('NEO')

    print("\nSimulando 300 pasos...")

    for t in range(1, 301):
        # Simular métricas que varían
        # Crisis rate oscila
        crisis_rate = 0.1 + 0.15 * np.sin(t / 30) + np.random.randn() * 0.02
        crisis_rate = np.clip(crisis_rate, 0, 1)

        # Lyapunov decrece lentamente (mejora)
        V_t = 2.0 / (1 + t / 100) + np.random.randn() * 0.1

        # Otros métricas
        metrics = HealthMetrics(
            crisis_rate=crisis_rate,
            V_t=V_t,
            CF_score=0.5 + np.random.randn() * 0.1,
            CI_score=0.5 + np.random.randn() * 0.1,
            ethics_score=0.8 + np.random.randn() * 0.05,
            narrative_continuity=0.6 + np.random.randn() * 0.1,
            symbolic_stability=0.7 + np.random.randn() * 0.1,
            self_coherence=0.6 + np.random.randn() * 0.1,
            tom_accuracy=0.5 + t / 600 + np.random.randn() * 0.05,
            config_entropy=0.5 + np.random.randn() * 0.1,
            wellbeing=0.6 + np.random.randn() * 0.1,
            metacognition=0.5 + np.random.randn() * 0.1
        )

        monitor.observe(t, metrics)

        if t % 50 == 0:
            assessment = monitor.assess()
            print(f"\n  t={t}:")
            print(f"    H_t: {assessment.H_t:.3f}")
            print(f"    Level: {assessment.level.value}")
            print(f"    Trend: {assessment.trend:.4f}")
            print(f"    Risk factors: {assessment.risk_factors[:3]}")

    print("\n" + "=" * 70)
    print("ESTADÍSTICAS FINALES")
    print("=" * 70)

    stats = monitor.get_statistics()
    print(f"\n  Agente: {stats['agent_id']}")
    print(f"  H actual: {stats['H_current']:.3f}")
    print(f"  H medio: {stats['H_mean']:.3f}")
    print(f"  Tendencia: {stats['trend']:.4f}")
    print(f"  Nivel: {stats['level']}")
    print(f"  Umbral: {stats['threshold']:.3f}")
    print(f"  Necesita intervención: {stats['needs_intervention']}")

    print("\n  Pesos de métricas:")
    for name, weight in stats['metric_weights'].items():
        print(f"    {name}: {weight:.3f}")

    return monitor


if __name__ == "__main__":
    test_health_monitor()
