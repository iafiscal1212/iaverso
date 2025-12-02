"""
Λ-Field: Campo Meta-Dinámico
=============================

El Λ-Field observa qué régimen interno domina en cada momento:

- CIRCADIAN: ciclos de actividad/descanso
- NARRATIVE: identidad, coherencia existencial, entropía narrativa
- QUANTUM: Q-Field, ComplexField, decoherencia, colapso
- TELEO: Omega Spaces, teleología, objetivos largos
- SOCIAL: interacciones multi-agente, TensorMind
- CREATIVE: Genesis, ideas, materialización (nuevo)

⚠️ IMPORTANTE:
Λ NO CONTROLA NADA.
Solo mide y devuelve:
- Vector de pesos π_r(t) (uno por régimen)
- Escalar Λ(t) que indica concentración de la dinámica

Todo es endógeno:
- Pesos por varianza inversa
- Z-scores históricos
- Softmax sin parámetros
- Entropía normalizada

100% endógeno. Sin números mágicos (excepto eps de máquina).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class Regime(Enum):
    """Regímenes internos que puede dominar la dinámica."""
    CIRCADIAN = "circadian"     # Ciclos de actividad/descanso
    NARRATIVE = "narrative"     # Identidad, CE, entropía narrativa
    QUANTUM = "quantum"         # Q-Field, ComplexField
    TELEO = "teleo"            # Omega Spaces, teleología
    SOCIAL = "social"          # Multi-agente, TensorMind
    CREATIVE = "creative"      # Genesis, ideas, materialización


@dataclass
class LambdaSnapshot:
    """Estado del Λ-Field en un instante."""
    t: int
    lambda_scalar: float                    # Λ(t) ∈ [0, 1]
    regime_weights: Dict[str, float]        # π_r(t) por régimen
    regime_features: Dict[str, float]       # φ_r(t) por régimen
    dominant_regime: str                    # argmax π_r(t)
    regime_entropy: float                   # H_π(t)
    metric_z: Dict[str, float]             # z_k(t) por métrica
    metric_weights: Dict[str, float]       # w̃_k(t) por métrica


@dataclass
class MetricStats:
    """Estadísticas históricas de una métrica."""
    mu: float = 0.0           # Media histórica
    sigma: float = 1.0        # Desviación estándar
    var_z: float = 1.0        # Varianza de z-scores
    n: int = 0                # Número de observaciones


class LambdaField:
    """
    Campo Λ meta-dinámico.

    Observa métricas internas y devuelve:
    - Pesos de régimen π_r(t)
    - Escalar Λ(t) de concentración

    NO modifica el estado de los agentes.
    Todo es endógeno (pesos por varianza inversa, z-scores históricos).
    """

    # Mapeo por defecto de métricas a regímenes
    DEFAULT_REGIME_MAP = {
        Regime.CIRCADIAN.value: [
            "phase_stability",
            "energy_cycle_variance",
            "circadian_amplitude"
        ],
        Regime.NARRATIVE.value: [
            "CE_mean",
            "H_narr_mean",
            "var_S_minus_I",
            "identity_stability"
        ],
        Regime.QUANTUM.value: [
            "E_Q_mean",
            "C_Q_mean",
            "lambda_decoherence_mean",
            "collapse_pressure_mean",
            "psi_norm_mean"
        ],
        Regime.TELEO.value: [
            "omega_modes_active",
            "omega_reconstruction_error",
            "phase_curvature_mean"
        ],
        Regime.SOCIAL.value: [
            "tensor_power",
            "identity_divergence",
            "n_communities"
        ],
        Regime.CREATIVE.value: [
            "ideas_per_step",
            "adoption_rate",
            "objects_created",
            "inspiration_potential"
        ]
    }

    def __init__(
        self,
        metric_names: Optional[List[str]] = None,
        regime_map: Optional[Dict[str, List[str]]] = None
    ):
        """
        Inicializa el Λ-Field.

        Args:
            metric_names: Lista de nombres de métricas escalares.
                         Si None, se infiere del regime_map.
            regime_map: Asignación régimen -> lista de métricas.
                       Si None, usa DEFAULT_REGIME_MAP.
        """
        self.eps = np.finfo(float).eps
        self.t = 0

        # Configurar régimen map
        self.regime_map = regime_map if regime_map else self.DEFAULT_REGIME_MAP.copy()

        # Inferir métricas si no se proporcionan
        if metric_names is None:
            metric_names = []
            for metrics in self.regime_map.values():
                metric_names.extend(metrics)
            metric_names = list(set(metric_names))

        self.metric_names = metric_names

        # Historial de métricas
        self._history: Dict[str, List[float]] = {m: [] for m in metric_names}

        # Estadísticas por métrica (actualización incremental)
        self._stats: Dict[str, MetricStats] = {m: MetricStats() for m in metric_names}

        # Historial de z-scores para calcular var_z
        self._z_history: Dict[str, List[float]] = {m: [] for m in metric_names}

        # Historial de snapshots
        self._snapshots: List[LambdaSnapshot] = []

    def _update_stats_incremental(self, name: str, value: float):
        """
        Actualiza estadísticas de forma incremental (Welford's algorithm).

        Evita recalcular sobre todo el historial.
        """
        stats = self._stats[name]
        stats.n += 1
        n = stats.n

        if n == 1:
            stats.mu = value
            stats.sigma = 0.0
        else:
            # Actualización incremental de media y varianza
            delta = value - stats.mu
            stats.mu += delta / n
            delta2 = value - stats.mu
            # M2 acumulado para varianza
            if not hasattr(stats, '_M2'):
                stats._M2 = 0.0
            stats._M2 += delta * delta2
            stats.sigma = np.sqrt(stats._M2 / n) if n > 1 else 0.0

    def _compute_z(self, name: str, value: float) -> float:
        """Calcula z-score de un valor."""
        stats = self._stats[name]
        return (value - stats.mu) / (stats.sigma + self.eps)

    def _update_var_z(self, name: str, z: float):
        """Actualiza varianza de z-scores."""
        self._z_history[name].append(z)

        # Limitar historial
        max_len = 200
        if len(self._z_history[name]) > max_len:
            self._z_history[name] = self._z_history[name][-max_len:]

        # Calcular var_z
        z_arr = np.array(self._z_history[name])
        if len(z_arr) > 1:
            self._stats[name].var_z = float(np.var(z_arr))
        else:
            self._stats[name].var_z = 1.0

    def _compute_regime_feature(
        self,
        regime: str,
        z_now: Dict[str, float],
        w_norm: Dict[str, float]
    ) -> float:
        """
        Calcula φ_r(t) = Σ w̃_k(t) · z_k(t) para un régimen.
        """
        metrics = self.regime_map.get(regime, [])

        if not metrics:
            return 0.0

        acc = 0.0
        total_weight = 0.0

        for m in metrics:
            if m in z_now and m in w_norm:
                acc += w_norm[m] * z_now[m]
                total_weight += w_norm[m]

        # Normalizar por peso total del régimen
        if total_weight > self.eps:
            return acc / total_weight
        return 0.0

    def _softmax(self, values: np.ndarray) -> np.ndarray:
        """Softmax numéricamente estable."""
        v_shift = values - np.max(values)
        exp_v = np.exp(v_shift)
        return exp_v / (np.sum(exp_v) + self.eps)

    def _compute_entropy(self, probs: np.ndarray) -> float:
        """Calcula entropía de Shannon."""
        H = 0.0
        for p in probs:
            if p > self.eps:
                H -= p * np.log(p)
        return float(H)

    def step(self, metrics: Dict[str, float]) -> LambdaSnapshot:
        """
        Paso principal del Λ-Field.

        Args:
            metrics: Dict nombre_métrica -> valor escalar en tiempo t.

        Returns:
            LambdaSnapshot con todos los valores calculados.

        No modifica nada externo, solo observa.
        """
        self.t += 1

        # Actualizar historial y estadísticas
        for name in self.metric_names:
            if name in metrics:
                value = float(metrics[name])
                self._history[name].append(value)
                self._update_stats_incremental(name, value)

        # Calcular z-scores y pesos
        z_now: Dict[str, float] = {}
        w_raw: Dict[str, float] = {}

        for name in self.metric_names:
            if name in metrics:
                value = metrics[name]
                z = self._compute_z(name, value)
                z_now[name] = z

                # Actualizar var_z
                self._update_var_z(name, z)

                # Peso = 1 / var_z (métricas estables pesan más)
                var_z = self._stats[name].var_z
                w_raw[name] = 1.0 / (var_z + self.eps)

        # Normalizar pesos
        w_sum = sum(w_raw.values()) if w_raw else 1.0
        w_norm = {name: w / w_sum for name, w in w_raw.items()}

        # Calcular φ_r(t) para cada régimen
        regime_features: Dict[str, float] = {}
        for regime in self.regime_map.keys():
            phi = self._compute_regime_feature(regime, z_now, w_norm)
            regime_features[regime] = phi

        # Calcular π_r(t) via softmax
        regime_names = list(regime_features.keys())
        a_vals = np.array([regime_features[r] for r in regime_names])

        pi_vals = self._softmax(a_vals)

        regime_weights = {
            regime_names[i]: float(pi_vals[i])
            for i in range(len(regime_names))
        }

        # Entropía y Λ(t)
        H = self._compute_entropy(pi_vals)
        R = len(regime_names)
        H_max = np.log(R) if R > 1 else 1.0

        lambda_scalar = 1.0 - (H / H_max) if H_max > self.eps else 0.0
        lambda_scalar = float(np.clip(lambda_scalar, 0, 1))

        # Régimen dominante
        dominant_idx = int(np.argmax(pi_vals))
        dominant_regime = regime_names[dominant_idx]

        # Crear snapshot
        snapshot = LambdaSnapshot(
            t=self.t,
            lambda_scalar=lambda_scalar,
            regime_weights=regime_weights,
            regime_features=regime_features,
            dominant_regime=dominant_regime,
            regime_entropy=float(H),
            metric_z=z_now,
            metric_weights=w_norm
        )

        self._snapshots.append(snapshot)

        # Limitar historial de snapshots
        max_snapshots = 500
        if len(self._snapshots) > max_snapshots:
            self._snapshots = self._snapshots[-max_snapshots:]

        return snapshot

    def get_regime_history(self, regime: str) -> List[float]:
        """Obtiene historial de pesos para un régimen."""
        return [s.regime_weights.get(regime, 0) for s in self._snapshots]

    def get_lambda_history(self) -> List[float]:
        """Obtiene historial de Λ(t)."""
        return [s.lambda_scalar for s in self._snapshots]

    def get_dominant_regime_sequence(self) -> List[str]:
        """Obtiene secuencia de regímenes dominantes."""
        return [s.dominant_regime for s in self._snapshots]

    def get_regime_transitions(self) -> List[Dict[str, Any]]:
        """
        Detecta transiciones entre regímenes dominantes.

        Returns:
            Lista de transiciones {t, from, to}
        """
        transitions = []
        seq = self.get_dominant_regime_sequence()

        for i in range(1, len(seq)):
            if seq[i] != seq[i-1]:
                transitions.append({
                    't': self._snapshots[i].t,
                    'from': seq[i-1],
                    'to': seq[i]
                })

        return transitions

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas del Λ-Field."""
        if not self._snapshots:
            return {'t': 0, 'n_snapshots': 0}

        lambda_hist = self.get_lambda_history()
        transitions = self.get_regime_transitions()

        # Tiempo en cada régimen
        dom_seq = self.get_dominant_regime_sequence()
        regime_time = {}
        for r in dom_seq:
            regime_time[r] = regime_time.get(r, 0) + 1

        total = len(dom_seq) if dom_seq else 1
        regime_proportion = {r: t / total for r, t in regime_time.items()}

        return {
            't': self.t,
            'n_snapshots': len(self._snapshots),
            'lambda_mean': float(np.mean(lambda_hist)),
            'lambda_std': float(np.std(lambda_hist)),
            'lambda_current': lambda_hist[-1] if lambda_hist else 0,
            'dominant_now': self._snapshots[-1].dominant_regime if self._snapshots else None,
            'n_transitions': len(transitions),
            'regime_proportion': regime_proportion,
            'metrics_tracked': len(self.metric_names)
        }


class LambdaFieldMultiAgent:
    """
    Λ-Field para sistemas multi-agente.

    Mantiene un LambdaField por agente más uno global
    que agrega las métricas de todos los agentes.
    """

    def __init__(
        self,
        agents: List[str],
        metric_names: Optional[List[str]] = None,
        regime_map: Optional[Dict[str, List[str]]] = None
    ):
        """
        Args:
            agents: Lista de IDs de agentes
            metric_names: Métricas por agente
            regime_map: Mapeo régimen -> métricas
        """
        self.agents = agents

        # Un LambdaField por agente
        self._agent_fields: Dict[str, LambdaField] = {
            agent: LambdaField(metric_names, regime_map)
            for agent in agents
        }

        # LambdaField global (métricas agregadas)
        global_metrics = [f"{m}_mean" for m in (metric_names or [])]
        global_metrics += [f"{m}_var" for m in (metric_names or [])]
        self._global_field = LambdaField(global_metrics, regime_map)

    def step(
        self,
        agent_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, LambdaSnapshot]:
        """
        Paso para todos los agentes.

        Args:
            agent_metrics: {agent_id: {metric_name: value}}

        Returns:
            {agent_id: LambdaSnapshot, "global": LambdaSnapshot}
        """
        results = {}

        # Por agente
        for agent, metrics in agent_metrics.items():
            if agent in self._agent_fields:
                results[agent] = self._agent_fields[agent].step(metrics)

        # Global: agregar métricas
        all_metrics = list(agent_metrics.values())
        if all_metrics:
            # Calcular medias y varianzas
            global_metrics = {}
            metric_keys = set()
            for m in all_metrics:
                metric_keys.update(m.keys())

            for key in metric_keys:
                values = [m.get(key, 0) for m in all_metrics if key in m]
                if values:
                    global_metrics[f"{key}_mean"] = float(np.mean(values))
                    global_metrics[f"{key}_var"] = float(np.var(values))

            results["global"] = self._global_field.step(global_metrics)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Estadísticas de todos los campos."""
        stats = {
            "global": self._global_field.get_statistics()
        }
        for agent in self.agents:
            stats[agent] = self._agent_fields[agent].get_statistics()
        return stats
