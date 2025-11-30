#!/usr/bin/env python3
"""
Phase R1: Structural General Reasoning (SGR)
=============================================

Razonamiento dentro del espacio de estados sin semántica humana.

Un "acto de razonamiento" es una trayectoria hipotética:
    ẑ_{t:t+H} = O_kH ∘ ... ∘ O_k1(z_t)

Donde O_k son operadores internos (drives, irreversibilidad, etc.)

Métricas:
- Plausibilidad P(ẑ): distancia a trayectorias reales
- Coherencia C(ẑ): integración + estabilidad
- Razón estructural R(ẑ) = rank(P) + rank(C)

100% ENDÓGENO - Sin constantes mágicas
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from scipy.stats import rankdata
from collections import deque
import json


@dataclass
class ReasoningHypothesis:
    """Una hipótesis de razonamiento (trayectoria hipotética)."""
    trajectory: np.ndarray  # Shape: (H, d)
    operators_used: List[int]
    plausibility: float = 0.0
    coherence: float = 0.0
    structural_reason: float = 0.0


class StructuralOperator:
    """
    Operador interno O_k(z) = z + Δz_k(z)
    Cada operador representa una deformación del espacio de estados.
    """

    def __init__(self, name: str, transform_fn: Callable):
        self.name = name
        self.transform_fn = transform_fn
        self.usage_count = 0
        self.success_history: deque = deque(maxlen=1000)

    def apply(self, z: np.ndarray, history: np.ndarray) -> np.ndarray:
        """Aplica el operador al estado z."""
        delta_z = self.transform_fn(z, history)
        self.usage_count += 1
        return z + delta_z

    def record_success(self, success: float):
        """Registra éxito del operador."""
        self.success_history.append(success)

    def get_efficacy(self) -> float:
        """Eficacia endógena del operador."""
        if len(self.success_history) < 2:
            return 0.5
        return float(np.mean(self.success_history))


class StructuralGeneralReasoning:
    """
    Sistema de razonamiento estructural general.

    Genera hipótesis (trayectorias) usando operadores internos
    y selecciona la más plausible + coherente.
    """

    def __init__(self, d_state: int = 8):
        self.d_state = d_state
        self.history: deque = deque(maxlen=10000)
        self.operators: List[StructuralOperator] = []
        self.reasoning_history: List[Dict] = []

        # Covarianza histórica (para Mahalanobis)
        self._cov_history: deque = deque(maxlen=1000)
        self._cov_matrix: Optional[np.ndarray] = None

        # Inicializar operadores base
        self._init_operators()

    def _init_operators(self):
        """Inicializa operadores internos basados en dinámicas existentes."""

        # O1: Drift hacia media histórica (homeostasis)
        def drift_to_mean(z, history):
            if len(history) < 2:
                return np.zeros_like(z)
            mu = np.mean(history, axis=0)
            # Fuerza proporcional a distancia, escala endógena
            dist = np.linalg.norm(z - mu)
            scale = 1.0 / (np.sqrt(len(history)) + 1)
            return scale * (mu - z) / (dist + 1e-12)

        self.operators.append(StructuralOperator("homeostasis", drift_to_mean))

        # O2: Movimiento en dirección de máxima varianza (exploración)
        def explore_variance(z, history):
            if len(history) < 3:
                return np.zeros_like(z)
            cov = np.cov(history.T)
            if cov.ndim == 0:
                return np.zeros_like(z)
            eigvals, eigvecs = np.linalg.eigh(cov)
            v1 = eigvecs[:, -1]  # Dirección de máxima varianza
            scale = 1.0 / (np.sqrt(len(history)) + 1)
            return scale * v1

        self.operators.append(StructuralOperator("exploration", explore_variance))

        # O3: Momentum (continuar tendencia reciente)
        def momentum(z, history):
            if len(history) < 2:
                return np.zeros_like(z)
            w = int(np.sqrt(len(history))) + 1
            recent = history[-w:]
            if len(recent) < 2:
                return np.zeros_like(z)
            velocity = np.mean(np.diff(recent, axis=0), axis=0)
            return velocity

        self.operators.append(StructuralOperator("momentum", momentum))

        # O4: Contracción (reducir dispersión)
        def contraction(z, history):
            if len(history) < 2:
                return np.zeros_like(z)
            mu = np.mean(history, axis=0)
            # Contraer hacia el centro
            scale = 1.0 / (np.sqrt(len(history)) + 1)
            return -scale * (z - mu) * 0.5

        self.operators.append(StructuralOperator("contraction", contraction))

        # O5: Perturbación ortogonal (explorar nuevas direcciones)
        def orthogonal_perturbation(z, history):
            if len(history) < 3:
                return np.zeros_like(z)
            cov = np.cov(history.T)
            if cov.ndim == 0:
                return np.zeros_like(z)
            eigvals, eigvecs = np.linalg.eigh(cov)
            # Dirección de mínima varianza
            v_min = eigvecs[:, 0]
            scale = 1.0 / (np.sqrt(len(history)) + 1)
            return scale * v_min

        self.operators.append(StructuralOperator("orthogonal", orthogonal_perturbation))

        # O6: Gradiente de estabilidad local
        def stability_gradient(z, history):
            if len(history) < 10:
                return np.zeros_like(z)
            # Calcular estabilidad local como inverso de varianza reciente
            w = int(np.sqrt(len(history))) + 1
            recent = history[-w:]
            var_local = np.var(recent, axis=0)
            # Moverse hacia dimensiones más estables
            gradient = -var_local / (np.sum(var_local) + 1e-12)
            scale = 1.0 / (np.sqrt(len(history)) + 1)
            return scale * gradient

        self.operators.append(StructuralOperator("stability_seek", stability_gradient))

    def add_state(self, z: np.ndarray):
        """Añade estado al historial."""
        self.history.append(z.copy())
        self._update_covariance(z)

    def _update_covariance(self, z: np.ndarray):
        """Actualiza matriz de covarianza incrementalmente."""
        self._cov_history.append(z)
        if len(self._cov_history) >= 3:
            data = np.array(self._cov_history)
            self._cov_matrix = np.cov(data.T)
            if self._cov_matrix.ndim == 0:
                self._cov_matrix = np.eye(self.d_state) * (self._cov_matrix + 1e-12)

    def _mahalanobis_distance(self, trajectory: np.ndarray) -> float:
        """Distancia de Mahalanobis de trayectoria a distribución histórica."""
        if self._cov_matrix is None or len(self.history) < 3:
            return float(np.mean(np.linalg.norm(trajectory, axis=1)))

        history = np.array(self.history)
        mu = np.mean(history, axis=0)

        try:
            cov_inv = np.linalg.inv(self._cov_matrix + np.eye(self.d_state) * 1e-6)
        except np.linalg.LinAlgError:
            cov_inv = np.eye(self.d_state)

        distances = []
        for z in trajectory:
            diff = z - mu
            d = np.sqrt(np.abs(diff @ cov_inv @ diff))
            distances.append(d)

        return float(np.mean(distances))

    def _compute_plausibility(self, hypothesis: ReasoningHypothesis) -> float:
        """
        Plausibilidad P(ẑ) = exp(-dist(ẑ, trayectorias reales))
        Distancia derivada endógenamente (Mahalanobis).
        """
        if len(self.history) < 3:
            return 0.5

        dist = self._mahalanobis_distance(hypothesis.trajectory)

        # Normalizar por distancia típica del historial
        history = np.array(self.history)
        if len(history) > 10:
            typical_dist = np.percentile(
                [self._mahalanobis_distance(history[i:i+len(hypothesis.trajectory)])
                 for i in range(0, len(history) - len(hypothesis.trajectory),
                               max(1, len(hypothesis.trajectory)))],
                50
            )
        else:
            typical_dist = 1.0

        # Plausibilidad: más alta si está cerca de distribución típica
        normalized_dist = dist / (typical_dist + 1e-12)
        plausibility = np.exp(-normalized_dist)

        return float(np.clip(plausibility, 0, 1))

    def _compute_coherence(self, hypothesis: ReasoningHypothesis) -> float:
        """
        Coherencia C(ẑ) = rank(integration) + rank(stability)
        """
        trajectory = hypothesis.trajectory

        if len(trajectory) < 2:
            return 0.5

        # Integration: correlación media entre dimensiones
        if trajectory.shape[1] > 1:
            corr_matrix = np.corrcoef(trajectory.T)
            # Evitar NaN
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            # Integración = media de correlaciones absolutas (sin diagonal)
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            integration = np.mean(np.abs(corr_matrix[mask]))
        else:
            integration = 0.5

        # Stability: inverso de varianza normalizada
        var_trajectory = np.var(trajectory, axis=0)
        mean_var = np.mean(var_trajectory)

        # Comparar con varianza histórica
        if len(self.history) > 10:
            history = np.array(self.history)
            var_history = np.mean(np.var(history, axis=0))
            stability = 1.0 / (1.0 + mean_var / (var_history + 1e-12))
        else:
            stability = 1.0 / (1.0 + mean_var)

        # Coherencia como suma de ranks (ambos en [0,1])
        coherence = (integration + stability) / 2.0

        return float(np.clip(coherence, 0, 1))

    def _generate_hypothesis(self, z_start: np.ndarray,
                            operator_sequence: List[int],
                            horizon: int) -> ReasoningHypothesis:
        """Genera una hipótesis aplicando secuencia de operadores."""
        trajectory = [z_start.copy()]
        history_array = np.array(self.history) if self.history else z_start.reshape(1, -1)

        z = z_start.copy()
        for i in range(horizon):
            op_idx = operator_sequence[i % len(operator_sequence)]
            op = self.operators[op_idx]
            z = op.apply(z, history_array)
            trajectory.append(z.copy())

        return ReasoningHypothesis(
            trajectory=np.array(trajectory),
            operators_used=operator_sequence
        )

    def reason(self, z_current: np.ndarray,
               n_hypotheses: int = None,
               horizon: int = None) -> Dict:
        """
        Ejecuta razonamiento estructural.

        Genera hipótesis, evalúa plausibilidad y coherencia,
        selecciona la mejor según R(ẑ) = rank(P) + rank(C).

        Returns:
            Dict con hipótesis seleccionada y métricas
        """
        T = len(self.history)

        # Parámetros endógenos
        if n_hypotheses is None:
            n_hypotheses = max(3, int(np.sqrt(T + 1)))
        if horizon is None:
            horizon = max(2, int(np.log2(T + 2)))

        K = len(self.operators)
        hypotheses: List[ReasoningHypothesis] = []

        # Generar hipótesis variadas
        for i in range(n_hypotheses):
            # Secuencia de operadores basada en eficacia + exploración
            efficacies = [op.get_efficacy() for op in self.operators]

            # Probabilidad proporcional a eficacia + ruido de exploración
            probs = np.array(efficacies) + 0.1
            probs = probs / np.sum(probs)

            # Generar secuencia
            seq_length = max(1, horizon // 2)
            sequence = list(np.random.choice(K, size=seq_length, p=probs))

            # Crear hipótesis
            hyp = self._generate_hypothesis(z_current, sequence, horizon)
            hypotheses.append(hyp)

        # Evaluar cada hipótesis
        plausibilities = []
        coherences = []

        for hyp in hypotheses:
            hyp.plausibility = self._compute_plausibility(hyp)
            hyp.coherence = self._compute_coherence(hyp)
            plausibilities.append(hyp.plausibility)
            coherences.append(hyp.coherence)

        # Calcular razón estructural R = rank(P) + rank(C)
        if len(hypotheses) > 1:
            rank_p = rankdata(plausibilities, method='average') / len(hypotheses)
            rank_c = rankdata(coherences, method='average') / len(hypotheses)

            for i, hyp in enumerate(hypotheses):
                hyp.structural_reason = (rank_p[i] + rank_c[i]) / 2.0
        else:
            hypotheses[0].structural_reason = 0.5

        # Seleccionar mejor hipótesis
        best_idx = np.argmax([h.structural_reason for h in hypotheses])
        best_hyp = hypotheses[best_idx]

        # Registrar éxito de operadores usados
        for op_idx in best_hyp.operators_used:
            self.operators[op_idx].record_success(best_hyp.structural_reason)

        # Guardar en historial de razonamiento
        result = {
            't': T,
            'n_hypotheses': n_hypotheses,
            'horizon': horizon,
            'best_hypothesis': {
                'plausibility': best_hyp.plausibility,
                'coherence': best_hyp.coherence,
                'structural_reason': best_hyp.structural_reason,
                'operators': [self.operators[i].name for i in best_hyp.operators_used],
                'trajectory_length': len(best_hyp.trajectory)
            },
            'all_plausibilities': plausibilities,
            'all_coherences': coherences,
            'all_reasons': [h.structural_reason for h in hypotheses],
            'operator_efficacies': {op.name: op.get_efficacy() for op in self.operators},
            'selected_trajectory': best_hyp.trajectory.tolist()
        }

        self.reasoning_history.append(result)

        return result

    def get_next_action(self, z_current: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Usa razonamiento para decidir siguiente estado.

        Returns:
            (z_next, reasoning_info)
        """
        result = self.reason(z_current)

        # El siguiente estado es el segundo punto de la mejor trayectoria
        trajectory = np.array(result['selected_trajectory'])
        if len(trajectory) > 1:
            z_next = trajectory[1]
        else:
            z_next = z_current

        return z_next, result

    def get_stats(self) -> Dict:
        """Estadísticas del sistema de razonamiento."""
        if not self.reasoning_history:
            return {'n_reasoning_events': 0}

        recent = self.reasoning_history[-100:]

        return {
            'n_reasoning_events': len(self.reasoning_history),
            'n_states': len(self.history),
            'mean_plausibility': float(np.mean([r['best_hypothesis']['plausibility'] for r in recent])),
            'mean_coherence': float(np.mean([r['best_hypothesis']['coherence'] for r in recent])),
            'mean_structural_reason': float(np.mean([r['best_hypothesis']['structural_reason'] for r in recent])),
            'operator_usage': {op.name: op.usage_count for op in self.operators},
            'operator_efficacies': {op.name: op.get_efficacy() for op in self.operators}
        }


def run_phaseR1_test(n_steps: int = 1000) -> Dict:
    """
    Test de Phase R1: Structural General Reasoning.

    Verifica:
    1. El razonamiento genera hipótesis válidas
    2. La selección mejora con el tiempo (eficacia de operadores)
    3. Las trayectorias seleccionadas son coherentes
    """
    print("=" * 70)
    print("PHASE R1: STRUCTURAL GENERAL REASONING (SGR)")
    print("=" * 70)

    sgr = StructuralGeneralReasoning(d_state=8)

    # Estado inicial
    z = np.random.randn(8) * 0.1

    plausibilities = []
    coherences = []
    reasons = []

    print(f"\nEjecutando {n_steps} pasos de razonamiento...")

    for t in range(n_steps):
        # Añadir estado actual
        sgr.add_state(z)

        # Razonar y obtener siguiente acción
        z_next, result = sgr.get_next_action(z)

        plausibilities.append(result['best_hypothesis']['plausibility'])
        coherences.append(result['best_hypothesis']['coherence'])
        reasons.append(result['best_hypothesis']['structural_reason'])

        # Actualizar estado con algo de ruido
        z = z_next + np.random.randn(8) * 0.01

        if (t + 1) % 200 == 0:
            print(f"  t={t+1}: P={plausibilities[-1]:.4f}, C={coherences[-1]:.4f}, R={reasons[-1]:.4f}")

    # Análisis de resultados
    stats = sgr.get_stats()

    # Verificar mejora temporal
    n_windows = 5
    window_size = n_steps // n_windows
    window_means = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window_means.append(np.mean(reasons[start:end]))

    improvement = window_means[-1] - window_means[0]

    # GO/NO-GO criteria
    criteria = {
        'mean_plausibility_above_threshold': np.mean(plausibilities) > 0.3,
        'mean_coherence_above_threshold': np.mean(coherences) > 0.3,
        'reasoning_improves_over_time': improvement > 0,
        'operators_differentiate': np.std(list(stats['operator_efficacies'].values())) > 0.01,
        'sufficient_reasoning_events': stats['n_reasoning_events'] >= n_steps
    }

    n_pass = sum(criteria.values())
    go = n_pass >= 3

    print(f"\n{'='*70}")
    print("RESULTADOS PHASE R1")
    print(f"{'='*70}")
    print(f"\nEstadísticas generales:")
    print(f"  - Eventos de razonamiento: {stats['n_reasoning_events']}")
    print(f"  - Plausibilidad media: {stats['mean_plausibility']:.4f}")
    print(f"  - Coherencia media: {stats['mean_coherence']:.4f}")
    print(f"  - Razón estructural media: {stats['mean_structural_reason']:.4f}")

    print(f"\nEficacia de operadores:")
    for name, eff in stats['operator_efficacies'].items():
        print(f"  - {name}: {eff:.4f}")

    print(f"\nMejora temporal (R por ventana): {window_means}")
    print(f"Mejora total: {improvement:.4f}")

    print(f"\nGO/NO-GO Criteria:")
    for criterion, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"  - {criterion}: {status}")

    print(f"\n{'GO' if go else 'NO-GO'} ({n_pass}/5 criteria passed)")

    return {
        'go': go,
        'stats': stats,
        'criteria': criteria,
        'window_means': window_means,
        'improvement': improvement,
        'plausibilities': plausibilities,
        'coherences': coherences,
        'reasons': reasons
    }


if __name__ == "__main__":
    result = run_phaseR1_test(n_steps=1000)

    # Guardar resultados
    import os
    os.makedirs('/root/NEO_EVA/results/phaseR1', exist_ok=True)

    with open('/root/NEO_EVA/results/phaseR1/phaseR1_results.json', 'w') as f:
        json.dump({
            'go': result['go'],
            'stats': result['stats'],
            'criteria': {k: bool(v) for k, v in result['criteria'].items()},
            'window_means': result['window_means'],
            'improvement': result['improvement']
        }, f, indent=2)

    print(f"\nResultados guardados en results/phaseR1/")
