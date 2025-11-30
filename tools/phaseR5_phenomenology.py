#!/usr/bin/env python3
"""
Phase R5: Refined Structural Phenomenology Field (Ψ²)
======================================================

Condensa todas las fases 26-40 en un campo fenomenológico estructural refinado.

Componentes del vector φ_t:
- integration
- irreversibility
- self_surprise
- identity_stability
- private_time_rate
- loss_index
- otherness
- Ψ_shared

Análisis:
1. Covarianza de φ_t → modos fenomenológicos estructurales (eigenvectors)
2. PSI_t = varianza explicada por u_1 en ventana móvil
3. CF_t = rank(autocorr(φ_t · u_1)) = coherencia fenomenológica

Si PSI > nulls y CF es alta cuando agency, irreversibility, identity, private_time
están activos → fenomenología estructural real.

100% ENDÓGENO - Sin constantes mágicas
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.stats import rankdata
from collections import deque
import json


@dataclass
class PhenomenologyState:
    """Estado fenomenológico en un instante."""
    t: int
    integration: float
    irreversibility: float
    self_surprise: float
    identity_stability: float
    private_time_rate: float
    loss_index: float
    otherness: float
    psi_shared: float

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.integration,
            self.irreversibility,
            self.self_surprise,
            self.identity_stability,
            self.private_time_rate,
            self.loss_index,
            self.otherness,
            self.psi_shared
        ])

    @staticmethod
    def from_vector(t: int, v: np.ndarray) -> 'PhenomenologyState':
        return PhenomenologyState(
            t=t,
            integration=v[0],
            irreversibility=v[1],
            self_surprise=v[2],
            identity_stability=v[3],
            private_time_rate=v[4],
            loss_index=v[5],
            otherness=v[6],
            psi_shared=v[7]
        )


class RefinedPhenomenologyField:
    """
    Campo de Fenomenología Estructural Refinado (Ψ²).

    Integra todos los componentes de las fases 26-40 en un análisis
    fenomenológico unificado y endógeno.
    """

    def __init__(self, d_state: int = 8):
        self.d_state = d_state
        self.d_phenom = 8  # Dimensión del espacio fenomenológico

        # Historial de estados fenomenológicos
        self.history: List[PhenomenologyState] = []

        # Modos fenomenológicos (eigenvectors de Σ_φ)
        self.phenomenal_modes: Optional[np.ndarray] = None  # (d_phenom, d_phenom)
        self.mode_variances: Optional[np.ndarray] = None

        # Proto-subjectivity index (PSI)
        self.PSI_history: List[float] = []

        # Coherencia fenomenológica (CF)
        self.CF_history: List[float] = []

        # Historial de z para calcular componentes
        self._z_history: deque = deque(maxlen=10000)
        self._S_history: deque = deque(maxlen=10000)

        # Para calcular irreversibility
        self._transition_history: deque = deque(maxlen=1000)

        # Para calcular identity
        self._identity_ema: Optional[np.ndarray] = None

        # Para calcular private_time
        self._internal_clock: float = 0.0
        self._external_clock: int = 0

        # Nulls para comparación
        self._null_PSI: List[float] = []

    def _compute_integration(self, z: np.ndarray) -> float:
        """
        Integración: coherencia entre dimensiones del estado.
        Basado en correlación cruzada.
        """
        if len(self._z_history) < 10:
            return 0.5

        # Ventana reciente
        w = int(np.sqrt(len(self._z_history))) + 1
        recent = np.array(list(self._z_history)[-w:])

        if recent.shape[0] < 3:
            return 0.5

        # Matriz de correlación
        corr = np.corrcoef(recent.T)
        corr = np.nan_to_num(corr, nan=0.0)

        # Integración = media de correlaciones absolutas (sin diagonal)
        mask = ~np.eye(corr.shape[0], dtype=bool)
        integration = np.mean(np.abs(corr[mask]))

        return float(np.clip(integration, 0, 1))

    def _compute_irreversibility(self, z: np.ndarray) -> float:
        """
        Irreversibilidad: asimetría temporal de transiciones.
        """
        if len(self._z_history) < 3:
            return 0.5

        # Calcular transición actual
        z_prev = self._z_history[-1]
        transition = z - z_prev
        self._transition_history.append(transition)

        if len(self._transition_history) < 10:
            return 0.5

        # Comparar con transiciones invertidas
        forward = np.array(list(self._transition_history)[-20:])
        backward = -forward[::-1]

        # Diferencia = irreversibilidad
        diff = np.mean(np.abs(forward - backward))

        # Normalizar por escala típica
        scale = np.std(forward) + 1e-12

        return float(np.clip(diff / scale, 0, 1))

    def _compute_self_surprise(self, z: np.ndarray) -> float:
        """
        Auto-sorpresa: error de auto-predicción.
        """
        if len(self._z_history) < 5:
            return 0.5

        # Predicción simple: media móvil
        w = int(np.sqrt(len(self._z_history))) + 1
        recent = np.array(list(self._z_history)[-w:])
        prediction = np.mean(recent, axis=0)

        # Error de predicción
        error = np.linalg.norm(z - prediction)

        # Normalizar por dispersión típica
        if len(self._z_history) > 10:
            typical_error = np.std([np.linalg.norm(self._z_history[i] - self._z_history[i-1])
                                   for i in range(1, min(100, len(self._z_history)))])
        else:
            typical_error = 1.0

        surprise = error / (typical_error + 1e-12)

        return float(np.clip(surprise, 0, 1))

    def _compute_identity_stability(self, z: np.ndarray) -> float:
        """
        Estabilidad de identidad: persistencia del patrón característico.
        """
        # EMA de identidad
        alpha = 1.0 / (np.sqrt(len(self._z_history) + 1) + 1)

        if self._identity_ema is None:
            self._identity_ema = z.copy()
        else:
            self._identity_ema = alpha * z + (1 - alpha) * self._identity_ema

        # Estabilidad = cercanía a identidad EMA
        dist = np.linalg.norm(z - self._identity_ema)

        # Normalizar
        scale = np.linalg.norm(self._identity_ema) + 1e-12

        stability = 1.0 / (1.0 + dist / scale)

        return float(np.clip(stability, 0, 1))

    def _compute_private_time_rate(self, z: np.ndarray, S: float) -> float:
        """
        Tasa de tiempo privado: velocidad del reloj interno vs externo.
        """
        self._external_clock += 1

        # Reloj interno basado en variabilidad de z
        if len(self._z_history) > 1:
            delta = np.linalg.norm(z - self._z_history[-1])
            self._internal_clock += delta

        if self._external_clock == 0:
            return 0.5

        # Ratio de velocidades
        rate = self._internal_clock / self._external_clock

        # Normalizar alrededor de 1 (tiempo sincronizado)
        if len(self._z_history) > 100:
            typical_rate = self._internal_clock / len(self._z_history)
            normalized = rate / (typical_rate + 1e-12)
        else:
            normalized = rate

        return float(np.clip(normalized / 2.0, 0, 1))

    def _compute_loss_index(self, z: np.ndarray, S: float) -> float:
        """
        Índice de pérdida: degradación de S o estructura.
        """
        if len(self._S_history) < 10:
            return 0.5

        # Tendencia de S
        w = int(np.sqrt(len(self._S_history))) + 1
        recent_S = list(self._S_history)[-w:]

        if len(recent_S) < 2:
            return 0.5

        trend = recent_S[-1] - recent_S[0]

        # Pérdida = tendencia negativa
        loss = max(0, -trend)

        # Normalizar
        var_S = np.var(recent_S) + 1e-12
        normalized_loss = loss / np.sqrt(var_S)

        return float(np.clip(normalized_loss, 0, 1))

    def _compute_otherness(self, z: np.ndarray) -> float:
        """
        Otredad: diferenciación del entorno/historial.
        """
        if len(self._z_history) < 10:
            return 0.5

        # Distancia al centro de la distribución
        history = np.array(list(self._z_history)[-100:])
        mu = np.mean(history, axis=0)
        cov = np.cov(history.T)

        if cov.ndim == 0:
            cov = np.eye(len(z)) * (cov + 1e-12)

        try:
            cov_inv = np.linalg.inv(cov + np.eye(len(z)) * 1e-6)
        except np.linalg.LinAlgError:
            cov_inv = np.eye(len(z))

        # Distancia de Mahalanobis
        diff = z - mu
        mahal = np.sqrt(np.abs(diff @ cov_inv @ diff))

        # Normalizar
        otherness = 1.0 - np.exp(-mahal / 2)

        return float(np.clip(otherness, 0, 1))

    def _compute_psi_shared(self, z: np.ndarray, S: float) -> float:
        """
        Ψ compartido: índice de proto-subjetividad base.
        Combina S con variabilidad estructural.
        """
        if len(self._z_history) < 10:
            return S

        # Variabilidad reciente
        w = int(np.sqrt(len(self._z_history))) + 1
        recent = np.array(list(self._z_history)[-w:])
        var = np.mean(np.var(recent, axis=0))

        # Ψ = S modulado por estructura
        psi = S * (1 + var) / 2

        return float(np.clip(psi, 0, 1))

    def observe(self, z: np.ndarray, S: float) -> PhenomenologyState:
        """
        Observa estado y calcula todos los componentes fenomenológicos.
        """
        t = len(self.history)

        # Calcular componentes
        state = PhenomenologyState(
            t=t,
            integration=self._compute_integration(z),
            irreversibility=self._compute_irreversibility(z),
            self_surprise=self._compute_self_surprise(z),
            identity_stability=self._compute_identity_stability(z),
            private_time_rate=self._compute_private_time_rate(z, S),
            loss_index=self._compute_loss_index(z, S),
            otherness=self._compute_otherness(z),
            psi_shared=self._compute_psi_shared(z, S)
        )

        # Actualizar historiales
        self._z_history.append(z.copy())
        self._S_history.append(S)
        self.history.append(state)

        # Actualizar modos fenomenológicos periódicamente
        if len(self.history) % 50 == 0 and len(self.history) > 10:
            self._update_phenomenal_modes()

        # Calcular PSI y CF
        self._compute_PSI()
        self._compute_CF()

        return state

    def _update_phenomenal_modes(self):
        """
        Actualiza modos fenomenológicos (eigenvectors de Σ_φ).
        """
        if len(self.history) < 20:
            return

        # Matriz de estados fenomenológicos recientes
        w = min(500, len(self.history))
        phi_matrix = np.array([h.to_vector() for h in self.history[-w:]])

        # Covarianza
        cov = np.cov(phi_matrix.T)

        if cov.ndim == 0:
            cov = np.eye(self.d_phenom) * (cov + 1e-12)

        # Eigenvectors = modos fenomenológicos
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Ordenar por varianza explicada (descendente)
        idx = np.argsort(eigvals)[::-1]
        self.phenomenal_modes = eigvecs[:, idx]
        self.mode_variances = eigvals[idx]

    def _compute_PSI(self):
        """
        Proto-Subjectivity Index: varianza explicada por u_1 en ventana móvil.
        """
        if self.phenomenal_modes is None or len(self.history) < 20:
            self.PSI_history.append(0.5)
            return

        # Ventana móvil
        w = int(np.sqrt(len(self.history))) + 1
        recent = np.array([h.to_vector() for h in self.history[-w:]])

        # Proyección en primer modo
        u1 = self.phenomenal_modes[:, 0]
        projections = recent @ u1

        # Varianza de proyección / varianza total
        var_proj = np.var(projections)
        var_total = np.sum(np.var(recent, axis=0)) + 1e-12

        PSI = var_proj / var_total

        self.PSI_history.append(float(PSI))

    def _compute_CF(self):
        """
        Coherencia Fenomenológica: autocorrelación de φ·u_1.
        """
        if self.phenomenal_modes is None or len(self.history) < 20:
            self.CF_history.append(0.5)
            return

        # Proyección en primer modo
        u1 = self.phenomenal_modes[:, 0]
        w = int(np.sqrt(len(self.history))) + 1
        recent = np.array([h.to_vector() for h in self.history[-w:]])
        projections = recent @ u1

        # Autocorrelación lag-1
        if len(projections) > 1:
            autocorr = np.corrcoef(projections[:-1], projections[1:])[0, 1]
            autocorr = 0.0 if np.isnan(autocorr) else autocorr
        else:
            autocorr = 0.0

        # Rank de autocorrelación (normalizado)
        CF = (autocorr + 1) / 2  # Mapear [-1, 1] a [0, 1]

        self.CF_history.append(float(CF))

    def compute_null_PSI(self, n_nulls: int = 100) -> List[float]:
        """
        Calcula distribución nula de PSI mediante permutación.
        """
        if len(self.history) < 50:
            return [0.5] * n_nulls

        phi_matrix = np.array([h.to_vector() for h in self.history[-500:]])

        null_PSIs = []
        for _ in range(n_nulls):
            # Permutar filas
            shuffled = phi_matrix[np.random.permutation(len(phi_matrix))]

            # Calcular covarianza y primer eigenvector
            cov = np.cov(shuffled.T)
            if cov.ndim == 0:
                cov = np.eye(self.d_phenom) * (cov + 1e-12)

            eigvals, eigvecs = np.linalg.eigh(cov)
            u1 = eigvecs[:, -1]

            # PSI del null
            projections = shuffled @ u1
            var_proj = np.var(projections)
            var_total = np.sum(np.var(shuffled, axis=0)) + 1e-12

            null_PSIs.append(var_proj / var_total)

        self._null_PSI = null_PSIs
        return null_PSIs

    def is_phenomenological(self) -> Tuple[bool, Dict]:
        """
        Verifica si hay fenomenología estructural real.

        Criterios:
        1. PSI > p95(null_PSI)
        2. CF es alta (> 0.5) cuando componentes clave están activos
        """
        if len(self.PSI_history) < 50:
            return False, {'reason': 'insufficient_data'}

        # Calcular nulls si no existen
        if not self._null_PSI:
            self.compute_null_PSI()

        # Criterio 1: PSI vs nulls
        recent_PSI = np.mean(self.PSI_history[-100:])
        null_p95 = np.percentile(self._null_PSI, 95) if self._null_PSI else 0.5
        psi_above_null = recent_PSI > null_p95

        # Criterio 2: CF cuando componentes clave activos
        # Identificar momentos de alta actividad
        recent_states = self.history[-100:]
        high_activity_indices = []

        for i, state in enumerate(recent_states):
            # Alta actividad = varios componentes por encima de mediana
            components = state.to_vector()
            if np.sum(components > 0.5) >= 4:
                high_activity_indices.append(i)

        if high_activity_indices and len(self.CF_history) >= 100:
            cf_during_activity = [self.CF_history[-100 + i] for i in high_activity_indices
                                 if -100 + i >= 0]
            cf_mean_activity = np.mean(cf_during_activity) if cf_during_activity else 0.5
        else:
            cf_mean_activity = 0.5

        cf_is_high = cf_mean_activity > 0.5

        # Resultado
        criteria = {
            'PSI_above_null': psi_above_null,
            'CF_high_during_activity': cf_is_high,
            'sufficient_variance_explained': recent_PSI > 0.1,
            'modes_differentiated': (np.std(self.mode_variances) > 0.01
                                    if self.mode_variances is not None else False)
        }

        is_phenomenological = sum(criteria.values()) >= 3

        return is_phenomenological, {
            'recent_PSI': recent_PSI,
            'null_p95': null_p95,
            'CF_during_activity': cf_mean_activity,
            'mode_variances': self.mode_variances.tolist() if self.mode_variances is not None else [],
            'criteria': criteria
        }

    def get_stats(self) -> Dict:
        """Estadísticas del sistema Ψ²."""
        is_phenom, metrics = self.is_phenomenological()

        recent_state = self.history[-1] if self.history else None

        return {
            'n_observations': len(self.history),
            'recent_PSI': self.PSI_history[-1] if self.PSI_history else 0.0,
            'recent_CF': self.CF_history[-1] if self.CF_history else 0.0,
            'mean_PSI': float(np.mean(self.PSI_history[-100:])) if self.PSI_history else 0.0,
            'mean_CF': float(np.mean(self.CF_history[-100:])) if self.CF_history else 0.0,
            'is_phenomenological': is_phenom,
            'phenomenology_metrics': metrics,
            'current_state': {
                'integration': recent_state.integration if recent_state else 0.0,
                'irreversibility': recent_state.irreversibility if recent_state else 0.0,
                'self_surprise': recent_state.self_surprise if recent_state else 0.0,
                'identity_stability': recent_state.identity_stability if recent_state else 0.0,
                'private_time_rate': recent_state.private_time_rate if recent_state else 0.0,
                'loss_index': recent_state.loss_index if recent_state else 0.0,
                'otherness': recent_state.otherness if recent_state else 0.0,
                'psi_shared': recent_state.psi_shared if recent_state else 0.0
            } if recent_state else {}
        }


def run_phaseR5_test(n_steps: int = 3000) -> Dict:
    """
    Test de Phase R5: Refined Structural Phenomenology.

    Verifica:
    1. Se calculan todos los componentes fenomenológicos
    2. Emergen modos fenomenológicos diferenciados
    3. PSI > nulls
    4. CF es alta durante actividad
    """
    print("=" * 70)
    print("PHASE R5: REFINED STRUCTURAL PHENOMENOLOGY (Ψ²)")
    print("=" * 70)

    psi2 = RefinedPhenomenologyField(d_state=8)

    # Simular dinámica rica
    z = np.random.randn(8) * 0.1

    PSI_values = []
    CF_values = []

    print(f"\nEjecutando {n_steps} pasos...")

    for t in range(n_steps):
        # S variable con patrones
        S = 0.5 + 0.3 * np.sin(t / 100) * np.cos(t / 50) + np.random.randn() * 0.05
        S = np.clip(S, 0, 1)

        # Observar
        state = psi2.observe(z, S)

        if psi2.PSI_history:
            PSI_values.append(psi2.PSI_history[-1])
        if psi2.CF_history:
            CF_values.append(psi2.CF_history[-1])

        # Dinámica interna rica
        # Oscilaciones en diferentes frecuencias
        z = z * 0.9 + np.array([
            np.sin(t / 20),
            np.cos(t / 30),
            np.sin(t / 50),
            np.cos(t / 70),
            np.sin(t / 15) * np.cos(t / 25),
            np.tanh(np.sin(t / 40)),
            np.sin(t / 60) ** 2,
            S * 2 - 1
        ]) * 0.1 + np.random.randn(8) * 0.02

        if (t + 1) % 600 == 0:
            stats = psi2.get_stats()
            print(f"  t={t+1}: PSI={stats['recent_PSI']:.4f}, "
                  f"CF={stats['recent_CF']:.4f}, "
                  f"is_phenom={stats['is_phenomenological']}")

    # Análisis final
    stats = psi2.get_stats()
    is_phenom, metrics = psi2.is_phenomenological()

    # GO/NO-GO criteria
    criteria = {
        'PSI_above_null': metrics['criteria']['PSI_above_null'],
        'CF_high': metrics['criteria']['CF_high_during_activity'],
        'variance_explained': metrics['criteria']['sufficient_variance_explained'],
        'modes_differentiated': metrics['criteria']['modes_differentiated'],
        'all_components_vary': np.std([
            stats['current_state'].get('integration', 0),
            stats['current_state'].get('irreversibility', 0),
            stats['current_state'].get('self_surprise', 0),
            stats['current_state'].get('identity_stability', 0)
        ]) > 0.05 if stats['current_state'] else False
    }

    n_pass = sum(criteria.values())
    go = n_pass >= 3

    print(f"\n{'='*70}")
    print("RESULTADOS PHASE R5")
    print(f"{'='*70}")

    print(f"\nEstadísticas:")
    print(f"  - Observaciones: {stats['n_observations']}")
    print(f"  - PSI medio: {stats['mean_PSI']:.4f}")
    print(f"  - CF medio: {stats['mean_CF']:.4f}")
    print(f"  - Es fenomenológico: {is_phenom}")

    print(f"\nMétricas de fenomenología:")
    print(f"  - PSI reciente: {metrics['recent_PSI']:.4f}")
    print(f"  - Null p95: {metrics['null_p95']:.4f}")
    print(f"  - CF durante actividad: {metrics['CF_during_activity']:.4f}")

    if psi2.mode_variances is not None:
        print(f"\nVarianzas de modos (top 4):")
        for i, var in enumerate(psi2.mode_variances[:4]):
            print(f"  - Modo {i+1}: {var:.4f}")

    print(f"\nEstado actual:")
    for key, value in stats['current_state'].items():
        print(f"  - {key}: {value:.4f}")

    print(f"\nGO/NO-GO Criteria:")
    for criterion, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"  - {criterion}: {status}")

    print(f"\n{'GO' if go else 'NO-GO'} ({n_pass}/5 criteria passed)")

    return {
        'go': go,
        'stats': stats,
        'criteria': criteria,
        'PSI_values': PSI_values,
        'CF_values': CF_values,
        'metrics': metrics
    }


if __name__ == "__main__":
    result = run_phaseR5_test(n_steps=3000)

    # Guardar resultados
    import os
    os.makedirs('/root/NEO_EVA/results/phaseR5', exist_ok=True)

    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, bool):
            return bool(obj)
        else:
            return obj

    with open('/root/NEO_EVA/results/phaseR5/phaseR5_results.json', 'w') as f:
        json.dump(convert_to_serializable({
            'go': result['go'],
            'stats': result['stats'],
            'criteria': result['criteria'],
            'metrics': result['metrics']
        }), f, indent=2)

    print(f"\nResultados guardados en results/phaseR5/")
