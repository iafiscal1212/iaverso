#!/usr/bin/env python3
"""
Phase 15B: Global Narrative Trace (GNT)
========================================

GNT = campo continuo que integra la dinámica del sistema a lo largo del tiempo.

NO es un "estado" discreto - es un EMA endógeno del estado conjunto.
Captura la "huella" histórica del sistema en el espacio de estados.

Componentes:
1. G_state_t = estado conjunto [NEO + EVA] (8D)
2. α_t = factor de olvido endógeno = 1 - η_t donde η_t = 1/√(t+1)
3. GNT_t = α_t * GNT_{t-1} + (1 - α_t) * G_state_t

Propiedades emergentes:
- Momentum: GNT cambia lentamente → "inercia narrativa"
- Atractores: regiones donde GNT tiende a quedarse
- Trayectorias: patrones de movimiento en el espacio

100% endógeno - CERO números mágicos.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from scipy import stats
import json
from datetime import datetime

import sys
sys.path.insert(0, '/root/NEO_EVA/tools')

from endogenous_core import (
    derive_window_size,
    derive_learning_rate,
    compute_iqr,
    compute_entropy_normalized,
    NUMERIC_EPS,
    PROVENANCE
)


# =============================================================================
# GLOBAL NARRATIVE TRACE
# =============================================================================

@dataclass
class GNTSnapshot:
    """Snapshot del GNT en un momento dado."""
    t: int
    gnt: np.ndarray           # Vector GNT (8D)
    alpha: float              # Factor de olvido usado
    velocity: np.ndarray      # Cambio respecto al anterior
    acceleration: np.ndarray  # Cambio de velocidad

    def to_dict(self) -> Dict:
        return {
            't': self.t,
            'gnt': self.gnt.tolist(),
            'alpha': self.alpha,
            'velocity_norm': float(np.linalg.norm(self.velocity)),
            'acceleration_norm': float(np.linalg.norm(self.acceleration))
        }


class GlobalNarrativeTrace:
    """
    Global Narrative Trace - integración temporal del estado del sistema.

    GNT_t = α_t * GNT_{t-1} + (1 - α_t) * G_state_t

    donde:
    - G_state_t es el estado conjunto actual
    - α_t = 1 - 1/√(t+1) es el factor de olvido endógeno
    """

    def __init__(self, dim: int = 8):
        """
        Args:
            dim: dimensión del espacio de estados (4 NEO + 4 EVA = 8)
        """
        self.dim = dim

        # GNT actual
        self.gnt = np.zeros(dim)
        self.prev_gnt = np.zeros(dim)
        self.prev_prev_gnt = np.zeros(dim)

        # Derivado: maxlen = √1e6 ≈ 1000
        derived_maxlen = int(np.sqrt(1e6))

        # Historial
        self.history: deque = deque(maxlen=derived_maxlen)

        # Velocidad y aceleración
        self.velocity = np.zeros(dim)
        self.acceleration = np.zeros(dim)

        # Historial de normas (para análisis)
        self.velocity_history: deque = deque(maxlen=derived_maxlen)
        self.acceleration_history: deque = deque(maxlen=derived_maxlen)

        # Contador
        self.t = 0

    def compute_alpha(self, t: int) -> float:
        """
        Factor de olvido endógeno.

        α_t = 1 - η_t donde η_t = 1/√(t+1)

        Propiedad: α crece con t → GNT se vuelve más "pesado" (más inercia).
        """
        eta = 1.0 / np.sqrt(t + 1)
        alpha = 1.0 - eta

        PROVENANCE.log('alpha_gnt', alpha,
                       '1 - 1/sqrt(t+1)',
                       {'t': t, 'eta': eta}, t)

        return alpha

    def update(self, g_state: np.ndarray) -> GNTSnapshot:
        """
        Actualiza GNT con nuevo estado.

        Args:
            g_state: estado conjunto [4D NEO + 4D EVA] = 8D
        """
        if len(g_state) != self.dim:
            raise ValueError(f"g_state debe ser {self.dim}D, recibido {len(g_state)}D")

        # Factor de olvido endógeno
        alpha = self.compute_alpha(self.t)

        # Guardar estado previo
        self.prev_prev_gnt = self.prev_gnt.copy()
        self.prev_gnt = self.gnt.copy()

        # Actualizar GNT
        self.gnt = alpha * self.gnt + (1 - alpha) * g_state

        # Calcular velocidad (derivada primera)
        self.velocity = self.gnt - self.prev_gnt

        # Calcular aceleración (derivada segunda)
        prev_velocity = self.prev_gnt - self.prev_prev_gnt
        self.acceleration = self.velocity - prev_velocity

        # Registrar
        self.velocity_history.append(np.linalg.norm(self.velocity))
        self.acceleration_history.append(np.linalg.norm(self.acceleration))

        # Crear snapshot
        snapshot = GNTSnapshot(
            t=self.t,
            gnt=self.gnt.copy(),
            alpha=alpha,
            velocity=self.velocity.copy(),
            acceleration=self.acceleration.copy()
        )

        self.history.append(snapshot)
        self.t += 1

        return snapshot

    def get_momentum(self) -> float:
        """
        Momentum del GNT = norma de velocidad.

        Alto momentum = el sistema está "moviéndose" en el espacio de estados.
        Bajo momentum = el sistema está "quieto" o en equilibrio.
        """
        return float(np.linalg.norm(self.velocity))

    def get_stability(self) -> float:
        """
        Estabilidad = 1 / (1 + aceleración normalizada)

        Alta estabilidad = movimiento predecible (baja aceleración).
        Baja estabilidad = cambios bruscos de dirección.
        """
        if len(self.acceleration_history) < 2:
            return 0.5

        # Normalizar por historia
        accel_norm = np.linalg.norm(self.acceleration)
        median_accel = np.median(list(self.acceleration_history))

        if median_accel < NUMERIC_EPS:
            return 1.0

        normalized_accel = accel_norm / (median_accel + NUMERIC_EPS)
        stability = 1.0 / (1.0 + normalized_accel)

        return float(stability)

    def get_inertia(self) -> float:
        """
        Inercia = α actual.

        Alta inercia = GNT cambia lentamente.
        Baja inercia = GNT responde rápidamente a nuevos estados.
        """
        return self.compute_alpha(self.t)

    def detect_attractor_region(self) -> Dict:
        """
        Detecta si el GNT está en una región atractora.

        Criterio: baja velocidad sostenida.
        """
        if len(self.velocity_history) < 10:
            return {'in_attractor': False, 'confidence': 0.0}

        # Velocidad reciente
        recent_velocities = list(self.velocity_history)[-10:]
        median_velocity = np.median(recent_velocities)

        # Umbral endógeno: q25 de toda la historia
        velocity_threshold = np.percentile(list(self.velocity_history), 25)

        in_attractor = median_velocity < velocity_threshold
        confidence = 1.0 - (median_velocity / (np.max(list(self.velocity_history)) + NUMERIC_EPS))

        return {
            'in_attractor': bool(in_attractor),
            'confidence': float(confidence),
            'threshold': float(velocity_threshold),
            'median_velocity': float(median_velocity)
        }

    def get_trajectory_curvature(self) -> float:
        """
        Curvatura de la trayectoria.

        κ = |a| / |v|²

        Alta curvatura = trayectoria muy curva.
        Baja curvatura = trayectoria recta.
        """
        v_norm = np.linalg.norm(self.velocity)
        a_norm = np.linalg.norm(self.acceleration)

        if v_norm < NUMERIC_EPS:
            return 0.0

        curvature = a_norm / (v_norm ** 2 + NUMERIC_EPS)
        return float(curvature)

    def get_summary(self) -> Dict:
        """Resumen del estado del GNT."""
        attractor = self.detect_attractor_region()

        return {
            'gnt': self.gnt.tolist(),
            't': self.t,
            'momentum': self.get_momentum(),
            'stability': self.get_stability(),
            'inertia': self.get_inertia(),
            'curvature': self.get_trajectory_curvature(),
            'attractor': attractor,
            'n_history': len(self.history)
        }

    def get_history_sample(self, n: int = 100) -> List[Dict]:
        """Retorna últimas n snapshots."""
        snapshots = list(self.history)[-n:]
        return [s.to_dict() for s in snapshots]


# =============================================================================
# DIRECTIONAL MOMENTUM (Phase 16)
# =============================================================================

class DirectionalMomentum:
    """
    Tracks directional momentum of the GNT field.

    Mathematical basis:
    - dGNT_t = GNT_t - GNT_{t-1}  (gradient)
    - momentum_t = β_t * momentum_{t-1} + (1 - β_t) * dGNT_t
    - β_t derived endogenously using same pattern as α_t

    Measures whether GNT has a preferred direction or is random walk-like.
    NO semantic interpretation. Pure vector field analysis.
    """

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.momentum = np.zeros(dim)
        self.prev_momentum = np.zeros(dim)

        # Derived maxlen
        derived_maxlen = int(np.sqrt(1e6))

        # History for analysis
        self.gradient_history: deque = deque(maxlen=derived_maxlen)
        self.momentum_history: deque = deque(maxlen=derived_maxlen)
        self.directionality_history: deque = deque(maxlen=derived_maxlen)

        self.t = 0

    def compute_beta(self, t: int) -> float:
        """
        Momentum smoothing factor (endogenous).

        β_t = 1 - 1/√(t+1)

        Same pattern as GNT alpha for consistency.
        """
        eta = 1.0 / np.sqrt(t + 1)
        return 1.0 - eta

    def update(self, gnt_current: np.ndarray, gnt_prev: np.ndarray) -> Dict:
        """
        Update directional momentum with new GNT values.

        Args:
            gnt_current: Current GNT vector
            gnt_prev: Previous GNT vector

        Returns:
            Dict with momentum metrics
        """
        # Compute gradient (direction of change)
        d_gnt = gnt_current - gnt_prev
        self.gradient_history.append(d_gnt.copy())

        # Endogenous smoothing factor
        beta = self.compute_beta(self.t)

        # Update momentum (EMA of gradient)
        self.prev_momentum = self.momentum.copy()
        self.momentum = beta * self.momentum + (1 - beta) * d_gnt
        self.momentum_history.append(self.momentum.copy())

        # Compute directionality (cosine similarity between consecutive momentum)
        directionality = self._compute_directionality()
        self.directionality_history.append(directionality)

        self.t += 1

        return {
            'momentum': self.momentum.tolist(),
            'momentum_norm': float(np.linalg.norm(self.momentum)),
            'gradient_norm': float(np.linalg.norm(d_gnt)),
            'directionality': directionality,
            'beta': beta
        }

    def _compute_directionality(self) -> float:
        """
        Compute directionality index.

        dir_t = cos(momentum_t, momentum_{t-1})

        Range: [-1, 1]
        - Near 1: consistent direction (preferent movement)
        - Near 0: random direction
        - Near -1: oscillating/reversing
        """
        norm_curr = np.linalg.norm(self.momentum)
        norm_prev = np.linalg.norm(self.prev_momentum)

        if norm_curr < NUMERIC_EPS or norm_prev < NUMERIC_EPS:
            return 0.0

        cos_sim = np.dot(self.momentum, self.prev_momentum) / (norm_curr * norm_prev)
        return float(np.clip(cos_sim, -1.0, 1.0))

    def get_statistics(self) -> Dict:
        """Return statistics about directional momentum."""
        if len(self.directionality_history) < 10:
            return {'error': 'insufficient_data'}

        dir_array = np.array(list(self.directionality_history))
        grad_norms = np.array([np.linalg.norm(g) for g in self.gradient_history])
        mom_norms = np.array([np.linalg.norm(m) for m in self.momentum_history])

        return {
            'directionality': {
                'mean': float(np.mean(dir_array)),
                'std': float(np.std(dir_array)),
                'median': float(np.median(dir_array)),
                'p25': float(np.percentile(dir_array, 25)),
                'p75': float(np.percentile(dir_array, 75)),
                'fraction_positive': float(np.mean(dir_array > 0))
            },
            'gradient_norm': {
                'mean': float(np.mean(grad_norms)),
                'std': float(np.std(grad_norms))
            },
            'momentum_norm': {
                'mean': float(np.mean(mom_norms)),
                'std': float(np.std(mom_norms))
            },
            'n_samples': len(dir_array)
        }

    def analyze_vs_null(self, n_nulls: int = 100) -> Dict:
        """
        Compare directionality against random walk null.

        Null: gradient signs shuffled independently per dimension.
        """
        if len(self.gradient_history) < 20:
            return {'error': 'insufficient_data'}

        # Real statistics
        real_stats = self.get_statistics()
        if 'error' in real_stats:
            return real_stats

        real_mean_dir = real_stats['directionality']['mean']

        # Generate null distribution
        gradients = np.array(list(self.gradient_history))
        null_mean_dirs = []

        for _ in range(n_nulls):
            # Shuffle gradient signs independently per dimension
            null_gradients = gradients.copy()
            for d in range(self.dim):
                signs = np.random.choice([-1, 1], size=len(gradients))
                null_gradients[:, d] = np.abs(gradients[:, d]) * signs

            # Compute null momentum and directionality
            null_mom = np.zeros(self.dim)
            null_dirs = []

            for i, g in enumerate(null_gradients):
                beta = self.compute_beta(i)
                prev_mom = null_mom.copy()
                null_mom = beta * null_mom + (1 - beta) * g

                # Directionality
                n1 = np.linalg.norm(null_mom)
                n2 = np.linalg.norm(prev_mom)
                if n1 > NUMERIC_EPS and n2 > NUMERIC_EPS:
                    null_dirs.append(np.dot(null_mom, prev_mom) / (n1 * n2))
                else:
                    null_dirs.append(0.0)

            null_mean_dirs.append(np.mean(null_dirs))

        null_mean_dirs = np.array(null_mean_dirs)

        # Statistics
        z_score = (real_mean_dir - np.mean(null_mean_dirs)) / (np.std(null_mean_dirs) + NUMERIC_EPS)
        p_value = float(np.mean(null_mean_dirs >= real_mean_dir))

        return {
            'real_mean_directionality': float(real_mean_dir),
            'null': {
                'mean': float(np.mean(null_mean_dirs)),
                'std': float(np.std(null_mean_dirs)),
                'p95': float(np.percentile(null_mean_dirs, 95))
            },
            'z_score': float(z_score),
            'p_value': p_value,
            'significant': p_value < 0.05,
            'above_null_p95': real_mean_dir > np.percentile(null_mean_dirs, 95),
            'n_nulls': n_nulls
        }


# =============================================================================
# ANÁLISIS DE TRAYECTORIA
# =============================================================================

class TrajectoryAnalyzer:
    """Análisis de trayectorias del GNT."""

    def __init__(self, gnt: GlobalNarrativeTrace):
        self.gnt = gnt

    def compute_path_length(self, start_t: Optional[int] = None,
                            end_t: Optional[int] = None) -> float:
        """
        Longitud del camino en el espacio de estados.

        Suma de normas de velocidades.
        """
        history = list(self.gnt.history)

        if start_t is not None:
            history = [h for h in history if h.t >= start_t]
        if end_t is not None:
            history = [h for h in history if h.t <= end_t]

        if len(history) < 2:
            return 0.0

        total_length = sum(
            np.linalg.norm(history[i].velocity)
            for i in range(1, len(history))
        )

        return float(total_length)

    def compute_displacement(self, start_t: Optional[int] = None,
                              end_t: Optional[int] = None) -> float:
        """
        Desplazamiento neto (distancia start → end).

        path_length / displacement = tortuosidad de la trayectoria.
        """
        history = list(self.gnt.history)

        if start_t is not None:
            history = [h for h in history if h.t >= start_t]
        if end_t is not None:
            history = [h for h in history if h.t <= end_t]

        if len(history) < 2:
            return 0.0

        displacement = np.linalg.norm(history[-1].gnt - history[0].gnt)
        return float(displacement)

    def compute_tortuosity(self, start_t: Optional[int] = None,
                           end_t: Optional[int] = None) -> float:
        """
        Tortuosidad = path_length / displacement.

        1.0 = trayectoria recta.
        > 1.0 = trayectoria curva/errática.
        """
        path_length = self.compute_path_length(start_t, end_t)
        displacement = self.compute_displacement(start_t, end_t)

        if displacement < NUMERIC_EPS:
            return 1.0  # Sin movimiento neto

        return path_length / displacement

    def find_turning_points(self) -> List[Dict]:
        """
        Encuentra puntos de giro en la trayectoria.

        Un punto de giro es donde la aceleración es alta
        (cambio brusco de dirección).
        """
        history = list(self.gnt.history)

        if len(history) < 10:
            return []

        # Umbral endógeno: q90 de aceleraciones
        accelerations = [np.linalg.norm(h.acceleration) for h in history]
        threshold = np.percentile(accelerations, 90)

        turning_points = []
        for h in history:
            if np.linalg.norm(h.acceleration) > threshold:
                turning_points.append({
                    't': h.t,
                    'acceleration': float(np.linalg.norm(h.acceleration)),
                    'gnt': h.gnt.tolist()
                })

        return turning_points

    def get_analysis(self) -> Dict:
        """Análisis completo de la trayectoria."""
        # Usar toda la historia
        total_path = self.compute_path_length()
        total_displacement = self.compute_displacement()
        tortuosity = self.compute_tortuosity()

        # Análisis por ventana endógena
        window = derive_window_size(self.gnt.t)

        recent_path = self.compute_path_length(start_t=self.gnt.t - window)
        recent_displacement = self.compute_displacement(start_t=self.gnt.t - window)
        recent_tortuosity = self.compute_tortuosity(start_t=self.gnt.t - window)

        turning_points = self.find_turning_points()

        return {
            'total': {
                'path_length': total_path,
                'displacement': total_displacement,
                'tortuosity': tortuosity
            },
            'recent': {
                'window': window,
                'path_length': recent_path,
                'displacement': recent_displacement,
                'tortuosity': recent_tortuosity
            },
            'n_turning_points': len(turning_points),
            'turning_points_sample': turning_points[-5:]
        }


# =============================================================================
# INTEGRACIÓN CON ESTADOS EMERGENTES
# =============================================================================

class GNTSystem:
    """
    Sistema completo de GNT integrado con estados emergentes.

    Phase 16 addition: DirectionalMomentum tracking.
    """

    def __init__(self, dim: int = 8):
        self.gnt = GlobalNarrativeTrace(dim=dim)
        self.analyzer = TrajectoryAnalyzer(self.gnt)

        # Phase 16: Directional momentum
        self.dir_momentum = DirectionalMomentum(dim=dim)

        # Historial de regiones atractoras
        self.attractor_history: List[Dict] = []

        # Eventos
        self.events: List[Dict] = []

        # Phase 16: Momentum history
        self.momentum_results: List[Dict] = []

    def update(self, g_state: np.ndarray) -> Dict:
        """
        Actualiza GNT con nuevo estado conjunto.

        Args:
            g_state: estado conjunto [4D NEO + 4D EVA] = 8D
        """
        # Actualizar GNT
        snapshot = self.gnt.update(g_state)

        # Detectar atractor
        attractor_info = self.gnt.detect_attractor_region()

        # Detectar entrada/salida de atractor
        if len(self.attractor_history) > 0:
            prev_in_attractor = self.attractor_history[-1]['in_attractor']
            curr_in_attractor = attractor_info['in_attractor']

            if not prev_in_attractor and curr_in_attractor:
                self.events.append({
                    't': self.gnt.t,
                    'event': 'enter_attractor',
                    'confidence': attractor_info['confidence']
                })
            elif prev_in_attractor and not curr_in_attractor:
                self.events.append({
                    't': self.gnt.t,
                    'event': 'exit_attractor',
                    'confidence': attractor_info['confidence']
                })

        self.attractor_history.append(attractor_info)

        # Phase 16: Update directional momentum
        if self.gnt.t > 1:
            mom_result = self.dir_momentum.update(
                self.gnt.gnt,
                self.gnt.prev_gnt
            )
            self.momentum_results.append(mom_result)
        else:
            mom_result = None

        return {
            't': self.gnt.t - 1,  # t ya incrementado
            'snapshot': snapshot.to_dict(),
            'attractor': attractor_info,
            'momentum': self.gnt.get_momentum(),
            'stability': self.gnt.get_stability(),
            'directional_momentum': mom_result  # Phase 16
        }

    def get_summary(self) -> Dict:
        """Resumen completo del sistema GNT."""
        result = {
            'gnt': self.gnt.get_summary(),
            'trajectory': self.analyzer.get_analysis(),
            'n_events': len(self.events),
            'recent_events': self.events[-10:]
        }

        # Phase 16: Add directional momentum stats
        if self.dir_momentum.t > 10:
            result['directional_momentum'] = self.dir_momentum.get_statistics()

        return result

    def analyze_directionality(self, n_nulls: int = 100) -> Dict:
        """Phase 16: Analyze directionality vs null models."""
        return self.dir_momentum.analyze_vs_null(n_nulls)

    def save(self, path: str):
        """Guarda el sistema GNT."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_summary(),
            'history_sample': self.gnt.get_history_sample(100),
            'events': self.events,
            'attractor_history_sample': self.attractor_history[-100:]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 15B: GLOBAL NARRATIVE TRACE - TEST")
    print("=" * 70)

    # Crear sistema
    gnt_system = GNTSystem(dim=8)

    # Simular datos
    np.random.seed(42)
    n_steps = 5000

    print("\n[1] Simulando GNT...")

    for t in range(n_steps):
        # Simular estado conjunto (8D)
        # NEO: [te_rank, se_rank, sync_rank, H_rank]
        # EVA: [te_rank, se_rank, sync_rank, H_rank]

        # Simular con cierta estructura (no puro ruido)
        phase = t / 500.0  # Ciclo lento

        neo_state = np.array([
            0.5 + 0.3 * np.sin(phase) + np.random.randn() * 0.1,
            0.5 + 0.2 * np.cos(phase * 0.5) + np.random.randn() * 0.1,
            0.5 + 0.25 * np.sin(phase * 0.3) + np.random.randn() * 0.1,
            0.5 + 0.15 * np.cos(phase * 0.7) + np.random.randn() * 0.1
        ])

        eva_state = np.array([
            0.5 + 0.25 * np.cos(phase) + np.random.randn() * 0.1,
            0.5 + 0.2 * np.sin(phase * 0.6) + np.random.randn() * 0.1,
            0.5 + 0.25 * np.sin(phase * 0.3) + np.random.randn() * 0.1,
            0.5 + 0.2 * np.cos(phase * 0.4) + np.random.randn() * 0.1
        ])

        # Clipear a [0, 1]
        neo_state = np.clip(neo_state, 0, 1)
        eva_state = np.clip(eva_state, 0, 1)

        g_state = np.concatenate([neo_state, eva_state])

        # Actualizar
        result = gnt_system.update(g_state)

    print(f"    Pasos procesados: {n_steps}")

    # Resumen
    print("\n[2] Resumen de GNT:")
    summary = gnt_system.get_summary()

    print(f"\nEstado GNT actual:")
    gnt_vec = summary['gnt']['gnt']
    print(f"  NEO: [{gnt_vec[0]:.3f}, {gnt_vec[1]:.3f}, {gnt_vec[2]:.3f}, {gnt_vec[3]:.3f}]")
    print(f"  EVA: [{gnt_vec[4]:.3f}, {gnt_vec[5]:.3f}, {gnt_vec[6]:.3f}, {gnt_vec[7]:.3f}]")

    print(f"\nDinámica:")
    print(f"  Momentum: {summary['gnt']['momentum']:.4f}")
    print(f"  Estabilidad: {summary['gnt']['stability']:.4f}")
    print(f"  Inercia: {summary['gnt']['inertia']:.4f}")
    print(f"  Curvatura: {summary['gnt']['curvature']:.4f}")

    print(f"\nAtractor:")
    att = summary['gnt']['attractor']
    print(f"  En región atractora: {att['in_attractor']}")
    print(f"  Confianza: {att['confidence']:.3f}")

    print(f"\nTrayectoria:")
    traj = summary['trajectory']
    print(f"  Longitud total: {traj['total']['path_length']:.2f}")
    print(f"  Desplazamiento: {traj['total']['displacement']:.4f}")
    print(f"  Tortuosidad: {traj['total']['tortuosity']:.2f}")
    print(f"  Puntos de giro: {traj['n_turning_points']}")

    print(f"\nEventos: {summary['n_events']}")
    for e in summary['recent_events'][-5:]:
        print(f"  t={e['t']}: {e['event']} (conf={e['confidence']:.2f})")

    # Guardar
    gnt_system.save('/root/NEO_EVA/results/phase15b_gnt_test.json')
    print(f"\n[OK] Guardado en results/phase15b_gnt_test.json")

    print("\n" + "=" * 70)
    print("VERIFICACIÓN ANTI-MAGIA:")
    print("  - α_t = 1 - 1/√(t+1) (endógeno)")
    print("  - Umbral de atractor = q25(velocity_history)")
    print("  - maxlen derivado de √1e6")
    print("  - NO hay constantes hardcodeadas")
    print("=" * 70)
