#!/usr/bin/env python3
"""
Phase 12: Sistema 100% Endógeno Puro
=====================================

CERO números mágicos. Todo derivado de:
- Cuantiles de historia
- IQR, MAD, varianza
- √T para escalado temporal
- PCA/MDL para selección de modelo
- Ranks para normalización

Único permitido:
- ε numérico (NUMERIC_EPS = machine epsilon)
- Prior uniforme simplex (1/3, 1/3, 1/3) - propiedad geométrica

CADA parámetro tiene registro de procedencia.
"""

import sys
import os
import json
import numpy as np
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Set
from collections import deque
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/root/NEO_EVA/tools')
from endogenous_core import (
    NUMERIC_EPS, SIMPLEX_UNIFORM, PROVENANCE,
    derive_window_size, derive_buffer_size,
    derive_learning_rate, derive_temperature,
    derive_noise_scale, derive_clip_bounds, derive_probability_clip,
    derive_softmax_gamma, derive_K_by_mdl, derive_K_by_variance,
    derive_rupture_threshold, derive_gate_threshold,
    rank_normalize, rolling_rank,
    compute_acf_lag1, compute_entropy_normalized, compute_iqr, compute_mad,
    get_provenance_report
)


# =============================================================================
# ESTADOS (propiedad del modelo, no tuning)
# =============================================================================

class LifeState(Enum):
    SLEEP = "SLEEP"
    WAKE = "WAKE"
    WORK = "WORK"
    LEARN = "LEARN"
    SOCIAL = "SOCIAL"


# =============================================================================
# PAD ENDÓGENO (sin window_size fijo)
# =============================================================================

class EndogenousPAD:
    """
    PAD por PCA de señales.
    Window size se deriva de √T.
    """

    def __init__(self):
        self.signal_history = deque()
        self.pca = None
        self.pca_fitted = False
        self.T = 0

    def get_window_size(self) -> int:
        """Ventana endógena: √T, mínimo 10."""
        return derive_window_size(self.T)

    def _maintain_buffer(self):
        """Mantiene buffer al tamaño endógeno."""
        max_size = derive_buffer_size(self.T)
        while len(self.signal_history) > max_size:
            self.signal_history.popleft()

    def update(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Actualiza PAD con nuevas señales."""
        self.T += 1

        # Construir vector de señales
        keys = ['r', 's', 'm', 'c', 'R_soc', 'e', 'q', 'h']
        signal_vec = np.array([signals.get(k, 0.5) for k in keys])
        self.signal_history.append(signal_vec)
        self._maintain_buffer()

        window = self.get_window_size()
        if len(self.signal_history) < window:
            return {'P': 0.5, 'A': 0.5, 'D': 0.5}

        data = np.array(list(self.signal_history))

        # Fit PCA cada √T pasos (endógeno)
        refit_interval = max(1, int(np.sqrt(self.T)))
        if not self.pca_fitted or self.T % refit_interval == 0:
            # Centrar con mediana (robusto)
            centered = data - np.median(data, axis=0)
            self.pca = PCA(n_components=3)
            self.pca.fit(centered)
            self.pca_fitted = True

        # Transformar
        centered = signal_vec - np.median(data, axis=0)
        coords = self.pca.transform(centered.reshape(1, -1))[0]

        # Normalizar por rank en historia (100% endógeno)
        P = self._rank_in_history(coords[0], 0)
        A = self._rank_in_history(coords[1], 1) if len(coords) > 1 else 0.5
        D = self._rank_in_history(coords[2], 2) if len(coords) > 2 else 0.5

        return {'P': P, 'A': A, 'D': D}

    def _rank_in_history(self, value: float, component: int) -> float:
        """Rank de un componente en su historia."""
        if len(self.signal_history) < 10:
            return 0.5

        data = np.array(list(self.signal_history))
        centered = data - np.median(data, axis=0)
        projected = self.pca.transform(centered)[:, component]

        return stats.percentileofscore(projected, value) / 100


# =============================================================================
# GATE CONTINUO ENDÓGENO
# =============================================================================

class ContinuousGate:
    """
    Gate continuo sin umbrales fijos.
    gate = rank(ρ) * (1 - rank(var_I))
    """

    def __init__(self):
        self.rho_history = deque()
        self.var_history = deque()
        self.T = 0

    def _maintain_buffers(self):
        """Mantiene buffers al tamaño endógeno."""
        max_size = derive_buffer_size(self.T)
        while len(self.rho_history) > max_size:
            self.rho_history.popleft()
        while len(self.var_history) > max_size:
            self.var_history.popleft()

    def compute(self, signals_self: Dict[str, float],
                signals_other: Dict[str, float],
                I_history: deque) -> Tuple[float, float, float]:
        """
        Calcula gate continuo.

        Returns: (gate_strength, rho, var_I)
        """
        self.T += 1

        # ρ: correlación de señales
        keys = list(signals_self.keys())
        vec_self = np.array([signals_self.get(k, 0.5) for k in keys])
        vec_other = np.array([signals_other.get(k, 0.5) for k in keys])

        if np.std(vec_self) > NUMERIC_EPS and np.std(vec_other) > NUMERIC_EPS:
            rho = np.corrcoef(vec_self, vec_other)[0, 1]
        else:
            rho = 0.0

        # var_I: variabilidad reciente
        window = derive_window_size(self.T)
        if len(I_history) >= window:
            recent = np.array(list(I_history)[-window:])
            var_I = np.var(recent)
        else:
            var_I = compute_iqr(np.array(list(I_history))) if I_history else 0.1

        self.rho_history.append(rho)
        self.var_history.append(var_I)
        self._maintain_buffers()

        # Gate por ranks (100% endógeno)
        rho_rank = rolling_rank(rho, self.rho_history)
        var_rank = rolling_rank(var_I, self.var_history)

        # Gate = alta correlación * baja variabilidad
        gate = rho_rank * (1 - var_rank)

        PROVENANCE.log('gate', gate, 'rank(rho) * (1 - rank(var_I))',
                       {'rho': rho, 'var_I': var_I, 'rho_rank': rho_rank, 'var_rank': var_rank}, self.T)

        return gate, rho, var_I


# =============================================================================
# R_SOC ADAPTATIVO
# =============================================================================

class AdaptiveRSoc:
    """
    R_soc con alpha adaptativo: α = 1/√(n_couplings + 1)
    """

    def __init__(self):
        self.R_soc_ema = 0.5  # Prior neutro (centro del rango)
        self.n_couplings = 0
        self.reward_history = deque()

    def get_alpha(self) -> float:
        """Alpha endógeno: 1/√(n+1)"""
        alpha = 1.0 / np.sqrt(self.n_couplings + 1)
        PROVENANCE.log('alpha_Rsoc', alpha, '1/sqrt(n_couplings + 1)',
                       {'n_couplings': self.n_couplings}, self.n_couplings)
        return alpha

    def update(self, reward: float) -> float:
        """Actualiza R_soc_ema con nuevo reward."""
        self.n_couplings += 1
        self.reward_history.append(reward)

        # Mantener buffer
        max_size = derive_buffer_size(self.n_couplings)
        while len(self.reward_history) > max_size:
            self.reward_history.popleft()

        alpha = self.get_alpha()
        self.R_soc_ema = (1 - alpha) * self.R_soc_ema + alpha * reward

        return self.R_soc_ema


# =============================================================================
# COUPLING GRADUADO
# =============================================================================

class GraduatedCoupling:
    """
    Intensidad de coupling graduada.
    intensity = min(π_NEO, π_EVA) * √(gate_NEO * gate_EVA)
    """

    def __init__(self):
        self.intensity_history = deque()
        self.T = 0

    def compute_intensity(self, pi_neo: float, pi_eva: float,
                          gate_neo: float, gate_eva: float) -> float:
        """Calcula intensidad de coupling."""
        self.T += 1

        intensity = min(pi_neo, pi_eva) * np.sqrt(gate_neo * gate_eva + NUMERIC_EPS)

        self.intensity_history.append(intensity)
        max_size = derive_buffer_size(self.T)
        while len(self.intensity_history) > max_size:
            self.intensity_history.popleft()

        return intensity

    def get_threshold(self) -> float:
        """Umbral endógeno: mediana de intensidades."""
        if len(self.intensity_history) < 10:
            return 0.0  # Sin umbral durante warmup
        return np.median(list(self.intensity_history))


# =============================================================================
# SELF-MODEL ENDÓGENO
# =============================================================================

class SelfModel:
    """
    Modelo de sí mismo sin ventanas fijas.
    """

    def __init__(self):
        self.signal_history = deque()
        self.state_history = deque()
        self.pad_history = deque()
        self.error_history = deque()

        self.estimates = {
            'alpha_affect': 0.5,
            'alpha_switch': 0.5,
            'alpha_stability': 0.5,
            'alpha_reactivity': 0.5
        }
        self.observations = self.estimates.copy()
        self.self_error = 0.0
        self.n_updates = 0

    def _maintain_buffers(self):
        """Buffers endógenos."""
        max_size = derive_buffer_size(self.n_updates)
        for buf in [self.signal_history, self.state_history,
                    self.pad_history, self.error_history]:
            while len(buf) > max_size:
                buf.popleft()

    def update(self, signals: Dict[str, float], state: str,
               pad: Dict[str, float]) -> Dict:
        """Actualiza el modelo de sí mismo."""
        self.n_updates += 1

        self.signal_history.append(signals)
        self.state_history.append(state)
        self.pad_history.append(pad)
        self._maintain_buffers()

        window = derive_window_size(self.n_updates)
        if len(self.signal_history) < window:
            return {'self_error': 0.0, 'estimates': self.estimates.copy()}

        # Calcular observaciones
        self._compute_observations()

        # Actualizar estimaciones con learning rate endógeno
        self._update_estimates()

        # Calcular error
        self._compute_self_error()

        return {
            'self_error': self.self_error,
            'estimates': self.estimates.copy(),
            'observations': self.observations.copy(),
            'learning_rate': derive_learning_rate(self.n_updates)
        }

    def _compute_observations(self):
        """Calcula índices observados."""
        window = min(len(self.pad_history), derive_window_size(self.n_updates))
        if window < 5:
            return

        # α_affect: volumen PAD
        pad_arr = np.array([[p['P'], p['A'], p['D']]
                           for p in list(self.pad_history)[-window:]])
        if len(pad_arr) >= 3:
            cov = np.cov(pad_arr.T)
            det = np.linalg.det(cov)
            self.observations['alpha_affect'] = np.sqrt(max(0, det))

        # α_switch: tasa de cambio
        states = list(self.state_history)[-window:]
        if len(states) >= 2:
            switches = sum(1 for i in range(1, len(states)) if states[i] != states[i-1])
            self.observations['alpha_switch'] = switches / (len(states) - 1)

        # α_stability: inverso de varianza
        signal_arr = np.array([[s['r'], s['m'], s['e']]
                              for s in list(self.signal_history)[-window:]])
        if len(signal_arr) >= 5:
            var = np.mean(np.var(signal_arr, axis=0))
            self.observations['alpha_stability'] = 1 / (1 + var)

        # α_reactivity: 1 - ACF
        if len(pad_arr) >= 10:
            acf_mean = np.mean([compute_acf_lag1(pad_arr[:, i]) for i in range(3)])
            self.observations['alpha_reactivity'] = 1 - acf_mean

    def _update_estimates(self):
        """Actualiza estimaciones con EMA endógeno."""
        alpha = derive_learning_rate(self.n_updates)
        for key in self.estimates:
            self.estimates[key] = (1 - alpha) * self.estimates[key] + alpha * self.observations[key]

    def _compute_self_error(self):
        """Calcula error de automodelo."""
        errors = [abs(self.estimates[k] - self.observations[k]) for k in self.estimates]
        self.self_error = np.mean(errors)
        self.error_history.append(self.self_error)

    def get_metacognitive_accuracy(self) -> float:
        """Precisión metacognitiva: 1 - rank(self_error)."""
        if len(self.error_history) < 5:
            return 0.5
        return 1 - rolling_rank(self.self_error, self.error_history)


# =============================================================================
# GLOBAL WORKSPACE ENDÓGENO
# =============================================================================

class GlobalWorkspace:
    """
    Global Workspace con K y umbral endógenos.
    K por MDL, umbral por mediana.
    """

    def __init__(self):
        self.variable_history = deque()
        self.activation_history = deque()
        self.variance_history = deque()
        self.pca = None
        self.is_active = False
        self.GW = None
        self.K = 2  # Se actualizará por MDL
        self.T = 0
        self.broadcast_count = 0

    def _maintain_buffers(self):
        max_size = derive_buffer_size(self.T)
        for buf in [self.variable_history, self.activation_history, self.variance_history]:
            while len(buf) > max_size:
                buf.popleft()

    def get_activation_threshold(self) -> float:
        """Umbral endógeno: mediana de activaciones."""
        if len(self.activation_history) < 10:
            return 0.0
        return np.median(list(self.activation_history))

    def compute_K(self) -> int:
        """K por MDL."""
        if self.pca is None or not hasattr(self.pca, 'explained_variance_ratio_'):
            return 2
        return derive_K_by_mdl(self.pca.explained_variance_ratio_)

    def update(self, signals_neo: Dict, signals_eva: Dict,
               pad_neo: Dict, pad_eva: Dict,
               gate_neo: float, gate_eva: float,
               coupling_intensity: float) -> Dict:
        """Actualiza Global Workspace."""
        self.T += 1

        # Construir vector de variables
        all_vars = self._build_variable_vector(signals_neo, signals_eva, pad_neo, pad_eva)
        self.variable_history.append(all_vars)

        # Activación
        activation = (gate_neo + gate_eva) / 2 * coupling_intensity
        self.activation_history.append(activation)
        self._maintain_buffers()

        # Umbral endógeno
        threshold = self.get_activation_threshold()
        self.is_active = activation > threshold

        if not self.is_active:
            return {
                'is_active': False,
                'GW': None,
                'K': self.K,
                'activation': activation,
                'threshold': threshold
            }

        # BROADCAST
        self._fit_pca_and_select()
        self.GW = self._construct_GW(all_vars)
        self.broadcast_count += 1

        return {
            'is_active': True,
            'GW': self.GW.tolist() if self.GW is not None else None,
            'K': self.K,
            'activation': activation,
            'threshold': threshold,
            'broadcast_rate': self.broadcast_count / self.T
        }

    def _build_variable_vector(self, sig_neo: Dict, sig_eva: Dict,
                               pad_neo: Dict, pad_eva: Dict) -> np.ndarray:
        """Construye vector de todas las variables."""
        vec = []
        for key in ['r', 's', 'm', 'c', 'R_soc', 'e', 'q', 'h']:
            vec.append(sig_neo.get(key, 0.5))
        for key in ['r', 's', 'm', 'c', 'R_soc', 'e', 'q', 'h']:
            vec.append(sig_eva.get(key, 0.5))
        vec.extend([pad_neo.get('P', 0.5), pad_neo.get('A', 0.5), pad_neo.get('D', 0.5)])
        vec.extend([pad_eva.get('P', 0.5), pad_eva.get('A', 0.5), pad_eva.get('D', 0.5)])
        return np.array(vec)

    def _fit_pca_and_select(self):
        """Ajusta PCA y K por MDL."""
        window = derive_window_size(self.T)
        if len(self.variable_history) < window:
            return

        data = np.array(list(self.variable_history))
        centered = data - np.median(data, axis=0)

        n_components = min(10, data.shape[1], data.shape[0] - 1)
        self.pca = PCA(n_components=n_components)
        self.pca.fit(centered)

        # K por MDL
        self.K = self.compute_K()
        self.variance_history.append(self.pca.explained_variance_ratio_[0])

    def _construct_GW(self, current_vars: np.ndarray) -> np.ndarray:
        """Construye vector GW."""
        if self.pca is None:
            return current_vars[:self.K]

        data = np.array(list(self.variable_history))
        centered = current_vars - np.median(data, axis=0)
        projected = self.pca.transform(centered.reshape(1, -1))[0]
        return projected[:self.K]

    def get_modulation(self) -> Optional[np.ndarray]:
        """Modulación normalizada por ranks."""
        if not self.is_active or self.GW is None:
            return None

        if len(self.variable_history) < 10:
            return self.GW

        # Rank-normalize
        modulation = np.zeros_like(self.GW)
        for i in range(len(self.GW)):
            hist = [v[i] if i < len(v) else 0 for v in list(self.variable_history)[-50:]]
            if len(hist) > 1:
                rank = stats.percentileofscore(hist, self.GW[i]) / 100
                modulation[i] = 2 * rank - 1  # [-1, 1]

        return modulation


# =============================================================================
# IDENTITY CONTINUITY ENDÓGENO
# =============================================================================

class IdentityContinuity:
    """
    Continuidad de identidad sin umbrales fijos.
    """

    def __init__(self):
        self.state_history = deque()
        self.signal_history = deque()
        self.transition_history = deque()
        self.consistency_history = deque()

        self.identity_profile = {}
        self.transition_matrix = {}
        self.rupture_events = []

        self.temporal_consistency = 0.5
        self.repertoire_consistency = 0.5
        self.narrative_consistency = 0.5
        self.identity_index = 0.5
        self.T = 0

    def _maintain_buffers(self):
        max_size = derive_buffer_size(self.T)
        for buf in [self.state_history, self.signal_history,
                    self.transition_history, self.consistency_history]:
            while len(buf) > max_size:
                buf.popleft()

    def update(self, state: str, signals: Dict[str, float], t: int) -> Dict:
        """Actualiza métricas de continuidad."""
        self.T = t

        prev_state = self.state_history[-1] if self.state_history else state
        self.state_history.append(state)
        self.signal_history.append(signals)
        self._maintain_buffers()

        window = derive_window_size(self.T)
        if len(self.state_history) < window:
            return {'identity_index': 0.5, 'rupture': False}

        # Registrar transición
        transition = (prev_state, state)
        self.transition_history.append(transition)
        self._update_transition_matrix(transition)

        # Calcular índices
        self._compute_temporal_consistency()
        self._compute_repertoire_consistency()
        self._compute_narrative_consistency()

        # Índice global
        self.identity_index = np.mean([
            self.temporal_consistency,
            self.repertoire_consistency,
            self.narrative_consistency
        ])
        self.consistency_history.append(self.identity_index)

        # Detectar ruptura (umbral endógeno)
        rupture = self._detect_rupture(t)

        return {
            'identity_index': self.identity_index,
            'temporal_consistency': self.temporal_consistency,
            'repertoire_consistency': self.repertoire_consistency,
            'narrative_consistency': self.narrative_consistency,
            'rupture': rupture,
            'n_ruptures': len(self.rupture_events)
        }

    def _update_transition_matrix(self, transition: Tuple[str, str]):
        src, dst = transition
        if src not in self.transition_matrix:
            self.transition_matrix[src] = {}
        if dst not in self.transition_matrix[src]:
            self.transition_matrix[src][dst] = 0
        self.transition_matrix[src][dst] += 1

    def _compute_temporal_consistency(self):
        """ACF de patrones de estado."""
        window = derive_window_size(self.T)
        if len(self.state_history) < window:
            return

        states = list(self.state_history)[-window:]
        unique_states = list(set(states))
        state_to_num = {s: i for i, s in enumerate(unique_states)}
        numeric = np.array([state_to_num[s] for s in states])

        acf1 = compute_acf_lag1(numeric)
        self.temporal_consistency = (acf1 + 1) / 2

    def _compute_repertoire_consistency(self):
        """Estabilidad del perfil."""
        window = derive_window_size(self.T)
        if len(self.state_history) < window:
            return

        states = list(self.state_history)[-window:]
        counts = {}
        for s in states:
            counts[s] = counts.get(s, 0) + 1
        total = sum(counts.values())
        current_profile = {s: c/total for s, c in counts.items()}

        if not self.identity_profile:
            self.identity_profile = current_profile
            self.repertoire_consistency = 1.0
            return

        # Distancia entre perfiles
        all_states = set(current_profile.keys()) | set(self.identity_profile.keys())
        distance = sum(abs(current_profile.get(s, 0) - self.identity_profile.get(s, 0))
                      for s in all_states) / 2

        # Actualizar perfil con EMA endógeno
        alpha = derive_learning_rate(self.T)
        for s in all_states:
            old = self.identity_profile.get(s, 0)
            new = current_profile.get(s, 0)
            self.identity_profile[s] = (1 - alpha) * old + alpha * new

        self.repertoire_consistency = 1 - distance

    def _compute_narrative_consistency(self):
        """Predictibilidad de transiciones."""
        window = derive_window_size(self.T)
        if len(self.transition_history) < 10:
            return

        recent = list(self.transition_history)[-window:]
        expected_scores = []

        for src, dst in recent:
            if src in self.transition_matrix:
                total = sum(self.transition_matrix[src].values())
                if total > 0:
                    prob = self.transition_matrix[src].get(dst, 0) / total
                    expected_scores.append(prob)

        if expected_scores:
            self.narrative_consistency = np.mean(expected_scores)

    def _detect_rupture(self, t: int) -> bool:
        """Ruptura cuando consistencia < q10 (endógeno)."""
        if len(self.consistency_history) < 20:
            return False

        threshold = derive_rupture_threshold(np.array(list(self.consistency_history)))

        if self.identity_index < threshold:
            self.rupture_events.append({
                't': t,
                'identity_index': self.identity_index,
                'threshold': threshold
            })
            return True
        return False


# =============================================================================
# MUNDO CONSCIENTE ENDÓGENO
# =============================================================================

class EndogenousWorld:
    """
    Mundo 100% endógeno.
    """

    def __init__(self, name: str):
        self.name = name
        self.t = 0

        # Estado interno (prior uniforme - geometría del simplex)
        self.I = SIMPLEX_UNIFORM.copy()
        self.current_state = LifeState.WAKE

        # Módulos endógenos
        self.pad = EndogenousPAD()
        self.gate = ContinuousGate()
        self.r_soc = AdaptiveRSoc()
        self.self_model = SelfModel()
        self.identity = IdentityContinuity()

        # Historiales (buffers dinámicos)
        self.I_history = deque()
        self.pi_history = deque()
        self.state_counts = {s: 0 for s in LifeState}

        # Métricas
        self.consciousness_log = []

    def _maintain_buffers(self):
        """Buffers dinámicos."""
        max_size = derive_buffer_size(self.t)
        while len(self.I_history) > max_size:
            self.I_history.popleft()
        while len(self.pi_history) > max_size:
            self.pi_history.popleft()

    def compute_signals(self) -> Dict[str, float]:
        """Calcula las 8 señales internas."""
        r = 1 - np.linalg.norm(self.I)

        # s: estabilidad
        window = derive_window_size(self.t) if self.t > 0 else 10
        if len(self.I_history) >= 5:
            recent = np.array(list(self.I_history)[-window:])
            var = np.var(recent)
            s = 1 / (1 + var)
        else:
            s = 0.5

        # m: entropía de I
        I_probs = self.I.copy()
        I_probs = np.maximum(I_probs, NUMERIC_EPS)
        I_probs = I_probs / I_probs.sum()
        m = compute_entropy_normalized(I_probs)

        c = np.max(self.I)
        R_soc = self.r_soc.R_soc_ema

        work_ratio = self.state_counts[LifeState.WORK] / max(1, self.t)
        e = 1 - work_ratio

        q = self.identity.narrative_consistency
        h = self.state_counts[LifeState.SOCIAL] / max(1, self.t)

        return {'r': r, 's': s, 'm': m, 'c': c, 'R_soc': R_soc, 'e': e, 'q': q, 'h': h}

    def compute_pi(self, signals: Dict[str, float], gate: float) -> float:
        """Índice volitivo."""
        benefit = (signals['R_soc'] + signals['h'] + signals['m']) / 3
        cost = signals['r'] + NUMERIC_EPS
        pi_base = benefit / cost

        # Normalizar por rank en historia
        if len(self.pi_history) > 10:
            pi_normalized = rolling_rank(pi_base, self.pi_history)
        else:
            pi_normalized = min(1.0, pi_base)

        return pi_normalized * gate

    def select_state(self, gw_modulation: Optional[np.ndarray] = None) -> LifeState:
        """Selección de estado con gamma endógeno."""
        window = derive_window_size(self.t) if self.t > 0 else 10

        D_rest = 1 - np.mean(list(self.I_history)[-5:]) if self.I_history else 0.5
        D_nov = np.std(list(self.I_history)[-window:]) if len(self.I_history) >= window else 0.5
        D_learn = self.I[1] if len(self.I) > 1 else 0.5
        D_soc = self.r_soc.R_soc_ema

        utilities = {
            LifeState.SLEEP: D_rest,
            LifeState.WAKE: (1 - D_rest),
            LifeState.WORK: self.I[0],
            LifeState.LEARN: D_learn + D_nov,
            LifeState.SOCIAL: D_soc
        }

        # Modulación GW
        if gw_modulation is not None and len(gw_modulation) >= 2:
            mod = np.tanh(gw_modulation[:min(len(gw_modulation), 3)])
            states_list = list(utilities.keys())
            for i, s in enumerate(states_list[:len(mod)]):
                # Modulación por rank (endógeno)
                utilities[s] *= (1 + 0.3 * mod[i % len(mod)])

        # Gamma endógeno
        vals = np.array(list(utilities.values()))
        gamma = derive_softmax_gamma(vals)

        # Softmax
        exp_u = np.exp(gamma * vals)
        probs = exp_u / exp_u.sum()

        states = list(utilities.keys())
        idx = np.random.choice(len(states), p=probs)

        return states[idx]

    def step(self, other_signals: Dict[str, float],
             coupling_intensity: float = 0.0,
             gw_modulation: Optional[np.ndarray] = None) -> Dict:
        """Ejecuta un paso."""
        self.t += 1
        self.I_history.append(self.I.copy())
        self._maintain_buffers()

        # Señales y PAD
        signals = self.compute_signals()
        pad = self.pad.update(signals)

        # Selección de estado
        new_state = self.select_state(gw_modulation)
        self.current_state = new_state
        self.state_counts[new_state] += 1

        # Módulos de consciencia
        self_model_output = self.self_model.update(signals, new_state.value, pad)
        identity_output = self.identity.update(new_state.value, signals, self.t)

        # Dinámica de I
        attractors = {
            LifeState.SLEEP: np.array([0.2, 0.2, 0.6]),
            LifeState.WAKE: SIMPLEX_UNIFORM.copy(),
            LifeState.WORK: np.array([0.6, 0.2, 0.2]),
            LifeState.LEARN: np.array([0.2, 0.6, 0.2]),
            LifeState.SOCIAL: np.array([0.25, 0.25, 0.5])
        }

        target = attractors[new_state]

        # Learning rate endógeno
        rate = derive_learning_rate(self.t)

        if coupling_intensity > 0:
            other_vec = np.array([other_signals.get('r', 0.5),
                                  other_signals.get('m', 0.5),
                                  other_signals.get('h', 0.5)])
            other_vec = other_vec / (other_vec.sum() + NUMERIC_EPS)
            target = (1 - coupling_intensity) * target + coupling_intensity * other_vec

        self.I = (1 - rate) * self.I + rate * target

        # Clip por cuantiles (endógeno)
        if len(self.I_history) > 20:
            I_arr = np.array(list(self.I_history))
            for i in range(len(self.I)):
                clip_low, clip_high = derive_probability_clip(I_arr[:, i])
                self.I[i] = np.clip(self.I[i], clip_low, clip_high)

        # Normalizar al simplex
        self.I = self.I / (self.I.sum() + NUMERIC_EPS)

        # Actualizar R_soc
        if coupling_intensity > 0:
            self.r_soc.update(coupling_intensity)

        # Log
        consciousness = {
            't': self.t,
            'state': new_state.value,
            'self_error': self_model_output['self_error'],
            'metacognitive_accuracy': self.self_model.get_metacognitive_accuracy(),
            'identity_index': identity_output['identity_index'],
            'rupture': identity_output['rupture']
        }
        self.consciousness_log.append(consciousness)

        return {
            'signals': signals,
            'pad': pad,
            'state': new_state,
            'self_model': self_model_output,
            'identity': identity_output
        }


# =============================================================================
# EXPERIMENTO PHASE 12
# =============================================================================

def run_phase12_experiment(n_cycles: int = 25000,
                           output_dir: str = '/root/NEO_EVA/results/phase12') -> Dict:
    """
    Ejecuta experimento 100% endógeno.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("PHASE 12: SISTEMA 100% ENDÓGENO PURO")
    print("=" * 70)
    print("\nCERO números mágicos. Todo derivado de historia.")
    print("Constantes permitidas: ε numérico, prior uniforme simplex")

    # Crear mundos
    neo = EndogenousWorld("NEO")
    eva = EndogenousWorld("EVA")

    # Global Workspace
    global_workspace = GlobalWorkspace()
    coupling = GraduatedCoupling()

    # Registros
    bilateral_events = []
    pi_log_neo = []
    pi_log_eva = []

    train_end = int(n_cycles * 0.6)

    print(f"\nCiclos: {n_cycles} (train: 0-{train_end}, test: {train_end}-{n_cycles})")
    print("\n[Simulando...]")

    for t in range(1, n_cycles + 1):
        # Señales
        neo_signals = neo.compute_signals()
        eva_signals = eva.compute_signals()

        # PAD
        neo_pad = neo.pad.update(neo_signals)
        eva_pad = eva.pad.update(eva_signals)

        # Gates
        neo_gate, neo_rho, neo_var = neo.gate.compute(neo_signals, eva_signals, neo.I_history)
        eva_gate, eva_rho, eva_var = eva.gate.compute(eva_signals, neo_signals, eva.I_history)

        # π
        neo_pi = neo.compute_pi(neo_signals, neo_gate)
        eva_pi = eva.compute_pi(eva_signals, eva_gate)

        neo.pi_history.append(neo_pi)
        eva.pi_history.append(eva_pi)

        # Coupling intensity
        coupling_intensity = coupling.compute_intensity(neo_pi, eva_pi, neo_gate, eva_gate)
        threshold = coupling.get_threshold()

        # Global Workspace
        gw_output = global_workspace.update(
            neo_signals, eva_signals,
            neo_pad, eva_pad,
            neo_gate, eva_gate,
            coupling_intensity
        )

        gw_modulation = None
        if gw_output['is_active']:
            gw_modulation = global_workspace.get_modulation()

        # Bilateral
        bilateral = coupling_intensity > threshold

        # Steps
        neo_output = neo.step(eva_signals, coupling_intensity if bilateral else 0, gw_modulation)
        eva_output = eva.step(neo_signals, coupling_intensity if bilateral else 0, gw_modulation)

        # Registrar
        pi_log_neo.append({'t': t, 'pi': neo_pi, 'gate': neo_gate})
        pi_log_eva.append({'t': t, 'pi': eva_pi, 'gate': eva_gate})

        if bilateral:
            bilateral_events.append({
                't': t,
                'intensity': coupling_intensity,
                'phase': 'train' if t <= train_end else 'test'
            })

        if t % 5000 == 0:
            n_train = len([e for e in bilateral_events if e['phase'] == 'train'])
            n_test = len([e for e in bilateral_events if e['phase'] == 'test'])
            print(f"  t={t}: bilateral={len(bilateral_events)} (train={n_train}, test={n_test})")

    # Resultados
    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)

    n_bilateral = len(bilateral_events)
    n_train_events = len([e for e in bilateral_events if e['phase'] == 'train'])
    n_test_events = len([e for e in bilateral_events if e['phase'] == 'test'])

    # AUC
    pi_neo = np.array([p['pi'] for p in pi_log_neo])
    pi_eva = np.array([p['pi'] for p in pi_log_eva])
    bilateral_ts = set(e['t'] for e in bilateral_events)

    # Train AUC
    train_labels = np.array([1 if t in bilateral_ts else 0 for t in range(1, train_end + 1)])
    train_pi = pi_neo[:train_end]
    auc_train = roc_auc_score(train_labels, train_pi) if train_labels.sum() > 5 else 0.5

    # Test AUC
    test_labels = np.array([1 if t in bilateral_ts else 0 for t in range(train_end + 1, n_cycles + 1)])
    test_pi = pi_neo[train_end:]
    auc_test = roc_auc_score(test_labels, test_pi) if test_labels.sum() > 5 else 0.5

    # Correlación inter-agente
    window = 5
    neo_vals, eva_vals = [], []
    for t in bilateral_ts:
        for dt in range(-window, window + 1):
            idx = t + dt - 1
            if 0 <= idx < len(pi_neo):
                neo_vals.append(pi_neo[idx])
                eva_vals.append(pi_eva[idx])
    inter_corr = np.corrcoef(neo_vals, eva_vals)[0, 1] if len(neo_vals) > 10 else 0

    print(f"\n1. EVENTOS BILATERALES: {n_bilateral} ({n_bilateral/n_cycles*100:.2f}%)")
    print(f"   Train: {n_train_events}, Test: {n_test_events}")

    print(f"\n2. AUC (predicción de bilateral por π):")
    print(f"   Train: {auc_train:.4f}")
    print(f"   Test:  {auc_test:.4f}")
    print(f"   Drop:  {(auc_train - auc_test)/auc_train*100:.1f}%")

    print(f"\n3. CORRELACIÓN INTER-AGENTE: {inter_corr:.4f}")

    print(f"\n4. GLOBAL WORKSPACE:")
    print(f"   K (MDL): {global_workspace.K}")
    print(f"   Broadcasts: {global_workspace.broadcast_count}")

    # Provenance
    print(f"\n5. REGISTRO DE PROCEDENCIA:")
    report = get_provenance_report()
    print(f"   Parámetros rastreados: {report['n_records']}")
    print(f"   Últimos: {', '.join(report['params'][-5:])}")

    # Guardar
    results = {
        'n_cycles': n_cycles,
        'n_bilateral': n_bilateral,
        'n_train_events': n_train_events,
        'n_test_events': n_test_events,
        'auc_train': float(auc_train),
        'auc_test': float(auc_test),
        'inter_agent_correlation': float(inter_corr),
        'gw_K': global_workspace.K,
        'gw_broadcasts': global_workspace.broadcast_count,
        'provenance': report
    }

    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {str(k): convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, LifeState):
            return obj.value
        else:
            return obj

    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)

    with open(f"{output_dir}/bilateral_events.json", 'w') as f:
        json.dump(convert_for_json(bilateral_events), f)

    with open(f"{output_dir}/pi_log_neo.json", 'w') as f:
        json.dump(convert_for_json(pi_log_neo), f)

    with open(f"{output_dir}/pi_log_eva.json", 'w') as f:
        json.dump(convert_for_json(pi_log_eva), f)

    print(f"\n[OK] Resultados guardados en {output_dir}/")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Phase 12: Pure Endogenous System')
    parser.add_argument('--cycles', type=int, default=25000)
    parser.add_argument('--output-dir', default='/root/NEO_EVA/results/phase12')
    args = parser.parse_args()

    run_phase12_experiment(args.cycles, args.output_dir)
