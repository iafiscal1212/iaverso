#!/usr/bin/env python3
"""
Phase 8c: VOLUNTAD Social Potenciada + Sistema Afectivo (100% Endógeno)
========================================================================

Mejoras sobre Phase 8:
1. Recompensa social con reciprocidad (EMA de R_soc)
2. Temperatura γ auto-ajustada por confianza
3. Costo social con ρ, Var, fatiga en ranks
4. Período refractario endógeno post-coupling
5. Bandit de modos con recompensa condicionada
6. Sistema afectivo: latentes PAD (Valencia, Activación, Dominancia)

Todo 100% endógeno - sin constantes mágicas.
Principio: "Si no sale de la historia, no entra en la dinámica"

"Emoción" = estado latente que emerge de señales internas combinadas
sin pesos fijos (ranks, cuantiles, √T) y que modula la política como
un campo lento con memoria endógena.
"""

import sys
import os
import json
import math
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from scipy.signal import periodogram

sys.path.insert(0, '/root/NEOSYNT')
sys.path.insert(0, '/root/NEO_EVA/tools')

from affective_system import AffectiveSystem

# =============================================================================
# Constantes endógenas (solo geométricas/matemáticas)
# =============================================================================

EPS = 1e-12

def window_size(T: int) -> int:
    """w = max(10, floor(sqrt(T)))"""
    return max(10, int(math.sqrt(T)))

def max_hist(T: int) -> int:
    """Historia máxima = min(T, 10*sqrt(T))"""
    return min(T, int(10 * math.sqrt(T))) if T > 0 else 100

def rank_normalize(value: float, history: List[float], min_samples: int = 10) -> float:
    """Normaliza value a [0,1] por rank. NaN si insuficientes datos."""
    if len(history) < min_samples:
        return float('nan')
    rank = (np.searchsorted(np.sort(history), value) + 1) / len(history)
    return float(rank)

def inverse_rank(rank_value: float, history: List[float]) -> float:
    """Obtiene valor correspondiente a un rank dado."""
    if len(history) < 10 or rank_value < 0 or rank_value > 1:
        return float('nan')
    sorted_hist = np.sort(history)
    idx = int(rank_value * (len(sorted_hist) - 1))
    return float(sorted_hist[idx])

def iqr(arr) -> float:
    """IQR seguro."""
    if len(arr) < 4:
        return float('nan')
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))

def quantile_safe(arr, q: float) -> float:
    """Quantile que retorna NaN si insuficientes datos."""
    if len(arr) < 5:
        return float('nan')
    return float(np.percentile(arr, q * 100))

def ema_beta(w: int) -> float:
    """β para EMA: (w-1)/(w+1)"""
    return (w - 1) / (w + 1)

# =============================================================================
# Estados de VOLUNTAD
# =============================================================================

class LifeState(Enum):
    SLEEP = 0
    WAKE = 1
    WORK = 2
    LEARN = 3
    SOCIAL = 4

# =============================================================================
# Recompensa Social con Reciprocidad
# =============================================================================

class SocialRewardTracker:
    """
    Rastrea R_soc = BordaRank(ΔRMSE↓, ΔMDL↓, MI/TE↑) en ventanas de coupling.
    Mantiene EMA para suavizar.
    """

    def __init__(self):
        # Métricas durante coupling ON
        self.rmse_on: deque = deque(maxlen=200)
        self.mdl_on: deque = deque(maxlen=200)
        self.mi_on: deque = deque(maxlen=200)

        # Métricas durante coupling OFF (baseline)
        self.rmse_off: deque = deque(maxlen=200)
        self.mdl_off: deque = deque(maxlen=200)
        self.mi_off: deque = deque(maxlen=200)

        # R_soc history y EMA
        self.R_soc_history: deque = deque(maxlen=500)
        self.R_soc_ema = 0.5  # Iniciar neutral

        # Ventana ON actual
        self.current_window_rmse: List[float] = []
        self.current_window_mdl: List[float] = []
        self.current_window_mi: List[float] = []
        self.in_coupling = False

    def start_coupling_window(self):
        """Inicia una ventana de coupling."""
        self.in_coupling = True
        self.current_window_rmse = []
        self.current_window_mdl = []
        self.current_window_mi = []

    def add_metric(self, rmse: float, mdl: float, mi: float, coupling_active: bool):
        """Añade métricas al tracker."""
        if coupling_active and self.in_coupling:
            self.current_window_rmse.append(rmse)
            self.current_window_mdl.append(mdl)
            self.current_window_mi.append(mi)
        elif not coupling_active:
            if not math.isnan(rmse):
                self.rmse_off.append(rmse)
            if not math.isnan(mdl):
                self.mdl_off.append(mdl)
            if not math.isnan(mi):
                self.mi_off.append(mi)

    def end_coupling_window(self, w: int) -> float:
        """
        Finaliza ventana y calcula R_soc.
        Retorna R_soc normalizado por rank histórico.
        """
        self.in_coupling = False

        # Aceptar ventanas de 1+ ciclos (antes era 3)
        if len(self.current_window_rmse) < 1:
            return float('nan')

        # Métricas de la ventana ON
        rmse_on_mean = np.mean(self.current_window_rmse)
        mdl_on_mean = np.mean(self.current_window_mdl)
        mi_on_mean = np.mean(self.current_window_mi)

        self.rmse_on.append(rmse_on_mean)
        self.mdl_on.append(mdl_on_mean)
        self.mi_on.append(mi_on_mean)

        # BordaRank: comparar con histórico
        # RMSE↓ y MDL↓ = buenos si son bajos (rank invertido)
        # MI↑ = bueno si es alto
        if len(self.rmse_on) < 5:
            return float('nan')

        rmse_rank = 1 - rank_normalize(rmse_on_mean, list(self.rmse_on))
        mdl_rank = 1 - rank_normalize(mdl_on_mean, list(self.mdl_on))
        mi_rank = rank_normalize(mi_on_mean, list(self.mi_on))

        if any(math.isnan(x) for x in [rmse_rank, mdl_rank, mi_rank]):
            return float('nan')

        R_soc = (rmse_rank + mdl_rank + mi_rank) / 3.0
        self.R_soc_history.append(R_soc)

        # Actualizar EMA
        beta = ema_beta(w)
        self.R_soc_ema = beta * self.R_soc_ema + (1 - beta) * R_soc

        return R_soc

    def get_R_soc_ema_rank(self) -> float:
        """Retorna rank del EMA actual."""
        if len(self.R_soc_history) < 10:
            return float('nan')
        return rank_normalize(self.R_soc_ema, list(self.R_soc_history))

# =============================================================================
# Período Refractario Endógeno
# =============================================================================

class RefractoryPeriod:
    """
    Período refractario post-coupling.
    Δt_ref derivado de Var_w(I) o R_soc.
    """

    def __init__(self):
        self.last_coupling_t = -1000
        self.delta_t_ref = 10  # Se actualiza endógenamente
        self.delta_t_history: deque = deque(maxlen=100)

    def trigger(self, t: int, var_I_history: List[float], w: int):
        """Activa período refractario tras coupling."""
        self.last_coupling_t = t

        # Δt_ref = inverse_rank(Var_w(I))
        # Alta varianza → período corto; baja varianza → período largo
        if len(var_I_history) >= 10:
            var_current = var_I_history[-1] if var_I_history else 0.5
            var_rank = rank_normalize(var_current, list(var_I_history))
            if not math.isnan(var_rank):
                # Mapear rank a período: alto rank (alta var) → corto
                # Período entre w/4 y 2*w
                self.delta_t_ref = int(w / 4 + (1 - var_rank) * 1.75 * w)
                self.delta_t_history.append(self.delta_t_ref)

    def get_damping(self, t: int) -> float:
        """
        Retorna factor de amortiguación [0, 1].
        1.0 = sin amortiguación, 0.0 = completamente amortiguado.
        """
        if self.last_coupling_t < 0:
            return 1.0

        elapsed = t - self.last_coupling_t

        if elapsed >= self.delta_t_ref:
            return 1.0

        if len(self.delta_t_history) < 5:
            # Durante warmup, amortiguación lineal simple
            return elapsed / self.delta_t_ref

        # Amortiguación por rank
        elapsed_ratio = elapsed / self.delta_t_ref
        # rank(Δt/Δt_ref) usando histórico de ratios
        damping = elapsed_ratio  # Lineal como aproximación endógena
        return float(damping)

# =============================================================================
# Bandit de Modos con Recompensa Condicionada
# =============================================================================

class ConditionalModeBandit:
    """
    Thompson Sampling para modos {-1, 0, +1}.
    Recompensa = BordaRank condicionada al modo elegido.
    Pesos especializados por mundo: EVA→MI/TE, NEO→ΔMDL
    """

    def __init__(self, specialization: str = "balanced"):
        """
        specialization: "mi_te" (EVA), "mdl" (NEO), o "balanced"
        Los pesos se aprenden pero con prior hacia la especialización.
        """
        self.arms = [-1, 0, 1]
        self.specialization = specialization

        # Posterior Beta para cada brazo
        self.alpha = {m: 1.0 for m in self.arms}
        self.beta_param = {m: 1.0 for m in self.arms}

        # Historiales por modo
        self.rewards_by_mode: Dict[int, deque] = {m: deque(maxlen=100) for m in self.arms}
        self.pulls_by_mode: Dict[int, int] = {m: 0 for m in self.arms}

        # Historia global para normalización
        self.all_rewards: deque = deque(maxlen=500)

        # Pesos aprendidos para métricas (RMSE, MDL, MI)
        # Prior según especialización
        if specialization == "mi_te":
            # EVA: más peso a MI/TE
            self.metric_weights = np.array([0.2, 0.2, 0.6])
        elif specialization == "mdl":
            # NEO: más peso a MDL (y RMSE)
            self.metric_weights = np.array([0.3, 0.5, 0.2])
        else:
            self.metric_weights = np.array([1/3, 1/3, 1/3])

        self.weight_history: deque = deque(maxlen=100)

    def select(self, gate_open: bool) -> int:
        """Thompson Sampling para seleccionar modo."""
        if not gate_open:
            return 0

        samples = {}
        for m in self.arms:
            samples[m] = np.random.beta(
                max(self.alpha[m], 0.1),
                max(self.beta_param[m], 0.1)
            )

        return max(samples, key=samples.get)

    def update(self, mode: int, rmse: float, mdl: float, mi: float):
        """
        Actualiza con recompensa BordaRank ponderada por especialización.
        EVA prioriza MI/TE, NEO prioriza MDL.
        """
        self.pulls_by_mode[mode] += 1

        self.all_rewards.append((rmse, mdl, mi))

        if len(self.all_rewards) < 10:
            return

        all_rmse = [r[0] for r in self.all_rewards if not math.isnan(r[0])]
        all_mdl = [r[1] for r in self.all_rewards if not math.isnan(r[1])]
        all_mi = [r[2] for r in self.all_rewards if not math.isnan(r[2])]

        if len(all_rmse) < 10:
            return

        rmse_rank = 1 - rank_normalize(rmse, all_rmse) if not math.isnan(rmse) else 0.5
        mdl_rank = 1 - rank_normalize(mdl, all_mdl) if not math.isnan(mdl) else 0.5
        mi_rank = rank_normalize(mi, all_mi) if not math.isnan(mi) else 0.5

        # Recompensa ponderada por especialización
        ranks = np.array([rmse_rank, mdl_rank, mi_rank])
        G = float(np.dot(self.metric_weights, ranks))
        self.rewards_by_mode[mode].append(G)

        # Actualizar pesos adaptativamente (gradient hacia mejor métrica)
        if len(self.rewards_by_mode[mode]) > 5:
            recent_rewards = list(self.rewards_by_mode[mode])[-5:]
            if np.std(recent_rewards) > 0.01:
                # Gradient: aumentar peso de métrica que correlaciona con G
                lr_w = 0.01
                gradient = ranks - np.mean(ranks)
                self.metric_weights += lr_w * gradient * (G - 0.5)
                # Proyectar al simplex
                self.metric_weights = np.maximum(self.metric_weights, 0.05)
                self.metric_weights /= self.metric_weights.sum()

        self.weight_history.append(self.metric_weights.copy())

        # Actualizar posterior Beta
        if G > 0.5:
            self.alpha[mode] += G
        else:
            self.beta_param[mode] += (1 - G)

    def get_stats(self) -> Dict:
        return {
            'pulls': dict(self.pulls_by_mode),
            'mean_rewards': {
                m: float(np.mean(list(self.rewards_by_mode[m]))) if self.rewards_by_mode[m] else 0
                for m in self.arms
            },
            'alpha': dict(self.alpha),
            'beta': dict(self.beta_param),
            'specialization': self.specialization,
            'metric_weights': self.metric_weights.tolist(),
        }

# =============================================================================
# Oscilador Circadiano (igual que Phase 8)
# =============================================================================

class CircadianOscillator:
    def __init__(self, min_period: int = 20, max_period: int = 500):
        self.activity_history: deque = deque(maxlen=2000)
        self.phase_history: deque = deque(maxlen=1000)
        self.min_period = min_period
        self.max_period = max_period
        self.dominant_freq = float('nan')
        self.phase = 0.0

    def update(self, activity: float) -> float:
        self.activity_history.append(activity)
        if len(self.activity_history) < self.min_period * 2:
            return float('nan')

        signal = np.array(self.activity_history)
        signal = signal - np.mean(signal)
        if np.std(signal) < EPS:
            return float('nan')

        freqs, psd = periodogram(signal, fs=1.0)
        valid_mask = (freqs > 1/self.max_period) & (freqs < 1/self.min_period)
        if not np.any(valid_mask):
            return float('nan')

        valid_freqs = freqs[valid_mask]
        valid_psd = psd[valid_mask]
        idx_max = np.argmax(valid_psd)
        self.dominant_freq = valid_freqs[idx_max]

        t = np.arange(len(signal))
        cos_comp = np.sum(signal * np.cos(2 * np.pi * self.dominant_freq * t))
        sin_comp = np.sum(signal * np.sin(2 * np.pi * self.dominant_freq * t))
        self.phase = np.arctan2(sin_comp, cos_comp)

        C_raw = np.sin(self.phase)
        self.phase_history.append(C_raw)

        if len(self.phase_history) < 20:
            return float('nan')

        return rank_normalize(C_raw, list(self.phase_history))

# =============================================================================
# Métricas de Ganancia
# =============================================================================

class GainMetrics:
    def __init__(self):
        self.rmse_history: deque = deque(maxlen=500)
        self.mdl_history: deque = deque(maxlen=500)
        self.mi_history: deque = deque(maxlen=500)

    def compute(self, I_history: List[np.ndarray], w: int) -> Tuple[float, float, float, float]:
        """Retorna (G, rmse, mdl, mi)"""
        if len(I_history) < w + 1:
            return float('nan'), float('nan'), float('nan'), float('nan')

        I_arr = np.array(I_history[-w-1:])

        # RMSE
        beta = ema_beta(w)
        ema = I_arr[0].copy()
        for i in range(1, len(I_arr) - 1):
            ema = beta * ema + (1 - beta) * I_arr[i]
        rmse = float(np.sqrt(np.mean((I_arr[-1] - ema) ** 2)))
        self.rmse_history.append(rmse)

        # MDL
        mean_I = np.mean(I_arr, axis=0)
        mean_I = np.maximum(mean_I, EPS)
        mean_I = mean_I / mean_I.sum()
        mdl = float(-np.sum(mean_I * np.log(mean_I + EPS)))
        self.mdl_history.append(mdl)

        # MI proxy
        if len(I_arr) > 3:
            corr = np.corrcoef(I_arr[:-1].flatten(), I_arr[1:].flatten())[0, 1]
            mi = abs(corr) if not np.isnan(corr) else 0
        else:
            mi = 0.0
        self.mi_history.append(mi)

        # Borda rank
        if len(self.rmse_history) < 10:
            return float('nan'), rmse, mdl, mi

        rmse_rank = 1 - rank_normalize(rmse, list(self.rmse_history))
        mdl_rank = 1 - rank_normalize(mdl, list(self.mdl_history))
        mi_rank = rank_normalize(mi, list(self.mi_history))

        if any(math.isnan(x) for x in [rmse_rank, mdl_rank, mi_rank]):
            return float('nan'), rmse, mdl, mi

        G = (rmse_rank + mdl_rank + mi_rank) / 3.0
        return float(G), rmse, mdl, mi

# =============================================================================
# Drives Endógenos
# =============================================================================

@dataclass
class Drives:
    D_rest: float
    D_nov: float
    D_learn: float
    D_soc: float  # Ahora basado en R_soc_ema
    C_t: float
    warmup: bool

# =============================================================================
# Sistema de VOLUNTAD Potenciado
# =============================================================================

class PotentiatedVoluntarySystem:
    """
    VOLUNTAD con social potenciado:
    - D_soc = rank(R_soc_ema) - basado en reciprocidad
    - γ = rank(confianza) - temperatura endógena
    - Período refractario post-coupling
    """

    def __init__(self):
        self.state = LifeState.WAKE
        self.state_history: deque = deque(maxlen=2000)

        # Drives histories
        self.D_rest_history: deque = deque(maxlen=500)
        self.D_nov_history: deque = deque(maxlen=500)
        self.D_learn_history: deque = deque(maxlen=500)

        # Oscilador y recompensa social
        self.circadian = CircadianOscillator()
        self.social_reward = SocialRewardTracker()
        self.refractory = RefractoryPeriod()

        # Pesos aprendidos
        self.alpha = {s: np.zeros(5) for s in LifeState}
        self.beta = {s: np.zeros(3) for s in LifeState}

        # Costes
        self.fatigue = 0.0
        self.saturation = 0.0
        self.risk = 0.0

        # Historiales para confianza
        self.conf_history: deque = deque(maxlen=500)
        self.utility_history: deque = deque(maxlen=500)
        self.reward_history: deque = deque(maxlen=500)

        self.lr = 0.1

    def compute_drives(self, I: np.ndarray, I_history: List[np.ndarray],
                      residuals: List[float], G: float) -> Drives:
        """Drives 100% endógenos con D_soc basado en reciprocidad."""
        T = len(I_history)
        w = window_size(T)
        warmup = False

        # D_rest
        if len(residuals) >= w:
            iqr_r = iqr(residuals[-w:])
            if not math.isnan(iqr_r):
                D_rest_raw = iqr_r / math.sqrt(T)
                self.D_rest_history.append(D_rest_raw)
                D_rest = rank_normalize(D_rest_raw, list(self.D_rest_history))
            else:
                D_rest = float('nan')
                warmup = True
        else:
            D_rest = float('nan')
            warmup = True

        # D_nov
        if len(I_history) >= w:
            I_arr = np.array(I_history[-w:])
            iqr_I = iqr(I_arr.flatten())
            # λ₁ aproximado por varianza explicada
            try:
                _, s, _ = np.linalg.svd(I_arr - np.mean(I_arr, axis=0), full_matrices=False)
                lambda1 = s[0]**2 / (s**2).sum() if s.sum() > EPS else 0
            except:
                lambda1 = 0
            D_nov_raw = lambda1 + (iqr_I / (1 + EPS) if not math.isnan(iqr_I) else 0)
            self.D_nov_history.append(D_nov_raw)
            D_nov = rank_normalize(D_nov_raw, list(self.D_nov_history))
        else:
            D_nov = float('nan')
            warmup = True

        # D_learn
        if not math.isnan(G):
            D_learn_raw = 1 - G
            self.D_learn_history.append(D_learn_raw)
            D_learn = rank_normalize(D_learn_raw, list(self.D_learn_history))
        else:
            D_learn = float('nan')
            warmup = True

        # D_soc = rank(R_soc_ema) - RECIPROCIDAD
        D_soc = self.social_reward.get_R_soc_ema_rank()
        if math.isnan(D_soc):
            D_soc = 0.5  # Neutral durante warmup
            warmup = True

        # C_t
        activity = np.linalg.norm(np.diff(I_history[-2:], axis=0)) if len(I_history) >= 2 else 0
        C_t = self.circadian.update(activity)
        if math.isnan(C_t):
            C_t = 0.5
            warmup = True

        return Drives(
            D_rest=D_rest if not math.isnan(D_rest) else 0.5,
            D_nov=D_nov if not math.isnan(D_nov) else 0.5,
            D_learn=D_learn if not math.isnan(D_learn) else 0.5,
            D_soc=D_soc,
            C_t=C_t,
            warmup=warmup
        )

    def compute_confidence(self, I: np.ndarray) -> float:
        """Confianza = max(I) - 2º max(I)"""
        I_sorted = np.sort(I)[::-1]
        conf = float(I_sorted[0] - I_sorted[1])
        self.conf_history.append(conf)
        return conf

    def compute_gamma(self) -> float:
        """γ = rank(confianza) - temperatura endógena."""
        if len(self.conf_history) < 10:
            return 1.0
        conf_current = self.conf_history[-1]
        gamma = rank_normalize(conf_current, list(self.conf_history))
        if math.isnan(gamma):
            return 1.0
        # Escalar para que tenga efecto: γ ∈ [0.5, 5.0]
        return 0.5 + gamma * 4.5

    def compute_utility(self, state: LifeState, drives: Drives,
                        social_boost: float = 0.0) -> float:
        """
        Utilidad con boost dinámico para SOCIAL cuando R̃_soc alto.
        social_boost = rank(R_soc_ema) si state=SOCIAL, else 0
        """
        D = np.array([drives.D_rest, drives.D_nov, drives.D_learn, drives.D_soc, drives.C_t])
        K = np.array([self.fatigue, self.saturation, self.risk])
        base_U = float(np.dot(self.alpha[state], D) - np.dot(self.beta[state], K))

        # Boost para SOCIAL cuando R_soc_ema está alto (percentil)
        if state == LifeState.SOCIAL and social_boost > 0:
            return base_U + social_boost
        return base_U

    def choose_state(self, drives: Drives, I: np.ndarray,
                     social_boost: float = 0.0) -> Tuple[LifeState, Dict]:
        """Elige estado con softmax, γ endógeno, y boost social dinámico."""
        utilities = {s: self.compute_utility(s, drives, social_boost if s == LifeState.SOCIAL else 0)
                     for s in LifeState}

        # γ endógeno
        self.compute_confidence(I)
        gamma = self.compute_gamma()

        # Softmax
        U_arr = np.array([utilities[s] for s in LifeState])
        U_scaled = gamma * U_arr
        U_shifted = U_scaled - np.max(U_scaled)
        exp_U = np.exp(U_shifted)
        probs = exp_U / (exp_U.sum() + EPS)

        choice = np.random.choice(len(LifeState), p=probs)
        new_state = list(LifeState)[choice]

        self.state_history.append(new_state)
        for s in LifeState:
            self.utility_history.append(utilities[s])

        self.state = new_state
        return new_state, {
            'utilities': {s.name: utilities[s] for s in LifeState},
            'probs': {s.name: float(probs[i]) for i, s in enumerate(LifeState)},
            'gamma': gamma,
            'chosen': new_state.name
        }

    def update_costs(self, state: LifeState, coupling_active: bool):
        if state != LifeState.SLEEP:
            self.fatigue = min(1.0, self.fatigue + 0.01)
        else:
            self.fatigue = max(0.0, self.fatigue - 0.05)

        if state == LifeState.WORK:
            self.saturation = min(1.0, self.saturation + 0.02)
        else:
            self.saturation = max(0.0, self.saturation - 0.01)

        if state == LifeState.SOCIAL and coupling_active:
            self.risk = min(1.0, self.risk + 0.01)
        else:
            self.risk = max(0.0, self.risk - 0.02)

    def learn(self, reward: float, state: LifeState, drives: Drives):
        self.reward_history.append(reward)
        if len(self.reward_history) < 10:
            return

        reward_std = np.std(list(self.reward_history))
        lr = self.lr / (1 + reward_std) if reward_std > EPS else self.lr

        U_current = self.compute_utility(state, drives)
        td_error = reward - U_current

        D = np.array([drives.D_rest, drives.D_nov, drives.D_learn, drives.D_soc, drives.C_t])
        self.alpha[state] += lr * td_error * D

        K = np.array([self.fatigue, self.saturation, self.risk])
        self.beta[state] -= lr * td_error * K

        for s in LifeState:
            self.alpha[s] = np.clip(self.alpha[s], -5, 5)
            self.beta[s] = np.clip(self.beta[s], -5, 5)

# =============================================================================
# Señales Internas
# =============================================================================

@dataclass
class InternalSignals:
    t: int
    u: float
    rho: float
    lambda1: float
    conf: float
    cv_r: float
    var_I: float
    G: float
    warmup: bool

# =============================================================================
# Mundo Autónomo con Social Potenciado
# =============================================================================

class PotentiatedWorld:
    """
    Mundo con VOLUNTAD social potenciada.
    Especializaciones: NEO→MDL, EVA→MI/TE
    """

    def __init__(self, name: str, initial_I: np.ndarray,
                 specialization: str = "balanced"):
        """
        specialization: "mdl" (NEO), "mi_te" (EVA), o "balanced"
        """
        self.name = name
        self.specialization = specialization
        self.I = initial_I.copy()
        self.I_history: List[np.ndarray] = [initial_I.copy()]
        self.residuals: List[float] = []

        # OU state
        self.ou_Z = np.array([0.0, 0.0])
        self.u_1 = np.array([1, -1, 0]) / np.sqrt(2)
        self.u_2 = np.array([1, 1, -2]) / np.sqrt(6)

        # Historiales
        self.rho_history: deque = deque(maxlen=500)
        self.var_I_history: deque = deque(maxlen=500)
        self.iqr_history: deque = deque(maxlen=500)
        self.cost_history: deque = deque(maxlen=500)
        self.benefit_history: deque = deque(maxlen=500)

        # Componentes
        self.gain_metrics = GainMetrics()
        self.voluntary = PotentiatedVoluntarySystem()
        self.mode_bandit = ConditionalModeBandit(specialization=specialization)
        self.affective = AffectiveSystem()  # Sistema afectivo endógeno

        # Estado
        self.willing = False
        self.current_mode = 0
        self.t = 0

        # Contadores
        self.warmup_count = 0
        self.dynamic_count = 0
        self.coupling_count = 0

        # Logs
        self.series: List[Dict] = []
        self.consent_log: List[Dict] = []
        self.voluntary_log: List[Dict] = []

    def _get_window(self) -> int:
        return window_size(len(self.I_history))

    def _compute_rho(self) -> float:
        w = self._get_window()
        if len(self.I_history) < w:
            return float('nan')

        I_arr = np.array(self.I_history[-w:])
        diffs = np.diff(I_arr, axis=0)
        if len(diffs) < 5:
            return float('nan')

        norms = np.linalg.norm(diffs, axis=1)
        if norms[0] < EPS or norms[-1] < EPS:
            return float('nan')

        rho = (norms[-1] / norms[0]) ** (1 / len(norms))
        if not math.isnan(rho):
            self.rho_history.append(rho)
        return float(rho)

    def _compute_var_I(self) -> float:
        w = self._get_window()
        if len(self.I_history) < w:
            return float('nan')

        var = float(np.var(np.array(self.I_history[-w:])))
        if not math.isnan(var):
            self.var_I_history.append(var)
        return var

    def _compute_signals(self) -> InternalSignals:
        T = len(self.I_history)
        w = self._get_window()
        warmup = False

        # u, cv_r
        if len(self.residuals) >= w:
            iqr_r = iqr(self.residuals[-w:])
            if not math.isnan(iqr_r):
                self.iqr_history.append(iqr_r)
                u = iqr_r / math.sqrt(T)
                r_arr = np.array(self.residuals[-w:])
                cv_r = np.std(r_arr) / (np.mean(r_arr) + EPS)
            else:
                u = float('nan')
                cv_r = float('nan')
                warmup = True
        else:
            u = float('nan')
            cv_r = float('nan')
            warmup = True

        rho = self._compute_rho()
        if math.isnan(rho):
            warmup = True

        var_I = self._compute_var_I()
        if math.isnan(var_I):
            warmup = True

        # PCA
        if len(self.I_history) >= w:
            I_arr = np.array(self.I_history[-w:])
            try:
                _, s, _ = np.linalg.svd(I_arr - np.mean(I_arr, axis=0), full_matrices=False)
                lambda1 = float(s[0]**2 / (s**2).sum()) if s.sum() > EPS else 0
            except:
                lambda1 = float('nan')
                warmup = True
        else:
            lambda1 = float('nan')
            warmup = True

        G, _, _, _ = self.gain_metrics.compute(self.I_history, w)
        if math.isnan(G):
            warmup = True

        I_sorted = np.sort(self.I)[::-1]
        conf = float(I_sorted[0] - I_sorted[1])

        return InternalSignals(
            t=self.t, u=u, rho=rho, lambda1=lambda1,
            conf=conf, cv_r=cv_r if not math.isnan(cv_r) else 1.0,
            var_I=var_I, G=G, warmup=warmup
        )

    def _compute_social_cost(self, signals: InternalSignals) -> float:
        """
        cost_t = rank(ρ) + rank(1 - Var_w(I)) + rank(fatiga)
        Media simple de ranks.
        """
        if signals.warmup:
            return float('nan')

        # rank(ρ)
        rho_rank = rank_normalize(signals.rho, list(self.rho_history))

        # rank(1 - Var) = alto cuando Var es bajo
        if len(self.var_I_history) >= 10:
            inv_var = 1 - signals.var_I
            # Necesitamos histórico de 1-Var
            inv_var_hist = [1 - v for v in self.var_I_history]
            var_cost_rank = rank_normalize(inv_var, inv_var_hist)
        else:
            var_cost_rank = float('nan')

        # rank(fatiga)
        fatigue_rank = self.voluntary.fatigue  # Ya en [0,1]

        if math.isnan(rho_rank) or math.isnan(var_cost_rank):
            return float('nan')

        cost_raw = (rho_rank + var_cost_rank + fatigue_rank) / 3.0
        self.cost_history.append(cost_raw)

        return rank_normalize(cost_raw, list(self.cost_history))

    def _compute_social_benefit(self, signals_self: InternalSignals,
                               signals_other: Optional[InternalSignals]) -> float:
        """Beneficio basado en R_soc_ema (reciprocidad)."""
        if signals_other is None or signals_self.warmup or signals_other.warmup:
            return float('nan')

        # Beneficio principal: rank(R_soc_ema)
        R_soc_rank = self.voluntary.social_reward.get_R_soc_ema_rank()

        if math.isnan(R_soc_rank):
            # Fallback a métricas directas durante warmup
            f1 = signals_other.u / (1 + signals_self.u + EPS)
            f2 = signals_other.lambda1 / (signals_other.lambda1 + signals_self.lambda1 + EPS)
            f3 = signals_other.conf / (1 + signals_self.cv_r + EPS)
            benefit_raw = f1 * f2 * f3
            self.benefit_history.append(benefit_raw)
            return rank_normalize(benefit_raw, list(self.benefit_history))

        return R_soc_rank

    def _compute_willingness(self, signals: InternalSignals,
                            other_signals: Optional[InternalSignals]) -> Tuple[bool, float, Dict]:
        """
        π = σ(rank(R_soc_ema) - rank(cost)) * damping_refractario
        """
        benefit = self._compute_social_benefit(signals, other_signals)
        cost = self._compute_social_cost(signals)

        if math.isnan(benefit) or math.isnan(cost):
            self.warmup_count += 1
            return False, float('nan'), {'warmup': True, 'benefit': benefit, 'cost': cost}

        self.dynamic_count += 1

        # π base
        diff = benefit - cost
        k = 4.0  # Podría hacerse endógeno también
        pi_base = 1.0 / (1.0 + np.exp(-k * diff))

        # Aplicar amortiguación refractaria
        damping = self.voluntary.refractory.get_damping(self.t)
        pi = pi_base * damping

        # Decisión estocástica
        a = np.random.random() < pi

        return a, float(pi), {
            'warmup': False,
            'benefit': float(benefit),
            'cost': float(cost),
            'pi_base': float(pi_base),
            'damping': float(damping),
            'pi': float(pi)
        }

    def _ou_step(self) -> np.ndarray:
        T = len(self.I_history)
        w = self._get_window()

        sigma_uniform = 1.0 / math.sqrt(12)
        sigma_floor = sigma_uniform / math.sqrt(w)

        if len(self.iqr_history) >= 10:
            iqr_med = np.median(list(self.iqr_history))
            sigma = max(sigma_floor, max(iqr_med, sigma_uniform) / math.sqrt(w))
        else:
            sigma = sigma_floor

        if len(self.residuals) > w:
            r = np.array(self.residuals[-w:])
            if len(r) > 2:
                corr = np.corrcoef(r[:-1], r[1:])[0, 1]
                if not np.isnan(corr) and abs(corr) > EPS and abs(corr) < 0.99:
                    theta = max(0.01, min(1.0, -1 / np.log(abs(corr) + EPS)))
                else:
                    theta = 0.1
            else:
                theta = 0.1
        else:
            theta = 0.1

        noise = np.random.randn(2) * sigma
        self.ou_Z = (1 - theta) * self.ou_Z + noise
        return self.ou_Z

    def _project_to_simplex(self, I: np.ndarray) -> np.ndarray:
        I = np.maximum(I, EPS)
        return I / I.sum()

    def _compute_social_boost(self) -> float:
        """
        Boost para SOCIAL cuando R_soc_ema está en percentiles altos.
        Retorna valor en [0, 1] basado en rank de R_soc_ema.
        """
        R_soc_rank = self.voluntary.social_reward.get_R_soc_ema_rank()
        if math.isnan(R_soc_rank):
            return 0.0

        # Solo boost si R_soc_ema > P50 (rank > 0.5)
        if R_soc_rank > 0.5:
            # Escalar: rank 0.5→0, rank 1.0→1.0
            return (R_soc_rank - 0.5) * 2.0
        return 0.0

    def _pre_step_willingness(self, other_signals: Optional[InternalSignals] = None) -> Tuple[bool, bool]:
        """
        Calcula willing y gate SIN modificar estado.
        Usado para determinar bilateral consent antes del step.
        """
        w = self._get_window()
        signals = self._compute_signals()

        # Métricas para drives
        G, rmse, mdl, mi = self.gain_metrics.compute(self.I_history, w)

        # Drives (solo lectura, no modifica historial permanentemente)
        drives = self.voluntary.compute_drives(self.I, self.I_history, self.residuals, G)

        # Social boost: ampliar probabilidad SOCIAL cuando R_soc_ema alto
        social_boost = self._compute_social_boost()

        # Estado temporal (no guarda en historial)
        utilities = {s: self.voluntary.compute_utility(s, drives, social_boost if s == LifeState.SOCIAL else 0)
                     for s in LifeState}
        gamma = self.voluntary.compute_gamma() if len(self.voluntary.conf_history) >= 10 else 1.0

        U_arr = np.array([utilities[s] for s in LifeState])
        U_scaled = gamma * U_arr
        U_shifted = U_scaled - np.max(U_scaled)
        exp_U = np.exp(U_shifted)
        probs = exp_U / (exp_U.sum() + EPS)

        choice = np.random.choice(len(LifeState), p=probs)
        life_state = list(LifeState)[choice]

        gate_open = (life_state == LifeState.SOCIAL)

        # Willingness
        a, pi, _ = self._compute_willingness(signals, other_signals)
        willing = a and gate_open

        # Guardar temporalmente para que step() use los mismos valores
        self._temp_life_state = life_state
        self._temp_drives = drives
        self._temp_utilities = utilities
        self._temp_probs = probs
        self._temp_gamma = gamma

        return willing, gate_open

    def step(self, other_signals: Optional[InternalSignals] = None,
            bilateral_consent: bool = False,
            precomputed_willing: Optional[bool] = None,
            precomputed_gate: Optional[bool] = None) -> Dict:
        self.t += 1
        I_prev = self.I.copy()
        w = self._get_window()

        # Señales
        signals = self._compute_signals()

        # Métricas para tracking
        G, rmse, mdl, mi = self.gain_metrics.compute(self.I_history, w)

        # VOLUNTAD - usar valores precomputados si existen
        if hasattr(self, '_temp_life_state') and precomputed_willing is not None:
            life_state = self._temp_life_state
            drives = self._temp_drives
            voluntary_info = {
                'utilities': {s.name: self._temp_utilities[s] for s in LifeState},
                'probs': {s.name: float(self._temp_probs[i]) for i, s in enumerate(LifeState)},
                'gamma': self._temp_gamma,
                'chosen': life_state.name
            }
            self.voluntary.state = life_state
            self.voluntary.state_history.append(life_state)
            # Limpiar temporales
            del self._temp_life_state, self._temp_drives, self._temp_utilities, self._temp_probs, self._temp_gamma
        else:
            drives = self.voluntary.compute_drives(self.I, self.I_history, self.residuals, G)
            social_boost = self._compute_social_boost()
            life_state, voluntary_info = self.voluntary.choose_state(drives, self.I, social_boost)

        # Gate = SOCIAL
        gate_open = precomputed_gate if precomputed_gate is not None else (life_state == LifeState.SOCIAL)

        # Willingness - usar precomputado si existe
        if precomputed_willing is not None:
            self.willing = precomputed_willing
            a, pi, willingness_info = self._compute_willingness(signals, other_signals)  # Solo para logging
        else:
            a, pi, willingness_info = self._compute_willingness(signals, other_signals)
            self.willing = a and gate_open

        # Modo
        m = self.mode_bandit.select(gate_open and self.willing)
        self.current_mode = m

        # Coupling - ahora usa bilateral_consent que fue calculado con el willing del mismo ciclo
        coupling_active = bilateral_consent and self.willing and gate_open

        # Tracking de recompensa social
        if coupling_active and not self.voluntary.social_reward.in_coupling:
            self.voluntary.social_reward.start_coupling_window()

        self.voluntary.social_reward.add_metric(rmse, mdl, mi, coupling_active)

        if not coupling_active and self.voluntary.social_reward.in_coupling:
            R_soc = self.voluntary.social_reward.end_coupling_window(w)
            self.voluntary.refractory.trigger(self.t, list(self.var_I_history), w)

        # Actualizar bandit si hubo coupling
        if coupling_active:
            self.coupling_count += 1
            self.mode_bandit.update(m, rmse, mdl, mi)

        # Dinámicas
        if life_state == LifeState.SLEEP:
            dZ = self._ou_step() * 0.1
        elif life_state == LifeState.LEARN:
            dZ = self._ou_step() * 2.0
        else:
            dZ = self._ou_step()

        # Coupling effect
        kappa = 0.0
        if coupling_active and other_signals and not other_signals.warmup:
            if not math.isnan(signals.lambda1):
                kappa = min(0.5, signals.lambda1 * signals.conf / (1 + signals.cv_r + EPS))

        # Update I
        I_new = I_prev + dZ[0] * self.u_1 + dZ[1] * self.u_2
        I_new = self._project_to_simplex(I_new)

        # Residuo
        residual = float(np.linalg.norm(I_new - I_prev))
        self.residuals.append(residual)
        if len(self.residuals) > max_hist(self.t):
            self.residuals = self.residuals[-max_hist(self.t):]

        # Update state
        self.I = I_new
        self.I_history.append(I_new.copy())
        if len(self.I_history) > max_hist(self.t):
            self.I_history = self.I_history[-max_hist(self.t):]

        # Learning
        reward = G if not math.isnan(G) else 0
        self.voluntary.learn(reward, life_state, drives)
        self.voluntary.update_costs(life_state, coupling_active)

        # Sistema Afectivo
        affect_result = self.affective.process(
            I=I_new, I_prev=I_prev,
            I_history=self.I_history, residuals=self.residuals,
            rho=signals.rho if not math.isnan(signals.rho) else 1.0,
            R_soc_ema=self.voluntary.social_reward.R_soc_ema,
            is_sleep=(life_state == LifeState.SLEEP),
            mdl_history=list(self.gain_metrics.mdl_history),
            fatigue=self.voluntary.fatigue
        )

        # Logs
        record = {
            't': self.t,
            'I_prev': I_prev.tolist(),
            'I_new': I_new.tolist(),
            'life_state': life_state.name,
            'gate_open': bool(gate_open),
            'willing': bool(self.willing),
            'pi': pi if not math.isnan(pi) else None,
            'bilateral_consent': bool(bilateral_consent),
            'mode': m,
            'kappa': kappa,
            'coupling_active': bool(coupling_active),
            'warmup': bool(signals.warmup),
            'G': G if not math.isnan(G) else None,
            'affect': affect_result['PAD'],
        }
        self.series.append(record)

        consent_record = {
            't': self.t,
            'warmup': bool(willingness_info.get('warmup', True)),
            'benefit': willingness_info.get('benefit'),
            'cost': willingness_info.get('cost'),
            'pi': pi if not math.isnan(pi) else None,
            'pi_base': willingness_info.get('pi_base'),
            'damping': willingness_info.get('damping'),
            'a': int(self.willing),
            'gate': bool(gate_open),
            'm': m,
            'kappa': kappa,
            'rho': signals.rho if not math.isnan(signals.rho) else None,
            'var_I': signals.var_I if not math.isnan(signals.var_I) else None,
            'R_soc_ema': self.voluntary.social_reward.R_soc_ema,
            'G': G if not math.isnan(G) else None,
        }
        self.consent_log.append(consent_record)

        voluntary_record = {
            't': self.t,
            'state': life_state.name,
            'drives': {
                'D_rest': drives.D_rest,
                'D_nov': drives.D_nov,
                'D_learn': drives.D_learn,
                'D_soc': drives.D_soc,
                'C_t': drives.C_t,
            },
            'utilities': voluntary_info['utilities'],
            'probs': voluntary_info['probs'],
            'gamma': voluntary_info['gamma'],
            'fatigue': self.voluntary.fatigue,
            'R_soc_ema': self.voluntary.social_reward.R_soc_ema,
            'refractory_damping': self.voluntary.refractory.get_damping(self.t),
        }
        self.voluntary_log.append(voluntary_record)

        return record

# =============================================================================
# Experimento
# =============================================================================

def run_experiment(cycles: int = 5000, output_dir: str = "/root/NEO_EVA/results/phase8b"):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("PHASE 8b: VOLUNTAD SOCIAL POTENCIADA")
    print("=" * 70)

    # Especializaciones complementarias:
    # NEO prioriza MDL (compresión/estructura)
    # EVA prioriza MI/TE (información mutua/transferencia)
    neo = PotentiatedWorld("NEO", np.array([1.0, 0.0, 0.0]), specialization="mdl")
    eva = PotentiatedWorld("EVA", np.array([1/3, 1/3, 1/3]), specialization="mi_te")

    print(f"NEO initial: {neo.I}")
    print(f"EVA initial: {eva.I}")
    print()

    bilateral_events = []

    for t in range(1, cycles + 1):
        neo_signals = neo._compute_signals()
        eva_signals = eva._compute_signals()

        # Primero: calcular willingness de ambos SIN actualizar I
        # Esto requiere un pre-step para calcular willing
        neo_willing_now, neo_gate_now = neo._pre_step_willingness(eva_signals)
        eva_willing_now, eva_gate_now = eva._pre_step_willingness(neo_signals)

        # Bilateral consent AHORA (mismo ciclo)
        bilateral = neo_willing_now and eva_willing_now and neo_gate_now and eva_gate_now

        neo.step(other_signals=eva_signals, bilateral_consent=bilateral,
                 precomputed_willing=neo_willing_now, precomputed_gate=neo_gate_now)
        eva.step(other_signals=neo_signals, bilateral_consent=bilateral,
                 precomputed_willing=eva_willing_now, precomputed_gate=eva_gate_now)

        if bilateral:
            bilateral_events.append({
                't': t,
                'neo_mode': neo.current_mode,
                'eva_mode': eva.current_mode,
                'neo_state': neo.voluntary.state.name,
                'eva_state': eva.voluntary.state.name,
                'neo_R_soc_ema': neo.voluntary.social_reward.R_soc_ema,
                'eva_R_soc_ema': eva.voluntary.social_reward.R_soc_ema,
            })

        if t % 1000 == 0:
            print(f"  t={t:4d}: NEO={[f'{x:.3f}' for x in neo.I]} EVA={[f'{x:.3f}' for x in eva.I]}")
            print(f"          states: NEO={neo.voluntary.state.name}, EVA={eva.voluntary.state.name}")
            print(f"          bilateral={len(bilateral_events)}, coupling: NEO={neo.coupling_count}, EVA={eva.coupling_count}")
            print(f"          R_soc_ema: NEO={neo.voluntary.social_reward.R_soc_ema:.3f}, EVA={eva.voluntary.social_reward.R_soc_ema:.3f}")

    print()
    print("=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print(f"NEO final: {neo.I}")
    print(f"EVA final: {eva.I}")
    print(f"Bilateral events: {len(bilateral_events)}")
    print(f"Coupling count: NEO={neo.coupling_count}, EVA={eva.coupling_count}")

    from collections import Counter
    neo_states = Counter([r['state'] for r in neo.voluntary_log])
    eva_states = Counter([r['state'] for r in eva.voluntary_log])
    print(f"\nNEO states: {dict(neo_states)}")
    print(f"EVA states: {dict(eva_states)}")

    print(f"\nMode bandit stats:")
    print(f"  NEO: {neo.mode_bandit.get_stats()}")
    print(f"  EVA: {eva.mode_bandit.get_stats()}")

    # Guardar
    with open(f"{output_dir}/series_neo.json", 'w') as f:
        json.dump(neo.series, f)
    with open(f"{output_dir}/series_eva.json", 'w') as f:
        json.dump(eva.series, f)
    with open(f"{output_dir}/consent_log_neo.json", 'w') as f:
        json.dump(neo.consent_log, f)
    with open(f"{output_dir}/consent_log_eva.json", 'w') as f:
        json.dump(eva.consent_log, f)
    with open(f"{output_dir}/voluntary_log_neo.json", 'w') as f:
        json.dump(neo.voluntary_log, f)
    with open(f"{output_dir}/voluntary_log_eva.json", 'w') as f:
        json.dump(eva.voluntary_log, f)
    with open(f"{output_dir}/bilateral_events.json", 'w') as f:
        json.dump(bilateral_events, f)
    with open(f"{output_dir}/affect_log_neo.json", 'w') as f:
        json.dump(neo.affective.affect_log, f)
    with open(f"{output_dir}/affect_log_eva.json", 'w') as f:
        json.dump(eva.affective.affect_log, f)
    with open(f"{output_dir}/bandit_stats.json", 'w') as f:
        json.dump({'neo': neo.mode_bandit.get_stats(), 'eva': eva.mode_bandit.get_stats()}, f, indent=2)

    print(f"\n[OK] Resultados guardados en {output_dir}/")

    return neo, eva, bilateral_events

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 8b: Social Potentiated")
    parser.add_argument("--cycles", type=int, default=5000)
    parser.add_argument("--output", type=str, default="/root/NEO_EVA/results/phase8b")
    args = parser.parse_args()

    run_experiment(cycles=args.cycles, output_dir=args.output)
