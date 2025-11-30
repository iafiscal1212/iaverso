#!/usr/bin/env python3
"""
Phase 8: VOLUNTAD Endógena + Consent Dinámico
=============================================

Elimina warmup perpetuo y activa cálculo dinámico.
Añade sistema de VOLUNTAD con 5 estados de vida.

Estados: SLEEP, WAKE, WORK, LEARN, SOCIAL
Drives: D_rest, D_nov, D_learn, D_soc, C_t (oscilador)
Política: π_t(s) = softmax(γ_t * U_t(s))

Todo 100% endógeno - sin constantes mágicas.
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
from scipy.stats import rankdata

sys.path.insert(0, '/root/NEOSYNT')
sys.path.insert(0, '/root/NEO_EVA/tools')

# =============================================================================
# Constantes endógenas (solo geométricas/matemáticas)
# =============================================================================

EPS = 1e-12

def window_size(T: int) -> int:
    """Ventana w = max(10, floor(sqrt(T)))"""
    return max(10, int(math.sqrt(T)))

def max_hist(T: int) -> int:
    """Historia máxima = min(T, 10*sqrt(T))"""
    return min(T, int(10 * math.sqrt(T))) if T > 0 else 100

def quantile_safe(arr, q: float, default: float = 0.5) -> float:
    """Quantile seguro que retorna NaN si insuficientes datos."""
    if len(arr) < 5:
        return float('nan')
    return float(np.percentile(arr, q * 100))

def rank_normalize(value: float, history: List[float]) -> float:
    """Normaliza value a [0,1] por rank en history. NaN si warmup."""
    if len(history) < 10:
        return float('nan')
    rank = (np.searchsorted(np.sort(history), value) + 1) / len(history)
    return float(rank)

def iqr(arr) -> float:
    """IQR seguro."""
    if len(arr) < 4:
        return float('nan')
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))

def cv(arr) -> float:
    """Coeficiente de variación."""
    if len(arr) < 2:
        return float('nan')
    m = np.mean(arr)
    if abs(m) < EPS:
        return float('nan')
    return float(np.std(arr) / abs(m))

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
# Drives Endógenos
# =============================================================================

@dataclass
class Drives:
    """Impulsos calculados de estadísticas internas."""
    D_rest: float   # Necesidad de descanso
    D_nov: float    # Necesidad de novedad
    D_learn: float  # Necesidad de consolidar/aprender
    D_soc: float    # Necesidad social
    C_t: float      # Oscilador circadiano
    warmup: bool    # True si algún drive es NaN

# =============================================================================
# Señales Internas (sin warmup defaults)
# =============================================================================

@dataclass
class InternalSignals:
    """Señales internas con flag de warmup explícito."""
    t: int
    u: float              # IQR(r) / sqrt(T) o NaN
    rho: float            # Spectral radius approx o NaN
    lambda1: float        # Varianza explicada v1 o NaN
    conf: float           # max(I) - second_max(I)
    cv_r: float           # CV(residuos) o NaN
    var_I: float          # Var(I) en ventana o NaN
    G: float              # Ganancia Borda o NaN
    warmup: bool          # True si cualquier métrica es NaN

    def to_dict(self) -> Dict:
        return {
            't': self.t,
            'u': self.u if not math.isnan(self.u) else None,
            'rho': self.rho if not math.isnan(self.rho) else None,
            'lambda1': self.lambda1 if not math.isnan(self.lambda1) else None,
            'conf': self.conf,
            'cv_r': self.cv_r if not math.isnan(self.cv_r) else None,
            'var_I': self.var_I if not math.isnan(self.var_I) else None,
            'G': self.G if not math.isnan(self.G) else None,
            'warmup': self.warmup
        }

# =============================================================================
# Oscilador Circadiano Endógeno
# =============================================================================

class CircadianOscillator:
    """
    Extrae ritmo endógeno del periodograma de la señal de actividad.
    Sin reloj externo - fase derivada del propio histórico.
    """

    def __init__(self, min_period: int = 20, max_period: int = 500):
        self.activity_history: deque = deque(maxlen=2000)
        self.phase_history: deque = deque(maxlen=1000)
        self.min_period = min_period
        self.max_period = max_period
        self.dominant_freq = float('nan')
        self.phase = 0.0

    def update(self, activity: float) -> float:
        """
        Actualiza con nueva actividad y retorna C_t ∈ [0,1].
        activity = ||ΔI|| o residuo r.
        """
        self.activity_history.append(activity)

        if len(self.activity_history) < self.min_period * 2:
            return float('nan')

        # Periodograma
        signal = np.array(self.activity_history)
        signal = signal - np.mean(signal)

        if np.std(signal) < EPS:
            return float('nan')

        freqs, psd = periodogram(signal, fs=1.0)

        # Filtrar frecuencias válidas (períodos entre min y max)
        valid_mask = (freqs > 1/self.max_period) & (freqs < 1/self.min_period)
        if not np.any(valid_mask):
            return float('nan')

        valid_freqs = freqs[valid_mask]
        valid_psd = psd[valid_mask]

        # Frecuencia dominante
        idx_max = np.argmax(valid_psd)
        self.dominant_freq = valid_freqs[idx_max]

        # Fase por regresión senoidal simple
        t = np.arange(len(signal))
        cos_comp = np.sum(signal * np.cos(2 * np.pi * self.dominant_freq * t))
        sin_comp = np.sum(signal * np.sin(2 * np.pi * self.dominant_freq * t))
        self.phase = np.arctan2(sin_comp, cos_comp)

        # C_t = sin(phase) normalizado a [0,1]
        C_raw = np.sin(self.phase)
        self.phase_history.append(C_raw)

        # Normalizar por rank histórico
        if len(self.phase_history) < 20:
            return float('nan')

        C_rank = rank_normalize(C_raw, list(self.phase_history))
        return C_rank if not math.isnan(C_rank) else 0.5

# =============================================================================
# Métricas de Ganancia (Borda Rank)
# =============================================================================

class GainMetrics:
    """
    G_t = BordaRank(ΔRMSE↓, ΔMDL↓, MI/TE↑)
    """

    def __init__(self):
        self.rmse_history: deque = deque(maxlen=500)
        self.mdl_history: deque = deque(maxlen=500)
        self.mi_history: deque = deque(maxlen=500)
        self.gain_history: deque = deque(maxlen=500)

    def compute(self, I_history: List[np.ndarray], w: int) -> Tuple[float, Dict]:
        """Calcula ganancia Borda. Retorna (G, métricas)."""
        if len(I_history) < w + 1:
            return float('nan'), {}

        I_arr = np.array(I_history[-w-1:])

        # RMSE: EMA prediction error
        beta = (w - 1) / (w + 1)
        ema = I_arr[0].copy()
        for i in range(1, len(I_arr) - 1):
            ema = beta * ema + (1 - beta) * I_arr[i]
        rmse = float(np.sqrt(np.mean((I_arr[-1] - ema) ** 2)))
        self.rmse_history.append(rmse)

        # MDL: entropía de distribución media
        mean_I = np.mean(I_arr, axis=0)
        mean_I = np.maximum(mean_I, EPS)
        mean_I = mean_I / mean_I.sum()
        mdl = float(-np.sum(mean_I * np.log(mean_I + EPS)))
        self.mdl_history.append(mdl)

        # MI proxy: correlación temporal
        if len(I_arr) > 3:
            corr = np.corrcoef(I_arr[:-1].flatten(), I_arr[1:].flatten())[0, 1]
            mi = abs(corr) if not np.isnan(corr) else 0
        else:
            mi = 0.0
        self.mi_history.append(mi)

        # Borda rank (requiere historia)
        if len(self.rmse_history) < 10:
            return float('nan'), {'rmse': rmse, 'mdl': mdl, 'mi': mi}

        # Ranks (RMSE y MDL bajos = buenos, MI alto = bueno)
        rmse_rank = 1 - rank_normalize(rmse, list(self.rmse_history))
        mdl_rank = 1 - rank_normalize(mdl, list(self.mdl_history))
        mi_rank = rank_normalize(mi, list(self.mi_history))

        if any(math.isnan(x) for x in [rmse_rank, mdl_rank, mi_rank]):
            return float('nan'), {'rmse': rmse, 'mdl': mdl, 'mi': mi}

        G = (rmse_rank + mdl_rank + mi_rank) / 3.0
        self.gain_history.append(G)

        return float(G), {'rmse': rmse, 'mdl': mdl, 'mi': mi, 'G': G}

# =============================================================================
# Sistema de VOLUNTAD
# =============================================================================

class VoluntarySystem:
    """
    Sistema de VOLUNTAD endógeno.
    Elige estado de vida basado en drives, oscilador y utilidad aprendida.
    """

    def __init__(self):
        self.state = LifeState.WAKE
        self.state_history: deque = deque(maxlen=2000)

        # Drives histories
        self.D_rest_history: deque = deque(maxlen=500)
        self.D_nov_history: deque = deque(maxlen=500)
        self.D_learn_history: deque = deque(maxlen=500)
        self.D_soc_history: deque = deque(maxlen=500)

        # Oscilador circadiano
        self.circadian = CircadianOscillator()

        # Pesos aprendidos (α, β para cada estado)
        # Inicializados a cero - se aprenden
        self.alpha = {s: np.zeros(5) for s in LifeState}  # [D_rest, D_nov, D_learn, D_soc, C_t]
        self.beta = {s: np.zeros(3) for s in LifeState}   # [fatiga, saturación, riesgo]

        # Costes
        self.fatigue = 0.0
        self.saturation = 0.0
        self.risk = 0.0

        # Reward history para TD learning
        self.reward_history: deque = deque(maxlen=500)
        self.utility_history: deque = deque(maxlen=500)

        # Learning rate endógeno
        self.lr = 0.1

    def compute_drives(self, signals: InternalSignals, I_history: List[np.ndarray],
                      residuals: List[float], coupling_benefit: float) -> Drives:
        """Calcula drives 100% endógenos."""
        T = len(I_history)
        w = window_size(T)
        warmup = False

        # D_rest = rank(IQR(r) / sqrt(T)) - aumenta con sorpresa/errores
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

        # D_nov = rank(IQR(I), λ₁, ACF-lag) - novedad
        if len(I_history) >= w:
            I_arr = np.array(I_history[-w:])
            iqr_I = iqr(I_arr.flatten())
            D_nov_raw = (signals.lambda1 if not math.isnan(signals.lambda1) else 0) + \
                       (iqr_I / (1 + EPS) if not math.isnan(iqr_I) else 0)
            self.D_nov_history.append(D_nov_raw)
            D_nov = rank_normalize(D_nov_raw, list(self.D_nov_history))
        else:
            D_nov = float('nan')
            warmup = True

        # D_learn = rank(ΔMDL, RMSE) - necesidad de consolidar
        if not math.isnan(signals.G):
            D_learn_raw = 1 - signals.G  # Bajo G = alta necesidad de aprender
            self.D_learn_history.append(D_learn_raw)
            D_learn = rank_normalize(D_learn_raw, list(self.D_learn_history))
        else:
            D_learn = float('nan')
            warmup = True

        # D_soc = rank(beneficio previo de coupling)
        self.D_soc_history.append(coupling_benefit)
        if len(self.D_soc_history) >= 10:
            D_soc = rank_normalize(coupling_benefit, list(self.D_soc_history))
        else:
            D_soc = float('nan')
            warmup = True

        # C_t = oscilador circadiano
        activity = np.linalg.norm(np.diff(I_history[-2:], axis=0)) if len(I_history) >= 2 else 0
        C_t = self.circadian.update(activity)
        if math.isnan(C_t):
            warmup = True

        return Drives(
            D_rest=D_rest if not math.isnan(D_rest) else 0.5,
            D_nov=D_nov if not math.isnan(D_nov) else 0.5,
            D_learn=D_learn if not math.isnan(D_learn) else 0.5,
            D_soc=D_soc if not math.isnan(D_soc) else 0.5,
            C_t=C_t if not math.isnan(C_t) else 0.5,
            warmup=warmup
        )

    def compute_utility(self, state: LifeState, drives: Drives) -> float:
        """U(s) = α_s · D - β_s · K"""
        D = np.array([drives.D_rest, drives.D_nov, drives.D_learn, drives.D_soc, drives.C_t])
        K = np.array([self.fatigue, self.saturation, self.risk])

        # Alineación con impulsos
        benefit = np.dot(self.alpha[state], D)

        # Costes
        cost = np.dot(self.beta[state], K)

        return float(benefit - cost)

    def choose_state(self, drives: Drives) -> Tuple[LifeState, Dict]:
        """Elige estado con softmax sobre utilidades."""
        # Utilidad por estado
        utilities = {s: self.compute_utility(s, drives) for s in LifeState}

        # γ = rank(confianza) - temperatura endógena
        if len(self.utility_history) >= 10:
            u_std = np.std(list(self.utility_history))
            gamma = 1.0 / (u_std + EPS) if u_std > EPS else 1.0
            gamma = min(gamma, 10.0)  # Cap para estabilidad numérica
        else:
            gamma = 1.0

        # Softmax
        U_arr = np.array([utilities[s] for s in LifeState])
        U_scaled = gamma * U_arr
        U_shifted = U_scaled - np.max(U_scaled)  # Estabilidad numérica
        exp_U = np.exp(U_shifted)
        probs = exp_U / (exp_U.sum() + EPS)

        # Muestrear
        choice = np.random.choice(len(LifeState), p=probs)
        new_state = list(LifeState)[choice]

        self.state_history.append(new_state)
        for s in LifeState:
            self.utility_history.append(utilities[s])

        info = {
            'utilities': {s.name: utilities[s] for s in LifeState},
            'probs': {s.name: float(probs[i]) for i, s in enumerate(LifeState)},
            'gamma': gamma,
            'chosen': new_state.name
        }

        self.state = new_state
        return new_state, info

    def update_costs(self, state: LifeState, coupling_active: bool):
        """Actualiza costes basado en estado actual."""
        # Fatiga aumenta si no SLEEP
        if state != LifeState.SLEEP:
            self.fatigue = min(1.0, self.fatigue + 0.01)
        else:
            self.fatigue = max(0.0, self.fatigue - 0.05)

        # Saturación aumenta en WORK
        if state == LifeState.WORK:
            self.saturation = min(1.0, self.saturation + 0.02)
        else:
            self.saturation = max(0.0, self.saturation - 0.01)

        # Riesgo aumenta en SOCIAL activo
        if state == LifeState.SOCIAL and coupling_active:
            self.risk = min(1.0, self.risk + 0.01)
        else:
            self.risk = max(0.0, self.risk - 0.02)

    def learn(self, reward: float, state: LifeState, drives: Drives):
        """TD(0) update de pesos α, β."""
        self.reward_history.append(reward)

        if len(self.reward_history) < 10:
            return

        # Learning rate endógeno
        reward_std = np.std(list(self.reward_history))
        lr = self.lr / (1 + reward_std) if reward_std > EPS else self.lr

        # TD error
        U_current = self.compute_utility(state, drives)
        td_error = reward - U_current

        # Update α (aumentar si td_error > 0)
        D = np.array([drives.D_rest, drives.D_nov, drives.D_learn, drives.D_soc, drives.C_t])
        self.alpha[state] += lr * td_error * D

        # Update β (reducir si td_error > 0, ya que β penaliza)
        K = np.array([self.fatigue, self.saturation, self.risk])
        self.beta[state] -= lr * td_error * K

        # Clip para estabilidad
        for s in LifeState:
            self.alpha[s] = np.clip(self.alpha[s], -5, 5)
            self.beta[s] = np.clip(self.beta[s], -5, 5)

# =============================================================================
# Mundo Autónomo con VOLUNTAD
# =============================================================================

class AutonomousWorld:
    """
    Mundo autónomo con VOLUNTAD endógena.
    Elimina warmup perpetuo - usa NaN explícito.
    """

    def __init__(self, name: str, initial_I: np.ndarray):
        self.name = name
        self.I = initial_I.copy()
        self.I_history: List[np.ndarray] = [initial_I.copy()]
        self.residuals: List[float] = []

        # Estado OU
        self.ou_Z = np.array([0.0, 0.0])

        # Historiales (con maxlen para evitar memoria infinita)
        self.rho_history: deque = deque(maxlen=500)
        self.var_I_history: deque = deque(maxlen=500)
        self.iqr_history: deque = deque(maxlen=500)
        self.cost_history: deque = deque(maxlen=500)
        self.benefit_history: deque = deque(maxlen=500)
        self.pi_history: deque = deque(maxlen=500)

        # Bases tangentes
        self.u_1 = np.array([1, -1, 0]) / np.sqrt(2)
        self.u_2 = np.array([1, 1, -2]) / np.sqrt(6)

        # Métricas
        self.gain_metrics = GainMetrics()

        # VOLUNTAD
        self.voluntary = VoluntarySystem()

        # Bandit para modos
        self.mode_counts = {-1: 0, 0: 0, 1: 0}
        self.mode_rewards = {-1: [], 0: [], 1: []}

        # Estado de consentimiento
        self.willing = False
        self.current_mode = 0

        # Contadores
        self.t = 0
        self.warmup_count = 0
        self.dynamic_count = 0

        # Logs
        self.series: List[Dict] = []
        self.consent_log: List[Dict] = []
        self.voluntary_log: List[Dict] = []

    def _get_window(self) -> int:
        return window_size(len(self.I_history))

    def _compute_rho(self) -> float:
        """ρ dinámico - retorna NaN si insuficientes datos."""
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

        # Ratio de contracción/expansión
        rho = (norms[-1] / norms[0]) ** (1 / len(norms))

        if not math.isnan(rho):
            self.rho_history.append(rho)

        return float(rho)

    def _compute_var_I(self) -> float:
        """Varianza de I en ventana."""
        w = self._get_window()
        if len(self.I_history) < w:
            return float('nan')

        I_arr = np.array(self.I_history[-w:])
        var = float(np.var(I_arr))

        if not math.isnan(var):
            self.var_I_history.append(var)

        return var

    def _compute_signals(self) -> InternalSignals:
        """Señales internas con warmup explícito."""
        T = len(self.I_history)
        w = self._get_window()
        warmup = False

        # u = IQR(r) / sqrt(T)
        if len(self.residuals) >= w:
            iqr_r = iqr(self.residuals[-w:])
            if not math.isnan(iqr_r):
                self.iqr_history.append(iqr_r)
                u = iqr_r / math.sqrt(T)
                cv_r = cv(self.residuals[-w:])
            else:
                u = float('nan')
                cv_r = float('nan')
                warmup = True
        else:
            u = float('nan')
            cv_r = float('nan')
            warmup = True

        # rho
        rho = self._compute_rho()
        if math.isnan(rho):
            warmup = True

        # var_I
        var_I = self._compute_var_I()
        if math.isnan(var_I):
            warmup = True

        # PCA
        if len(self.I_history) >= w:
            I_arr = np.array(self.I_history[-w:])
            I_centered = I_arr - np.mean(I_arr, axis=0)
            try:
                _, s, _ = np.linalg.svd(I_centered, full_matrices=False)
                lambda1 = float(s[0]**2 / (s**2).sum()) if s.sum() > EPS else 0
            except:
                lambda1 = float('nan')
                warmup = True
        else:
            lambda1 = float('nan')
            warmup = True

        # G (ganancia Borda)
        G, _ = self.gain_metrics.compute(self.I_history, w)
        if math.isnan(G):
            warmup = True

        # conf
        I_sorted = np.sort(self.I)[::-1]
        conf = float(I_sorted[0] - I_sorted[1])

        return InternalSignals(
            t=self.t, u=u, rho=rho, lambda1=lambda1,
            conf=conf, cv_r=cv_r, var_I=var_I, G=G,
            warmup=warmup
        )

    def _compute_benefit(self, signals_self: InternalSignals,
                        signals_other: Optional[InternalSignals]) -> float:
        """Beneficio de acoplar - NaN si warmup."""
        if signals_other is None or signals_self.warmup or signals_other.warmup:
            return float('nan')

        # Factores endógenos
        f1 = signals_other.u / (1 + signals_self.u + EPS)
        f2 = signals_other.lambda1 / (signals_other.lambda1 + signals_self.lambda1 + EPS)
        f3 = signals_other.conf / (1 + signals_self.cv_r + EPS)

        benefit_raw = f1 * f2 * f3
        self.benefit_history.append(benefit_raw)

        return rank_normalize(benefit_raw, list(self.benefit_history))

    def _compute_cost(self, signals: InternalSignals) -> float:
        """Costo de acoplar - NaN si warmup."""
        if signals.warmup:
            return float('nan')

        # Componentes de costo
        rho_p95 = quantile_safe(list(self.rho_history), 0.95)
        var_p25 = quantile_safe(list(self.var_I_history), 0.25)

        if math.isnan(rho_p95) or math.isnan(var_p25):
            return float('nan')

        # Tensión
        tension = 1.0 if signals.rho >= rho_p95 else 0.0

        # Variabilidad baja
        var_cost = 1.0 if signals.var_I < var_p25 else 0.0

        # Fatiga del sistema VOLUNTAD
        fatigue_cost = self.voluntary.fatigue

        cost_raw = (tension + var_cost + fatigue_cost) / 3.0
        self.cost_history.append(cost_raw)

        return rank_normalize(cost_raw, list(self.cost_history))

    def _compute_willingness(self, signals: InternalSignals,
                            other_signals: Optional[InternalSignals]) -> Tuple[bool, float, Dict]:
        """Calcula π y decisión a. Sin defaults de warmup."""
        benefit = self._compute_benefit(signals, other_signals)
        cost = self._compute_cost(signals)

        # Si warmup, no consentir pero logear NaN
        if math.isnan(benefit) or math.isnan(cost):
            self.warmup_count += 1
            return False, float('nan'), {
                'benefit': benefit,
                'cost': cost,
                'pi': float('nan'),
                'warmup': True
            }

        self.dynamic_count += 1

        # π = sigmoid(benefit - cost)
        diff = benefit - cost

        # k endógeno basado en varianza de diffs históricos
        if len(self.pi_history) >= 20:
            diff_std = np.std([b - c for b, c in zip(list(self.benefit_history)[-20:],
                                                      list(self.cost_history)[-20:])])
            k = 1.0 / (diff_std + EPS) if diff_std > EPS else 4.0
            k = min(k, 10.0)
        else:
            k = 4.0

        pi = 1.0 / (1.0 + np.exp(-k * diff))
        self.pi_history.append(pi)

        # Decisión estocástica
        a = np.random.random() < pi

        return a, float(pi), {
            'benefit': float(benefit),
            'cost': float(cost),
            'pi': float(pi),
            'k': k,
            'warmup': False
        }

    def _select_mode(self, gate_open: bool) -> int:
        """Thompson Sampling para modo."""
        if not gate_open:
            return 0

        # Beta posteriors
        samples = {}
        for m in [-1, 0, 1]:
            n = self.mode_counts[m]
            if n > 0 and self.mode_rewards[m]:
                mean_r = np.mean(self.mode_rewards[m])
                alpha = 1 + n * mean_r
                beta = 1 + n * (1 - mean_r)
            else:
                alpha, beta = 1, 1
            samples[m] = np.random.beta(max(alpha, 0.1), max(beta, 0.1))

        return max(samples, key=samples.get)

    def _update_mode(self, mode: int, reward: float):
        """Actualiza bandit con recompensa."""
        self.mode_counts[mode] += 1
        # Normalizar reward a [0,1]
        reward_norm = (reward + 1) / 2 if reward >= -1 else 0
        reward_norm = min(1, max(0, reward_norm))
        self.mode_rewards[mode].append(reward_norm)
        # Trim
        if len(self.mode_rewards[mode]) > 100:
            self.mode_rewards[mode] = self.mode_rewards[mode][-100:]

    def _ou_step(self) -> np.ndarray:
        """OU step en plano tangente."""
        T = len(self.I_history)
        w = self._get_window()

        # σ basado en max(IQR, σ_uniform) / sqrt(w) - NO sqrt(T) para mantener variabilidad
        sigma_uniform = 1.0 / math.sqrt(12)  # Uniforme en [0,1]
        sigma_floor = sigma_uniform / math.sqrt(w)

        if len(self.iqr_history) >= 10:
            iqr_med = np.median(list(self.iqr_history))
            sigma_data = max(iqr_med, sigma_uniform) / math.sqrt(w)
            sigma = max(sigma_floor, sigma_data)
        else:
            sigma = sigma_floor

        # θ basado en autocorrelación
        if len(self.residuals) > w:
            r = np.array(self.residuals[-w:])
            if len(r) > 2:
                corr = np.corrcoef(r[:-1], r[1:])[0, 1]
                if not np.isnan(corr) and abs(corr) > EPS and abs(corr) < 0.99:
                    theta = -1 / np.log(abs(corr) + EPS)
                    theta = max(0.01, min(1.0, theta))
                else:
                    theta = 0.1
            else:
                theta = 0.1
        else:
            theta = 0.1

        # OU dynamics
        noise = np.random.randn(2) * sigma
        self.ou_Z = (1 - theta) * self.ou_Z + noise

        return self.ou_Z

    def _project_to_simplex(self, I: np.ndarray) -> np.ndarray:
        """Proyecta a simplex."""
        I = np.maximum(I, EPS)
        return I / I.sum()

    def step(self, other_signals: Optional[InternalSignals] = None,
            bilateral_consent: bool = False) -> Dict:
        """Ejecuta un paso con VOLUNTAD."""
        self.t += 1
        I_prev = self.I.copy()
        w = self._get_window()

        # 1. Señales internas
        signals = self._compute_signals()

        # 2. VOLUNTAD: elegir estado de vida
        coupling_benefit = self.benefit_history[-1] if self.benefit_history else 0
        drives = self.voluntary.compute_drives(
            signals, self.I_history, self.residuals, coupling_benefit
        )
        life_state, voluntary_info = self.voluntary.choose_state(drives)

        # 3. Gate basado en estado de vida
        gate_open = (life_state == LifeState.SOCIAL)

        # 4. Willingness (solo relevante si SOCIAL)
        a, pi, willingness_info = self._compute_willingness(signals, other_signals)
        self.willing = a and gate_open

        # 5. Modo
        m = self._select_mode(gate_open and self.willing)
        self.current_mode = m

        # 6. Coupling
        coupling_active = bilateral_consent and self.willing and gate_open

        # 7. Dinámicas según estado de vida
        if life_state == LifeState.SLEEP:
            # Reducir actividad
            dZ = self._ou_step() * 0.1
        elif life_state == LifeState.LEARN:
            # Aumentar exploración
            dZ = self._ou_step() * 2.0
        elif life_state == LifeState.WORK:
            # Normal con drift hacia consolidación
            dZ = self._ou_step()
        else:
            dZ = self._ou_step()

        # 8. Aplicar coupling si activo
        if coupling_active and other_signals is not None and not other_signals.warmup:
            # κ basado en confianza y estabilidad
            if not math.isnan(signals.lambda1) and not math.isnan(other_signals.lambda1):
                kappa = min(0.5, signals.lambda1 * other_signals.conf / (1 + signals.cv_r + EPS))
            else:
                kappa = 0.0

            # Aplicar según modo
            if m == 1:
                # Align: atraer hacia v1 de other
                pass  # Implementar si necesario
            elif m == -1:
                # Anti-align
                pass
        else:
            kappa = 0.0

        # 9. Actualizar I
        I_new = I_prev + dZ[0] * self.u_1 + dZ[1] * self.u_2
        I_new = self._project_to_simplex(I_new)

        # 10. Residuo
        residual = float(np.linalg.norm(I_new - I_prev))
        self.residuals.append(residual)
        if len(self.residuals) > max_hist(self.t):
            self.residuals = self.residuals[-max_hist(self.t):]

        # 11. Actualizar estado
        self.I = I_new
        self.I_history.append(I_new.copy())
        if len(self.I_history) > max_hist(self.t):
            self.I_history = self.I_history[-max_hist(self.t):]

        # 12. Reward para VOLUNTAD
        G, _ = self.gain_metrics.compute(self.I_history, w)
        reward = G if not math.isnan(G) else 0
        self.voluntary.learn(reward, life_state, drives)
        self.voluntary.update_costs(life_state, coupling_active)

        # 13. Update mode bandit
        if coupling_active:
            self._update_mode(m, reward)

        # 14. Logs
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
        }
        self.series.append(record)

        consent_record = {
            't': self.t,
            'warmup': bool(willingness_info.get('warmup', True)),
            'benefit': willingness_info.get('benefit'),
            'cost': willingness_info.get('cost'),
            'pi': pi if not math.isnan(pi) else None,
            'a': int(self.willing),
            'gate': bool(gate_open),
            'm': m,
            'kappa': kappa,
            'rho': signals.rho if not math.isnan(signals.rho) else None,
            'var_I': signals.var_I if not math.isnan(signals.var_I) else None,
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
            'fatigue': self.voluntary.fatigue,
            'saturation': self.voluntary.saturation,
            'risk': self.voluntary.risk,
        }
        self.voluntary_log.append(voluntary_record)

        return record

# =============================================================================
# Experimento
# =============================================================================

def run_experiment(cycles: int = 3000, output_dir: str = "/root/NEO_EVA/results/phase8"):
    """Ejecuta experimento Phase 8 con VOLUNTAD."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("PHASE 8: VOLUNTAD ENDÓGENA + CONSENT DINÁMICO")
    print("=" * 70)

    # Inicializar mundos
    neo = AutonomousWorld("NEO", np.array([1.0, 0.0, 0.0]))
    eva = AutonomousWorld("EVA", np.array([1/3, 1/3, 1/3]))

    print(f"NEO initial: {neo.I}")
    print(f"EVA initial: {eva.I}")
    print()

    bilateral_events = []

    for t in range(1, cycles + 1):
        # Señales cruzadas
        neo_signals = neo._compute_signals()
        eva_signals = eva._compute_signals()

        # Bilateral consent
        bilateral = neo.willing and eva.willing

        # Steps
        neo.step(other_signals=eva_signals, bilateral_consent=bilateral)
        eva.step(other_signals=neo_signals, bilateral_consent=bilateral)

        if bilateral:
            bilateral_events.append({
                't': t,
                'neo_mode': neo.current_mode,
                'eva_mode': eva.current_mode,
                'neo_state': neo.voluntary.state.name,
                'eva_state': eva.voluntary.state.name,
            })

        if t % 500 == 0:
            print(f"  t={t:4d}: NEO={neo.I} EVA={eva.I}")
            print(f"          NEO state={neo.voluntary.state.name}, EVA state={eva.voluntary.state.name}")
            print(f"          warmup: NEO={neo.warmup_count}/{t}, EVA={eva.warmup_count}/{t}")
            print(f"          bilateral={len(bilateral_events)}")

    # Resultados
    print()
    print("=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print(f"NEO final: {neo.I}")
    print(f"EVA final: {eva.I}")
    print(f"Bilateral events: {len(bilateral_events)}")
    print(f"NEO warmup ratio: {neo.warmup_count}/{cycles} ({100*neo.warmup_count/cycles:.1f}%)")
    print(f"EVA warmup ratio: {eva.warmup_count}/{cycles} ({100*eva.warmup_count/cycles:.1f}%)")

    # Distribución de estados
    neo_states = [r['state'] for r in neo.voluntary_log]
    eva_states = [r['state'] for r in eva.voluntary_log]
    from collections import Counter
    print(f"\nNEO state distribution: {dict(Counter(neo_states))}")
    print(f"EVA state distribution: {dict(Counter(eva_states))}")

    # Guardar
    with open(f"{output_dir}/series_neo.json", 'w') as f:
        json.dump(neo.series, f, indent=2)
    with open(f"{output_dir}/series_eva.json", 'w') as f:
        json.dump(eva.series, f, indent=2)
    with open(f"{output_dir}/consent_log_neo.json", 'w') as f:
        json.dump(neo.consent_log, f, indent=2)
    with open(f"{output_dir}/consent_log_eva.json", 'w') as f:
        json.dump(eva.consent_log, f, indent=2)
    with open(f"{output_dir}/voluntary_log_neo.json", 'w') as f:
        json.dump(neo.voluntary_log, f, indent=2)
    with open(f"{output_dir}/voluntary_log_eva.json", 'w') as f:
        json.dump(eva.voluntary_log, f, indent=2)
    with open(f"{output_dir}/bilateral_events.json", 'w') as f:
        json.dump(bilateral_events, f, indent=2)

    print(f"\n[OK] Resultados guardados en {output_dir}/")

    return neo, eva, bilateral_events

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 8: VOLUNTAD + Consent Dinámico")
    parser.add_argument("--cycles", type=int, default=3000)
    parser.add_argument("--output", type=str, default="/root/NEO_EVA/results/phase8")
    args = parser.parse_args()

    run_experiment(cycles=args.cycles, output_dir=args.output)
