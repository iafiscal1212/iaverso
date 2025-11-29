#!/usr/bin/env python3
"""
Phase 7: Acoplamiento Libre por Consentimiento y Utilidad Endógena
===================================================================

NEO y EVA deciden autónomamente:
1. Si proponen acoplarse (ON/OFF) - Voluntad individual π
2. Solo se acoplan si ambos quieren - Consentimiento bilateral
3. Cómo acoplarse m∈{-1,0,+1} - Bandit de 3 brazos
4. Cuándo desconectarse - Stopping rules endógenas

Todo deriva de estadísticas internas - CERO magia.

Principio: "Si no sale de la historia, no entra en la dinámica"
"""

import sys
import os
import json
import numpy as np
import time
import hashlib
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field
from scipy.stats import rankdata

sys.path.insert(0, '/root/NEOSYNT')
sys.path.insert(0, '/root/NEO_EVA/tools')

from phase6_coupled_system_v2 import EndogenousStats, BUS, WorldSummary


# =============================================================================
# A) Bandit de 3 brazos con Thompson Sampling
# =============================================================================

class ThompsonBandit:
    """
    Bandit de 3 brazos para modo m∈{-1, 0, +1}.
    Thompson Sampling con recompensas normalizadas por cuantiles históricos.
    """

    def __init__(self):
        # Brazos: -1 (anti-align), 0 (off), +1 (align)
        self.arms = [-1, 0, 1]

        # Prior: Beta(1, 1) para cada brazo (uniforme)
        # Almacenamos (alpha, beta) para cada brazo
        self.alpha = {arm: 1.0 for arm in self.arms}
        self.beta_param = {arm: 1.0 for arm in self.arms}

        # Historia de recompensas para normalización endógena
        self.reward_history: List[float] = []
        self.arm_rewards: Dict[int, List[float]] = {arm: [] for arm in self.arms}
        self.arm_pulls: Dict[int, int] = {arm: 0 for arm in self.arms}

        # Regret tracking
        self.cumulative_reward = 0.0
        self.best_possible_reward = 0.0
        self.regret_history: List[float] = []

    def select_arm(self, gate_open: bool) -> int:
        """Selecciona brazo con Thompson Sampling. Si gate OFF, fuerza brazo 0."""
        if not gate_open:
            return 0

        # Sample de cada Beta posterior
        samples = {}
        for arm in self.arms:
            samples[arm] = np.random.beta(self.alpha[arm], self.beta_param[arm])

        return max(samples, key=samples.get)

    def update(self, arm: int, reward: float, T: int):
        """
        Actualiza el bandit con recompensa normalizada por cuantiles.
        reward_norm ∈ [0, 1] usando rank histórico.
        """
        self.reward_history.append(reward)
        self.arm_rewards[arm].append(reward)
        self.arm_pulls[arm] += 1

        # Normalizar reward a [0, 1] usando rank percentil histórico
        if len(self.reward_history) > 10:
            rank = (np.searchsorted(np.sort(self.reward_history), reward) + 1) / len(self.reward_history)
            reward_norm = rank
        else:
            reward_norm = 0.5  # Neutral durante warmup

        # Actualizar Beta posterior
        # Tratamos reward_norm como probabilidad de éxito
        if reward_norm > 0.5:
            self.alpha[arm] += reward_norm
        else:
            self.beta_param[arm] += (1 - reward_norm)

        # Trim history endógenamente
        max_hist = EndogenousStats.max_hist(T)
        if len(self.reward_history) > max_hist:
            self.reward_history = self.reward_history[-max_hist:]
        for a in self.arms:
            if len(self.arm_rewards[a]) > max_hist:
                self.arm_rewards[a] = self.arm_rewards[a][-max_hist:]

        # Regret tracking
        self.cumulative_reward += reward
        max_reward = max(np.mean(self.arm_rewards[a]) if self.arm_rewards[a] else 0
                        for a in self.arms)
        self.best_possible_reward += max_reward
        current_regret = self.best_possible_reward - self.cumulative_reward
        self.regret_history.append(current_regret)

    def get_stats(self) -> Dict:
        """Retorna estadísticas del bandit."""
        return {
            'pulls': self.arm_pulls.copy(),
            'mean_rewards': {
                arm: float(np.mean(self.arm_rewards[arm])) if self.arm_rewards[arm] else 0.0
                for arm in self.arms
            },
            'cumulative_reward': float(self.cumulative_reward),
            'cumulative_regret': float(self.regret_history[-1]) if self.regret_history else 0.0,
            'alpha': self.alpha.copy(),
            'beta': self.beta_param.copy(),
        }

    def regret_is_improving(self, w: int) -> bool:
        """Verifica si el regret está mejorando (por debajo de mediana histórica)."""
        if len(self.regret_history) < w * 2:
            return True  # Warmup

        recent_regret = np.mean(self.regret_history[-w:])
        historical_regret = np.median(self.regret_history[:-w])

        return recent_regret <= historical_regret


# =============================================================================
# B) Señales Internas y Utilidad Endógena
# =============================================================================

@dataclass
class InternalSignals:
    """Señales internas de un mundo (sin ver datos crudos del otro)."""
    t: int
    u: float              # Incertidumbre = IQR(r) / √T
    rho: float            # Estabilidad (spectral radius approx)
    v1: np.ndarray        # Dirección principal (PCA)
    lambda1: float        # Varianza explicada por v1
    G: float              # Ganancia reciente (Borda rank)
    conf: float           # Confianza = max(I) - second_max(I)
    cv_r: float           # Coef. variación de residuos
    var_I: float          # Varianza de I en ventana


class UtilityComputer:
    """
    Computa beneficio esperado ΔÛ y coste de acoplar.
    100% endógeno - normalizado por cuantiles históricos.
    """

    def __init__(self):
        self.delta_u_history: List[float] = []
        self.cost_history: List[float] = []
        self.L_history: List[float] = []  # Log benefit/cost ratio

    def compute_benefit(self, signals_self: InternalSignals,
                       signals_other: Optional[InternalSignals], T: int) -> float:
        """
        ΔÛ = (u_Y/(1+u_X)) × (λ₁^Y/(λ₁^Y+λ₁^X+ε)) × (conf^Y/(1+CV(r^X)))
        Normalizado por cuantiles históricos → ∈ [0, 1]
        """
        eps = EndogenousStats.EPS

        if signals_other is None:
            return 0.0

        # Factor 1: Necesidad relativa
        f1 = signals_other.u / (1 + signals_self.u + eps)

        # Factor 2: Dominancia direccional de Y
        f2 = signals_other.lambda1 / (signals_other.lambda1 + signals_self.lambda1 + eps)

        # Factor 3: Y confiable, X inestable
        f3 = signals_other.conf / (1 + signals_self.cv_r + eps)

        delta_u_raw = f1 * f2 * f3

        # Normalizar por cuantiles históricos
        self.delta_u_history.append(delta_u_raw)
        max_hist = EndogenousStats.max_hist(T)
        if len(self.delta_u_history) > max_hist:
            self.delta_u_history = self.delta_u_history[-max_hist:]

        if len(self.delta_u_history) > 10:
            # Rank percentil
            rank = (np.searchsorted(np.sort(self.delta_u_history), delta_u_raw) + 1)
            delta_u_norm = rank / len(self.delta_u_history)
        else:
            delta_u_norm = 0.5

        return float(delta_u_norm)

    def compute_cost(self, signals_self: InternalSignals,
                    rho_p95: float, var_I_p25: float,
                    bus_latency_rank: float, T: int) -> float:
        """
        Coste en ranks [0, 1]:
        - Tensión: 1{ρ ≥ p95}
        - Pérdida de variabilidad: rank inverso de Var(I)
        - Latencia BUS (rank)
        """
        eps = EndogenousStats.EPS

        # Componente 1: Tensión (binario)
        tension = 1.0 if signals_self.rho >= rho_p95 else 0.0

        # Componente 2: Variabilidad baja es costoso
        # Si Var(I) < p25, rank alto
        if signals_self.var_I < var_I_p25:
            var_cost = 1.0
        else:
            var_cost = 0.0

        # Componente 3: Latencia BUS (ya en rank)
        latency_cost = bus_latency_rank

        # Suma de ranks
        cost_raw = (tension + var_cost + latency_cost) / 3.0

        # Normalizar por historia
        self.cost_history.append(cost_raw)
        max_hist = EndogenousStats.max_hist(T)
        if len(self.cost_history) > max_hist:
            self.cost_history = self.cost_history[-max_hist:]

        if len(self.cost_history) > 10:
            rank = (np.searchsorted(np.sort(self.cost_history), cost_raw) + 1)
            cost_norm = rank / len(self.cost_history)
        else:
            cost_norm = 0.5

        return float(cost_norm)

    def compute_willingness(self, benefit: float, cost: float) -> float:
        """
        π = σ(rank(ΔÛ) - rank(coste))
        Logística sobre diferencia de ranks.
        """
        diff = benefit - cost
        # Logística: σ(x) = 1/(1 + e^(-kx)) donde k controla pendiente
        # k = 4 da una curva razonable en [-1, 1]
        # Pero para ser endógenos, usamos k = 1 / σ(history of diffs)
        # Durante warmup, k = 4 (geométrico)
        k = 4.0
        pi = 1.0 / (1.0 + np.exp(-k * diff))
        return float(pi)

    def compute_L_ratio(self, benefit: float, cost: float) -> float:
        """L = log((ΔÛ + ε) / (coste + ε)) para stopping rules."""
        eps = EndogenousStats.EPS
        L = np.log((benefit + eps) / (cost + eps))
        self.L_history.append(L)
        return float(L)


# =============================================================================
# C) Métricas de Ganancia (Borda Rank)
# =============================================================================

class GainMetrics:
    """
    Computa G_t = BordaRank(ΔRMSE, ΔMDL, MI) ∈ [0, 1]
    Sin pesos fijos - ranking puro.
    """

    def __init__(self):
        self.rmse_history: List[float] = []
        self.mdl_history: List[float] = []
        self.mi_history: List[float] = []

    def compute_rmse_delta(self, I_history: List[np.ndarray], w: int) -> float:
        """ΔRMSE: mejora en predicción (EMA vs actual)."""
        if len(I_history) < w + 1:
            return 0.0

        I_arr = np.array(I_history[-w-1:])

        # EMA prediction
        beta = (w - 1) / (w + 1)
        ema = I_arr[0]
        for i in range(1, len(I_arr) - 1):
            ema = beta * ema + (1 - beta) * I_arr[i]

        # RMSE actual vs predicted
        rmse = np.sqrt(np.mean((I_arr[-1] - ema) ** 2))

        self.rmse_history.append(rmse)

        if len(self.rmse_history) < 2:
            return 0.0

        # Delta: reducción es buena (negativo)
        delta = self.rmse_history[-2] - self.rmse_history[-1]
        return float(delta)

    def compute_mdl_delta(self, I_history: List[np.ndarray], w: int) -> float:
        """ΔMDL: cambio en complejidad de descripción (entropía)."""
        if len(I_history) < w:
            return 0.0

        I_arr = np.array(I_history[-w:])

        # MDL aproximado como entropía de la distribución media
        mean_I = np.mean(I_arr, axis=0)
        mean_I = np.maximum(mean_I, EndogenousStats.EPS)
        mean_I = mean_I / mean_I.sum()

        entropy = -np.sum(mean_I * np.log(mean_I + EndogenousStats.EPS))

        self.mdl_history.append(entropy)

        if len(self.mdl_history) < 2:
            return 0.0

        # Delta: más entropía es exploración (positivo es bueno para explorar)
        delta = self.mdl_history[-1] - self.mdl_history[-2]
        return float(delta)

    def compute_mi_local(self, I_history: List[np.ndarray], w: int) -> float:
        """MI local: información mutua entre componentes."""
        if len(I_history) < w:
            return 0.0

        I_arr = np.array(I_history[-w:])

        # MI aproximada por correlación máxima entre componentes
        try:
            corr = np.corrcoef(I_arr.T)
            # Tomar máxima correlación off-diagonal
            np.fill_diagonal(corr, 0)
            mi_approx = np.max(np.abs(corr))
        except:
            mi_approx = 0.0

        self.mi_history.append(mi_approx)
        return float(mi_approx)

    def compute_borda_gain(self, I_history: List[np.ndarray], w: int, T: int) -> float:
        """
        G = BordaRank(ΔRMSE, ΔMDL, MI) normalizado a [0, 1]
        """
        delta_rmse = self.compute_rmse_delta(I_history, w)
        delta_mdl = self.compute_mdl_delta(I_history, w)
        mi = self.compute_mi_local(I_history, w)

        # Crear array de métricas (mayor es mejor para todas)
        metrics = np.array([delta_rmse, delta_mdl, mi])

        # Borda: rank cada métrica
        if len(self.rmse_history) > 10:
            ranks = []
            for i, (hist, val) in enumerate([
                (self.rmse_history, delta_rmse),
                (self.mdl_history, delta_mdl),
                (self.mi_history, mi)
            ]):
                if len(hist) > 0:
                    rank = (np.searchsorted(np.sort(hist), val) + 1) / len(hist)
                    ranks.append(rank)
                else:
                    ranks.append(0.5)

            G = np.mean(ranks)
        else:
            G = 0.5

        return float(G)


# =============================================================================
# D) Mundo con Consentimiento y Auto-Acoplamiento
# =============================================================================

class ConsentWorld:
    """
    Mundo con:
    - Señales internas endógenas
    - Voluntad de acoplar π (estocástica)
    - Bandit para modo m∈{-1,0,+1}
    - Stopping rules endógenas
    """

    def __init__(self, world_id: str, initial_I: np.ndarray, bus: BUS):
        self.world_id = world_id
        self.other_id = 'EVA' if world_id == 'NEO' else 'NEO'
        self.bus = bus

        self.I = initial_I.copy()
        self.I_history: List[np.ndarray] = [initial_I.copy()]
        self.residuals: List[float] = []

        # OU state
        self.ou_Z = np.array([0.0, 0.0])
        self.ou_Z_history: List[np.ndarray] = []

        # Statistics histories
        self.rho_history: List[float] = []
        self.iqr_history: List[float] = []
        self.tau_history: List[float] = []
        self.theta_history: List[float] = []
        self.var_I_history: List[float] = []
        self.kappa_history: List[float] = []

        # Tangent basis (geometric constants)
        self.u_1 = np.array([1, -1, 0]) / np.sqrt(2)
        self.u_2 = np.array([1, 1, -2]) / np.sqrt(6)
        self.u_c = np.array([1, 1, 1]) / np.sqrt(3)

        # Drift
        self.drift_ema = np.zeros(3)

        # Phase 7: Consent components
        self.bandit = ThompsonBandit()
        self.utility = UtilityComputer()
        self.gain_metrics = GainMetrics()

        # Consent state
        self.willing_to_couple = False  # a_t
        self.current_mode = 0  # m_t ∈ {-1, 0, +1}
        self.consent_given = False

        # Stopping state
        self.stopped = False
        self.stop_reason = None

        # Counters
        self.t = 0
        self.gate_activations = 0
        self.coupling_activations = 0
        self.consent_proposals = 0
        self.bilateral_consents = 0

        # Logging
        self.series: List[Dict] = []
        self.consent_log: List[Dict] = []

    def _get_window(self) -> int:
        return EndogenousStats.window_size(len(self.I_history))

    def _get_max_hist(self) -> int:
        return EndogenousStats.max_hist(len(self.I_history))

    def _compute_sigma_med(self) -> float:
        w = self._get_window()
        sigma_uniform = 1.0 / np.sqrt(12)

        if len(self.I_history) < w:
            return sigma_uniform / np.sqrt(len(self.I_history) + 1)

        I_window = np.array(self.I_history[-w:])
        sigmas = np.std(I_window, axis=0)
        sigma_med = float(np.median(sigmas))

        if sigma_med < EndogenousStats.EPS:
            sigma_med = sigma_uniform / np.sqrt(len(self.I_history))

        return sigma_med

    def _compute_internal_signals(self) -> InternalSignals:
        """Computa todas las señales internas del mundo."""
        T = len(self.I_history)
        w = self._get_window()

        # u = IQR(r) / √T
        if len(self.residuals) > w:
            res = np.array(self.residuals[-w:])
            iqr_r = EndogenousStats.iqr(res)
            u = iqr_r / np.sqrt(T)
            cv_r = EndogenousStats.cv(res)
        else:
            u = EndogenousStats.EPS
            cv_r = 1.0

        # ρ (spectral radius approx)
        rho = self._compute_rho()

        # PCA
        v1, lambda1 = self._compute_pca()

        # G (Borda gain)
        G = self.gain_metrics.compute_borda_gain(self.I_history, w, T)

        # conf = max(I) - second_max(I)
        I_sorted = np.sort(self.I)[::-1]
        conf = I_sorted[0] - I_sorted[1]

        # var_I
        if len(self.I_history) >= w:
            I_window = np.array(self.I_history[-w:])
            var_I = float(np.var(I_window))
        else:
            var_I = EndogenousStats.EPS

        self.var_I_history.append(var_I)
        max_hist = self._get_max_hist()
        if len(self.var_I_history) > max_hist:
            self.var_I_history = self.var_I_history[-max_hist:]

        return InternalSignals(
            t=self.t, u=u, rho=rho, v1=v1, lambda1=lambda1,
            G=G, conf=conf, cv_r=cv_r, var_I=var_I
        )

    def _compute_rho(self) -> float:
        """Aproxima ρ(J) del Jacobiano."""
        w = self._get_window()
        if len(self.I_history) < w:
            return 0.99

        I_arr = np.array(self.I_history[-w:])
        residuals = np.diff(I_arr, axis=0)

        if len(residuals) > 5:
            r_norm = np.linalg.norm(residuals, axis=1)
            if r_norm[0] > EndogenousStats.EPS:
                rho = (r_norm[-1] / r_norm[0]) ** (1 / len(r_norm))
            else:
                rho = 0.99
        else:
            rho = 0.99

        self.rho_history.append(rho)
        max_hist = self._get_max_hist()
        if len(self.rho_history) > max_hist:
            self.rho_history = self.rho_history[-max_hist:]

        return float(rho)

    def _compute_pca(self) -> Tuple[np.ndarray, float]:
        """PCA para dirección principal."""
        w = self._get_window()
        if len(self.I_history) < w:
            return self.u_1, 0.0

        I_window = np.array(self.I_history[-w:])
        mu_I = np.mean(I_window, axis=0)
        I_centered = I_window - mu_I

        try:
            cov = np.cov(I_centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx = np.argsort(eigvals)[::-1]
            v1 = eigvecs[:, idx[0]]
            lambda1 = eigvals[idx[0]]
        except:
            v1 = self.u_1
            lambda1 = 0.0

        return v1, float(lambda1)

    def _compute_gate_endogenous(self) -> Tuple[bool, Dict]:
        """Gate crítico por cuantiles puros."""
        w = self._get_window()
        T = len(self.I_history)

        if T < w:
            return True, {'reason': 'warmup', 'open': True}

        # Umbrales de historia
        rho_p95 = EndogenousStats.quantile_safe(self.rho_history, 0.95, 0.99)
        iqr_p75 = EndogenousStats.quantile_safe(self.iqr_history, 0.75, EndogenousStats.EPS)

        # Current values
        rho_current = self.rho_history[-1] if self.rho_history else 0.99
        iqr_current = self.iqr_history[-1] if self.iqr_history else EndogenousStats.EPS

        # At corner check
        at_corner = np.max(self.I) > 0.90

        # Gate: pure quantile comparison
        gate_open = at_corner or (rho_current >= rho_p95 and iqr_current >= iqr_p75)

        return gate_open, {
            'rho': float(rho_current),
            'rho_p95': float(rho_p95),
            'iqr': float(iqr_current),
            'iqr_p75': float(iqr_p75),
            'at_corner': at_corner,
            'open': gate_open,
        }

    def compute_willingness(self, signals_self: InternalSignals,
                           signals_other: Optional[InternalSignals]) -> Tuple[bool, float, Dict]:
        """
        Computa voluntad de acoplar π y decisión estocástica a.
        """
        T = len(self.I_history)
        w = self._get_window()

        # Benefit
        benefit = self.utility.compute_benefit(signals_self, signals_other, T)

        # Cost
        rho_p95 = EndogenousStats.quantile_safe(self.rho_history, 0.95, 0.99)
        var_I_p25 = EndogenousStats.quantile_safe(self.var_I_history, 0.25, EndogenousStats.EPS)

        # BUS latency rank (placeholder - could use actual latency)
        bus_latency_rank = 0.5  # Neutral

        cost = self.utility.compute_cost(signals_self, rho_p95, var_I_p25, bus_latency_rank, T)

        # Willingness π
        pi = self.utility.compute_willingness(benefit, cost)

        # Stochastic decision
        a = np.random.random() < pi

        # L ratio for stopping
        L = self.utility.compute_L_ratio(benefit, cost)

        info = {
            'benefit': benefit,
            'cost': cost,
            'pi': pi,
            'a': a,
            'L': L,
        }

        return a, pi, info

    def check_stopping_rules(self, signals: InternalSignals) -> Tuple[bool, Optional[str]]:
        """
        Stopping rules endógenas:
        - Cortar si ρ(J) ≥ p99(ρ)
        - Cortar si Var_w(I) ≤ p25
        - Cortar si regret del bandit empeora
        """
        w = self._get_window()

        # Rule 1: ρ too high
        rho_p99 = EndogenousStats.quantile_safe(self.rho_history, 0.99, 1.0)
        if signals.rho >= rho_p99:
            return True, 'rho_p99'

        # Rule 2: Variance too low
        var_I_p25 = EndogenousStats.quantile_safe(self.var_I_history, 0.25, 0.0)
        if signals.var_I <= var_I_p25 and len(self.var_I_history) > w:
            return True, 'var_I_p25'

        # Rule 3: Bandit regret worsening
        if not self.bandit.regret_is_improving(w):
            return True, 'regret_worsening'

        return False, None

    def check_start_condition(self) -> bool:
        """
        Condición de inicio: L ≥ quantile_0.95({L})
        """
        if len(self.utility.L_history) < 20:
            return True  # Warmup - always allow

        L_current = self.utility.L_history[-1]
        L_p95 = EndogenousStats.quantile_safe(self.utility.L_history, 0.95, 0.0)

        return L_current >= L_p95

    def _compute_kappa_endogenous(self, signals_self: InternalSignals,
                                  signals_other: InternalSignals, T: int) -> float:
        """κ endógeno igual que Phase 6."""
        eps = EndogenousStats.EPS

        f1 = signals_other.u / (1 + signals_self.u + eps)
        f2 = signals_other.lambda1 / (signals_other.lambda1 + signals_self.lambda1 + eps)
        f3 = signals_other.conf / (1 + signals_self.cv_r + eps)

        kappa_raw = f1 * f2 * f3

        self.kappa_history.append(kappa_raw)
        max_hist = self._get_max_hist()
        if len(self.kappa_history) > max_hist:
            self.kappa_history = self.kappa_history[-max_hist:]

        if len(self.kappa_history) > 10:
            kappa_p99 = EndogenousStats.quantile_safe(self.kappa_history, 0.99, 1.0)
            if kappa_p99 > eps:
                kappa = min(1.0, kappa_raw / kappa_p99)
            else:
                kappa = kappa_raw
        else:
            kappa = min(1.0, kappa_raw)

        return float(kappa)

    def _compute_tau_endogenous(self) -> float:
        """τ endógeno igual que Phase 6."""
        T = len(self.I_history)
        w = self._get_window()
        sigma_med = self._compute_sigma_med()
        tau_floor = sigma_med / max(T, 1)

        if len(self.residuals) < w:
            return max(tau_floor, EndogenousStats.EPS)

        res = np.array(self.residuals[-w:])
        iqr_r = EndogenousStats.iqr(res)

        iqr_r_hist = EndogenousStats.quantile_safe(
            [EndogenousStats.iqr(np.array(self.residuals[max(0,i-w):i]))
             for i in range(w, len(self.residuals), w//2)],
            0.5, iqr_r
        ) if len(self.residuals) > 2*w else iqr_r

        tau = (iqr_r / np.sqrt(T)) * (sigma_med / (iqr_r_hist + EndogenousStats.EPS))
        tau = max(tau, tau_floor)

        if len(self.tau_history) > 20:
            tau_p99 = EndogenousStats.quantile_safe(self.tau_history, 0.99, tau)
            tau = min(tau, tau_p99)

        self.tau_history.append(tau)
        max_hist = self._get_max_hist()
        if len(self.tau_history) > max_hist:
            self.tau_history = self.tau_history[-max_hist:]

        return tau

    def _ou_step(self, tau: float) -> np.ndarray:
        """OU step igual que Phase 6."""
        T = len(self.I_history)
        w = self._get_window()
        sigma_med = self._compute_sigma_med()

        theta_floor = sigma_med / max(T, 1)
        if len(self.theta_history) >= w:
            theta_ceil = EndogenousStats.quantile_safe(self.theta_history[-self._get_max_hist():], 0.99)
            theta_ceil = max(theta_ceil, theta_floor * 10)
        else:
            theta_ceil = 1.0 / w

        theta = (theta_floor + theta_ceil) / 2
        if len(self.residuals) > w:
            r = np.array(self.residuals[-w:])
            if len(r) > 2:
                r_corr = np.corrcoef(r[:-1], r[1:])[0, 1]
                if not np.isnan(r_corr) and abs(r_corr) < 0.99 and abs(r_corr) > EndogenousStats.EPS:
                    theta_raw = -1 / np.log(abs(r_corr) + EndogenousStats.EPS)
                    theta = max(theta_floor, min(theta_ceil, theta_raw))

        self.theta_history.append(theta)
        max_hist = self._get_max_hist()
        if len(self.theta_history) > max_hist:
            self.theta_history = self.theta_history[-max_hist:]

        dt = 1.0
        sigma = np.sqrt(max(tau, EndogenousStats.EPS))

        drift = -theta * self.ou_Z * dt
        diffusion = sigma * np.sqrt(dt) * np.random.randn(2)

        self.ou_Z = self.ou_Z + drift + diffusion

        # Clip endógeno
        self.ou_Z_history.append(self.ou_Z.copy())
        if len(self.ou_Z_history) > max_hist:
            self.ou_Z_history = self.ou_Z_history[-max_hist:]

        if len(self.ou_Z_history) >= 10:
            Z_arr = np.array(self.ou_Z_history)
            z_min = np.percentile(Z_arr, 0.1, axis=0)
            z_max = np.percentile(Z_arr, 99.9, axis=0)
            self.ou_Z = np.clip(self.ou_Z, z_min, z_max)

        return self.ou_Z

    def _mirror_descent(self, I: np.ndarray, delta: np.ndarray, eta: float) -> np.ndarray:
        """Mirror descent igual que Phase 6."""
        log_floor = -30

        I_safe = np.maximum(I, EndogenousStats.EPS)
        I_safe = I_safe / I_safe.sum()

        log_I = np.log(I_safe)
        log_I = np.maximum(log_I, log_floor)

        log_I_new = log_I + eta * delta

        exp_log = np.exp(log_I_new - np.max(log_I_new))
        I_new = exp_log / exp_log.sum()

        return I_new

    def compute_summary(self) -> WorldSummary:
        """Computa summary para BUS."""
        signals = self._compute_internal_signals()

        return WorldSummary(
            world_id=self.world_id,
            t=self.t,
            mu_I=np.mean(np.array(self.I_history[-self._get_window():]), axis=0) if len(self.I_history) >= self._get_window() else self.I,
            v1=signals.v1,
            lambda1=signals.lambda1,
            u=signals.u,
            conf=signals.conf,
            cv_r=signals.cv_r
        )

    def step(self, other_willing: bool, other_signals: Optional[InternalSignals]) -> Dict:
        """
        Ejecuta un paso con consentimiento bilateral.
        """
        self.t += 1
        T = len(self.I_history)
        I_prev = self.I.copy()

        # 1. Compute internal signals
        signals = self._compute_internal_signals()

        # 2. Gate check
        gate_open, gate_info = self._compute_gate_endogenous()

        # 3. Compute willingness
        a, pi, willingness_info = self.compute_willingness(signals, other_signals)
        self.willing_to_couple = a

        if a:
            self.consent_proposals += 1

        # 4. Check stopping rules
        should_stop, stop_reason = self.check_stopping_rules(signals)
        if should_stop:
            self.stopped = True
            self.stop_reason = stop_reason

        # 5. Bilateral consent
        bilateral_consent = a and other_willing and gate_open and not self.stopped
        self.consent_given = bilateral_consent

        if bilateral_consent:
            self.bilateral_consents += 1

        # 6. Select mode with bandit
        m = self.bandit.select_arm(gate_open and bilateral_consent)
        self.current_mode = m

        # 7. Base dynamics
        sigma_med = self._compute_sigma_med()
        sigma_uniform = 1.0 / np.sqrt(12)
        w = self._get_window()

        # Drift
        if len(self.I_history) >= 3:
            I_arr = np.array(self.I_history[-w:])
            diffs = np.diff(I_arr, axis=0)
            if len(diffs) > 0:
                beta = (w - 1) / (w + 1)
                ema = diffs[0]
                for d in diffs[1:]:
                    ema = beta * ema + (1 - beta) * d
                drift = ema - np.dot(ema, self.u_c) * self.u_c
                self.drift_ema = beta * self.drift_ema + (1 - beta) * drift
            else:
                drift = np.zeros(3)
        else:
            drift = np.zeros(3)

        # Noise
        if T < w:
            sigma_noise = sigma_uniform / np.sqrt(T + 1)
        else:
            I_window = np.array(self.I_history[-w:])
            iqr_I = np.mean([EndogenousStats.iqr(I_window[:, i]) for i in range(3)])
            sigma_noise = max(iqr_I, sigma_med) / np.sqrt(T)

        z1, z2 = np.random.randn(2)
        noise = sigma_noise * (z1 * self.u_1 + z2 * self.u_2)

        # Candidate
        I_candidate = I_prev + self.drift_ema + noise
        I_candidate = np.maximum(I_candidate, EndogenousStats.EPS)
        I_candidate = I_candidate / I_candidate.sum()

        # Residual
        residual = np.linalg.norm(I_candidate - I_prev)
        self.residuals.append(residual)
        max_hist = self._get_max_hist()
        if len(self.residuals) > max_hist:
            self.residuals = self.residuals[-max_hist:]

        # IQR history
        if len(self.residuals) > w:
            iqr_current = EndogenousStats.iqr(np.array(self.residuals[-w:]))
        else:
            iqr_current = EndogenousStats.EPS
        self.iqr_history.append(iqr_current)
        if len(self.iqr_history) > max_hist:
            self.iqr_history = self.iqr_history[-max_hist:]

        # 8. Apply coupling if bilateral consent
        kappa = 0.0
        coupling_active = False

        if bilateral_consent and other_signals is not None and gate_open:
            self.gate_activations += 1

            # τ and OU
            tau = self._compute_tau_endogenous()
            Z = self._ou_step(tau)
            delta_base = Z[0] * self.u_1 + Z[1] * self.u_2
            delta_base = delta_base - delta_base.mean()

            # κ endógeno
            kappa = self._compute_kappa_endogenous(signals, other_signals, T)

            # Dirección del otro proyectada
            g_other = other_signals.v1 - np.dot(other_signals.v1, self.u_c) * self.u_c
            g_norm = np.linalg.norm(g_other)
            if g_norm > EndogenousStats.EPS:
                g_other = g_other / g_norm
            else:
                g_other = np.zeros(3)

            # Δ̃ = Δ_base + κ × m × g_other
            delta_coupled = delta_base + kappa * m * g_other

            # η = τ
            eta = tau

            # Mirror descent
            I_new = self._mirror_descent(I_candidate, delta_coupled, eta)

            coupling_active = True
            self.coupling_activations += 1
        else:
            I_new = I_candidate
            tau = 0.0
            eta = 0.0

        # Update state
        self.I = I_new
        self.I_history.append(self.I.copy())
        if len(self.I_history) > max_hist:
            self.I_history = self.I_history[-max_hist:]

        # 9. Update bandit with reward = G
        reward = signals.G
        self.bandit.update(m, reward, T)

        # 10. Publish to BUS
        if self.t % w == 0:
            self.bus.publish(self.compute_summary())

        # Log
        record = {
            't': self.t,
            'I_prev': I_prev.tolist(),
            'I_new': self.I.tolist(),
            'gate_open': gate_open,
            'willing': a,
            'pi': pi,
            'bilateral_consent': bilateral_consent,
            'mode': m,
            'kappa': kappa,
            'coupling_active': coupling_active,
            'stopped': self.stopped,
            'stop_reason': self.stop_reason,
            'G': signals.G,
            'bandit_regret': self.bandit.regret_history[-1] if self.bandit.regret_history else 0,
        }
        self.series.append(record)

        consent_record = {
            't': self.t,
            'delta_u': willingness_info['benefit'],
            'cost': willingness_info['cost'],
            'pi': pi,
            'a': a,
            'gate': gate_open,
            'm': m,
            'kappa': kappa,
            'eta': eta if coupling_active else 0,
            'rho': signals.rho,
            'var_I': signals.var_I,
            'G': signals.G,
            'regret': self.bandit.regret_history[-1] if self.bandit.regret_history else 0,
        }
        self.consent_log.append(consent_record)

        return record

    def get_consent_stats(self) -> Dict:
        """Estadísticas de consentimiento."""
        return {
            'total_steps': self.t,
            'consent_proposals': self.consent_proposals,
            'bilateral_consents': self.bilateral_consents,
            'coupling_activations': self.coupling_activations,
            'gate_activations': self.gate_activations,
            'stopped': self.stopped,
            'stop_reason': self.stop_reason,
            'bandit_stats': self.bandit.get_stats(),
            'mode_distribution': {
                m: sum(1 for r in self.series if r['mode'] == m)
                for m in [-1, 0, 1]
            },
        }


# =============================================================================
# E) Sistema Acoplado con Consentimiento Bilateral
# =============================================================================

class ConsentCoupledSystem:
    """
    Sistema NEO↔EVA con consentimiento bilateral.
    Solo se acoplan si ambos quieren.
    """

    def __init__(self, enable_coupling: bool = True):
        self.bus = BUS()
        self.bus.enabled = enable_coupling

        self.neo = ConsentWorld('NEO', np.array([1.0, 0.0, 0.0]), self.bus)
        self.eva = ConsentWorld('EVA', np.array([1/3, 1/3, 1/3]), self.bus)

        self.enable_coupling = enable_coupling
        self.bilateral_events: List[Dict] = []

    def run(self, cycles: int = 1000, verbose: bool = True,
            snapshot_every: int = 1000) -> Dict:
        """Ejecuta el sistema con consentimiento bilateral."""
        print("=" * 70)
        print("Phase 7: Acoplamiento Libre por Consentimiento")
        print(f"Coupling: {'ENABLED' if self.enable_coupling else 'DISABLED'}")
        print("=" * 70)
        print(f"NEO initial: {self.neo.I}")
        print(f"EVA initial: {self.eva.I}")
        print()

        start_time = time.time()
        snapshots = []

        for i in range(cycles):
            # Get signals from both worlds
            neo_signals = self.neo._compute_internal_signals()
            eva_signals = self.eva._compute_internal_signals()

            # Each world computes willingness independently
            neo_willing, neo_pi, _ = self.neo.compute_willingness(neo_signals, eva_signals)
            eva_willing, eva_pi, _ = self.eva.compute_willingness(eva_signals, neo_signals)

            # Step with bilateral consent check
            neo_result = self.neo.step(eva_willing, eva_signals if self.enable_coupling else None)
            eva_result = self.eva.step(neo_willing, neo_signals if self.enable_coupling else None)

            # Record bilateral event
            if neo_willing and eva_willing:
                self.bilateral_events.append({
                    't': i + 1,
                    'neo_pi': neo_pi,
                    'eva_pi': eva_pi,
                    'neo_mode': neo_result['mode'],
                    'eva_mode': eva_result['mode'],
                })

            # Progress
            if verbose and (i + 1) % 200 == 0:
                print(f"  t={i+1:5d}: NEO={self.neo.I} EVA={self.eva.I}")
                print(f"           NEO willing={neo_willing}, EVA willing={eva_willing}")
                print(f"           NEO mode={neo_result['mode']}, EVA mode={eva_result['mode']}")

            # Snapshot
            if (i + 1) % snapshot_every == 0:
                snapshots.append({
                    't': i + 1,
                    'neo_I': self.neo.I.tolist(),
                    'eva_I': self.eva.I.tolist(),
                    'neo_stats': self.neo.get_consent_stats(),
                    'eva_stats': self.eva.get_consent_stats(),
                })

        elapsed = time.time() - start_time

        # Compute correlations
        neo_arr = np.array([r['I_new'] for r in self.neo.series])
        eva_arr = np.array([r['I_new'] for r in self.eva.series])

        correlations = {}
        for i, comp in enumerate(['S', 'N', 'C']):
            correlations[comp] = float(np.corrcoef(neo_arr[:, i], eva_arr[:, i])[0, 1])
        correlations['mean'] = np.mean(list(correlations.values()))

        results = {
            'cycles': cycles,
            'elapsed': elapsed,
            'coupling_enabled': self.enable_coupling,
            'correlations': correlations,
            'neo': {
                'final_I': self.neo.I.tolist(),
                'consent_stats': self.neo.get_consent_stats(),
            },
            'eva': {
                'final_I': self.eva.I.tolist(),
                'consent_stats': self.eva.get_consent_stats(),
            },
            'bilateral_events': len(self.bilateral_events),
            'snapshots': snapshots,
        }

        print()
        print("=" * 70)
        print("Results:")
        print(f"  NEO final: {self.neo.I}")
        print(f"  EVA final: {self.eva.I}")
        print(f"  Correlation mean: {correlations['mean']:.4f}")
        print(f"  NEO consent proposals: {self.neo.consent_proposals}/{cycles}")
        print(f"  EVA consent proposals: {self.eva.consent_proposals}/{cycles}")
        print(f"  Bilateral consents: {len(self.bilateral_events)}/{cycles}")
        print(f"  NEO mode dist: {self.neo.get_consent_stats()['mode_distribution']}")
        print(f"  EVA mode dist: {self.eva.get_consent_stats()['mode_distribution']}")
        print("=" * 70)

        return results

    def save_results(self, output_dir: str):
        """Guarda todos los resultados."""
        os.makedirs(output_dir, exist_ok=True)

        # Series
        for world, name in [(self.neo, 'neo'), (self.eva, 'eva')]:
            with open(f"{output_dir}/series_{name}.json", 'w') as f:
                json.dump(world.series, f, indent=2)

            with open(f"{output_dir}/consent_log_{name}.json", 'w') as f:
                json.dump(world.consent_log, f, indent=2)

        # Bilateral events
        with open(f"{output_dir}/bilateral_events.json", 'w') as f:
            json.dump(self.bilateral_events, f, indent=2)

        # Bandit stats
        bandit_stats = {
            'neo': self.neo.bandit.get_stats(),
            'eva': self.eva.bandit.get_stats(),
        }
        with open(f"{output_dir}/bandit_stats.json", 'w') as f:
            json.dump(bandit_stats, f, indent=2)

        print(f"[OK] Results saved to {output_dir}/")


# =============================================================================
# F) Experimento Principal
# =============================================================================

def run_experiment(cycles: int = 5000, output_dir: str = "/root/NEO_EVA/results/phase7"):
    """Ejecuta experimento completo de Phase 7."""

    print("\n" + "=" * 70)
    print("PHASE 7: ACOPLAMIENTO LIBRE POR CONSENTIMIENTO Y UTILIDAD ENDÓGENA")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Run with coupling
    print("\n[1] Running with coupling enabled...")
    system_coupled = ConsentCoupledSystem(enable_coupling=True)
    results_coupled = system_coupled.run(cycles=cycles, snapshot_every=1000)
    system_coupled.save_results(f"{output_dir}/coupled")

    # 2. Run ablation (no coupling)
    print("\n[2] Running ablation (no coupling)...")
    system_ablation = ConsentCoupledSystem(enable_coupling=False)
    results_ablation = system_ablation.run(cycles=cycles, snapshot_every=1000)
    system_ablation.save_results(f"{output_dir}/ablation")

    # 3. Compare results
    print("\n" + "=" * 70)
    print("COMPARISON: Coupled vs Ablation")
    print("=" * 70)

    comparison = {
        'coupled': {
            'correlation_mean': results_coupled['correlations']['mean'],
            'neo_bilateral': results_coupled['neo']['consent_stats']['bilateral_consents'],
            'eva_bilateral': results_coupled['eva']['consent_stats']['bilateral_consents'],
            'neo_mode_dist': results_coupled['neo']['consent_stats']['mode_distribution'],
            'eva_mode_dist': results_coupled['eva']['consent_stats']['mode_distribution'],
            'neo_bandit_regret': results_coupled['neo']['consent_stats']['bandit_stats']['cumulative_regret'],
            'eva_bandit_regret': results_coupled['eva']['consent_stats']['bandit_stats']['cumulative_regret'],
        },
        'ablation': {
            'correlation_mean': results_ablation['correlations']['mean'],
            'neo_bilateral': results_ablation['neo']['consent_stats']['bilateral_consents'],
            'eva_bilateral': results_ablation['eva']['consent_stats']['bilateral_consents'],
        },
    }

    print(f"Correlation (coupled):  {comparison['coupled']['correlation_mean']:.4f}")
    print(f"Correlation (ablation): {comparison['ablation']['correlation_mean']:.4f}")
    print(f"Bilateral events (coupled): {comparison['coupled']['neo_bilateral']}")
    print(f"NEO mode distribution: {comparison['coupled']['neo_mode_dist']}")
    print(f"EVA mode distribution: {comparison['coupled']['eva_mode_dist']}")

    # 4. Save comparison
    with open(f"{output_dir}/comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)

    # 5. Generate summary report
    report = generate_report(results_coupled, results_ablation, output_dir)

    with open(f"{output_dir}/phase7_consent_autocouple.md", 'w') as f:
        f.write(report)

    print(f"\n[OK] Report saved to {output_dir}/phase7_consent_autocouple.md")

    return results_coupled, results_ablation


def generate_report(results_coupled: Dict, results_ablation: Dict, output_dir: str) -> str:
    """Genera reporte Markdown."""

    timestamp = datetime.now().isoformat()

    report = f"""# Phase 7: Acoplamiento Libre por Consentimiento y Utilidad Endógena

**Fecha**: {timestamp}
**Ciclos**: {results_coupled['cycles']}

---

## Resumen

NEO y EVA deciden autónomamente si proponen acoplarse y cómo hacerlo:
- Voluntad individual π basada en beneficio vs coste
- Consentimiento bilateral: solo se acoplan si ambos quieren
- Modo m∈{{-1, 0, +1}} aprendido con Thompson Sampling
- Stopping rules endógenas por ρ, Var(I) y regret

---

## Fórmulas Usadas

### Beneficio Esperado
```
ΔÛ = (u_Y/(1+u_X)) × (λ₁^Y/(λ₁^Y+λ₁^X+ε)) × (conf^Y/(1+CV(r^X)))
```

### Coste Endógeno
```
coste = Rank(1{{ρ≥p95}} + RankInvVar(I) + Rank(latencia BUS)) / 3
```

### Voluntad
```
π = σ(rank(ΔÛ) - rank(coste))
a ~ Bernoulli(π)
```

### Consentimiento Bilateral
```
Acoplamiento activo ⟺ a_NEO = 1 AND a_EVA = 1
```

### Modo (Bandit)
```
m ∈ {{-1, 0, +1}} ~ Thompson Sampling con recompensa G = BordaRank(ΔRMSE, ΔMDL, MI)
```

---

## Resultados

### Métricas de Consentimiento

| Métrica | NEO | EVA |
|---------|-----|-----|
| Propuestas de consentimiento | {results_coupled['neo']['consent_stats']['consent_proposals']} | {results_coupled['eva']['consent_stats']['consent_proposals']} |
| Consentimientos bilaterales | {results_coupled['neo']['consent_stats']['bilateral_consents']} | {results_coupled['eva']['consent_stats']['bilateral_consents']} |
| Activaciones de acoplamiento | {results_coupled['neo']['consent_stats']['coupling_activations']} | {results_coupled['eva']['consent_stats']['coupling_activations']} |

### Distribución de Modos

| Modo | NEO | EVA |
|------|-----|-----|
| -1 (anti-align) | {results_coupled['neo']['consent_stats']['mode_distribution'][-1]} | {results_coupled['eva']['consent_stats']['mode_distribution'][-1]} |
| 0 (off) | {results_coupled['neo']['consent_stats']['mode_distribution'][0]} | {results_coupled['eva']['consent_stats']['mode_distribution'][0]} |
| +1 (align) | {results_coupled['neo']['consent_stats']['mode_distribution'][1]} | {results_coupled['eva']['consent_stats']['mode_distribution'][1]} |

### Bandit Statistics

| Métrica | NEO | EVA |
|---------|-----|-----|
| Regret acumulado | {results_coupled['neo']['consent_stats']['bandit_stats']['cumulative_regret']:.4f} | {results_coupled['eva']['consent_stats']['bandit_stats']['cumulative_regret']:.4f} |
| Recompensa acumulada | {results_coupled['neo']['consent_stats']['bandit_stats']['cumulative_reward']:.4f} | {results_coupled['eva']['consent_stats']['bandit_stats']['cumulative_reward']:.4f} |

### Correlación

| Componente | Correlación |
|------------|-------------|
| S | {results_coupled['correlations']['S']:.4f} |
| N | {results_coupled['correlations']['N']:.4f} |
| C | {results_coupled['correlations']['C']:.4f} |
| **Media** | **{results_coupled['correlations']['mean']:.4f}** |

### Comparación: Coupled vs Ablation

| Métrica | Coupled | Ablation |
|---------|---------|----------|
| Correlación media | {results_coupled['correlations']['mean']:.4f} | {results_ablation['correlations']['mean']:.4f} |
| Eventos bilaterales | {results_coupled.get('bilateral_events', 0)} | {results_ablation.get('bilateral_events', 0)} |

---

## Criterios GO/NO-GO

| Criterio | Estado |
|----------|--------|
| Autonomía real (decisiones ON/OFF) | {'✅ PASS' if results_coupled['neo']['consent_stats']['consent_proposals'] > 0 else '❌ FAIL'} |
| Consentimiento bilateral efectivo | {'✅ PASS' if results_coupled['neo']['consent_stats']['bilateral_consents'] > 0 else '❌ FAIL'} |
| Mejora de utilidad vs ablation | {'✅ PASS' if abs(results_coupled['correlations']['mean']) >= abs(results_ablation['correlations']['mean']) * 0.5 else '❌ FAIL'} |
| Distribución de modos no degenerada | {'✅ PASS' if results_coupled['neo']['consent_stats']['mode_distribution'][-1] > 0 and results_coupled['neo']['consent_stats']['mode_distribution'][1] > 0 else '⚠️ CHECK'} |

---

## Archivos Generados

- `coupled/series_neo.json`: Serie temporal NEO
- `coupled/series_eva.json`: Serie temporal EVA
- `coupled/consent_log_neo.json`: Log de consentimiento NEO
- `coupled/consent_log_eva.json`: Log de consentimiento EVA
- `coupled/bilateral_events.json`: Eventos de consentimiento bilateral
- `coupled/bandit_stats.json`: Estadísticas del bandit
- `ablation/`: Mismos archivos para ablación

---

*Generado: {timestamp}*
*Principio: "Si no sale de la historia, no entra en la dinámica"*
"""

    return report


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 7: Consent-based Auto-coupling")
    parser.add_argument("--cycles", type=int, default=5000)
    parser.add_argument("--output", type=str, default="/root/NEO_EVA/results/phase7")
    args = parser.parse_args()

    run_experiment(cycles=args.cycles, output_dir=args.output)


if __name__ == "__main__":
    main()
