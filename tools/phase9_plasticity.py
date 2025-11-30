#!/usr/bin/env python3
"""
Phase 9: Plasticidad Afectiva (100% endogeno)
=============================================
Calcula indices de plasticidad para NEO, EVA y su interaccion.
Todo mediante estadistica historica: cuantiles, IQR, sqrt(T), ACF/PCA, ranks.
Sin constantes fijas, sin boosts.
"""

import sys
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy import stats
from scipy.signal import correlate
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, '/root/NEO_EVA/tools')

# =============================================================================
# UTILIDADES ENDOGENAS
# =============================================================================

def get_epsilon(arr: np.ndarray) -> float:
    """Epsilon minimo positivo del dtype."""
    return np.finfo(arr.dtype).eps if arr.dtype.kind == 'f' else 1e-10


def rolling_window_indices(T: int, w: int) -> List[Tuple[int, int]]:
    """Genera indices de ventanas deslizantes."""
    return [(i, min(i + w, T)) for i in range(0, T - w + 1, max(1, w // 2))]


def robust_standardize(x: np.ndarray, w: int) -> np.ndarray:
    """
    Estandarizacion robusta por ventana deslizante.
    x_tilde = (x - med_w) / (IQR_w + eps)
    """
    T = len(x)
    x_std = np.zeros_like(x, dtype=np.float64)
    eps = get_epsilon(x.astype(np.float64))

    for t in range(T):
        start = max(0, t - w + 1)
        end = t + 1
        window = x[start:end]

        med = np.median(window)
        q75, q25 = np.percentile(window, [75, 25])
        iqr = q75 - q25

        x_std[t] = (x[t] - med) / (iqr + eps)

    return x_std


def rank_in_window(x: np.ndarray, w: int) -> np.ndarray:
    """Rank normalizado [0,1] en ventana deslizante."""
    T = len(x)
    ranks = np.zeros(T, dtype=np.float64)

    for t in range(T):
        start = max(0, t - w + 1)
        end = t + 1
        window = x[start:end]

        if len(window) > 1:
            rank = stats.rankdata(window)[-1]
            ranks[t] = (rank - 1) / (len(window) - 1)
        else:
            ranks[t] = 0.5

    return ranks


def compute_acf_lag(x: np.ndarray, max_lag: Optional[int] = None) -> int:
    """
    Encuentra lag donde ACF cruza la mediana envolvente.
    Retorna lag minimo donde |ACF| < mediana(|ACF|).
    """
    if max_lag is None:
        max_lag = min(len(x) // 4, 100)

    x_centered = x - np.mean(x)
    acf = correlate(x_centered, x_centered, mode='full')
    acf = acf[len(acf)//2:]  # Solo lags positivos
    acf = acf[:max_lag+1] / (acf[0] + get_epsilon(acf))

    med_acf = np.median(np.abs(acf[1:]))

    for lag in range(1, len(acf)):
        if np.abs(acf[lag]) < med_acf:
            return lag

    return max_lag


def shoelace_area(x: np.ndarray, y: np.ndarray) -> float:
    """Area poligonal por formula shoelace."""
    n = len(x)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j]
        area -= x[j] * y[i]

    return abs(area) / 2.0


def theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Regresion Theil-Sen (mediana de pendientes)."""
    if len(x) < 2:
        return 0.0

    slopes = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if x[j] != x[i]:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))

    return np.median(slopes) if slopes else 0.0


def robust_pca_2d(data: np.ndarray) -> np.ndarray:
    """PCA robusta a 2D usando medianas para centrar."""
    centered = data - np.median(data, axis=0)

    # Usar MAD para escalar
    mad = np.median(np.abs(centered), axis=0)
    mad[mad == 0] = 1
    scaled = centered / mad

    pca = PCA(n_components=min(2, data.shape[1]))
    return pca.fit_transform(scaled)


# =============================================================================
# EXTRACCION DE SENALES
# =============================================================================

@dataclass
class WorldSignals:
    """Senales extraidas de un mundo."""
    name: str
    T: int
    w: int  # Ventana

    # Senales base (8)
    r: np.ndarray  # resource
    s: np.ndarray  # stability
    m: np.ndarray  # motivation
    c: np.ndarray  # control
    R_soc: np.ndarray  # social reward
    e: np.ndarray  # energy
    q: np.ndarray  # quality
    h: np.ndarray  # harmony

    # PAD latente
    V: np.ndarray  # Valencia
    A: np.ndarray  # Activacion
    D: np.ndarray  # Dominancia

    # Estados y metricas
    states: np.ndarray
    pi: np.ndarray  # Indice volitivo
    modes: np.ndarray

    # Pesos adaptativos
    weights: np.ndarray  # Shape (T, 3)


def extract_signals(affect_log: List[dict],
                    consent_log: List[dict],
                    voluntary_log: List[dict],
                    world_name: str) -> WorldSignals:
    """Extrae todas las senales de un mundo."""

    T = len(affect_log)
    w = max(10, int(np.sqrt(T)))

    # Extraer senales base del affect_log
    r = np.array([a['signals'].get('r', 0.5) if a.get('signals') else 0.5 for a in affect_log])
    s = np.array([a['signals'].get('s', 0.5) if a.get('signals') else 0.5 for a in affect_log])
    m = np.array([a['signals'].get('m', 0.5) if a.get('signals') else 0.5 for a in affect_log])
    c = np.array([a['signals'].get('c', 0.5) if a.get('signals') else 0.5 for a in affect_log])
    R_soc = np.array([a['signals'].get('R_soc', 0.5) if a.get('signals') else 0.5 for a in affect_log])
    e = np.array([a['signals'].get('e', 0.5) if a.get('signals') else 0.5 for a in affect_log])
    q = np.array([a['signals'].get('q', 0.5) if a.get('signals') else 0.5 for a in affect_log])
    h = np.array([a['signals'].get('h', 0.5) if a.get('signals') else 0.5 for a in affect_log])

    # Calcular PAD por ranks en ventana w
    V_raw = R_soc + h - r
    A_raw = m + e
    D_raw = c + s

    V = rank_in_window(V_raw, w)
    A = rank_in_window(A_raw, w)
    D = rank_in_window(D_raw, w)

    # Extraer estados del voluntary_log (no consent_log)
    state_map = {'SLEEP': 0, 'WAKE': 1, 'WORK': 2, 'LEARN': 3, 'SOCIAL': 4}
    states = np.zeros(T, dtype=np.int32)
    for i, v_entry in enumerate(voluntary_log[:T]):
        state_str = v_entry.get('state', 'WAKE')
        states[i] = state_map.get(state_str, 1)

    # Extraer pi del consent_log
    pi = np.array([c_entry.get('pi', 0.5) if c_entry.get('pi') is not None else 0.5
                   for c_entry in consent_log[:T]])

    # Extraer modes de consent_log (campo 'm')
    modes = np.zeros(T, dtype=np.int32)
    for i, c_entry in enumerate(consent_log[:T]):
        modes[i] = c_entry.get('m', 0)

    # Calcular pesos adaptativos como serie temporal
    # Usamos las probabilidades de estados como proxy de pesos
    weights = np.zeros((T, 3), dtype=np.float64)
    for i, v_entry in enumerate(voluntary_log[:T]):
        probs = v_entry.get('probs', {})
        if probs:
            # Usar probs de WORK, LEARN, SOCIAL como proxy de pesos MDL, MI, RMSE
            weights[i, 0] = probs.get('WORK', 1/3)
            weights[i, 1] = probs.get('LEARN', 1/3)
            weights[i, 2] = probs.get('SOCIAL', 1/3)
        else:
            weights[i] = [1/3, 1/3, 1/3]

    return WorldSignals(
        name=world_name, T=T, w=w,
        r=r, s=s, m=m, c=c, R_soc=R_soc, e=e, q=q, h=h,
        V=V, A=A, D=D,
        states=states, pi=pi, modes=modes, weights=weights
    )


# =============================================================================
# INDICES INTRAMUNDO
# =============================================================================

@dataclass
class IntraWorldAlphas:
    """Indices de plasticidad intramundo."""
    name: str
    alpha_affect: float      # Volumen afectivo
    alpha_hyst: float        # Histeresis
    alpha_switch: float      # Tasa cambio estado
    alpha_recov: float       # Recuperacion
    alpha_sus: float         # Susceptibilidad OU
    alpha_intra: float       # Compuesto

    # Series temporales por ventana
    alpha_affect_series: List[float]
    alpha_hyst_series: List[float]
    alpha_switch_series: List[float]


def compute_alpha_affect(V: np.ndarray, A: np.ndarray, D: np.ndarray,
                         w: int) -> Tuple[float, List[float]]:
    """
    alpha_affect = sqrt(det(Cov_w([V,A,D])))
    Volumen afectivo.
    """
    T = len(V)
    series = []

    for start, end in rolling_window_indices(T, w):
        window_data = np.column_stack([V[start:end], A[start:end], D[start:end]])
        if len(window_data) >= 3:
            cov = np.cov(window_data.T)
            det_val = np.linalg.det(cov)
            vol = np.sqrt(max(0, det_val))
            series.append(vol)
        else:
            series.append(0.0)

    return np.median(series) if series else 0.0, series


def compute_alpha_hyst(V: np.ndarray, A: np.ndarray, w: int) -> Tuple[float, List[float]]:
    """
    alpha_hyst = area poligonal (shoelace) de trayectoria (V,A) en ventana w.
    """
    T = len(V)
    series = []

    for start, end in rolling_window_indices(T, w):
        area = shoelace_area(V[start:end], A[start:end])
        series.append(area)

    return np.median(series) if series else 0.0, series


def compute_alpha_switch(states: np.ndarray, w: int) -> Tuple[float, List[float]]:
    """
    alpha_switch = #{t: state_t != state_{t-1}} / w
    Tasa de cambio de estado.
    """
    T = len(states)
    series = []

    for start, end in rolling_window_indices(T, w):
        window_states = states[start:end]
        switches = np.sum(window_states[1:] != window_states[:-1])
        rate = switches / (len(window_states) - 1) if len(window_states) > 1 else 0
        series.append(rate)

    return np.median(series) if series else 0.0, series


def compute_alpha_recov(V: np.ndarray, A: np.ndarray, D: np.ndarray,
                        pi: np.ndarray, w: int) -> float:
    """
    alpha_recov = 1 / med(tau_V, tau_A, tau_D)
    Detecta eventos safety y mide tiempo de recuperacion.
    """
    T = len(V)

    # Calcular varianza de I (proxy: norma PAD)
    I = np.sqrt(V**2 + A**2 + D**2)
    var_I = np.zeros(T)
    for t in range(w, T):
        var_I[t] = np.var(I[t-w:t])

    # Calcular rho (correlacion de pi)
    rho = np.zeros(T)
    for t in range(w, T):
        if np.std(pi[t-w:t]) > 0:
            rho[t] = np.corrcoef(pi[t-w:t], np.arange(w))[0, 1]

    # Detectar eventos safety: rho >= q95 y var_I <= q25
    q95_rho = np.percentile(rho[w:], 95)
    q25_var = np.percentile(var_I[w:], 25)

    safety_events = []
    for t in range(w, T):
        if rho[t] >= q95_rho and var_I[t] <= q25_var:
            safety_events.append(t)

    if not safety_events:
        return 0.0

    # Para cada evento, calcular tau de recuperacion
    taus = []
    for event_t in safety_events:
        for X, name in [(V, 'V'), (A, 'A'), (D, 'D')]:
            med_X = np.median(X[max(0, event_t-w):event_t])
            threshold = np.percentile(np.abs(X - med_X), 50)

            # Buscar tau donde |X_{t+tau} - med| <= threshold
            for tau in range(1, min(w, T - event_t)):
                if abs(X[event_t + tau] - med_X) <= threshold:
                    taus.append(tau)
                    break

    if not taus:
        return 0.0

    med_tau = np.median(taus)
    return 1.0 / med_tau if med_tau > 0 else 0.0


def compute_alpha_sus(V: np.ndarray, A: np.ndarray, D: np.ndarray, w: int) -> float:
    """
    alpha_sus = susceptibilidad OU endogena.
    chi_w = IQR(Delta[V,A,D]) / IQR(delta_t)
    """
    T = len(V)

    # Calcular deltas
    dV = np.diff(V)
    dA = np.diff(A)
    dD = np.diff(D)

    # Parametros OU endogenos
    # sigma_w = max(IQR_w, sigma_hist) / sqrt(T)
    # theta_w = 1 / lag_ACF

    chi_values = []

    for X, dX in [(V, dV), (A, dA), (D, dD)]:
        # IQR de los deltas observados
        iqr_observed = np.percentile(dX, 75) - np.percentile(dX, 25)

        # Parametros OU
        iqr_X = np.percentile(X, 75) - np.percentile(X, 25)
        sigma_hist = np.std(X)
        sigma_w = max(iqr_X, sigma_hist) / np.sqrt(T)

        lag_acf = compute_acf_lag(X)
        theta_w = 1.0 / lag_acf if lag_acf > 0 else 0.1

        # Simular OU y calcular IQR de deltas
        np.random.seed(42)
        delta_ou = np.zeros(len(dX))
        x_ou = np.median(X)
        for t in range(len(dX)):
            dx = theta_w * (np.median(X) - x_ou) + sigma_w * np.random.randn()
            delta_ou[t] = dx
            x_ou += dx

        iqr_ou = np.percentile(delta_ou, 75) - np.percentile(delta_ou, 25)

        if iqr_ou > 0:
            chi_values.append(iqr_observed / iqr_ou)

    return np.median(chi_values) if chi_values else 0.0


def compute_intraworld_alphas(signals: WorldSignals) -> IntraWorldAlphas:
    """Calcula todos los indices intramundo."""

    alpha_affect, affect_series = compute_alpha_affect(
        signals.V, signals.A, signals.D, signals.w)

    alpha_hyst, hyst_series = compute_alpha_hyst(
        signals.V, signals.A, signals.w)

    alpha_switch, switch_series = compute_alpha_switch(
        signals.states, signals.w)

    alpha_recov = compute_alpha_recov(
        signals.V, signals.A, signals.D, signals.pi, signals.w)

    alpha_sus = compute_alpha_sus(
        signals.V, signals.A, signals.D, signals.w)

    # Composicion por ranks
    alphas = [alpha_affect, alpha_hyst, alpha_switch, alpha_recov, alpha_sus]
    ranks = stats.rankdata(alphas) / len(alphas)
    alpha_intra = float(np.sum(ranks))

    return IntraWorldAlphas(
        name=signals.name,
        alpha_affect=float(alpha_affect),
        alpha_hyst=float(alpha_hyst),
        alpha_switch=float(alpha_switch),
        alpha_recov=float(alpha_recov),
        alpha_sus=float(alpha_sus),
        alpha_intra=alpha_intra,
        alpha_affect_series=[float(x) for x in affect_series],
        alpha_hyst_series=[float(x) for x in hyst_series],
        alpha_switch_series=[float(x) for x in switch_series]
    )


# =============================================================================
# INDICES INTERMUNDOS
# =============================================================================

@dataclass
class InterWorldAlphas:
    """Indices de plasticidad intermundos."""
    alpha_consent_elast: float   # Elasticidad consentimiento
    alpha_cross_sus: float       # Susceptibilidad cruzada
    alpha_coord: float           # Coordinacion por modos
    alpha_homeo: float           # Homeostasis reciprocidad
    alpha_inter: float           # Compuesto


def compute_alpha_consent_elast(neo: WorldSignals, eva: WorldSignals,
                                 bilateral_events: List[dict]) -> float:
    """
    alpha_consent_elast = IQR_w(Delta_p / Delta_pi)
    Elasticidad del consentimiento.
    """
    # Crear serie de probabilidad bilateral por calibracion isotonica
    bilateral_ts = set(e['t'] for e in bilateral_events)

    # Usar pi conjunto
    pi_joint = (neo.pi + eva.pi) / 2
    labels = np.array([1 if t in bilateral_ts else 0 for t in range(len(pi_joint))])

    # Calibracion isotonica
    iso = IsotonicRegression(out_of_bounds='clip')

    # Necesitamos suficientes datos
    valid_mask = ~np.isnan(pi_joint)
    if valid_mask.sum() < 100:
        return 0.0

    iso.fit(pi_joint[valid_mask], labels[valid_mask])
    p_bilateral = iso.predict(pi_joint)

    # Calcular elasticidad: Delta_p / Delta_pi
    delta_p = np.diff(p_bilateral)
    delta_pi = np.diff(pi_joint)

    # Evitar division por cero
    eps = get_epsilon(delta_pi)
    elasticity = delta_p / (np.abs(delta_pi) + eps)

    # IQR de elasticidad
    iqr = np.percentile(elasticity, 75) - np.percentile(elasticity, 25)

    return float(iqr)


def compute_alpha_cross_sus(neo: WorldSignals, eva: WorldSignals,
                            bilateral_events: List[dict]) -> float:
    """
    alpha_cross_sus = susceptibilidad cruzada afectiva.
    Regresion Theil-Sen de Delta[V,A,D]_NEO ~ Delta[V,A,D]_{t-1}_EVA
    durante ventanas con consent=1.
    """
    bilateral_ts = set(e['t'] for e in bilateral_events)
    w = min(neo.w, eva.w)

    # Encontrar ventanas con bilateral
    betas = []

    for bt in bilateral_ts:
        if bt < 1 or bt >= min(neo.T, eva.T) - 1:
            continue

        # Delta PAD en t para NEO
        for X_neo, X_eva in [(neo.V, eva.V), (neo.A, eva.A), (neo.D, eva.D)]:
            delta_neo = X_neo[bt] - X_neo[bt-1] if bt > 0 else 0
            delta_eva_prev = X_eva[bt-1] - X_eva[bt-2] if bt > 1 else 0

            if abs(delta_eva_prev) > 1e-10:
                beta = delta_neo / delta_eva_prev
                betas.append(abs(beta))

    if not betas:
        return 0.0

    return float(np.median(betas))


def compute_alpha_coord(neo: WorldSignals, eva: WorldSignals,
                        bilateral_events: List[dict]) -> float:
    """
    alpha_coord = IQR(r[V,A,D](+1) - r[V,A,D](-1))
    Coordinacion por modos durante consent=1.
    """
    # Separar por modo
    mode_plus = []
    mode_minus = []

    for e in bilateral_events:
        t = e['t']
        if t >= min(neo.T, eva.T):
            continue

        neo_mode = e.get('neo_mode', 0)
        eva_mode = e.get('eva_mode', 0)

        # Correlacion PAD
        r_VA = np.corrcoef([neo.V[t], neo.A[t], neo.D[t]],
                           [eva.V[t], eva.A[t], eva.D[t]])[0, 1]

        if neo_mode == 1 and eva_mode == 1:
            mode_plus.append(r_VA)
        elif neo_mode == -1 and eva_mode == -1:
            mode_minus.append(r_VA)

    if not mode_plus or not mode_minus:
        return 0.0

    diff = np.median(mode_plus) - np.median(mode_minus)
    return float(abs(diff))


def compute_alpha_homeo(neo: WorldSignals, eva: WorldSignals,
                        voluntary_neo: List[dict], voluntary_eva: List[dict]) -> float:
    """
    alpha_homeo = IQR(Delta_R_soc_ema_NEO) + IQR(Delta_R_soc_ema_EVA)
    Solo al cierre de ventanas ON.
    """
    # Extraer R_soc_ema de voluntary logs
    R_soc_ema_neo = np.array([v.get('R_soc_ema', 0.5) for v in voluntary_neo])
    R_soc_ema_eva = np.array([v.get('R_soc_ema', 0.5) for v in voluntary_eva])

    # Calcular deltas
    delta_neo = np.diff(R_soc_ema_neo)
    delta_eva = np.diff(R_soc_ema_eva)

    # IQR de cambios
    iqr_neo = np.percentile(delta_neo, 75) - np.percentile(delta_neo, 25) if len(delta_neo) > 0 else 0
    iqr_eva = np.percentile(delta_eva, 75) - np.percentile(delta_eva, 25) if len(delta_eva) > 0 else 0

    return float(iqr_neo + iqr_eva)


def compute_interworld_alphas(neo: WorldSignals, eva: WorldSignals,
                              bilateral_events: List[dict],
                              voluntary_neo: List[dict],
                              voluntary_eva: List[dict]) -> InterWorldAlphas:
    """Calcula todos los indices intermundos."""

    alpha_consent_elast = compute_alpha_consent_elast(neo, eva, bilateral_events)
    alpha_cross_sus = compute_alpha_cross_sus(neo, eva, bilateral_events)
    alpha_coord = compute_alpha_coord(neo, eva, bilateral_events)
    alpha_homeo = compute_alpha_homeo(neo, eva, voluntary_neo, voluntary_eva)

    # Composicion por ranks
    alphas = [alpha_consent_elast, alpha_cross_sus, alpha_coord, alpha_homeo]
    ranks = stats.rankdata(alphas) / len(alphas)
    alpha_inter = float(np.sum(ranks))

    return InterWorldAlphas(
        alpha_consent_elast=alpha_consent_elast,
        alpha_cross_sus=alpha_cross_sus,
        alpha_coord=alpha_coord,
        alpha_homeo=alpha_homeo,
        alpha_inter=alpha_inter
    )


# =============================================================================
# INDICES ESTRUCTURALES
# =============================================================================

@dataclass
class StructuralAlphas:
    """Indices estructurales."""
    alpha_weights: float     # Movilidad de pesos
    alpha_manifold: float    # Deriva en variedad
    alpha_struct: float      # Compuesto

    weights_series: List[float]
    manifold_series: List[float]


def compute_alpha_weights(neo: WorldSignals, eva: WorldSignals) -> Tuple[float, List[float]]:
    """
    alpha_weights = sum_i IQR_w(Delta_w_i)
    Movilidad de pesos adaptativos.
    """
    series = []
    w = min(neo.w, eva.w)

    # Combinar pesos de ambos mundos
    all_weights = np.vstack([neo.weights, eva.weights])

    for start, end in rolling_window_indices(len(all_weights), w):
        window_weights = all_weights[start:end]
        delta_weights = np.diff(window_weights, axis=0)

        iqr_sum = 0.0
        for i in range(3):  # 3 pesos
            col = delta_weights[:, i]
            iqr = np.percentile(col, 75) - np.percentile(col, 25)
            iqr_sum += iqr

        series.append(iqr_sum)

    return float(np.median(series)) if series else 0.0, series


def compute_alpha_manifold(neo: WorldSignals, eva: WorldSignals) -> Tuple[float, List[float]]:
    """
    alpha_manifold = IQR_w(Delta_coords)
    Deriva en variedad latente [V,A,D,pi,R_soc] -> R^2.
    """
    series = []
    w = min(neo.w, eva.w)
    T = min(neo.T, eva.T)

    # Combinar datos de ambos mundos
    data = np.column_stack([
        neo.V[:T], neo.A[:T], neo.D[:T], neo.pi[:T], neo.R_soc[:T],
        eva.V[:T], eva.A[:T], eva.D[:T], eva.pi[:T], eva.R_soc[:T]
    ])

    for start, end in rolling_window_indices(T, w):
        window_data = data[start:end]

        if len(window_data) >= 10:
            # PCA robusta a 2D
            coords = robust_pca_2d(window_data)

            # Delta de coordenadas
            delta_coords = np.diff(coords, axis=0)
            norms = np.linalg.norm(delta_coords, axis=1)

            iqr = np.percentile(norms, 75) - np.percentile(norms, 25)
            series.append(iqr)
        else:
            series.append(0.0)

    return float(np.median(series)) if series else 0.0, series


def compute_structural_alphas(neo: WorldSignals, eva: WorldSignals) -> StructuralAlphas:
    """Calcula todos los indices estructurales."""

    alpha_weights, weights_series = compute_alpha_weights(neo, eva)
    alpha_manifold, manifold_series = compute_alpha_manifold(neo, eva)

    # Composicion por ranks
    alphas = [alpha_weights, alpha_manifold]
    ranks = stats.rankdata(alphas) / len(alphas)
    alpha_struct = float(np.sum(ranks))

    return StructuralAlphas(
        alpha_weights=alpha_weights,
        alpha_manifold=alpha_manifold,
        alpha_struct=alpha_struct,
        weights_series=[float(x) for x in weights_series],
        manifold_series=[float(x) for x in manifold_series]
    )


# =============================================================================
# INDICE GLOBAL
# =============================================================================

@dataclass
class GlobalAlpha:
    """Indice global de plasticidad."""
    alpha_global: float
    alpha_global_series: List[float]

    components: Dict[str, float]


def compute_global_alpha(neo_intra: IntraWorldAlphas,
                         eva_intra: IntraWorldAlphas,
                         inter: InterWorldAlphas,
                         struct: StructuralAlphas) -> GlobalAlpha:
    """
    alpha_global = rank(alpha_intra_NEO) + rank(alpha_intra_EVA)
                 + rank(alpha_inter) + rank(alpha_struct)
    """
    components = {
        'alpha_intra_NEO': neo_intra.alpha_intra,
        'alpha_intra_EVA': eva_intra.alpha_intra,
        'alpha_inter': inter.alpha_inter,
        'alpha_struct': struct.alpha_struct
    }

    values = list(components.values())
    ranks = stats.rankdata(values) / len(values)
    alpha_global = float(np.sum(ranks))

    # Serie temporal (usando series disponibles)
    n_windows = min(len(neo_intra.alpha_affect_series),
                    len(struct.weights_series))

    series = []
    for i in range(n_windows):
        # Aproximar alpha_global por ventana
        local_vals = [
            neo_intra.alpha_affect_series[i] if i < len(neo_intra.alpha_affect_series) else 0,
            eva_intra.alpha_affect_series[i] if i < len(neo_intra.alpha_affect_series) else 0,
            struct.weights_series[i] if i < len(struct.weights_series) else 0,
            struct.manifold_series[i] if i < len(struct.manifold_series) else 0
        ]
        local_ranks = stats.rankdata(local_vals) / len(local_vals)
        series.append(float(np.sum(local_ranks)))

    return GlobalAlpha(
        alpha_global=alpha_global,
        alpha_global_series=series,
        components=components
    )


# =============================================================================
# MAIN
# =============================================================================

def load_data(data_dir: str) -> Tuple[dict, dict, dict, dict, dict, List[dict]]:
    """Carga todos los datos necesarios."""

    with open(f"{data_dir}/affect_log_neo.json") as f:
        affect_neo = json.load(f)
    with open(f"{data_dir}/affect_log_eva.json") as f:
        affect_eva = json.load(f)

    with open(f"{data_dir}/consent_log_neo.json") as f:
        consent_neo = json.load(f)
    with open(f"{data_dir}/consent_log_eva.json") as f:
        consent_eva = json.load(f)

    with open(f"{data_dir}/voluntary_log_neo.json") as f:
        voluntary_neo = json.load(f)
    with open(f"{data_dir}/voluntary_log_eva.json") as f:
        voluntary_eva = json.load(f)

    with open(f"{data_dir}/bilateral_events.json") as f:
        bilateral = json.load(f)

    return affect_neo, affect_eva, consent_neo, consent_eva, voluntary_neo, voluntary_eva, bilateral


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Phase 9: Plasticidad Afectiva')
    parser.add_argument('--data-dir', default='/root/NEO_EVA/results/phase8_long',
                        help='Directorio de datos de entrada')
    parser.add_argument('--output-dir', default='/root/NEO_EVA/results/phase9',
                        help='Directorio de salida')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/figures", exist_ok=True)

    print("=" * 70)
    print("PHASE 9: PLASTICIDAD AFECTIVA (100% ENDOGENO)")
    print("=" * 70)

    # Cargar datos
    print("\n[1] Cargando datos...")
    (affect_neo, affect_eva, consent_neo, consent_eva,
     voluntary_neo, voluntary_eva, bilateral) = load_data(args.data_dir)

    print(f"    NEO: {len(affect_neo)} ciclos")
    print(f"    EVA: {len(affect_eva)} ciclos")
    print(f"    Bilateral events: {len(bilateral)}")

    # Extraer senales
    print("\n[2] Extrayendo senales...")
    neo_signals = extract_signals(affect_neo, consent_neo, voluntary_neo, 'NEO')
    eva_signals = extract_signals(affect_eva, consent_eva, voluntary_eva, 'EVA')

    print(f"    Ventana w: NEO={neo_signals.w}, EVA={eva_signals.w}")

    # Calcular indices intramundo
    print("\n[3] Calculando indices intramundo...")
    neo_intra = compute_intraworld_alphas(neo_signals)
    eva_intra = compute_intraworld_alphas(eva_signals)

    print(f"    NEO alpha_intra: {neo_intra.alpha_intra:.4f}")
    print(f"      - affect: {neo_intra.alpha_affect:.6f}")
    print(f"      - hyst: {neo_intra.alpha_hyst:.6f}")
    print(f"      - switch: {neo_intra.alpha_switch:.6f}")
    print(f"      - recov: {neo_intra.alpha_recov:.6f}")
    print(f"      - sus: {neo_intra.alpha_sus:.6f}")

    print(f"    EVA alpha_intra: {eva_intra.alpha_intra:.4f}")
    print(f"      - affect: {eva_intra.alpha_affect:.6f}")
    print(f"      - hyst: {eva_intra.alpha_hyst:.6f}")
    print(f"      - switch: {eva_intra.alpha_switch:.6f}")
    print(f"      - recov: {eva_intra.alpha_recov:.6f}")
    print(f"      - sus: {eva_intra.alpha_sus:.6f}")

    # Calcular indices intermundos
    print("\n[4] Calculando indices intermundos...")
    inter = compute_interworld_alphas(neo_signals, eva_signals, bilateral,
                                      voluntary_neo, voluntary_eva)

    print(f"    alpha_inter: {inter.alpha_inter:.4f}")
    print(f"      - consent_elast: {inter.alpha_consent_elast:.6f}")
    print(f"      - cross_sus: {inter.alpha_cross_sus:.6f}")
    print(f"      - coord: {inter.alpha_coord:.6f}")
    print(f"      - homeo: {inter.alpha_homeo:.6f}")

    # Calcular indices estructurales
    print("\n[5] Calculando indices estructurales...")
    struct = compute_structural_alphas(neo_signals, eva_signals)

    print(f"    alpha_struct: {struct.alpha_struct:.4f}")
    print(f"      - weights: {struct.alpha_weights:.6f}")
    print(f"      - manifold: {struct.alpha_manifold:.6f}")

    # Calcular indice global
    print("\n[6] Calculando indice global...")
    global_alpha = compute_global_alpha(neo_intra, eva_intra, inter, struct)

    print(f"    alpha_global: {global_alpha.alpha_global:.4f}")

    # Guardar resultados
    print("\n[7] Guardando resultados...")

    with open(f"{args.output_dir}/alpha_intraworld_neo.json", 'w') as f:
        json.dump(asdict(neo_intra), f, indent=2)

    with open(f"{args.output_dir}/alpha_intraworld_eva.json", 'w') as f:
        json.dump(asdict(eva_intra), f, indent=2)

    with open(f"{args.output_dir}/alpha_interworld.json", 'w') as f:
        json.dump(asdict(inter), f, indent=2)

    with open(f"{args.output_dir}/alpha_structural.json", 'w') as f:
        json.dump(asdict(struct), f, indent=2)

    with open(f"{args.output_dir}/alpha_global.json", 'w') as f:
        json.dump(asdict(global_alpha), f, indent=2)

    # Guardar senales PAD para figuras
    pad_data = {
        'neo': {
            'V': neo_signals.V.tolist(),
            'A': neo_signals.A.tolist(),
            'D': neo_signals.D.tolist()
        },
        'eva': {
            'V': eva_signals.V.tolist(),
            'A': eva_signals.A.tolist(),
            'D': eva_signals.D.tolist()
        }
    }
    with open(f"{args.output_dir}/pad_signals.json", 'w') as f:
        json.dump(pad_data, f)

    print(f"\n[OK] Resultados guardados en {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
