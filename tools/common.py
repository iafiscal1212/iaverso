#!/usr/bin/env python3
"""
NEO_EVA Common Utilities - 100% Endógeno
=========================================
Todas las funciones derivan parámetros de datos históricos.
Sin constantes hardcodeadas excepto priors de máxima entropía (1/3,1/3,1/3) cuando T=0.
"""
import os
import json
import math
import hashlib
import statistics as st
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

# ============================================================================
# PROYECCIÓN AL SIMPLEX
# ============================================================================

def proj_simplex(S: float, N: float, C: float) -> Tuple[float, float, float]:
    """
    Proyecta (S,N,C) al simplex: S+N+C=1, todos ≥0.
    Prior de máxima entropía si degenera: (1/3, 1/3, 1/3).
    """
    S = max(0.0, S)
    N = max(0.0, N)
    C = max(0.0, C)
    total = S + N + C
    if total <= 0:
        return (1/3, 1/3, 1/3)  # Único prior no informativo defendible
    return (S / total, N / total, C / total)

# ============================================================================
# CARGA DE HISTÓRICOS
# ============================================================================

def load_hist(path: str) -> List[Dict]:
    """Carga histórico JSONL."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records

def append_hist(path: str, record: Dict) -> None:
    """Añade registro al histórico JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')

def last_I(hist: List[Dict]) -> Tuple[float, float, float]:
    """Obtiene última intención del histórico, o prior si vacío."""
    if not hist:
        return (1/3, 1/3, 1/3)
    r = hist[-1].get("I", {"S": 1/3, "N": 1/3, "C": 1/3})
    return (float(r["S"]), float(r["N"]), float(r["C"]))

def triplet(hist: List[Dict], key: str) -> List[float]:
    """Extrae serie de una componente de I."""
    return [h["I"][key] for h in hist if "I" in h and key in h["I"]]

# ============================================================================
# ESTADÍSTICAS ENDÓGENAS
# ============================================================================

def sigmas(hist: List[Dict]) -> Tuple[float, float, float]:
    """Desviaciones estándar de S, N, C desde histórico."""
    def sd(xs):
        if len(xs) < 2:
            return 0.0
        return float(np.std(xs, ddof=0))  # Población, no muestra
    S = sd(triplet(hist, "S"))
    N = sd(triplet(hist, "N"))
    C = sd(triplet(hist, "C"))
    return (S, N, C)

def means(hist: List[Dict]) -> Tuple[float, float, float]:
    """Medias de S, N, C desde histórico."""
    def mu(xs):
        if len(xs) == 0:
            return 1/3
        return float(np.mean(xs))
    S = mu(triplet(hist, "S"))
    N = mu(triplet(hist, "N"))
    C = mu(triplet(hist, "C"))
    return (S, N, C)

def cov3(hist: List[Dict]) -> np.ndarray:
    """Matriz de covarianza 3x3 desde histórico."""
    if len(hist) < 2:
        return np.zeros((3, 3))
    S = triplet(hist, "S")
    N = triplet(hist, "N")
    C = triplet(hist, "C")
    X = np.array([S, N, C], dtype=float)
    return np.cov(X)

def iqr(xs: List[float]) -> float:
    """Rango intercuartílico."""
    if len(xs) < 4:
        return float(np.std(xs)) if xs else 0.0
    return float(np.percentile(xs, 75) - np.percentile(xs, 25))

def cv(xs: List[float]) -> float:
    """Coeficiente de variación = std/mean."""
    if len(xs) < 2:
        return 0.0
    mu = np.mean(xs)
    if abs(mu) < 1e-12:
        return 0.0
    return float(np.std(xs) / abs(mu))

def pctl(xs: List[float], p: float) -> float:
    """Percentil p (0-1) de xs."""
    if len(xs) == 0:
        return 0.0
    return float(np.percentile(xs, p * 100))

def quantiles_dict(xs: List[float]) -> Dict[str, float]:
    """Diccionario con percentiles estándar."""
    if len(xs) == 0:
        return {"p25": 0, "p50": 0, "p75": 0, "p90": 0, "p95": 0, "p99": 0}
    return {
        "p25": float(np.percentile(xs, 25)),
        "p50": float(np.percentile(xs, 50)),
        "p75": float(np.percentile(xs, 75)),
        "p90": float(np.percentile(xs, 90)),
        "p95": float(np.percentile(xs, 95)),
        "p99": float(np.percentile(xs, 99))
    }

# ============================================================================
# ACF Y VENTANA ENDÓGENA
# ============================================================================

def acf(series: List[float], maxlag: Optional[int] = None) -> List[float]:
    """
    Autocorrelación desde lag 0 hasta maxlag.
    maxlag = max(3, floor(sqrt(T))) si no se especifica.
    """
    T = len(series)
    if T < 3:
        return [1.0]
    if maxlag is None:
        maxlag = max(3, int(math.sqrt(T)))
    maxlag = min(maxlag, T // 2)

    x = np.array(series, dtype=float)
    mu = np.mean(x)
    var = np.var(x)
    if var < 1e-12:
        return [1.0] * (maxlag + 1)

    acf_vals = [1.0]
    for lag in range(1, maxlag + 1):
        cov = np.mean((x[lag:] - mu) * (x[:-lag] - mu))
        acf_vals.append(float(cov / var))
    return acf_vals

def acf_window(series: List[float]) -> int:
    """
    Ventana endógena = primer lag donde ACF cae bajo mediana de envolvente
    o primer cruce a cero.
    Escala con T, no constante fija.
    """
    T = len(series)
    if T < 8:
        return max(3, int(math.log1p(T) + 1))  # Cold start

    maxlag = max(3, int(math.sqrt(T)))
    acf_vals = acf(series, maxlag)

    # Envolvente = |acf|
    envelope = [abs(r) for r in acf_vals[1:]]
    if not envelope:
        return max(3, int(math.log1p(T) + 1))

    med_env = float(np.median(envelope))
    cross_zero = None

    for lag in range(1, len(acf_vals)):
        if acf_vals[lag] <= 0 and cross_zero is None:
            cross_zero = lag
        if abs(acf_vals[lag]) < med_env:
            return lag

    return cross_zero or max(3, int(math.log1p(T) + 1))

# ============================================================================
# PCA ENDÓGENO
# ============================================================================

def pca_full(COV: np.ndarray) -> Tuple[List[float], List[List[float]], List[float]]:
    """
    PCA completo: eigenvalues, eigenvectors, varianza explicada.
    Retorna (lambdas, [v1, v2, v3], varexp_ratios).
    """
    if COV.shape != (3, 3):
        return ([0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [1/3, 1/3, 1/3])

    vals, vecs = np.linalg.eigh(COV)
    # Ordenar por eigenvalue descendente
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    # Normalizar vectores
    V = []
    for i in range(3):
        v = vecs[:, i]
        n = np.linalg.norm(v)
        if n > 1e-12:
            v = v / n
        V.append(v.tolist())

    # Varianza explicada
    total = float(np.sum(np.abs(vals)))
    if total < 1e-12:
        varexp = [1/3, 1/3, 1/3]
    else:
        varexp = [float(abs(v)) / total for v in vals]

    return (vals.tolist(), V, varexp)

def pca_v1(hist: List[Dict]) -> Tuple[List[float], float, float]:
    """
    Vector principal v1 desde histórico.
    Retorna (v1, lambda1, var_explicada_1).
    """
    COV = cov3(hist)
    lambdas, vecs, varexp = pca_full(COV)
    return (vecs[0], lambdas[0], varexp[0])

# ============================================================================
# ALPHA ENDÓGENO
# ============================================================================

def median_alpha(sigs: Tuple[float, float, float], T: int) -> float:
    """
    Alpha endógeno = median(σ_S, σ_N, σ_C) / sqrt(T).
    Escala con T, sin constantes.
    """
    med = st.median(list(sigs))
    return med / max(1.0, math.sqrt(max(1, T)))

def percentile_alpha(sigs: Tuple[float, float, float], T: int, p: float) -> float:
    """Alpha por percentil de sigmas."""
    val = pctl(list(sigs), p)
    return val / max(1.0, math.sqrt(max(1, T)))

# ============================================================================
# k ENDÓGENO PARA RECALL
# ============================================================================

def k_endogenous(T: int) -> int:
    """
    k = max(3, floor(log(T))).
    Número de episodios/vecinos para recall.
    """
    if T < 3:
        return 3
    return max(3, int(math.log(T)))

# ============================================================================
# INFORMACIÓN MUTUA (kNN estimator)
# ============================================================================

def mi_knn(x: np.ndarray, y: np.ndarray, k: Optional[int] = None) -> float:
    """
    Estimador kNN de información mutua.
    k = floor(T^{1/3}) si no se especifica.
    """
    from scipy.spatial import KDTree

    n = len(x)
    if n < 4:
        return 0.0

    if k is None:
        k = max(1, int(n ** (1/3)))
    k = min(k, n - 1)

    x = np.array(x).reshape(-1, 1) if x.ndim == 1 else x
    y = np.array(y).reshape(-1, 1) if y.ndim == 1 else y
    xy = np.hstack([x, y])

    try:
        tree_xy = KDTree(xy)
        tree_x = KDTree(x)
        tree_y = KDTree(y)

        mi = 0.0
        for i in range(n):
            # Distancia al k-ésimo vecino en espacio conjunto
            d_xy, _ = tree_xy.query(xy[i], k + 1)
            eps = d_xy[-1]
            if eps < 1e-12:
                continue

            # Contar vecinos dentro de eps en espacios marginales
            nx = len(tree_x.query_ball_point(x[i], eps)) - 1
            ny = len(tree_y.query_ball_point(y[i], eps)) - 1

            from scipy.special import digamma
            mi += digamma(k) - digamma(max(1, nx)) - digamma(max(1, ny)) + digamma(n)

        return max(0.0, mi / n)
    except Exception:
        return 0.0

# ============================================================================
# TRANSFER ENTROPY
# ============================================================================

def transfer_entropy(source: List[float], target: List[float],
                     lag: int = 1, k: Optional[int] = None) -> float:
    """
    Transfer Entropy de source a target.
    TE(S→T) = I(T_t+1 ; S_t | T_t)
    """
    n = len(source)
    if n < lag + 3:
        return 0.0

    # Construir vectores
    T_past = np.array(target[:-lag]).reshape(-1, 1)
    T_future = np.array(target[lag:]).reshape(-1, 1)
    S_past = np.array(source[:-lag]).reshape(-1, 1)

    # TE = I(T_future; S_past | T_past)
    # Aproximamos como: I(T_future; [S_past, T_past]) - I(T_future; T_past)
    joint = np.hstack([S_past, T_past])

    mi_joint = mi_knn(T_future.flatten(), joint, k)
    mi_cond = mi_knn(T_future.flatten(), T_past.flatten(), k)

    return max(0.0, mi_joint - mi_cond)

# ============================================================================
# CHECKSUMS
# ============================================================================

def sha256_file(path: str) -> str:
    """SHA256 de archivo."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def sha256_str(s: str) -> str:
    """SHA256 de string."""
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

# ============================================================================
# SELECCIÓN DE MODELO (BIC/AIC)
# ============================================================================

def ar_bic(series: List[float], max_p: Optional[int] = None) -> Tuple[int, float]:
    """
    Selecciona orden AR por BIC.
    max_p = floor(log(T)) si no se especifica.
    Retorna (p_óptimo, BIC_óptimo).
    """
    n = len(series)
    if n < 5:
        return (1, float('inf'))

    if max_p is None:
        max_p = max(1, int(math.log(n)))
    max_p = min(max_p, n // 3)

    x = np.array(series, dtype=float)
    best_p = 1
    best_bic = float('inf')

    for p in range(1, max_p + 1):
        # Construir matrices para regresión
        X = np.column_stack([x[p-i-1:n-i-1] for i in range(p)])
        y = x[p:]

        if len(y) < p + 1:
            continue

        # OLS
        try:
            beta, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
            if len(residuals) == 0:
                residuals = np.sum((y - X @ beta) ** 2)
            else:
                residuals = residuals[0]

            n_eff = len(y)
            sigma2 = residuals / n_eff
            if sigma2 < 1e-12:
                sigma2 = 1e-12

            # BIC = n*log(sigma2) + p*log(n)
            bic = n_eff * math.log(sigma2) + p * math.log(n_eff)

            if bic < best_bic:
                best_bic = bic
                best_p = p
        except Exception:
            continue

    return (best_p, best_bic)

# ============================================================================
# NULOS POR PERMUTACIÓN
# ============================================================================

def null_permutation(metric_func, x: np.ndarray, y: np.ndarray,
                     B: Optional[int] = None) -> Tuple[float, List[float], float]:
    """
    Test de permutación para métricas bivariadas.
    B = floor(10*sqrt(T)) si no se especifica.
    Retorna (valor_observado, distribución_nula, p_hat).
    """
    n = len(x)
    if B is None:
        B = max(10, int(10 * math.sqrt(n)))

    obs = metric_func(x, y)
    null_dist = []

    for _ in range(B):
        y_perm = np.random.permutation(y)
        null_dist.append(metric_func(x, y_perm))

    # p_hat = fracción de nulos >= observado
    p_hat = sum(1 for v in null_dist if v >= obs) / max(1, B)

    return (obs, null_dist, p_hat)

# ============================================================================
# ALEATORIZACIÓN DE FASE (preserva espectro)
# ============================================================================

def phase_randomize(x: np.ndarray) -> np.ndarray:
    """
    Aleatoriza fase preservando magnitud espectral.
    Genera surrogate con misma estructura de autocorrelación.
    """
    X = np.fft.rfft(x)
    mag = np.abs(X)
    ph = np.angle(X)

    # Fases aleatorias (preservar simetría)
    rnd = np.random.uniform(-np.pi, np.pi, size=ph.shape)
    rnd[0] = ph[0]  # DC component
    if len(ph) > 1:
        rnd[-1] = ph[-1]  # Nyquist

    Y = mag * np.exp(1j * rnd)
    y = np.fft.irfft(Y, n=len(x))
    return np.real(y)
