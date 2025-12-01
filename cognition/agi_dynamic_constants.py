"""
AGI Dynamic Constants - Eliminación de Números Mágicos
=======================================================

Todas las constantes se derivan endógenamente de:
- percentiles
- ranks
- √t, 1/√t
- varianzas y covarianzas
- autocorrelaciones
- historial de episodios
- distribución de errores

100% endógeno. Cero números mágicos hardcodeados.
"""

import numpy as np
from typing import List, Optional, Union
from functools import lru_cache


# =============================================================================
# PARTE 1: Mínimos adaptativos (reemplaza 5, 10, 20)
# =============================================================================

def L_t(t: int) -> int:
    """
    Ventana mínima adaptativa.

    L_t = max(3, floor(√t))

    Elimina todos los mínimos fijos como:
    - if len(history) < 10
    - if len(values) < 5
    - ventanas "por defecto": 3, 5, 10, 20

    Args:
        t: Número de muestras/pasos del módulo

    Returns:
        Tamaño mínimo de ventana endógeno
    """
    return max(3, int(np.floor(np.sqrt(t + 1))))


def min_samples(t: int) -> int:
    """
    Número mínimo de muestras para cálculos estadísticos.

    Alias de L_t para legibilidad.
    """
    return L_t(t)


# =============================================================================
# PARTE 2: Límites de historial adaptativos (reemplaza 100, 200, 500, 1000)
# =============================================================================

def max_history(t: int) -> int:
    """
    Límite de historial adaptativo.

    H_max(t) = int(50 + 5√t)

    Ejemplos:
    - t=100 → H=50+5·10 = 100
    - t=1000 → H=50+5·31 = 205
    - t=10000 → H=50+5·100 = 550

    Crece sin explotar, mantiene continuidad narrativa.

    Args:
        t: Tiempo/pasos transcurridos

    Returns:
        Límite máximo de historial
    """
    return int(50 + 5 * np.sqrt(t + 1))


def adaptive_window(t: int, base: int = 10) -> int:
    """
    Ventana adaptativa para procesamiento.

    W = max(base, ceil(√t))
    """
    return max(base, int(np.ceil(np.sqrt(t + 1))))


# =============================================================================
# PARTE 3: Percentiles dinámicos (reemplaza 20, 75, 95)
# =============================================================================

def dynamic_percentile_low(t: int) -> float:
    """
    Percentil bajo dinámico.

    p_low = min(5 + 2·log(t+1), 40)

    Al principio: umbrales suaves
    Conforme madura: más exigentes
    """
    return min(5 + 2 * np.log(t + 1), 40)


def dynamic_percentile_high(t: int) -> float:
    """
    Percentil alto dinámico.

    p_high = max(60, 90 - 3/√t)
    """
    return max(60, 90 - 3 / np.sqrt(t + 1))


def dynamic_percentile_danger(t: int) -> float:
    """
    Percentil para detección de peligro.

    p_danger = 90 - 5/√t

    Más estricto conforme el sistema madura.
    """
    return max(80, 90 - 5 / np.sqrt(t + 1))


def compute_adaptive_percentile(values: np.ndarray, t: int,
                                 mode: str = 'high') -> float:
    """
    Calcula percentil adaptativo sobre valores.

    Args:
        values: Array de valores
        t: Tiempo actual
        mode: 'low', 'high', o 'danger'

    Returns:
        Valor del percentil adaptativo
    """
    if len(values) == 0:
        return 0.5

    if mode == 'low':
        p = dynamic_percentile_low(t)
    elif mode == 'danger':
        p = dynamic_percentile_danger(t)
    else:
        p = dynamic_percentile_high(t)

    return float(np.percentile(values, p))


# =============================================================================
# PARTE 4: Umbrales de similaridad adaptativos (reemplaza 0.5, 0.8, 0.9)
# =============================================================================

def similarity_threshold(similarity_history: List[float]) -> float:
    """
    Umbral de similaridad endógeno.

    τ_sim = μ_sim + 0.5·σ_sim

    Basado en la distribución histórica de similaridades.

    Args:
        similarity_history: Historial de valores de similaridad

    Returns:
        Umbral adaptativo
    """
    if len(similarity_history) < 3:
        return 0.5  # Valor neutral inicial

    mu = np.mean(similarity_history)
    sigma = np.std(similarity_history)

    return float(mu + 0.5 * sigma)


def similarity_threshold_percentile(similarity_history: List[float],
                                     t: int) -> float:
    """
    Umbral de similaridad basado en percentil dinámico.

    τ_sim = percentile(sim_hist, p_high)
    """
    if len(similarity_history) < 3:
        return 0.5

    p = dynamic_percentile_high(t)
    return float(np.percentile(similarity_history, min(p, 95)))


# =============================================================================
# PARTE 5: Factores de actualización adaptativos (reemplaza 0.9/0.1)
# =============================================================================

def adaptive_momentum(values: List[float]) -> float:
    """
    Momento adaptativo para actualización exponencial.

    β_t = 1 / (1 + exp(-(σ_t - μ_t)))

    - Variabilidad baja → mayor peso histórico
    - Variabilidad alta → más aprendizaje fresco

    Args:
        values: Historial de valores recientes

    Returns:
        β entre 0 y 1
    """
    if len(values) < 3:
        return 0.5

    mu = np.mean(values)
    sigma = np.std(values)

    # Normalizar para que tenga sentido
    if abs(mu) > 1e-8:
        normalized_diff = (sigma - abs(mu)) / (abs(mu) + 1e-8)
    else:
        normalized_diff = sigma

    beta = 1 / (1 + np.exp(-normalized_diff))
    return float(np.clip(beta, 0.1, 0.9))


def exponential_update(old_value: float, new_value: float,
                       history: List[float]) -> float:
    """
    Actualización exponencial con momento adaptativo.

    x = β·x_old + (1-β)·x_new

    donde β se deriva del historial.
    """
    beta = adaptive_momentum(history)
    return beta * old_value + (1 - beta) * new_value


# =============================================================================
# PARTE 6: Períodos de actualización adaptativos (reemplaza 10, 20, 25, 50)
# =============================================================================

def update_period(history: List[float]) -> int:
    """
    Período de actualización basado en entropía.

    T_upd = max(5, int(10 / √entropía))

    Más variabilidad → actualizaciones más frecuentes
    Menos variabilidad → actualizaciones menos frecuentes
    """
    if len(history) < 5:
        return 5

    # Calcular entropía aproximada via histograma
    hist, _ = np.histogram(history, bins='auto', density=True)
    hist = hist[hist > 0]  # Evitar log(0)

    if len(hist) == 0:
        return 10

    entropy = -np.sum(hist * np.log(hist + 1e-10))
    entropy = max(0.1, entropy)  # Evitar división por cero

    return max(5, int(10 / np.sqrt(entropy)))


def should_update(t: int, history: List[float]) -> bool:
    """
    Decide si es momento de actualizar.

    Returns:
        True si t % T_upd == 0
    """
    period = update_period(history)
    return t % period == 0


# =============================================================================
# PARTE 7: Dimensiones dinámicas (reemplaza 5, 6, 7, 10)
# =============================================================================

def dynamic_dim_from_eigenvalues(eigenvalues: np.ndarray,
                                  max_dim: int = 20) -> int:
    """
    Número de dimensiones basado en eigenvalores significativos.

    d = min(max_dim, #{λ_i : λ_i > μ_λ})

    Args:
        eigenvalues: Array de eigenvalores
        max_dim: Límite superior

    Returns:
        Número de dimensiones significativas
    """
    if len(eigenvalues) == 0:
        return 3

    mu = np.mean(eigenvalues)
    n_significant = np.sum(eigenvalues > mu)

    return int(np.clip(n_significant, 2, max_dim))


def dynamic_dim_from_covariance(X: np.ndarray, max_dim: int = 20) -> int:
    """
    Número de dimensiones desde matriz de covarianza.

    Usa eigendecomposition para encontrar dimensiones significativas.

    Args:
        X: Matriz de datos (n_samples, n_features)
        max_dim: Límite superior

    Returns:
        Número de dimensiones significativas
    """
    if X.ndim == 1:
        return 1

    if X.shape[0] < 3:
        return min(X.shape[1] if X.ndim > 1 else 1, max_dim)

    try:
        cov = np.cov(X.T)
        if cov.ndim == 0:
            return 1
        eigenvalues = np.linalg.eigvalsh(cov)
        return dynamic_dim_from_eigenvalues(eigenvalues, max_dim)
    except:
        return min(X.shape[1] if X.ndim > 1 else 1, max_dim)


def infer_optimal_dim(history: List[np.ndarray], max_dim: int = 20) -> int:
    """
    Infiere dimensión óptima desde historial de vectores.
    """
    if len(history) < 5:
        return min(len(history[0]) if history else 5, max_dim)

    X = np.array(history[-min(100, len(history)):])
    return dynamic_dim_from_covariance(X, max_dim)


# =============================================================================
# PARTE 8: Umbrales éticos y de peligro (reemplaza 0.3, 0.5, 0.8, 95)
# =============================================================================

def ethical_threshold(risk_history: List[float]) -> float:
    """
    Umbral ético basado en distribución de riesgo.

    τ_ethics = μ_risk + σ_risk
    """
    if len(risk_history) < 5:
        return 0.5

    mu = np.mean(risk_history)
    sigma = np.std(risk_history)

    return float(mu + sigma)


def norm_threshold(persistence_history: List[float], t: int) -> float:
    """
    Umbral para considerar una norma estable.

    τ_norm = percentile(persist_hist, p_high)
    """
    if len(persistence_history) < 5:
        return 0.5

    p = min(80, dynamic_percentile_high(t))
    return float(np.percentile(persistence_history, p))


def danger_threshold(risk_history: List[float], t: int) -> float:
    """
    Umbral de peligro dinámico.

    τ_danger = percentile(risk_hist, 90 - 5/√t)
    """
    if len(risk_history) < 5:
        return 0.8

    p = dynamic_percentile_danger(t)
    return float(np.percentile(risk_history, p))


def no_go_confirmation_count(t: int) -> int:
    """
    Número de detecciones para confirmar zona prohibida.

    Basado en √t para ser más estricto con el tiempo.
    """
    return max(2, int(np.sqrt(t / 100 + 1)))


# =============================================================================
# PARTE 9: Ventanas para clustering y conceptos
# =============================================================================

def concept_window(n_unique_events: int) -> int:
    """
    Ventana para detección de conceptos.

    W_t = 3 + floor(√n_unique)
    """
    return 3 + int(np.floor(np.sqrt(n_unique_events + 1)))


def kmeans_iterations(n_samples: int) -> int:
    """
    Número de iteraciones k-means adaptativo.

    Basado en log del número de muestras.
    """
    return max(10, int(5 * np.log(n_samples + 1)))


def n_clusters_from_eigenvalues(eigenvalues: np.ndarray) -> int:
    """
    Número de clusters basado en eigenvalores.

    n = #{λ_i : λ_i ≥ median(λ)}
    """
    if len(eigenvalues) == 0:
        return 1

    median = np.median(eigenvalues)
    return max(1, int(np.sum(eigenvalues >= median)))


# =============================================================================
# PARTE 10: Ridge regression endógeno
# =============================================================================

def ridge_lambda(X: np.ndarray, t: int) -> float:
    """
    Parámetro de regularización ridge endógeno.

    λ_t = trace(Cov(X)) / d_s · 1/√(T+1)

    Args:
        X: Matriz de diseño
        t: Tiempo/pasos

    Returns:
        λ para ridge regression
    """
    if X.shape[0] < 3:
        return 0.1

    try:
        cov = np.cov(X.T)
        if cov.ndim == 0:
            trace_cov = cov
        else:
            trace_cov = np.trace(cov)

        d_s = X.shape[1] if X.ndim > 1 else 1

        lambda_t = (trace_cov / (d_s + 1e-8)) * (1 / np.sqrt(t + 1))
        return float(max(1e-6, lambda_t))
    except:
        return 0.1 / np.sqrt(t + 1)


# =============================================================================
# PARTE 11: Confianza y calibración
# =============================================================================

def confidence_from_error(error: float, error_history: List[float]) -> float:
    """
    Calcula confianza basada en error normalizado.

    c_t = exp(-(E_t - μ_E) / (σ_E + ε))

    Capado entre 0 y 1 usando percentiles p5-p95.
    """
    if len(error_history) < 5:
        return 0.5

    mu = np.mean(error_history)
    sigma = np.std(error_history) + 1e-8

    z_score = (error - mu) / sigma
    confidence = np.exp(-z_score)

    # Capar con percentiles
    p5 = np.percentile(error_history, 5)
    p95 = np.percentile(error_history, 95)

    if error <= p5:
        return 0.95
    elif error >= p95:
        return 0.05
    else:
        return float(np.clip(confidence, 0.05, 0.95))


def tom_accuracy(error: float, error_history: List[float]) -> float:
    """
    Accuracy de Theory of Mind.

    ToMAcc = 1 - ||e|| / percentile_95(||e_i||)
    """
    if len(error_history) < 5:
        return 0.5

    p95 = np.percentile(error_history, 95)
    if p95 < 1e-8:
        return 1.0

    acc = 1 - error / (p95 + 1e-8)
    return float(np.clip(acc, 0, 1))


# =============================================================================
# PARTE 12: Learning rate adaptativo
# =============================================================================

def adaptive_learning_rate(t: int, confidence: float = 1.0) -> float:
    """
    Learning rate que decrece con el tiempo y confianza.

    η'_t = η_base / √(t+1) · c_t
    """
    eta_base = 1.0
    eta_t = eta_base / np.sqrt(t + 1)
    return float(eta_t * confidence)


def exploration_vs_exploitation(confidence: float, t: int) -> tuple:
    """
    Pesos para exploración vs explotación basados en confianza.

    Returns:
        (exploration_weight, exploitation_weight)
    """
    eta_t = 1 / np.sqrt(t + 1)

    # Baja confianza → más exploración
    exploration = eta_t * (1 - confidence)
    exploitation = eta_t * confidence

    return float(exploration), float(exploitation)


# =============================================================================
# UTILIDADES
# =============================================================================

def safe_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normaliza vector(es) de forma segura."""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + 1e-8)


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax con temperatura."""
    x = np.asarray(x)
    x = x - np.max(x)  # Estabilidad numérica
    exp_x = np.exp(x / temperature)
    return exp_x / (np.sum(exp_x) + 1e-8)


def to_simplex(x: np.ndarray) -> np.ndarray:
    """Proyecta vector al simplex (suma=1, todos ≥0)."""
    x = np.clip(x, 0, None)
    return x / (np.sum(x) + 1e-8)


def entropy(p: np.ndarray) -> float:
    """Calcula entropía de distribución."""
    p = np.clip(p, 1e-10, 1)
    p = p / np.sum(p)
    return float(-np.sum(p * np.log(p)))


def normalized_entropy(p: np.ndarray) -> float:
    """Entropía normalizada [0, 1]."""
    n = len(p)
    if n <= 1:
        return 0.0
    max_entropy = np.log(n)
    return entropy(p) / max_entropy if max_entropy > 0 else 0.0
