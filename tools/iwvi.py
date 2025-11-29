#!/usr/bin/env python3
"""
IWVI — Inter-World Validation Interface
========================================
Sistema de retos/respuestas entre NEO y EVA.
100% endógeno: todos los parámetros derivan de historia local.

Tareas:
1. PREDICCIÓN: predecir x_{t+1..t+h}
2. COMPRESIÓN: estimar MDL con modelo AR
3. POLÍTICA: proponer ΔI que minimice residuales

Scoring:
- ΔRMSE vs baseline AR(p) elegido por BIC
- ΔMDL (bits ahorrados)
- MI kNN (k = floor(T^{1/3}))
- TE (Transfer Entropy)
- Nulo empírico por permutaciones B = floor(10*sqrt(T))
"""
import os
import sys
import json
import math
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

sys.path.insert(0, '/root/NEO_EVA/tools')
from common import (
    load_hist, triplet, sigmas, means, cov3, pctl, quantiles_dict,
    acf_window, pca_v1, median_alpha, k_endogenous, ar_bic,
    mi_knn, transfer_entropy, null_permutation, phase_randomize,
    sha256_str
)
import numpy as np

# ============================================================================
# RUTAS
# ============================================================================
NEO_HIST = "/root/NEOSYNT/state/neo_state.yaml"
EVA_HIST = "/root/EVASYNT/state/history.jsonl"
IWVI_LOG = "/root/NEO_EVA/results/iwvi_log.jsonl"
IWVI_CSV = "/root/NEO_EVA/results/iwvi_scores.csv"
IWVI_REPORT = "/root/NEO_EVA/results/iwvi_report.md"

# ============================================================================
# CARGA DE DATOS
# ============================================================================

def load_neo_history() -> List[Dict]:
    """Carga histórico de NEO desde YAML."""
    import yaml
    if not os.path.exists(NEO_HIST):
        return []
    try:
        with open(NEO_HIST, 'r') as f:
            state = yaml.safe_load(f)
        history = state.get("autonomy", {}).get("history_intention", [])
        records = []
        for i, vec in enumerate(history):
            if isinstance(vec, list) and len(vec) == 3:
                records.append({
                    "t": i,
                    "I": {"S": float(vec[0]), "N": float(vec[1]), "C": float(vec[2])}
                })
        return records
    except:
        return []

def load_eva_history() -> List[Dict]:
    """Carga histórico de EVA."""
    return load_hist(EVA_HIST)

# ============================================================================
# MODELOS PREDICTIVOS
# ============================================================================

def fit_ar(series: List[float], p: int) -> Tuple[np.ndarray, float]:
    """
    Ajusta modelo AR(p).
    Retorna (coeficientes, sigma_residual).
    """
    n = len(series)
    if n < p + 2:
        return (np.zeros(p), 1.0)

    x = np.array(series, dtype=float)
    X = np.column_stack([x[p-i-1:n-i-1] for i in range(p)])
    y = x[p:]

    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta
        sigma = float(np.std(residuals)) if len(residuals) > 0 else 1.0
        return (beta, sigma)
    except:
        return (np.zeros(p), 1.0)

def predict_ar(series: List[float], beta: np.ndarray, h: int) -> List[float]:
    """Predice h pasos adelante con AR."""
    p = len(beta)
    x = list(series[-p:])  # últimos p valores
    preds = []

    for _ in range(h):
        pred = sum(beta[i] * x[-(i+1)] for i in range(p))
        preds.append(pred)
        x.append(pred)

    return preds

def mdl_score(series: List[float], p: int) -> float:
    """
    MDL (Minimum Description Length) para AR(p).
    MDL = -log L + (p/2) * log(n)
    """
    beta, sigma = fit_ar(series, p)
    n = len(series)
    if sigma < 1e-12:
        sigma = 1e-12

    # -log L ≈ (n/2) * log(2π σ²) + n/2
    log_likelihood = -(n/2) * math.log(2 * math.pi * sigma**2) - n/2
    mdl = -log_likelihood + (p/2) * math.log(n)

    return mdl

# ============================================================================
# TAREAS DE RETO
# ============================================================================

class Challenge:
    """Reto inter-mundos."""

    def __init__(self, sender: str, epoch: int, task_type: str,
                 series: List[float], h: int, w: int, p: int):
        self.sender = sender
        self.epoch = epoch
        self.task_type = task_type  # PREDICCION, COMPRESION, POLITICA
        self.series = series
        self.h = h  # horizonte de predicción
        self.w = w  # ventana ACF
        self.p = p  # orden AR (por BIC)
        self.payload = self._create_payload()

    def _create_payload(self) -> Dict:
        """Crea payload del reto."""
        return {
            "task": self.task_type,
            "h": self.h,
            "w": self.w,
            "p": self.p,
            "n": len(self.series),
            "stats": {
                "mean": float(np.mean(self.series)) if self.series else 0,
                "std": float(np.std(self.series)) if self.series else 0,
                "last": self.series[-1] if self.series else 0
            }
        }

    def to_dict(self) -> Dict:
        return {
            "sender": self.sender,
            "epoch": self.epoch,
            "task_type": self.task_type,
            "payload": self.payload,
            "checksum": sha256_str(json.dumps(self.payload))[:16]
        }

class Response:
    """Respuesta a un reto."""

    def __init__(self, responder: str, epoch: int, task_type: str,
                 prediction: Optional[List[float]] = None,
                 compression: Optional[float] = None,
                 policy: Optional[List[float]] = None):
        self.responder = responder
        self.epoch = epoch
        self.task_type = task_type
        self.prediction = prediction
        self.compression = compression
        self.policy = policy

    def to_dict(self) -> Dict:
        return {
            "responder": self.responder,
            "epoch": self.epoch,
            "task_type": self.task_type,
            "prediction": self.prediction,
            "compression": self.compression,
            "policy": self.policy
        }

# ============================================================================
# SCORING
# ============================================================================

def score_prediction(truth: List[float], pred: List[float],
                     baseline_pred: List[float]) -> Dict:
    """Evalúa predicción vs baseline AR."""
    if len(truth) == 0 or len(pred) == 0:
        return {"delta_rmse": 0, "rmse": 0, "baseline_rmse": 0}

    # Truncar a longitud mínima
    n = min(len(truth), len(pred), len(baseline_pred))
    truth = np.array(truth[:n])
    pred = np.array(pred[:n])
    baseline = np.array(baseline_pred[:n])

    rmse = float(np.sqrt(np.mean((truth - pred)**2)))
    baseline_rmse = float(np.sqrt(np.mean((truth - baseline)**2)))
    delta_rmse = baseline_rmse - rmse  # Positivo = mejor que baseline

    return {
        "rmse": rmse,
        "baseline_rmse": baseline_rmse,
        "delta_rmse": delta_rmse
    }

def score_compression(mdl_response: float, mdl_baseline: float) -> Dict:
    """Evalúa compresión vs baseline."""
    delta_mdl = mdl_baseline - mdl_response  # Positivo = mejor
    return {
        "mdl": mdl_response,
        "baseline_mdl": mdl_baseline,
        "delta_mdl": delta_mdl
    }

def score_mutual_info(x: np.ndarray, y: np.ndarray, T: int) -> Dict:
    """Calcula MI kNN con k = floor(T^{1/3})."""
    k = max(1, int(T ** (1/3)))
    mi = mi_knn(x, y, k)
    return {"mi": mi, "k": k}

def score_transfer_entropy(source: List[float], target: List[float],
                           T: int) -> Dict:
    """Calcula TE con k endógeno."""
    k = max(1, int(T ** (1/3)))
    te = transfer_entropy(source, target, lag=1, k=k)
    return {"te": te, "k": k}

# ============================================================================
# NULOS EMPÍRICOS
# ============================================================================

def compute_null_distribution(metric_func, x: np.ndarray, y: np.ndarray,
                              T: int) -> Tuple[float, List[float], float]:
    """
    Calcula distribución nula por permutaciones.
    B = floor(10 * sqrt(T))
    Retorna (observado, nulos, p_hat).
    """
    B = max(10, int(10 * math.sqrt(T)))
    return null_permutation(metric_func, x, y, B)

# ============================================================================
# IWVI ENGINE
# ============================================================================

class IWVIEngine:
    """Motor de validación inter-mundos."""

    def __init__(self):
        self.challenges: List[Challenge] = []
        self.responses: List[Response] = []
        self.scores: List[Dict] = []

    def create_challenge(self, sender: str, epoch: int,
                         hist: List[Dict], component: str = "S") -> Challenge:
        """Crea un reto desde el histórico del emisor."""
        series = triplet(hist, component)
        T = len(series)

        # Parámetros endógenos
        h = max(1, int(math.log(T))) if T > 1 else 1
        w = acf_window(series) if T > 3 else max(3, int(math.log1p(T) + 1))
        p, _ = ar_bic(series)

        # Elegir tarea
        tasks = ["PREDICCION", "COMPRESION", "POLITICA"]
        task_idx = epoch % len(tasks)
        task_type = tasks[task_idx]

        return Challenge(sender, epoch, task_type, series, h, w, p)

    def generate_response(self, responder: str, challenge: Challenge,
                          responder_hist: List[Dict],
                          component: str = "S") -> Response:
        """Genera respuesta usando solo historia local del receptor."""
        series = triplet(responder_hist, component)
        T = len(series)

        if challenge.task_type == "PREDICCION":
            # Predecir usando AR del receptor
            p, _ = ar_bic(series) if T > 5 else (1, 0)
            beta, _ = fit_ar(series, p)
            pred = predict_ar(series, beta, challenge.h)
            return Response(responder, challenge.epoch, "PREDICCION",
                            prediction=pred)

        elif challenge.task_type == "COMPRESION":
            # MDL del receptor
            p, _ = ar_bic(series) if T > 5 else (1, 0)
            mdl = mdl_score(series, p)
            return Response(responder, challenge.epoch, "COMPRESION",
                            compression=mdl)

        elif challenge.task_type == "POLITICA":
            # Propuesta de ΔI basada en v1 del receptor
            if T > 3:
                v1, _, _ = pca_v1(responder_hist)
                alpha = median_alpha(sigmas(responder_hist), T)
                policy = [v1[0] * alpha, v1[1] * alpha, v1[2] * alpha]
            else:
                policy = [0, 0, 0]
            return Response(responder, challenge.epoch, "POLITICA",
                            policy=policy)

        return Response(responder, challenge.epoch, challenge.task_type)

    def evaluate(self, challenge: Challenge, response: Response,
                 truth_series: List[float]) -> Dict:
        """Evalúa respuesta vs baseline del emisor."""
        T = len(truth_series)
        scores = {
            "epoch": challenge.epoch,
            "task": challenge.task_type,
            "sender": challenge.sender,
            "responder": response.responder
        }

        if challenge.task_type == "PREDICCION" and response.prediction:
            # Baseline AR del emisor
            beta, _ = fit_ar(challenge.series, challenge.p)
            baseline_pred = predict_ar(challenge.series, beta, challenge.h)

            # Verdad (si hay suficientes datos)
            if len(truth_series) > len(challenge.series):
                truth = truth_series[len(challenge.series):
                                     len(challenge.series) + challenge.h]
                pred_scores = score_prediction(truth, response.prediction,
                                               baseline_pred)
                scores.update(pred_scores)

        elif challenge.task_type == "COMPRESION" and response.compression:
            baseline_mdl = mdl_score(challenge.series, challenge.p)
            comp_scores = score_compression(response.compression, baseline_mdl)
            scores.update(comp_scores)

        elif challenge.task_type == "POLITICA" and response.policy:
            # Evaluar por correlación con v1 del emisor
            if T > 3:
                sender_hist = [{"I": {"S": s}} for s in challenge.series]
                # Construir hist completo (proxy)
                full_hist = []
                for i, s in enumerate(challenge.series):
                    full_hist.append({"t": i, "I": {"S": s, "N": 0, "C": 0}})
                v1_sender, _, _ = pca_v1(full_hist)
                # Similitud de dirección
                dot = sum(a*b for a, b in zip(response.policy, v1_sender))
                scores["policy_alignment"] = float(dot)

        # MI y TE
        if len(truth_series) > 10:
            x = np.array(challenge.series[-min(50, T):])
            y = np.array(truth_series[-min(50, len(truth_series)):])
            if len(x) > 5 and len(y) > 5:
                # MI
                min_len = min(len(x), len(y))
                mi_scores = score_mutual_info(x[:min_len], y[:min_len], T)
                scores.update(mi_scores)

                # TE
                te_scores = score_transfer_entropy(
                    list(x[:min_len]), list(y[:min_len]), T)
                scores["te"] = te_scores["te"]

                # Nulos para MI
                obs_mi, null_mi, p_mi = compute_null_distribution(
                    lambda a, b: mi_knn(a, b, max(1, int(len(a)**(1/3)))),
                    x[:min_len], y[:min_len], T
                )
                scores["mi_p_hat"] = p_mi
                scores["mi_null_median"] = float(np.median(null_mi))

        self.scores.append(scores)
        return scores

    def compute_borda_score(self, scores: Dict) -> float:
        """Calcula score Borda (rank sum normalizado)."""
        metrics = ["delta_rmse", "delta_mdl", "mi", "te"]
        valid_metrics = [m for m in metrics if m in scores and scores[m] is not None]

        if not valid_metrics:
            return 0.0

        # Para simplificar, normalizamos cada métrica a [0,1]
        # (en producción, usaríamos ranks sobre todos los scores)
        normalized = []
        for m in valid_metrics:
            val = scores[m]
            if m in ["delta_rmse", "delta_mdl"]:
                # Positivo es mejor
                normalized.append(1.0 if val > 0 else 0.0)
            else:
                # Mayor es mejor para MI, TE
                normalized.append(min(1.0, max(0.0, val)))

        return float(np.mean(normalized)) if normalized else 0.0

# ============================================================================
# EJECUCIÓN
# ============================================================================

def run_iwvi_cycle(neo_hist: List[Dict], eva_hist: List[Dict],
                   epoch: int, engine: IWVIEngine) -> Dict:
    """Ejecuta un ciclo de IWVI."""
    results = {"epoch": epoch}

    # NEO reta a EVA
    if neo_hist:
        neo_challenge = engine.create_challenge("NEO", epoch, neo_hist)
        eva_response = engine.generate_response("EVA", neo_challenge, eva_hist)
        neo_truth = triplet(neo_hist, "S")
        neo_scores = engine.evaluate(neo_challenge, eva_response, neo_truth)
        neo_scores["borda"] = engine.compute_borda_score(neo_scores)
        results["neo_to_eva"] = neo_scores

    # EVA reta a NEO
    if eva_hist:
        eva_challenge = engine.create_challenge("EVA", epoch, eva_hist)
        neo_response = engine.generate_response("NEO", eva_challenge, neo_hist)
        eva_truth = triplet(eva_hist, "S")
        eva_scores = engine.evaluate(eva_challenge, neo_response, eva_truth)
        eva_scores["borda"] = engine.compute_borda_score(eva_scores)
        results["eva_to_neo"] = eva_scores

    # Log
    os.makedirs(os.path.dirname(IWVI_LOG), exist_ok=True)
    with open(IWVI_LOG, 'a') as f:
        f.write(json.dumps(results) + '\n')

    return results

def generate_iwvi_report(engine: IWVIEngine) -> str:
    """Genera reporte markdown."""
    scores = engine.scores
    if not scores:
        return "# IWVI Report\n\nNo hay datos de validación."

    # Extraer métricas
    delta_rmse = [s.get("delta_rmse", 0) for s in scores if "delta_rmse" in s]
    delta_mdl = [s.get("delta_mdl", 0) for s in scores if "delta_mdl" in s]
    mi_vals = [s.get("mi", 0) for s in scores if "mi" in s]
    te_vals = [s.get("te", 0) for s in scores if "te" in s]
    p_hats = [s.get("mi_p_hat", 1) for s in scores if "mi_p_hat" in s]
    borda = [s.get("borda", 0) for s in scores if "borda" in s]

    report = []
    report.append("# IWVI — Reporte de Validación Inter-Mundos\n")
    report.append(f"Generado: {datetime.now().isoformat()}\n")
    report.append(f"Total de evaluaciones: {len(scores)}\n")

    report.append("\n## Distribución de Métricas\n")
    report.append("| Métrica | Mediana | IQR | p95 |\n")
    report.append("|---------|---------|-----|-----|\n")

    for name, vals in [("ΔRMSE", delta_rmse), ("ΔMDL", delta_mdl),
                       ("MI", mi_vals), ("TE", te_vals), ("Borda", borda)]:
        if vals:
            med = float(np.median(vals))
            iqr_val = float(np.percentile(vals, 75) - np.percentile(vals, 25))
            p95 = float(np.percentile(vals, 95))
            report.append(f"| {name} | {med:.4f} | {iqr_val:.4f} | {p95:.4f} |\n")

    report.append("\n## Significancia Estadística\n")
    if p_hats:
        n_signif = sum(1 for p in p_hats if p <= 0.05)
        frac = n_signif / len(p_hats)
        report.append(f"- Fracción de retos con MI p̂ ≤ 0.05: {frac:.2%}\n")
        report.append(f"- p̂ mediano: {float(np.median(p_hats)):.4f}\n")

    report.append("\n## Parámetros Derivados\n")
    report.append("- Todos los parámetros (h, w, p, k, B) derivados de T local\n")
    report.append("- h = floor(log T)\n")
    report.append("- w = primer lag con ACF < mediana envolvente\n")
    report.append("- p = orden AR por BIC\n")
    report.append("- k = floor(T^{1/3})\n")
    report.append("- B = floor(10√T)\n")

    return "".join(report)

# ============================================================================
# MAIN
# ============================================================================

def main(cycles: int = 100):
    """Ejecuta IWVI por cycles ciclos."""
    print(f"IWVI iniciando ({cycles} ciclos)...")

    engine = IWVIEngine()

    for epoch in range(cycles):
        neo_hist = load_neo_history()
        eva_hist = load_eva_history()

        if not neo_hist and not eva_hist:
            print(f"  Epoch {epoch}: Sin datos, esperando...")
            continue

        results = run_iwvi_cycle(neo_hist, eva_hist, epoch, engine)

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: NEO_T={len(neo_hist)}, EVA_T={len(eva_hist)}")

    # Generar reporte
    report = generate_iwvi_report(engine)
    os.makedirs(os.path.dirname(IWVI_REPORT), exist_ok=True)
    with open(IWVI_REPORT, 'w') as f:
        f.write(report)

    # CSV
    import csv
    if engine.scores:
        keys = set()
        for s in engine.scores:
            keys.update(s.keys())
        keys = sorted(keys)

        with open(IWVI_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for s in engine.scores:
                writer.writerow(s)

    print(f"\nIWVI completado.")
    print(f"Reporte: {IWVI_REPORT}")
    print(f"CSV: {IWVI_CSV}")
    print(f"Log: {IWVI_LOG}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=100)
    args = parser.parse_args()
    main(args.cycles)
