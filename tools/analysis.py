#!/usr/bin/env python3
"""
NEO_EVA Analysis Tools
======================
Herramientas de análisis dinámico:
1. Jacobiano local
2. Mapa de susceptibilidad χ(α), τ(α)
3. Nulos por aleatorización de fase
4. Ablaciones

100% endógeno. Sin hardcodeo.
"""
import os
import sys
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

sys.path.insert(0, '/root/NEO_EVA/tools')
from common import (
    load_hist, triplet, sigmas, cov3, pctl, acf_window,
    pca_full, median_alpha, phase_randomize, sha256_file
)
import numpy as np

# ============================================================================
# RUTAS
# ============================================================================
NEO_HIST_YAML = "/root/NEOSYNT/state/neo_state.yaml"
EVA_HIST = "/root/EVASYNT/state/history.jsonl"
RESULTS_DIR = "/root/NEO_EVA/results"

# ============================================================================
# JACOBIANO LOCAL
# ============================================================================

def estimate_jacobian(hist: List[Dict], eps_scale: float = 1.0) -> Dict:
    """
    Estima Jacobiano local del sistema de intención.
    Usa diferencias finitas con escala endógena.

    J_ij ≈ ∂I_i(t+1)/∂I_j(t)

    Retorna J (3x3), eigenvalores, radio espectral ρ.
    """
    T = len(hist)
    if T < 10:
        return {"error": "Historial insuficiente", "T": T}

    # Extraer transiciones
    transitions = []
    for i in range(1, T):
        I_prev = [hist[i-1]["I"]["S"], hist[i-1]["I"]["N"], hist[i-1]["I"]["C"]]
        I_curr = [hist[i]["I"]["S"], hist[i]["I"]["N"], hist[i]["I"]["C"]]
        transitions.append((I_prev, I_curr))

    # Construir matrices para regresión
    # I_{t+1} ≈ J @ I_t + c
    X = np.array([t[0] for t in transitions])  # I_t
    Y = np.array([t[1] for t in transitions])  # I_{t+1}

    # Añadir columna de 1s para bias
    X_aug = np.column_stack([X, np.ones(len(X))])

    # Regresión lineal: [J | c] = (X^T X)^{-1} X^T Y
    try:
        beta = np.linalg.lstsq(X_aug, Y, rcond=None)[0]
        J = beta[:3, :].T  # 3x3
        c = beta[3, :]     # bias

        # Eigenvalores
        eigvals, eigvecs = np.linalg.eig(J)
        rho = float(max(abs(eigvals)))  # Radio espectral

        # Residuales
        Y_pred = X_aug @ beta
        residuals = Y - Y_pred
        rmse = float(np.sqrt(np.mean(residuals**2)))

        return {
            "J": J.tolist(),
            "eigvals_real": [float(np.real(e)) for e in eigvals],
            "eigvals_imag": [float(np.imag(e)) for e in eigvals],
            "eigvecs_real": [[float(np.real(x)) for x in v] for v in eigvecs.T],
            "eigvecs_imag": [[float(np.imag(x)) for x in v] for v in eigvecs.T],
            "rho": rho,
            "bias": c.tolist(),
            "rmse": rmse,
            "T": T,
            "stable": rho < 1.0
        }
    except Exception as e:
        return {"error": str(e), "T": T}

# ============================================================================
# SUSCEPTIBILIDAD
# ============================================================================

def compute_susceptibility_map(hist: List[Dict],
                               directions: Optional[List[List[float]]] = None,
                               n_alphas: int = 6) -> Dict:
    """
    Computa mapa de susceptibilidad χ(α) y tiempo de relajación τ(α).

    χ(α) = ||ΔI_ss|| / α
    τ(α) = pasos hasta ||I - I*|| < ε

    Direcciones y alphas derivados endógenamente.
    """
    T = len(hist)
    if T < 20:
        return {"error": "Historial insuficiente", "T": T}

    # Estado estacionario = media de últimos w valores
    w = acf_window(triplet(hist, "S"))
    I_star = np.array([
        np.mean(triplet(hist[-w:], "S")),
        np.mean(triplet(hist[-w:], "N")),
        np.mean(triplet(hist[-w:], "C"))
    ])

    # Direcciones desde PCA si no se especifican
    if directions is None:
        COV = cov3(hist)
        lambdas, vecs, varexp = pca_full(COV)
        directions = vecs  # v1, v2, v3

    # Alphas por percentiles de σ/√T
    sig = sigmas(hist)
    percentiles = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    alphas = []
    for p in percentiles[:n_alphas]:
        alpha = pctl(list(sig), p) / max(1.0, math.sqrt(T))
        alphas.append(alpha)

    # Mapa
    results = []
    for d in directions:
        v = np.array(d)
        v = v / (np.linalg.norm(v) + 1e-12)

        for alpha in alphas:
            # Perturbación inicial
            I_pert = I_star + alpha * v

            # χ = ||ΔI|| / α (respuesta estacionaria)
            # Aproximamos con la varianza observada en esa dirección
            proj = np.array([np.dot([h["I"]["S"], h["I"]["N"], h["I"]["C"]] - I_star, v)
                            for h in hist[-w:]])
            chi = float(np.std(proj) / (alpha + 1e-12))

            # τ = escala de autocorrelación en esa dirección
            tau = acf_window(list(proj)) if len(proj) > 3 else w

            results.append({
                "dir": d,
                "alpha": float(alpha),
                "chi": chi,
                "tau": tau
            })

    return {
        "I_star": I_star.tolist(),
        "w": w,
        "results": results,
        "T": T
    }

# ============================================================================
# NULOS POR FASE ALEATORIA
# ============================================================================

def null_phase_test(hist: List[Dict],
                    metric: str = "variance",
                    B: Optional[int] = None) -> Dict:
    """
    Test de nulos por aleatorización de fase.
    Preserva espectro pero destruye estructura temporal.

    B = floor(10*sqrt(T)) si no se especifica.
    """
    T = len(hist)
    if T < 20:
        return {"error": "Historial insuficiente", "T": T}

    series = np.array(triplet(hist, "S"))

    if B is None:
        B = max(10, int(10 * math.sqrt(T)))

    # Métrica observada
    if metric == "variance":
        obs = float(np.var(series))
        metric_func = np.var
    elif metric == "autocorr_sum":
        # Suma de |ACF| hasta lag w
        w = acf_window(list(series))
        from common import acf
        obs = float(sum(abs(r) for r in acf(list(series), w)))
        metric_func = lambda x: sum(abs(r) for r in acf(list(x), w))
    else:
        obs = float(np.var(series))
        metric_func = np.var

    # Distribución nula
    null_dist = []
    for _ in range(B):
        surrogate = phase_randomize(series)
        null_dist.append(float(metric_func(surrogate)))

    # p-valor (fracción de nulos >= observado)
    p_hat = sum(1 for v in null_dist if v >= obs) / B

    return {
        "metric": metric,
        "observed": obs,
        "null_median": float(np.median(null_dist)),
        "null_p25": float(np.percentile(null_dist, 25)),
        "null_p75": float(np.percentile(null_dist, 75)),
        "p_hat": p_hat,
        "B": B,
        "T": T
    }

# ============================================================================
# ABLACIONES
# ============================================================================

ABLATION_STATE = "/root/NEO_EVA/state/ablations.json"

def get_ablation_flags() -> Dict:
    """Obtiene flags de ablación actuales."""
    if not os.path.exists(ABLATION_STATE):
        return {
            "no_recall_eva": False,
            "no_gate": False,
            "no_bus": False,
            "no_pca": False
        }
    try:
        with open(ABLATION_STATE, 'r') as f:
            return json.load(f)
    except:
        return {}

def set_ablation(flag: str, value: bool = True) -> Dict:
    """Activa/desactiva flag de ablación."""
    flags = get_ablation_flags()
    flags[flag] = value
    os.makedirs(os.path.dirname(ABLATION_STATE), exist_ok=True)
    with open(ABLATION_STATE, 'w') as f:
        json.dump(flags, f, indent=2)
    return flags

def clear_ablations() -> Dict:
    """Limpia todas las ablaciones."""
    flags = {
        "no_recall_eva": False,
        "no_gate": False,
        "no_bus": False,
        "no_pca": False
    }
    os.makedirs(os.path.dirname(ABLATION_STATE), exist_ok=True)
    with open(ABLATION_STATE, 'w') as f:
        json.dump(flags, f, indent=2)
    return flags

# ============================================================================
# PRE-REGISTRO
# ============================================================================

def generate_preregistration() -> str:
    """Genera documento de pre-registro."""
    content = []
    content.append("# Pre-registro de Experimentos NEO↔EVA")
    content.append(f"\nFecha: {datetime.now().isoformat()}")
    content.append("\n## Hipótesis")
    content.append("H1: ρ(J) << 1 en todas las direcciones locales (atractor fuerte).")
    content.append("H2: Existe α* (cuantil ≥ p95 de τ) donde aparece régimen no-lineal.")
    content.append("H3: MI/TE inter-mundos supera p95 del nulo empírico en ≥ una ventana.")
    content.append("H4: La ablación de componentes (recall, gate, bus) degrada el score IWVI.")

    content.append("\n## Umbrales (endógenos)")
    content.append("- α: cuantiles de median(σ)/√T : {p25, p50, p75, p90, p95, p99}")
    content.append("- τ*: p95(τ) por dirección")
    content.append("- Significancia: percentil ≥ p95 del nulo, no α fijos")
    content.append("- k (MI/TE): floor(T^{1/3})")
    content.append("- B (permutaciones): floor(10√T)")

    content.append("\n## Métricas")
    content.append("- ΔRMSE: mejora vs baseline AR(p) con p por BIC")
    content.append("- ΔMDL: bits ahorrados vs baseline")
    content.append("- MI kNN: información mutua por k-vecinos")
    content.append("- TE: transfer entropy source→target")
    content.append("- Score Borda: rank sum normalizado")

    content.append("\n## Ablaciones planificadas")
    content.append("- no_recall_eva: desactivar memoria episódica de EVA")
    content.append("- no_gate: desactivar gate de exploración conjunta")
    content.append("- no_bus: desactivar comunicación inter-mundos")
    content.append("- no_pca: usar dirección aleatoria en lugar de v1")

    content.append("\n## Criterios de éxito")
    content.append("1. Aumentos selectivos de exploración cuando sorpresa/MI del otro sube")
    content.append("2. Cero constantes hardcodeadas en el código")
    content.append("3. Retorno a S alto tras episodios exploratorios (sin colapso)")
    content.append("4. p̂ < 0.05 para MI en al menos 20% de las evaluaciones")

    return "\n".join(content)

# ============================================================================
# REPRODUCIBILIDAD
# ============================================================================

def generate_reproducibility_info() -> Dict:
    """Genera información de reproducibilidad."""
    import platform
    import hashlib

    files_to_hash = [
        "/root/NEO_EVA/tools/common.py",
        "/root/NEO_EVA/tools/iwvi.py",
        "/root/NEO_EVA/tools/analysis.py",
        "/root/NEO_EVA/bus.py",
        "/root/EVASYNT/core/evasynt.py",
        "/root/NEOSYNT/neo_bus_listener.py"
    ]

    hashes = {}
    for f in files_to_hash:
        if os.path.exists(f):
            hashes[f] = sha256_file(f)

    return {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "file_hashes": hashes,
        "ablation_state": get_ablation_flags()
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="NEO_EVA Analysis Tools")
    parser.add_argument("command", choices=[
        "jacobian", "susceptibility", "nulls",
        "ablate", "clear_ablations", "prereg", "repro"
    ])
    parser.add_argument("--world", choices=["neo", "eva"], default="neo")
    parser.add_argument("--flag", type=str, help="Ablation flag name")
    parser.add_argument("--B", type=int, help="Number of permutations")
    args = parser.parse_args()

    # Cargar histórico
    if args.world == "neo":
        import yaml
        if os.path.exists(NEO_HIST_YAML):
            with open(NEO_HIST_YAML, 'r') as f:
                state = yaml.safe_load(f)
            raw = state.get("autonomy", {}).get("history_intention", [])
            hist = [{"t": i, "I": {"S": v[0], "N": v[1], "C": v[2]}}
                   for i, v in enumerate(raw) if len(v) == 3]
        else:
            hist = []
    else:
        hist = load_hist(EVA_HIST)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.command == "jacobian":
        result = estimate_jacobian(hist)
        print(json.dumps(result, indent=2))
        with open(f"{RESULTS_DIR}/jacobian_{args.world}.json", 'w') as f:
            json.dump(result, f, indent=2)

    elif args.command == "susceptibility":
        result = compute_susceptibility_map(hist)
        print(json.dumps(result, indent=2))
        with open(f"{RESULTS_DIR}/susceptibility_{args.world}.json", 'w') as f:
            json.dump(result, f, indent=2)

    elif args.command == "nulls":
        result = null_phase_test(hist, B=args.B)
        print(json.dumps(result, indent=2))
        with open(f"{RESULTS_DIR}/nulls_{args.world}.json", 'w') as f:
            json.dump(result, f, indent=2)

    elif args.command == "ablate":
        if args.flag:
            result = set_ablation(args.flag, True)
            print(json.dumps(result, indent=2))
        else:
            print("Uso: --flag <flag_name>")

    elif args.command == "clear_ablations":
        result = clear_ablations()
        print(json.dumps(result, indent=2))

    elif args.command == "prereg":
        prereg = generate_preregistration()
        print(prereg)
        with open(f"{RESULTS_DIR}/preregistration.md", 'w') as f:
            f.write(prereg)

    elif args.command == "repro":
        result = generate_reproducibility_info()
        print(json.dumps(result, indent=2))
        with open(f"{RESULTS_DIR}/reproducibility.json", 'w') as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
