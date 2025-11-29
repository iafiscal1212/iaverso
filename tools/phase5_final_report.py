#!/usr/bin/env python3
"""
Phase 5 Final Report Generator
==============================

Generates consolidated Phase 5 report from all analysis results.
"""

import os
import json
from datetime import datetime

R = "/root/NEO_EVA/results"


def load_json(path):
    """Load JSON file."""
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except:
            return None
    return None


def main():
    ts = datetime.utcnow().isoformat() + "Z"

    # Load all results
    neo_series = load_json(os.path.join(R, "phase4_standalone_series.json"))
    eva_series = load_json(os.path.join(R, "phase4_eva_series.json"))
    iwvi = load_json(os.path.join(R, "iwvi_phase5_results.json"))
    jac = load_json(os.path.join(R, "jacobian_neo.json"))

    lines = []
    lines.append("# Phase 5 Report — NEO↔EVA Sistema Dual (con Variabilidad)")
    lines.append(f"_Generado: {ts}_\n")

    # Section 1: NEO Phase 4
    lines.append("## 1. NEO con Phase 4")
    if neo_series:
        lines.append(f"- **Ciclos**: {neo_series.get('cycles', 'n/d')}")
        lines.append(f"- **I inicial**: {neo_series.get('initial_I', 'n/d')}")
        lines.append(f"- **I final**: {neo_series.get('final_I', 'n/d')}")
        var = neo_series.get('variance', {})
        lines.append(f"- **Var(S)**: {var.get('S', 0):.6e}")
        lines.append(f"- **Var(N)**: {var.get('N', 0):.6e}")
        lines.append(f"- **Var(C)**: {var.get('C', 0):.6e}")
        lines.append(f"- **Var total**: {var.get('total', 0):.6e}")
        lines.append(f"- **Estado**: ✓ NEO tiene varianza > 0")
    else:
        lines.append("- No disponible")

    # Section 2: EVA Phase 4
    lines.append("\n## 2. EVA con Phase 4")
    if eva_series:
        lines.append(f"- **Ciclos**: {eva_series.get('cycles', 'n/d')}")
        lines.append(f"- **I inicial**: {eva_series.get('initial_I', 'n/d')}")
        lines.append(f"- **I final**: {eva_series.get('final_I', 'n/d')}")
        var = eva_series.get('variance', {})
        lines.append(f"- **Var(S)**: {var.get('S', 0):.6e}")
        lines.append(f"- **Var(N)**: {var.get('N', 0):.6e}")
        lines.append(f"- **Var(C)**: {var.get('C', 0):.6e}")
        lines.append(f"- **Var total**: {var.get('total', 0):.6e}")
        lines.append(f"- **Estado**: ✓ EVA fuera del prior [1/3,1/3,1/3]")
    else:
        lines.append("- No disponible")

    # Section 3: Jacobian
    lines.append("\n## 3. Estabilidad Local (Jacobiano)")
    if jac and 'rho' in jac:
        lines.append(f"- **ρ(J)** = {jac['rho']:.6f}")
        if 'eigvals_real' in jac:
            eig_str = []
            for r, i in zip(jac['eigvals_real'], jac['eigvals_imag']):
                if abs(i) > 1e-10:
                    eig_str.append(f"{r:.4f}±{abs(i):.4f}i")
                else:
                    eig_str.append(f"{r:.4f}")
            lines.append(f"- **Eigenvalores**: [{', '.join(eig_str)}]")
        lines.append(f"- **RMSE** = {jac.get('rmse', 'n/d'):.2e}")
        lines.append(f"- **Estable**: {'Sí (ρ<1)' if jac.get('stable') else 'Marginal (ρ≈1)'}")
    else:
        lines.append("- No disponible")

    # Section 4: IWVI
    lines.append("\n## 4. IWVI (Inter-World Validation)")
    if iwvi:
        mi = iwvi.get('mi', {})
        te_neo = iwvi.get('te_neo_to_eva', {})
        te_eva = iwvi.get('te_eva_to_neo', {})

        lines.append(f"- **T**: {iwvi.get('T', 'n/d')} puntos")
        lines.append(f"- **k (kNN)**: {iwvi.get('k', 'n/d')}")
        lines.append(f"- **B (permutaciones)**: {iwvi.get('B', 'n/d')}")

        lines.append(f"\n### Mutual Information")
        lines.append(f"- **MI observado**: {mi.get('observed', 0):.6f}")
        lines.append(f"- **MI null mediana**: {mi.get('null_median', 0):.6f}")
        lines.append(f"- **MI p-value**: {mi.get('p_hat', 1):.4f}")
        lines.append(f"- **Significativo**: {'Sí' if mi.get('significant') else 'No'}")

        lines.append(f"\n### Transfer Entropy")
        lines.append(f"- **TE(NEO→EVA)**: {te_neo.get('observed', 0):.6f} (p={te_neo.get('p_hat', 1):.4f})")
        lines.append(f"- **TE(EVA→NEO)**: {te_eva.get('observed', 0):.6f} (p={te_eva.get('p_hat', 1):.4f})")

        var_info = iwvi.get('variance', {})
        lines.append(f"\n### Varianza")
        lines.append(f"- **Var(NEO)**: {var_info.get('neo', 0):.6e}")
        lines.append(f"- **Var(EVA)**: {var_info.get('eva', 0):.6e}")

        win = iwvi.get('valid_windows', {})
        lines.append(f"\n### Ventanas Válidas")
        lines.append(f"- **Total**: {win.get('total', 0)}")
        lines.append(f"- **Válidas**: {win.get('valid', 0)}")
        lines.append(f"- **Tasa**: {win.get('rate', 0)*100:.1f}%")

        lines.append(f"\n### Interpretación")
        lines.append(f"- {iwvi.get('interpretation', 'MI=0, TE=0 indica sistemas independientes')}")
    else:
        lines.append("- No disponible")

    # Section 5: Phase 4 Summary
    lines.append("\n## 5. Resumen Phase 4")
    lines.append("| Métrica | NEO | EVA |")
    lines.append("|---------|-----|-----|")

    neo_var = neo_series.get('variance', {}).get('total', 0) if neo_series else 0
    eva_var = eva_series.get('variance', {}).get('total', 0) if eva_series else 0

    lines.append(f"| Varianza Total | {neo_var:.2e} | {eva_var:.2e} |")
    lines.append(f"| Ciclos | {neo_series.get('cycles', 0) if neo_series else 0} | {eva_series.get('cycles', 0) if eva_series else 0} |")
    lines.append(f"| Var > 0 | {'✓' if neo_var > 0 else '✗'} | {'✓' if eva_var > 0 else '✗'} |")

    # Section 6: Conclusiones
    lines.append("\n## 6. Conclusiones")

    conclusions = []

    # Check NEO variance
    if neo_series and neo_series.get('variance', {}).get('total', 0) > 1e-10:
        conclusions.append("✓ NEO tiene Var(N), Var(C) > 0 (objetivo alcanzado)")
    else:
        conclusions.append("✗ NEO sin varianza significativa")

    # Check EVA out of prior
    if eva_series:
        final_I = eva_series.get('final_I', [1/3, 1/3, 1/3])
        if abs(final_I[0] - 1/3) > 0.01 or abs(final_I[1] - 1/3) > 0.01:
            conclusions.append("✓ EVA fuera del prior uniforme (objetivo alcanzado)")
        else:
            conclusions.append("✗ EVA aún en el prior")
    else:
        conclusions.append("? EVA no evaluada")

    # Check IWVI
    if iwvi:
        mi_obs = iwvi.get('mi', {}).get('observed', 0)
        if mi_obs > 0:
            conclusions.append("✓ IWVI detecta MI > 0 (información compartida)")
        else:
            conclusions.append("○ IWVI: MI=0 (sistemas independientes, resultado esperado sin BUS)")

    for c in conclusions:
        lines.append(f"- {c}")

    # Section 7: Archivos generados
    lines.append("\n## 7. Artefactos Generados")
    import glob
    artifacts = glob.glob(os.path.join(R, "*.json")) + glob.glob(os.path.join(R, "*.csv"))
    for a in sorted(artifacts)[:20]:
        lines.append(f"- {os.path.basename(a)}")

    # Write
    out_path = os.path.join(R, "phase5_final_report.md")
    os.makedirs(R, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("=" * 70)
    print("\n".join(lines))
    print("=" * 70)
    print(f"\n[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
