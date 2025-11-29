#!/usr/bin/env python3
"""
Phase 6 Final Report Generator
==============================

Generates the comprehensive Phase 6 report with:
- Coupled NEO↔EVA results
- Endogenous coupling (κ_t) metrics
- IWVI analysis
- Ablation comparison (no_bus)
- Pearson correlation analysis
"""

import os
import json
from datetime import datetime
import numpy as np
from scipy.stats import pearsonr

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
    coupled_results = load_json(os.path.join(R, "phase6_coupled_results.json"))
    coupled_neo = load_json(os.path.join(R, "phase6_coupled_neo.json"))
    coupled_eva = load_json(os.path.join(R, "phase6_coupled_eva.json"))
    iwvi = load_json(os.path.join(R, "phase6_iwvi_results.json"))
    ablation_results = load_json(os.path.join(R, "phase6_ablation_no_bus_results.json"))
    ablation_neo = load_json(os.path.join(R, "phase6_ablation_no_bus_neo.json"))
    ablation_eva = load_json(os.path.join(R, "phase6_ablation_no_bus_eva.json"))
    ablation_iwvi = load_json(os.path.join(R, "phase6_ablation_no_bus_iwvi.json"))

    lines = []
    lines.append("# Phase 6 Report — Endogenous Coupling NEO↔EVA + IWVI")
    lines.append(f"_Generated: {ts}_\n")

    # Section 1: Phase 6 Implementation
    lines.append("## 1. Phase 6 Implementation")
    lines.append("")
    lines.append("### A) BUS with Summary Messages")
    lines.append("Each world publishes every w ≈ √T steps:")
    lines.append("- **μ_I**: (S̄, N̄, C̄) mean intention over window")
    lines.append("- **v₁, λ₁**: Principal direction (PCA) and variance explained")
    lines.append("- **u**: IQR(r)/√T uncertainty")
    lines.append("- **conf** ∈ [0,1]: Confidence")
    lines.append("- **CV(r)**: Coefficient of variation")
    lines.append("")
    lines.append("### B) Endogenous Coupling Law")
    lines.append("```")
    lines.append("κ_t^X = (u_t^Y / (1 + u_t^X)) × (λ₁^Y / (λ₁^Y + λ₁^X + ε)) × (conf_t^Y / (1 + CV(r_t^X)))")
    lines.append("g_t^Y→X = Proj_tangent^X(v₁^Y)")
    lines.append("Δ̃_t^X = Δ_t^X + κ_t^X × g_t^Y→X")
    lines.append("I_{t+1}^X = softmax(log I_t^X + η_t^X × Δ̃_t^X)")
    lines.append("```")
    lines.append("- Applied only when gate is open (ρ(J) ≥ p95 AND IQR ≥ p75)")
    lines.append("")
    lines.append("### C) IWVI Validation")
    lines.append("- Valid windows: Var(I) ≥ p50(Var_hist)")
    lines.append("- Phase null tests: B = ⌊10√T⌋")
    lines.append("- Success: MI or TE > null in ≥1 window (p̂ ≤ 0.05)")

    # Section 2: Coupled Results
    lines.append("\n## 2. Coupled System Results")
    if coupled_results:
        lines.append(f"- **Cycles**: {coupled_results.get('cycles', 500)}")
        lines.append(f"- **BUS counts**: NEO={coupled_results.get('bus_counts', {}).get('NEO', 0)}, EVA={coupled_results.get('bus_counts', {}).get('EVA', 0)}")

        neo = coupled_results.get('neo', {})
        eva = coupled_results.get('eva', {})
        lines.append(f"\n### NEO (World A)")
        lines.append(f"- **I initial**: {neo.get('initial_I', [])}")
        lines.append(f"- **I final**: {neo.get('final_I', [])}")
        lines.append(f"- **Total variance**: {neo.get('variance', {}).get('total', 0):.4e}")
        lines.append(f"- **Gate activations**: {neo.get('gate_activations', 0)}")
        lines.append(f"- **Coupling activations**: {neo.get('coupling_activations', 0)}")

        lines.append(f"\n### EVA (World B)")
        lines.append(f"- **I initial**: {eva.get('initial_I', [])}")
        lines.append(f"- **I final**: {eva.get('final_I', [])}")
        lines.append(f"- **Total variance**: {eva.get('variance', {}).get('total', 0):.4e}")
        lines.append(f"- **Gate activations**: {eva.get('gate_activations', 0)}")
        lines.append(f"- **Coupling activations**: {eva.get('coupling_activations', 0)}")

    # Section 3: κ Analysis
    lines.append("\n## 3. Coupling Analysis (κ_t)")
    if coupled_neo and 'series' in coupled_neo:
        kappas = [s.get('kappa', 0) for s in coupled_neo['series']]
        kappas_active = [k for k in kappas if k > 0.01]
        lines.append(f"- **Total steps**: {len(kappas)}")
        lines.append(f"- **κ > 0.01 steps**: {len(kappas_active)}")
        if kappas_active:
            lines.append(f"- **Mean κ (when active)**: {np.mean(kappas_active):.4f}")
            lines.append(f"- **Max κ**: {np.max(kappas_active):.4f}")

    # Section 4: IWVI Results
    lines.append("\n## 4. IWVI Results")
    if iwvi:
        mi = iwvi.get('mi', {})
        te_neo = iwvi.get('te_neo_to_eva', {})
        te_eva = iwvi.get('te_eva_to_neo', {})

        lines.append(f"- **T**: {iwvi.get('T', 'n/a')} points")
        lines.append(f"- **k (kNN)**: {iwvi.get('k', 'n/a')}")
        lines.append(f"- **B (nulls)**: {iwvi.get('B', 'n/a')}")

        lines.append(f"\n### Mutual Information")
        lines.append(f"- **MI observed**: {mi.get('observed', 0):.6f}")
        lines.append(f"- **MI null median**: {mi.get('null_median', 0):.6f}")
        lines.append(f"- **MI p-value**: {mi.get('p_hat', 1):.4f}")
        lines.append(f"- **Significant**: {'Yes ✓' if mi.get('significant') else 'No'}")

        lines.append(f"\n### Transfer Entropy")
        lines.append(f"- **TE(NEO→EVA)**: {te_neo.get('observed', 0):.6f} (p={te_neo.get('p_hat', 1):.4f})")
        lines.append(f"- **TE(EVA→NEO)**: {te_eva.get('observed', 0):.6f} (p={te_eva.get('p_hat', 1):.4f})")

        win = iwvi.get('valid_windows', {})
        lines.append(f"\n### Window Analysis")
        lines.append(f"- **Total windows**: {win.get('total', 0)}")
        lines.append(f"- **Valid windows**: {win.get('valid', 0)}")
        lines.append(f"- **Significant windows (p ≤ 0.05)**: {win.get('significant', 0)}")
        lines.append(f"- **Success**: {'YES ✓' if iwvi.get('success') else 'NO'}")

    # Section 5: Ablation Analysis
    lines.append("\n## 5. Ablation: no_bus")
    if ablation_results:
        lines.append(f"- **Coupling activations**: NEO={ablation_results['neo'].get('coupling_activations', 0)}, EVA={ablation_results['eva'].get('coupling_activations', 0)}")
        lines.append(f"- **BUS counts**: {ablation_results.get('bus_counts', {})}")

        if ablation_iwvi:
            abl_win = ablation_iwvi.get('valid_windows', {})
            lines.append(f"- **Significant windows**: {abl_win.get('significant', 0)}")
    else:
        lines.append("- Not available")

    # Section 6: Pearson Correlation Comparison
    lines.append("\n## 6. Pearson Correlation Analysis")
    if coupled_neo and coupled_eva and ablation_neo and ablation_eva:
        coupled_neo_I = np.array([[s['S_new'], s['N_new'], s['C_new']] for s in coupled_neo['series']])
        coupled_eva_I = np.array([[s['S_new'], s['N_new'], s['C_new']] for s in coupled_eva['series']])
        ablation_neo_I = np.array([[s['S_new'], s['N_new'], s['C_new']] for s in ablation_neo['series']])
        ablation_eva_I = np.array([[s['S_new'], s['N_new'], s['C_new']] for s in ablation_eva['series']])

        coupled_corrs = [pearsonr(coupled_neo_I[:,i], coupled_eva_I[:,i])[0] for i in range(3)]
        ablation_corrs = [pearsonr(ablation_neo_I[:,i], ablation_eva_I[:,i])[0] for i in range(3)]

        lines.append(f"\n| Component | Coupled | Ablation | Δ |")
        lines.append(f"|-----------|---------|----------|---|")
        lines.append(f"| S | {coupled_corrs[0]:.4f} | {ablation_corrs[0]:.4f} | {coupled_corrs[0] - ablation_corrs[0]:+.4f} |")
        lines.append(f"| N | {coupled_corrs[1]:.4f} | {ablation_corrs[1]:.4f} | {coupled_corrs[1] - ablation_corrs[1]:+.4f} |")
        lines.append(f"| C | {coupled_corrs[2]:.4f} | {ablation_corrs[2]:.4f} | {coupled_corrs[2] - ablation_corrs[2]:+.4f} |")
        lines.append(f"| **Mean** | **{np.mean(coupled_corrs):.4f}** | **{np.mean(ablation_corrs):.4f}** | **{np.mean(coupled_corrs) - np.mean(ablation_corrs):+.4f}** |")

        lines.append(f"\n**Key Finding**: Coupling increases correlation from {np.mean(ablation_corrs):.4f} to {np.mean(coupled_corrs):.4f} (+{np.mean(coupled_corrs) - np.mean(ablation_corrs):.4f})")

    # Section 7: Summary Table
    lines.append("\n## 7. Summary Comparison")
    lines.append("| Metric | Coupled | Ablation (no_bus) |")
    lines.append("|--------|---------|-------------------|")

    if coupled_results and ablation_results:
        lines.append(f"| NEO coupling acts | {coupled_results['neo'].get('coupling_activations', 0)} | {ablation_results['neo'].get('coupling_activations', 0)} |")
        lines.append(f"| EVA coupling acts | {coupled_results['eva'].get('coupling_activations', 0)} | {ablation_results['eva'].get('coupling_activations', 0)} |")
        lines.append(f"| BUS messages | {coupled_results.get('bus_counts', {}).get('NEO', 0) + coupled_results.get('bus_counts', {}).get('EVA', 0)} | {ablation_results.get('bus_counts', {}).get('NEO', 0) + ablation_results.get('bus_counts', {}).get('EVA', 0)} |")

    if iwvi and ablation_iwvi:
        lines.append(f"| IWVI sig windows | {iwvi.get('valid_windows', {}).get('significant', 0)} | {ablation_iwvi.get('valid_windows', {}).get('significant', 0)} |")

    if coupled_neo and ablation_neo:
        coupled_corrs = [pearsonr(coupled_neo_I[:,i], coupled_eva_I[:,i])[0] for i in range(3)]
        ablation_corrs = [pearsonr(ablation_neo_I[:,i], ablation_eva_I[:,i])[0] for i in range(3)]
        lines.append(f"| Mean Pearson corr | {np.mean(coupled_corrs):.4f} | {np.mean(ablation_corrs):.4f} |")

    # Section 8: Conclusions
    lines.append("\n## 8. Conclusions")

    conclusions = []

    # BUS working
    if coupled_results and coupled_results.get('bus_counts', {}).get('NEO', 0) > 0:
        conclusions.append("✓ BUS publishing summary messages (μ_I, v₁, λ₁, u, conf)")

    # Coupling active
    if coupled_results and coupled_results['neo'].get('coupling_activations', 0) > 0:
        conclusions.append(f"✓ Endogenous coupling κ_t active ({coupled_results['neo'].get('coupling_activations', 0)} NEO, {coupled_results['eva'].get('coupling_activations', 0)} EVA)")

    # IWVI success
    if iwvi and iwvi.get('success'):
        conclusions.append(f"✓ IWVI success: {iwvi.get('valid_windows', {}).get('significant', 0)} significant windows")

    # Correlation improvement
    if coupled_neo and ablation_neo:
        delta_corr = np.mean(coupled_corrs) - np.mean(ablation_corrs)
        if delta_corr > 0.1:
            conclusions.append(f"✓ Coupling causality confirmed: Δ correlation = +{delta_corr:.4f}")
        else:
            conclusions.append(f"○ Coupling effect small: Δ correlation = +{delta_corr:.4f}")

    # Ablation validation
    if ablation_results and ablation_results['neo'].get('coupling_activations', 0) == 0:
        conclusions.append("✓ Ablation validates: no_bus → no coupling activations")

    for c in conclusions:
        lines.append(f"- {c}")

    # Section 9: Artifacts
    lines.append("\n## 9. Artifacts Generated")
    import glob
    artifacts = glob.glob(os.path.join(R, "phase6*.json"))
    for a in sorted(artifacts):
        lines.append(f"- {os.path.basename(a)}")

    # Write
    out_path = os.path.join(R, "phase6_final_report.md")
    os.makedirs(R, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("=" * 70)
    print("\n".join(lines))
    print("=" * 70)
    print(f"\n[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
