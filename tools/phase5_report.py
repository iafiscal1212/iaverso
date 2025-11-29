#!/usr/bin/env python3
import os, json, glob, math
from datetime import datetime
import numpy as np

R = "/root/NEO_EVA/results"
NEO_H = "/root/NEOSYNT/state/neo_state.yaml"
EVA_H = "/root/EVASYNT/state/history.jsonl"

OUT = os.path.join(R, "phase5_report.md")

def load_json(path):
    if os.path.exists(path):
        try: return json.load(open(path))
        except: return None
    return None

def load_neo_hist():
    if not os.path.exists(NEO_H): return []
    import yaml
    with open(NEO_H) as f:
        state = yaml.safe_load(f)
    raw = state.get("autonomy", {}).get("history_intention", [])
    return [{"t": i, "I": {"S": v[0], "N": v[1], "C": v[2]}} for i, v in enumerate(raw) if len(v) == 3]

def load_eva_hist():
    if not os.path.exists(EVA_H): return []
    with open(EVA_H) as f:
        return [json.loads(l) for l in f if l.strip()]

def last_I(hist):
    if not hist: return None
    j=hist[-1]; I=j.get("I",{})
    return (j.get("t"), I.get("S"), I.get("N"), I.get("C"))

def var_window(hist):
    if len(hist)<10: return None
    T=len(hist); w=max(10,int(math.sqrt(T)))
    S=[h["I"]["S"] for h in hist[-w:]]
    N=[h["I"]["N"] for h in hist[-w:]]
    C=[h["I"]["C"] for h in hist[-w:]]
    return {"w":w,"VarS":float(np.var(S)),"VarN":float(np.var(N)),"VarC":float(np.var(C))}

def main():
    ts = datetime.utcnow().isoformat()+"Z"

    # Load data
    jac = load_json(os.path.join(R,"jacobian_neo.json"))
    susc = load_json(os.path.join(R,"susceptibility_neo.json"))
    nulls = load_json(os.path.join(R,"nulls_neo.json"))
    iwvi_nulls = load_json(os.path.join(R,"iwvi_null_tests.json"))
    p4 = load_json(os.path.join(R,"phase4_integration_test.json"))
    ablations = load_json(os.path.join(R,"ablation_expected.json"))

    neo_hist = load_neo_hist()
    eva_hist = load_eva_hist()

    neo_last = last_I(neo_hist)
    eva_last = last_I(eva_hist)
    neo_var = var_window(neo_hist)
    eva_var = var_window(eva_hist)

    lines=[]
    lines.append(f"# Phase 5 Report — NEO↔EVA Sistema Dual")
    lines.append(f"_Generado: {ts}_\n")

    # Section 1: State
    lines.append("## 1. Estado Actual")
    lines.append("\n### NEO")
    lines.append(f"- **T** = {len(neo_hist)} ciclos")
    if neo_last:
        lines.append(f"- **Última I**: t={neo_last[0]}, S={neo_last[1]:.6f}, N={neo_last[2]:.2e}, C={neo_last[3]:.2e}")
    if neo_var:
        lines.append(f"- **Var (w={neo_var['w']})**: S={neo_var['VarS']:.3e}, N={neo_var['VarN']:.3e}, C={neo_var['VarC']:.3e}")

    lines.append("\n### EVA")
    lines.append(f"- **T** = {len(eva_hist)} ciclos")
    if eva_last:
        lines.append(f"- **Última I**: t={eva_last[0]}, S={eva_last[1]:.6f}, N={eva_last[2]:.6f}, C={eva_last[3]:.6f}")
    if eva_var:
        lines.append(f"- **Var (w={eva_var['w']})**: S={eva_var['VarS']:.3e}, N={eva_var['VarN']:.3e}, C={eva_var['VarC']:.3e}")

    # Section 2: Jacobian
    lines.append("\n## 2. Estabilidad Local (Jacobiano)")
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
        lines.append(f"- **Estable**: {'Sí' if jac.get('stable') else 'No (ρ≥1)'}")
    else:
        lines.append("- No disponible")

    # Section 3: Susceptibility
    lines.append("\n## 3. Susceptibilidad")
    if susc and 'results' in susc:
        results = susc['results']
        if results:
            chis = [r['chi'] for r in results if 'chi' in r]
            taus = [r['tau'] for r in results if 'tau' in r]
            if chis:
                lines.append(f"- **χ**: min={min(chis):.3e}, max={max(chis):.3e}, median={np.median(chis):.3e}")
            if taus:
                lines.append(f"- **τ**: min={min(taus)}, max={max(taus)}, median={np.median(taus):.1f}")
    else:
        lines.append("- No disponible")

    # Section 4: Null tests
    lines.append("\n## 4. Tests de Nulos")
    if nulls:
        lines.append(f"- **Métrica**: {nulls.get('metric', 'variance')}")
        lines.append(f"- **Observado**: {nulls.get('observed', 'n/d'):.6f}")
        lines.append(f"- **Nulo mediana**: {nulls.get('null_median', 'n/d'):.6f}")
        lines.append(f"- **p̂**: {nulls.get('p_hat', 'n/d'):.4f}")
        lines.append(f"- **B**: {nulls.get('B', 'n/d')}")
    else:
        lines.append("- No disponible")

    # Section 5: IWVI
    lines.append("\n## 5. IWVI (Inter-World Validation)")
    if iwvi_nulls:
        mi = iwvi_nulls.get('mi', {})
        te = iwvi_nulls.get('te', {})
        lines.append(f"- **MI observado**: {mi.get('observed', 'n/d'):.6f}")
        lines.append(f"- **MI p̂**: {mi.get('p_hat', 'n/d'):.4f}")
        lines.append(f"- **TE observado**: {te.get('observed', 'n/d'):.6f}")
        lines.append(f"- **TE p̂**: {te.get('p_hat', 'n/d'):.4f}")
        lines.append(f"- **k (kNN)**: {iwvi_nulls.get('k', 'n/d')}")
        lines.append(f"- **B**: {iwvi_nulls.get('B', 'n/d')}")
    else:
        lines.append("- No disponible")

    # Section 6: Phase 4
    lines.append("\n## 6. Phase 4 (Variabilidad Endógena)")
    if p4:
        lines.append(f"- **Activaciones**: {p4.get('activations', 0)}")
        lines.append(f"- **Tasa**: {p4.get('activation_rate', 0)*100:.1f}%")
        lines.append(f"- **IWVI válido**: {p4.get('iwvi_valid', 'n/d')}")
        diag = p4.get('diagnostics', {})
        if 'controller_state' in diag:
            cs = diag['controller_state']
            lines.append(f"- **Gate activations**: {cs.get('gate_activations', 0)}")
    else:
        lines.append("- No ejecutado")

    # Section 7: Ablations
    lines.append("\n## 7. Ablaciones (Esperadas)")
    if ablations:
        lines.append("| Ablación | Factor | ρ esperado |")
        lines.append("|----------|--------|-----------|")
        for a in ablations:
            lines.append(f"| {a['name']} | x{a['degradation_factor']:.2f} | {a['expected_rho']:.4f} |")
    else:
        lines.append("- No disponible")

    # Section 8: Files
    lines.append("\n## 8. Artefactos Generados")
    artifacts = glob.glob(os.path.join(R, "*.json")) + glob.glob(os.path.join(R, "*.csv")) + glob.glob(os.path.join(R, "*.md"))
    for a in sorted(artifacts)[:20]:
        lines.append(f"- {os.path.basename(a)}")

    # Write
    os.makedirs(R, exist_ok=True)
    open(OUT,"w").write("\n".join(lines)+"\n")
    print(f"\n{'='*60}")
    print("\n".join(lines))
    print(f"\n{'='*60}")
    print(f"\nSaved: {OUT}")

if __name__=="__main__":
    main()
