#!/usr/bin/env python3
import os, json, glob, math, csv, time, statistics as st
from datetime import datetime
import numpy as np

# Rutas conocidas
NEO_HIST = "/root/NEOSYNT/state/neo_state.yaml"  # NEO usa YAML
EVA_HIST = "/root/EVASYNT/state/history.jsonl"
BUS_LOG  = "/root/NEO_EVA/logs/bus.log"
JAC_FILE = "/root/NEO_EVA/results/jacobian_neo.json"
P4_LOG   = "/root/NEO_EVA/logs/phase4.log"
IWVI_DIR = "/root/NEO_EVA/results"

OUT_MD   = "/root/NEO_EVA/results/neo_eva_status.md"
OUT_CSV  = "/root/NEO_EVA/results/neo_eva_status_timeseries.csv"

def load_neo_hist():
    if not os.path.exists(NEO_HIST): return []
    import yaml
    with open(NEO_HIST) as f:
        state = yaml.safe_load(f)
    raw = state.get("autonomy", {}).get("history_intention", [])
    return [{"t": i, "I": {"S": v[0], "N": v[1], "C": v[2]}} for i, v in enumerate(raw) if len(v) == 3]

def load_eva_hist():
    if not os.path.exists(EVA_HIST): return []
    with open(EVA_HIST) as f:
        return [json.loads(l) for l in f if l.strip()]

def last_intent(hist):
    if not hist: return (None, None, None, None)
    r = hist[-1]
    t = r.get("t")
    I = r.get("I",{})
    return (t, I.get("S"), I.get("N"), I.get("C"))

def var_last_window(hist):
    if len(hist)<10: return None
    T=len(hist)
    w=max(10,int(math.sqrt(T)))
    S=[h["I"]["S"] for h in hist[-w:]]
    N=[h["I"]["N"] for h in hist[-w:]]
    C=[h["I"]["C"] for h in hist[-w:]]
    return {"w":w,"varS":float(np.var(S)), "varN":float(np.var(N)), "varC":float(np.var(C))}

def load_json_safe(p):
    if os.path.exists(p):
        try: return json.load(open(p))
        except: return None
    return None

def list_iwvi_files():
    cands = []
    for pat in ["iwvi_*.json","iwvi_*.csv","iwvi_report*.md","iwvi_scores*.csv","iwvi_metrics*.json","iwvi_null*.json"]:
        cands += glob.glob(os.path.join(IWVI_DIR, pat))
    return sorted(cands)

def bus_stats():
    if not os.path.exists(BUS_LOG): 
        return {"count":0,"by_agent":{},"last_msg":None,"last_epoch":None}
    by_agent={}
    last=None; last_epoch=None
    with open(BUS_LOG) as f:
        for line in f:
            if not line.strip(): continue
            try:
                j=json.loads(line)
                a=j.get("agent","?")
                by_agent[a]=by_agent.get(a,0)+1
                last=j; last_epoch=j.get("epoch",last_epoch)
            except: pass
    return {
        "count": sum(by_agent.values()),
        "by_agent": by_agent,
        "last_msg": last,
        "last_epoch": last_epoch
    }

def write_csv_timeseries(histN, histE, path):
    with open(path,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["idx","neo_t","neo_S","neo_N","neo_C","eva_t","eva_S","eva_N","eva_C"])
        m=max(len(histN), len(histE))
        for i in range(m):
            n = histN[i] if i<len(histN) else {}
            e = histE[i] if i<len(histE) else {}
            w.writerow([
                i,
                (n.get("t") if n else ""), (n.get("I",{}).get("S") if n else ""),
                (n.get("I",{}).get("N") if n else ""), (n.get("I",{}).get("C") if n else ""),
                (e.get("t") if e else ""), (e.get("I",{}).get("S") if e else ""),
                (e.get("I",{}).get("N") if e else ""), (e.get("I",{}).get("C") if e else "")
            ])

def main():
    ts = datetime.utcnow().isoformat()+"Z"
    neoH = load_neo_hist()
    evaH = load_eva_hist()

    neo_last = last_intent(neoH)
    eva_last = last_intent(evaH)
    neo_var = var_last_window(neoH)
    eva_var = var_last_window(evaH)

    jac = load_json_safe(JAC_FILE)
    rho = jac.get("rho") if jac else None
    eig_real = jac.get("eigvals_real") if jac else None
    eig_imag = jac.get("eigvals_imag") if jac else None

    iwvi_files = list_iwvi_files()
    bus = bus_stats()

    write_csv_timeseries(neoH, evaH, OUT_CSV)

    lines=[]
    lines.append(f"# NEO↔EVA — Estado (solo lectura)  \nGenerado: {ts}")
    lines.append("\n## Mundo NEO")
    lines.append(f"- T = {len(neoH)} ciclos")
    lines.append(f"- Última intención: t={neo_last[0]}, S={neo_last[1]:.6f}, N={neo_last[2]:.2e}, C={neo_last[3]:.2e}" if neo_last[1] else "- Última intención: n/d")
    if neo_var:
        lines.append(f"- Var últimas {neo_var['w']} muestras: VarS={neo_var['varS']:.3e}, VarN={neo_var['varN']:.3e}, VarC={neo_var['varC']:.3e}")
    if rho is not None:
        eig_str = ", ".join([f"{r:.4f}{'+' if i>=0 else ''}{i:.4f}i" if abs(i)>1e-10 else f"{r:.4f}" for r,i in zip(eig_real or [], eig_imag or [])])
        lines.append(f"- ρ(J) = {rho:.6f}")
        lines.append(f"- Eigenvalores: [{eig_str}]")
        lines.append(f"- Estable (ρ<1): {'Sí' if rho < 1 else 'No'}")

    lines.append("\n## Mundo EVA")
    lines.append(f"- T = {len(evaH)} ciclos")
    if eva_last[1] is not None:
        lines.append(f"- Última intención: t={eva_last[0]}, S={eva_last[1]:.6f}, N={eva_last[2]:.6f}, C={eva_last[3]:.6f}")
    else:
        lines.append("- Última intención: n/d")
    if eva_var:
        lines.append(f"- Var últimas {eva_var['w']} muestras: VarS={eva_var['varS']:.3e}, VarN={eva_var['varN']:.3e}, VarC={eva_var['varC']:.3e}")

    lines.append("\n## Vínculo NEO↔EVA (BUS + IWVI)")
    lines.append(f"- Mensajes BUS: total={bus['count']}, por agente={bus['by_agent']}")
    if bus['last_epoch']:
        lines.append(f"- Último epoch: {bus['last_epoch']}")
    if iwvi_files:
        lines.append(f"- Artefactos IWVI ({len(iwvi_files)}):")
        for p in iwvi_files[:10]:
            lines.append(f"  - {os.path.basename(p)}")
    else:
        lines.append("- Artefactos IWVI: ninguno")

    # Phase 4 status
    p4_result = load_json_safe("/root/NEO_EVA/results/phase4_integration_test.json")
    if p4_result:
        lines.append("\n## Phase 4 (Variabilidad Endógena)")
        lines.append(f"- Activaciones: {p4_result.get('activations', 'n/d')}")
        lines.append(f"- Tasa activación: {p4_result.get('activation_rate', 0)*100:.1f}%")
        lines.append(f"- IWVI válido: {p4_result.get('iwvi_valid', 'n/d')}")

    lines.append("\n## Ficheros generados")
    lines.append(f"- CSV series: {OUT_CSV}")
    lines.append(f"- Este reporte: {OUT_MD}")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    open(OUT_MD,"w").write("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
