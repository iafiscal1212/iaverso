#!/usr/bin/env python3
import os, json, math, time, glob, hashlib
from statistics import median
from collections import Counter, defaultdict

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

NOW = time.time()
HOURS = float(os.environ.get("HOURS", "12"))   # ventana en horas para "reciente"
RECENT_SEC = HOURS*3600

def load_jsonl(path):
    if path is None or not os.path.exists(path): return []
    out=[]
    with open(path,'r') as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try: out.append(json.loads(ln))
            except: pass
    return out

def load_yaml(path):
    if path is None or not os.path.exists(path): return {}
    if not HAS_YAML:
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}

def qtile(vals, q):
    if not vals: return None
    vals=sorted(vals)
    k=(len(vals)-1)*q
    i=math.floor(k); j=math.ceil(k)
    if i==j: return vals[i]
    return vals[i]+(vals[j]-vals[i])*(k-i)

def safe(var, default=None):
    return default if var is None or (isinstance(var,float) and math.isnan(var)) else var

def summarize_world_from_yaml(tag, yaml_path):
    """Para NEO que usa neo_state.yaml"""
    state = load_yaml(yaml_path)
    if not state:
        return {"world":tag, "t":"-", "last_I":"-", "n":0, "source":"yaml_not_found"}

    auto = state.get("autonomy", {})
    current_I = auto.get("current_intention", [None, None, None])

    # Obtener t desde el estado del daemon
    t = state.get("cycle_count", state.get("t", "-"))

    # Historial de errores (proxy para actividad)
    hist_errors = auto.get("history_abs_errors", [])
    n = len(hist_errors) if hist_errors else 0

    # Varianza de errores como proxy
    def var(x):
        if len(x)<2: return 0.0
        m=sum(x)/len(x)
        return sum((xi-m)**2 for xi in x)/max(1,(len(x)-1))

    var_errors = var(hist_errors[-1000:]) if hist_errors else 0.0

    # Intentar obtener info de population
    pop = state.get("population", {})
    pop_size = len(pop.get("individuals", [])) if isinstance(pop.get("individuals"), list) else pop.get("size", 0)

    # Modos del ethos
    ethos = state.get("ethos", {})

    return {
        "world": tag,
        "t": t,
        "n": n,
        "last_I": [safe(current_I[0]), safe(current_I[1]), safe(current_I[2])] if len(current_I)>=3 else current_I,
        "var_last1k": {"errors": var_errors},
        "prop_rate": 0.0,
        "gate_rate": None,
        "modes": {},
        "cuts": {"count":0, "top":[]},
        "recent_len": min(n, 1000),
        "recent_last_hours": HOURS,
        "population_size": pop_size,
        "source": "yaml"
    }

def summarize_world(tag, hist_path, bandit_path):
    hist=load_jsonl(hist_path)
    band=load_jsonl(bandit_path)
    if not hist:
        return {"world":tag, "t":"-", "last_I":"-", "n":0}

    # tiempos
    t_last = hist[-1].get("t", None)
    t0 = hist[0].get("t", None)
    # filtro reciente por timestamp UNIX si existe, si no por índice relativo
    def stamp(e): return e.get("ts", None)
    has_ts = stamp(hist[0]) is not None
    recent = [e for e in hist if (NOW - (e.get("ts", NOW))) <= RECENT_SEC] if has_ts else hist[-min(len(hist),3000):]

    # intención y varianzas
    S=[e["I"]["S"] for e in hist if "I" in e]
    N=[e["I"]["N"] for e in hist if "I" in e]
    C=[e["I"]["C"] for e in hist if "I" in e]
    lastI = [S[-1] if S else None, N[-1] if N else None, C[-1] if C else None]

    def var(x):
        if len(x)<2: return 0.0
        m=sum(x)/len(x)
        return sum((xi-m)**2 for xi in x)/max(1,(len(x)-1))

    varS,varN,varC = var(S[-1000:]), var(N[-1000:]), var(C[-1000:])

    # propuestas y consentimiento
    a = [e.get("a",None) for e in hist if "a" in e]  # propuesta individual
    prop_rate = sum(1 for v in a if v==1)/len(a) if a else 0.0

    # modos (bandit)
    modes=[b.get("mode") for b in band if "mode" in b]
    mcount=Counter(modes)

    # gates/cortes
    gates=[e.get("gate_on") for e in hist if "gate_on" in e]
    gate_rate = sum(1 for g in gates if g)/len(gates) if gates else None

    cuts=[e.get("cut_reason") for e in hist if "cut_reason" in e and e["cut_reason"]]
    cut_count = len(cuts); cut_top = Counter(cuts).most_common(3)

    # métricas recientes
    recent_I = [e["I"] for e in recent if "I" in e]
    recS = [i["S"] for i in recent_I]; recN=[i["N"] for i in recent_I]; recC=[i["C"] for i in recent_I]

    return {
        "world":tag,
        "t0":t0, "t":t_last, "n":len(hist),
        "last_I": [safe(lastI[0]), safe(lastI[1]), safe(lastI[2])],
        "var_last1k": {"S":varS,"N":varN,"C":varC},
        "prop_rate": prop_rate,
        "gate_rate": gate_rate,
        "modes": dict(mcount),
        "cuts": {"count":cut_count, "top":cut_top},
        "recent_len": len(recent),
        "recent_last_hours": HOURS,
        "source": "jsonl"
    }

def summarize_coupling(cpl_path):
    cpl=load_jsonl(cpl_path)
    if not cpl: return {"events":0}

    # bilaterales, modos, consent lift aproximado
    both=[e for e in cpl if e.get("both_on")==1]
    events=len(both)

    # prob de propuestas individuales si vienen logueadas
    pN = qN = None
    aN=[e.get("a_NEO") for e in cpl if "a_NEO" in e]
    aE=[e.get("a_EVA") for e in cpl if "a_EVA" in e]
    if aN: pN=sum(1 for x in aN if x==1)/len(aN)
    if aE: qN=sum(1 for x in aE if x==1)/len(aE)
    bothrate = sum(1 for e in cpl if e.get("both_on")==1)/len(cpl) if cpl else 0.0
    lift = (bothrate / (pN*qN)) if (pN and qN and pN*qN>0) else None

    modes=[(e.get("mode_NEO"),e.get("mode_EVA")) for e in both]
    from collections import Counter
    mh = Counter(modes).most_common(6)

    return {"events":events, "pNEO":pN, "pEVA":qN, "both_rate":bothrate, "lift":lift, "mode_pairs_top": mh}

def try_path(*cands):
    for p in cands:
        if p and os.path.exists(p): return p
    return None

# rutas típicas - priorizar Phase 7 live data
neo_hist = try_path(
    "/root/NEO_EVA/state/neo_history.jsonl",  # Phase 7 converted
    "/root/NEOSYNT/state/history.jsonl",
    "/root/NEOSYNT/state/intent_history.jsonl"
)
neo_yaml = try_path("/root/NEOSYNT/state/neo_state.yaml")
neo_band = try_path(
    "/root/NEO_EVA/results/phase7_live/coupled/bandit_stats.json",
    "/root/NEOSYNT/state/bandit.log.jsonl",
    "/root/NEOSYNT/state/mode_bandit.jsonl"
)
eva_hist = try_path(
    "/root/NEO_EVA/state/eva_history.jsonl",  # Phase 7 converted
    "/root/EVASYNT/state/history.jsonl"
)
eva_band = try_path("/root/EVASYNT/state/bandit.log.jsonl")
coupling = try_path(
    "/root/NEO_EVA/logs/coupling.jsonl",
    "/root/NEO_EVA/results/phase7_coupling.jsonl"
)

# NEO: usa YAML si no hay history.jsonl
if neo_hist:
    neo = summarize_world("NEO", neo_hist, neo_band)
else:
    neo = summarize_world_from_yaml("NEO", neo_yaml)

# EVA: usa JSONL
eva = summarize_world("EVA", eva_hist, eva_band)
cpl = summarize_coupling(coupling)

report = {"ts": time.time(), "hours_window": HOURS, "NEO": neo, "EVA": eva, "COUPLING": cpl}

outdir="/root/NEO_EVA/results/"
os.makedirs(outdir, exist_ok=True)
with open(os.path.join(outdir, "phase7_live_digest.json"),"w") as f:
    json.dump(report,f,indent=2)

# print resumen humano
def fmtI(I):
    if isinstance(I, list) and len(I) >= 3 and None not in I:
        return f"[S={I[0]:.4f}, N={I[1]:.4f}, C={I[2]:.4f}]"
    return str(I)

print("=== PHASE 7 LIVE DIGEST ===")
for w in ("NEO","EVA"):
    d=report[w]
    src = d.get("source", "unknown")
    print(f"\n{w} ({src}): t={d.get('t')}  n={d.get('n')}  last I={fmtI(d.get('last_I'))}")
    v=d.get("var_last1k",{})
    if "S" in v:
        print(f"  var@last1k: S={v.get('S',0):.3e} N={v.get('N',0):.3e} C={v.get('C',0):.3e}")
    elif "errors" in v:
        print(f"  var@last1k (errors): {v.get('errors',0):.3e}")
    pr = d.get('prop_rate', 0) or 0
    gr = d.get('gate_rate')
    print(f"  prop_rate={pr:.3f}  gate_rate={gr if gr is not None else 'NA'}")
    print(f"  modes={d.get('modes')}")
    cuts=d.get("cuts",{})
    print(f"  cuts={cuts.get('count',0)}  top={cuts.get('top')}")
    if "population_size" in d:
        print(f"  population_size={d.get('population_size')}")
print("\nCOUPLING:")
print(f"  events(bilateral)={cpl.get('events')}")
print(f"  p(a_NEO)={cpl.get('pNEO')}  p(a_EVA)={cpl.get('pEVA')}  both_rate={cpl.get('both_rate')}")
print(f"  consent_lift≈{cpl.get('lift')}")
print(f"  mode_pairs_top={cpl.get('mode_pairs_top')}")
