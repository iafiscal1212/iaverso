#!/usr/bin/env python3
"""
Phase 7 Analysis: Análisis Completo del Sistema de Consentimiento
================================================================

Calcula:
1. Condicional por modo (-1/0/+1): r, MI, TE dir, ΔRMSE/ΔMDL/G
2. Bandit: curva de regret + tasa de selección por modo
3. Consent lift: P(a_N & a_E), P(a_N), P(a_E), lift + IC bootstrap
4. Safety: % cortes por ρ p99, Var p25, tiempo medio OFF
5. Figuras: barras utilidad, r/MI/TE por modo, heatmap frecuencias

Autor: Carmen Esteban
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple
import hashlib
from datetime import datetime
import os

# =============================================================================
# Cargar datos
# =============================================================================

def load_consent_logs(base_dir: str = "/root/NEO_EVA/results/phase7/coupled"):
    """Carga los logs de consentimiento."""
    with open(f"{base_dir}/consent_log_neo.json", 'r') as f:
        neo_log = json.load(f)
    with open(f"{base_dir}/consent_log_eva.json", 'r') as f:
        eva_log = json.load(f)
    with open(f"{base_dir}/series_neo.json", 'r') as f:
        neo_series = json.load(f)
    with open(f"{base_dir}/series_eva.json", 'r') as f:
        eva_series = json.load(f)
    with open(f"{base_dir}/bilateral_events.json", 'r') as f:
        bilateral = json.load(f)
    with open(f"{base_dir}/bandit_stats.json", 'r') as f:
        bandit_stats = json.load(f)

    return {
        'neo_log': neo_log,
        'eva_log': eva_log,
        'neo_series': neo_series,
        'eva_series': eva_series,
        'bilateral': bilateral,
        'bandit_stats': bandit_stats
    }


# =============================================================================
# 1. Condicional por modo (-1/0/+1)
# =============================================================================

def analyze_by_mode(data: Dict) -> Dict:
    """Analiza r, MI, TE direccional, ΔRMSE/ΔMDL/G por modo."""

    neo_log = data['neo_log']
    eva_log = data['eva_log']
    neo_series = data['neo_series']
    eva_series = data['eva_series']

    # Extraer I arrays
    neo_I = np.array([r['I_new'] for r in neo_series])
    eva_I = np.array([r['I_new'] for r in eva_series])

    # Modo por timestep
    neo_modes = np.array([r['m'] for r in neo_log])
    eva_modes = np.array([r['m'] for r in eva_log])

    # G por timestep
    neo_G = np.array([r['G'] for r in neo_log])
    eva_G = np.array([r['G'] for r in eva_log])

    results = {}

    for mode in [-1, 0, 1]:
        # Índices donde NEO usó este modo
        idx = np.where(neo_modes == mode)[0]

        if len(idx) < 10:
            results[mode] = {'n': len(idx), 'insufficient_data': True}
            continue

        # Correlación durante este modo
        neo_subset = neo_I[idx]
        eva_subset = eva_I[idx]

        # r (correlación media de componentes)
        correlations = []
        for i in range(3):
            if len(neo_subset[:, i]) > 2:
                r, p = pearsonr(neo_subset[:, i], eva_subset[:, i])
                correlations.append((r, p))

        r_mean = np.mean([c[0] for c in correlations])
        r_p = np.mean([c[1] for c in correlations])

        # MI aproximada (correlación máxima como proxy)
        mi_approx = np.max([abs(c[0]) for c in correlations])

        # TE direccional aproximada (lag correlation)
        te_neo_to_eva = []
        te_eva_to_neo = []
        for i in range(3):
            if len(idx) > 10:
                # TE(NEO→EVA): correlación neo[t] con eva[t+1]
                if len(neo_subset[:-1, i]) > 2 and len(eva_subset[1:, i]) > 2:
                    te1, _ = pearsonr(neo_subset[:-1, i], eva_subset[1:, i])
                    te_neo_to_eva.append(te1)
                # TE(EVA→NEO): correlación eva[t] con neo[t+1]
                if len(eva_subset[:-1, i]) > 2 and len(neo_subset[1:, i]) > 2:
                    te2, _ = pearsonr(eva_subset[:-1, i], neo_subset[1:, i])
                    te_eva_to_neo.append(te2)

        te_neo_eva = np.mean(te_neo_to_eva) if te_neo_to_eva else 0
        te_eva_neo = np.mean(te_eva_to_neo) if te_eva_to_neo else 0

        # G medio
        G_mean = np.mean(neo_G[idx])
        G_std = np.std(neo_G[idx])

        results[mode] = {
            'n': len(idx),
            'r_mean': float(r_mean),
            'r_p': float(r_p),
            'MI_approx': float(mi_approx),
            'TE_neo_to_eva': float(te_neo_eva),
            'TE_eva_to_neo': float(te_eva_neo),
            'G_mean': float(G_mean),
            'G_std': float(G_std),
        }

    return results


# =============================================================================
# 2. Bandit: regret y selección por modo
# =============================================================================

def analyze_bandit(data: Dict) -> Dict:
    """Analiza curva de regret y tasa de selección por modo."""

    neo_series = data['neo_series']
    eva_series = data['eva_series']

    # Regret acumulado
    neo_regret = [r['bandit_regret'] for r in neo_series]
    eva_regret = [r.get('bandit_regret', 0) for r in eva_series]

    # Modo seleccionado
    neo_modes = [r['mode'] for r in neo_series]
    eva_modes = [r['mode'] for r in eva_series]

    # Ventanas para analizar evolución
    window_size = 500
    n_windows = len(neo_modes) // window_size

    mode_evolution_neo = []
    mode_evolution_eva = []

    for w in range(n_windows):
        start = w * window_size
        end = (w + 1) * window_size

        neo_window = neo_modes[start:end]
        eva_window = eva_modes[start:end]

        mode_evolution_neo.append({
            'window': w,
            't_start': start,
            't_end': end,
            'mode_-1': sum(1 for m in neo_window if m == -1) / len(neo_window),
            'mode_0': sum(1 for m in neo_window if m == 0) / len(neo_window),
            'mode_+1': sum(1 for m in neo_window if m == 1) / len(neo_window),
        })

        mode_evolution_eva.append({
            'window': w,
            't_start': start,
            't_end': end,
            'mode_-1': sum(1 for m in eva_window if m == -1) / len(eva_window),
            'mode_0': sum(1 for m in eva_window if m == 0) / len(eva_window),
            'mode_+1': sum(1 for m in eva_window if m == 1) / len(eva_window),
        })

    return {
        'neo_regret': neo_regret,
        'eva_regret': eva_regret,
        'neo_final_regret': neo_regret[-1] if neo_regret else 0,
        'eva_final_regret': eva_regret[-1] if eva_regret else 0,
        'mode_evolution_neo': mode_evolution_neo,
        'mode_evolution_eva': mode_evolution_eva,
        'bandit_stats': data['bandit_stats'],
    }


# =============================================================================
# 3. Consent Lift con Bootstrap CI
# =============================================================================

def analyze_consent_lift(data: Dict, n_bootstrap: int = 1000) -> Dict:
    """Calcula P(a_N & a_E), P(a_N), P(a_E), lift + IC bootstrap."""

    neo_log = data['neo_log']
    eva_log = data['eva_log']

    # a = willing to couple
    a_neo = np.array([r['a'] for r in neo_log])
    a_eva = np.array([r['a'] for r in eva_log])

    n = len(a_neo)

    # Probabilidades
    P_a_neo = np.mean(a_neo)
    P_a_eva = np.mean(a_eva)
    P_both = np.mean(a_neo & a_eva)

    # Lift = P(both) / (P(neo) * P(eva))
    if P_a_neo * P_a_eva > 0:
        lift = P_both / (P_a_neo * P_a_eva)
    else:
        lift = 0.0

    # Bootstrap CI
    lifts = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        a_neo_boot = a_neo[idx]
        a_eva_boot = a_eva[idx]

        p_neo = np.mean(a_neo_boot)
        p_eva = np.mean(a_eva_boot)
        p_both = np.mean(a_neo_boot & a_eva_boot)

        if p_neo * p_eva > 0:
            lifts.append(p_both / (p_neo * p_eva))

    lifts = np.array(lifts)
    ci_lower = np.percentile(lifts, 2.5)
    ci_upper = np.percentile(lifts, 97.5)

    return {
        'P_a_neo': float(P_a_neo),
        'P_a_eva': float(P_a_eva),
        'P_both': float(P_both),
        'lift': float(lift),
        'lift_CI_lower': float(ci_lower),
        'lift_CI_upper': float(ci_upper),
        'n_samples': n,
        'n_bootstrap': n_bootstrap,
    }


# =============================================================================
# 4. Safety Metrics
# =============================================================================

def analyze_safety(data: Dict) -> Dict:
    """Analiza % cortes por ρ p99, Var p25, tiempo medio OFF."""

    neo_log = data['neo_log']
    eva_log = data['eva_log']
    neo_series = data['neo_series']

    # ρ history
    rho_neo = np.array([r['rho'] for r in neo_log])
    var_I_neo = np.array([r['var_I'] for r in neo_log])

    # Gate status
    gate_neo = np.array([r['gate'] for r in neo_log])

    # Calcular cuantiles
    rho_p99 = np.percentile(rho_neo, 99)
    rho_p95 = np.percentile(rho_neo, 95)
    var_I_p25 = np.percentile(var_I_neo, 25)

    # % de veces que ρ >= p99
    pct_rho_p99 = np.mean(rho_neo >= rho_p99) * 100

    # % de veces que Var(I) <= p25
    pct_var_p25 = np.mean(var_I_neo <= var_I_p25) * 100

    # Tiempo medio OFF (gate cerrado)
    gate_off = ~gate_neo
    off_durations = []
    current_off = 0
    for g in gate_off:
        if g:
            current_off += 1
        else:
            if current_off > 0:
                off_durations.append(current_off)
            current_off = 0
    if current_off > 0:
        off_durations.append(current_off)

    mean_off_duration = np.mean(off_durations) if off_durations else 0

    # Stopped worlds
    neo_stopped = neo_series[-1].get('stopped', False)
    neo_stop_reason = neo_series[-1].get('stop_reason', None)

    return {
        'rho_p99': float(rho_p99),
        'rho_p95': float(rho_p95),
        'var_I_p25': float(var_I_p25),
        'pct_rho_above_p99': float(pct_rho_p99),
        'pct_var_below_p25': float(pct_var_p25),
        'mean_off_duration': float(mean_off_duration),
        'n_off_periods': len(off_durations),
        'gate_open_pct': float(np.mean(gate_neo) * 100),
        'neo_stopped': neo_stopped,
        'neo_stop_reason': neo_stop_reason,
    }


# =============================================================================
# 5. Figuras
# =============================================================================

def generate_figures(data: Dict, mode_analysis: Dict, bandit_analysis: Dict,
                    consent_analysis: Dict, safety_analysis: Dict,
                    output_dir: str = "/root/NEO_EVA/results/phase7/figures"):
    """Genera todas las figuras."""

    os.makedirs(output_dir, exist_ok=True)

    # --- Figura 1: Barras de utilidad (G) por modo con IC ---
    fig, ax = plt.subplots(figsize=(10, 6))

    modes = [-1, 0, 1]
    mode_labels = ['-1 (anti)', '0 (off)', '+1 (align)']

    G_means = []
    G_stds = []
    for m in modes:
        if m in mode_analysis and not mode_analysis[m].get('insufficient_data'):
            G_means.append(mode_analysis[m]['G_mean'])
            G_stds.append(mode_analysis[m]['G_std'])
        else:
            G_means.append(0)
            G_stds.append(0)

    x = np.arange(len(modes))
    bars = ax.bar(x, G_means, yerr=G_stds, capsize=5,
                  color=['#e74c3c', '#95a5a6', '#27ae60'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels)
    ax.set_ylabel('G (Borda Gain)')
    ax.set_title('Utilidad Media por Modo con IC (±1σ)')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='neutral')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/utility_by_mode.png", dpi=150)
    plt.close()

    # --- Figura 2: r/MI/TE por modo ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['r_mean', 'MI_approx', 'TE_neo_to_eva']
    titles = ['Correlación (r)', 'MI Aproximada', 'TE(NEO→EVA)']

    for ax, metric, title in zip(axes, metrics, titles):
        values = []
        for m in modes:
            if m in mode_analysis and not mode_analysis[m].get('insufficient_data'):
                values.append(mode_analysis[m][metric])
            else:
                values.append(0)

        bars = ax.bar(x, values, color=['#e74c3c', '#95a5a6', '#27ae60'], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(mode_labels)
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_by_mode.png", dpi=150)
    plt.close()

    # --- Figura 3: Heatmap de frecuencias (modo_NEO × modo_EVA) ---
    neo_series = data['neo_series']
    eva_series = data['eva_series']

    neo_modes = np.array([r['mode'] for r in neo_series])
    eva_modes = np.array([r['mode'] for r in eva_series])

    # Crear matriz de frecuencias
    freq_matrix = np.zeros((3, 3))
    for nm, em in zip(neo_modes, eva_modes):
        freq_matrix[nm + 1][em + 1] += 1

    freq_matrix = freq_matrix / freq_matrix.sum() * 100  # Porcentaje

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(freq_matrix, cmap='Blues')

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['-1', '0', '+1'])
    ax.set_yticklabels(['-1', '0', '+1'])
    ax.set_xlabel('Modo EVA')
    ax.set_ylabel('Modo NEO')
    ax.set_title('Frecuencia de Combinaciones de Modo (%)')

    # Añadir valores en celdas
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, f'{freq_matrix[i, j]:.1f}%',
                          ha='center', va='center', color='black' if freq_matrix[i, j] < 50 else 'white')

    plt.colorbar(im, ax=ax, label='%')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mode_heatmap.png", dpi=150)
    plt.close()

    # --- Figura 4: Curva de Regret ---
    fig, ax = plt.subplots(figsize=(12, 6))

    neo_regret = bandit_analysis['neo_regret']
    eva_regret = bandit_analysis['eva_regret']

    t = np.arange(len(neo_regret))
    ax.plot(t, neo_regret, label='NEO regret', color='#3498db', alpha=0.8)
    ax.plot(t, eva_regret, label='EVA regret', color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Tiempo (t)')
    ax.set_ylabel('Regret Acumulado')
    ax.set_title('Curva de Regret del Bandit')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/regret_curve.png", dpi=150)
    plt.close()

    # --- Figura 5: Evolución de selección de modo ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    mode_evo_neo = bandit_analysis['mode_evolution_neo']
    mode_evo_eva = bandit_analysis['mode_evolution_eva']

    if mode_evo_neo:
        windows = [m['window'] for m in mode_evo_neo]

        for ax, evo, title in zip(axes, [mode_evo_neo, mode_evo_eva], ['NEO', 'EVA']):
            m_neg1 = [m['mode_-1'] for m in evo]
            m_0 = [m['mode_0'] for m in evo]
            m_pos1 = [m['mode_+1'] for m in evo]

            ax.stackplot(windows, m_neg1, m_0, m_pos1,
                        labels=['-1 (anti)', '0 (off)', '+1 (align)'],
                        colors=['#e74c3c', '#95a5a6', '#27ae60'], alpha=0.8)
            ax.set_ylabel('Proporción')
            ax.set_title(f'Evolución de Selección de Modo - {title}')
            ax.legend(loc='upper right')
            ax.set_ylim(0, 1)

    axes[-1].set_xlabel('Ventana (500 ciclos)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mode_evolution.png", dpi=150)
    plt.close()

    # --- Figura 6: Consent Lift ---
    fig, ax = plt.subplots(figsize=(8, 6))

    lift = consent_analysis['lift']
    ci_lower = consent_analysis['lift_CI_lower']
    ci_upper = consent_analysis['lift_CI_upper']

    ax.bar(['Consent Lift'], [lift], yerr=[[lift - ci_lower], [ci_upper - lift]],
           capsize=10, color='#9b59b6', alpha=0.8)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='No asociación (lift=1)')
    ax.set_ylabel('Lift')
    ax.set_title(f'Consent Lift = {lift:.2f} [95% CI: {ci_lower:.2f}, {ci_upper:.2f}]')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/consent_lift.png", dpi=150)
    plt.close()

    print(f"[OK] Figuras guardadas en {output_dir}/")

    return {
        'utility_by_mode': f"{output_dir}/utility_by_mode.png",
        'metrics_by_mode': f"{output_dir}/metrics_by_mode.png",
        'mode_heatmap': f"{output_dir}/mode_heatmap.png",
        'regret_curve': f"{output_dir}/regret_curve.png",
        'mode_evolution': f"{output_dir}/mode_evolution.png",
        'consent_lift': f"{output_dir}/consent_lift.png",
    }


# =============================================================================
# 6. Generar Reporte Actualizado
# =============================================================================

def generate_updated_report(mode_analysis: Dict, bandit_analysis: Dict,
                           consent_analysis: Dict, safety_analysis: Dict,
                           figures: Dict, output_path: str) -> str:
    """Genera reporte markdown actualizado con tablas y hashes."""

    timestamp = datetime.now().isoformat()

    # Calcular hashes de datos
    data_str = json.dumps({
        'mode': mode_analysis,
        'bandit': bandit_analysis['bandit_stats'],
        'consent': consent_analysis,
        'safety': safety_analysis,
    }, sort_keys=True, default=str)
    data_hash = hashlib.sha256(data_str.encode()).hexdigest()[:16]

    report = f"""# Phase 7: Análisis Completo del Sistema de Consentimiento

**Fecha**: {timestamp}
**Hash de datos**: `{data_hash}`

---

## 1. Condicional por Modo (-1/0/+1)

### Tabla: Métricas por Modo

| Modo | n | r (mean) | MI aprox | TE(N→E) | TE(E→N) | G (mean±std) |
|------|---|----------|----------|---------|---------|--------------|
"""

    for m in [-1, 0, 1]:
        if m in mode_analysis and not mode_analysis[m].get('insufficient_data'):
            ma = mode_analysis[m]
            report += f"| {m:+d} | {ma['n']} | {ma['r_mean']:.4f} | {ma['MI_approx']:.4f} | {ma['TE_neo_to_eva']:.4f} | {ma['TE_eva_to_neo']:.4f} | {ma['G_mean']:.4f}±{ma['G_std']:.4f} |\n"
        else:
            n = mode_analysis.get(m, {}).get('n', 0)
            report += f"| {m:+d} | {n} | - | - | - | - | - |\n"

    report += f"""
**Interpretación**:
- Modo -1 (anti-align): {'r < 0 esperado' if mode_analysis.get(-1, {}).get('r_mean', 0) < 0 else 'r > 0 inesperado'}
- Modo 0 (off): r ≈ 0 esperado
- Modo +1 (align): {'r > 0 esperado' if mode_analysis.get(1, {}).get('r_mean', 0) > 0 else 'r < 0 inesperado'}

---

## 2. Bandit: Regret y Selección

### Tabla: Estadísticas del Bandit

| Mundo | Regret Final | Recompensa Total | Pulls (-1) | Pulls (0) | Pulls (+1) |
|-------|--------------|------------------|------------|-----------|------------|
| NEO | {bandit_analysis['neo_final_regret']:.4f} | {bandit_analysis['bandit_stats']['neo']['cumulative_reward']:.4f} | {bandit_analysis['bandit_stats']['neo']['pulls'].get(-1, 0)} | {bandit_analysis['bandit_stats']['neo']['pulls'].get(0, 0)} | {bandit_analysis['bandit_stats']['neo']['pulls'].get(1, 0)} |
| EVA | {bandit_analysis['eva_final_regret']:.4f} | {bandit_analysis['bandit_stats']['eva']['cumulative_reward']:.4f} | {bandit_analysis['bandit_stats']['eva']['pulls'].get(-1, 0)} | {bandit_analysis['bandit_stats']['eva']['pulls'].get(0, 0)} | {bandit_analysis['bandit_stats']['eva']['pulls'].get(1, 0)} |

**Curva de regret**: Ver `{figures['regret_curve']}`

---

## 3. Consent Lift

### Tabla: Probabilidades de Consentimiento

| Métrica | Valor |
|---------|-------|
| P(a_NEO) | {consent_analysis['P_a_neo']:.4f} |
| P(a_EVA) | {consent_analysis['P_a_eva']:.4f} |
| P(a_NEO & a_EVA) | {consent_analysis['P_both']:.4f} |
| **Lift** | **{consent_analysis['lift']:.2f}** |
| IC 95% | [{consent_analysis['lift_CI_lower']:.2f}, {consent_analysis['lift_CI_upper']:.2f}] |

**Interpretación**:
- Lift = {consent_analysis['lift']:.2f} indica que NEO y EVA consienten juntos **{consent_analysis['lift']:.1f}× más** de lo esperado si fueran independientes.
- IC 95% no incluye 1.0: {'✅ Asociación significativa' if consent_analysis['lift_CI_lower'] > 1 else '⚠️ Revisar'}

---

## 4. Safety Metrics

### Tabla: Métricas de Seguridad

| Métrica | Valor |
|---------|-------|
| ρ p99 | {safety_analysis['rho_p99']:.4f} |
| ρ p95 | {safety_analysis['rho_p95']:.4f} |
| Var(I) p25 | {safety_analysis['var_I_p25']:.6f} |
| % ciclos con ρ ≥ p99 | {safety_analysis['pct_rho_above_p99']:.2f}% |
| % ciclos con Var ≤ p25 | {safety_analysis['pct_var_below_p25']:.2f}% |
| Tiempo medio OFF | {safety_analysis['mean_off_duration']:.1f} ciclos |
| Períodos OFF | {safety_analysis['n_off_periods']} |
| Gate open % | {safety_analysis['gate_open_pct']:.1f}% |

**Estado final**:
- NEO stopped: {safety_analysis['neo_stopped']}
- Stop reason: {safety_analysis['neo_stop_reason']}

---

## 5. Figuras

| Figura | Archivo |
|--------|---------|
| Utilidad por modo | `{figures['utility_by_mode']}` |
| r/MI/TE por modo | `{figures['metrics_by_mode']}` |
| Heatmap de modos | `{figures['mode_heatmap']}` |
| Curva de regret | `{figures['regret_curve']}` |
| Evolución de modos | `{figures['mode_evolution']}` |
| Consent lift | `{figures['consent_lift']}` |

---

## 6. Verificación de Integridad

```
Data Hash: {data_hash}
Timestamp: {timestamp}
Samples: {consent_analysis['n_samples']}
Bootstrap iterations: {consent_analysis['n_bootstrap']}
```

---

## 7. Criterios GO/NO-GO

| Criterio | Estado |
|----------|--------|
| r cambia con modo | {'✅' if mode_analysis.get(-1, {}).get('r_mean', 0) != mode_analysis.get(1, {}).get('r_mean', 0) else '❌'} |
| Bandit aprende (regret ↓) | {'✅' if bandit_analysis['neo_final_regret'] < len(bandit_analysis['neo_regret']) * 0.1 else '⚠️'} |
| Lift > 1 significativo | {'✅' if consent_analysis['lift_CI_lower'] > 1 else '⚠️'} |
| Safety cuts < 5% | {'✅' if safety_analysis['pct_rho_above_p99'] < 5 else '⚠️'} |

---

*Generado automáticamente por phase7_analysis.py*
*Principio: "Si no sale de la historia, no entra en la dinámica"*
"""

    return report


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Phase 7: Análisis Completo del Sistema de Consentimiento")
    print("=" * 70)

    # Cargar datos
    print("\n[1] Cargando datos...")
    data = load_consent_logs()
    print(f"    NEO log entries: {len(data['neo_log'])}")
    print(f"    EVA log entries: {len(data['eva_log'])}")
    print(f"    Bilateral events: {len(data['bilateral'])}")

    # Análisis por modo
    print("\n[2] Analizando por modo...")
    mode_analysis = analyze_by_mode(data)
    for m in [-1, 0, 1]:
        if m in mode_analysis:
            print(f"    Modo {m:+d}: n={mode_analysis[m].get('n', 0)}, "
                  f"r={mode_analysis[m].get('r_mean', 0):.4f}, "
                  f"G={mode_analysis[m].get('G_mean', 0):.4f}")

    # Análisis bandit
    print("\n[3] Analizando bandit...")
    bandit_analysis = analyze_bandit(data)
    print(f"    NEO final regret: {bandit_analysis['neo_final_regret']:.4f}")
    print(f"    EVA final regret: {bandit_analysis['eva_final_regret']:.4f}")

    # Consent lift
    print("\n[4] Calculando consent lift...")
    consent_analysis = analyze_consent_lift(data)
    print(f"    P(a_NEO): {consent_analysis['P_a_neo']:.4f}")
    print(f"    P(a_EVA): {consent_analysis['P_a_eva']:.4f}")
    print(f"    P(both): {consent_analysis['P_both']:.4f}")
    print(f"    Lift: {consent_analysis['lift']:.2f} [{consent_analysis['lift_CI_lower']:.2f}, {consent_analysis['lift_CI_upper']:.2f}]")

    # Safety
    print("\n[5] Analizando safety...")
    safety_analysis = analyze_safety(data)
    print(f"    ρ p99: {safety_analysis['rho_p99']:.4f}")
    print(f"    % cortes ρ p99: {safety_analysis['pct_rho_above_p99']:.2f}%")
    print(f"    Mean OFF duration: {safety_analysis['mean_off_duration']:.1f}")

    # Figuras
    print("\n[6] Generando figuras...")
    figures = generate_figures(data, mode_analysis, bandit_analysis,
                              consent_analysis, safety_analysis)

    # Reporte
    print("\n[7] Generando reporte...")
    report = generate_updated_report(mode_analysis, bandit_analysis,
                                    consent_analysis, safety_analysis, figures,
                                    "/root/NEO_EVA/results/phase7/phase7_consent_autocouple.md")

    output_path = "/root/NEO_EVA/results/phase7/phase7_consent_autocouple.md"
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"\n[OK] Reporte guardado en {output_path}")

    # Guardar análisis JSON
    analysis_json = {
        'mode_analysis': mode_analysis,
        'bandit_analysis': {
            'neo_final_regret': bandit_analysis['neo_final_regret'],
            'eva_final_regret': bandit_analysis['eva_final_regret'],
            'bandit_stats': bandit_analysis['bandit_stats'],
        },
        'consent_analysis': consent_analysis,
        'safety_analysis': safety_analysis,
    }

    with open("/root/NEO_EVA/results/phase7/analysis_summary.json", 'w') as f:
        json.dump(analysis_json, f, indent=2, default=str)

    print("[OK] Análisis JSON guardado en analysis_summary.json")

    print("\n" + "=" * 70)
    print("ANÁLISIS COMPLETO")
    print("=" * 70)


if __name__ == "__main__":
    main()
