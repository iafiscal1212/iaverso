#!/usr/bin/env python3
"""
Phase 14: Objetivos Emergentes con Datos Reales
===============================================

Ejecuta el sistema de objetivos emergentes sobre datos de Phase 12/13.
100% endógeno.
"""

import json
import numpy as np
import sys
sys.path.insert(0, '/root/NEO_EVA/tools')

from narrative import NarrativeSystem
from emergent_objectives import (
    EmergentObjectiveSystem,
    compute_narrative_tension,
    compute_tension_history,
    find_narrative_attractors,
    derive_proto_preferences
)
from endogenous_core import derive_window_size

import matplotlib.pyplot as plt


def load_data():
    """Carga datos reales."""
    print("[1] Cargando datos...")

    with open('/root/NEO_EVA/results/phase12/pi_log_neo.json') as f:
        pi_log_neo = json.load(f)
    with open('/root/NEO_EVA/results/phase12/pi_log_eva.json') as f:
        pi_log_eva = json.load(f)
    with open('/root/NEO_EVA/results/phase12/bilateral_events.json') as f:
        bilateral_events = json.load(f)

    pi_neo = np.array([p['pi'] for p in pi_log_neo])
    pi_eva = np.array([p['pi'] for p in pi_log_eva])

    n = len(pi_neo)
    print(f"    Ciclos: {n}")

    gw_intensity = np.zeros(n)
    gw_active = np.zeros(n, dtype=bool)
    for e in bilateral_events:
        t = e['t']
        if 0 < t <= n:
            gw_intensity[t-1] = e.get('intensity', 0)
            gw_active[t-1] = True

    states = []
    for t in range(n):
        hour = t % 24
        if hour < 6:
            states.append('SLEEP')
        elif hour < 12:
            states.append('WORK')
        elif hour < 18:
            states.append('SOCIAL')
        else:
            states.append('WAKE')

    window = derive_window_size(n)
    te_neo_to_eva = np.zeros(n)
    te_eva_to_neo = np.zeros(n)

    for t in range(window, n):
        start = t - window
        corr = np.corrcoef(pi_neo[start:t], pi_eva[start:t])[0, 1]
        if not np.isnan(corr):
            intensity = gw_intensity[start:t].mean()
            te_neo_to_eva[t] = max(0, corr * intensity * 0.5)
            te_eva_to_neo[t] = max(0, corr * intensity * 0.5)

    self_error_neo = np.zeros(n)
    self_error_eva = np.zeros(n)
    for t in range(1, n):
        self_error_neo[t] = abs(pi_neo[t] - pi_neo[t-1])
        self_error_eva[t] = abs(pi_eva[t] - pi_eva[t-1])

    return {
        'pi_neo': pi_neo,
        'pi_eva': pi_eva,
        'te_neo_to_eva': te_neo_to_eva,
        'te_eva_to_neo': te_eva_to_neo,
        'states': states,
        'gw_active': gw_active,
        'gw_intensity': gw_intensity,
        'self_error_neo': self_error_neo,
        'self_error_eva': self_error_eva,
        'n': n
    }


def run_full_system(data):
    """Ejecuta sistema narrativo + objetivos."""
    print("\n[2] Ejecutando sistema narrativo...")

    ns = NarrativeSystem()
    n = data['n']

    report_interval = n // 10
    for t in range(n):
        if t > 0 and t % report_interval == 0:
            print(f"    Progreso: {t}/{n} ({100*t//n}%)")

        ns.process_step(
            t=t,
            neo_pi=data['pi_neo'][t],
            eva_pi=data['pi_eva'][t],
            te_neo_to_eva=data['te_neo_to_eva'][t],
            te_eva_to_neo=data['te_eva_to_neo'][t],
            neo_state=data['states'][t],
            eva_state=data['states'][t],
            gw_active=data['gw_active'][t],
            gw_intensity=data['gw_intensity'][t],
            neo_self_error=data['self_error_neo'][t],
            eva_self_error=data['self_error_eva'][t]
        )

    print(f"    Episodios NEO: {len(ns.neo_memory.episodes)}")
    print(f"    Episodios EVA: {len(ns.eva_memory.episodes)}")

    print("\n[3] Analizando objetivos emergentes...")
    eos = EmergentObjectiveSystem(ns)

    # Actualizar en intervalos
    window = derive_window_size(n)
    for t in range(0, n, window // 4):
        eos.update(t)

    return ns, eos


def analyze_results(ns, eos):
    """Analiza resultados."""
    print("\n[4] Análisis de resultados...")

    summary = eos.get_summary()

    print(f"\n{'='*60}")
    print("TENSIÓN NARRATIVA")
    print(f"{'='*60}")
    print(f"  NEO: {summary['neo']['mean_tension']:.3f} (tendencia: {summary['neo']['tension_trend']})")
    print(f"  EVA: {summary['eva']['mean_tension']:.3f} (tendencia: {summary['eva']['tension_trend']})")

    print(f"\n{'='*60}")
    print("ATRACTORES NARRATIVOS")
    print(f"{'='*60}")
    print("  NEO:")
    for a in summary['neo']['top_attractors'][:3]:
        print(f"    Secuencia {a['sequence']}: freq={a['frequency']}, estabilidad={a['stability']:.4f}")
    print("  EVA:")
    for a in summary['eva']['top_attractors'][:3]:
        print(f"    Secuencia {a['sequence']}: freq={a['frequency']}, estabilidad={a['stability']:.4f}")

    print(f"\n{'='*60}")
    print("PROTO-PREFERENCIAS EMERGENTES")
    print(f"{'='*60}")
    print("  NEO:")
    for p in summary['neo']['preferences']:
        print(f"    {p['direction'].upper()} {p['name']}: fuerza={p['strength']:.3f}")
        if 'state' in p['evidence']:
            print(f"      → Estado preferido con TE={p['evidence'].get('mean_te', 0):.3f}")
        if 'role' in p['evidence']:
            print(f"      → Rol con mejor outcome TE={p['evidence'].get('outcome_te', 0):.3f}")

    print("  EVA:")
    for p in summary['eva']['preferences']:
        print(f"    {p['direction'].upper()} {p['name']}: fuerza={p['strength']:.3f}")
        if 'state' in p['evidence']:
            print(f"      → Estado preferido con TE={p['evidence'].get('mean_te', 0):.3f}")
        if 'role' in p['evidence']:
            print(f"      → Rol con mejor outcome TE={p['evidence'].get('outcome_te', 0):.3f}")

    print(f"\n{'='*60}")
    print("CORRELACIÓN DE PREFERENCIAS")
    print(f"{'='*60}")
    corr = summary['preference_correlation']
    if corr > 0.5:
        interpretation = "ALINEADAS (buscan lo mismo)"
    elif corr < -0.5:
        interpretation = "OPUESTAS (buscan cosas diferentes)"
    else:
        interpretation = "INDEPENDIENTES (cada uno su camino)"
    print(f"  Correlación: {corr:.3f} - {interpretation}")

    return summary


def plot_results(eos, output_dir='/root/NEO_EVA/figures'):
    """Genera visualizaciones."""
    print("\n[5] Generando visualizaciones...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Tensión a lo largo del tiempo
    ax = axes[0, 0]
    if eos.neo_tension_history:
        ax.plot(eos.neo_tension_history, label='NEO', color='#E74C3C', alpha=0.7)
        ax.plot(eos.eva_tension_history, label='EVA', color='#3498DB', alpha=0.7)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Narrative Tension')
        ax.set_title('Tension Evolution')
        ax.legend()
        ax.set_ylim(0, 1)

    # 2. Preferencias comparadas
    ax = axes[0, 1]
    neo_prefs = {p.name: p.strength for p in eos.neo_preferences}
    eva_prefs = {p.name: p.strength for p in eos.eva_preferences}
    all_prefs = list(set(neo_prefs.keys()) | set(eva_prefs.keys()))

    if all_prefs:
        x = np.arange(len(all_prefs))
        width = 0.35
        neo_vals = [neo_prefs.get(p, 0) for p in all_prefs]
        eva_vals = [eva_prefs.get(p, 0) for p in all_prefs]

        ax.barh(x - width/2, neo_vals, width, label='NEO', color='#E74C3C', alpha=0.8)
        ax.barh(x + width/2, eva_vals, width, label='EVA', color='#3498DB', alpha=0.8)
        ax.set_yticks(x)
        ax.set_yticklabels([p.replace('_', '\n') for p in all_prefs], fontsize=8)
        ax.set_xlabel('Preference Strength')
        ax.set_title('Proto-Preferences Comparison')
        ax.legend()

    # 3. Atractores
    ax = axes[1, 0]
    neo_attract = [(str(a.type_sequence), a.stability_score) for a in eos.neo_attractors[:5]]
    eva_attract = [(str(a.type_sequence), a.stability_score) for a in eos.eva_attractors[:5]]

    if neo_attract or eva_attract:
        labels = []
        neo_vals = []
        eva_vals = []

        all_seqs = list(set([a[0] for a in neo_attract] + [a[0] for a in eva_attract]))[:6]
        neo_dict = dict(neo_attract)
        eva_dict = dict(eva_attract)

        for seq in all_seqs:
            labels.append(seq[:15])
            neo_vals.append(neo_dict.get(seq, 0))
            eva_vals.append(eva_dict.get(seq, 0))

        x = np.arange(len(labels))
        ax.bar(x - 0.2, neo_vals, 0.4, label='NEO', color='#E74C3C', alpha=0.8)
        ax.bar(x + 0.2, eva_vals, 0.4, label='EVA', color='#3498DB', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Stability Score')
        ax.set_title('Top Narrative Attractors')
        ax.legend()

    # 4. Evolución de preferencias
    ax = axes[1, 1]
    if eos.preference_evolution:
        ts = [p['t'] for p in eos.preference_evolution]

        neo_stability = []
        eva_stability = []
        for p in eos.preference_evolution:
            neo_stab = sum(pref['strength'] for pref in p['neo']
                         if 'stability' in pref['name']) if p['neo'] else 0
            eva_stab = sum(pref['strength'] for pref in p['eva']
                         if 'stability' in pref['name']) if p['eva'] else 0
            neo_stability.append(neo_stab)
            eva_stability.append(eva_stab)

        ax.plot(ts, neo_stability, label='NEO stability-seeking', color='#E74C3C', alpha=0.8)
        ax.plot(ts, eva_stability, label='EVA stability-seeking', color='#3498DB', alpha=0.8)
        ax.set_xlabel('Time')
        ax.set_ylabel('Stability Preference')
        ax.set_title('Preference Evolution')
        ax.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/phase14_objectives_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Guardado: {output_dir}/phase14_objectives_analysis.png")

    # Figura LinkedIn
    fig, ax = plt.subplots(figsize=(10, 6))

    if eos.neo_preferences or eos.eva_preferences:
        neo_prefs = {p.name: p.strength for p in eos.neo_preferences}
        eva_prefs = {p.name: p.strength for p in eos.eva_preferences}
        all_prefs = list(set(neo_prefs.keys()) | set(eva_prefs.keys()))

        x = np.arange(len(all_prefs))
        width = 0.35

        bars1 = ax.barh(x - width/2, [neo_prefs.get(p, 0) for p in all_prefs],
                       width, label='NEO', color='#E74C3C', alpha=0.85)
        bars2 = ax.barh(x + width/2, [eva_prefs.get(p, 0) for p in all_prefs],
                       width, label='EVA', color='#3498DB', alpha=0.85)

        ax.set_yticks(x)
        labels_clean = []
        for p in all_prefs:
            if 'state_' in p:
                labels_clean.append(f"Prefer {p.replace('state_', '')} state")
            elif 'role_' in p:
                labels_clean.append(f"Prefer {p.replace('role_', '')} role")
            elif 'stability' in p:
                labels_clean.append("Seek narrative stability")
            elif 'attractor' in p:
                labels_clean.append("Follow familiar patterns")
            else:
                labels_clean.append(p)
        ax.set_yticklabels(labels_clean, fontsize=11)

        ax.set_xlabel('Preference Strength', fontsize=12)
        ax.set_title('Emergent Proto-Preferences\nDerived from Narrative History (No External Rewards)',
                    fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/phase14_preferences_linkedin.png', dpi=250,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Guardado: {output_dir}/phase14_preferences_linkedin.png")


def main():
    print("=" * 70)
    print("PHASE 14: OBJETIVOS EMERGENTES - DATOS REALES")
    print("=" * 70)

    data = load_data()
    ns, eos = run_full_system(data)
    summary = analyze_results(ns, eos)
    plot_results(eos)

    print("\n[6] Guardando resultados...")
    eos.save('/root/NEO_EVA/results/phase14_objectives_real.json')

    # Resumen
    results_summary = {
        'neo_mean_tension': summary['neo']['mean_tension'],
        'eva_mean_tension': summary['eva']['mean_tension'],
        'neo_tension_trend': summary['neo']['tension_trend'],
        'eva_tension_trend': summary['eva']['tension_trend'],
        'neo_n_attractors': len(summary['neo']['top_attractors']),
        'eva_n_attractors': len(summary['eva']['top_attractors']),
        'neo_n_preferences': len(summary['neo']['preferences']),
        'eva_n_preferences': len(summary['eva']['preferences']),
        'preference_correlation': summary['preference_correlation'],
        'neo_top_preference': summary['neo']['preferences'][0]['name'] if summary['neo']['preferences'] else None,
        'eva_top_preference': summary['eva']['preferences'][0]['name'] if summary['eva']['preferences'] else None
    }

    with open('/root/NEO_EVA/results/phase14_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"    Guardado: results/phase14_objectives_real.json")
    print(f"    Guardado: results/phase14_summary.json")

    print("\n" + "=" * 70)
    print("RESUMEN PHASE 14")
    print("=" * 70)
    print(f"  Tensión media: NEO={summary['neo']['mean_tension']:.3f}, EVA={summary['eva']['mean_tension']:.3f}")
    print(f"  Atractores: NEO={len(summary['neo']['top_attractors'])}, EVA={len(summary['eva']['top_attractors'])}")
    print(f"  Preferencias: NEO={len(summary['neo']['preferences'])}, EVA={len(summary['eva']['preferences'])}")
    print(f"  Correlación preferencias: {summary['preference_correlation']:.3f}")
    if summary['neo']['preferences']:
        print(f"  Top preferencia NEO: {summary['neo']['preferences'][0]['name']}")
    if summary['eva']['preferences']:
        print(f"  Top preferencia EVA: {summary['eva']['preferences'][0]['name']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
