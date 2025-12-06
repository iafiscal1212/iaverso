#!/usr/bin/env python3
"""
Test Explorer v2 - 100% Endógeno
================================
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/root/NEO_EVA')

from core.explorer_agent_v2 import ExplorerAgentV2, create_explorer


def load_data() -> pd.DataFrame:
    """Carga datos unificados más recientes."""
    data_dir = Path('/root/NEO_EVA/data')
    files = list(data_dir.glob('unified_*.csv'))
    latest = max(files, key=lambda f: f.stat().st_mtime)
    print(f"Datos: {latest}")
    return pd.read_csv(latest, index_col=0, parse_dates=True)


def run_test():
    print("=" * 70)
    print("EXPLORER v2 - 100% ENDÓGENO")
    print("=" * 70)
    print()

    df = load_data()
    variables = list(df.columns)

    print(f"Variables: {len(variables)}")
    print(f"Timestamps: {len(df)}")
    print()

    # Crear 5 agentes
    agents = {name: create_explorer(name, variables)
              for name in ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']}

    print("PERSONALIDADES (derivadas del hash del nombre):")
    print("-" * 50)
    for name, agent in agents.items():
        print(f"  {name}: exploration_base={agent._base_exploration:.3f}")
    print()

    print("EXPLORANDO...")
    print("-" * 50)

    results = []

    for t in range(len(df)):
        row = df.iloc[t]
        observation = {k: v for k, v in row.items() if not pd.isna(v)}

        for name, agent in agents.items():
            step_result = agent.observe(observation)
            results.append(step_result)

        if (t + 1) % 30 == 0:
            ce_strs = [f"{name}:{agents[name].CE_history[-1]:.3f}" for name in agents]
            hyp_strs = [f"{name}:{len(agents[name].world_model.hypotheses)}" for name in agents]
            disc_strs = [f"{name}:{len(agents[name].discoveries)}" for name in agents]
            print(f"  t={t+1:3d}")
            print(f"    CE: {' '.join(ce_strs)}")
            print(f"    Hip: {' '.join(hyp_strs)}")
            print(f"    Desc: {' '.join(disc_strs)}")

    print()
    print("=" * 70)
    print("RESULTADOS FINALES")
    print("=" * 70)
    print()

    # Por agente
    for name, agent in agents.items():
        status = agent.get_status()
        print(f"{name}:")
        print(f"  CE = {status['CE']:.4f}")
        print(f"  Hipótesis = {status['n_hypotheses']} (significativas: {status['n_significant']})")
        print(f"  Descubrimientos = {status['n_discoveries']}")
        print(f"  Curiosidad = {status['emotions']['curiosity']:.3f}")
        print(f"  Confianza = {status['emotions']['confidence']:.3f}")
        print()

    # Todos los descubrimientos
    print("=" * 70)
    print("DESCUBRIMIENTOS")
    print("=" * 70)
    print()

    all_discoveries = []
    for name, agent in agents.items():
        for d in agent.get_discoveries():
            d['agent'] = name
            all_discoveries.append(d)

    # Ordenar por confianza
    all_discoveries.sort(key=lambda x: x['confidence'], reverse=True)

    # Agrupar por tipo de relación
    cross_domain = []
    same_domain = []

    domains = {
        'crypto': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT'],
        'solar': ['solar_', 'geomag'],
        'climate': ['climate_'],
        'seismic': ['seismic_'],
    }

    def get_domain(var):
        for domain, patterns in domains.items():
            for pattern in patterns:
                if pattern in var:
                    return domain
        return 'other'

    for d in all_discoveries:
        src_domain = get_domain(d['source'])
        tgt_domain = get_domain(d['target'])

        if src_domain != tgt_domain:
            cross_domain.append(d)
        else:
            same_domain.append(d)

    print("CROSS-DOMAIN (los más interesantes):")
    print("-" * 50)
    for d in cross_domain[:20]:
        print(f"  [{d['agent']}] {d['source']} -> {d['target']} (lag={d['lag']})")
        print(f"         corr={d['correlation']:.3f}, conf={d['confidence']:.2f}, "
              f"éxito={d['success_rate']:.2f}, tests={d['n_tests']}")

    print()
    print(f"SAME-DOMAIN: {len(same_domain)} descubrimientos")
    print(f"CROSS-DOMAIN: {len(cross_domain)} descubrimientos")

    # Top hipótesis por agente
    print()
    print("=" * 70)
    print("TOP HIPÓTESIS POR AGENTE")
    print("=" * 70)

    for name, agent in agents.items():
        top = agent.get_top_hypotheses(5)
        if top:
            print(f"\n{name}:")
            for h in top:
                print(f"  {h.source} -> {h.target} (lag={h.lag})")
                print(f"    corr={h.strength:.3f}, conf={h.confidence:.2f}, "
                      f"éxito={h.success_rate:.2f}, tests={h.n_tests}")

    # Guardar
    output_dir = Path('/root/NEO_EVA/logs/explorer_v2')
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convertir numpy types a Python nativos
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        return obj

    with open(output_dir / f'results_{ts}.jsonl', 'w') as f:
        for r in results:
            f.write(json.dumps(convert_numpy(r)) + '\n')

    with open(output_dir / f'discoveries_{ts}.json', 'w') as f:
        json.dump(convert_numpy(all_discoveries), f, indent=2)

    print(f"\nGuardado en: {output_dir}")

    return agents, all_discoveries


if __name__ == '__main__':
    run_test()
