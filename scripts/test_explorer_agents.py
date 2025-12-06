#!/usr/bin/env python3
"""
Test: 5 Agentes Explorando el Mundo Real
========================================

Cada agente (NEO, EVA, ALEX, ADAM, IRIS) explora datos reales
con su propia personalidad y descubre estructura causal.

Datos: cripto, solar, clima, sismos (última semana)
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/root/NEO_EVA')

from core.explorer_agent import ExplorerAgent, create_agent, PERSONALITIES


def load_unified_data() -> pd.DataFrame:
    """Carga el dataset unificado más reciente."""
    data_dir = Path('/root/NEO_EVA/data')
    files = list(data_dir.glob('unified_*.csv'))

    if not files:
        raise FileNotFoundError("No unified data found. Run world_data_collector.py first.")

    latest = max(files, key=lambda f: f.stat().st_mtime)
    print(f"Cargando: {latest}")

    df = pd.read_csv(latest, index_col=0, parse_dates=True)
    return df


def run_exploration(n_steps: int = None):
    """
    Ejecuta exploración con los 5 agentes.
    """
    print("=" * 70)
    print("5 AGENTES EXPLORANDO EL MUNDO REAL")
    print("=" * 70)
    print()

    # Cargar datos
    df = load_unified_data()
    variables = list(df.columns)

    print(f"Variables: {len(variables)}")
    print(f"Timestamps: {len(df)}")
    print()

    # Limitar steps si no se especifica
    if n_steps is None:
        n_steps = len(df)

    # Crear agentes
    agents = {name: create_agent(name, variables) for name in PERSONALITIES.keys()}

    print("PERSONALIDADES:")
    print("-" * 50)
    for name, agent in agents.items():
        p = agent.personality
        pref = p['domain_preference'] or 'all'
        print(f"  {name}: curiosidad={p['curiosity_base']:.1f}, "
              f"riesgo={p['risk_tolerance']:.1f}, dominio={pref}")
    print()

    # Simular exploración
    print("EXPLORANDO...")
    print("-" * 50)

    results = []

    for t in range(min(n_steps, len(df))):
        # Obtener observación actual
        row = df.iloc[t]
        observation = row.to_dict()

        # Cada agente observa
        for name, agent in agents.items():
            step_result = agent.observe(observation)
            step_result['timestamp'] = str(df.index[t])
            results.append(step_result)

        # Log cada 20 pasos
        if (t + 1) % 20 == 0:
            ce_strs = [f"{name}:{agents[name].CE_history[-1]:.3f}" for name in agents]
            hyp_strs = [f"{name}:{len(agents[name].world_model.hypotheses)}" for name in agents]
            print(f"  t={t+1:3d} | CE: {' '.join(ce_strs)}")
            print(f"        | Hipótesis: {' '.join(hyp_strs)}")

    print()
    print("=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print()

    # Resumen por agente
    print("CE FINAL POR AGENTE:")
    print("-" * 40)
    for name, agent in agents.items():
        status = agent.get_status()
        print(f"  {name}:")
        print(f"    CE = {status['CE']:.4f}")
        print(f"    Hipótesis = {status['n_hypotheses']} (confiables: {status['n_confident']})")
        print(f"    Descubrimientos = {status['n_discoveries']}")
        print(f"    Emoción dominante = {status['dominant_emotion']}")
        print()

    # Descubrimientos
    print("DESCUBRIMIENTOS (hipótesis confirmadas):")
    print("-" * 50)
    all_discoveries = []
    for name, agent in agents.items():
        for d in agent.get_discoveries():
            d['agent'] = name
            all_discoveries.append(d)

    if all_discoveries:
        # Ordenar por confianza
        all_discoveries.sort(key=lambda x: x['confidence'], reverse=True)
        for d in all_discoveries[:15]:  # Top 15
            print(f"  [{d['agent']}] {d['hypothesis']}")
            print(f"         confianza={d['confidence']:.2f}, éxito={d['success_rate']:.2f}")
    else:
        print("  (ninguno aún - necesita más datos)")

    print()

    # Mejores hipótesis por agente
    print("TOP HIPÓTESIS POR AGENTE:")
    print("-" * 50)
    for name, agent in agents.items():
        best = agent.get_best_hypotheses(3)
        if best:
            print(f"  {name}:")
            for h in best:
                print(f"    {h.source} -> {h.target} (lag={h.lag})")
                print(f"      conf={h.confidence:.2f}, éxito={h.success_rate:.2f}, tests={h.total_tests}")
        else:
            print(f"  {name}: (sin hipótesis aún)")
        print()

    # Guardar resultados
    output_dir = Path('/root/NEO_EVA/logs/explorer')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f'exploration_{timestamp}.jsonl'

    with open(results_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f"Resultados guardados: {results_file}")

    # Emociones finales
    print()
    print("ESTADO EMOCIONAL FINAL:")
    print("-" * 50)
    for name, agent in agents.items():
        e = agent.emotions
        print(f"  {name}: curiosidad={e.curiosity:.2f}, sorpresa={e.surprise:.2f}, "
              f"confianza={e.confidence:.2f}, confusión={e.confusion:.2f}")

    return agents, results


if __name__ == '__main__':
    agents, results = run_exploration()
