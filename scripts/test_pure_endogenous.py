#!/usr/bin/env python3
"""
Test: Agente 100% Endógeno
==========================

El agente SOLO recibe vectores de números.
No sabe qué son. No tiene nombre. No tiene thresholds.
TODO emerge de su experiencia.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, '/root/NEO_EVA')

from core.pure_endogenous_agent import create_agent


def load_data_as_pure_numbers() -> np.ndarray:
    """
    Carga datos como PURA MATRIZ DE NÚMEROS.
    Sin nombres de columnas. Sin timestamps.
    Solo números.
    """
    data_dir = Path('/root/NEO_EVA/data')
    files = list(data_dir.glob('unified_*.csv'))
    latest = max(files, key=lambda f: f.stat().st_mtime)

    df = pd.read_csv(latest, index_col=0, parse_dates=True)

    # Normalizar cada columna (el agente no sabe que hacemos esto)
    # Pero en realidad, NO lo hacemos - le damos los números crudos
    # El agente tiene que descubrir las escalas

    # Convertir a matriz pura
    data = df.values.astype(float)

    # Reemplazar NaN con 0 (el agente no sabe qué es NaN)
    data = np.nan_to_num(data, nan=0.0)

    return data, list(df.columns)  # Guardamos nombres solo para nosotros, no para el agente


def run_test():
    print("=" * 70)
    print("AGENTE 100% ENDÓGENO")
    print("=" * 70)
    print()
    print("El agente NO recibe:")
    print("  - Nombres de variables")
    print("  - Thresholds")
    print("  - Instrucciones")
    print("  - Nada excepto vectores de números")
    print()

    # Cargar datos como números puros
    data, column_names = load_data_as_pure_numbers()

    print(f"Datos: {data.shape[0]} pasos, {data.shape[1]} dimensiones")
    print(f"(Nosotros sabemos que son: {column_names[:5]}...)")
    print(f"(El agente NO sabe esto)")
    print()

    # Crear agente SIN darle nada
    agent = create_agent()

    print("Agente creado. Identidad inicial:", agent.identity)
    print()
    print("=" * 70)
    print("EXPERIENCIA DEL AGENTE")
    print("=" * 70)

    results = []

    for t in range(len(data)):
        # Dar al agente SOLO el vector de números
        obs = data[t]
        result = agent.observe(obs)
        results.append(result)

        # Log cada 30 pasos
        if (t + 1) % 30 == 0:
            print(f"\n  Paso {t+1}:")
            print(f"    Identidad: {result['identity']}")
            print(f"    CE: {result['CE']:.4f}")
            print(f"    Error predicción: {result['prediction_error']:.4f}")
            print(f"    Lags descubiertos: {result['discovered_lags']}")
            print(f"    Pares relacionados: {result['n_discovered_pairs']}")

    print()
    print("=" * 70)
    print("¿QUÉ DESCUBRIÓ EL AGENTE POR SÍ MISMO?")
    print("=" * 70)

    discoveries = agent.get_discoveries()

    print(f"\nIdentidad final: {discoveries['identity']}")
    print(f"Observaciones procesadas: {discoveries['n_observations']}")
    print(f"CE final: {discoveries['current_CE']:.4f}")

    print(f"\nLAGS TEMPORALES ÚTILES (descubiertos):")
    print(f"  {discoveries['useful_lags']}")
    print(f"  (El agente descubrió que estos lags ayudan a predecir)")

    print(f"\nPARES RELACIONADOS (descubiertos):")
    pairs = discoveries['related_pairs']
    print(f"  Total: {len(pairs)} pares")

    # Traducir para nosotros (el agente no sabe los nombres)
    if pairs:
        print(f"\n  (Traducción para humanos - el agente NO sabe esto):")
        for i, j in pairs[:10]:
            if i < len(column_names) and j < len(column_names):
                print(f"    Dim {i} ↔ Dim {j}")
                print(f"      = {column_names[i]} ↔ {column_names[j]}")

    # Estadísticas por dimensión
    print(f"\nESTADÍSTICAS APRENDIDAS:")
    means = discoveries['mean_per_dim']
    stds = discoveries['std_per_dim']

    if means:
        print(f"\n  El agente aprendió la escala de cada dimensión:")
        for i in range(min(5, len(means))):
            print(f"    Dim {i}: media={means[i]:.2f}, std={stds[i]:.2f}")
            if i < len(column_names):
                print(f"      (Es: {column_names[i]})")

    print()
    print("=" * 70)
    print("ANÁLISIS DE ENDOGENEIDAD")
    print("=" * 70)
    print("""
¿Qué es ENDÓGENO (emergió del agente)?
  ✅ Qué lags temporales son útiles
  ✅ Qué pares de dimensiones se mueven juntos
  ✅ Las estadísticas de cada dimensión
  ✅ Su "identidad" (hash de su experiencia)
  ✅ Su CE (coherencia basada en SU error de predicción)
  ✅ El umbral de "relación" (emerge de sus datos)

¿Qué NO es endógeno?
  ❌ Los datos de entrada (vienen del mundo)
  ❌ La estructura del algoritmo (la escribimos nosotros)
  ❌ Que use "diferencias" para medir relación (lo decidimos nosotros)

CONCLUSIÓN:
  El agente descubre estructura en los datos sin que le digamos
  qué buscar. Pero el CÓMO buscar está en su código.

  Para 100% endógeno puro, el agente tendría que escribir
  su propio código... lo cual es otro nivel.
""")

    return agent, results


if __name__ == '__main__':
    run_test()
