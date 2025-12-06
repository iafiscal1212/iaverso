#!/usr/bin/env python3
"""
Validación de Hipótesis - Script de Falsación
=============================================
Ejecutar después de 48-72 horas para validar predicciones.
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import pandas as pd
import numpy as np
from scipy import stats
from data.world_data_collector import WorldDataCollector
from pathlib import Path

# Hipótesis a validar
HYPOTHESES = [
    {
        'name': 'GEOMAG → SOLAR_DENSITY',
        'source': 'geomag_kp',
        'target': 'solar_density',
        'lag': 3,
        'expected_corr': -0.918,
        'falsification_threshold': -0.5,
    },
    {
        'name': 'SOLAR_SPEED → CLIMATE_PRESSURE',
        'source': 'solar_speed',
        'target': 'climate_pressure',
        'lag': 3,
        'expected_corr': 0.866,
        'falsification_threshold': 0.4,
    },
    {
        'name': 'CLIMATE_PRESSURE → BTC',
        'source': 'climate_pressure',
        'target': 'crypto_BTCUSDT_close',
        'lag': 3,
        'expected_corr': -0.918,
        'falsification_threshold': -0.3,
    },
    {
        'name': 'SOLAR_TEMP → SEISMIC',
        'source': 'solar_temperature',
        'target': 'seismic_count',
        'lag': 7,
        'expected_corr': 0.43,
        'falsification_threshold': 0.1,
    },
]


def validate_hypothesis(df, hyp):
    """Valida una hipótesis con datos nuevos."""
    src = hyp['source']
    tgt = hyp['target']
    lag = hyp['lag']
    
    if src not in df.columns or tgt not in df.columns:
        return {'status': 'NO_DATA', 'reason': f'Missing {src} or {tgt}'}
    
    # Alinear con lag
    src_data = df[src].iloc[:-lag].reset_index(drop=True) if lag > 0 else df[src].reset_index(drop=True)
    tgt_data = df[tgt].iloc[lag:].reset_index(drop=True) if lag > 0 else df[tgt].reset_index(drop=True)

    # Limpiar NaN
    valid = ~(src_data.isna() | tgt_data.isna())
    src_clean = src_data[valid]
    tgt_clean = tgt_data[valid]
    
    if len(src_clean) < 10:
        return {'status': 'INSUFFICIENT_DATA', 'n': len(src_clean)}
    
    # Correlación
    corr, pvalue = stats.pearsonr(src_clean, tgt_clean)
    
    # Evaluar
    expected = hyp['expected_corr']
    threshold = hyp['falsification_threshold']
    
    if expected > 0:
        falsified = corr < threshold
    else:
        falsified = corr > threshold
    
    return {
        'status': 'FALSIFIED' if falsified else 'SUPPORTED',
        'observed_corr': corr,
        'expected_corr': expected,
        'pvalue': pvalue,
        'n_samples': len(src_clean),
        'threshold': threshold,
    }


def main():
    print("=" * 70)
    print("VALIDACIÓN DE HIPÓTESIS")
    print("=" * 70)
    
    # Cargar datos más recientes
    data_dir = Path('/root/NEO_EVA/data')
    files = list(data_dir.glob('unified_*.csv'))
    
    if not files:
        print("No hay datos. Ejecuta world_data_collector.py primero.")
        return
    
    latest = max(files, key=lambda f: f.stat().st_mtime)
    print(f"Datos: {latest}")
    
    df = pd.read_csv(latest, index_col=0, parse_dates=True)
    print(f"Timestamps: {len(df)}")
    print()
    
    # Validar cada hipótesis
    for hyp in HYPOTHESES:
        print(f"\n{hyp['name']}:")
        print("-" * 50)
        
        result = validate_hypothesis(df, hyp)
        
        if result['status'] in ['NO_DATA', 'INSUFFICIENT_DATA']:
            print(f"  ⚠️  {result['status']}: {result.get('reason', result.get('n', ''))}")
        else:
            status_emoji = "❌" if result['status'] == 'FALSIFIED' else "✅"
            print(f"  {status_emoji} {result['status']}")
            print(f"     Correlación observada: {result['observed_corr']:+.3f}")
            print(f"     Correlación esperada:  {result['expected_corr']:+.3f}")
            print(f"     p-value: {result['pvalue']:.4f}")
            print(f"     Muestras: {result['n_samples']}")


if __name__ == '__main__':
    main()
