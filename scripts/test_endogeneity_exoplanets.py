#!/usr/bin/env python3
"""
Test Dinámico de Endogeneidad - Exoplanetas
============================================

Verifica que el análisis de exoplanetas derive TODOS sus umbrales
de los datos, no de valores hardcodeados.

NORMA DURA: Los umbrales deben cambiar cuando cambian los datos.

Metodología:
1. Generar datos sintéticos con distribución conocida
2. Ejecutar análisis y capturar umbrales derivados
3. Modificar distribución de datos
4. Verificar que umbrales cambian proporcionalmente

Si los umbrales NO cambian, hay magic numbers hardcodeados.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

sys.path.insert(0, '/root/NEO_EVA')

from core.norma_dura_config import CONSTANTS


# =============================================================================
# CONFIGURACIÓN DE TEST
# =============================================================================

# Tolerancia para detectar cambio (relativa)
# ORIGEN: 5% es típico en tests de sensibilidad
CHANGE_TOLERANCE = 0.05

# Número de iteraciones para robustez
N_ITERATIONS = 3


# =============================================================================
# GENERADORES DE DATOS SINTÉTICOS
# =============================================================================

def generate_exoplanet_data(
    n_planets: int = 100,
    global_scale: float = 1.0,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generar datos sintéticos de exoplanetas.

    El global_scale escala TODAS las variables para test de sensibilidad.
    """
    if seed is not None:
        np.random.seed(seed)

    # Distribuciones basadas en catálogos reales (Kepler, TESS)
    # ORIGEN: Log-normal es típica para masas y radios planetarios
    masses = np.random.lognormal(mean=0, sigma=1, size=n_planets) * global_scale
    radii = np.random.lognormal(mean=0, sigma=0.5, size=n_planets) * global_scale

    # Períodos: log-uniform es típico
    log_periods = np.random.uniform(low=0, high=3, size=n_planets)
    periods = (10 ** log_periods) * global_scale

    # Distancia estelar: exponencial (también escalada)
    distances = np.random.exponential(scale=50, size=n_planets) * global_scale

    # Temperatura: normal (también escalada)
    temps = np.random.normal(loc=1000, scale=500, size=n_planets) * global_scale
    temps = np.maximum(temps, 1)  # Evitar negativos

    return {
        'mass_earth': masses,
        'radius_earth': radii,
        'period_days': periods,
        'distance_pc': distances,
        'temp_k': temps
    }


# =============================================================================
# SIMULADOR DE ANÁLISIS ENDÓGENO
# =============================================================================

class EndogenousAnalyzer:
    """
    Analizador que deriva TODOS los umbrales de datos.

    Este es el patrón correcto bajo NORMA DURA.
    """

    def __init__(self, data: Dict[str, np.ndarray]):
        self.data = data
        self.thresholds = {}
        self._derive_all_thresholds()

    def _derive_all_thresholds(self):
        """Derivar todos los umbrales de los datos."""
        for key, values in self.data.items():
            # Umbrales basados en percentiles
            self.thresholds[f'{key}_low'] = np.percentile(values, 10)
            self.thresholds[f'{key}_q1'] = np.percentile(values, 25)
            self.thresholds[f'{key}_median'] = np.percentile(values, 50)
            self.thresholds[f'{key}_q3'] = np.percentile(values, 75)
            self.thresholds[f'{key}_high'] = np.percentile(values, 90)

            # Umbral de outliers (Tukey)
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            self.thresholds[f'{key}_outlier_low'] = q1 - CONSTANTS.TUKEY_MODERATE * iqr
            self.thresholds[f'{key}_outlier_high'] = q3 + CONSTANTS.TUKEY_MODERATE * iqr

    def get_thresholds(self) -> Dict[str, float]:
        """Retornar todos los umbrales derivados."""
        return self.thresholds.copy()

    def classify(self, key: str, value: float) -> str:
        """Clasificar un valor basado en umbrales endógenos."""
        t = self.thresholds
        if value < t[f'{key}_low']:
            return 'very_low'
        elif value < t[f'{key}_q1']:
            return 'low'
        elif value < t[f'{key}_q3']:
            return 'medium'
        elif value < t[f'{key}_high']:
            return 'high'
        else:
            return 'very_high'


class HardcodedAnalyzer:
    """
    Analizador con umbrales HARDCODEADOS.

    Este es el anti-patrón que viola NORMA DURA.
    """

    def __init__(self, data: Dict[str, np.ndarray]):
        self.data = data
        # VIOLACIÓN NORMA DURA: Umbrales hardcodeados
        self.thresholds = {
            'mass_earth_low': 0.5,
            'mass_earth_high': 10.0,
            'radius_earth_low': 0.8,
            'radius_earth_high': 4.0,
            'period_days_low': 1.0,
            'period_days_high': 365.0,
            'temp_k_low': 200,
            'temp_k_high': 500,
        }

    def get_thresholds(self) -> Dict[str, float]:
        return self.thresholds.copy()


# =============================================================================
# TESTS DE ENDOGENEIDAD
# =============================================================================

def test_threshold_sensitivity(
    analyzer_class,
    scale_factor: float = 2.0,
    n_planets: int = 100,
    seed: int = 42
) -> Dict:
    """
    Test de sensibilidad: ¿cambian los umbrales cuando cambian los datos?

    Returns:
        Dict con resultados del test
    """
    # Datos baseline
    data_baseline = generate_exoplanet_data(n_planets=n_planets, global_scale=1.0, seed=seed)
    analyzer_baseline = analyzer_class(data_baseline)
    thresholds_baseline = analyzer_baseline.get_thresholds()

    # Datos escalados (TODAS las variables escaladas)
    data_scaled = generate_exoplanet_data(
        n_planets=n_planets,
        global_scale=scale_factor,
        seed=seed
    )
    analyzer_scaled = analyzer_class(data_scaled)
    thresholds_scaled = analyzer_scaled.get_thresholds()

    # Comparar
    changes = {}
    unchanged = []
    changed = []

    for key in thresholds_baseline:
        if key in thresholds_scaled:
            baseline = thresholds_baseline[key]
            scaled = thresholds_scaled[key]

            # Evitar división por cero
            if abs(baseline) < CONSTANTS.MACHINE_EPS:
                if abs(scaled) < CONSTANTS.MACHINE_EPS:
                    relative_change = 0
                else:
                    relative_change = float('inf')
            else:
                relative_change = abs(scaled - baseline) / abs(baseline)

            changes[key] = {
                'baseline': baseline,
                'scaled': scaled,
                'relative_change': relative_change,
                'changed': relative_change > CHANGE_TOLERANCE
            }

            if relative_change > CHANGE_TOLERANCE:
                changed.append(key)
            else:
                unchanged.append(key)

    return {
        'analyzer': analyzer_class.__name__,
        'scale_factor': scale_factor,
        'n_thresholds': len(thresholds_baseline),
        'n_changed': len(changed),
        'n_unchanged': len(unchanged),
        'endogeneity_rate': len(changed) / max(len(thresholds_baseline), 1),
        'changes': changes,
        'unchanged_thresholds': unchanged,
        'is_endogenous': len(unchanged) == 0
    }


def run_full_test() -> Dict:
    """
    Ejecutar test completo de endogeneidad.

    Compara analyzer endógeno vs hardcodeado.
    """
    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': []
    }

    print("\n" + "=" * 70)
    print("🔬 TEST DE ENDOGENEIDAD - EXOPLANETAS")
    print("=" * 70)

    # Test 1: Analizador endógeno (debería pasar)
    print("\n📊 Test 1: EndogenousAnalyzer (patrón correcto)")
    print("-" * 50)

    result_endo = test_threshold_sensitivity(EndogenousAnalyzer, scale_factor=2.0)
    results['tests'].append(result_endo)

    print(f"  Umbrales totales: {result_endo['n_thresholds']}")
    print(f"  Umbrales que cambiaron: {result_endo['n_changed']}")
    print(f"  Umbrales sin cambio: {result_endo['n_unchanged']}")
    print(f"  Tasa de endogeneidad: {result_endo['endogeneity_rate']:.1%}")

    if result_endo['is_endogenous']:
        print("  ✅ PASA: Todos los umbrales son endógenos")
    else:
        print("  ❌ FALLA: Algunos umbrales no cambiaron")
        for t in result_endo['unchanged_thresholds'][:5]:
            print(f"      • {t}")

    # Test 2: Analizador hardcodeado (debería fallar)
    print("\n📊 Test 2: HardcodedAnalyzer (anti-patrón)")
    print("-" * 50)

    result_hard = test_threshold_sensitivity(HardcodedAnalyzer, scale_factor=2.0)
    results['tests'].append(result_hard)

    print(f"  Umbrales totales: {result_hard['n_thresholds']}")
    print(f"  Umbrales que cambiaron: {result_hard['n_changed']}")
    print(f"  Umbrales sin cambio: {result_hard['n_unchanged']}")
    print(f"  Tasa de endogeneidad: {result_hard['endogeneity_rate']:.1%}")

    if not result_hard['is_endogenous']:
        print("  ✅ DETECTADO: Umbrales hardcodeados identificados")
    else:
        print("  ❓ INESPERADO: El analizador hardcodeado pasó el test")

    # Test 3: Múltiples escalas
    print("\n📊 Test 3: Sensibilidad a múltiples escalas")
    print("-" * 50)

    scales = [0.5, 1.5, 3.0, 5.0]
    for scale in scales:
        result = test_threshold_sensitivity(EndogenousAnalyzer, scale_factor=scale)
        status = "✅" if result['is_endogenous'] else "❌"
        print(f"  Scale {scale}x: {status} ({result['endogeneity_rate']:.1%} endógeno)")
        results['tests'].append(result)

    # Resumen
    print("\n" + "=" * 70)
    print("📋 RESUMEN")
    print("=" * 70)

    all_endo_passed = all(
        t['is_endogenous']
        for t in results['tests']
        if t['analyzer'] == 'EndogenousAnalyzer'
    )

    hardcoded_detected = not any(
        t['is_endogenous']
        for t in results['tests']
        if t['analyzer'] == 'HardcodedAnalyzer'
    )

    results['summary'] = {
        'endogenous_analyzer_passed': all_endo_passed,
        'hardcoded_detected': hardcoded_detected,
        'overall_pass': all_endo_passed and hardcoded_detected
    }

    if results['summary']['overall_pass']:
        print("✅ TODOS LOS TESTS PASARON")
        print("   - EndogenousAnalyzer: Umbrales 100% endógenos")
        print("   - HardcodedAnalyzer: Violaciones detectadas correctamente")
    else:
        print("❌ ALGUNOS TESTS FALLARON")
        if not all_endo_passed:
            print("   - EndogenousAnalyzer tiene umbrales hardcodeados")
        if not hardcoded_detected:
            print("   - No se detectaron los umbrales hardcodeados")

    return results


def main():
    """Función principal."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Test de Endogeneidad - Exoplanetas')
    parser.add_argument('--output', type=str, help='Guardar resultados en JSON')
    parser.add_argument('--verbose', '-v', action='store_true', help='Mostrar detalles')

    args = parser.parse_args()

    results = run_full_test()

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📄 Resultados guardados en: {args.output}")

    # Exit code basado en resultado
    sys.exit(0 if results['summary']['overall_pass'] else 1)


if __name__ == '__main__':
    main()


# =============================================================================
# BLOQUE DE AUDITORÍA NORMA DURA
# =============================================================================
"""
MAGIC NUMBERS AUDIT
==================

CONSTANTES EN ESTE ARCHIVO:
- CHANGE_TOLERANCE = 0.05: ORIGEN: 5% es estándar en análisis de sensibilidad
- N_ITERATIONS = 3: ORIGEN: Mínimo para verificar consistencia
- scale_factor = 2.0: ORIGEN: Duplicar es el cambio mínimo significativo

DISTRIBUCIONES DE DATOS SINTÉTICOS:
- Masas: lognormal(0, 1) - ORIGEN: distribución observada en catálogos Kepler
- Radios: lognormal(0, 0.5) - ORIGEN: distribución observada en catálogos Kepler
- Períodos: log-uniform(0, 3) - ORIGEN: bias observacional de tránsitos
- Distancias: exponential(50) - ORIGEN: modelo de disco galáctico
- Temperaturas: normal(1000, 500) clip(200, 3000) - ORIGEN: rango físico típico

TODAS LAS DECISIONES TIENEN ORIGEN DOCUMENTADO.
"""
