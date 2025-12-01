#!/usr/bin/env python3
"""
Test: ¿De dónde viene el período ~45?

Hipótesis: viene de las ventanas de 20 pasos hardcodeadas.

Verificación:
- Ventana 10 → período ~25?
- Ventana 20 → período ~45? (actual)
- Ventana 40 → período ~85?

Si esto se confirma, el período NO es endógeno,
es un artefacto de las constantes de detección.
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import json
import os


class ParametricAutonomousAgent:
    """
    Agente con ventanas PARAMETRIZABLES.
    """

    def __init__(self, name: str, dim: int = 6, window_size: int = 20,
                 step_rate: float = 0.1, identity_pull: float = 0.05):
        self.name = name
        self.dim = dim

        # PARÁMETROS CONFIGURABLES (antes hardcodeados)
        self.window_size = window_size  # Era 20
        self.step_rate = step_rate  # Era 0.1
        self.identity_pull_strength = identity_pull  # Era 0.05

        # Estado
        self.z = np.ones(dim) / dim
        self.z_history: List[np.ndarray] = []

        # Identidad
        self.identity_core = self.z.copy()
        self.identity_strength = 0.5
        self.identity_history: List[float] = []

        # Crisis
        self.in_crisis = False
        self.crises: List[int] = []

        self.t = 0

    def _compute_identity_strength(self) -> float:
        W = self.window_size
        if len(self.z_history) < W // 2:
            return 0.5

        dist_to_core = np.linalg.norm(self.z - self.identity_core)
        recent_dists = [np.linalg.norm(zh - self.identity_core)
                       for zh in self.z_history[-W:]]
        typical_dist = np.mean(recent_dists) + 1e-10

        return float(1.0 / (1.0 + dist_to_core / typical_dist))

    def _detect_crisis(self) -> bool:
        W = self.window_size
        if len(self.identity_history) < W:
            return False

        # Comparar últimos W/4 vs W anterior
        recent_window = W // 4
        recent = np.mean(self.identity_history[-recent_window:])
        baseline = np.mean(self.identity_history[-W:-recent_window])

        drop = baseline - recent

        # Umbral endógeno
        if len(self.identity_history) > W * 2:
            drops = [self.identity_history[i] - self.identity_history[i + recent_window]
                    for i in range(len(self.identity_history) - recent_window)]
            threshold = np.percentile(drops, 90)
        else:
            threshold = 0.15

        return drop > threshold

    def step(self, stimulus: np.ndarray, other_z: Optional[np.ndarray] = None) -> Dict:
        self.t += 1

        # Dinámica
        response = 0.5 * stimulus[:self.dim]

        if other_z is not None:
            other_influence = 0.1 * (other_z[:self.dim] - self.z)
        else:
            other_influence = 0

        identity_pull = self.identity_pull_strength * self.identity_strength * (self.identity_core - self.z)

        noise = np.random.randn(self.dim) * (0.02 if not self.in_crisis else 0.08)

        # Update con tasa configurable
        self.z = self.z + self.step_rate * response + other_influence + identity_pull + noise
        self.z = np.clip(self.z, 0.01, 0.99)
        self.z = self.z / self.z.sum()

        self.z_history.append(self.z.copy())

        # Identidad
        self.identity_strength = self._compute_identity_strength()
        self.identity_history.append(self.identity_strength)

        # Crisis
        was_in_crisis = self.in_crisis
        if self._detect_crisis() and not self.in_crisis:
            self.in_crisis = True
            self.crises.append(self.t)

        # Salir de crisis
        W = self.window_size
        if self.in_crisis and len(self.identity_history) > W // 2:
            recent = np.mean(self.identity_history[-(W//4):])
            past = np.mean(self.identity_history[-(W//2):-(W//4)])
            if recent > past:
                self.in_crisis = False

        # Actualizar núcleo identitario
        if len(self.z_history) >= W // 2:
            eta = 0.01 * (1 - self.identity_strength)
            self.identity_core = (1 - eta) * self.identity_core + eta * self.z

        return {
            't': self.t,
            'identity': self.identity_strength,
            'in_crisis': self.in_crisis
        }


def analyze_period(identity_history: List[float]) -> float:
    """Calcula período dominante usando FFT."""
    if len(identity_history) < 100:
        return 0

    signal = np.array(identity_history) - np.mean(identity_history)
    spectrum = np.abs(np.fft.rfft(signal))
    spectrum[0] = 0  # Ignorar DC

    n = len(signal)
    freqs = np.fft.rfftfreq(n)

    # Encontrar pico
    peak_idx = np.argmax(spectrum[1:50]) + 1  # Buscar en frecuencias razonables
    if freqs[peak_idx] > 0:
        period = 1.0 / freqs[peak_idx]
        return period

    return 0


def analyze_crisis_intervals(crises: List[int]) -> float:
    """Calcula intervalo medio entre crisis."""
    if len(crises) < 2:
        return 0
    intervals = np.diff(crises)
    return np.mean(intervals)


def run_parametric_test(window_sizes: List[int] = [10, 20, 40],
                       T: int = 2000, seed: int = 42) -> Dict:
    """
    Test paramétrico: ¿el período depende del window_size?
    """
    print("=" * 70)
    print("TEST: ORIGEN DEL PERÍODO ~45")
    print("=" * 70)
    print(f"Hipótesis: período ≈ 2 × window_size")
    print(f"Window sizes: {window_sizes}")
    print()

    np.random.seed(seed)

    results = {}

    for W in window_sizes:
        print(f"\n--- Window size = {W} ---")

        # Crear agentes
        neo = ParametricAutonomousAgent("NEO", window_size=W)
        eva = ParametricAutonomousAgent("EVA", window_size=W)

        for t in range(T):
            stimulus = np.random.dirichlet(np.ones(6) * 2)
            neo.step(stimulus, eva.z)
            eva.step(stimulus, neo.z)

        # Análisis
        period_neo = analyze_period(neo.identity_history)
        period_eva = analyze_period(eva.identity_history)

        interval_neo = analyze_crisis_intervals(neo.crises)
        interval_eva = analyze_crisis_intervals(eva.crises)

        results[W] = {
            'window_size': W,
            'neo_period': period_neo,
            'eva_period': period_eva,
            'neo_interval': interval_neo,
            'eva_interval': interval_eva,
            'neo_crises': len(neo.crises),
            'eva_crises': len(eva.crises),
            'predicted_period': 2 * W  # Si hipótesis correcta
        }

        print(f"  NEO: período={period_neo:.1f}, intervalo crisis={interval_neo:.1f}, n_crisis={len(neo.crises)}")
        print(f"  EVA: período={period_eva:.1f}, intervalo crisis={interval_eva:.1f}, n_crisis={len(eva.crises)}")
        print(f"  Predicción (2×W): {2 * W}")

    # Análisis de correlación
    print("\n" + "=" * 70)
    print("ANÁLISIS")
    print("=" * 70)

    Ws = list(results.keys())
    periods = [(results[W]['neo_period'] + results[W]['eva_period']) / 2 for W in Ws]
    intervals = [(results[W]['neo_interval'] + results[W]['eva_interval']) / 2 for W in Ws]
    predicted = [2 * W for W in Ws]

    # Correlación período vs window
    if len(Ws) >= 3:
        corr_period_W = np.corrcoef(Ws, periods)[0, 1]
        corr_interval_W = np.corrcoef(Ws, intervals)[0, 1]

        print(f"\nCorrelación window_size vs período: {corr_period_W:.3f}")
        print(f"Correlación window_size vs intervalo: {corr_interval_W:.3f}")

        # Ratio período/window
        ratios = [p / W for p, W in zip(periods, Ws)]
        print(f"\nRatio período/window: {ratios}")
        print(f"Ratio medio: {np.mean(ratios):.2f}")

        if corr_period_W > 0.9:
            print("\n→ CONFIRMADO: El período DEPENDE del window_size")
            print("→ El período ~45 viene de la ventana de 20 hardcodeada")
            print("→ NO es endógeno - es un ARTEFACTO")
        else:
            print("\n→ La correlación no es perfecta")
            print("→ Puede haber otros factores contribuyendo")

    # Tabla resumen
    print("\n" + "-" * 60)
    print(f"{'Window':>8} {'Período':>10} {'Intervalo':>10} {'Pred(2W)':>10} {'Error%':>10}")
    print("-" * 60)
    for W in Ws:
        r = results[W]
        avg_period = (r['neo_period'] + r['eva_period']) / 2
        pred = 2 * W
        error = abs(avg_period - pred) / pred * 100 if pred > 0 else 0
        print(f"{W:>8} {avg_period:>10.1f} {(r['neo_interval']+r['eva_interval'])/2:>10.1f} {pred:>10} {error:>9.1f}%")

    # Guardar
    os.makedirs('/root/NEO_EVA/results/period_origin', exist_ok=True)

    final_results = {
        'timestamp': datetime.now().isoformat(),
        'T': T,
        'seed': seed,
        'results': results,
        'conclusion': 'period_depends_on_window' if corr_period_W > 0.9 else 'period_partially_depends'
    }

    with open('/root/NEO_EVA/results/period_origin/test.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    return final_results


def run_dynamics_test(step_rates: List[float] = [0.05, 0.1, 0.2],
                     T: int = 2000, seed: int = 42) -> Dict:
    """
    Test: ¿el período depende de la tasa de cambio (step_rate)?
    """
    print("\n" + "=" * 70)
    print("TEST: ¿PERÍODO DEPENDE DE STEP_RATE?")
    print("=" * 70)

    np.random.seed(seed)

    results = {}

    for rate in step_rates:
        print(f"\n--- Step rate = {rate} ---")

        neo = ParametricAutonomousAgent("NEO", step_rate=rate)
        eva = ParametricAutonomousAgent("EVA", step_rate=rate)

        for t in range(T):
            stimulus = np.random.dirichlet(np.ones(6) * 2)
            neo.step(stimulus, eva.z)
            eva.step(stimulus, neo.z)

        period_neo = analyze_period(neo.identity_history)
        period_eva = analyze_period(eva.identity_history)

        results[rate] = {
            'step_rate': rate,
            'neo_period': period_neo,
            'eva_period': period_eva,
            'avg_period': (period_neo + period_eva) / 2
        }

        print(f"  Período promedio: {results[rate]['avg_period']:.1f}")

    # Análisis
    rates = list(results.keys())
    periods = [results[r]['avg_period'] for r in rates]

    if len(rates) >= 3:
        # El período debería ser inversamente proporcional a step_rate
        inv_rates = [1/r for r in rates]
        corr = np.corrcoef(inv_rates, periods)[0, 1]
        print(f"\nCorrelación 1/step_rate vs período: {corr:.3f}")

    return results


if __name__ == "__main__":
    # Test 1: Window size
    results_window = run_parametric_test(
        window_sizes=[10, 15, 20, 30, 40],
        T=2000,
        seed=42
    )

    # Test 2: Step rate
    results_rate = run_dynamics_test(
        step_rates=[0.05, 0.1, 0.15, 0.2],
        T=2000,
        seed=42
    )
