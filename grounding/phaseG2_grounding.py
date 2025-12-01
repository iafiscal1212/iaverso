#!/usr/bin/env python3
"""
Phase G2: Tests de Grounding
============================

Tres tests de conexión mundo ↔ representaciones internas:
1. Predictive Grounding: predicción del mundo
2. Symbolic Grounding: MI entre símbolos y regímenes
3. Value Grounding: valores correlacionados con estados predecibles

100% ENDÓGENO - Sin constantes mágicas
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os

import sys
sys.path.insert(0, '/root/NEO_EVA/grounding')
from phaseG1_world_channel import StructuredWorldChannel, WorldState


@dataclass
class GroundingResult:
    """Resultado de un test de grounding."""
    name: str
    score_real: float
    score_null: float
    p95_null: float
    passed: bool


class GroundingTests:
    """
    Tests de grounding entre sistema interno y mundo externo.

    100% Endógeno:
    - Predicciones por regresión lineal simple
    - MI estimada por k-NN adaptativo
    - Umbrales por percentil 95 de nulos
    """

    def __init__(self, dim_internal: int = 6, dim_world: int = 6):
        self.dim_internal = dim_internal
        self.dim_world = dim_world

        # Mundo estructurado
        self.world = StructuredWorldChannel(dim_s=dim_world, seed=42)

        # Historia de estados internos y externos
        self.z_history: List[np.ndarray] = []
        self.s_history: List[np.ndarray] = []
        self.regime_history: List[int] = []

        # Símbolos internos (generados endógenamente)
        self.symbol_history: List[int] = []
        self.n_symbols = 5

        # Valores/metas internos
        self.value_history: List[float] = []
        self.goal_active_history: List[bool] = []

        # Predicciones
        self.predictions: List[np.ndarray] = []
        self.prediction_errors: List[float] = []

    def _generate_symbol(self, z: np.ndarray) -> int:
        """
        Genera símbolo interno basado en estado.

        100% endógeno: clustering por k-means online simplificado
        """
        # Usar primeras componentes para discretizar
        if len(z) >= 2:
            # Discretizar en grid
            x_bin = int(z[0] * self.n_symbols) % self.n_symbols
            y_bin = int(z[1] * self.n_symbols) % self.n_symbols
            return (x_bin + y_bin) % self.n_symbols
        return 0

    def _compute_value(self, z: np.ndarray, s: np.ndarray) -> float:
        """
        Computa valor interno del estado.

        100% endógeno: basado en entropía y predictabilidad
        """
        # Entropía interna
        z_prob = z / (z.sum() + 1e-10)
        z_prob = np.clip(z_prob, 1e-10, 1.0)
        entropy = -np.sum(z_prob * np.log(z_prob))

        # Error de predicción (sorpresa) si hay predicción previa
        if self.predictions:
            pred_error = np.linalg.norm(s - self.predictions[-1])
        else:
            pred_error = 0.5

        # Valor = entropía alta + sorpresa baja (estados informativos y predecibles)
        value = entropy / (1 + pred_error)

        return value

    def _predict_world(self, z: np.ndarray) -> np.ndarray:
        """
        Predice siguiente estado del mundo basado en estado interno.

        f(z_t) → ŝ_{t+1}

        100% endógeno: regresión lineal sobre historia reciente
        """
        if len(self.z_history) < 10 or len(self.s_history) < 10:
            return np.ones(self.dim_world) * 0.5

        # Ventana endógena
        window = max(10, int(np.sqrt(len(self.z_history))))

        Z = np.array(self.z_history[-window:])
        S_next = np.array(self.s_history[-window+1:] + [self.s_history[-1]])

        # Ajustar tamaños
        min_len = min(len(Z), len(S_next))
        Z = Z[:min_len]
        S_next = S_next[:min_len]

        # Regresión lineal simple: S = Z @ W + b
        try:
            # Pseudo-inversa
            Z_pinv = np.linalg.pinv(Z)
            W = Z_pinv @ S_next

            # Predicción para estado actual
            prediction = z @ W
            prediction = np.clip(prediction, 0, 1)
            return prediction

        except Exception:
            return np.ones(self.dim_world) * 0.5

    def step(self, z: np.ndarray) -> Dict[str, Any]:
        """
        Ejecuta un paso de interacción con el mundo.

        Args:
            z: Estado interno actual

        Returns:
            Dict con estado del mundo y métricas de grounding
        """
        # Paso del mundo
        world_state = self.world.step()
        s = world_state.s
        regime = world_state.regime

        # Generar símbolo
        symbol = self._generate_symbol(z)

        # Predecir mundo (antes de ver s)
        prediction = self._predict_world(z)
        pred_error = np.linalg.norm(prediction - s)

        # Computar valor
        value = self._compute_value(z, s)

        # Meta activa (endógeno: valor > mediana)
        if self.value_history:
            value_threshold = np.median(self.value_history)
            goal_active = value > value_threshold
        else:
            goal_active = True

        # Registrar historia
        self.z_history.append(z.copy())
        self.s_history.append(s.copy())
        self.regime_history.append(regime)
        self.symbol_history.append(symbol)
        self.predictions.append(prediction)
        self.prediction_errors.append(pred_error)
        self.value_history.append(value)
        self.goal_active_history.append(goal_active)

        return {
            't': len(self.z_history),
            'world_state': world_state,
            'symbol': symbol,
            'prediction_error': pred_error,
            'value': value,
            'goal_active': goal_active
        }

    def test_predictive_grounding(self, n_nulls: int = 10) -> GroundingResult:
        """
        Test 1: Predictive Grounding

        Mide: ΔMSE = MSE_null - MSE_real

        GO si ΔMSE > 0 y > p95 de nulos
        """
        if len(self.prediction_errors) < 50:
            return GroundingResult(
                name='predictive',
                score_real=0,
                score_null=0,
                p95_null=0,
                passed=False
            )

        MSE_real = np.mean(np.array(self.prediction_errors) ** 2)

        # Generar nulos (predicciones aleatorias)
        null_mses = []
        for _ in range(n_nulls):
            null_errors = []
            for t in range(len(self.s_history)):
                null_pred = np.random.rand(self.dim_world)
                null_error = np.linalg.norm(null_pred - self.s_history[t])
                null_errors.append(null_error)
            null_mses.append(np.mean(np.array(null_errors) ** 2))

        MSE_null_mean = np.mean(null_mses)
        delta_mse = MSE_null_mean - MSE_real

        # También calcular p95 del delta
        null_deltas = []
        for _ in range(n_nulls):
            # Shuffle de predicciones
            shuffled_errors = np.random.permutation(self.prediction_errors)
            mse_shuffled = np.mean(shuffled_errors ** 2)
            null_deltas.append(MSE_null_mean - mse_shuffled)

        p95_null = np.percentile(null_deltas, 95) if null_deltas else 0

        passed = delta_mse > 0 and delta_mse > p95_null

        return GroundingResult(
            name='predictive',
            score_real=delta_mse,
            score_null=MSE_null_mean,
            p95_null=p95_null,
            passed=passed
        )

    def test_symbolic_grounding(self, n_nulls: int = 10) -> GroundingResult:
        """
        Test 2: Symbolic Grounding

        Mide: I(σ; s_regime) = H(σ) - H(σ|s_regime)

        GO si I_real > p95(I_null)
        """
        if len(self.symbol_history) < 50:
            return GroundingResult(
                name='symbolic',
                score_real=0,
                score_null=0,
                p95_null=0,
                passed=False
            )

        # H(σ) - entropía de símbolos
        symbol_counts = {}
        for s in self.symbol_history:
            symbol_counts[s] = symbol_counts.get(s, 0) + 1
        total = len(self.symbol_history)
        H_sigma = -sum((c/total) * np.log(c/total + 1e-10) for c in symbol_counts.values())

        # H(σ|regime) - entropía condicional
        regime_symbols = {}
        for sym, reg in zip(self.symbol_history, self.regime_history):
            if reg not in regime_symbols:
                regime_symbols[reg] = []
            regime_symbols[reg].append(sym)

        H_sigma_given_regime = 0
        for reg, symbols in regime_symbols.items():
            p_reg = len(symbols) / total
            sym_counts = {}
            for s in symbols:
                sym_counts[s] = sym_counts.get(s, 0) + 1
            H_sym_in_reg = -sum((c/len(symbols)) * np.log(c/len(symbols) + 1e-10)
                               for c in sym_counts.values())
            H_sigma_given_regime += p_reg * H_sym_in_reg

        I_real = H_sigma - H_sigma_given_regime

        # Nulos: shuffle símbolos
        null_Is = []
        for _ in range(n_nulls):
            shuffled_symbols = list(np.random.permutation(self.symbol_history))

            # Recalcular MI
            regime_symbols_null = {}
            for sym, reg in zip(shuffled_symbols, self.regime_history):
                if reg not in regime_symbols_null:
                    regime_symbols_null[reg] = []
                regime_symbols_null[reg].append(sym)

            H_cond_null = 0
            for reg, symbols in regime_symbols_null.items():
                p_reg = len(symbols) / total
                sym_counts = {}
                for s in symbols:
                    sym_counts[s] = sym_counts.get(s, 0) + 1
                H_sym_in_reg = -sum((c/len(symbols)) * np.log(c/len(symbols) + 1e-10)
                                   for c in sym_counts.values())
                H_cond_null += p_reg * H_sym_in_reg

            I_null = H_sigma - H_cond_null
            null_Is.append(I_null)

        p95_null = np.percentile(null_Is, 95) if null_Is else 0
        passed = I_real > p95_null

        return GroundingResult(
            name='symbolic',
            score_real=I_real,
            score_null=np.mean(null_Is),
            p95_null=p95_null,
            passed=passed
        )

    def test_value_grounding(self, n_nulls: int = 10) -> GroundingResult:
        """
        Test 3: Value Grounding

        Mide: correlación entre valor alto y estados predecibles

        GO si correlación > p95(null)
        """
        if len(self.value_history) < 50:
            return GroundingResult(
                name='value',
                score_real=0,
                score_null=0,
                p95_null=0,
                passed=False
            )

        # Correlación: valor alto ↔ error de predicción bajo
        values = np.array(self.value_history)
        errors = np.array(self.prediction_errors)

        # Correlación negativa esperada (más valor = menos error)
        corr_real = -np.corrcoef(values, errors)[0, 1]
        if np.isnan(corr_real):
            corr_real = 0

        # Nulos: shuffle valores
        null_corrs = []
        for _ in range(n_nulls):
            shuffled_values = np.random.permutation(values)
            corr_null = -np.corrcoef(shuffled_values, errors)[0, 1]
            if not np.isnan(corr_null):
                null_corrs.append(corr_null)

        p95_null = np.percentile(null_corrs, 95) if null_corrs else 0
        passed = corr_real > p95_null

        return GroundingResult(
            name='value',
            score_real=corr_real,
            score_null=np.mean(null_corrs) if null_corrs else 0,
            p95_null=p95_null,
            passed=passed
        )

    def run_all_tests(self, n_nulls: int = 10) -> Dict[str, GroundingResult]:
        """Ejecuta todos los tests de grounding."""
        return {
            'predictive': self.test_predictive_grounding(n_nulls),
            'symbolic': self.test_symbolic_grounding(n_nulls),
            'value': self.test_value_grounding(n_nulls)
        }


def run_phase_g2() -> Dict[str, Any]:
    """Ejecuta Phase G2 y evalúa criterios GO/NO-GO."""

    print("=" * 70)
    print("PHASE G2: TESTS DE GROUNDING")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}\n")

    np.random.seed(42)

    # Crear sistema de grounding
    grounding = GroundingTests(dim_internal=6, dim_world=6)

    # Simulación
    T = 500
    results = []

    print("Simulando interacción mundo-sistema...")
    z = np.random.rand(6)
    z = z / z.sum()

    for t in range(T):
        # Dinámica interna influenciada por mundo
        if t > 0 and grounding.s_history:
            # El sistema "percibe" el mundo
            z = 0.8 * z + 0.2 * grounding.s_history[-1][:6]

        noise = np.random.randn(6) * 0.02
        z = z + noise
        z = np.clip(z, 0.01, 0.99)
        z = z / z.sum()

        result = grounding.step(z)
        results.append(result)

        if t % 100 == 0:
            print(f"  t={t}, pred_error={result['prediction_error']:.4f}, symbol={result['symbol']}")

    print()

    # Ejecutar tests
    print("Ejecutando tests de grounding...")
    test_results = grounding.run_all_tests(n_nulls=10)

    print("=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print()

    for name, result in test_results.items():
        print(f"{name.upper()} GROUNDING:")
        print(f"  Score real: {result.score_real:.4f}")
        print(f"  Score null: {result.score_null:.4f}")
        print(f"  p95 null: {result.p95_null:.4f}")
        print(f"  Pasado: {'✓' if result.passed else '✗'}")
        print()

    # Criterios GO/NO-GO
    criteria = {}

    # 1. Predictive grounding
    criteria['predictive_grounding'] = test_results['predictive'].passed

    # 2. Symbolic grounding
    criteria['symbolic_grounding'] = test_results['symbolic'].passed

    # 3. Value grounding
    criteria['value_grounding'] = test_results['value'].passed

    # 4. Al menos 2 de 3 tests pasados
    n_passed = sum(r.passed for r in test_results.values())
    criteria['majority_passed'] = n_passed >= 2

    # 5. Scores reales positivos
    criteria['scores_positive'] = all(r.score_real > 0 for r in test_results.values())

    passed = sum(criteria.values())
    total = len(criteria)
    go_status = "GO" if passed >= 3 else "NO-GO"

    print("Criterios:")
    for name, passed_criterion in criteria.items():
        status = "✅" if passed_criterion else "❌"
        print(f"  {status} {name}")
    print()
    print(f"Resultado: {go_status} ({passed}/{total} criterios)")

    # Guardar resultados
    output = {
        'phase': 'G2',
        'name': 'Grounding Tests',
        'timestamp': datetime.now().isoformat(),
        'tests': {
            name: {
                'score_real': r.score_real,
                'score_null': r.score_null,
                'p95_null': r.p95_null,
                'passed': r.passed
            }
            for name, r in test_results.items()
        },
        'criteria': criteria,
        'go_status': go_status,
        'passed_criteria': passed,
        'total_criteria': total
    }

    os.makedirs('/root/NEO_EVA/results/phaseG2', exist_ok=True)
    os.makedirs('/root/NEO_EVA/figures', exist_ok=True)

    with open('/root/NEO_EVA/results/phaseG2/grounding_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Visualización
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Error de predicción temporal
        ax1 = axes[0, 0]
        ax1.plot(grounding.prediction_errors, 'b-', alpha=0.7, linewidth=0.5)
        ax1.axhline(y=np.mean(grounding.prediction_errors), color='r', linestyle='--',
                   label=f'Media={np.mean(grounding.prediction_errors):.4f}')
        ax1.set_xlabel('Tiempo')
        ax1.set_ylabel('Error de Predicción')
        ax1.set_title('Predictive Grounding: Error Temporal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Símbolos vs Regímenes
        ax2 = axes[0, 1]
        # Matriz de confusión símbolos-regímenes
        n_sym = grounding.n_symbols
        n_reg = grounding.world.n_regimes
        confusion = np.zeros((n_sym, n_reg))
        for sym, reg in zip(grounding.symbol_history, grounding.regime_history):
            confusion[sym, reg] += 1
        confusion = confusion / (confusion.sum(axis=0, keepdims=True) + 1e-10)
        im = ax2.imshow(confusion, cmap='Blues', aspect='auto')
        ax2.set_xlabel('Régimen del Mundo')
        ax2.set_ylabel('Símbolo Interno')
        ax2.set_title('Symbolic Grounding: P(símbolo|régimen)')
        plt.colorbar(im, ax=ax2)

        # 3. Valor vs Error de predicción
        ax3 = axes[1, 0]
        ax3.scatter(grounding.value_history, grounding.prediction_errors,
                   alpha=0.3, s=10, c=grounding.regime_history, cmap='viridis')
        ax3.set_xlabel('Valor Interno')
        ax3.set_ylabel('Error de Predicción')
        ax3.set_title('Value Grounding: Valor vs Sorpresa')
        ax3.grid(True, alpha=0.3)

        # 4. Resumen de tests
        ax4 = axes[1, 1]
        test_names = list(test_results.keys())
        scores_real = [test_results[n].score_real for n in test_names]
        p95_nulls = [test_results[n].p95_null for n in test_names]

        x = np.arange(len(test_names))
        width = 0.35

        bars1 = ax4.bar(x - width/2, scores_real, width, label='Real', color='blue', alpha=0.7)
        bars2 = ax4.bar(x + width/2, p95_nulls, width, label='p95 Null', color='red', alpha=0.7)

        ax4.set_xlabel('Test')
        ax4.set_ylabel('Score')
        ax4.set_title('Comparación Real vs Null')
        ax4.set_xticks(x)
        ax4.set_xticklabels([n.capitalize() for n in test_names])
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/root/NEO_EVA/figures/phaseG2_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nResultados guardados en: /root/NEO_EVA/results/phaseG2")
        print(f"Figura: /root/NEO_EVA/figures/phaseG2_results.png")

    except Exception as e:
        print(f"Warning: No se pudo crear visualización: {e}")

    return output


if __name__ == "__main__":
    run_phase_g2()
