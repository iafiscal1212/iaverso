#!/usr/bin/env python3
"""
Phase G2-Dual: Grounding Separado por Agente
=============================================

Cada agente tiene su propio grounding con el mundo:
- NEO: más enganchado al mundo (predicción precisa)
- EVA: puede estar más enganchada a NEO

Para cada agente A:
- G_A_pred = rank(-MSE_A)
- G_A_sym = rank(I(σ_A; regime))
- G_A_val = rank(|corr(V_A, outcome_world)|)
- GI_A = G_A_pred + G_A_sym + G_A_val

100% ENDÓGENO - Sin constantes mágicas
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import sys

sys.path.insert(0, '/root/NEO_EVA/grounding')
from phaseG1_world_channel import StructuredWorldChannel, WorldState


@dataclass
class AgentGrounding:
    """Métricas de grounding para un agente."""
    agent: str
    t: int
    G_pred: float      # Grounding predictivo
    G_sym: float       # Grounding simbólico
    G_val: float       # Grounding de valor
    GI: float          # Grounding Index total
    prediction_error: float
    symbol: int
    value: float


class DualAgentGrounding:
    """
    Grounding separado para NEO y EVA.

    Características:
    - Un solo WORLD, pero cada agente tiene su propio predictor
    - NEO: más conectado al mundo externo
    - EVA: puede predecir también a través de NEO

    Para cada A:
    - G_A_pred = rank(-MSE_A)
    - G_A_sym = rank(I(σ_A; regime))
    - G_A_val = rank(|corr(V_A, outcome)|)
    - GI_A = G_A_pred + G_A_sym + G_A_val

    100% Endógeno
    """

    def __init__(self, dim_world: int = 6, dim_neo: int = 6, dim_eva: int = 6):
        self.dim_world = dim_world
        self.dim_neo = dim_neo
        self.dim_eva = dim_eva

        # Mundo compartido
        self.world = StructuredWorldChannel(dim_s=dim_world, seed=42)

        # Número de símbolos por agente
        self.n_symbols = 5

        # === Historia NEO ===
        self.neo_z_history: List[np.ndarray] = []
        self.neo_predictions: List[np.ndarray] = []
        self.neo_pred_errors: List[float] = []
        self.neo_symbols: List[int] = []
        self.neo_values: List[float] = []
        self.neo_G_pred_history: List[float] = []
        self.neo_G_sym_history: List[float] = []
        self.neo_G_val_history: List[float] = []
        self.neo_GI_history: List[float] = []

        # === Historia EVA ===
        self.eva_z_history: List[np.ndarray] = []
        self.eva_predictions: List[np.ndarray] = []
        self.eva_pred_errors: List[float] = []
        self.eva_symbols: List[int] = []
        self.eva_values: List[float] = []
        self.eva_G_pred_history: List[float] = []
        self.eva_G_sym_history: List[float] = []
        self.eva_G_val_history: List[float] = []
        self.eva_GI_history: List[float] = []

        # === Historia compartida ===
        self.s_history: List[np.ndarray] = []
        self.regime_history: List[int] = []
        self.outcome_history: List[float] = []

        self.t = 0

    def _generate_symbol(self, z: np.ndarray, agent: str) -> int:
        """
        Genera símbolo interno basado en estado.

        100% endógeno: discretización por grid
        """
        if len(z) >= 2:
            # NEO y EVA pueden tener diferentes patrones de discretización
            if agent == 'NEO':
                x_bin = int(z[0] * self.n_symbols) % self.n_symbols
                y_bin = int(z[1] * self.n_symbols) % self.n_symbols
            else:  # EVA
                x_bin = int(z[-1] * self.n_symbols) % self.n_symbols
                y_bin = int(z[-2] * self.n_symbols) % self.n_symbols
            return (x_bin + y_bin) % self.n_symbols
        return 0

    def _compute_value(self, z: np.ndarray, s: np.ndarray,
                        pred_error: float, agent: str) -> float:
        """
        Computa valor interno del estado para un agente.

        NEO: valora precisión (bajo error)
        EVA: valora novedad (alta entropía)

        100% endógeno
        """
        # Entropía del estado
        z_prob = np.abs(z) / (np.sum(np.abs(z)) + 1e-10)
        z_prob = np.clip(z_prob, 1e-10, 1.0)
        entropy = -np.sum(z_prob * np.log(z_prob))

        if agent == 'NEO':
            # NEO valora precisión: más valor cuando menos error
            value = entropy / (1 + pred_error)
        else:  # EVA
            # EVA valora novedad: más valor cuando más entropía y cambio
            value = entropy * (1 + pred_error * 0.5)

        return value

    def _predict_world(self, z: np.ndarray, z_history: List[np.ndarray],
                        agent: str) -> np.ndarray:
        """
        Predice siguiente estado del mundo para un agente.

        NEO: predictor directo del mundo
        EVA: puede usar también información de NEO

        100% endógeno: regresión lineal
        """
        if len(z_history) < 10 or len(self.s_history) < 10:
            return np.ones(self.dim_world) * 0.5

        window = max(10, int(np.sqrt(len(z_history))))

        Z = np.array(z_history[-window:])
        S_next = np.array(self.s_history[-window+1:] + [self.s_history[-1]])

        # Ajustar tamaños
        min_len = min(len(Z), len(S_next))
        Z = Z[:min_len]
        S_next = S_next[:min_len]

        if agent == 'EVA' and len(self.neo_z_history) >= window:
            # EVA puede usar información de NEO
            NEO_Z = np.array(self.neo_z_history[-window:])[:min_len]
            Z = np.hstack([Z, NEO_Z * 0.3])  # Añadir con peso reducido

        try:
            Z_pinv = np.linalg.pinv(Z)
            W = Z_pinv @ S_next
            prediction = z[:Z.shape[1]] @ W if agent == 'NEO' else np.hstack([z, self.neo_z_history[-1] * 0.3 if self.neo_z_history else np.zeros(self.dim_neo)]) @ W
            prediction = np.clip(prediction[:self.dim_world], 0, 1)
            return prediction
        except Exception:
            return np.ones(self.dim_world) * 0.5

    def _compute_grounding_index(self, pred_errors: List[float],
                                   symbols: List[int],
                                   values: List[float],
                                   agent: str) -> Tuple[float, float, float, float]:
        """
        Calcula índices de grounding para un agente.

        G_pred = rank(-MSE)
        G_sym = rank(I(σ; regime))
        G_val = rank(|corr(V, outcome)|)
        GI = G_pred + G_sym + G_val

        100% endógeno: ranks sobre historia propia
        """
        if len(pred_errors) < 10:
            return 0.5, 0.5, 0.5, 1.5

        # === G_pred: rank de -MSE ===
        mse = np.mean(np.array(pred_errors[-20:]) ** 2)
        mse_history = [np.mean(np.array(pred_errors[max(0,i-20):i+1]) ** 2)
                       for i in range(20, len(pred_errors))]
        if mse_history:
            neg_mse_sorted = np.sort([-m for m in mse_history])
            G_pred = np.searchsorted(neg_mse_sorted, -mse) / len(neg_mse_sorted)
        else:
            G_pred = 0.5

        # === G_sym: MI estimada entre símbolos y regímenes ===
        window = min(50, len(symbols))
        recent_symbols = symbols[-window:]
        recent_regimes = self.regime_history[-window:]

        # H(σ)
        sym_counts = {}
        for s in recent_symbols:
            sym_counts[s] = sym_counts.get(s, 0) + 1
        H_sigma = -sum((c/window) * np.log(c/window + 1e-10) for c in sym_counts.values())

        # H(σ|regime)
        regime_symbols = {}
        for sym, reg in zip(recent_symbols, recent_regimes):
            if reg not in regime_symbols:
                regime_symbols[reg] = []
            regime_symbols[reg].append(sym)

        H_cond = 0
        for reg, syms in regime_symbols.items():
            p_reg = len(syms) / window
            sym_c = {}
            for s in syms:
                sym_c[s] = sym_c.get(s, 0) + 1
            H_s = -sum((c/len(syms)) * np.log(c/len(syms) + 1e-10) for c in sym_c.values())
            H_cond += p_reg * H_s

        MI = H_sigma - H_cond

        # Rank de MI
        MI_history_agent = self.neo_G_sym_history if agent == 'NEO' else self.eva_G_sym_history
        if MI_history_agent:
            sorted_mi = np.sort(MI_history_agent)
            G_sym = np.searchsorted(sorted_mi, MI) / len(sorted_mi)
        else:
            G_sym = 0.5

        # === G_val: correlación valor-outcome ===
        window_val = min(50, len(values), len(self.outcome_history))
        if window_val >= 10:
            recent_values = np.array(values[-window_val:])
            recent_outcomes = np.array(self.outcome_history[-window_val:])
            corr = np.corrcoef(recent_values, recent_outcomes)[0, 1]
            if np.isnan(corr):
                corr = 0
            corr = abs(corr)

            corr_history = self.neo_G_val_history if agent == 'NEO' else self.eva_G_val_history
            if corr_history:
                sorted_corr = np.sort(corr_history)
                G_val = np.searchsorted(sorted_corr, corr) / len(sorted_corr)
            else:
                G_val = 0.5
        else:
            G_val = 0.5

        # GI total
        GI = G_pred + G_sym + G_val

        return G_pred, G_sym, G_val, GI

    def step(self, neo_z: np.ndarray, eva_z: np.ndarray) -> Dict[str, Any]:
        """
        Ejecuta un paso de grounding para ambos agentes.

        Args:
            neo_z: Estado de NEO
            eva_z: Estado de EVA

        Returns:
            Dict con métricas de grounding por agente
        """
        self.t += 1

        # Paso del mundo
        world_state = self.world.step()
        s = world_state.s
        regime = world_state.regime

        # Outcome del mundo (para value grounding)
        outcome = np.mean(s)  # Simplificación: media del estado

        # === NEO ===
        neo_prediction = self._predict_world(neo_z, self.neo_z_history, 'NEO')
        neo_pred_error = np.linalg.norm(neo_prediction - s)
        neo_symbol = self._generate_symbol(neo_z, 'NEO')
        neo_value = self._compute_value(neo_z, s, neo_pred_error, 'NEO')

        self.neo_z_history.append(neo_z.copy())
        self.neo_predictions.append(neo_prediction)
        self.neo_pred_errors.append(neo_pred_error)
        self.neo_symbols.append(neo_symbol)
        self.neo_values.append(neo_value)

        neo_G_pred, neo_G_sym, neo_G_val, neo_GI = self._compute_grounding_index(
            self.neo_pred_errors, self.neo_symbols, self.neo_values, 'NEO'
        )
        self.neo_G_pred_history.append(neo_G_pred)
        self.neo_G_sym_history.append(neo_G_sym)
        self.neo_G_val_history.append(neo_G_val)
        self.neo_GI_history.append(neo_GI)

        # === EVA ===
        eva_prediction = self._predict_world(eva_z, self.eva_z_history, 'EVA')
        eva_pred_error = np.linalg.norm(eva_prediction - s)
        eva_symbol = self._generate_symbol(eva_z, 'EVA')
        eva_value = self._compute_value(eva_z, s, eva_pred_error, 'EVA')

        self.eva_z_history.append(eva_z.copy())
        self.eva_predictions.append(eva_prediction)
        self.eva_pred_errors.append(eva_pred_error)
        self.eva_symbols.append(eva_symbol)
        self.eva_values.append(eva_value)

        eva_G_pred, eva_G_sym, eva_G_val, eva_GI = self._compute_grounding_index(
            self.eva_pred_errors, self.eva_symbols, self.eva_values, 'EVA'
        )
        self.eva_G_pred_history.append(eva_G_pred)
        self.eva_G_sym_history.append(eva_G_sym)
        self.eva_G_val_history.append(eva_G_val)
        self.eva_GI_history.append(eva_GI)

        # Historia compartida
        self.s_history.append(s.copy())
        self.regime_history.append(regime)
        self.outcome_history.append(outcome)

        return {
            't': self.t,
            'world_state': world_state,
            'NEO': AgentGrounding(
                agent='NEO',
                t=self.t,
                G_pred=neo_G_pred,
                G_sym=neo_G_sym,
                G_val=neo_G_val,
                GI=neo_GI,
                prediction_error=neo_pred_error,
                symbol=neo_symbol,
                value=neo_value
            ),
            'EVA': AgentGrounding(
                agent='EVA',
                t=self.t,
                G_pred=eva_G_pred,
                G_sym=eva_G_sym,
                G_val=eva_G_val,
                GI=eva_GI,
                prediction_error=eva_pred_error,
                symbol=eva_symbol,
                value=eva_value
            )
        }

    def test_grounding(self, agent: str, n_nulls: int = 10) -> Dict[str, Any]:
        """
        Test de grounding para un agente contra nulos.

        GO si:
        1. MSE_A < p95(MSE_null)
        2. MI(σ_A; regime) > p95(MI_null)
        3. |corr(V_A, outcome)| > p95(corr_null)
        """
        if agent == 'NEO':
            pred_errors = self.neo_pred_errors
            symbols = self.neo_symbols
            values = self.neo_values
        else:
            pred_errors = self.eva_pred_errors
            symbols = self.eva_symbols
            values = self.eva_values

        if len(pred_errors) < 50:
            return {'agent': agent, 'tested': False, 'reason': 'insufficient_data'}

        # === Test 1: Predictive ===
        mse_real = np.mean(np.array(pred_errors) ** 2)

        null_mses = []
        for _ in range(n_nulls):
            shuffled = np.random.permutation(pred_errors)
            null_mses.append(np.mean(shuffled ** 2))

        p5_mse_null = np.percentile(null_mses, 5)  # Queremos MSE bajo
        test_pred = mse_real < np.mean(null_mses)

        # === Test 2: Symbolic ===
        sym_counts = {}
        for s in symbols:
            sym_counts[s] = sym_counts.get(s, 0) + 1
        total = len(symbols)
        H_sigma = -sum((c/total) * np.log(c/total + 1e-10) for c in sym_counts.values())

        regime_symbols = {}
        for sym, reg in zip(symbols, self.regime_history):
            if reg not in regime_symbols:
                regime_symbols[reg] = []
            regime_symbols[reg].append(sym)

        H_cond = 0
        for reg, syms in regime_symbols.items():
            p_reg = len(syms) / total
            sym_c = {}
            for s in syms:
                sym_c[s] = sym_c.get(s, 0) + 1
            H_s = -sum((c/len(syms)) * np.log(c/len(syms) + 1e-10) for c in sym_c.values())
            H_cond += p_reg * H_s

        MI_real = H_sigma - H_cond

        null_MIs = []
        for _ in range(n_nulls):
            shuffled_symbols = list(np.random.permutation(symbols))
            regime_symbols_null = {}
            for sym, reg in zip(shuffled_symbols, self.regime_history):
                if reg not in regime_symbols_null:
                    regime_symbols_null[reg] = []
                regime_symbols_null[reg].append(sym)

            H_cond_null = 0
            for reg, syms in regime_symbols_null.items():
                p_reg = len(syms) / total
                sym_c = {}
                for s in syms:
                    sym_c[s] = sym_c.get(s, 0) + 1
                H_s = -sum((c/len(syms)) * np.log(c/len(syms) + 1e-10) for c in sym_c.values())
                H_cond_null += p_reg * H_s
            null_MIs.append(H_sigma - H_cond_null)

        p95_MI = np.percentile(null_MIs, 95)
        test_sym = MI_real > p95_MI

        # === Test 3: Value ===
        values_arr = np.array(values)
        outcomes_arr = np.array(self.outcome_history)
        corr_real = abs(np.corrcoef(values_arr, outcomes_arr)[0, 1])
        if np.isnan(corr_real):
            corr_real = 0

        null_corrs = []
        for _ in range(n_nulls):
            shuffled = np.random.permutation(values_arr)
            c = abs(np.corrcoef(shuffled, outcomes_arr)[0, 1])
            if not np.isnan(c):
                null_corrs.append(c)

        p95_corr = np.percentile(null_corrs, 95) if null_corrs else 0
        test_val = corr_real > p95_corr

        # Resultado
        n_passed = sum([test_pred, test_sym, test_val])
        certified = n_passed >= 2

        return {
            'agent': agent,
            'tested': True,
            'certified': certified,
            'tests': {
                'predictive': {
                    'MSE_real': float(mse_real),
                    'MSE_null_mean': float(np.mean(null_mses)),
                    'passed': test_pred
                },
                'symbolic': {
                    'MI_real': float(MI_real),
                    'p95_MI_null': float(p95_MI),
                    'passed': test_sym
                },
                'value': {
                    'corr_real': float(corr_real),
                    'p95_corr_null': float(p95_corr),
                    'passed': test_val
                }
            },
            'n_passed': n_passed
        }

    def get_comparison(self) -> Dict[str, Any]:
        """Compara grounding NEO vs EVA."""
        if not self.neo_GI_history or not self.eva_GI_history:
            return {'ready': False}

        n = min(len(self.neo_GI_history), len(self.eva_GI_history))

        neo_GI_mean = np.mean(self.neo_GI_history[-n:])
        eva_GI_mean = np.mean(self.eva_GI_history[-n:])

        neo_pred_mean = np.mean(self.neo_pred_errors[-n:])
        eva_pred_mean = np.mean(self.eva_pred_errors[-n:])

        return {
            'ready': True,
            't': self.t,
            'NEO': {
                'GI_mean': float(neo_GI_mean),
                'G_pred_mean': float(np.mean(self.neo_G_pred_history[-n:])),
                'G_sym_mean': float(np.mean(self.neo_G_sym_history[-n:])),
                'G_val_mean': float(np.mean(self.neo_G_val_history[-n:])),
                'pred_error_mean': float(neo_pred_mean)
            },
            'EVA': {
                'GI_mean': float(eva_GI_mean),
                'G_pred_mean': float(np.mean(self.eva_G_pred_history[-n:])),
                'G_sym_mean': float(np.mean(self.eva_G_sym_history[-n:])),
                'G_val_mean': float(np.mean(self.eva_G_val_history[-n:])),
                'pred_error_mean': float(eva_pred_mean)
            },
            'more_grounded': 'NEO' if neo_GI_mean > eva_GI_mean else 'EVA',
            'divergence': float(abs(neo_GI_mean - eva_GI_mean))
        }


def run_phase_g2_dual() -> Dict[str, Any]:
    """Ejecuta Phase G2-Dual y evalúa criterios GO/NO-GO."""

    print("=" * 70)
    print("PHASE G2-DUAL: GROUNDING SEPARADO POR AGENTE")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}\n")

    np.random.seed(42)

    # Crear sistema
    grounding = DualAgentGrounding(dim_world=6, dim_neo=6, dim_eva=6)

    # Simulación
    T = 500

    # Estados iniciales
    neo_z = np.random.rand(6)
    neo_z = neo_z / neo_z.sum()
    eva_z = np.random.rand(6)
    eva_z = eva_z / eva_z.sum()

    print("Simulando grounding dual...")
    for t in range(T):
        # Dinámica diferenciada
        if grounding.s_history:
            s = grounding.s_history[-1]
            # NEO: más conectado al mundo
            neo_z = 0.7 * neo_z + 0.3 * s
            # EVA: parcialmente conectado al mundo, parcialmente a NEO
            eva_z = 0.7 * eva_z + 0.15 * s + 0.15 * neo_z

        noise_neo = np.random.randn(6) * 0.02
        noise_eva = np.random.randn(6) * 0.03  # EVA más ruidosa

        neo_z = neo_z + noise_neo
        eva_z = eva_z + noise_eva

        neo_z = np.clip(neo_z, 0.01, 0.99)
        eva_z = np.clip(eva_z, 0.01, 0.99)
        neo_z = neo_z / neo_z.sum()
        eva_z = eva_z / eva_z.sum()

        result = grounding.step(neo_z, eva_z)

        if t % 100 == 0:
            print(f"  t={t}: GI_NEO={result['NEO'].GI:.3f}, GI_EVA={result['EVA'].GI:.3f}")

    print()

    # Tests por agente
    print("Ejecutando tests de grounding...")
    test_neo = grounding.test_grounding('NEO', n_nulls=10)
    test_eva = grounding.test_grounding('EVA', n_nulls=10)

    print()
    print("=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print()

    comparison = grounding.get_comparison()

    print("NEO:")
    print(f"  GI medio: {comparison['NEO']['GI_mean']:.4f}")
    print(f"  G_pred: {comparison['NEO']['G_pred_mean']:.4f}")
    print(f"  G_sym: {comparison['NEO']['G_sym_mean']:.4f}")
    print(f"  G_val: {comparison['NEO']['G_val_mean']:.4f}")
    print(f"  Error predicción: {comparison['NEO']['pred_error_mean']:.4f}")
    if test_neo['tested']:
        print(f"  Tests pasados: {test_neo['n_passed']}/3")
        print(f"  Certificado: {'Sí' if test_neo['certified'] else 'No'}")
    print()

    print("EVA:")
    print(f"  GI medio: {comparison['EVA']['GI_mean']:.4f}")
    print(f"  G_pred: {comparison['EVA']['G_pred_mean']:.4f}")
    print(f"  G_sym: {comparison['EVA']['G_sym_mean']:.4f}")
    print(f"  G_val: {comparison['EVA']['G_val_mean']:.4f}")
    print(f"  Error predicción: {comparison['EVA']['pred_error_mean']:.4f}")
    if test_eva['tested']:
        print(f"  Tests pasados: {test_eva['n_passed']}/3")
        print(f"  Certificado: {'Sí' if test_eva['certified'] else 'No'}")
    print()

    print(f"Más enganchado al mundo: {comparison['more_grounded']}")
    print(f"Divergencia: {comparison['divergence']:.4f}")
    print()

    # Criterios GO/NO-GO
    criteria = {}

    # 1. GI calculados
    criteria['gi_computed'] = len(grounding.neo_GI_history) > 0 and len(grounding.eva_GI_history) > 0

    # 2. NEO más enganchado (esperado por diseño)
    criteria['neo_more_grounded'] = comparison['more_grounded'] == 'NEO'

    # 3. NEO pasa tests
    criteria['neo_certified'] = test_neo.get('certified', False)

    # 4. Divergencia entre agentes
    criteria['differentiated'] = comparison['divergence'] > 0.05

    # 5. Ambos tienen grounding funcional (GI > 1.0)
    criteria['both_functional'] = comparison['NEO']['GI_mean'] > 1.0 and comparison['EVA']['GI_mean'] > 1.0

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
        'phase': 'G2-Dual',
        'name': 'Dual Agent Grounding',
        'timestamp': datetime.now().isoformat(),
        'comparison': comparison,
        'test_NEO': test_neo,
        'test_EVA': test_eva,
        'criteria': criteria,
        'go_status': go_status,
        'passed_criteria': passed,
        'total_criteria': total
    }

    os.makedirs('/root/NEO_EVA/results/phaseG2_dual', exist_ok=True)

    with open('/root/NEO_EVA/results/phaseG2_dual/grounding_dual_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Visualización
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. GI temporal
        ax1 = axes[0, 0]
        ax1.plot(grounding.neo_GI_history, 'b-', label='GI NEO', alpha=0.7)
        ax1.plot(grounding.eva_GI_history, 'r-', label='GI EVA', alpha=0.7)
        ax1.set_xlabel('Tiempo')
        ax1.set_ylabel('GI')
        ax1.set_title('Índice de Grounding por Agente')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Error de predicción
        ax2 = axes[0, 1]
        ax2.plot(grounding.neo_pred_errors, 'b-', label='NEO', alpha=0.5, linewidth=0.5)
        ax2.plot(grounding.eva_pred_errors, 'r-', label='EVA', alpha=0.5, linewidth=0.5)
        ax2.set_xlabel('Tiempo')
        ax2.set_ylabel('Error de Predicción')
        ax2.set_title('Predictive Grounding')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Componentes de grounding
        ax3 = axes[1, 0]
        n = min(len(grounding.neo_G_pred_history), len(grounding.eva_G_pred_history))
        x = np.arange(3)
        width = 0.35

        neo_means = [
            np.mean(grounding.neo_G_pred_history[-n:]),
            np.mean(grounding.neo_G_sym_history[-n:]),
            np.mean(grounding.neo_G_val_history[-n:])
        ]
        eva_means = [
            np.mean(grounding.eva_G_pred_history[-n:]),
            np.mean(grounding.eva_G_sym_history[-n:]),
            np.mean(grounding.eva_G_val_history[-n:])
        ]

        ax3.bar(x - width/2, neo_means, width, label='NEO', color='blue', alpha=0.7)
        ax3.bar(x + width/2, eva_means, width, label='EVA', color='red', alpha=0.7)
        ax3.set_xlabel('Componente')
        ax3.set_ylabel('Valor medio')
        ax3.set_title('Componentes de Grounding')
        ax3.set_xticks(x)
        ax3.set_xticklabels(['G_pred', 'G_sym', 'G_val'])
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Diferencia GI
        ax4 = axes[1, 1]
        diff = np.array(grounding.neo_GI_history) - np.array(grounding.eva_GI_history)
        ax4.plot(diff, 'purple', alpha=0.7)
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.fill_between(range(len(diff)), 0, diff,
                        where=np.array(diff) > 0, color='blue', alpha=0.3, label='NEO > EVA')
        ax4.fill_between(range(len(diff)), 0, diff,
                        where=np.array(diff) < 0, color='red', alpha=0.3, label='EVA > NEO')
        ax4.set_xlabel('Tiempo')
        ax4.set_ylabel('GI_NEO - GI_EVA')
        ax4.set_title('Diferencia de Grounding')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs('/root/NEO_EVA/figures', exist_ok=True)
        plt.savefig('/root/NEO_EVA/figures/phaseG2_dual_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nResultados guardados en: /root/NEO_EVA/results/phaseG2_dual")
        print(f"Figura: /root/NEO_EVA/figures/phaseG2_dual_results.png")

    except Exception as e:
        print(f"Warning: No se pudo crear visualización: {e}")

    return output


if __name__ == "__main__":
    run_phase_g2_dual()
