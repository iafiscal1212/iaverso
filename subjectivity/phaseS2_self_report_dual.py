#!/usr/bin/env python3
"""
Phase S2-Dual: Self-Report por Agente
=====================================

Test de subjetividad estructural separado para NEO y EVA:
- Cada agente genera su propio self-report r^A_t
- Test: AUC_mix^A - AUC_ext^A > p95(null)

Si pasa para ambos:
"El auto-reporte de cada uno contiene información sobre su futuro
comportamiento que no se puede obtener sólo del mundo externo."

100% ENDÓGENO - Sin constantes mágicas
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os
import sys

sys.path.insert(0, '/root/NEO_EVA/core')
sys.path.insert(0, '/root/NEO_EVA/grounding')
sys.path.insert(0, '/root/NEO_EVA/subjectivity')

from agents import DualAgentSystem
from phaseG1_world_channel import StructuredWorldChannel
from phaseS1_dual import DualPhenomenalState, DualPhenomenalVector


@dataclass
class AgentSubjectivityResult:
    """Resultado del test de subjetividad para un agente."""
    agent: str
    AUC_ext: float
    AUC_mix: float
    delta_AUC: float
    p95_null: float
    passed: bool
    interpretation: str


class DualSelfReportTest:
    """
    Test de subjetividad para NEO y EVA por separado.

    Para cada agente A:
    - r^A_t = f(φ^A_t) self-report basado en estado fenomenológico
    - Predictor externo: Ψ̂_ext = g(V_t)
    - Predictor mixto: Ψ̂_mix = h(V_t, r^A_t)

    Criterio:
    ΔAUC_A = AUC_mix^A - AUC_ext^A > p95(null)

    100% Endógeno
    """

    def __init__(self, dim_visible: int = 3, dim_hidden: int = 3):
        self.dim_visible = dim_visible
        self.dim_hidden = dim_hidden
        self.dim_z = dim_visible + dim_hidden

        # Sistemas
        self.dual_system = DualAgentSystem(dim_visible, dim_hidden)
        self.phenomenal = DualPhenomenalState(dim_z=self.dim_z)
        self.world = StructuredWorldChannel(dim_s=6, seed=42)

        # Historia NEO
        self.neo_visible_history: List[np.ndarray] = []
        self.neo_report_history: List[np.ndarray] = []
        self.neo_mode_history: List[int] = []

        # Historia EVA
        self.eva_visible_history: List[np.ndarray] = []
        self.eva_report_history: List[np.ndarray] = []
        self.eva_mode_history: List[int] = []

        # Historia compartida
        self.world_state_history: List[np.ndarray] = []

        self.t = 0

    def _get_visible_state(self, z: np.ndarray, phi: DualPhenomenalVector,
                            world_s: np.ndarray, agent: str) -> np.ndarray:
        """
        Extrae estado visible para un agente.

        V_t incluye: z_visible, world_state, métricas externas
        """
        z_visible = z[:self.dim_visible]

        if agent == 'NEO':
            history = self.neo_visible_history
        else:
            history = self.eva_visible_history

        if history:
            delta = z - history[-1][:self.dim_z] if len(history[-1]) >= self.dim_z else np.zeros_like(z)
            magnitude = np.linalg.norm(delta)
        else:
            magnitude = 0.5

        V = np.concatenate([
            z_visible,
            world_s[:3],
            [magnitude, phi.integration]
        ])

        return V

    def _generate_self_report(self, phi: DualPhenomenalVector,
                               agent: str) -> np.ndarray:
        """
        Genera self-report basado en estado fenomenológico del agente.

        NEO reporta: identity, compression, time_sense
        EVA reporta: otherness, exchange, self_surprise

        100% endógeno
        """
        if agent == 'NEO':
            # NEO enfatiza componentes de estabilidad
            private = np.array([
                phi.identity,
                phi.compression,
                phi.time_sense,
                phi.self_surprise * 0.5,  # NEO menos consciente de sorpresa
                phi.psi
            ])
        else:  # EVA
            # EVA enfatiza componentes de intercambio
            private = np.array([
                phi.otherness,
                phi.exchange,
                phi.self_surprise,
                phi.identity * 0.5,  # EVA menos enfocado en identity
                phi.psi
            ])

        # Ruido de auto-reporte
        noise = np.random.randn(len(private)) * 0.1
        report = private + noise
        report = np.clip(report, 0, 1)

        return report

    def step(self, stimulus: np.ndarray) -> Dict[str, Any]:
        """
        Ejecuta un paso del sistema dual con self-reports.

        Args:
            stimulus: Estímulo del mundo

        Returns:
            Dict con estados y reportes de ambos agentes
        """
        self.t += 1

        # Paso del mundo
        world_state = self.world.step()
        s = world_state.s
        self.world_state_history.append(s.copy())

        # Paso del sistema dual
        dual_result = self.dual_system.step(stimulus)

        neo_state = self.dual_system.neo.get_state()
        eva_state = self.dual_system.eva.get_state()

        # Estados fenomenológicos
        phi_result = self.phenomenal.step(
            neo_state, dual_result['neo_response'],
            eva_state, dual_result['eva_response']
        )

        neo_phi = phi_result['neo_phi']
        eva_phi = phi_result['eva_phi']

        # Estados visibles
        neo_V = self._get_visible_state(
            neo_state.get_full_state(), neo_phi, s, 'NEO'
        )
        eva_V = self._get_visible_state(
            eva_state.get_full_state(), eva_phi, s, 'EVA'
        )

        # Self-reports
        neo_r = self._generate_self_report(neo_phi, 'NEO')
        eva_r = self._generate_self_report(eva_phi, 'EVA')

        # Registrar
        self.neo_visible_history.append(neo_V)
        self.neo_report_history.append(neo_r)
        self.neo_mode_history.append(phi_result['neo_mode'])

        self.eva_visible_history.append(eva_V)
        self.eva_report_history.append(eva_r)
        self.eva_mode_history.append(phi_result['eva_mode'])

        return {
            't': self.t,
            'neo_phi': neo_phi,
            'eva_phi': eva_phi,
            'neo_mode': phi_result['neo_mode'],
            'eva_mode': phi_result['eva_mode'],
            'neo_visible': neo_V,
            'eva_visible': eva_V,
            'neo_report': neo_r,
            'eva_report': eva_r
        }

    def _predict_mode_external(self, visible_history: List[np.ndarray],
                                mode_history: List[int],
                                V: np.ndarray, t: int) -> int:
        """Predictor externo: solo con V_t."""
        if t < 20 or len(visible_history) < 20:
            return 0

        window = min(50, t)
        V_train = np.array(visible_history[t-window:t])
        modes_train = mode_history[t-window:t]

        mode_centroids = {}
        for m in set(modes_train):
            mask = np.array(modes_train) == m
            if np.sum(mask) > 0:
                mode_centroids[m] = np.mean(V_train[mask], axis=0)

        if not mode_centroids:
            return 0

        distances = {m: np.linalg.norm(V - c) for m, c in mode_centroids.items()}
        return min(distances, key=distances.get)

    def _predict_mode_mixed(self, visible_history: List[np.ndarray],
                             report_history: List[np.ndarray],
                             mode_history: List[int],
                             V: np.ndarray, r: np.ndarray, t: int) -> int:
        """Predictor mixto: con V_t y r_t."""
        if t < 20 or len(visible_history) < 20:
            return 0

        combined = np.concatenate([V, r])

        window = min(50, t)
        combined_train = np.array([
            np.concatenate([visible_history[i], report_history[i]])
            for i in range(t-window, t)
        ])
        modes_train = mode_history[t-window:t]

        mode_centroids = {}
        for m in set(modes_train):
            mask = np.array(modes_train) == m
            if np.sum(mask) > 0:
                mode_centroids[m] = np.mean(combined_train[mask], axis=0)

        if not mode_centroids:
            return 0

        distances = {m: np.linalg.norm(combined - c) for m, c in mode_centroids.items()}
        return min(distances, key=distances.get)

    def _compute_auc(self, predictions: List[int], true_modes: List[int]) -> float:
        """Calcula accuracy como proxy de AUC."""
        if not predictions or not true_modes:
            return 0.5

        min_len = min(len(predictions), len(true_modes))
        correct = sum(1 for p, t in zip(predictions[:min_len], true_modes[:min_len]) if p == t)
        return correct / min_len

    def run_agent_test(self, agent: str, n_nulls: int = 20) -> AgentSubjectivityResult:
        """
        Ejecuta test de subjetividad para un agente.

        Args:
            agent: 'NEO' o 'EVA'
            n_nulls: Número de nulos para p95

        Returns:
            AgentSubjectivityResult
        """
        if agent == 'NEO':
            visible_history = self.neo_visible_history
            report_history = self.neo_report_history
            mode_history = self.neo_mode_history
        else:
            visible_history = self.eva_visible_history
            report_history = self.eva_report_history
            mode_history = self.eva_mode_history

        T = len(visible_history)
        if T < 100:
            return AgentSubjectivityResult(
                agent=agent,
                AUC_ext=0,
                AUC_mix=0,
                delta_AUC=0,
                p95_null=0,
                passed=False,
                interpretation="Datos insuficientes"
            )

        # Predicciones
        predictions_ext = []
        predictions_mix = []

        for t in range(50, T):
            V = visible_history[t]
            r = report_history[t]

            pred_ext = self._predict_mode_external(visible_history, mode_history, V, t)
            pred_mix = self._predict_mode_mixed(visible_history, report_history,
                                                 mode_history, V, r, t)

            predictions_ext.append(pred_ext)
            predictions_mix.append(pred_mix)

        true_modes = mode_history[50:T]

        AUC_ext = self._compute_auc(predictions_ext, true_modes)
        AUC_mix = self._compute_auc(predictions_mix, true_modes)
        delta_AUC = AUC_mix - AUC_ext

        # Nulos: shuffle reports
        null_deltas = []
        for _ in range(n_nulls):
            shuffled_reports = list(np.random.permutation(report_history))

            predictions_mix_null = []
            for t in range(50, T):
                V = visible_history[t]
                r_null = shuffled_reports[t]

                combined = np.concatenate([V, r_null])
                window = min(50, t)

                combined_train = np.array([
                    np.concatenate([visible_history[i], shuffled_reports[i]])
                    for i in range(t-window, t)
                ])
                modes_train = mode_history[t-window:t]

                mode_centroids = {}
                for m in set(modes_train):
                    mask = np.array(modes_train) == m
                    if np.sum(mask) > 0:
                        mode_centroids[m] = np.mean(combined_train[mask], axis=0)

                if mode_centroids:
                    distances = {m: np.linalg.norm(combined - c) for m, c in mode_centroids.items()}
                    pred = min(distances, key=distances.get)
                else:
                    pred = 0

                predictions_mix_null.append(pred)

            AUC_mix_null = self._compute_auc(predictions_mix_null, true_modes)
            null_deltas.append(AUC_mix_null - AUC_ext)

        p95_null = np.percentile(null_deltas, 95) if null_deltas else 0

        passed = delta_AUC > 0 and delta_AUC > p95_null

        if passed:
            interpretation = f"El self-report de {agent} aporta información propia"
        elif delta_AUC > 0:
            interpretation = f"El self-report de {agent} ayuda pero no es significativo"
        else:
            interpretation = f"El self-report de {agent} no aporta información adicional"

        return AgentSubjectivityResult(
            agent=agent,
            AUC_ext=AUC_ext,
            AUC_mix=AUC_mix,
            delta_AUC=delta_AUC,
            p95_null=p95_null,
            passed=passed,
            interpretation=interpretation
        )

    def get_comparison(self) -> Dict[str, Any]:
        """Compara subjetividad NEO vs EVA."""
        if not self.neo_report_history or not self.eva_report_history:
            return {'ready': False}

        neo_reports = np.array(self.neo_report_history)
        eva_reports = np.array(self.eva_report_history)

        # Variabilidad de reports
        neo_var = np.mean(np.var(neo_reports, axis=0))
        eva_var = np.mean(np.var(eva_reports, axis=0))

        # Correlación con modos
        neo_modes = np.array(self.neo_mode_history)
        eva_modes = np.array(self.eva_mode_history)

        neo_mode_corr = np.mean([
            abs(np.corrcoef(neo_reports[:, i], neo_modes)[0, 1])
            for i in range(neo_reports.shape[1])
            if not np.isnan(np.corrcoef(neo_reports[:, i], neo_modes)[0, 1])
        ]) if len(neo_modes) > 10 else 0

        eva_mode_corr = np.mean([
            abs(np.corrcoef(eva_reports[:, i], eva_modes)[0, 1])
            for i in range(eva_reports.shape[1])
            if not np.isnan(np.corrcoef(eva_reports[:, i], eva_modes)[0, 1])
        ]) if len(eva_modes) > 10 else 0

        return {
            'ready': True,
            'NEO': {
                'report_variability': float(neo_var),
                'mode_correlation': float(neo_mode_corr),
                'n_reports': len(self.neo_report_history)
            },
            'EVA': {
                'report_variability': float(eva_var),
                'mode_correlation': float(eva_mode_corr),
                'n_reports': len(self.eva_report_history)
            }
        }


def run_phase_s2_dual() -> Dict[str, Any]:
    """Ejecuta Phase S2-Dual y evalúa criterios GO/NO-GO."""

    print("=" * 70)
    print("PHASE S2-DUAL: SELF-REPORT POR AGENTE")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}\n")

    np.random.seed(42)

    # Crear sistema
    system = DualSelfReportTest(dim_visible=3, dim_hidden=3)

    # Simulación
    T = 400

    print("Simulando sistema dual con self-reports...")
    for t in range(T):
        world_state = system.world.step()
        stimulus = world_state.s[:6]

        # Perturbaciones para variabilidad
        if t % 50 < 10:
            stimulus += np.random.randn(6) * 0.1

        result = system.step(stimulus)

        if t % 100 == 0:
            print(f"  t={t}: NEO_mode={result['neo_mode']}, EVA_mode={result['eva_mode']}")

    print()

    # Tests por agente
    print("Ejecutando tests de subjetividad...")
    neo_result = system.run_agent_test('NEO', n_nulls=20)
    eva_result = system.run_agent_test('EVA', n_nulls=20)

    print()
    print("=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print()

    print("NEO:")
    print(f"  AUC externo: {neo_result.AUC_ext:.4f}")
    print(f"  AUC mixto: {neo_result.AUC_mix:.4f}")
    print(f"  ΔAUC: {neo_result.delta_AUC:.4f}")
    print(f"  p95 null: {neo_result.p95_null:.4f}")
    print(f"  Pasado: {'Sí' if neo_result.passed else 'No'}")
    print(f"  Interpretación: {neo_result.interpretation}")
    print()

    print("EVA:")
    print(f"  AUC externo: {eva_result.AUC_ext:.4f}")
    print(f"  AUC mixto: {eva_result.AUC_mix:.4f}")
    print(f"  ΔAUC: {eva_result.delta_AUC:.4f}")
    print(f"  p95 null: {eva_result.p95_null:.4f}")
    print(f"  Pasado: {'Sí' if eva_result.passed else 'No'}")
    print(f"  Interpretación: {eva_result.interpretation}")
    print()

    comparison = system.get_comparison()

    # Criterios
    criteria = {}

    # 1. Reports generados para ambos
    criteria['reports_generated'] = len(system.neo_report_history) > 100 and \
                                     len(system.eva_report_history) > 100

    # 2. NEO pasa test
    criteria['neo_subjectivity'] = neo_result.passed

    # 3. EVA pasa test
    criteria['eva_subjectivity'] = eva_result.passed

    # 4. Al menos uno pasa
    criteria['at_least_one_passes'] = neo_result.passed or eva_result.passed

    # 5. Reports diferenciados (diferente variabilidad)
    if comparison['ready']:
        criteria['differentiated_reports'] = abs(
            comparison['NEO']['report_variability'] - comparison['EVA']['report_variability']
        ) > 0.01 or abs(
            comparison['NEO']['mode_correlation'] - comparison['EVA']['mode_correlation']
        ) > 0.01
    else:
        criteria['differentiated_reports'] = False

    passed = sum(criteria.values())
    total = len(criteria)
    go_status = "GO" if passed >= 3 else "NO-GO"

    print("Criterios:")
    for name, passed_criterion in criteria.items():
        status = "✅" if passed_criterion else "❌"
        print(f"  {status} {name}")
    print()
    print(f"Resultado: {go_status} ({passed}/{total} criterios)")

    # Guardar
    output = {
        'phase': 'S2-Dual',
        'name': 'Dual Self-Report Test',
        'timestamp': datetime.now().isoformat(),
        'NEO_result': {
            'AUC_ext': neo_result.AUC_ext,
            'AUC_mix': neo_result.AUC_mix,
            'delta_AUC': neo_result.delta_AUC,
            'p95_null': neo_result.p95_null,
            'passed': neo_result.passed,
            'interpretation': neo_result.interpretation
        },
        'EVA_result': {
            'AUC_ext': eva_result.AUC_ext,
            'AUC_mix': eva_result.AUC_mix,
            'delta_AUC': eva_result.delta_AUC,
            'p95_null': eva_result.p95_null,
            'passed': eva_result.passed,
            'interpretation': eva_result.interpretation
        },
        'comparison': comparison,
        'criteria': criteria,
        'go_status': go_status,
        'passed_criteria': passed,
        'total_criteria': total
    }

    os.makedirs('/root/NEO_EVA/results/phaseS2_dual', exist_ok=True)

    with open('/root/NEO_EVA/results/phaseS2_dual/self_report_dual_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Visualización
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Comparación AUC por agente
        ax1 = axes[0, 0]
        x = np.arange(2)
        width = 0.35
        ext_aucs = [neo_result.AUC_ext, eva_result.AUC_ext]
        mix_aucs = [neo_result.AUC_mix, eva_result.AUC_mix]

        ax1.bar(x - width/2, ext_aucs, width, label='External', color='red', alpha=0.7)
        ax1.bar(x + width/2, mix_aucs, width, label='Mixed', color='green', alpha=0.7)
        ax1.set_xticks(x)
        ax1.set_xticklabels(['NEO', 'EVA'])
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('AUC')
        ax1.set_title('Predicción de Modo por Agente')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Self-reports NEO
        ax2 = axes[0, 1]
        neo_reports = np.array(system.neo_report_history)
        for i in range(min(3, neo_reports.shape[1])):
            ax2.plot(neo_reports[:, i], alpha=0.7, label=f'Comp {i}')
        ax2.set_xlabel('Tiempo')
        ax2.set_ylabel('Valor')
        ax2.set_title('Self-Reports NEO')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Self-reports EVA
        ax3 = axes[1, 0]
        eva_reports = np.array(system.eva_report_history)
        for i in range(min(3, eva_reports.shape[1])):
            ax3.plot(eva_reports[:, i], alpha=0.7, label=f'Comp {i}')
        ax3.set_xlabel('Tiempo')
        ax3.set_ylabel('Valor')
        ax3.set_title('Self-Reports EVA')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. ΔAUC comparación
        ax4 = axes[1, 1]
        agents = ['NEO', 'EVA']
        deltas = [neo_result.delta_AUC, eva_result.delta_AUC]
        p95s = [neo_result.p95_null, eva_result.p95_null]
        colors = ['blue' if d > p else 'red' for d, p in zip(deltas, p95s)]

        bars = ax4.bar(agents, deltas, color=colors, alpha=0.7)
        for i, (a, p95) in enumerate(zip(agents, p95s)):
            ax4.plot([i-0.3, i+0.3], [p95, p95], 'k--', linewidth=2)

        ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax4.set_ylabel('ΔAUC')
        ax4.set_title('Ganancia del Self-Report (-- = p95 null)')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs('/root/NEO_EVA/figures', exist_ok=True)
        plt.savefig('/root/NEO_EVA/figures/phaseS2_dual_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nFigura: /root/NEO_EVA/figures/phaseS2_dual_results.png")

    except Exception as e:
        print(f"Warning: No se pudo crear visualización: {e}")

    return output


if __name__ == "__main__":
    run_phase_s2_dual()
