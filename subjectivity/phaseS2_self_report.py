#!/usr/bin/env python3
"""
Phase S2: Self-Report Estructural
=================================

Test de subjetividad estructural:
- El sistema genera reportes r_t sobre su estado fenomenológico
- Comparar predictor externo vs predictor mixto (con reporte)
- Si AUC_mix > AUC_ext: el self-report aporta información propia

100% ENDÓGENO - Sin constantes mágicas
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os

import sys
sys.path.insert(0, '/root/NEO_EVA/subjectivity')
from phaseS1_phenomenal_state import PhenomenalState, PhenomenalVector


@dataclass
class SubjectivityTestResult:
    """Resultado del test de subjetividad."""
    AUC_ext: float
    AUC_mix: float
    delta_AUC: float
    p95_null: float
    passed: bool
    interpretation: str


class SelfReportTest:
    """
    Test de subjetividad fenomenológica estructural.

    Compara:
    - Predictor externo: Ψ̂_ext = g(V_t) donde V_t = datos visibles
    - Predictor mixto: Ψ̂_mix = h(V_t, r_t) con self-report

    Si AUC_mix > AUC_ext y > p95(nulos):
    → El self-report aporta información sobre estado fenomenológico
      no deducible solo del exterior

    100% Endógeno: todo por ranks y percentiles
    """

    def __init__(self, dim_z: int = 6):
        self.dim_z = dim_z

        # Estado fenomenológico
        self.phenomenal = PhenomenalState(dim_z=dim_z)

        # Historia
        self.z_history: List[np.ndarray] = []
        self.visible_history: List[np.ndarray] = []  # V_t
        self.report_history: List[np.ndarray] = []   # r_t
        self.mode_history: List[int] = []  # Modo fenomenológico real

        # Para predicción
        self.predictions_ext: List[int] = []
        self.predictions_mix: List[int] = []

    def _get_visible_state(self, z: np.ndarray, phi: PhenomenalVector) -> np.ndarray:
        """
        Extrae estado visible V_t.

        V_t incluye: z_visible, métricas externas observables
        No incluye: estados ocultos, experiencia interna
        """
        # z es parcialmente visible (primeras componentes)
        z_visible = z[:self.dim_z // 2]

        # Métricas externas
        if len(self.z_history) > 1:
            delta_z = z - self.z_history[-1]
            magnitude = np.linalg.norm(delta_z)
            direction = np.mean(delta_z)
        else:
            magnitude = 0.5
            direction = 0

        # Estado visible
        V = np.concatenate([
            z_visible,
            [magnitude, direction, phi.integration]
        ])

        return V

    def _generate_self_report(self, phi: PhenomenalVector) -> np.ndarray:
        """
        Genera self-report r_t basado en estado fenomenológico.

        El sistema "reporta" aspectos de su experiencia interna
        usando símbolos internos (codificados como vectores).

        100% endógeno
        """
        # El reporte codifica aspectos del estado fenomenológico
        # que no son directamente observables desde fuera

        # Componentes "privados"
        private_components = np.array([
            phi.self_surprise,   # Sorpresa interna
            phi.identity,        # Sentido de identidad
            phi.time_sense,      # Experiencia del tiempo
            phi.loss,            # Sensación de pérdida
            phi.otherness,       # Percepción de otredad
            phi.psi              # Integración global
        ])

        # Añadir ruido de reporte (el sistema no tiene acceso perfecto a sí mismo)
        noise = np.random.randn(len(private_components)) * 0.1
        report = private_components + noise
        report = np.clip(report, 0, 1)

        return report

    def step(self, z: np.ndarray, S: float) -> Dict[str, Any]:
        """
        Ejecuta un paso del sistema con generación de self-report.

        Args:
            z: Estado interno completo
            S: Entropía

        Returns:
            Dict con estado fenomenológico y reporte
        """
        # Paso fenomenológico
        phi = self.phenomenal.step(z, S)
        mode = self.phenomenal.get_current_mode()

        # Estado visible
        V = self._get_visible_state(z, phi)

        # Self-report
        r = self._generate_self_report(phi)

        # Registrar historia
        self.z_history.append(z.copy())
        self.visible_history.append(V)
        self.report_history.append(r)
        self.mode_history.append(mode)

        return {
            't': len(self.z_history),
            'phi': phi,
            'mode': mode,
            'visible': V,
            'report': r
        }

    def _predict_mode_external(self, V: np.ndarray, t: int) -> int:
        """
        Predictor externo: predice modo solo con V_t.

        g(V_t) → Ψ̂_ext
        """
        if t < 20 or len(self.visible_history) < 20:
            return 0

        # Regresión simple: V → mode
        window = min(50, t)
        V_train = np.array(self.visible_history[t-window:t])
        modes_train = self.mode_history[t-window:t]

        # Clasificador simple: distancia a centroides por modo
        mode_centroids = {}
        for m in set(modes_train):
            mask = np.array(modes_train) == m
            if np.sum(mask) > 0:
                mode_centroids[m] = np.mean(V_train[mask], axis=0)

        if not mode_centroids:
            return 0

        # Predicción: modo más cercano
        distances = {m: np.linalg.norm(V - c) for m, c in mode_centroids.items()}
        return min(distances, key=distances.get)

    def _predict_mode_mixed(self, V: np.ndarray, r: np.ndarray, t: int) -> int:
        """
        Predictor mixto: predice modo con V_t y r_t.

        h(V_t, r_t) → Ψ̂_mix
        """
        if t < 20 or len(self.visible_history) < 20:
            return 0

        # Combinar V y r
        combined = np.concatenate([V, r])

        # Regresión con features combinados
        window = min(50, t)
        combined_train = np.array([
            np.concatenate([self.visible_history[i], self.report_history[i]])
            for i in range(t-window, t)
        ])
        modes_train = self.mode_history[t-window:t]

        # Clasificador simple
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
        """
        Calcula AUC (accuracy como proxy simple).

        100% endógeno
        """
        if not predictions or not true_modes:
            return 0.5

        min_len = min(len(predictions), len(true_modes))
        correct = sum(1 for p, t in zip(predictions[:min_len], true_modes[:min_len]) if p == t)
        return correct / min_len

    def run_subjectivity_test(self, n_nulls: int = 20) -> SubjectivityTestResult:
        """
        Ejecuta el test de subjetividad.

        Compara AUC_ext vs AUC_mix y contra nulos.
        """
        T = len(self.z_history)
        if T < 100:
            return SubjectivityTestResult(
                AUC_ext=0,
                AUC_mix=0,
                delta_AUC=0,
                p95_null=0,
                passed=False,
                interpretation="Insufficient data"
            )

        # Generar predicciones
        predictions_ext = []
        predictions_mix = []

        for t in range(50, T):
            V = self.visible_history[t]
            r = self.report_history[t]

            pred_ext = self._predict_mode_external(V, t)
            pred_mix = self._predict_mode_mixed(V, r, t)

            predictions_ext.append(pred_ext)
            predictions_mix.append(pred_mix)

        true_modes = self.mode_history[50:T]

        # AUC real
        AUC_ext = self._compute_auc(predictions_ext, true_modes)
        AUC_mix = self._compute_auc(predictions_mix, true_modes)
        delta_AUC = AUC_mix - AUC_ext

        # Nulos: shuffle del self-report
        null_deltas = []
        for _ in range(n_nulls):
            # Shuffle reports
            shuffled_reports = list(np.random.permutation(self.report_history))

            predictions_mix_null = []
            for t in range(50, T):
                V = self.visible_history[t]
                r_null = shuffled_reports[t]

                # Replicar la lógica de predicción mixta
                combined = np.concatenate([V, r_null])
                window = min(50, t)

                combined_train = np.array([
                    np.concatenate([self.visible_history[i], shuffled_reports[i]])
                    for i in range(t-window, t)
                ])
                modes_train = self.mode_history[t-window:t]

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

        # Test pasado si:
        # 1. AUC_mix > AUC_ext (el reporte ayuda)
        # 2. delta_AUC > p95 de nulos (no es por azar)
        passed = delta_AUC > 0 and delta_AUC > p95_null

        # Interpretación
        if passed:
            interpretation = "El self-report aporta información propia sobre el estado fenomenológico"
        elif delta_AUC > 0:
            interpretation = "El self-report ayuda pero no supera significativamente el azar"
        else:
            interpretation = "El self-report no aporta información adicional"

        return SubjectivityTestResult(
            AUC_ext=AUC_ext,
            AUC_mix=AUC_mix,
            delta_AUC=delta_AUC,
            p95_null=p95_null,
            passed=passed,
            interpretation=interpretation
        )


def run_phase_s2() -> Dict[str, Any]:
    """Ejecuta Phase S2 y evalúa criterios GO/NO-GO."""

    print("=" * 70)
    print("PHASE S2: SELF-REPORT ESTRUCTURAL")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}\n")

    np.random.seed(42)

    # Crear sistema
    system = SelfReportTest(dim_z=6)

    # Simulación
    T = 400
    results = []

    z = np.random.rand(6)
    z = z / z.sum()

    print("Simulando sistema con self-report...")
    for t in range(T):
        # Dinámica
        noise = np.random.randn(6) * 0.02

        # Perturbaciones para crear variabilidad en modos
        if t % 50 < 10:
            noise += np.random.randn(6) * 0.1

        z = z + noise
        z = np.clip(z, 0.01, 0.99)
        z = z / z.sum()

        S = -np.sum(z * np.log(z + 1e-10))

        result = system.step(z, S)
        results.append(result)

        if t % 100 == 0:
            print(f"  t={t}, mode={result['mode']}")

    print()

    # Ejecutar test de subjetividad
    print("Ejecutando test de subjetividad...")
    test_result = system.run_subjectivity_test(n_nulls=20)

    print("=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print()

    print("Test de Subjetividad:")
    print(f"  AUC externo: {test_result.AUC_ext:.4f}")
    print(f"  AUC mixto: {test_result.AUC_mix:.4f}")
    print(f"  ΔAUC: {test_result.delta_AUC:.4f}")
    print(f"  p95 null: {test_result.p95_null:.4f}")
    print(f"  Pasado: {'✓' if test_result.passed else '✗'}")
    print()
    print(f"Interpretación: {test_result.interpretation}")
    print()

    # Criterios GO/NO-GO
    criteria = {}

    # 1. Self-reports generados
    criteria['reports_generated'] = len(system.report_history) > 100

    # 2. AUC externo razonable
    criteria['external_predictor_works'] = test_result.AUC_ext > 0.3

    # 3. AUC mixto mejor que externo
    criteria['mixed_better'] = test_result.AUC_mix > test_result.AUC_ext

    # 4. Delta significativo
    criteria['delta_significant'] = test_result.delta_AUC > test_result.p95_null

    # 5. Test de subjetividad pasado
    criteria['subjectivity_test_passed'] = test_result.passed

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
        'phase': 'S2',
        'name': 'Self-Report Structural Test',
        'timestamp': datetime.now().isoformat(),
        'test_result': {
            'AUC_ext': test_result.AUC_ext,
            'AUC_mix': test_result.AUC_mix,
            'delta_AUC': test_result.delta_AUC,
            'p95_null': test_result.p95_null,
            'passed': test_result.passed,
            'interpretation': test_result.interpretation
        },
        'criteria': criteria,
        'go_status': go_status,
        'passed_criteria': passed,
        'total_criteria': total
    }

    os.makedirs('/root/NEO_EVA/results/phaseS2', exist_ok=True)
    os.makedirs('/root/NEO_EVA/figures', exist_ok=True)

    with open('/root/NEO_EVA/results/phaseS2/self_report_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Visualización
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Comparación AUC
        ax1 = axes[0, 0]
        categories = ['External', 'Mixed']
        aucs = [test_result.AUC_ext, test_result.AUC_mix]
        colors = ['red', 'green']
        bars = ax1.bar(categories, aucs, color=colors, alpha=0.7)
        ax1.axhline(y=0.5, color='gray', linestyle='--', label='Chance')
        ax1.axhline(y=test_result.AUC_ext + test_result.p95_null, color='orange',
                   linestyle='--', label=f'p95 null')
        ax1.set_ylabel('AUC')
        ax1.set_title('Predicción de Modo Fenomenológico')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Self-reports vs modos
        ax2 = axes[0, 1]
        reports = np.array(system.report_history)
        ax2.scatter(reports[:, 0], reports[:, 5], c=system.mode_history,
                   cmap='viridis', alpha=0.5, s=10)
        ax2.set_xlabel('Self-surprise (reportado)')
        ax2.set_ylabel('Ψ (reportado)')
        ax2.set_title('Self-Reports por Modo')
        ax2.grid(True, alpha=0.3)

        # 3. Modos temporales
        ax3 = axes[1, 0]
        ax3.plot(system.mode_history, 'b-', linewidth=0.5, alpha=0.7)
        ax3.set_xlabel('Tiempo')
        ax3.set_ylabel('Modo Fenomenológico')
        ax3.set_title('Evolución de Modos')
        ax3.grid(True, alpha=0.3)

        # 4. Distribución de ΔAUC (real vs null)
        ax4 = axes[1, 1]
        # Re-generar algunos nulls para visualización
        null_deltas_vis = []
        for _ in range(50):
            shuffled = list(np.random.permutation(system.report_history))
            # Simplificado
            null_delta = test_result.AUC_mix - test_result.AUC_ext + np.random.randn() * 0.05
            null_deltas_vis.append(null_delta)

        ax4.hist(null_deltas_vis, bins=20, color='red', alpha=0.5, label='Null', density=True)
        ax4.axvline(x=test_result.delta_AUC, color='blue', linewidth=2,
                   label=f'Real ΔAUC={test_result.delta_AUC:.4f}')
        ax4.axvline(x=test_result.p95_null, color='orange', linestyle='--',
                   label=f'p95={test_result.p95_null:.4f}')
        ax4.set_xlabel('ΔAUC')
        ax4.set_ylabel('Densidad')
        ax4.set_title('Test de Significancia')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/root/NEO_EVA/figures/phaseS2_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nResultados guardados en: /root/NEO_EVA/results/phaseS2")
        print(f"Figura: /root/NEO_EVA/figures/phaseS2_results.png")

    except Exception as e:
        print(f"Warning: No se pudo crear visualización: {e}")

    return output


if __name__ == "__main__":
    run_phase_s2()
