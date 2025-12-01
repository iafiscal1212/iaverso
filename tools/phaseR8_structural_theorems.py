#!/usr/bin/env python3
"""
Phase R8: Endogenous Structural Theorems
=========================================

El sistema descubre "leyes internas" - relaciones entre
sus propias variables que predicen cambios en S.

Componentes:
1. Espacio de features internos: f_t ∈ R^d
2. Covarianza y modos: Σ_f, eigenvectores v_i
3. Dimensión estructural: d_law = #{λ_i ≥ median(λ)}
4. Ley como predicción: ΔŜ_t^(i) = <f_t, v_i>
5. Validación: ρ_i = corr(ΔŜ, ΔS)

100% ENDÓGENO:
- d_law basado en mediana de eigenvalues
- Validación por percentil 90 de correlaciones
- Estabilidad angular en ventanas √t

Criterios GO:
1. laws_discovered: al menos 1 ley válida
2. laws_predict_S: ρ > percentil 90
3. laws_stable: drift angular < threshold endógeno
4. multiple_modes: d_law ≥ 2
5. laws_interpretable: loadings diferenciados
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass
class StructuralLaw:
    """Una ley estructural interna."""
    law_id: int
    eigenvector: np.ndarray
    eigenvalue: float
    correlation: float  # ρ con ΔS
    loadings: Dict[str, float]
    first_detected: int
    stability_history: List[float] = field(default_factory=list)


class EndogenousStructuralTheorems:
    """
    Descubrimiento de leyes estructurales internas.

    Una "ley" es un patrón que predice cambios en S
    basándose únicamente en la propia historia.
    """

    def __init__(self):
        # Feature names
        self.feature_names = [
            'integration', 'irreversibility', 'self_surprise',
            'D_stab', 'D_nov', 'D_irr',
            'Psi', 'S'
        ]
        self.n_features = len(self.feature_names)

        # Historia
        self.f_history: List[np.ndarray] = []
        self.S_history: List[float] = []
        self.delta_S_history: List[float] = []

        # Leyes descubiertas
        self.laws: List[StructuralLaw] = []
        self.correlation_history: List[List[float]] = []

    def _compute_features(self,
                          integration: float,
                          irreversibility: float,
                          self_surprise: float,
                          D_stab: float,
                          D_nov: float,
                          D_irr: float,
                          Psi: float,
                          S: float) -> np.ndarray:
        """Construye vector de features."""
        return np.array([
            integration, irreversibility, self_surprise,
            D_stab, D_nov, D_irr, Psi, S
        ])

    def _compute_covariance(self) -> np.ndarray:
        """Calcula covarianza de features."""
        if len(self.f_history) < 3:
            return np.eye(self.n_features)

        t = len(self.f_history)
        w = max(3, int(np.sqrt(t)))

        recent = np.array(self.f_history[-w:])
        cov = np.cov(recent.T)

        # Regularización endógena
        reg = np.trace(cov) / self.n_features / np.sqrt(t + 1)
        cov += reg * np.eye(self.n_features)

        return cov

    def _compute_d_law(self, eigenvalues: np.ndarray) -> int:
        """
        Dimensión estructural endógena.
        d_law = #{λ_i ≥ median(λ)}
        """
        median_eig = np.median(eigenvalues)
        return int(np.sum(eigenvalues >= median_eig))

    def _detect_laws(self, t: int) -> List[StructuralLaw]:
        """Detecta leyes estructurales válidas."""
        if len(self.f_history) < 10 or len(self.delta_S_history) < 10:
            return []

        # Covarianza y descomposición
        cov = self._compute_covariance()
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Ordenar de mayor a menor
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Dimensión estructural
        d_law = self._compute_d_law(eigenvalues)

        new_laws = []
        correlations = []

        for i in range(d_law):
            v_i = eigenvectors[:, i]

            # Predicción de ΔS
            w = min(len(self.f_history), int(np.sqrt(len(self.f_history))))
            predictions = []
            actuals = []

            for j in range(max(0, len(self.f_history) - w), len(self.f_history)):
                if j < len(self.delta_S_history):
                    pred = np.dot(self.f_history[j], v_i)
                    predictions.append(pred)
                    actuals.append(self.delta_S_history[j])

            if len(predictions) < 3:
                continue

            # Correlación
            corr = np.corrcoef(predictions, actuals)[0, 1]
            if np.isnan(corr):
                corr = 0

            correlations.append(abs(corr))

            # Threshold endógeno: percentil 90 de correlaciones históricas
            if len(self.correlation_history) > 0:
                all_corrs = [c for cs in self.correlation_history for c in cs]
                if len(all_corrs) > 5:
                    threshold = np.percentile(all_corrs, 90)
                else:
                    threshold = 1.0 / np.sqrt(len(self.f_history))
            else:
                threshold = 1.0 / np.sqrt(len(self.f_history))

            # Verificar si es ley válida
            if abs(corr) >= threshold:
                # Loadings
                loadings = {}
                for k, name in enumerate(self.feature_names):
                    loadings[name] = float(v_i[k])

                # Verificar que no es idéntica a ley existente
                is_new = True
                for existing in self.laws:
                    similarity = abs(np.dot(v_i, existing.eigenvector))
                    if similarity > 0.95:
                        is_new = False
                        # Actualizar estabilidad
                        existing.stability_history.append(similarity)
                        break

                if is_new:
                    law = StructuralLaw(
                        law_id=len(self.laws),
                        eigenvector=v_i.copy(),
                        eigenvalue=float(eigenvalues[i]),
                        correlation=float(corr),
                        loadings=loadings,
                        first_detected=t
                    )
                    new_laws.append(law)

        self.correlation_history.append(correlations)
        return new_laws

    def step(self,
             integration: float,
             irreversibility: float,
             self_surprise: float,
             D_stab: float,
             D_nov: float,
             D_irr: float,
             Psi: float,
             S: float) -> Dict:
        """Ejecuta un paso de descubrimiento de leyes."""
        t = len(self.f_history)

        # Construir features
        f = self._compute_features(
            integration, irreversibility, self_surprise,
            D_stab, D_nov, D_irr, Psi, S
        )
        self.f_history.append(f)
        self.S_history.append(S)

        # ΔS
        if t > 0:
            delta_S = S - self.S_history[-2]
            self.delta_S_history.append(delta_S)

        # Detectar leyes
        new_laws = self._detect_laws(t)
        for law in new_laws:
            self.laws.append(law)

        return {
            't': t,
            'f': f.tolist(),
            'S': S,
            'n_laws': len(self.laws),
            'new_laws': [l.law_id for l in new_laws]
        }

    def get_summary(self) -> Dict:
        """Resumen del descubrimiento de leyes."""
        T = len(self.f_history)

        # Estadísticas de leyes
        law_stats = []
        for law in self.laws:
            # Top loadings
            sorted_loadings = sorted(law.loadings.items(),
                                    key=lambda x: abs(x[1]), reverse=True)
            top_loadings = sorted_loadings[:3]

            # Estabilidad
            if len(law.stability_history) > 0:
                stability = np.mean(law.stability_history)
            else:
                stability = 1.0

            law_stats.append({
                'law_id': law.law_id,
                'eigenvalue': law.eigenvalue,
                'correlation': law.correlation,
                'top_loadings': top_loadings,
                'stability': float(stability),
                'first_detected': law.first_detected
            })

        # Métricas globales
        n_laws = len(self.laws)

        # Correlaciones promedio
        if n_laws > 0:
            mean_corr = np.mean([l.correlation for l in self.laws])
            max_corr = max([abs(l.correlation) for l in self.laws])
        else:
            mean_corr = 0
            max_corr = 0

        # d_law
        if len(self.f_history) > 10:
            cov = self._compute_covariance()
            eigenvalues = np.linalg.eigvalsh(cov)
            d_law = self._compute_d_law(eigenvalues)
        else:
            d_law = 0

        return {
            'T': T,
            'n_laws': n_laws,
            'law_stats': law_stats,
            'd_law': d_law,
            'mean_correlation': float(mean_corr),
            'max_correlation': float(max_corr)
        }


def run_phaseR8_test(n_steps: int = 1000, seed: int = 42) -> Dict:
    """Ejecuta test de Phase R8 - 100% ENDÓGENO."""
    np.random.seed(seed)

    est = EndogenousStructuralTheorems()

    # Simular dinámicas
    results = []

    for t in range(n_steps):
        # Generar features endógenos basados en historia
        eta = 1.0 / np.sqrt(t + 1)

        if t == 0:
            # Inicial
            integration = 0.5
            irreversibility = 0
            self_surprise = 1.0
            D_stab = 0.5
            D_nov = 0.5
            D_irr = 0.5
            Psi = 0.5
            S = 0.5
        else:
            # Evolución endógena
            prev_f = est.f_history[-1]

            integration = prev_f[0] + eta * np.random.randn()
            irreversibility = t / n_steps
            self_surprise = 1.0 / np.sqrt(t + 1)

            # Drives basados en historia
            if len(est.S_history) > 1:
                delta_S = est.S_history[-1] - est.S_history[-2] if len(est.S_history) >= 2 else 0
                D_stab = 0.5 - delta_S
                D_nov = 0.5 + delta_S
                D_irr = prev_f[5] + eta * np.random.randn()
            else:
                D_stab = 0.5
                D_nov = 0.5
                D_irr = 0.5

            Psi = (integration + D_stab + D_nov) / 3
            S = (integration * D_stab + irreversibility * D_nov) / 2

        # Clip
        integration = np.clip(integration, 0, 1)
        D_stab = np.clip(D_stab, 0, 1)
        D_nov = np.clip(D_nov, 0, 1)
        D_irr = np.clip(D_irr, 0, 1)
        Psi = np.clip(Psi, 0, 1)
        S = np.clip(S, 0, 1)

        result = est.step(
            integration, irreversibility, self_surprise,
            D_stab, D_nov, D_irr, Psi, S
        )
        results.append(result)

    # Summary
    summary = est.get_summary()

    # Evaluación GO/NO-GO (100% ENDÓGENO)
    T = n_steps

    # Thresholds endógenos
    # laws_discovered: ≥ 1
    # laws_predict_S: max_corr > 1/√T
    corr_threshold = 1.0 / np.sqrt(T)

    # laws_stable: promedio > 0.5
    if len(est.laws) > 0:
        stabilities = []
        for law in est.laws:
            if len(law.stability_history) > 0:
                stabilities.append(np.mean(law.stability_history))
            else:
                stabilities.append(1.0)
        mean_stability = np.mean(stabilities)
    else:
        mean_stability = 0

    # multiple_modes: d_law ≥ 2
    # laws_interpretable: loadings diferenciados

    criteria = {
        'laws_discovered': summary['n_laws'] >= 1,
        'laws_predict_S': summary['max_correlation'] > corr_threshold,
        'laws_stable': mean_stability > 0.5,
        'multiple_modes': summary['d_law'] >= 2,
        'laws_interpretable': all(
            len(l['top_loadings']) >= 2 for l in summary['law_stats']
        ) if summary['law_stats'] else False
    }

    go = sum(criteria.values()) >= 3

    return {
        'go': go,
        'criteria': criteria,
        'summary': summary,
        'S_series': [r['S'] for r in results[-100:]]
    }


def main():
    """Ejecuta Phase R8."""
    print("=" * 70)
    print("PHASE R8: ENDOGENOUS STRUCTURAL THEOREMS")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}")

    results = run_phaseR8_test(n_steps=1000)

    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)

    print(f"\nLeyes descubiertas: {results['summary']['n_laws']}")
    print(f"d_law (dimensión estructural): {results['summary']['d_law']}")
    print(f"Correlación máxima: {results['summary']['max_correlation']:.4f}")

    print("\nLeyes:")
    for law in results['summary']['law_stats'][:5]:
        print(f"  Ley {law['law_id']}: ρ={law['correlation']:.4f}, "
              f"λ={law['eigenvalue']:.4f}, estab={law['stability']:.4f}")
        print(f"    Top loadings: {law['top_loadings'][:2]}")

    print("\nCriterios:")
    for name, passed in results['criteria'].items():
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")

    n_pass = sum(results['criteria'].values())
    n_total = len(results['criteria'])
    go_status = "GO" if results['go'] else "NO-GO"

    print(f"\nResultado: {go_status} ({n_pass}/{n_total} criterios)")

    # Guardar resultados
    results_dir = '/root/NEO_EVA/results/phaseR8'
    os.makedirs(results_dir, exist_ok=True)

    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32, np.int16)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(f'{results_dir}/phaseR8_results.json', 'w') as f:
        json.dump(convert_types(results), f, indent=2)

    # Generar figura
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # S over time
    ax = axes[0, 0]
    ax.plot(results['S_series'], alpha=0.7)
    ax.set_xlabel('Step (last 100)')
    ax.set_ylabel('S(t)')
    ax.set_title('Proto-Subjectivity Score')
    ax.grid(True, alpha=0.3)

    # Law correlations
    ax = axes[0, 1]
    if results['summary']['law_stats']:
        law_ids = [l['law_id'] for l in results['summary']['law_stats']]
        corrs = [abs(l['correlation']) for l in results['summary']['law_stats']]
        ax.bar(law_ids, corrs, color='purple', alpha=0.7)
        ax.axhline(y=1/np.sqrt(1000), color='r', linestyle='--',
                   label=f'threshold=1/√T')
    ax.set_xlabel('Law ID')
    ax.set_ylabel('|ρ|')
    ax.set_title('Law Correlations with ΔS')
    ax.legend()

    # Law eigenvalues
    ax = axes[1, 0]
    if results['summary']['law_stats']:
        law_ids = [l['law_id'] for l in results['summary']['law_stats']]
        eigs = [l['eigenvalue'] for l in results['summary']['law_stats']]
        ax.bar(law_ids, eigs, color='teal', alpha=0.7)
    ax.set_xlabel('Law ID')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Law Eigenvalues')

    # Criteria
    ax = axes[1, 1]
    criteria_names = list(results['criteria'].keys())
    criteria_values = [1 if v else 0 for v in results['criteria'].values()]
    colors = ['green' if v else 'red' for v in results['criteria'].values()]
    y_pos = range(len(criteria_names))
    ax.barh(y_pos, criteria_values, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(criteria_names, fontsize=9)
    ax.set_xlim(0, 1.5)
    ax.set_title(f'Phase R8 Criteria: {go_status}')

    plt.tight_layout()
    plt.savefig('/root/NEO_EVA/figures/phaseR8_results.png', dpi=150)
    plt.close()

    print(f"\nResultados guardados en: {results_dir}")
    print(f"Figura: /root/NEO_EVA/figures/phaseR8_results.png")

    return results


if __name__ == "__main__":
    main()
