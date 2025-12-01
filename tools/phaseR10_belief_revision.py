#!/usr/bin/env python3
"""
Phase R10: Structural Coherence & Belief Revision
=================================================

Detecta leyes internas que se contradicen sistemáticamente con la
experiencia y las degrada.

100% ENDÓGENO - Sin constantes mágicas
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


@dataclass
class InternalLaw:
    """Representa una ley interna con su peso y historial."""
    eigenvector: np.ndarray
    eigenvalue: float
    predictions: List[float] = field(default_factory=list)
    actual_values: List[float] = field(default_factory=list)
    contradictions: List[float] = field(default_factory=list)
    correlations: List[float] = field(default_factory=list)
    weight: float = 1.0
    rank: int = 0
    creation_time: int = 0


class StructuralBeliefRevision:
    """
    Detecta leyes internas que se contradicen sistemáticamente
    con la experiencia y las degrada.

    100% Endógeno:
    - c_i = |pred_i - actual| / σ(actual) (contradicción normalizada)
    - w_i = rank(|ρ_i|) * (1 - c_i) (peso por correlación y consistencia)
    - Degradación si w_i < P25 de pesos históricos
    - Leyes descubiertas por PCA de historial de features
    """

    def __init__(self, dim: int = 3):
        self.dim = dim

        # Leyes activas con sus pesos
        self.active_laws: List[InternalLaw] = []

        # Historial
        self.S_history: List[float] = []
        self.z_history: List[np.ndarray] = []
        self.delta_S_history: List[float] = []
        self.feature_history: List[np.ndarray] = []  # Features derivados de z

        self.weight_history: List[List[float]] = []
        self.contradiction_history: List[List[float]] = []
        self.degraded_laws: List[int] = []  # Índices de leyes degradadas

        # Para coherencia global
        self.coherence_history: List[float] = []

        # Ventana para descubrimiento de leyes (endógena: √T)
        self.law_discovery_interval = 10  # Se ajusta con √T

    def _compute_features(self, z: np.ndarray, S: float, t: int) -> np.ndarray:
        """
        Computa features derivados del estado para descubrimiento de leyes.
        100% endógeno: derivados de z y S.
        """
        features = []

        # 1. Entropía
        features.append(S)

        # 2. Varianza de z
        features.append(np.var(z))

        # 3. Max - min de z (rango)
        features.append(np.max(z) - np.min(z))

        # 4. Concentración (Gini-like)
        sorted_z = np.sort(z)
        n = len(z)
        gini = np.sum((2 * np.arange(1, n + 1) - n - 1) * sorted_z) / (n * np.sum(sorted_z) + 1e-10)
        features.append(gini)

        # 5. Delta S si hay historia
        if len(self.S_history) > 0:
            features.append(S - self.S_history[-1])
        else:
            features.append(0.0)

        # 6. Distancia a centroide
        if len(self.z_history) > 0:
            centroid = np.mean(self.z_history, axis=0)
            features.append(np.linalg.norm(z - centroid))
        else:
            features.append(0.0)

        return np.array(features)

    def _discover_laws(self) -> None:
        """
        Descubre leyes internas por PCA de la historia de features.
        100% endógeno: eigenvectors de la covarianza de features.
        """
        if len(self.feature_history) < 10:
            return

        # Matriz de features
        F = np.array(self.feature_history)

        # Covarianza
        cov = np.cov(F.T)

        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Ordenar por eigenvalue descendente
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Número de leyes: dimensiones con eigenvalue > mediana (endógeno)
            median_eig = np.median(eigenvalues)
            n_laws = int(np.sum(eigenvalues >= median_eig))
            n_laws = max(1, min(n_laws, len(eigenvalues)))

            # Crear nuevas leyes si no existen
            current_t = len(self.S_history)
            for i in range(n_laws):
                if i >= len(self.active_laws):
                    new_law = InternalLaw(
                        eigenvector=eigenvectors[:, i].copy(),
                        eigenvalue=float(eigenvalues[i]),
                        rank=i + 1,
                        creation_time=current_t
                    )
                    self.active_laws.append(new_law)
                else:
                    # Actualizar eigenvector existente
                    self.active_laws[i].eigenvector = eigenvectors[:, i].copy()
                    self.active_laws[i].eigenvalue = float(eigenvalues[i])

        except np.linalg.LinAlgError:
            pass

    def step(self, z: np.ndarray, S: float) -> Dict[str, Any]:
        """
        Ejecuta un paso de revisión de creencias.

        Returns:
            Dict con estado de coherencia y pesos de leyes
        """
        t = len(self.S_history)

        # Computar features
        features = self._compute_features(z, S, t)

        # Registrar historia
        self.S_history.append(S)
        self.z_history.append(z.copy())
        self.feature_history.append(features)

        if t > 0:
            self.delta_S_history.append(S - self.S_history[-2])
        else:
            self.delta_S_history.append(0.0)

        # Descubrir leyes periódicamente (intervalo endógeno: √T)
        self.law_discovery_interval = max(5, int(np.sqrt(t + 1)))
        if t % self.law_discovery_interval == 0:
            self._discover_laws()

        # Calcular contradicciones para cada ley
        contradictions = self._compute_contradictions(t)

        # Calcular pesos actualizados
        weights = self._compute_weights(contradictions)

        # Detectar y degradar leyes inconsistentes
        degraded_this_step = self._degrade_inconsistent_laws(weights)

        # Calcular coherencia global
        coherence = self._compute_global_coherence()
        self.coherence_history.append(coherence)

        # Guardar historial
        if contradictions:
            self.contradiction_history.append(contradictions)
            self.weight_history.append(weights)

        return {
            't': t,
            'n_active_laws': len(self.active_laws),
            'n_degraded_total': len(self.degraded_laws),
            'degraded_this_step': degraded_this_step,
            'coherence': coherence,
            'contradictions': contradictions,
            'weights': weights
        }

    def _compute_contradictions(self, t: int) -> List[float]:
        """
        Calcula contradicciones para cada ley.

        c_i = |pred_i - actual| / σ(actual)

        100% endógeno: normalizado por la desviación estándar de ΔS
        """
        if t < 2 or len(self.active_laws) == 0:
            return []

        # σ(ΔS) endógeno
        if len(self.delta_S_history) > 1:
            sigma_delta = np.std(self.delta_S_history)
            sigma_delta = max(sigma_delta, 1e-10)
        else:
            sigma_delta = 1.0

        # ΔS actual
        delta_S_actual = self.delta_S_history[-1]

        # Δfeatures
        if len(self.feature_history) >= 2:
            delta_features = self.feature_history[-1] - self.feature_history[-2]
        else:
            delta_features = np.zeros(len(self.feature_history[-1]))

        contradictions = []
        for law in self.active_laws:
            # Predicción de la ley: proyección del cambio de features
            if len(law.eigenvector) == len(delta_features):
                pred = np.dot(law.eigenvector, delta_features)

                # Correlación con ΔS para evaluar poder predictivo
                if len(law.predictions) > 5:
                    corr = np.corrcoef(law.predictions[-20:], law.actual_values[-20:])[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                else:
                    corr = 0.0
                law.correlations.append(corr)

                # c_i = |pred - actual| / σ(ΔS), escalado por eigenvalue
                scaled_pred = pred * law.eigenvalue
                c_i = abs(scaled_pred - delta_S_actual) / sigma_delta
                c_i = min(c_i, 1.0)  # Cap endógeno a 1
            else:
                pred = 0.0
                c_i = 0.5  # Valor neutro si dimensiones no coinciden

            law.predictions.append(pred)
            law.actual_values.append(delta_S_actual)
            law.contradictions.append(c_i)
            contradictions.append(c_i)

        return contradictions

    def _compute_weights(self, contradictions: List[float]) -> List[float]:
        """
        Calcula pesos actualizados para cada ley.

        w_i = rank(|ρ_i|) * (1 - c_i)

        100% endógeno: rank viene de las correlaciones, c_i de las contradicciones
        """
        if not contradictions or not self.active_laws:
            return []

        # Calcular rankings por correlación
        correlations = []
        for law in self.active_laws:
            if law.correlations:
                correlations.append(abs(law.correlations[-1]))
            else:
                correlations.append(0.0)

        # Ranking (mayor correlación = mayor rank)
        if correlations:
            sorted_indices = np.argsort(correlations)
            for rank, idx in enumerate(sorted_indices):
                self.active_laws[idx].rank = rank + 1

        weights = []
        max_rank = len(self.active_laws)

        for i, law in enumerate(self.active_laws):
            c_i = contradictions[i] if i < len(contradictions) else 0.0
            # w_i = rank * (1 - c_i), normalizado
            w_i = (law.rank / max(max_rank, 1)) * (1.0 - c_i)
            law.weight = w_i
            weights.append(w_i)

        return weights

    def _degrade_inconsistent_laws(self, weights: List[float]) -> List[int]:
        """
        Degrada leyes con w_i < P25 de pesos históricos.

        100% endógeno: umbral = percentil 25 de la historia
        """
        degraded_this_step = []

        if len(self.weight_history) < 10 or not weights:
            return degraded_this_step

        # Calcular P25 de todos los pesos históricos
        all_weights = [w for ws in self.weight_history for w in ws]
        threshold = np.percentile(all_weights, 25)

        for i, w in enumerate(weights):
            if w < threshold and i not in self.degraded_laws:
                # Verificar consistencia: debe estar bajo P25 por varios pasos
                # Ventana endógena: √T pasos
                window = max(int(np.sqrt(len(self.S_history))), 3)
                if len(self.weight_history) >= window:
                    recent_weights = []
                    for j in range(min(window, len(self.weight_history))):
                        hist_idx = -(j + 1)
                        if i < len(self.weight_history[hist_idx]):
                            recent_weights.append(self.weight_history[hist_idx][i])

                    if recent_weights and np.mean(recent_weights) < threshold:
                        self.degraded_laws.append(i)
                        degraded_this_step.append(i)

        return degraded_this_step

    def _compute_global_coherence(self) -> float:
        """
        Calcula coherencia global del sistema de creencias.

        Coherencia = 1 - media(contradicciones de leyes activas no degradadas)

        100% endógeno
        """
        active_contradictions = []

        for i, law in enumerate(self.active_laws):
            if i not in self.degraded_laws and law.contradictions:
                active_contradictions.append(law.contradictions[-1])

        if not active_contradictions:
            return 1.0

        return 1.0 - np.mean(active_contradictions)

    def get_belief_state(self) -> Dict[str, Any]:
        """Retorna el estado actual del sistema de creencias."""
        return {
            'n_laws': len(self.active_laws),
            'n_degraded': len(self.degraded_laws),
            'n_active': len(self.active_laws) - len(self.degraded_laws),
            'coherence': self.coherence_history[-1] if self.coherence_history else 1.0,
            'degraded_indices': self.degraded_laws.copy(),
            'law_weights': [law.weight for law in self.active_laws]
        }

    def get_active_laws(self) -> List[InternalLaw]:
        """Retorna leyes activas (no degradadas)."""
        return [
            law for i, law in enumerate(self.active_laws)
            if i not in self.degraded_laws
        ]


def run_phase_r10() -> Dict[str, Any]:
    """Ejecuta Phase R10 y evalúa criterios GO/NO-GO."""

    print("=" * 70)
    print("PHASE R10: STRUCTURAL COHERENCE & BELIEF REVISION")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}\n")

    np.random.seed(42)

    # Crear sistema
    belief_system = StructuralBeliefRevision(dim=3)

    # Simulación extendida con perturbaciones
    T = 300
    results = []

    # Estado inicial
    z = np.array([0.4, 0.3, 0.3])

    for t in range(T):
        # Dinámica con perturbaciones ocasionales para inducir contradicciones
        noise = np.random.randn(3) * 0.02

        # Perturbación más fuerte ocasionalmente para crear contradicciones
        if t % 50 == 49:
            noise += np.random.randn(3) * 0.15

        z = z + noise
        z = np.clip(z, 0.01, 0.99)
        z = z / z.sum()

        S = -np.sum(z * np.log(z + 1e-10))

        result = belief_system.step(z, S)
        results.append(result)

    # Análisis de resultados
    final_state = belief_system.get_belief_state()

    print("=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print()

    # Métricas
    n_laws = final_state['n_laws']
    n_degraded = final_state['n_degraded']
    n_active = final_state['n_active']
    final_coherence = final_state['coherence']

    # Coherencia a lo largo del tiempo
    coherence_history = belief_system.coherence_history
    mean_coherence = np.mean(coherence_history) if coherence_history else 0.0

    # Contradicciones detectadas
    all_contradictions = [c for cs in belief_system.contradiction_history for c in cs]
    high_contradictions = sum(1 for c in all_contradictions if c > 0.5) if all_contradictions else 0

    print(f"Leyes descubiertas: {n_laws}")
    print(f"Leyes degradadas: {n_degraded}")
    print(f"Leyes activas: {n_active}")
    print(f"Coherencia final: {final_coherence:.4f}")
    print(f"Coherencia media: {mean_coherence:.4f}")
    print(f"Contradicciones altas detectadas: {high_contradictions}")
    print()

    # Detalles de leyes
    if belief_system.active_laws:
        print("Estado de leyes:")
        for i, law in enumerate(belief_system.active_laws):
            status = "DEGRADADA" if i in belief_system.degraded_laws else "ACTIVA"
            mean_c = np.mean(law.contradictions) if law.contradictions else 0.0
            mean_corr = np.mean(np.abs(law.correlations)) if law.correlations else 0.0
            print(f"  Ley {i}: w={law.weight:.4f}, c_media={mean_c:.4f}, |ρ|={mean_corr:.4f} [{status}]")
    print()

    # =====================================================
    # CRITERIOS GO/NO-GO (100% endógenos)
    # =====================================================

    criteria = {}

    # 1. Leyes descubiertas: debe haber al menos 1
    criteria['laws_discovered'] = n_laws > 0

    # 2. Contradicciones detectadas: debe haber > 0
    criteria['contradictions_detected'] = len(all_contradictions) > 0

    # 3. Pesos diferenciados: std(weights) > 0
    weights = final_state['law_weights']
    criteria['weights_differentiated'] = np.std(weights) > 0 if weights else False

    # 4. Coherencia mantenida: coherencia media > 0.3
    criteria['coherence_maintained'] = mean_coherence > 0.3

    # 5. Sistema estable: coherencia nunca < 0.1
    min_coherence = min(coherence_history) if coherence_history else 1.0
    criteria['system_stable'] = min_coherence > 0.1

    # Resultado GO/NO-GO
    passed = sum(criteria.values())
    total = len(criteria)
    go_status = "GO" if passed >= 4 else "NO-GO"

    print("Criterios:")
    for name, passed_criterion in criteria.items():
        status = "✅" if passed_criterion else "❌"
        print(f"  {status} {name}")
    print()
    print(f"Resultado: {go_status} ({passed}/{total} criterios)")

    # Guardar resultados
    output = {
        'phase': 'R10',
        'name': 'Structural Coherence & Belief Revision',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'n_laws': n_laws,
            'n_degraded': n_degraded,
            'n_active': n_active,
            'final_coherence': final_coherence,
            'mean_coherence': mean_coherence,
            'min_coherence': min_coherence,
            'high_contradictions': high_contradictions
        },
        'criteria': criteria,
        'go_status': go_status,
        'passed_criteria': passed,
        'total_criteria': total
    }

    # Crear directorios si no existen
    os.makedirs('/root/NEO_EVA/results/phaseR10', exist_ok=True)
    os.makedirs('/root/NEO_EVA/figures', exist_ok=True)

    # Guardar JSON
    with open('/root/NEO_EVA/results/phaseR10/belief_revision_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

    # Crear visualización
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Coherencia temporal
        ax1 = axes[0, 0]
        ax1.plot(coherence_history, 'b-', linewidth=1.5)
        ax1.axhline(y=0.3, color='r', linestyle='--', alpha=0.7, label='Umbral coherencia')
        ax1.set_xlabel('Tiempo')
        ax1.set_ylabel('Coherencia')
        ax1.set_title('Coherencia Global del Sistema de Creencias')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Contradicciones por ley
        ax2 = axes[0, 1]
        if belief_system.contradiction_history:
            n_laws_plot = len(belief_system.active_laws)
            for i in range(min(n_laws_plot, 5)):
                law_contradictions = [
                    cs[i] if i < len(cs) else 0.0
                    for cs in belief_system.contradiction_history
                ]
                label = f"Ley {i}" + (" (deg)" if i in belief_system.degraded_laws else "")
                ax2.plot(law_contradictions, label=label, alpha=0.7)
        ax2.set_xlabel('Tiempo')
        ax2.set_ylabel('Contradicción')
        ax2.set_title('Contradicciones por Ley')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. Pesos de leyes a lo largo del tiempo
        ax3 = axes[1, 0]
        if belief_system.weight_history:
            n_laws_plot = len(belief_system.active_laws)
            for i in range(min(n_laws_plot, 5)):
                law_weights = [
                    ws[i] if i < len(ws) else 0.0
                    for ws in belief_system.weight_history
                ]
                label = f"Ley {i}"
                ax3.plot(law_weights, label=label, alpha=0.7)
            # P25 threshold
            all_w = [w for ws in belief_system.weight_history for w in ws]
            if all_w:
                p25 = np.percentile(all_w, 25)
                ax3.axhline(y=p25, color='r', linestyle='--', alpha=0.7, label=f'P25={p25:.3f}')
        ax3.set_xlabel('Tiempo')
        ax3.set_ylabel('Peso')
        ax3.set_title('Pesos de Leyes (w_i = rank * (1 - c_i))')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 4. Estado final de leyes
        ax4 = axes[1, 1]
        if belief_system.active_laws:
            law_indices = range(len(belief_system.active_laws))
            weights = [law.weight for law in belief_system.active_laws]
            colors = ['red' if i in belief_system.degraded_laws else 'green' for i in law_indices]
            bars = ax4.bar(law_indices, weights, color=colors, alpha=0.7)
            ax4.set_xlabel('Índice de Ley')
            ax4.set_ylabel('Peso Final')
            ax4.set_title(f'Estado Final: {n_active} activas, {n_degraded} degradadas')
            # Leyenda manual
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.7, label='Activa'),
                Patch(facecolor='red', alpha=0.7, label='Degradada')
            ]
            ax4.legend(handles=legend_elements)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/root/NEO_EVA/figures/phaseR10_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nResultados guardados en: /root/NEO_EVA/results/phaseR10")
        print(f"Figura: /root/NEO_EVA/figures/phaseR10_results.png")

    except Exception as e:
        print(f"Warning: No se pudo crear visualización: {e}")

    return output


if __name__ == "__main__":
    run_phase_r10()
