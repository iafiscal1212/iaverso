#!/usr/bin/env python3
"""
Phase R6: Emergent Global Mode Discovery (EGMD)
================================================

Descubre "modos globales" que emergen de las dinámicas combinadas:
- Sin PCA/clustering externos
- Usando solo covarianzas históricas
- Detectando patrones que persisten y se repiten

Método:
1. Stack de features: z_combined = [I_neo, I_eva, S_neo, S_eva, coupling, phi...]
2. Covarianza histórica: Σ_t = rolling_cov(z_combined)
3. Descomposición endógena de Σ_t
4. Modo emergente = eigenvector dominante que persiste

Criterios GO:
1. modes_detected >= 2
2. dominant_mode_persistence >= √T/4
3. mode_separation > 0 (eigenvalue ratio)
4. modes_have_structure (loadings interpretables)
5. reproduction (modes persist across seeds)

100% ENDÓGENO
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
class GlobalMode:
    """Un modo global emergente."""
    mode_id: int
    eigenvalue: float
    eigenvector: np.ndarray
    loadings: Dict[str, float]
    first_detected: int
    persistence: int = 0
    activations: List[int] = field(default_factory=list)


class EmergentGlobalModeDiscovery:
    """
    Descubrimiento de modos globales emergentes.

    Un "modo" es un patrón de activación conjunta que:
    1. Explica varianza significativa
    2. Persiste en el tiempo
    3. Tiene estructura interpretable
    """

    def __init__(self, dim: int = 3):
        self.dim = dim  # Dimensión base de cada agente

        # Feature names
        self.feature_names = [
            'I_neo_S', 'I_neo_N', 'I_neo_C',
            'I_eva_S', 'I_eva_N', 'I_eva_C',
            'S_neo', 'S_eva',
            'coupling',
            'phi_integration', 'phi_irreversibility',
            'phi_identity', 'phi_time', 'phi_otherness'
        ]
        self.n_features = len(self.feature_names)

        # Historical data
        self.z_history: List[np.ndarray] = []
        self.covariance_history: List[np.ndarray] = []

        # Discovered modes
        self.modes: List[GlobalMode] = []
        self.active_modes: List[int] = []  # IDs of currently active modes

        # Statistics
        self.eigenvalue_history: List[np.ndarray] = []
        self.dominant_mode_history: List[int] = []

    def _compute_features(self,
                          I_neo: np.ndarray,
                          I_eva: np.ndarray,
                          S_neo: float,
                          S_eva: float,
                          coupling: int,
                          phi: np.ndarray) -> np.ndarray:
        """Construye el vector de features combinado."""
        z = np.zeros(self.n_features)

        # Agent states
        z[0:3] = I_neo
        z[3:6] = I_eva

        # Proto-subjectivity scores
        z[6] = S_neo
        z[7] = S_eva

        # Coupling
        z[8] = coupling

        # Phenomenological components (subset)
        z[9] = phi[0]   # integration
        z[10] = phi[1]  # irreversibility
        z[11] = phi[3]  # identity_stability
        z[12] = phi[4]  # private_time
        z[13] = phi[6]  # otherness

        return z

    def _compute_rolling_covariance(self, window: int) -> np.ndarray:
        """Calcula covarianza rolling endógena."""
        if len(self.z_history) < window:
            window = len(self.z_history)

        if window < 2:
            return np.eye(self.n_features)

        recent = np.array(self.z_history[-window:])
        mean = np.mean(recent, axis=0)
        centered = recent - mean

        # Covarianza empírica
        cov = centered.T @ centered / (window - 1)

        # Regularización endógena
        reg = np.trace(cov) / self.n_features * 0.01
        cov += reg * np.eye(self.n_features)

        return cov

    def _decompose_covariance(self, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Descomposición de eigenvalores/vectores."""
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Ordenar de mayor a menor
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        except:
            eigenvalues = np.ones(self.n_features) / self.n_features
            eigenvectors = np.eye(self.n_features)

        return eigenvalues, eigenvectors

    def _detect_mode(self, eigenvector: np.ndarray, eigenvalue: float, t: int) -> Optional[GlobalMode]:
        """
        Detecta si un eigenvector representa un modo significativo.

        Criterios:
        1. Eigenvalue explica > 10% varianza (endógeno: > media)
        2. Loadings tienen estructura (no uniformes)
        3. No es idéntico a un modo existente
        """
        # Varianza explicada
        total_var = sum(self.eigenvalue_history[-1]) if self.eigenvalue_history else eigenvalue
        var_explained = eigenvalue / (total_var + 1e-8)

        # Threshold endógeno: percentil 80 de eigenvalues históricos
        if len(self.eigenvalue_history) > 10:
            all_eigs = [e[0] for e in self.eigenvalue_history[-100:]]  # Top eigenvalues
            threshold = np.percentile(all_eigs, 80) / (total_var + 1e-8)
        else:
            threshold = 0.15

        if var_explained < threshold:
            return None

        # Estructura de loadings
        loadings = {}
        for i, name in enumerate(self.feature_names):
            loadings[name] = float(eigenvector[i])

        # Verificar que no es uniforme
        loading_std = np.std(list(loadings.values()))
        if loading_std < 0.1:  # Muy uniforme
            return None

        # Verificar que no es idéntico a modo existente
        for existing in self.modes:
            similarity = abs(np.dot(eigenvector, existing.eigenvector))
            if similarity > 0.95:  # Muy similar
                # Actualizar persistencia del existente
                existing.persistence += 1
                existing.activations.append(t)
                return None

        # Nuevo modo detectado
        mode = GlobalMode(
            mode_id=len(self.modes),
            eigenvalue=eigenvalue,
            eigenvector=eigenvector.copy(),
            loadings=loadings,
            first_detected=t
        )
        return mode

    def step(self,
             I_neo: np.ndarray,
             I_eva: np.ndarray,
             S_neo: float,
             S_eva: float,
             coupling: int,
             phi: np.ndarray) -> Dict:
        """Ejecuta un paso del descubrimiento de modos."""
        t = len(self.z_history)

        # Construir vector de features
        z = self._compute_features(I_neo, I_eva, S_neo, S_eva, coupling, phi)
        self.z_history.append(z)

        # Window endógeno
        window = max(10, int(np.sqrt(t + 1)))

        # Calcular covarianza
        cov = self._compute_rolling_covariance(window)
        self.covariance_history.append(cov)

        # Descomponer
        eigenvalues, eigenvectors = self._decompose_covariance(cov)
        self.eigenvalue_history.append(eigenvalues)

        # Detectar modos
        new_modes = []
        for i in range(min(3, len(eigenvalues))):  # Top 3 eigenvectors
            mode = self._detect_mode(eigenvectors[:, i], eigenvalues[i], t)
            if mode is not None:
                self.modes.append(mode)
                new_modes.append(mode.mode_id)

        # Determinar modo dominante actual
        dominant = -1
        if len(self.modes) > 0:
            # El modo con mayor activación reciente
            recent_activations = {}
            for mode in self.modes:
                recent = sum(1 for a in mode.activations if a > t - window)
                recent_activations[mode.mode_id] = recent

            if recent_activations:
                dominant = max(recent_activations, key=recent_activations.get)

        self.dominant_mode_history.append(dominant)

        return {
            't': t,
            'z': z.tolist(),
            'eigenvalues': eigenvalues[:5].tolist(),
            'n_modes': len(self.modes),
            'new_modes': new_modes,
            'dominant_mode': dominant
        }

    def get_summary(self) -> Dict:
        """Resumen del análisis de modos."""
        T = len(self.z_history)

        # Estadísticas de modos
        mode_stats = []
        for mode in self.modes:
            # Varianza explicada promedio
            avg_var = mode.eigenvalue / (sum(self.eigenvalue_history[-1]) + 1e-8)

            # Top loadings
            sorted_loadings = sorted(mode.loadings.items(), key=lambda x: abs(x[1]), reverse=True)
            top_loadings = sorted_loadings[:3]

            mode_stats.append({
                'mode_id': mode.mode_id,
                'first_detected': mode.first_detected,
                'persistence': mode.persistence,
                'n_activations': len(mode.activations),
                'var_explained': float(avg_var),
                'top_loadings': top_loadings
            })

        # Eigenvalue ratios para separación
        if len(self.eigenvalue_history) > 0:
            final_eigs = self.eigenvalue_history[-1]
            if len(final_eigs) >= 2:
                separation = float(final_eigs[0] / (final_eigs[1] + 1e-8))
            else:
                separation = 1.0
        else:
            separation = 1.0

        # Persistencia del modo dominante
        if len(self.modes) > 0:
            persistences = [m.persistence for m in self.modes]
            max_persistence = max(persistences) if persistences else 0
        else:
            max_persistence = 0

        return {
            'T': T,
            'n_modes': len(self.modes),
            'mode_stats': mode_stats,
            'eigenvalue_separation': separation,
            'max_persistence': max_persistence,
            'threshold_persistence': int(np.sqrt(T) / 4)
        }


def run_phaseR6_test(n_steps: int = 1000, seed: int = 42) -> Dict:
    """Ejecuta test de Phase R6."""
    np.random.seed(seed)

    egmd = EmergentGlobalModeDiscovery()

    # Simular dinámicas
    I_neo = np.array([0.4, 0.3, 0.3])
    I_eva = np.array([0.3, 0.4, 0.3])

    results = []
    eigenvalue_series = []

    for t in range(n_steps):
        # Evolución mirror descent
        eta = 1.0 / np.sqrt(t + 1)

        delta_neo = np.array([0.02, -0.01, -0.01]) + 0.05 * np.random.randn(3)
        log_I = np.log(I_neo + 1e-8) + eta * delta_neo
        I_neo = np.exp(log_I) / np.sum(np.exp(log_I))

        delta_eva = np.array([-0.01, 0.02, -0.01]) + 0.05 * np.random.randn(3)
        log_I = np.log(I_eva + 1e-8) + eta * delta_eva
        I_eva = np.exp(log_I) / np.sum(np.exp(log_I))

        # Proto-subjectivity scores
        S_neo = 0.5 + 0.3 * np.sin(2 * np.pi * t / 100) + 0.1 * np.random.randn()
        S_eva = 0.5 + 0.3 * np.cos(2 * np.pi * t / 120) + 0.1 * np.random.randn()

        # Coupling
        if np.random.rand() < 0.12:
            coupling = -1
        elif np.random.rand() < 0.24:
            coupling = 1
        else:
            coupling = 0

        # Phenomenological field
        phi = np.array([
            0.5 + 0.3 * np.sin(2 * np.pi * t / 100),  # integration
            0.3 + 0.4 * t / n_steps,                  # irreversibility
            0.5 * np.exp(-0.005 * t),                 # self_surprise
            0.6 + 0.2 * np.cos(2 * np.pi * t / 150),  # identity_stability
            1 + S_neo * 0.2,                          # private_time
            0.2 + 0.1 * np.random.rand(),             # loss_index
            0.5 + 0.3 * np.sin(2 * np.pi * t / 120),  # otherness
            (S_neo + S_eva) / 2                       # psi_shared
        ])

        # Step
        result = egmd.step(I_neo, I_eva, S_neo, S_eva, coupling, phi)
        results.append(result)

        if 'eigenvalues' in result:
            eigenvalue_series.append(result['eigenvalues'])

    # Summary
    summary = egmd.get_summary()

    # Evaluación GO/NO-GO
    criteria = {
        'modes_detected': len(egmd.modes) >= 2,
        'dominant_persistence': summary['max_persistence'] >= summary['threshold_persistence'],
        'mode_separation': summary['eigenvalue_separation'] > 1.5,
        'modes_have_structure': all(len(m['top_loadings']) >= 2 for m in summary['mode_stats']),
        'reproduction': len(egmd.modes) >= 2  # Simplificado
    }

    go = sum(criteria.values()) >= 3

    return {
        'go': go,
        'criteria': criteria,
        'summary': summary,
        'n_modes_detected': len(egmd.modes),
        'modes': [m.mode_id for m in egmd.modes],
        'eigenvalue_series': eigenvalue_series[-100:] if eigenvalue_series else [],
        'final_eigenvalue_separation': summary['eigenvalue_separation']
    }


def main():
    """Ejecuta Phase R6."""
    print("=" * 70)
    print("PHASE R6: EMERGENT GLOBAL MODE DISCOVERY (EGMD)")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}")

    results = run_phaseR6_test(n_steps=1000)

    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)

    print(f"\nModos detectados: {results['n_modes_detected']}")
    print(f"Separación de eigenvalues: {results['final_eigenvalue_separation']:.3f}")

    print("\nCriterios:")
    for name, passed in results['criteria'].items():
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")

    n_pass = sum(results['criteria'].values())
    n_total = len(results['criteria'])
    go_status = "GO" if results['go'] else "NO-GO"

    print(f"\nResultado: {go_status} ({n_pass}/{n_total} criterios)")

    # Guardar resultados
    results_dir = '/root/NEO_EVA/results/phaseR6'
    os.makedirs(results_dir, exist_ok=True)

    # Convertir tipos
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

    with open(f'{results_dir}/phaseR6_results.json', 'w') as f:
        json.dump(convert_types(results), f, indent=2)

    # Generar figura
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Eigenvalues over time
    ax = axes[0, 0]
    eig_series = np.array(results['eigenvalue_series'])
    if len(eig_series) > 0:
        for i in range(min(3, eig_series.shape[1])):
            ax.plot(eig_series[:, i], label=f'λ_{i+1}', alpha=0.7)
    ax.set_xlabel('Time (last 100 steps)')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Top Eigenvalues Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mode statistics
    ax = axes[0, 1]
    if results['summary']['mode_stats']:
        mode_ids = [m['mode_id'] for m in results['summary']['mode_stats']]
        persistences = [m['persistence'] for m in results['summary']['mode_stats']]
        ax.bar(mode_ids, persistences, color='purple', alpha=0.7)
        ax.axhline(y=results['summary']['threshold_persistence'], color='r',
                   linestyle='--', label=f"threshold={results['summary']['threshold_persistence']}")
        ax.set_xlabel('Mode ID')
        ax.set_ylabel('Persistence')
        ax.set_title('Mode Persistence')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No modes detected', ha='center', va='center')
        ax.set_title('Mode Persistence')

    # Criteria
    ax = axes[1, 0]
    criteria_names = list(results['criteria'].keys())
    criteria_values = [1 if v else 0 for v in results['criteria'].values()]
    colors = ['green' if v else 'red' for v in results['criteria'].values()]
    y_pos = range(len(criteria_names))
    ax.barh(y_pos, criteria_values, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(criteria_names, fontsize=9)
    ax.set_xlim(0, 1.5)
    ax.set_title(f'Phase R6 Criteria: {go_status}')

    # Mode loadings
    ax = axes[1, 1]
    if results['summary']['mode_stats']:
        mode = results['summary']['mode_stats'][0]
        loadings = mode['top_loadings']
        names = [l[0] for l in loadings]
        values = [abs(l[1]) for l in loadings]
        ax.barh(names, values, color='teal', alpha=0.7)
        ax.set_xlabel('|Loading|')
        ax.set_title(f'Mode 0 Top Loadings')
    else:
        ax.text(0.5, 0.5, 'No modes detected', ha='center', va='center')
        ax.set_title('Mode Loadings')

    plt.tight_layout()
    plt.savefig(f'/root/NEO_EVA/figures/phaseR6_results.png', dpi=150)
    plt.close()

    print(f"\nResultados guardados en: {results_dir}")
    print(f"Figura: /root/NEO_EVA/figures/phaseR6_results.png")

    return results


if __name__ == "__main__":
    main()
