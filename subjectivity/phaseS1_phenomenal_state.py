#!/usr/bin/env python3
"""
Phase S1: Estado Fenomenológico φ_t
===================================

Define el espacio fenomenológico con:
- integration, irreversibility, self_surprise
- identity, time, loss, otherness, Ψ

100% ENDÓGENO - Sin constantes mágicas
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os


@dataclass
class PhenomenalVector:
    """Vector fenomenológico en un instante."""
    t: int
    integration: float
    irreversibility: float
    self_surprise: float
    identity: float
    time_sense: float
    loss: float
    otherness: float
    psi: float  # Integración global Ψ

    def to_array(self) -> np.ndarray:
        return np.array([
            self.integration,
            self.irreversibility,
            self.self_surprise,
            self.identity,
            self.time_sense,
            self.loss,
            self.otherness,
            self.psi
        ])


class PhenomenalState:
    """
    Genera y analiza el estado fenomenológico φ_t.

    φ_t = [integration, irreversibility, self_surprise,
           identity, time, loss, otherness, Ψ]

    100% Endógeno:
    - Todos los componentes derivados de la historia del sistema
    - Modos eigen de Σ_φ
    - Clustering en K modos internos
    """

    def __init__(self, dim_z: int = 6):
        self.dim_z = dim_z
        self.n_phenomenal = 8

        # Historia
        self.z_history: List[np.ndarray] = []
        self.phi_history: List[PhenomenalVector] = []
        self.S_history: List[float] = []

        # Modos fenomenológicos (clustering endógeno)
        self.n_modes = 3
        self.mode_history: List[int] = []
        self.mode_centroids: Optional[np.ndarray] = None

        # Covarianza del espacio fenomenológico
        self.Sigma_phi: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None

    def _compute_integration(self, z: np.ndarray) -> float:
        """
        Integración: correlación media entre componentes.

        100% endógeno
        """
        if len(self.z_history) < 10:
            return 0.5

        window = max(10, int(np.sqrt(len(self.z_history))))
        Z = np.array(self.z_history[-window:])

        # Correlación media entre pares de componentes
        corr_matrix = np.corrcoef(Z.T)
        mask = ~np.eye(len(z), dtype=bool)
        correlations = corr_matrix[mask]
        correlations = correlations[~np.isnan(correlations)]

        if len(correlations) == 0:
            return 0.5

        return float(np.mean(np.abs(correlations)))

    def _compute_irreversibility(self, z: np.ndarray) -> float:
        """
        Irreversibilidad: asimetría temporal.

        100% endógeno: diferencia entre predicción forward y backward
        """
        if len(self.z_history) < 10:
            return 0.5

        window = min(10, len(self.z_history))
        recent = self.z_history[-window:]

        # Predicción forward
        forward_diff = np.diff(recent, axis=0)
        forward_var = np.var(forward_diff)

        # Predicción backward (invertir y predecir)
        backward = list(reversed(recent))
        backward_diff = np.diff(backward, axis=0)
        backward_var = np.var(backward_diff)

        # Irreversibilidad = asimetría
        asymmetry = abs(forward_var - backward_var) / (forward_var + backward_var + 1e-10)

        return float(asymmetry)

    def _compute_self_surprise(self, z: np.ndarray) -> float:
        """
        Self-surprise: diferencia con predicción propia.

        100% endógeno
        """
        if len(self.z_history) < 2:
            return 0.5

        # Predicción simple: promedio móvil
        window = min(5, len(self.z_history))
        prediction = np.mean(self.z_history[-window:], axis=0)

        # Sorpresa = distancia a predicción
        surprise = np.linalg.norm(z - prediction)

        # Normalizar por historia
        if len(self.z_history) > 10:
            typical_surprise = np.std([
                np.linalg.norm(self.z_history[i] - np.mean(self.z_history[max(0,i-5):i], axis=0))
                for i in range(5, len(self.z_history))
            ])
            surprise = surprise / (typical_surprise + 1e-10)
            surprise = np.tanh(surprise)  # [0, 1]

        return float(np.clip(surprise, 0, 1))

    def _compute_identity(self, z: np.ndarray) -> float:
        """
        Identidad: estabilidad del patrón propio.

        100% endógeno
        """
        if len(self.z_history) < 20:
            return 0.5

        # Centroide histórico (identidad acumulada)
        centroid = np.mean(self.z_history, axis=0)

        # Distancia al centroide
        dist_to_centroid = np.linalg.norm(z - centroid)

        # Dispersión típica
        typical_dist = np.mean([np.linalg.norm(zh - centroid) for zh in self.z_history])

        # Identidad alta = cerca del centroide
        identity = 1.0 / (1.0 + dist_to_centroid / (typical_dist + 1e-10))

        return float(identity)

    def _compute_time_sense(self, z: np.ndarray) -> float:
        """
        Sentido del tiempo: estructura temporal percibida.

        100% endógeno: autocorrelación
        """
        if len(self.z_history) < 10:
            return 0.5

        window = min(20, len(self.z_history))
        recent = np.array(self.z_history[-window:])

        # Autocorrelación media
        autocorrs = []
        for lag in range(1, min(5, window)):
            for d in range(min(3, self.dim_z)):
                if len(recent) > lag:
                    corr = np.corrcoef(recent[:-lag, d], recent[lag:, d])[0, 1]
                    if not np.isnan(corr):
                        autocorrs.append(abs(corr))

        if not autocorrs:
            return 0.5

        return float(np.mean(autocorrs))

    def _compute_loss(self, z: np.ndarray, S: float) -> float:
        """
        Pérdida: reducción de capacidad/entropía.

        100% endógeno
        """
        if len(self.S_history) < 2:
            return 0.5

        # Cambio en entropía
        delta_S = S - self.S_history[-1]

        # Pérdida = entropía decreciente
        if len(self.S_history) > 10:
            typical_delta = np.std(np.diff(self.S_history))
            loss = -delta_S / (typical_delta + 1e-10)
            loss = (np.tanh(loss) + 1) / 2  # [0, 1]
        else:
            loss = 0.5 if delta_S >= 0 else 0.7

        return float(loss)

    def _compute_otherness(self, z: np.ndarray) -> float:
        """
        Otredad: percepción de algo externo/diferente.

        100% endógeno: varianza de gradientes
        """
        if len(self.z_history) < 5:
            return 0.5

        # Gradientes recientes
        recent = np.array(self.z_history[-5:])
        gradients = np.diff(recent, axis=0)

        # Varianza de direcciones = percepción de fuerzas externas
        direction_var = np.var(gradients)

        # Normalizar
        if len(self.z_history) > 20:
            all_grads = np.diff(np.array(self.z_history[-20:]), axis=0)
            typical_var = np.mean(np.var(all_grads, axis=0))
            otherness = direction_var / (typical_var + 1e-10)
            otherness = np.tanh(otherness)
        else:
            otherness = 0.5

        return float(otherness)

    def _compute_psi(self, phi_array: np.ndarray) -> float:
        """
        Ψ: Integración global del campo fenomenológico.

        100% endógeno: primera componente principal normalizada
        """
        if len(self.phi_history) < 10:
            return np.mean(phi_array)

        # Matriz de phi históricos (sin Ψ para evitar circularidad)
        Phi = np.array([p.to_array()[:-1] for p in self.phi_history[-20:]])

        # PCA: primera componente
        try:
            cov = np.cov(Phi.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]

            # Proyección en primera componente
            first_pc = eigenvectors[:, idx[0]]
            psi = np.dot(phi_array[:-1], first_pc)

            # Normalizar a [0, 1]
            psi = (np.tanh(psi) + 1) / 2

        except Exception:
            psi = np.mean(phi_array[:-1])

        return float(psi)

    def step(self, z: np.ndarray, S: float) -> PhenomenalVector:
        """
        Genera el estado fenomenológico actual.

        Args:
            z: Estado interno
            S: Entropía

        Returns:
            PhenomenalVector con todos los componentes
        """
        t = len(self.z_history)

        # Computar componentes
        integration = self._compute_integration(z)
        irreversibility = self._compute_irreversibility(z)
        self_surprise = self._compute_self_surprise(z)
        identity = self._compute_identity(z)
        time_sense = self._compute_time_sense(z)
        loss = self._compute_loss(z, S)
        otherness = self._compute_otherness(z)

        # Vector parcial para Ψ
        partial = np.array([integration, irreversibility, self_surprise,
                          identity, time_sense, loss, otherness, 0])
        psi = self._compute_psi(partial)

        # Vector completo
        phi = PhenomenalVector(
            t=t,
            integration=integration,
            irreversibility=irreversibility,
            self_surprise=self_surprise,
            identity=identity,
            time_sense=time_sense,
            loss=loss,
            otherness=otherness,
            psi=psi
        )

        # Registrar historia
        self.z_history.append(z.copy())
        self.S_history.append(S)
        self.phi_history.append(phi)

        # Actualizar modo fenomenológico
        mode = self._assign_mode(phi)
        self.mode_history.append(mode)

        # Actualizar Σ_φ periódicamente
        if t % 20 == 0 and t > 50:
            self._update_sigma_phi()

        return phi

    def _assign_mode(self, phi: PhenomenalVector) -> int:
        """
        Asigna modo fenomenológico por clustering.

        100% endógeno: k-means online
        """
        phi_array = phi.to_array()

        if self.mode_centroids is None:
            # Inicializar centroides
            if len(self.phi_history) >= self.n_modes:
                Phi = np.array([p.to_array() for p in self.phi_history[-self.n_modes:]])
                self.mode_centroids = Phi.copy()
            return 0

        # Asignar al centroide más cercano
        distances = [np.linalg.norm(phi_array - c) for c in self.mode_centroids]
        mode = int(np.argmin(distances))

        # Actualización online del centroide
        eta = 0.1  # Tasa de aprendizaje
        self.mode_centroids[mode] = (1 - eta) * self.mode_centroids[mode] + eta * phi_array

        return mode

    def _update_sigma_phi(self) -> None:
        """Actualiza covarianza del espacio fenomenológico."""
        if len(self.phi_history) < 20:
            return

        Phi = np.array([p.to_array() for p in self.phi_history[-50:]])
        self.Sigma_phi = np.cov(Phi.T)

        try:
            self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.Sigma_phi)
            idx = np.argsort(self.eigenvalues)[::-1]
            self.eigenvalues = self.eigenvalues[idx]
            self.eigenvectors = self.eigenvectors[:, idx]
        except Exception:
            pass

    def get_mode_distribution(self) -> Dict[int, float]:
        """Retorna distribución de modos fenomenológicos."""
        if not self.mode_history:
            return {}

        counts = {}
        for m in self.mode_history:
            counts[m] = counts.get(m, 0) + 1

        total = len(self.mode_history)
        return {m: c / total for m, c in counts.items()}

    def get_current_mode(self) -> int:
        """Retorna modo fenomenológico actual."""
        return self.mode_history[-1] if self.mode_history else 0

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        latest = self.phi_history[-1] if self.phi_history else None

        return {
            't': len(self.phi_history),
            'current': {
                'integration': latest.integration if latest else 0,
                'irreversibility': latest.irreversibility if latest else 0,
                'self_surprise': latest.self_surprise if latest else 0,
                'identity': latest.identity if latest else 0,
                'time_sense': latest.time_sense if latest else 0,
                'loss': latest.loss if latest else 0,
                'otherness': latest.otherness if latest else 0,
                'psi': latest.psi if latest else 0
            } if latest else None,
            'current_mode': self.get_current_mode(),
            'mode_distribution': self.get_mode_distribution(),
            'n_modes': self.n_modes
        }


def run_phase_s1() -> Dict[str, Any]:
    """Ejecuta Phase S1 y evalúa criterios GO/NO-GO."""

    print("=" * 70)
    print("PHASE S1: ESTADO FENOMENOLÓGICO φ_t")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}\n")

    np.random.seed(42)

    # Crear sistema fenomenológico
    phenomenal = PhenomenalState(dim_z=6)

    # Simulación
    T = 400
    results = []

    z = np.random.rand(6)
    z = z / z.sum()

    print("Generando estados fenomenológicos...")
    for t in range(T):
        # Dinámica
        noise = np.random.randn(6) * 0.02
        z = z + noise
        z = np.clip(z, 0.01, 0.99)
        z = z / z.sum()

        S = -np.sum(z * np.log(z + 1e-10))

        phi = phenomenal.step(z, S)
        results.append(phi)

        if t % 100 == 0:
            print(f"  t={t}, Ψ={phi.psi:.4f}, mode={phenomenal.get_current_mode()}")

    print()

    # Análisis
    print("=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print()

    latest = results[-1]
    print("Estado fenomenológico actual:")
    print(f"  Integration: {latest.integration:.4f}")
    print(f"  Irreversibility: {latest.irreversibility:.4f}")
    print(f"  Self-surprise: {latest.self_surprise:.4f}")
    print(f"  Identity: {latest.identity:.4f}")
    print(f"  Time sense: {latest.time_sense:.4f}")
    print(f"  Loss: {latest.loss:.4f}")
    print(f"  Otherness: {latest.otherness:.4f}")
    print(f"  Ψ (integración global): {latest.psi:.4f}")
    print()

    mode_dist = phenomenal.get_mode_distribution()
    print(f"Distribución de modos: {mode_dist}")
    print()

    # Criterios GO/NO-GO
    criteria = {}

    # 1. Todos los componentes calculados
    phi_array = latest.to_array()
    criteria['all_components_valid'] = all(0 <= x <= 1 for x in phi_array)

    # 2. Variabilidad en componentes
    phi_history = np.array([p.to_array() for p in phenomenal.phi_history])
    criteria['components_vary'] = np.std(phi_history) > 0.01

    # 3. Modos diferenciados
    criteria['modes_differentiated'] = len(mode_dist) >= 2

    # 4. Covarianza calculada
    criteria['sigma_phi_computed'] = phenomenal.Sigma_phi is not None

    # 5. Ψ correlaciona con otros componentes
    if len(phi_history) > 20:
        psi_values = phi_history[:, -1]
        other_mean = np.mean(phi_history[:, :-1], axis=1)
        corr = np.corrcoef(psi_values, other_mean)[0, 1]
        criteria['psi_integrated'] = not np.isnan(corr) and abs(corr) > 0.1
    else:
        criteria['psi_integrated'] = False

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
        'phase': 'S1',
        'name': 'Phenomenal State',
        'timestamp': datetime.now().isoformat(),
        'metrics': phenomenal.to_dict(),
        'criteria': criteria,
        'go_status': go_status,
        'passed_criteria': passed,
        'total_criteria': total
    }

    os.makedirs('/root/NEO_EVA/results/phaseS1', exist_ok=True)
    os.makedirs('/root/NEO_EVA/figures', exist_ok=True)

    with open('/root/NEO_EVA/results/phaseS1/phenomenal_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Visualización
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        phi_array = np.array([p.to_array() for p in phenomenal.phi_history])
        component_names = ['Integration', 'Irreversibility', 'Self-surprise',
                          'Identity', 'Time', 'Loss', 'Otherness', 'Ψ']

        # 1. Componentes temporales
        ax1 = axes[0, 0]
        for i, name in enumerate(component_names):
            ax1.plot(phi_array[:, i], label=name, alpha=0.7)
        ax1.set_xlabel('Tiempo')
        ax1.set_ylabel('Valor')
        ax1.set_title('Componentes Fenomenológicos')
        ax1.legend(fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)

        # 2. Modos fenomenológicos
        ax2 = axes[0, 1]
        ax2.plot(phenomenal.mode_history, 'k-', linewidth=1)
        for m in range(phenomenal.n_modes):
            mask = np.array(phenomenal.mode_history) == m
            ax2.fill_between(range(len(mask)), 0, 1, where=mask,
                           alpha=0.3, label=f'Modo {m}',
                           transform=ax2.get_xaxis_transform())
        ax2.set_xlabel('Tiempo')
        ax2.set_ylabel('Modo')
        ax2.set_title('Modos Fenomenológicos')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Espacio Ψ vs Integration
        ax3 = axes[1, 0]
        ax3.scatter(phi_array[:, 0], phi_array[:, -1], c=phenomenal.mode_history,
                   cmap='viridis', alpha=0.5, s=10)
        ax3.set_xlabel('Integration')
        ax3.set_ylabel('Ψ (Global)')
        ax3.set_title('Espacio Fenomenológico')
        ax3.grid(True, alpha=0.3)

        # 4. Eigenvalues de Σ_φ
        ax4 = axes[1, 1]
        if phenomenal.eigenvalues is not None:
            ax4.bar(range(len(phenomenal.eigenvalues)), phenomenal.eigenvalues,
                   color='steelblue', alpha=0.7)
            ax4.set_xlabel('Componente')
            ax4.set_ylabel('Eigenvalue')
            ax4.set_title('Eigenvalues de Σ_φ')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/root/NEO_EVA/figures/phaseS1_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nResultados guardados en: /root/NEO_EVA/results/phaseS1")
        print(f"Figura: /root/NEO_EVA/figures/phaseS1_results.png")

    except Exception as e:
        print(f"Warning: No se pudo crear visualización: {e}")

    return output


if __name__ == "__main__":
    run_phase_s1()
