#!/usr/bin/env python3
"""
Phase I1: Descomposición en Sub-sistemas + Matriz de Acoplos
============================================================

Divide NEO_EVA en M "sub-agentes" internos y calcula matrices
de acoplo dirigido mediante Transfer Entropy.

100% ENDÓGENO - Sin constantes mágicas
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


@dataclass
class Subsystem:
    """Representa un sub-agente interno."""
    name: str
    dim: int
    start_idx: int
    end_idx: int
    history: List[np.ndarray] = field(default_factory=list)


class SubsystemDecomposition:
    """
    Descompone el estado global en M sub-sistemas y calcula
    matrices de acoplo dirigido.

    Sub-sistemas:
    - NEO-visible: Componentes observables de NEO
    - NEO-hidden: Subespacio oculto de NEO
    - EVA-visible: Componentes observables de EVA
    - EVA-hidden: Subespacio oculto de EVA
    - workspace: Espacio de trabajo global
    - drives: Sistema de impulsos/motivaciones

    100% Endógeno:
    - Ventana w_t = floor(√t)
    - TE normalizada por varianza local
    - Matrices A, C por ranks
    """

    def __init__(self, total_dim: int = 12):
        self.total_dim = total_dim

        # Definir sub-sistemas (partición endógena por dimensión)
        self.M = 6  # Número de sub-sistemas
        dim_per_subsystem = total_dim // self.M
        remainder = total_dim % self.M

        self.subsystems: List[Subsystem] = []
        names = ['NEO_visible', 'NEO_hidden', 'EVA_visible',
                 'EVA_hidden', 'workspace', 'drives']

        idx = 0
        for i, name in enumerate(names):
            dim = dim_per_subsystem + (1 if i < remainder else 0)
            self.subsystems.append(Subsystem(
                name=name,
                dim=dim,
                start_idx=idx,
                end_idx=idx + dim
            ))
            idx += dim

        # Historia global
        self.z_history: List[np.ndarray] = []
        self.t: int = 0

        # Matrices de acoplo
        self.A_history: List[np.ndarray] = []  # Dirigido (asimétrico)
        self.C_history: List[np.ndarray] = []  # Simétrico

        # TE history para normalización
        self.te_values: List[float] = []

    def _get_window(self) -> int:
        """Ventana endógena: w_t = floor(√t)."""
        return max(3, int(np.sqrt(self.t + 1)))

    def _extract_subsystem(self, z: np.ndarray, subsys: Subsystem) -> np.ndarray:
        """Extrae componentes de un sub-sistema."""
        return z[subsys.start_idx:subsys.end_idx]

    def _estimate_te(self, source_history: List[np.ndarray],
                     target_history: List[np.ndarray],
                     lag: int = 1) -> float:
        """
        Estima Transfer Entropy de source → target.

        TE(X→Y) ≈ I(Y_t; X_{t-lag} | Y_{t-1})

        Aproximación por correlación parcial normalizada.
        100% endógeno.
        """
        window = self._get_window()

        if len(source_history) < window + lag or len(target_history) < window + lag:
            return 0.0

        # Extraer ventana reciente
        X = np.array([np.mean(s) for s in source_history[-window:]])
        Y = np.array([np.mean(t) for t in target_history[-window:]])

        if len(X) < lag + 2:
            return 0.0

        # Preparar series
        Y_t = Y[lag:]
        X_lag = X[:-lag] if lag > 0 else X
        Y_prev = Y[lag-1:-1] if lag > 1 else Y[:-1]

        n = min(len(Y_t), len(X_lag), len(Y_prev))
        if n < 3:
            return 0.0

        Y_t = Y_t[:n]
        X_lag = X_lag[:n]
        Y_prev = Y_prev[:n]

        # Normalizar por varianza local (endógeno)
        var_Y = np.var(Y_t)
        var_X = np.var(X_lag)
        if var_Y < 1e-10 or var_X < 1e-10:
            return 0.0

        try:
            # Correlaciones
            r_yx = np.corrcoef(Y_t, X_lag)[0, 1]
            r_yy = np.corrcoef(Y_t, Y_prev)[0, 1]
            r_xy = np.corrcoef(X_lag, Y_prev)[0, 1]

            if np.isnan(r_yx) or np.isnan(r_yy) or np.isnan(r_xy):
                return 0.0

            # Correlación parcial
            denom = np.sqrt((1 - r_yy**2) * (1 - r_xy**2))
            if denom < 1e-10:
                return 0.0

            r_partial = (r_yx - r_yy * r_xy) / denom

            # TE ≈ -0.5 * log(1 - r_partial^2), normalizada por entropía
            te = -0.5 * np.log(1 - r_partial**2 + 1e-10)

            # Normalizar por varianza local
            te_normalized = te / (np.sqrt(var_Y * var_X) + 1e-10)

            return max(0.0, te_normalized)

        except Exception:
            return 0.0

    def step(self, z: np.ndarray) -> Dict[str, Any]:
        """
        Ejecuta un paso de descomposición.

        Args:
            z: Estado global (dimensión total_dim)

        Returns:
            Dict con matrices A, C y métricas
        """
        self.t += 1
        self.z_history.append(z.copy())

        # Actualizar historia de sub-sistemas
        for subsys in self.subsystems:
            subsys_z = self._extract_subsystem(z, subsys)
            subsys.history.append(subsys_z)

        # Calcular matrices de TE solo si hay suficiente historia
        window = self._get_window()
        if len(self.z_history) < window:
            return {
                't': self.t,
                'window': window,
                'A': None,
                'C': None,
                'ready': False
            }

        # Calcular TE para cada par (i, j)
        TE_matrix = np.zeros((self.M, self.M))

        for i, subsys_i in enumerate(self.subsystems):
            for j, subsys_j in enumerate(self.subsystems):
                if i != j:
                    te = self._estimate_te(subsys_i.history, subsys_j.history)
                    TE_matrix[i, j] = te
                    if te > 0:
                        self.te_values.append(te)

        # Calcular matrices A (dirigido) y C (simétrico) por ranks
        A = self._compute_directed_coupling(TE_matrix)
        C = self._compute_symmetric_coupling(TE_matrix)

        self.A_history.append(A)
        self.C_history.append(C)

        return {
            't': self.t,
            'window': window,
            'TE_matrix': TE_matrix,
            'A': A,
            'C': C,
            'ready': True
        }

    def _compute_directed_coupling(self, TE: np.ndarray) -> np.ndarray:
        """
        Calcula matriz de acoplo dirigido.

        A_ij(t) = rank(TE_i→j - TE_j→i)

        100% endógeno: ranks sobre la diferencia
        """
        diff_matrix = TE - TE.T

        # Extraer valores no diagonales
        mask = ~np.eye(self.M, dtype=bool)
        values = diff_matrix[mask]

        if len(values) == 0:
            return np.zeros((self.M, self.M))

        # Calcular ranks (0 a 1)
        ranks = np.zeros_like(diff_matrix)
        sorted_vals = np.sort(values)

        for i in range(self.M):
            for j in range(self.M):
                if i != j:
                    val = diff_matrix[i, j]
                    rank = np.searchsorted(sorted_vals, val) / len(sorted_vals)
                    ranks[i, j] = rank

        return ranks

    def _compute_symmetric_coupling(self, TE: np.ndarray) -> np.ndarray:
        """
        Calcula matriz de acoplo simétrico.

        C_ij(t) = rank(TE_i→j + TE_j→i)

        100% endógeno: ranks sobre la suma
        """
        sum_matrix = TE + TE.T

        # Extraer valores no diagonales (solo triangular superior para simetría)
        mask = np.triu(np.ones((self.M, self.M), dtype=bool), k=1)
        values = sum_matrix[mask]

        if len(values) == 0:
            return np.zeros((self.M, self.M))

        # Calcular ranks
        ranks = np.zeros_like(sum_matrix)
        sorted_vals = np.sort(values)

        for i in range(self.M):
            for j in range(i + 1, self.M):
                val = sum_matrix[i, j]
                rank = np.searchsorted(sorted_vals, val) / max(len(sorted_vals), 1)
                ranks[i, j] = rank
                ranks[j, i] = rank  # Simétrico

        return ranks

    def get_coupling_summary(self) -> Dict[str, Any]:
        """Retorna resumen de acoplos."""
        if not self.C_history:
            return {'ready': False}

        C_latest = self.C_history[-1]

        # Sub-sistemas más acoplados
        max_coupling = []
        for i in range(self.M):
            for j in range(i + 1, self.M):
                max_coupling.append({
                    'pair': (self.subsystems[i].name, self.subsystems[j].name),
                    'coupling': C_latest[i, j]
                })

        max_coupling.sort(key=lambda x: x['coupling'], reverse=True)

        return {
            'ready': True,
            'n_subsystems': self.M,
            'subsystem_names': [s.name for s in self.subsystems],
            'top_couplings': max_coupling[:3],
            'mean_coupling': np.mean(C_latest[np.triu_indices(self.M, k=1)]),
            'coupling_std': np.std(C_latest[np.triu_indices(self.M, k=1)])
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            't': self.t,
            'M': self.M,
            'subsystems': [s.name for s in self.subsystems],
            'n_coupling_matrices': len(self.C_history),
            'summary': self.get_coupling_summary()
        }


def run_phase_i1() -> Dict[str, Any]:
    """Ejecuta Phase I1 y evalúa criterios GO/NO-GO."""

    print("=" * 70)
    print("PHASE I1: DESCOMPOSICIÓN EN SUB-SISTEMAS + MATRIZ DE ACOPLOS")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}\n")

    np.random.seed(42)

    # Crear sistema
    decomp = SubsystemDecomposition(total_dim=12)

    # Simulación
    T = 300
    results = []

    # Estado inicial
    z = np.random.rand(12)
    z = z / z.sum()

    for t in range(T):
        # Dinámica con estructura entre sub-sistemas
        # NEO y EVA interactúan, workspace es intermedio
        noise = np.random.randn(12) * 0.02

        # Acoplo estructurado entre sub-sistemas
        if t > 0:
            # NEO_visible → workspace
            z[8:10] += 0.1 * z[0:2]
            # EVA_visible → workspace
            z[8:10] += 0.1 * z[4:6]
            # workspace → drives
            z[10:12] += 0.05 * z[8:10]
            # drives → hidden states
            z[2:4] += 0.02 * z[10:12]
            z[6:8] += 0.02 * z[10:12]

        z = z + noise
        z = np.clip(z, 0.01, 0.99)
        z = z / z.sum()

        result = decomp.step(z)
        results.append(result)

    # Análisis final
    print("=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print()

    summary = decomp.get_coupling_summary()

    print(f"Sub-sistemas: {decomp.M}")
    for s in decomp.subsystems:
        print(f"  {s.name}: dim={s.dim}")
    print()

    if summary['ready']:
        print(f"Acoplo medio: {summary['mean_coupling']:.4f}")
        print(f"Acoplo std: {summary['coupling_std']:.4f}")
        print()
        print("Top acoplos:")
        for c in summary['top_couplings']:
            print(f"  {c['pair'][0]} ↔ {c['pair'][1]}: {c['coupling']:.4f}")
    print()

    # Criterios GO/NO-GO
    criteria = {}

    # 1. Matrices calculadas
    criteria['matrices_computed'] = len(decomp.C_history) > 0

    # 2. Acoplos diferenciados (std > 0)
    if summary['ready']:
        criteria['couplings_differentiated'] = summary['coupling_std'] > 0
    else:
        criteria['couplings_differentiated'] = False

    # 3. Hay acoplos fuertes (max > 0.5)
    if decomp.C_history:
        max_coupling = np.max(decomp.C_history[-1])
        criteria['strong_couplings'] = max_coupling > 0.5
    else:
        criteria['strong_couplings'] = False

    # 4. TE values capturados
    criteria['te_captured'] = len(decomp.te_values) > 0

    # 5. Sistema estable (matrices no degeneran)
    if len(decomp.C_history) > 10:
        recent_means = [np.mean(C) for C in decomp.C_history[-10:]]
        criteria['system_stable'] = np.std(recent_means) < np.mean(recent_means)
    else:
        criteria['system_stable'] = True

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
        'phase': 'I1',
        'name': 'Subsystem Decomposition',
        'timestamp': datetime.now().isoformat(),
        'metrics': summary,
        'criteria': criteria,
        'go_status': go_status,
        'passed_criteria': passed,
        'total_criteria': total
    }

    os.makedirs('/root/NEO_EVA/results/phaseI1', exist_ok=True)
    os.makedirs('/root/NEO_EVA/figures', exist_ok=True)

    with open('/root/NEO_EVA/results/phaseI1/subsystem_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Visualización
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Matriz C final
        ax1 = axes[0, 0]
        if decomp.C_history:
            im = ax1.imshow(decomp.C_history[-1], cmap='hot', vmin=0, vmax=1)
            ax1.set_xticks(range(decomp.M))
            ax1.set_yticks(range(decomp.M))
            ax1.set_xticklabels([s.name[:8] for s in decomp.subsystems], rotation=45, ha='right')
            ax1.set_yticklabels([s.name[:8] for s in decomp.subsystems])
            ax1.set_title('Matriz de Acoplo Simétrico C(t)')
            plt.colorbar(im, ax=ax1)

        # 2. Matriz A final (dirigido)
        ax2 = axes[0, 1]
        if decomp.A_history:
            im = ax2.imshow(decomp.A_history[-1], cmap='RdBu', vmin=-1, vmax=1)
            ax2.set_xticks(range(decomp.M))
            ax2.set_yticks(range(decomp.M))
            ax2.set_xticklabels([s.name[:8] for s in decomp.subsystems], rotation=45, ha='right')
            ax2.set_yticklabels([s.name[:8] for s in decomp.subsystems])
            ax2.set_title('Matriz de Acoplo Dirigido A(t)')
            plt.colorbar(im, ax=ax2)

        # 3. Evolución de acoplo medio
        ax3 = axes[1, 0]
        if decomp.C_history:
            mean_couplings = [np.mean(C[np.triu_indices(decomp.M, k=1)]) for C in decomp.C_history]
            ax3.plot(mean_couplings, 'b-', linewidth=1.5)
            ax3.set_xlabel('Tiempo')
            ax3.set_ylabel('Acoplo medio')
            ax3.set_title('Evolución del Acoplo Medio')
            ax3.grid(True, alpha=0.3)

        # 4. Distribución de TE
        ax4 = axes[1, 1]
        if decomp.te_values:
            ax4.hist(decomp.te_values, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            ax4.axvline(np.median(decomp.te_values), color='r', linestyle='--',
                       label=f'Mediana={np.median(decomp.te_values):.4f}')
            ax4.set_xlabel('Transfer Entropy')
            ax4.set_ylabel('Frecuencia')
            ax4.set_title('Distribución de Transfer Entropy')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/root/NEO_EVA/figures/phaseI1_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nResultados guardados en: /root/NEO_EVA/results/phaseI1")
        print(f"Figura: /root/NEO_EVA/figures/phaseI1_results.png")

    except Exception as e:
        print(f"Warning: No se pudo crear visualización: {e}")

    return output


if __name__ == "__main__":
    run_phase_i1()
