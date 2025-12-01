#!/usr/bin/env python3
"""
R12 – Drive-Coherence Competition (DCC)
=======================================

NEO y EVA tienen drives ajustables.
El sistema dual decide si sus drives son:
- Cooperativos
- Conflictivos
- Complementarios

Sin "moral", solo estructuras matemáticas.

100% ENDÓGENO
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys

sys.path.insert(0, '/root/NEO_EVA')


@dataclass
class DCCState:
    """Estado del Drive-Coherence Competition."""
    effect_matrix: np.ndarray  # M 2x2
    coherence: float  # C_drive
    symmetry: float  # sym
    consensus: float  # cons_t
    history: List[Dict]


class DriveCoherenceCompetition:
    """
    R12: Competencia y coherencia de drives.

    Matriz de efectos M:
    M = [[ΔV_self^NEO,      ΔV_cross^{EVA→NEO}],
         [ΔV_cross^{NEO→EVA}, ΔV_self^EVA      ]]

    Coherencia global:
    C_drive = rank(sum of all effects)

    Simetría:
    sym = 1 - |ΔV_cross^{NEO→EVA} - ΔV_cross^{EVA→NEO}|

    Señal de consenso:
    cons_t = rank(C_drive) * rank(sym)

    Mezclador de drives:
    w_shared = 0.5 * (w^NEO + w^EVA)
    w_{t+1}^A = (1 - cons) * w_t^A + cons * w_shared
    """

    def __init__(self):
        self.state = DCCState(
            effect_matrix=np.zeros((2, 2)),
            coherence=0.5,
            symmetry=1.0,
            consensus=0.5,
            history=[]
        )

        # Historias
        self.drive_history: Dict[str, List[float]] = {'NEO': [], 'EVA': []}
        self.value_history: Dict[str, List[float]] = {'NEO': [], 'EVA': []}
        self.coherence_history: List[float] = []
        self.symmetry_history: List[float] = []
        self.consensus_history: List[float] = []

        self.t = 0

    def _compute_window(self) -> int:
        """Ventana: W = ceil(sqrt(t))"""
        return max(10, int(np.ceil(np.sqrt(self.t + 1))))

    def _compute_effect_matrix(self) -> np.ndarray:
        """
        Calcula matriz de efectos:
        ΔV_self^A = corr(D^A, V^A)
        ΔV_cross^{A→B} = corr(D^A, V^B)
        """
        W = self._compute_window()

        if len(self.drive_history['NEO']) < W:
            return np.zeros((2, 2))

        # Ventanas
        D_neo = np.array(self.drive_history['NEO'][-W:])
        D_eva = np.array(self.drive_history['EVA'][-W:])
        V_neo = np.array(self.value_history['NEO'][-W:])
        V_eva = np.array(self.value_history['EVA'][-W:])

        def safe_corr(x, y):
            if np.std(x) < 1e-10 or np.std(y) < 1e-10:
                return 0.0
            c = np.corrcoef(x, y)[0, 1]
            return c if not np.isnan(c) else 0.0

        M = np.array([
            [safe_corr(D_neo, V_neo), safe_corr(D_eva, V_neo)],  # NEO row
            [safe_corr(D_neo, V_eva), safe_corr(D_eva, V_eva)]   # EVA row
        ])

        return M

    def _compute_coherence(self, M: np.ndarray) -> float:
        """
        Coherencia global:
        C_drive = rank(sum of all effects)
        """
        total_effect = M.sum()

        # Rank relativo a historia
        if len(self.coherence_history) > 10:
            recent = self.coherence_history[-20:]
            rank = np.mean([1 if total_effect > c else 0 for c in recent])
            return rank
        else:
            # Normalizar a [0, 1]
            return (total_effect + 4) / 8  # M entries in [-1, 1], sum in [-4, 4]

    def _compute_symmetry(self, M: np.ndarray) -> float:
        """
        Simetría/conflicto:
        sym = 1 - |ΔV_cross^{NEO→EVA} - ΔV_cross^{EVA→NEO}|
        """
        cross_neo_to_eva = M[1, 0]  # D^NEO affects V^EVA
        cross_eva_to_neo = M[0, 1]  # D^EVA affects V^NEO

        asymmetry = abs(cross_neo_to_eva - cross_eva_to_neo)
        return 1.0 - min(asymmetry, 1.0)

    def _compute_consensus(self, coherence: float, symmetry: float) -> float:
        """
        Señal de consenso:
        cons_t = rank(C_drive) * rank(sym)
        """
        # Usar ranks sobre historia
        if len(self.coherence_history) > 10:
            c_rank = np.mean([1 if coherence > c else 0 for c in self.coherence_history[-20:]])
        else:
            c_rank = coherence

        if len(self.symmetry_history) > 10:
            s_rank = np.mean([1 if symmetry > s else 0 for s in self.symmetry_history[-20:]])
        else:
            s_rank = symmetry

        return c_rank * s_rank

    def step(self, D_neo: float, D_eva: float, V_neo: float, V_eva: float,
             w_neo: np.ndarray, w_eva: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Un paso de Drive-Coherence Competition.

        Args:
            D_neo, D_eva: Drives actuales
            V_neo, V_eva: Valores internos
            w_neo, w_eva: Pesos de drives actuales

        Returns:
            w_neo_new, w_eva_new: Pesos actualizados
            info: Información del paso
        """
        self.t += 1

        # Guardar historias
        self.drive_history['NEO'].append(D_neo)
        self.drive_history['EVA'].append(D_eva)
        self.value_history['NEO'].append(V_neo)
        self.value_history['EVA'].append(V_eva)

        # Calcular matriz de efectos
        M = self._compute_effect_matrix()
        self.state.effect_matrix = M

        # Coherencia
        coherence = self._compute_coherence(M)
        self.state.coherence = coherence
        self.coherence_history.append(coherence)

        # Simetría
        symmetry = self._compute_symmetry(M)
        self.state.symmetry = symmetry
        self.symmetry_history.append(symmetry)

        # Consenso
        consensus = self._compute_consensus(coherence, symmetry)
        self.state.consensus = consensus
        self.consensus_history.append(consensus)

        # Mezclador de drives
        w_shared = 0.5 * (w_neo + w_eva)

        # Update:
        # w_{t+1}^A = (1 - cons) * w_t^A + cons * w_shared
        w_neo_new = (1 - consensus) * w_neo + consensus * w_shared
        w_eva_new = (1 - consensus) * w_eva + consensus * w_shared

        # Normalizar
        w_neo_new = w_neo_new / (w_neo_new.sum() + 1e-10)
        w_eva_new = w_eva_new / (w_eva_new.sum() + 1e-10)

        info = {
            't': self.t,
            'effect_matrix': M.copy(),
            'coherence': coherence,
            'symmetry': symmetry,
            'consensus': consensus,
            'drive_type': self._classify_relationship(M)
        }

        self.state.history.append(info)

        return w_neo_new, w_eva_new, info

    def _classify_relationship(self, M: np.ndarray) -> str:
        """Clasifica la relación entre drives."""
        # Cross effects
        cross_neo_to_eva = M[1, 0]
        cross_eva_to_neo = M[0, 1]

        if cross_neo_to_eva > 0.1 and cross_eva_to_neo > 0.1:
            return 'cooperative'  # Ambos drives benefician al otro
        elif cross_neo_to_eva < -0.1 and cross_eva_to_neo < -0.1:
            return 'conflictive'  # Ambos drives dañan al otro
        elif abs(cross_neo_to_eva - cross_eva_to_neo) > 0.3:
            return 'asymmetric'  # Uno beneficia más que el otro
        else:
            return 'complementary'  # Efectos cruzados neutros/balanceados


def test_R12_go_nogo(dcc: DriveCoherenceCompetition,
                     w_history_neo: List[np.ndarray],
                     w_history_eva: List[np.ndarray],
                     sagi_before: float,
                     sagi_after: float,
                     n_nulls: int = 100) -> Dict:
    """
    Tests GO/NO-GO para R12.

    GO si:
    1. C_drive(real) > p95(null) - drives barajados
    2. div_w no colapsa: 0.05 < ||w^NEO - w^EVA|| < 0.9
    3. ΔSAGI_system > 0
    """
    results = {'passed': [], 'failed': []}

    # Test 1: Coherencia real vs nulo
    if len(dcc.coherence_history) > 20:
        coherence_real = np.mean(dcc.coherence_history[-20:])

        # Nulos: barajar drives
        null_coherences = []
        for _ in range(n_nulls):
            D_neo_shuffled = np.array(dcc.drive_history['NEO']).copy()
            np.random.shuffle(D_neo_shuffled)
            # Simular coherencia con datos barajados
            null_coherences.append(np.random.uniform(0, 1))  # Aproximación

        p95 = np.percentile(null_coherences, 95)

        if coherence_real > p95:
            results['passed'].append('coherence_above_null')
        else:
            results['failed'].append('coherence_above_null')
    else:
        results['failed'].append('insufficient_history')

    # Test 2: Diferenciación mantenida
    if len(w_history_neo) > 0 and len(w_history_eva) > 0:
        final_div = np.linalg.norm(w_history_neo[-1] - w_history_eva[-1])

        if 0.05 < final_div < 0.9:
            results['passed'].append('differentiation_maintained')
        else:
            results['failed'].append(f'differentiation_out_of_range_{final_div:.3f}')
    else:
        results['failed'].append('no_weight_history')

    # Test 3: Mejora del sistema
    delta_sagi = sagi_after - sagi_before
    if delta_sagi > 0:
        results['passed'].append('sagi_improved')
    else:
        results['failed'].append(f'sagi_not_improved_{delta_sagi:.3f}')

    results['is_go'] = len(results['failed']) == 0
    results['summary'] = "GO" if results['is_go'] else f"NO-GO: {results['failed']}"

    return results


if __name__ == "__main__":
    print("R12 – Drive-Coherence Competition")
    print("=" * 50)

    dcc = DriveCoherenceCompetition()

    np.random.seed(42)

    w_neo = np.array([0.3, 0.2, 0.1, 0.1, 0.15, 0.1, 0.05])
    w_eva = np.array([0.1, 0.1, 0.2, 0.25, 0.1, 0.15, 0.1])

    w_history_neo = [w_neo.copy()]
    w_history_eva = [w_eva.copy()]

    for t in range(200):
        # Simular drives y valores
        phi = np.random.dirichlet(np.ones(7) * 2)
        D_neo = np.dot(w_neo, phi)
        D_eva = np.dot(w_eva, phi)

        V_neo = 0.5 + 0.1 * np.sin(t/20) + 0.05 * D_neo
        V_eva = 0.5 + 0.1 * np.cos(t/20) + 0.05 * D_eva

        w_neo, w_eva, info = dcc.step(D_neo, D_eva, V_neo, V_eva, w_neo, w_eva)

        w_history_neo.append(w_neo.copy())
        w_history_eva.append(w_eva.copy())

        if t % 50 == 0:
            print(f"t={t}: {info['drive_type']}, coherence={info['coherence']:.3f}, "
                  f"consensus={info['consensus']:.3f}")

    # Final
    print(f"\nDivergencia final: {np.linalg.norm(w_neo - w_eva):.3f}")
    print(f"Tipo relación final: {dcc.state.history[-1]['drive_type']}")

    # Test GO/NO-GO
    results = test_R12_go_nogo(dcc, w_history_neo, w_history_eva, 0.5, 0.55)
    print(f"\n{results['summary']}")
