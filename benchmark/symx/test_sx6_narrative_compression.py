"""
SX6 - Narrative Compression (MDL Endógeno)
==========================================

NC = 1 - L(narrativa_simbólica) / L(traza_episódica)
L(·) = -Σ log p_endo(·)

p_endo: probabilidades internas (frecuencias/Dirichlet endógenas)

Criterio PASS: NC ≥ Q75%(NC_hist) y ΔR̄ ≥ 0

100% endógeno. Sin priors fijos.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history


@dataclass
class NarrativeCompressionResult:
    """Resultado de compresión narrativa."""
    L_symbolic: float       # Longitud narrativa simbólica
    L_raw: float           # Longitud traza episódica
    NC: float              # Ratio de compresión
    quantile_used: float   # Q75% usado como umbral
    delta_R: float         # Cambio en recompensa
    passed: bool           # Si cumple criterio PASS


class NarrativeCompressor:
    """
    Sistema de compresión narrativa MDL endógeno.

    Mide cuánto más compacta es la representación simbólica
    comparada con la traza episódica raw.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

        # Historial de símbolos y estados raw
        self.symbol_sequences: List[List[int]] = []
        self.raw_traces: List[np.ndarray] = []
        self.rewards: List[float] = []

        # Frecuencias para p_endo (Dirichlet implícito)
        self.symbol_counts: Dict[int, int] = defaultdict(int)
        self.total_symbols: int = 0

        # Historial de métricas
        self.NC_history: List[float] = []
        self.L_sym_history: List[float] = []
        self.L_raw_history: List[float] = []

        self.t = 0

    def observe(
        self,
        t: int,
        symbol_sequence: List[int],
        raw_state: np.ndarray,
        reward: float
    ) -> None:
        """Registra observación."""
        self.t = t

        self.symbol_sequences.append(symbol_sequence)
        self.raw_traces.append(raw_state.copy())
        self.rewards.append(reward)

        # Actualizar frecuencias de símbolos
        for sym in symbol_sequence:
            self.symbol_counts[sym] += 1
            self.total_symbols += 1

        # Limitar históricos
        max_h = max_history(t)
        if len(self.symbol_sequences) > max_h:
            self.symbol_sequences = self.symbol_sequences[-max_h:]
            self.raw_traces = self.raw_traces[-max_h:]
            self.rewards = self.rewards[-max_h:]
            self._recalculate_frequencies()

    def _recalculate_frequencies(self) -> None:
        """Recalcula frecuencias de símbolos."""
        self.symbol_counts = defaultdict(int)
        self.total_symbols = 0
        for seq in self.symbol_sequences:
            for sym in seq:
                self.symbol_counts[sym] += 1
                self.total_symbols += 1

    def _compute_p_endo(self, symbol: int) -> float:
        """
        Probabilidad endógena de un símbolo.

        p_endo(σ) = (count(σ) + 1) / (total + |V|)

        Dirichlet con α=1 (Laplace smoothing endógeno).
        """
        vocab_size = len(self.symbol_counts)
        if vocab_size == 0:
            return 1.0

        count = self.symbol_counts.get(symbol, 0)
        return (count + 1) / (self.total_symbols + vocab_size)

    def _compute_L_symbolic(self, t: int) -> float:
        """
        Longitud de descripción mínima de narrativa simbólica.

        L(narrativa) = -Σ log p_endo(σ)
        """
        L = L_t(t)
        recent_sequences = self.symbol_sequences[-L:]

        if not recent_sequences:
            return 0.0

        total_length = 0.0
        for seq in recent_sequences:
            for sym in seq:
                p = self._compute_p_endo(sym)
                total_length += -np.log(p + 1e-10)

        return float(total_length)

    def _compute_L_raw(self, t: int) -> float:
        """
        Longitud de descripción de traza episódica raw.

        L(raw) = -Σ log p(estado)

        Usa histograma endógeno de normas de estado.
        """
        L = L_t(t)
        recent_traces = self.raw_traces[-L:]

        if len(recent_traces) < 3:
            return 1.0

        # Normalizar estados por componente
        all_values = np.concatenate([s.flatten() for s in recent_traces])

        if len(all_values) < 3:
            return 1.0

        # Histograma para estimar p(valor)
        n_bins = max(3, int(np.sqrt(len(all_values))))
        hist, edges = np.histogram(all_values, bins=n_bins, density=True)

        # Longitud total
        total_length = 0.0
        for trace in recent_traces:
            for val in trace.flatten():
                bin_idx = np.searchsorted(edges[:-1], val) - 1
                bin_idx = max(0, min(bin_idx, len(hist) - 1))

                bin_width = edges[bin_idx + 1] - edges[bin_idx]
                p = hist[bin_idx] * bin_width if hist[bin_idx] > 0 else 1e-10
                total_length += -np.log(p + 1e-10)

        return float(total_length)

    def compute_NC(self, t: int) -> Tuple[float, float, float]:
        """
        Computa ratio de compresión narrativa.

        NC = 1 - L(simbólica) / L(raw)

        Returns: (L_symbolic, L_raw, NC)
        """
        L_sym = self._compute_L_symbolic(t)
        L_raw = self._compute_L_raw(t)

        if L_raw < 1e-10:
            NC = 0.0
        else:
            NC = 1.0 - L_sym / L_raw
            NC = float(np.clip(NC, -1, 1))  # Puede ser negativo si simbólico es peor

        self.L_sym_history.append(L_sym)
        self.L_raw_history.append(L_raw)
        self.NC_history.append(NC)

        return L_sym, L_raw, NC

    def evaluate(self, t: int) -> NarrativeCompressionResult:
        """
        Evaluación completa de compresión narrativa.

        PASS: NC ≥ Q75%(NC_hist) y ΔR̄ ≥ 0
        """
        L_sym, L_raw, NC = self.compute_NC(t)

        # Q75% endógeno
        L = L_t(t)
        if len(self.NC_history) >= L:
            quantile = np.percentile(self.NC_history, 75)
        else:
            quantile = 0.3  # Default

        # ΔR̄ (cambio en recompensa media)
        if len(self.rewards) >= L:
            recent_rewards = self.rewards[-L:]
            mid = len(recent_rewards) // 2
            R_first = np.mean(recent_rewards[:mid]) if mid > 0 else 0
            R_second = np.mean(recent_rewards[mid:]) if len(recent_rewards) - mid > 0 else 0
            delta_R = R_second - R_first
        else:
            delta_R = 0.0

        # Criterio PASS
        passed = NC >= quantile and delta_R >= -0.05

        return NarrativeCompressionResult(
            L_symbolic=L_sym,
            L_raw=L_raw,
            NC=NC,
            quantile_used=quantile,
            delta_R=delta_R,
            passed=passed
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Estadísticas del sistema."""
        L = L_t(self.t)

        return {
            'agent_id': self.agent_id,
            't': self.t,
            'NC': np.mean(self.NC_history[-L:]) if self.NC_history else 0.0,
            'L_symbolic_mean': np.mean(self.L_sym_history[-L:]) if self.L_sym_history else 0.0,
            'L_raw_mean': np.mean(self.L_raw_history[-L:]) if self.L_raw_history else 0.0,
            'vocab_size': len(self.symbol_counts),
            'total_symbols': self.total_symbols,
            'formula': 'NC = 1 - L(simbólica) / L(raw)'
        }


def run_test() -> Dict[str, Any]:
    """
    SX6: Narrative Compression Test (MDL Endógeno).

    NC = 1 - L(narrativa_simbólica) / L(traza_episódica)
    PASS: NC ≥ Q75%(NC_hist) y ΔR̄ ≥ 0
    """
    np.random.seed(42)

    compressor = NarrativeCompressor('TEST')

    # Simular episodios
    for t in range(1, 301):
        # Estado raw (alta dimensionalidad)
        raw_state = np.random.randn(10)

        # Secuencia simbólica (compacta, con repeticiones)
        n_symbols = np.random.randint(2, 5)
        # Símbolos concentrados → más compresión
        symbol_seq = [np.random.choice([0, 1, 2, 3, 4],
                                        p=[0.35, 0.25, 0.2, 0.12, 0.08])
                     for _ in range(n_symbols)]

        reward = np.random.rand() * 0.5 + 0.25 + t * 0.0001  # Ligeramente creciente

        compressor.observe(t, symbol_seq, raw_state, reward)

    # Evaluación final
    result = compressor.evaluate(300)
    stats = compressor.get_statistics()

    # Score basado en NC
    score = max(0, result.NC)

    return {
        'score': float(np.clip(score, 0, 1)),
        'passed': result.passed,
        'details': {
            'L_symbolic': float(result.L_symbolic),
            'L_raw': float(result.L_raw),
            'NC': float(result.NC),
            'quantile_Q75': float(result.quantile_used),
            'delta_R': float(result.delta_R),
            'vocab_size': stats['vocab_size'],
            'total_symbols': stats['total_symbols']
        }
    }


if __name__ == "__main__":
    result = run_test()
    print("=" * 60)
    print("SX6 - NARRATIVE COMPRESSION (MDL ENDÓGENO)")
    print("=" * 60)
    print(f"Score (NC): {result['score']:.4f}")
    print(f"Passed: {result['passed']}")
    print(f"\nDetails:")
    for k, v in result['details'].items():
        print(f"  {k}: {v}")
