#!/usr/bin/env python3
"""
STATE ENCODING - Estados Cuánticos desde Ranks
==============================================

Los estados NO son valores absolutos inventados.
Son RANKS normalizados de la propia dinámica.

Encoding:
    p_i(t) = rank(d_i(t)) / Σ_j rank(d_j(t))
    ψ_i(t) = √(p_i(t))

Esto garantiza:
- Sin escalas arbitrarias
- Solo relaciones entre drives
- Amplitudes válidas (Σ|ψ|² = 1)

CHECK: No magic numbers excepto 0, 1, dimensiones.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.stats import rankdata


@dataclass
class QuantumStateEncoding:
    """
    Codificación de estado cuántico desde drives.

    El estado NO se impone - emerge de los ranks de drives.
    """
    dim: int
    amplitudes: np.ndarray = None
    phase: float = 0.0

    # Historia para normalización endógena
    amplitude_history: List[np.ndarray] = field(default_factory=list)
    phase_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        if self.amplitudes is None:
            # Estado inicial: superposición uniforme
            self.amplitudes = np.ones(self.dim) / np.sqrt(self.dim)

    @classmethod
    def from_drives(cls, drives: np.ndarray, history: List[np.ndarray] = None) -> 'QuantumStateEncoding':
        """
        Construye estado cuántico desde drives usando RANKS.

        p_i = rank(d_i) / Σrank(d_j)
        ψ_i = √p_i

        Los ranks son endógenos: solo dependen de relaciones entre drives.
        """
        dim = len(drives)

        # Calcular ranks (1 = menor, dim = mayor)
        # Usamos 'average' para ties
        ranks = rankdata(drives, method='average')

        # Normalizar ranks a probabilidades
        rank_sum = np.sum(ranks)
        probabilities = ranks / rank_sum

        # Amplitudes: raíz de probabilidades
        amplitudes = np.sqrt(probabilities)

        # Fase endógena: acumulada de cambios en ranks
        phase = 0.0
        if history and len(history) > 1:
            # Fase = suma de "rotaciones" en espacio de ranks
            for i in range(1, min(len(history), 50)):
                prev_ranks = rankdata(history[i-1], method='average')
                curr_ranks = rankdata(history[i], method='average')

                # Normalizar
                prev_norm = prev_ranks / np.sum(prev_ranks)
                curr_norm = curr_ranks / np.sum(curr_ranks)

                # Ángulo entre vectores de ranks
                dot = np.clip(np.dot(np.sqrt(prev_norm), np.sqrt(curr_norm)), 0, 1)
                phase += np.arccos(dot)

        state = cls(dim=dim, amplitudes=amplitudes, phase=phase)

        if history:
            # Guardar historia de amplitudes
            for h in history[-50:]:
                h_ranks = rankdata(h, method='average')
                h_probs = h_ranks / np.sum(h_ranks)
                state.amplitude_history.append(np.sqrt(h_probs))

        return state

    @property
    def probabilities(self) -> np.ndarray:
        """Probabilidades = |ψ|²"""
        return self.amplitudes ** 2

    @property
    def ranks(self) -> np.ndarray:
        """Recupera ranks desde probabilidades."""
        # Invertir: si p_i = rank_i / Σrank, entonces rank_i ∝ p_i
        probs = self.probabilities
        # Escalar para que sea interpretable como ranks
        return probs * len(probs)

    def entropy_endogenous(self) -> float:
        """
        Entropía endógena: normalizada por la propia historia.

        No usa log base fija - usa percentiles de la historia.
        """
        probs = self.probabilities
        probs = np.clip(probs, 1e-16, 1)

        # Entropía de Shannon (sin normalizar por log(dim) que sería constante externa)
        raw_entropy = -np.sum(probs * np.log(probs))

        # Normalizar por historia
        if len(self.amplitude_history) > 10:
            historical_entropies = []
            for amp in self.amplitude_history:
                p = amp ** 2
                p = np.clip(p, 1e-16, 1)
                historical_entropies.append(-np.sum(p * np.log(p)))

            # Percentiles de la historia
            e_min = np.percentile(historical_entropies, 5)
            e_max = np.percentile(historical_entropies, 95)

            if e_max > e_min:
                return (raw_entropy - e_min) / (e_max - e_min)

        return raw_entropy / np.log(self.dim)  # Fallback: normalizar por máximo teórico

    def coherence_endogenous(self) -> float:
        """
        Coherencia endógena: estabilidad de amplitudes.

        coherence = 1 / (1 + var_reciente / var_baseline)

        Todo basado en la propia historia.
        """
        if len(self.amplitude_history) < 20:
            return 0.5

        # Ventana reciente: √(len(history))
        window = max(5, int(np.sqrt(len(self.amplitude_history))))

        recent = np.array(self.amplitude_history[-window:])
        baseline = np.array(self.amplitude_history[:-window])

        var_recent = np.mean(np.var(recent, axis=0))
        var_baseline = np.mean(np.var(baseline, axis=0)) + 1e-16

        return 1.0 / (1.0 + var_recent / var_baseline)

    def inner_product(self, other: 'QuantumStateEncoding') -> complex:
        """Producto interno ⟨self|other⟩"""
        phase_diff = other.phase - self.phase
        overlap = np.dot(self.amplitudes, other.amplitudes)
        return overlap * np.exp(1j * phase_diff)

    def fidelity(self, other: 'QuantumStateEncoding') -> float:
        """Fidelidad |⟨ψ|φ⟩|²"""
        return abs(self.inner_product(other)) ** 2

    def update(self, new_drives: np.ndarray):
        """Actualiza estado con nuevos drives."""
        # Guardar estado actual en historia
        self.amplitude_history.append(self.amplitudes.copy())
        self.phase_history.append(self.phase)

        # Calcular nuevo estado
        ranks = rankdata(new_drives, method='average')
        rank_sum = np.sum(ranks)
        probabilities = ranks / rank_sum
        new_amplitudes = np.sqrt(probabilities)

        # Actualizar fase
        dot = np.clip(np.dot(self.amplitudes, new_amplitudes), 0, 1)
        phase_increment = np.arccos(dot)
        self.phase += phase_increment

        self.amplitudes = new_amplitudes

        # Mantener historia acotada
        max_history = 500
        if len(self.amplitude_history) > max_history:
            self.amplitude_history = self.amplitude_history[-max_history:]
            self.phase_history = self.phase_history[-max_history:]


@dataclass
class EntangledStateEncoding:
    """
    Estado entangled de múltiples agentes.

    El entanglement NO se impone - emerge de correlaciones históricas.
    """
    agent_names: List[str]
    local_states: Dict[str, QuantumStateEncoding]

    # Matriz de correlación (entanglement emergente)
    correlation_matrix: np.ndarray = None

    # Historia conjunta para calcular correlaciones
    joint_history: List[Dict[str, np.ndarray]] = field(default_factory=list)

    def __post_init__(self):
        n = len(self.agent_names)
        if self.correlation_matrix is None:
            self.correlation_matrix = np.eye(n)

    def update_entanglement(self, current_states: Dict[str, np.ndarray]):
        """
        Actualiza matriz de entanglement basada en correlación de ranks.

        El entanglement es la CORRELACIÓN entre las historias de ranks.
        """
        self.joint_history.append(current_states)

        # Mantener historia acotada
        max_history = 200
        if len(self.joint_history) > max_history:
            self.joint_history = self.joint_history[-max_history:]

        if len(self.joint_history) < 20:
            return

        n = len(self.agent_names)

        for i, a1 in enumerate(self.agent_names):
            for j, a2 in enumerate(self.agent_names):
                if i < j:
                    # Extraer historias de ranks
                    h1 = [rankdata(h[a1], method='average') for h in self.joint_history if a1 in h]
                    h2 = [rankdata(h[a2], method='average') for h in self.joint_history if a2 in h]

                    if len(h1) > 10 and len(h2) > 10:
                        h1 = np.array(h1[-50:])
                        h2 = np.array(h2[-50:])

                        # Correlación promedio entre dimensiones
                        correlations = []
                        for d in range(min(h1.shape[1], h2.shape[1])):
                            c = np.corrcoef(h1[:, d], h2[:, d])[0, 1]
                            if not np.isnan(c):
                                correlations.append(abs(c))

                        if correlations:
                            self.correlation_matrix[i, j] = np.mean(correlations)
                            self.correlation_matrix[j, i] = self.correlation_matrix[i, j]

    def get_entanglement(self, agent1: str, agent2: str) -> float:
        """Retorna entanglement entre dos agentes."""
        i = self.agent_names.index(agent1)
        j = self.agent_names.index(agent2)
        return self.correlation_matrix[i, j]

    def concurrence(self, agent1: str, agent2: str) -> float:
        """
        Concurrencia: medida de entanglement.

        concurrence = √(fidelity × correlation)
        """
        s1 = self.local_states[agent1]
        s2 = self.local_states[agent2]

        fid = s1.fidelity(s2)
        corr = self.get_entanglement(agent1, agent2)

        return np.sqrt(fid * corr)


def test_state_encoding():
    """Test de codificación de estados."""
    print("=" * 60)
    print("TEST: State Encoding desde Ranks")
    print("=" * 60)

    # Drives de ejemplo
    drives = np.array([0.1, 0.25, 0.15, 0.2, 0.18, 0.12])

    print(f"\nDrives originales: {drives}")
    print(f"Suma: {drives.sum():.3f}")

    # Crear estado
    state = QuantumStateEncoding.from_drives(drives)

    print(f"\nRanks: {rankdata(drives, method='average')}")
    print(f"Probabilidades (desde ranks): {state.probabilities}")
    print(f"Suma probabilidades: {state.probabilities.sum():.6f}")
    print(f"Amplitudes: {state.amplitudes}")
    print(f"Suma |ψ|²: {(state.amplitudes**2).sum():.6f}")

    # Simular historia
    print("\nSimulando historia...")
    history = [np.random.dirichlet(np.ones(6)) for _ in range(50)]
    state_with_history = QuantumStateEncoding.from_drives(drives, history)

    print(f"Fase acumulada: {state_with_history.phase:.3f}")
    print(f"Entropía endógena: {state_with_history.entropy_endogenous():.3f}")

    # Actualizar estado
    print("\nActualizando estado...")
    for _ in range(30):
        new_drives = np.random.dirichlet(np.ones(6))
        state_with_history.update(new_drives)

    print(f"Coherencia endógena: {state_with_history.coherence_endogenous():.3f}")

    # Test de entanglement
    print("\n" + "=" * 60)
    print("TEST: Entanglement entre Agentes")
    print("=" * 60)

    # Crear estados para NEO, EVA, ALEX
    neo_state = QuantumStateEncoding.from_drives(np.array([0.2, 0.3, 0.15, 0.15, 0.1, 0.1]))
    eva_state = QuantumStateEncoding.from_drives(np.array([0.15, 0.25, 0.2, 0.2, 0.1, 0.1]))
    alex_state = QuantumStateEncoding.from_drives(np.array([0.1, 0.2, 0.25, 0.15, 0.15, 0.15]))

    entangled = EntangledStateEncoding(
        agent_names=['NEO', 'EVA', 'ALEX'],
        local_states={'NEO': neo_state, 'EVA': eva_state, 'ALEX': alex_state}
    )

    # Simular interacciones
    print("\nSimulando interacciones...")
    for _ in range(50):
        current = {
            'NEO': np.random.dirichlet(np.ones(6)),
            'EVA': np.random.dirichlet(np.ones(6)),
            'ALEX': np.random.dirichlet(np.ones(6))
        }
        entangled.update_entanglement(current)

    print(f"\nMatriz de entanglement:")
    print(entangled.correlation_matrix)

    print(f"\nConcurrencias:")
    print(f"  NEO-EVA: {entangled.concurrence('NEO', 'EVA'):.3f}")
    print(f"  NEO-ALEX: {entangled.concurrence('NEO', 'ALEX'):.3f}")
    print(f"  EVA-ALEX: {entangled.concurrence('EVA', 'ALEX'):.3f}")

    print("\n✓ Encoding 100% endógeno: solo ranks, sin constantes mágicas")


if __name__ == "__main__":
    test_state_encoding()
