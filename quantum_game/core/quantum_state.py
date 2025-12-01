#!/usr/bin/env python3
"""
Estados Cuánticos Endógenos
===========================

Los estados cuánticos NO se definen externamente.
EMERGEN de la dinámica interna de los agentes.

Principios:
1. |ψ⟩ = normalización del vector de drives (ya es distribución)
2. Fase = deriva de la historia de cambios
3. Entanglement = correlación temporal emergente
4. Colapso = crisis (transición de fase endógena)

NO HAY:
- Constantes de Planck hardcodeadas
- Bases predefinidas
- Operadores externos

TODO EMERGE de:
- Geometría del espacio de drives
- Historia de interacciones
- Resonancia entre agentes
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class QuantumState:
    """
    Estado cuántico endógeno de un agente.

    El estado NO es impuesto - emerge de:
    - Amplitudes: pesos de drives normalizados
    - Fase: acumulada de cambios históricos
    - Coherencia: estabilidad temporal del estado
    """

    amplitudes: np.ndarray  # |α_i|² = probabilidades (drives)
    phase: float = 0.0  # Fase global emergente
    coherence: float = 1.0  # Grado de superposición vs mezcla

    # Historia para calcular propiedades emergentes
    amplitude_history: List[np.ndarray] = field(default_factory=list)
    phase_history: List[float] = field(default_factory=list)

    @classmethod
    def from_drives(cls, drives: np.ndarray, history: List[np.ndarray] = None) -> 'QuantumState':
        """
        Construye estado cuántico desde drives.

        Amplitudes = √(drives) (para que |α|² = drives)
        Fase = acumulada de cambios
        Coherencia = inverso de varianza histórica
        """
        # Asegurar normalización
        drives = np.clip(drives, 1e-10, None)
        drives = drives / drives.sum()

        # Amplitudes: √p para que |α|² = p
        amplitudes = np.sqrt(drives)

        # Fase emergente: acumulada de "rotaciones" en el espacio de drives
        phase = 0.0
        if history and len(history) > 1:
            # Fase = suma de ángulos entre estados consecutivos
            for i in range(1, len(history)):
                prev = np.sqrt(np.clip(history[i-1], 1e-10, None))
                curr = np.sqrt(np.clip(history[i], 1e-10, None))
                prev = prev / np.linalg.norm(prev)
                curr = curr / np.linalg.norm(curr)
                # Ángulo entre vectores
                dot = np.clip(np.dot(prev, curr), -1, 1)
                phase += np.arccos(dot)

        # Coherencia: qué tan estable ha sido el estado
        coherence = 1.0
        if history and len(history) > 10:
            recent = np.array(history[-10:])
            variance = np.mean(np.var(recent, axis=0))
            # Coherencia alta = baja varianza
            coherence = 1.0 / (1.0 + 10 * variance)

        state = cls(
            amplitudes=amplitudes,
            phase=phase,
            coherence=coherence
        )

        if history:
            state.amplitude_history = [np.sqrt(np.clip(h, 1e-10, None)) for h in history[-50:]]

        return state

    @property
    def probabilities(self) -> np.ndarray:
        """Probabilidades de medición = |α|²"""
        return self.amplitudes ** 2

    @property
    def entropy(self) -> float:
        """Entropía de von Neumann (endógena)."""
        p = self.probabilities
        p = p[p > 1e-10]  # Evitar log(0)
        return -np.sum(p * np.log(p))

    @property
    def purity(self) -> float:
        """Pureza del estado: Tr(ρ²)."""
        # Para estado puro: pureza = 1
        # Para mezcla máxima: pureza = 1/d
        return np.sum(self.probabilities ** 2)

    def inner_product(self, other: 'QuantumState') -> complex:
        """
        Producto interno ⟨self|other⟩.

        Incluye fase relativa.
        """
        phase_diff = other.phase - self.phase
        overlap = np.dot(self.amplitudes, other.amplitudes)
        return overlap * np.exp(1j * phase_diff)

    def fidelity(self, other: 'QuantumState') -> float:
        """
        Fidelidad |⟨ψ|φ⟩|² entre estados.

        Mide qué tan "parecidos" son cuánticamente.
        """
        return abs(self.inner_product(other)) ** 2

    def measure_in_basis(self, basis: np.ndarray) -> Tuple[int, 'QuantumState']:
        """
        Medición en base arbitraria.

        Args:
            basis: Matriz donde cada fila es un vector de base

        Returns:
            (índice del resultado, estado colapsado)
        """
        # Probabilidades en nueva base
        probs = np.array([abs(np.dot(self.amplitudes, b))**2 for b in basis])
        probs = probs / probs.sum()  # Normalizar

        # "Medir" = colapsar según probabilidades
        result = np.random.choice(len(basis), p=probs)

        # Estado colapsado
        collapsed_amp = basis[result].copy()
        collapsed_amp = collapsed_amp / np.linalg.norm(collapsed_amp)

        collapsed = QuantumState(
            amplitudes=collapsed_amp,
            phase=self.phase,  # Fase se preserva
            coherence=1.0  # Estado puro post-medición
        )

        return result, collapsed


@dataclass
class EntangledState:
    """
    Estado entangled de múltiples agentes.

    El entanglement NO se impone - emerge de:
    - Correlación temporal entre agentes
    - Resonancia de drives
    - Historia compartida
    """

    agents: List[str]
    local_states: Dict[str, QuantumState]

    # Matriz de correlaciones (entanglement emergente)
    correlation_matrix: np.ndarray = None

    # Historia conjunta
    joint_history: List[Dict[str, np.ndarray]] = field(default_factory=list)

    def __post_init__(self):
        n = len(self.agents)
        if self.correlation_matrix is None:
            self.correlation_matrix = np.eye(n)

    @classmethod
    def from_agents(cls, agent_states: Dict[str, QuantumState],
                    interaction_history: List[Dict] = None) -> 'EntangledState':
        """
        Construye estado entangled desde estados individuales + historia.

        El entanglement se calcula de la correlación histórica.
        """
        agents = list(agent_states.keys())
        n = len(agents)

        # Matriz de correlación emergente
        corr_matrix = np.eye(n)

        if interaction_history and len(interaction_history) > 20:
            # Calcular correlación de amplitudes históricas
            for i, a1 in enumerate(agents):
                for j, a2 in enumerate(agents):
                    if i < j:
                        # Extraer historias de amplitudes
                        h1 = [h.get(a1, np.zeros(6)) for h in interaction_history[-50:]]
                        h2 = [h.get(a2, np.zeros(6)) for h in interaction_history[-50:]]

                        if len(h1) > 10 and len(h2) > 10:
                            # Correlación promedio entre dimensiones
                            h1 = np.array(h1)
                            h2 = np.array(h2)

                            corrs = []
                            for d in range(min(h1.shape[1], h2.shape[1])):
                                if np.std(h1[:, d]) > 0 and np.std(h2[:, d]) > 0:
                                    c = np.corrcoef(h1[:, d], h2[:, d])[0, 1]
                                    if not np.isnan(c):
                                        corrs.append(abs(c))

                            if corrs:
                                corr_matrix[i, j] = np.mean(corrs)
                                corr_matrix[j, i] = corr_matrix[i, j]

        return cls(
            agents=agents,
            local_states=agent_states,
            correlation_matrix=corr_matrix,
            joint_history=interaction_history or []
        )

    def entanglement_measure(self, agent1: str, agent2: str) -> float:
        """
        Medida de entanglement entre dos agentes.

        Basada en correlación emergente (no impuesta).
        """
        i = self.agents.index(agent1)
        j = self.agents.index(agent2)
        return self.correlation_matrix[i, j]

    def concurrence(self, agent1: str, agent2: str) -> float:
        """
        Concurrencia (medida de entanglement para 2 qubits).

        Adaptada: usa fidelidad + correlación.
        """
        s1 = self.local_states[agent1]
        s2 = self.local_states[agent2]

        fid = s1.fidelity(s2)
        corr = self.entanglement_measure(agent1, agent2)

        # Concurrencia emergente: combinación de similitud y correlación
        return np.sqrt(fid * corr)

    def partial_trace(self, keep_agents: List[str]) -> 'EntangledState':
        """
        Traza parcial: marginalizamos sobre agentes no deseados.
        """
        keep_indices = [self.agents.index(a) for a in keep_agents]

        new_local = {a: self.local_states[a] for a in keep_agents}
        new_corr = self.correlation_matrix[np.ix_(keep_indices, keep_indices)]

        return EntangledState(
            agents=keep_agents,
            local_states=new_local,
            correlation_matrix=new_corr
        )

    def measure_coalition(self) -> Tuple[str, 'EntangledState']:
        """
        Medir qué coalición está más entangled.

        Retorna la coalición ganadora y el estado post-medición.
        """
        if len(self.agents) < 2:
            return None, self

        # Calcular entanglement de cada par
        pairs = []
        for i, a1 in enumerate(self.agents):
            for j, a2 in enumerate(self.agents):
                if i < j:
                    ent = self.concurrence(a1, a2)
                    pairs.append((f"{a1}-{a2}", ent))

        # Probabilidades proporcionales al entanglement
        total = sum(e for _, e in pairs)
        if total == 0:
            probs = [1/len(pairs)] * len(pairs)
        else:
            probs = [e/total for _, e in pairs]

        # "Medir" = colapsar a una coalición
        idx = np.random.choice(len(pairs), p=probs)
        winning_coalition = pairs[idx][0]

        # Post-medición: aumentar correlación de la coalición ganadora
        agents_in_coalition = winning_coalition.split('-')
        new_corr = self.correlation_matrix.copy()

        i = self.agents.index(agents_in_coalition[0])
        j = self.agents.index(agents_in_coalition[1])
        new_corr[i, j] = min(1.0, new_corr[i, j] + 0.1)
        new_corr[j, i] = new_corr[i, j]

        new_state = EntangledState(
            agents=self.agents,
            local_states=self.local_states,
            correlation_matrix=new_corr
        )

        return winning_coalition, new_state


def test_quantum_states():
    """Test básico de estados cuánticos."""
    print("=" * 50)
    print("TEST: Estados Cuánticos Endógenos")
    print("=" * 50)

    # Crear estado desde drives
    drives = np.array([0.1, 0.2, 0.3, 0.15, 0.15, 0.1])
    history = [np.random.dirichlet(np.ones(6)) for _ in range(20)]
    history.append(drives)

    state = QuantumState.from_drives(drives, history)

    print(f"\nEstado creado:")
    print(f"  Amplitudes: {state.amplitudes}")
    print(f"  Probabilidades: {state.probabilities}")
    print(f"  Fase: {state.phase:.3f}")
    print(f"  Coherencia: {state.coherence:.3f}")
    print(f"  Entropía: {state.entropy:.3f}")
    print(f"  Pureza: {state.purity:.3f}")

    # Segundo estado
    drives2 = np.array([0.3, 0.1, 0.1, 0.2, 0.2, 0.1])
    state2 = QuantumState.from_drives(drives2)

    print(f"\nFidelidad entre estados: {state.fidelity(state2):.3f}")

    # Estado entangled
    print("\n" + "=" * 50)
    print("TEST: Estado Entangled")
    print("=" * 50)

    agent_states = {
        'NEO': state,
        'EVA': state2,
        'ALEX': QuantumState.from_drives(np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1]))
    }

    # Simular historia de interacción
    interaction_history = []
    for _ in range(30):
        interaction_history.append({
            'NEO': np.random.dirichlet(np.ones(6)),
            'EVA': np.random.dirichlet(np.ones(6)),
            'ALEX': np.random.dirichlet(np.ones(6))
        })

    entangled = EntangledState.from_agents(agent_states, interaction_history)

    print(f"\nMatriz de correlación (entanglement):")
    print(entangled.correlation_matrix)

    print(f"\nConcurrencias:")
    print(f"  NEO-EVA: {entangled.concurrence('NEO', 'EVA'):.3f}")
    print(f"  NEO-ALEX: {entangled.concurrence('NEO', 'ALEX'):.3f}")
    print(f"  EVA-ALEX: {entangled.concurrence('EVA', 'ALEX'):.3f}")

    # Medir coalición
    coalition, new_state = entangled.measure_coalition()
    print(f"\nCoalición medida: {coalition}")


if __name__ == "__main__":
    test_quantum_states()
