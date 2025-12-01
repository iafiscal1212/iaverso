#!/usr/bin/env python3
"""
Operadores Cuánticos Endógenos
==============================

Los operadores NO son matrices predefinidas.
EMERGEN de la dinámica del sistema.

Operadores endógenos:
1. Rotación: deriva de gradientes de drives
2. Entangling: emerge de resonancia entre agentes
3. Medición: crisis = colapso natural
4. Decoherencia: pérdida de coherencia por entorno

Principio: U(t) = exp(-i·H·t) donde H emerge del sistema
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from .quantum_state import QuantumState, EntangledState


@dataclass
class EndogenousOperator:
    """
    Operador cuántico que emerge de la dinámica.

    No es una matriz fija - se recalcula según el estado del sistema.
    """

    name: str
    generator_fn: Callable  # Función que genera la matriz del operador

    # Historia para tracking
    application_history: List[Tuple[int, np.ndarray]] = None

    def __post_init__(self):
        if self.application_history is None:
            self.application_history = []

    def generate(self, state: QuantumState, context: Dict = None) -> np.ndarray:
        """
        Genera la matriz del operador dado el estado actual.

        El operador DEPENDE del estado - es endógeno.
        """
        return self.generator_fn(state, context or {})

    def apply(self, state: QuantumState, context: Dict = None) -> QuantumState:
        """Aplica el operador al estado."""
        matrix = self.generate(state, context)

        # Aplicar: |ψ'⟩ = U|ψ⟩
        new_amplitudes = matrix @ state.amplitudes

        # Normalizar
        norm = np.linalg.norm(new_amplitudes)
        if norm > 0:
            new_amplitudes = new_amplitudes / norm

        # Fase evoluciona
        # Fase acumulada = Tr(U†·dU) ≈ cambio angular
        phase_change = np.angle(np.dot(state.amplitudes.conj(), new_amplitudes))

        return QuantumState(
            amplitudes=np.abs(new_amplitudes),  # Amplitudes reales positivas
            phase=state.phase + phase_change,
            coherence=state.coherence * 0.99,  # Decoherencia natural
            amplitude_history=state.amplitude_history + [new_amplitudes]
        )


class EndogenousHamiltonian:
    """
    Hamiltoniano emergente del sistema.

    H = Σ hᵢⱼ |i⟩⟨j| donde hᵢⱼ emerge de:
    - Gradientes de energía libre
    - Interacciones entre drives
    - Potencial de otros agentes
    """

    def __init__(self, dim: int = 6):
        self.dim = dim

    def compute(self, state: QuantumState,
                other_state: Optional[QuantumState] = None,
                stimulus: np.ndarray = None) -> np.ndarray:
        """
        Calcula el Hamiltoniano emergente.

        H = H_local + H_interaction + H_external
        """
        H = np.zeros((self.dim, self.dim))

        # H_local: energía de cada drive (diagonal)
        # Energía ∝ -log(p) (menor prob = mayor energía)
        probs = state.probabilities
        for i in range(self.dim):
            H[i, i] = -np.log(probs[i] + 1e-10)

        # H_interaction: términos no diagonales de interacción entre drives
        # Emerge de correlaciones en la historia
        if state.amplitude_history and len(state.amplitude_history) > 5:
            history = np.array(state.amplitude_history[-20:])
            if history.shape[0] > 1:
                # Covarianza entre dimensiones
                cov = np.cov(history.T)
                # Términos de hopping ∝ covarianza
                for i in range(self.dim):
                    for j in range(i+1, self.dim):
                        if i < cov.shape[0] and j < cov.shape[1]:
                            H[i, j] = -cov[i, j]  # Signo negativo: favorece correlación
                            H[j, i] = H[i, j]

        # H_external: potencial del otro agente (si existe)
        if other_state is not None:
            other_probs = other_state.probabilities
            # Atracción hacia estado del otro
            for i in range(self.dim):
                H[i, i] -= 0.1 * other_probs[i]  # Reduce energía si coinciden

        # H_stimulus: campo externo
        if stimulus is not None:
            stimulus = stimulus / stimulus.sum()
            for i in range(self.dim):
                H[i, i] -= 0.05 * stimulus[i]

        return H

    def evolution_operator(self, state: QuantumState,
                          dt: float = 0.1,
                          other_state: QuantumState = None,
                          stimulus: np.ndarray = None) -> np.ndarray:
        """
        Operador de evolución U = exp(-i·H·dt)

        Calculado endógenamente.
        """
        H = self.compute(state, other_state, stimulus)

        # U = exp(-i·H·dt) ≈ I - i·H·dt (primer orden)
        # Para mejor precisión usamos exponencial de matriz
        # Pero H es real, así que exp(-i·H·dt) tiene partes real e imaginaria

        # Simplificación: tomamos la parte real de la evolución
        # que corresponde a redistribución de probabilidades
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        # U = V · exp(-i·λ·dt) · V†
        exp_eigenvalues = np.exp(-1j * eigenvalues * dt)
        U = eigenvectors @ np.diag(exp_eigenvalues) @ eigenvectors.T.conj()

        # Tomar magnitud (transición de probabilidades)
        return np.abs(U)


class EntanglementOperator:
    """
    Operador de entanglement emergente.

    El entanglement NO se impone - emerge de:
    - Resonancia (similaridad de estados)
    - Interacción prolongada
    - Sincronización de fases
    """

    def __init__(self):
        self.entanglement_history = []

    def compute_entangling_strength(self, state1: QuantumState,
                                    state2: QuantumState) -> float:
        """
        Calcula la fuerza de entanglement emergente.

        Basada en:
        1. Fidelidad (overlap de estados)
        2. Coherencia conjunta
        3. Sincronización de fases
        """
        # Fidelidad
        fidelity = state1.fidelity(state2)

        # Coherencia conjunta
        joint_coherence = np.sqrt(state1.coherence * state2.coherence)

        # Sincronización de fase
        phase_sync = np.cos(state1.phase - state2.phase)
        phase_factor = (1 + phase_sync) / 2  # Normalizado a [0,1]

        # Fuerza de entanglement
        strength = fidelity * joint_coherence * phase_factor

        return strength

    def create_bell_state(self, state1: QuantumState,
                         state2: QuantumState) -> EntangledState:
        """
        Crea un estado tipo Bell emergente.

        No es |00⟩ + |11⟩ predefinido, sino la superposición
        que maximiza el entanglement dado los estados actuales.
        """
        strength = self.compute_entangling_strength(state1, state2)

        # Correlación emergente basada en fuerza
        correlation_matrix = np.array([
            [1.0, strength],
            [strength, 1.0]
        ])

        return EntangledState(
            agents=['A', 'B'],
            local_states={'A': state1, 'B': state2},
            correlation_matrix=correlation_matrix
        )


class MeasurementOperator:
    """
    Operador de medición = colapso endógeno.

    La medición NO es proyección instantánea externa.
    Es una CRISIS que emerge cuando la coherencia cae.
    """

    def __init__(self, collapse_threshold: float = None):
        # Umbral endógeno: percentil de coherencia histórica
        self.collapse_threshold = collapse_threshold
        self.coherence_history = []

    def should_collapse(self, state: QuantumState) -> bool:
        """
        Determina si el estado debe colapsar (crisis).

        Criterio endógeno: coherencia < percentil histórico
        """
        self.coherence_history.append(state.coherence)

        if len(self.coherence_history) < 20:
            return False

        # Umbral endógeno: percentil 10 de la historia
        threshold = np.percentile(self.coherence_history, 10)

        return state.coherence < threshold

    def collapse(self, state: QuantumState,
                 basis: np.ndarray = None) -> Tuple[int, QuantumState]:
        """
        Realiza el colapso (medición).

        Si no se especifica base, usa la base natural (drives).
        """
        if basis is None:
            # Base natural: cada drive es un estado de base
            basis = np.eye(len(state.amplitudes))

        return state.measure_in_basis(basis)

    def soft_measurement(self, state: QuantumState,
                        strength: float = 0.1) -> QuantumState:
        """
        Medición débil: colapso parcial.

        Mezcla entre estado original y estado colapsado.
        """
        # Determinar hacia qué estado "tiende" a colapsar
        probs = state.probabilities
        dominant_idx = np.argmax(probs)

        # Estado colapsado
        collapsed_amp = np.zeros_like(state.amplitudes)
        collapsed_amp[dominant_idx] = 1.0

        # Mezcla
        new_amplitudes = (1 - strength) * state.amplitudes + strength * collapsed_amp
        new_amplitudes = new_amplitudes / np.linalg.norm(new_amplitudes)

        return QuantumState(
            amplitudes=new_amplitudes,
            phase=state.phase,
            coherence=state.coherence * (1 - strength)  # Pierde coherencia
        )


class DecoherenceOperator:
    """
    Decoherencia: pérdida de coherencia por interacción con entorno.

    Emerge de:
    - Varianza del estímulo externo
    - Ruido en las interacciones
    - Tiempo de vida finito de superposiciones
    """

    def __init__(self):
        self.noise_history = []

    def apply(self, state: QuantumState,
              environment_noise: float = None) -> QuantumState:
        """
        Aplica decoherencia al estado.

        El ruido es endógeno si no se especifica.
        """
        if environment_noise is None:
            # Ruido endógeno: varianza de amplitudes históricas
            if state.amplitude_history and len(state.amplitude_history) > 5:
                recent = np.array(state.amplitude_history[-10:])
                environment_noise = np.mean(np.var(recent, axis=0))
            else:
                environment_noise = 0.01

        self.noise_history.append(environment_noise)

        # Decoherencia: mezcla con estado máximamente mezclado
        dim = len(state.amplitudes)
        maximally_mixed = np.ones(dim) / np.sqrt(dim)

        # Factor de decoherencia
        decoherence_factor = 1 - np.exp(-environment_noise)

        new_amplitudes = (1 - decoherence_factor) * state.amplitudes + \
                         decoherence_factor * maximally_mixed
        new_amplitudes = new_amplitudes / np.linalg.norm(new_amplitudes)

        new_coherence = state.coherence * (1 - decoherence_factor)

        return QuantumState(
            amplitudes=new_amplitudes,
            phase=state.phase,
            coherence=new_coherence,
            amplitude_history=state.amplitude_history
        )


def test_operators():
    """Test de operadores endógenos."""
    print("=" * 50)
    print("TEST: Operadores Cuánticos Endógenos")
    print("=" * 50)

    # Estado inicial
    drives = np.array([0.1, 0.2, 0.3, 0.15, 0.15, 0.1])
    history = [np.random.dirichlet(np.ones(6)) for _ in range(20)]
    state = QuantumState.from_drives(drives, history)

    print(f"\nEstado inicial:")
    print(f"  Amplitudes: {state.amplitudes}")
    print(f"  Coherencia: {state.coherence:.3f}")

    # Hamiltoniano emergente
    H = EndogenousHamiltonian(dim=6)
    H_matrix = H.compute(state)
    print(f"\nHamiltoniano (diagonal):")
    print(f"  {np.diag(H_matrix)}")

    # Evolución
    U = H.evolution_operator(state, dt=0.1)
    print(f"\nOperador de evolución aplicado...")

    evolved_amplitudes = U @ state.amplitudes
    evolved_amplitudes = evolved_amplitudes / np.linalg.norm(evolved_amplitudes)
    print(f"  Nuevas amplitudes: {evolved_amplitudes}")

    # Entanglement
    print("\n" + "-" * 30)
    print("Test de Entanglement")

    state2 = QuantumState.from_drives(np.array([0.3, 0.1, 0.2, 0.2, 0.1, 0.1]))

    ent_op = EntanglementOperator()
    strength = ent_op.compute_entangling_strength(state, state2)
    print(f"  Fuerza de entanglement: {strength:.3f}")

    # Medición
    print("\n" + "-" * 30)
    print("Test de Medición")

    meas_op = MeasurementOperator()

    # Simular varios pasos para tener historia
    for _ in range(25):
        meas_op.coherence_history.append(np.random.uniform(0.5, 1.0))

    state_low_coherence = QuantumState(
        amplitudes=state.amplitudes,
        phase=state.phase,
        coherence=0.1  # Baja coherencia
    )

    should = meas_op.should_collapse(state_low_coherence)
    print(f"  ¿Debe colapsar?: {should}")

    if should:
        result, collapsed = meas_op.collapse(state_low_coherence)
        print(f"  Resultado de medición: {result}")
        print(f"  Estado colapsado: {collapsed.amplitudes}")

    # Decoherencia
    print("\n" + "-" * 30)
    print("Test de Decoherencia")

    decoh_op = DecoherenceOperator()
    decohered = decoh_op.apply(state)
    print(f"  Coherencia original: {state.coherence:.3f}")
    print(f"  Coherencia tras decoherencia: {decohered.coherence:.3f}")


if __name__ == "__main__":
    test_operators()
