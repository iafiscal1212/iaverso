"""
ComplexField: Campo de Estado Complejo Endógeno
================================================

Ofrece a cada agente un campo de estado complejo adicional ψ_i(t) ∈ C^d.

Principios:
- NO añade "saberes" externos
- NO fuerza decisiones
- NO impone colapsos
- Solo define dinámicas internas posibles sobre un vector complejo
- Todos los parámetros derivados endógenamente

Cada agente i tiene:
- Estado real habitual: S_i(t) ∈ R^d
- Identidad: I_i(t) ∈ R^d
- NUEVO: Estado complejo opcional ψ_i(t) ∈ C^d

El módulo ofrece:
- Inicialización de ψ desde S con fases endógenas
- Hamiltoniano H_i(t) derivado de covarianza histórica
- Evolución unitaria ψ(t+1/2) = U_i(t) ψ(t)
- Factor de decoherencia λ_i(t) basado en varianza narrativa
- Presión de colapso P_collapse (solo métrica, no acción)
- Proyección real opcional Π(ψ)

NO decide nada. NO colapsa nada.
Solo ofrece transformaciones posibles y métricas.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# Único epsilon "externo" (precisión de máquina)
EPS = np.finfo(float).eps


@dataclass
class ComplexState:
    """
    Estado complejo opcional de un agente.

    Attributes:
        psi: Vector de estado complejo ψ ∈ C^d
        history_real: Historial de estados reales S(t)
        history_ce: Historial de coherencia existencial CE(t)
        history_err: Historial de error interno E(t)
        history_narr_entropy: Historial de entropía narrativa H_narr(t)
    """
    psi: Optional[np.ndarray] = None
    history_real: List[np.ndarray] = field(default_factory=list)
    history_ce: List[float] = field(default_factory=list)
    history_err: List[float] = field(default_factory=list)
    history_narr_entropy: List[float] = field(default_factory=list)


class ComplexField:
    """
    Motor de dinámica compleja endógena.

    No decide nada, no colapsa nada,
    solo ofrece transformaciones posibles y métricas.

    Matemáticas:

    1. Inicialización:
       ψ_i(t) = Ŝ_i(t) ⊙ e^(iθ_i(t))
       donde Ŝ es normalizado y θ viene del ranking histórico

    2. Hamiltoniano endógeno:
       H_i(t) = C̃_i(t) + C̃_i(t)^T + α_i(t)·I_d
       donde C̃ es la covarianza normalizada del historial

    3. Evolución unitaria:
       ψ(t+1/2) = U_i(t)·ψ(t)
       U_i(t) = exp(i·Δt·H_i(t)) ≈ I + iH - 0.5·H²

    4. Decoherencia endógena:
       λ_i(t) = σ²_narr / (1 + σ²_narr)
       ψ(t+1) = ψ(t+1/2)·e^(-λ)

    5. Presión de colapso (solo métrica):
       P_collapse = (1 - Q_CE)·Q_E
       donde Q son posiciones cuantílicas
    """

    def __init__(self, dim: int):
        """
        Inicializa el campo complejo.

        Args:
            dim: Dimensión del espacio de estados
        """
        self.dim = dim

    # ==================== Utilidades internas ====================

    def _window_size(self, T: int) -> int:
        """
        Calcula tamaño de ventana endógeno: K = min(T, floor(√T)).

        Args:
            T: Número total de observaciones

        Returns:
            Tamaño de ventana K
        """
        if T <= 0:
            return 0
        return max(1, int(np.sqrt(T)))

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normaliza vector: x̂ = x / (||x||₂ + ε).

        Args:
            x: Vector a normalizar

        Returns:
            Vector normalizado
        """
        norm = np.linalg.norm(x)
        return x / (norm + EPS)

    def _rank_normalized(self, value: float, history: List[float]) -> float:
        """
        Devuelve la posición cuantílica de 'value' respecto a 'history'.

        Resultado en [0, 1]:
        - 0 si value es menor que todo el historial
        - 1 si value es mayor que todo el historial
        - 0.5 si no hay historial

        Args:
            value: Valor a evaluar
            history: Lista de valores históricos

        Returns:
            Posición cuantílica en [0, 1]
        """
        if not history:
            return 1/2  # Sin historial: centro
        arr = np.asarray(history)
        return float((arr <= value).sum()) / float(len(arr))

    # ==================== Inicialización ====================

    def init_complex_state(
        self,
        real_state: np.ndarray,
        cs: ComplexState,
    ) -> None:
        """
        Inicializa el estado complejo ψ desde el estado real S.

        ψ_i(t) = Ŝ_i(t) ⊙ e^(iθ_i(t))

        donde:
        - Ŝ = S / (||S|| + ε) es la normalización
        - θ_j = 2π · rank_norm(S_j; H_j) son las fases por componente

        Args:
            real_state: Estado real S(t)
            cs: ComplexState del agente (se modifica in-place)
        """
        real_state = np.asarray(real_state, dtype=float)

        # Ajustar dimensión si es necesario
        if len(real_state) != self.dim:
            if len(real_state) > self.dim:
                real_state = real_state[:self.dim]
            else:
                padded = np.zeros(self.dim)
                padded[:len(real_state)] = real_state
                real_state = padded

        # Normalizar estado
        real_state = self._normalize(real_state)

        # Fases endógenas basadas en ranking por dimensión
        if cs.history_real:
            # Construir matriz histórica
            H = np.stack([
                h[:self.dim] if len(h) >= self.dim
                else np.pad(h, (0, self.dim - len(h)))
                for h in cs.history_real
            ], axis=0)
            # Ranking por componente: proporción de valores históricos <= actual
            ranks = np.mean((H <= real_state[None, :]).astype(float), axis=0)
        else:
            # Sin historial: todas las fases en 0.5 (mitad del ciclo)
            ranks = np.full(self.dim, 1/2)

        # θ_j = 2π · rank_j
        phases = 2.0 * np.pi * ranks

        # ψ = Ŝ ⊙ e^(iθ)
        cs.psi = real_state * np.exp(1j * phases)

    # ==================== Hamiltoniano endógeno ====================

    def _build_hamiltonian(self, cs: ComplexState) -> Optional[np.ndarray]:
        """
        Construye Hamiltoniano endógeno H_i(t) desde el historial.

        H_i(t) = C̃_i(t) + C̃_i(t)^T + α_i(t)·I_d

        donde:
        - C_i(t) = Cov[S_i(τ)] es la covarianza de la ventana
        - C̃ = C / (Tr(|C|) + ε) es la normalización
        - α_i(t) = (1/d) Σ C̃_kk es el término de identidad endógeno

        Returns:
            Matriz Hermítica H_i(t), o None si no hay historial
        """
        if len(cs.history_real) == 0:
            return None

        # Construir matriz de historial con dimensión correcta
        H = np.stack([
            h[:self.dim] if len(h) >= self.dim
            else np.pad(h, (0, self.dim - len(h)))
            for h in cs.history_real
        ], axis=0)  # (T, d)

        T = H.shape[0]
        K = self._window_size(T)
        H_win = H[-K:]  # Ventana de tamaño K

        # Covarianza de la ventana
        if H_win.shape[0] < 2:
            # Con un solo punto, covarianza es cero
            C = np.zeros((self.dim, self.dim))
        else:
            C = np.cov(H_win, rowvar=False)

        # Asegurar forma (d, d)
        if C.ndim == 0:
            C = np.array([[float(C)]])
        elif C.ndim == 1:
            C = np.diag(C)

        # Ajustar tamaño si es necesario
        if C.shape[0] != self.dim:
            C_new = np.zeros((self.dim, self.dim))
            min_d = min(C.shape[0], self.dim)
            C_new[:min_d, :min_d] = C[:min_d, :min_d]
            C = C_new

        # Normalización: C̃ = C / (Σ|C_ij| + ε)
        denom = np.sum(np.abs(C)) + EPS
        C_tilde = C / denom

        # Hamiltoniano simétrico: H = C̃ + C̃^T
        H_mat = C_tilde + C_tilde.T

        # Término de identidad endógeno: α = (1/d) Tr(|C̃|)
        alpha = float(np.trace(np.abs(C_tilde))) / (self.dim + EPS)
        H_mat = H_mat + alpha * np.eye(self.dim)

        return H_mat

    # ==================== Paso de evolución ====================

    def step(
        self,
        cs: ComplexState,
        real_state: np.ndarray,
        ce: float,
        internal_error: float,
        narr_entropy: float,
    ) -> Dict[str, float]:
        """
        Actualiza ψ si existe y devuelve métricas.

        NO impone ninguna decisión.

        Pasos:
        1. Actualizar historiales
        2. Inicializar ψ si no existe
        3. Construir Hamiltoniano H_i(t)
        4. Evolución unitaria: ψ(t+1/2) = U·ψ(t)
        5. Decoherencia: ψ(t+1) = ψ(t+1/2)·e^(-λ)
        6. Calcular presión de colapso (solo métrica)

        Args:
            cs: ComplexState del agente
            real_state: Estado real S(t)
            ce: Coherencia existencial CE(t)
            internal_error: Error interno E(t) = Var[S - I]
            narr_entropy: Entropía narrativa H_narr(t)

        Returns:
            Dict con métricas:
            - lambda_decoherence: Factor de decoherencia λ ∈ [0, 1)
            - collapse_pressure: Presión de colapso P ∈ [0, 1]
        """
        # Actualizar historiales
        cs.history_real.append(np.asarray(real_state, dtype=float))
        cs.history_ce.append(float(ce))
        cs.history_err.append(float(internal_error))
        cs.history_narr_entropy.append(float(narr_entropy))

        # Limitar historial endógenamente
        max_hist = max(100, self._window_size(len(cs.history_real)) * 10)
        if len(cs.history_real) > max_hist:
            cs.history_real = cs.history_real[-max_hist:]
            cs.history_ce = cs.history_ce[-max_hist:]
            cs.history_err = cs.history_err[-max_hist:]
            cs.history_narr_entropy = cs.history_narr_entropy[-max_hist:]

        # Inicializar ψ si no existe
        if cs.psi is None:
            self.init_complex_state(real_state, cs)
            return {
                "lambda_decoherence": 0.0,
                "collapse_pressure": 0.0,
            }

        # Construir Hamiltoniano endógeno
        H_mat = self._build_hamiltonian(cs)
        if H_mat is None:
            return {
                "lambda_decoherence": 0.0,
                "collapse_pressure": 0.0,
            }

        # Evolución unitaria aproximada: U ≈ I + iH - 0.5·H²
        # (Aproximación de Taylor de exp(iH) con Δt=1)
        I = np.eye(self.dim, dtype=complex)
        Hc = H_mat.astype(complex)
        U = I + 1j * Hc - (1/2) * (Hc @ Hc)

        # ψ(t+1/2) = U·ψ(t)
        psi_half = U @ cs.psi

        # Decoherencia endógena basada en varianza narrativa
        arr_narr = np.asarray(cs.history_narr_entropy, dtype=float)
        if len(arr_narr) > 1:
            W = self._window_size(len(arr_narr))
            var_narr = float(np.var(arr_narr[-W:]))
        else:
            var_narr = 0.0

        # λ = σ²_narr / (1 + σ²_narr) ∈ [0, 1)
        lambda_dec = var_narr / (1.0 + var_narr)

        # ψ(t+1) = ψ(t+1/2)·e^(-λ)
        cs.psi = psi_half * np.exp(-lambda_dec)

        # Presión de colapso (SOLO métrica, no acción)
        # P = (1 - Q_CE) · Q_E
        # donde Q son posiciones cuantílicas
        q_ce = self._rank_normalized(ce, cs.history_ce)
        q_err = self._rank_normalized(internal_error, cs.history_err)
        collapse_pressure = (1.0 - q_ce) * q_err

        return {
            "lambda_decoherence": float(lambda_dec),
            "collapse_pressure": float(collapse_pressure),
        }

    # ==================== Proyección opcional ====================

    def project_real(self, cs: ComplexState) -> Optional[np.ndarray]:
        """
        Devuelve la proyección real normalizada de ψ.

        Π(ψ) = Re(ψ) / (||Re(ψ)||₂ + ε)

        NO se llama automáticamente desde este módulo.
        El agente decide si y cuándo usarla.

        Args:
            cs: ComplexState del agente

        Returns:
            Vector real normalizado, o None si ψ no existe
        """
        if cs.psi is None:
            return None
        real_part = np.real(cs.psi)
        return self._normalize(real_part)

    # ==================== Superposición de trayectorias ====================

    def superpose_trajectories(
        self,
        candidate_states: List[np.ndarray],
        candidate_errors: List[float],
        cs: ComplexState,
    ) -> Optional[np.ndarray]:
        """
        Crea superposición de estados candidatos con pesos endógenos.

        ψ = Σ_k α_k · ψ^(k)

        donde:
        - ψ^(k) = InitComplex(S^(k))
        - α_k = exp(-E^(k)) / Σ_j exp(-E^(j))  (softmax inverso del error)

        NO se llama automáticamente. El agente decide si usar esto.

        Args:
            candidate_states: Lista de estados candidatos S^(k)
            candidate_errors: Lista de errores internos E^(k)
            cs: ComplexState para inicialización de fases

        Returns:
            Estado complejo superpuesto ψ, o None si no hay candidatos
        """
        if not candidate_states:
            return None

        if len(candidate_states) != len(candidate_errors):
            return None

        # Convertir cada candidato a estado complejo
        psi_candidates = []
        for S_k in candidate_states:
            S_k = np.asarray(S_k, dtype=float)
            if len(S_k) != self.dim:
                if len(S_k) > self.dim:
                    S_k = S_k[:self.dim]
                else:
                    S_k = np.pad(S_k, (0, self.dim - len(S_k)))

            S_norm = self._normalize(S_k)

            # Fases del historial
            if cs.history_real:
                H = np.stack([
                    h[:self.dim] if len(h) >= self.dim
                    else np.pad(h, (0, self.dim - len(h)))
                    for h in cs.history_real
                ], axis=0)
                ranks = np.mean((H <= S_norm[None, :]).astype(float), axis=0)
            else:
                ranks = np.full(self.dim, 1/2)

            phases = 2.0 * np.pi * ranks
            psi_k = S_norm * np.exp(1j * phases)
            psi_candidates.append(psi_k)

        # Pesos softmax inverso del error: α_k = exp(-E_k) / Σ exp(-E_j)
        errors = np.asarray(candidate_errors, dtype=float)
        # Estabilizar softmax restando máximo
        neg_errors = -errors
        neg_errors = neg_errors - np.max(neg_errors)
        exp_neg = np.exp(neg_errors)
        alphas = exp_neg / (np.sum(exp_neg) + EPS)

        # Superposición: ψ = Σ α_k · ψ^(k)
        psi_super = np.zeros(self.dim, dtype=complex)
        for k, psi_k in enumerate(psi_candidates):
            psi_super += alphas[k] * psi_k

        return psi_super

    # ==================== Métricas adicionales ====================

    def get_amplitude_distribution(self, cs: ComplexState) -> Optional[np.ndarray]:
        """
        Retorna distribución de probabilidad de amplitudes |ψ_j|².

        Args:
            cs: ComplexState del agente

        Returns:
            Vector de probabilidades p_j = |ψ_j|² / Σ|ψ_k|², o None
        """
        if cs.psi is None:
            return None

        probs = np.abs(cs.psi) ** 2
        total = np.sum(probs)
        if total < EPS:
            return np.ones(self.dim) / self.dim
        return probs / total

    def get_phase_distribution(self, cs: ComplexState) -> Optional[np.ndarray]:
        """
        Retorna distribución de fases arg(ψ_j) ∈ [0, 2π).

        Args:
            cs: ComplexState del agente

        Returns:
            Vector de fases θ_j, o None
        """
        if cs.psi is None:
            return None

        phases = np.angle(cs.psi)  # En [-π, π]
        phases = np.mod(phases, 2 * np.pi)  # Llevar a [0, 2π)
        return phases

    def get_coherence(self, cs: ComplexState) -> float:
        """
        Calcula coherencia del estado complejo.

        C = |Σ ψ_j|² / (d · Σ|ψ_j|²)

        - C = 1 si todas las fases están alineadas
        - C → 1/d si las fases están uniformemente distribuidas

        Args:
            cs: ComplexState del agente

        Returns:
            Coherencia C ∈ [0, 1]
        """
        if cs.psi is None:
            return 0.0

        sum_psi = np.sum(cs.psi)
        sum_abs_sq = np.sum(np.abs(cs.psi) ** 2)

        if sum_abs_sq < EPS:
            return 0.0

        coherence = (np.abs(sum_psi) ** 2) / (self.dim * sum_abs_sq + EPS)
        return float(np.clip(coherence, 0, 1))

    def get_statistics(self, cs: ComplexState) -> Dict[str, Any]:
        """
        Retorna estadísticas completas del estado complejo.

        Args:
            cs: ComplexState del agente

        Returns:
            Dict con estadísticas
        """
        stats = {
            'has_psi': cs.psi is not None,
            'history_length': len(cs.history_real),
            'dim': self.dim,
        }

        if cs.psi is not None:
            stats['psi_norm'] = float(np.linalg.norm(cs.psi))
            stats['coherence'] = self.get_coherence(cs)
            stats['mean_amplitude'] = float(np.mean(np.abs(cs.psi)))
            stats['std_amplitude'] = float(np.std(np.abs(cs.psi)))

            phases = self.get_phase_distribution(cs)
            if phases is not None:
                stats['mean_phase'] = float(np.mean(phases))
                stats['std_phase'] = float(np.std(phases))

        if cs.history_ce:
            stats['mean_ce'] = float(np.mean(cs.history_ce))
        if cs.history_err:
            stats['mean_err'] = float(np.mean(cs.history_err))
        if cs.history_narr_entropy:
            stats['mean_narr_entropy'] = float(np.mean(cs.history_narr_entropy))

        return stats
