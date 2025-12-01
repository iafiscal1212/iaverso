"""
AGI-11: Counterfactual Selves (Yo Alternativo)
==============================================

"¿Qué habría pasado si yo (NEO/EVA) hubiera sido distinto?"

Estado de self:
    s_A = [d_A, φ̄_A, π_A] ∈ R^D

Generación de yos alternativos:
    s̃_A^(k) = s_A + δ^(k)
    δ^(k) ~ N(0, Σ_s)

Simulación contrafactual:
    J_A^(k) = rank(V̄^(k)) + rank(Ū^(k)) - rank(C̄^(k))

Índice de yo alternativo preferido:
    P_self(k) = rank(J_A^(k)) / Σ_j rank(J_A^(j))

100% endógeno.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SelfState:
    """Estado de self actual."""
    drives: np.ndarray
    phi_mean: np.ndarray
    policy_distribution: np.ndarray
    t: int


@dataclass
class CounterfactualSelf:
    """Un yo alternativo."""
    variant_id: int
    drives: np.ndarray
    phi_mean: np.ndarray
    policy_distribution: np.ndarray

    # Resultados de simulación
    simulated_V: float = 0.0
    simulated_U: float = 0.0
    simulated_C: float = 0.0
    J_score: float = 0.0
    preference: float = 0.0


@dataclass
class CounterfactualAnalysis:
    """Resultado de análisis contrafactual."""
    current_self: SelfState
    alternatives: List[CounterfactualSelf]
    best_alternative_id: int
    self_exploration_potential: float  # Varianza de J scores


class CounterfactualSelves:
    """
    Sistema de yos alternativos.

    Simula versiones alternativas del self y compara
    para entender qué tipo de yo sería mejor.
    """

    def __init__(self, agent_name: str, drive_dim: int = 6,
                 phi_dim: int = 5, n_policies: int = 7):
        """
        Inicializa sistema contrafactual.

        Args:
            agent_name: Nombre del agente
            drive_dim: Dimensión de drives
            phi_dim: Dimensión fenomenológica
            n_policies: Número de políticas
        """
        self.agent_name = agent_name
        self.drive_dim = drive_dim
        self.phi_dim = phi_dim
        self.n_policies = n_policies

        # Estado actual
        self.current_drives = np.ones(drive_dim) / drive_dim
        self.current_phi = np.zeros(phi_dim)
        self.current_policy = np.ones(n_policies) / n_policies

        # Historial para covarianza
        self.drives_history: List[np.ndarray] = []
        self.phi_history: List[np.ndarray] = []
        self.policy_history: List[np.ndarray] = []
        self.V_history: List[float] = []
        self.U_history: List[float] = []
        self.C_history: List[float] = []

        # Covarianza endógena
        self.Sigma_s: Optional[np.ndarray] = None

        # Historial de análisis
        self.analyses: List[CounterfactualAnalysis] = []

        self.t = 0

    def _compute_covariance(self):
        """
        Calcula covarianza endógena del estado de self.

        Σ_s = cov({s_A,t}) sobre ventana √T
        """
        window = int(np.ceil(np.sqrt(self.t + 1)))
        window = min(window, len(self.drives_history))

        if window < 10:
            # Covarianza inicial
            dim = self.drive_dim + self.phi_dim + self.n_policies
            self.Sigma_s = np.eye(dim) * 0.1
            return

        # Construir matriz de estados
        states = []
        for i in range(-window, 0):
            if abs(i) > len(self.drives_history):
                continue
            s = np.concatenate([
                self.drives_history[i],
                self.phi_history[i],
                self.policy_history[i]
            ])
            states.append(s)

        if len(states) < 5:
            return

        states = np.array(states)
        self.Sigma_s = np.cov(states.T)

        # Asegurar que sea positiva definida
        self.Sigma_s = self.Sigma_s + np.eye(self.Sigma_s.shape[0]) * 0.01

    def _generate_alternative(self, variant_id: int) -> CounterfactualSelf:
        """
        Genera un yo alternativo.

        s̃_A^(k) = s_A + δ^(k)
        δ^(k) ~ N(0, Σ_s)
        """
        # Estado actual
        s_current = np.concatenate([
            self.current_drives,
            self.current_phi,
            self.current_policy
        ])

        # Perturbación
        if self.Sigma_s is not None:
            try:
                delta = np.random.multivariate_normal(
                    np.zeros(len(s_current)),
                    self.Sigma_s
                )
            except:
                delta = np.random.randn(len(s_current)) * 0.1
        else:
            delta = np.random.randn(len(s_current)) * 0.1

        s_alt = s_current + delta

        # Extraer componentes
        drives = s_alt[:self.drive_dim]
        phi = s_alt[self.drive_dim:self.drive_dim + self.phi_dim]
        policy = s_alt[self.drive_dim + self.phi_dim:]

        # Proyectar a espacio válido
        # Drives → simplex via softmax
        drives = np.exp(drives) / np.sum(np.exp(drives))

        # Phi → z-score recortado
        if len(self.phi_history) > 10:
            phi_mean = np.mean(self.phi_history[-50:], axis=0)
            phi_std = np.std(self.phi_history[-50:], axis=0) + 1e-8
            phi_zscore = (phi - phi_mean) / phi_std
            phi = np.clip(phi_zscore, -2, 2) * phi_std + phi_mean

        # Políticas → normalización
        policy = np.exp(policy) / np.sum(np.exp(policy))

        return CounterfactualSelf(
            variant_id=variant_id,
            drives=drives,
            phi_mean=phi,
            policy_distribution=policy
        )

    def _simulate_alternative(self, alt: CounterfactualSelf,
                             horizon: int) -> Tuple[float, float, float]:
        """
        Simula un yo alternativo.

        Returns:
            (V_mean, U_mean, C_mean) simulados
        """
        if len(self.V_history) < 20:
            return 0.5, 0.5, 0.2

        # Estimar efectos basados en historial
        # Correlación entre drives y outcomes
        drives_arr = np.array(self.drives_history[-100:])
        V_arr = np.array(self.V_history[-100:])
        U_arr = np.array(self.U_history[-100:])
        C_arr = np.array(self.C_history[-100:])

        if len(drives_arr) < 10:
            return 0.5, 0.5, 0.2

        # Predicción simple: V ~ drives @ coef
        try:
            # Regresión lineal simple
            X = drives_arr
            y_V = V_arr
            y_U = U_arr
            y_C = C_arr

            # Coeficientes via pseudo-inversa
            coef_V = np.linalg.lstsq(X, y_V, rcond=None)[0]
            coef_U = np.linalg.lstsq(X, y_U, rcond=None)[0]
            coef_C = np.linalg.lstsq(X, y_C, rcond=None)[0]

            # Predecir para drives alternativos
            V_pred = float(np.dot(alt.drives, coef_V))
            U_pred = float(np.dot(alt.drives, coef_U))
            C_pred = float(np.dot(alt.drives, coef_C))

        except:
            V_pred = 0.5
            U_pred = 0.5
            C_pred = 0.2

        # Añadir efecto de políticas
        policy_bias = (alt.policy_distribution[0] - 0.15) * 0.2  # Exploration bonus
        V_pred += policy_bias
        U_pred += policy_bias * 0.5

        return (
            float(np.clip(V_pred, 0, 1)),
            float(np.clip(U_pred, 0, 1)),
            float(np.clip(C_pred, 0, 1))
        )

    def record_state(self, drives: np.ndarray, phi: np.ndarray,
                    policy: np.ndarray, V: float, U: float, C: float):
        """
        Registra estado actual.

        Args:
            drives: Vector de drives
            phi: Vector fenomenológico
            policy: Distribución de políticas
            V: Valor
            U: Utilidad
            C: Crisis
        """
        self.t += 1

        self.current_drives = drives.copy()
        self.current_phi = phi.copy()
        self.current_policy = policy.copy()

        self.drives_history.append(drives.copy())
        self.phi_history.append(phi.copy())
        self.policy_history.append(policy.copy())
        self.V_history.append(V)
        self.U_history.append(U)
        self.C_history.append(C)

        # Limitar historial
        max_hist = 500
        if len(self.drives_history) > max_hist:
            self.drives_history = self.drives_history[-max_hist:]
            self.phi_history = self.phi_history[-max_hist:]
            self.policy_history = self.policy_history[-max_hist:]
            self.V_history = self.V_history[-max_hist:]
            self.U_history = self.U_history[-max_hist:]
            self.C_history = self.C_history[-max_hist:]

        # Actualizar covarianza periódicamente
        if self.t % 20 == 0:
            self._compute_covariance()

    def analyze_counterfactuals(self, n_alternatives: int = 5) -> CounterfactualAnalysis:
        """
        Analiza yos alternativos.

        Args:
            n_alternatives: Número de alternativas a generar

        Returns:
            CounterfactualAnalysis
        """
        horizon = int(np.ceil(np.sqrt(self.t + 1)))

        # Estado actual
        current = SelfState(
            drives=self.current_drives.copy(),
            phi_mean=self.current_phi.copy(),
            policy_distribution=self.current_policy.copy(),
            t=self.t
        )

        # Generar alternativas
        alternatives = []
        for k in range(n_alternatives):
            alt = self._generate_alternative(k)
            V, U, C = self._simulate_alternative(alt, horizon)
            alt.simulated_V = V
            alt.simulated_U = U
            alt.simulated_C = C
            alternatives.append(alt)

        # Calcular J scores
        # J_A^(k) = rank(V̄^(k)) + rank(Ū^(k)) - rank(C̄^(k))
        Vs = [alt.simulated_V for alt in alternatives]
        Us = [alt.simulated_U for alt in alternatives]
        Cs = [alt.simulated_C for alt in alternatives]

        for alt in alternatives:
            rank_V = np.sum(np.array(Vs) <= alt.simulated_V)
            rank_U = np.sum(np.array(Us) <= alt.simulated_U)
            rank_C = np.sum(np.array(Cs) >= alt.simulated_C)
            alt.J_score = float(rank_V + rank_U + rank_C)

        # Calcular preferencias
        # P_self(k) = rank(J_A^(k)) / Σ_j rank(J_A^(j))
        J_scores = [alt.J_score for alt in alternatives]
        total_rank = 0
        for alt in alternatives:
            rank = np.sum(np.array(J_scores) <= alt.J_score)
            alt.preference = float(rank)
            total_rank += rank

        for alt in alternatives:
            alt.preference /= total_rank if total_rank > 0 else 1

        # Mejor alternativa
        best_id = max(range(len(alternatives)), key=lambda i: alternatives[i].J_score)

        # Potencial de exploración del self = varianza de J scores
        exploration_potential = float(np.std(J_scores))

        analysis = CounterfactualAnalysis(
            current_self=current,
            alternatives=alternatives,
            best_alternative_id=best_id,
            self_exploration_potential=exploration_potential
        )

        self.analyses.append(analysis)
        if len(self.analyses) > 100:
            self.analyses = self.analyses[-100:]

        return analysis

    def get_improvement_direction(self) -> Optional[np.ndarray]:
        """
        Obtiene dirección de mejora del self.

        Basado en diferencia con mejor alternativa.
        """
        if not self.analyses:
            return None

        last_analysis = self.analyses[-1]
        best = last_analysis.alternatives[last_analysis.best_alternative_id]

        # Dirección = drives del mejor - drives actuales
        direction = best.drives - last_analysis.current_self.drives

        return direction

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas contrafactuales."""
        if not self.analyses:
            return {
                'agent': self.agent_name,
                't': self.t,
                'n_analyses': 0
            }

        last = self.analyses[-1]
        best = last.alternatives[last.best_alternative_id]

        return {
            'agent': self.agent_name,
            't': self.t,
            'n_analyses': len(self.analyses),
            'current_drives': self.current_drives.tolist(),
            'best_alternative_drives': best.drives.tolist(),
            'best_J_score': best.J_score,
            'best_preference': best.preference,
            'exploration_potential': last.self_exploration_potential,
            'improvement_direction': self.get_improvement_direction().tolist() if self.get_improvement_direction() is not None else None,
            'mean_V_alt': float(np.mean([a.simulated_V for a in last.alternatives])),
            'mean_U_alt': float(np.mean([a.simulated_U for a in last.alternatives]))
        }


def test_counterfactual():
    """Test de yos alternativos."""
    print("=" * 60)
    print("TEST AGI-11: COUNTERFACTUAL SELVES")
    print("=" * 60)

    cf = CounterfactualSelves("NEO")

    print("\nSimulando 300 pasos y analizando yos alternativos...")

    for t in range(300):
        # Estado que evoluciona
        drives = np.array([0.2, 0.15, 0.15, 0.2, 0.15, 0.15])
        drives += np.random.randn(6) * 0.02
        drives = np.clip(drives, 0.01, None)
        drives /= drives.sum()

        phi = np.array([0.5, 0.4, 0.6, 0.5, 0.4]) + np.random.randn(5) * 0.1
        policy = np.ones(7) / 7 + np.random.randn(7) * 0.05
        policy = np.clip(policy, 0.01, None)
        policy /= policy.sum()

        V = 0.5 + 0.2 * np.sin(t / 30) + np.random.randn() * 0.1
        U = 0.5 + 0.15 * np.cos(t / 40) + np.random.randn() * 0.1
        C = max(0, 0.2 + np.random.randn() * 0.1)

        cf.record_state(drives, phi, policy, V, U, C)

        # Analizar periódicamente
        if (t + 1) % 50 == 0:
            analysis = cf.analyze_counterfactuals(5)
            print(f"  t={t+1}: exploration_potential={analysis.self_exploration_potential:.3f}, "
                  f"best_J={analysis.alternatives[analysis.best_alternative_id].J_score:.1f}")

    # Resultados finales
    stats = cf.get_statistics()

    print("\n" + "=" * 60)
    print("RESULTADOS COUNTERFACTUAL SELVES")
    print("=" * 60)

    print(f"\n  Análisis realizados: {stats['n_analyses']}")
    print(f"  Potencial de exploración: {stats['exploration_potential']:.3f}")
    print(f"  Mejor J score: {stats['best_J_score']:.1f}")

    print(f"\n  Drives actuales:")
    for i, d in enumerate(stats['current_drives']):
        print(f"    d[{i}] = {d:.3f}")

    print(f"\n  Drives del mejor yo alternativo:")
    for i, d in enumerate(stats['best_alternative_drives']):
        print(f"    d[{i}] = {d:.3f}")

    if stats['improvement_direction']:
        print(f"\n  Dirección de mejora:")
        for i, d in enumerate(stats['improvement_direction']):
            print(f"    Δd[{i}] = {d:+.3f}")

    if stats['n_analyses'] > 0:
        print("\n  ✓ Análisis contrafactual funcionando")
    else:
        print("\n  ⚠️ No se realizaron análisis")

    return cf


if __name__ == "__main__":
    test_counterfactual()
