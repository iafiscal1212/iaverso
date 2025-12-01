"""
AGI-12: Norm Emergence & Value Stabilization
=============================================

"Lo que hacemos todos, muchas veces, se vuelve 'lo correcto'."

Co-ocurrencias de políticas:
    p_t = [π_t^NEO, π_t^EVA, ..., π_t^IRIS]
    C_p = Σ_t p_t p_t^T
    C̃_ij = C_ij / √(C_ii * C_jj)

Normas como eigenvectores estables:
    C̃ v_ℓ = λ_ℓ v_ℓ
    Normas = eigenvectores con λ_ℓ ≥ median(λ)

Persistencia:
    pers_ℓ = corr(v_ℓ(t0), v_ℓ(t0+W))

Valor de norma:
    val_ℓ = corr((p_t · v_ℓ), G_t)
    donde G_t = mean_A V_t^A

Peso normativo:
    W_ℓ = rank(pers_ℓ) + rank(val_ℓ)

100% endógeno.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class EmergentNorm:
    """Una norma emergente."""
    norm_id: int
    eigenvector: np.ndarray
    eigenvalue: float
    agent_contributions: Dict[str, float]  # Contribución de cada agente
    persistence: float = 0.0
    value_correlation: float = 0.0
    weight: float = 0.0
    detection_time: int = 0
    is_stable: bool = False


class NormEmergence:
    """
    Sistema de emergencia de normas.

    Detecta patrones de comportamiento compartido que
    mejoran la vida global y se estabilizan.
    """

    def __init__(self, agent_names: List[str], n_policies: int = 7):
        """
        Inicializa sistema de normas.

        Args:
            agent_names: Lista de nombres de agentes
            n_policies: Número de políticas por agente
        """
        self.agent_names = agent_names
        self.n_agents = len(agent_names)
        self.n_policies = n_policies

        # Dimensión total del vector de políticas conjuntas
        self.policy_dim = self.n_agents * n_policies

        # Historial de políticas
        self.policy_history: List[np.ndarray] = []  # p_t concatenado

        # Historial de utilidad global
        self.G_history: List[float] = []

        # Normas detectadas
        self.norms: Dict[int, EmergentNorm] = {}
        self.next_norm_id = 0

        # Eigenvectores anteriores para persistencia
        self.prev_eigenvectors: Optional[np.ndarray] = None
        self.prev_eigenvalues: Optional[np.ndarray] = None

        # Learning rate para influencia normativa
        self.eta_t: float = 0.1

        self.t = 0

    def _compute_cooccurrence(self) -> Optional[np.ndarray]:
        """
        Calcula matriz de co-ocurrencia.

        C_p = Σ_t p_t p_t^T
        C̃_ij = C_ij / √(C_ii * C_jj)
        """
        window = int(np.ceil(np.sqrt(self.t + 1)))
        window = min(window, len(self.policy_history))

        if window < 10:
            return None

        # Tomar últimas políticas
        policies = np.array(self.policy_history[-window:])

        # C = Σ p p^T
        C = policies.T @ policies

        # Normalizar
        diag = np.diag(C)
        diag_sqrt = np.sqrt(diag + 1e-8)
        C_normalized = C / np.outer(diag_sqrt, diag_sqrt)

        return np.nan_to_num(C_normalized, 0)

    def _detect_norms(self, C: np.ndarray) -> List[EmergentNorm]:
        """
        Detecta normas como eigenvectores estables.

        Normas = eigenvectores con λ ≥ median(λ)
        """
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(C)
        except:
            return []

        median_eig = np.median(eigenvalues)
        significant = eigenvalues >= median_eig

        norms = []
        for i, is_sig in enumerate(significant):
            if not is_sig:
                continue

            v = eigenvectors[:, i]
            lam = eigenvalues[i]

            # Calcular contribución de cada agente
            contributions = {}
            for j, name in enumerate(self.agent_names):
                start = j * self.n_policies
                end = start + self.n_policies
                contributions[name] = float(np.mean(np.abs(v[start:end])))

            # Calcular persistencia
            persistence = 0.0
            if self.prev_eigenvectors is not None:
                # Correlación con eigenvectores anteriores
                for k in range(self.prev_eigenvectors.shape[1]):
                    if len(v) == len(self.prev_eigenvectors[:, k]):
                        corr = abs(np.dot(v, self.prev_eigenvectors[:, k]))
                        persistence = max(persistence, corr)

            norm = EmergentNorm(
                norm_id=self.next_norm_id,
                eigenvector=v,
                eigenvalue=float(lam),
                agent_contributions=contributions,
                persistence=float(persistence),
                detection_time=self.t
            )
            self.next_norm_id += 1
            norms.append(norm)

        # Guardar eigenvectores actuales
        self.prev_eigenvectors = eigenvectors
        self.prev_eigenvalues = eigenvalues

        return norms

    def _compute_norm_values(self, norms: List[EmergentNorm]):
        """
        Calcula valor de cada norma.

        val_ℓ = corr((p_t · v_ℓ), G_t)
        """
        if len(self.policy_history) < 20 or len(self.G_history) < 20:
            return

        window = min(50, len(self.policy_history))
        policies = np.array(self.policy_history[-window:])
        G = np.array(self.G_history[-window:])

        for norm in norms:
            # Activación de la norma en cada paso
            if len(norm.eigenvector) != policies.shape[1]:
                continue

            activations = policies @ norm.eigenvector

            # Correlación con utilidad global
            if np.std(activations) > 0 and np.std(G) > 0:
                corr = np.corrcoef(activations, G)[0, 1]
                norm.value_correlation = float(corr) if not np.isnan(corr) else 0.0

    def _compute_norm_weights(self, norms: List[EmergentNorm]):
        """
        Calcula pesos normativos.

        W_ℓ = rank(pers_ℓ) + rank(val_ℓ)
        """
        if not norms:
            return

        persistences = [n.persistence for n in norms]
        values = [n.value_correlation for n in norms]

        for norm in norms:
            rank_pers = np.sum(np.array(persistences) <= norm.persistence)
            rank_val = np.sum(np.array(values) <= norm.value_correlation)
            norm.weight = float(rank_pers + rank_val)

            # Norma es estable si persiste y tiene valor positivo
            norm.is_stable = norm.persistence > 0.5 and norm.value_correlation > 0

    def record_policies(self, agent_policies: Dict[str, np.ndarray],
                       agent_values: Dict[str, float]):
        """
        Registra políticas de todos los agentes.

        Args:
            agent_policies: {nombre: vector de política}
            agent_values: {nombre: valor V}
        """
        self.t += 1

        # Construir vector concatenado
        p_t = []
        for name in self.agent_names:
            if name in agent_policies:
                p_t.extend(agent_policies[name])
            else:
                p_t.extend(np.ones(self.n_policies) / self.n_policies)

        p_t = np.array(p_t)
        self.policy_history.append(p_t)

        # Utilidad global
        G_t = np.mean(list(agent_values.values())) if agent_values else 0.5
        self.G_history.append(G_t)

        # Limitar historial
        max_hist = 500
        if len(self.policy_history) > max_hist:
            self.policy_history = self.policy_history[-max_hist:]
            self.G_history = self.G_history[-max_hist:]

        # Actualizar learning rate
        self.eta_t = 1.0 / np.sqrt(self.t + 1)

        # Detectar normas periódicamente
        if self.t % 20 == 0:
            C = self._compute_cooccurrence()
            if C is not None:
                new_norms = self._detect_norms(C)
                self._compute_norm_values(new_norms)
                self._compute_norm_weights(new_norms)

                # Integrar nuevas normas
                for norm in new_norms:
                    if norm.is_stable:
                        # Buscar norma similar existente
                        found = False
                        for existing in self.norms.values():
                            similarity = abs(np.dot(norm.eigenvector, existing.eigenvector))
                            if similarity > 0.8:
                                # Actualizar existente
                                existing.persistence = 0.9 * existing.persistence + 0.1 * norm.persistence
                                existing.value_correlation = 0.9 * existing.value_correlation + 0.1 * norm.value_correlation
                                existing.weight = norm.weight
                                found = True
                                break

                        if not found:
                            self.norms[norm.norm_id] = norm

                # Limpiar normas inestables
                to_remove = [nid for nid, n in self.norms.items()
                            if not n.is_stable and self.t - n.detection_time > 100]
                for nid in to_remove:
                    del self.norms[nid]

    def get_normative_adjustment(self, agent_name: str,
                                current_policy: np.ndarray) -> np.ndarray:
        """
        Obtiene ajuste normativo para un agente.

        π_A^new ∝ π_A^old + η_t Σ_ℓ W_ℓ · (v_ℓ)_A

        Args:
            agent_name: Nombre del agente
            current_policy: Política actual

        Returns:
            Política ajustada
        """
        if not self.norms:
            return current_policy

        agent_idx = self.agent_names.index(agent_name) if agent_name in self.agent_names else 0
        start = agent_idx * self.n_policies
        end = start + self.n_policies

        adjustment = np.zeros(self.n_policies)

        for norm in self.norms.values():
            if norm.is_stable:
                # Componente de la norma para este agente
                norm_component = norm.eigenvector[start:end]
                adjustment += norm.weight * norm_component

        # Aplicar ajuste
        new_policy = current_policy + self.eta_t * adjustment

        # Normalizar
        new_policy = np.clip(new_policy, 0.01, None)
        new_policy /= new_policy.sum()

        return new_policy

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas de normas."""
        if not self.norms:
            return {
                'n_agents': self.n_agents,
                't': self.t,
                'n_norms': 0,
                'n_stable': 0,
                'eta_t': self.eta_t,
                'mean_G': float(np.mean(self.G_history[-50:])) if self.G_history else 0,
                'norms': []
            }

        stable_norms = [n for n in self.norms.values() if n.is_stable]

        norm_info = []
        for n in sorted(self.norms.values(), key=lambda x: x.weight, reverse=True)[:5]:
            norm_info.append({
                'id': n.norm_id,
                'eigenvalue': n.eigenvalue,
                'persistence': n.persistence,
                'value_corr': n.value_correlation,
                'weight': n.weight,
                'is_stable': n.is_stable,
                'contributions': n.agent_contributions
            })

        return {
            'n_agents': self.n_agents,
            't': self.t,
            'n_norms': len(self.norms),
            'n_stable': len(stable_norms),
            'eta_t': self.eta_t,
            'mean_G': float(np.mean(self.G_history[-50:])) if self.G_history else 0,
            'norms': norm_info
        }


def test_norms():
    """Test de emergencia de normas."""
    print("=" * 60)
    print("TEST AGI-12: NORM EMERGENCE")
    print("=" * 60)

    agents = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
    norms = NormEmergence(agents, n_policies=5)

    print(f"\nSimulando 500 pasos con {len(agents)} agentes...")

    for t in range(500):
        # Políticas con correlación (simulando comportamiento compartido)
        base_policy = np.array([0.3, 0.2, 0.2, 0.15, 0.15])
        if t % 50 < 25:
            # Patrón coordinado
            base_policy = np.array([0.4, 0.25, 0.15, 0.1, 0.1])

        policies = {}
        values = {}

        for agent in agents:
            # Política base + variación individual
            policy = base_policy + np.random.randn(5) * 0.05
            policy = np.clip(policy, 0.01, None)
            policy /= policy.sum()
            policies[agent] = policy

            # Valor correlacionado con coordinación
            coordination = 1 - np.std([p[0] for p in policies.values()])
            values[agent] = 0.5 + coordination * 0.3 + np.random.randn() * 0.1

        norms.record_policies(policies, values)

        if (t + 1) % 100 == 0:
            stats = norms.get_statistics()
            print(f"  t={t+1}: {stats['n_norms']} normas, "
                  f"{stats['n_stable']} estables, "
                  f"mean_G={stats['mean_G']:.3f}")

    # Resultados finales
    stats = norms.get_statistics()

    print("\n" + "=" * 60)
    print("RESULTADOS NORM EMERGENCE")
    print("=" * 60)

    print(f"\n  Normas detectadas: {stats['n_norms']}")
    print(f"  Normas estables: {stats['n_stable']}")
    print(f"  Learning rate: {stats['eta_t']:.4f}")
    print(f"  Utilidad global media: {stats['mean_G']:.3f}")

    print("\n  Top normas:")
    for n in stats['norms']:
        print(f"    Norma {n['id']}: λ={n['eigenvalue']:.2f}, "
              f"pers={n['persistence']:.2f}, val={n['value_corr']:.2f}, "
              f"stable={n['is_stable']}")

    # Probar ajuste normativo
    print("\n  Ajuste normativo para NEO:")
    test_policy = np.ones(5) / 5
    adjusted = norms.get_normative_adjustment('NEO', test_policy)
    print(f"    Original: {test_policy}")
    print(f"    Ajustado: {adjusted}")

    if stats['n_stable'] > 0:
        print("\n  ✓ Normas emergiendo y estabilizándose")
    else:
        print("\n  ⚠️ No se detectaron normas estables")

    return norms


if __name__ == "__main__":
    test_norms()
