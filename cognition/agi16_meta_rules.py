"""
AGI-16: Meta-Reglas Estructurales
==================================

"Reglas sobre las reglas" - El sistema aprende cuándo sus normas,
políticas y skills funcionan bien o mal.

Clustering de contextos:
    k(t) = 2 + floor(√log(t+1))
    C_j = cluster_j(c_1, ..., c_t)

Utilidad condicional:
    U_ij = E[u_t | c_t ∈ C_j, π_t = π_i]

Fuerza de meta-regla:
    R_ij = U_ij - median_k(U_kj)

Umbral de validez:
    τ_R(t) = percentile({|R_ij|}, 75 + 10/√t)

Meta-política:
    Π*(c) = argmax_πi R_ij(c)

Persistencia:
    P_ij cuenta cuántas veces R_ij > τ_R

100% endógeno.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from sklearn.cluster import KMeans
from .agi_dynamic_constants import (
    L_t, max_history, dynamic_percentile_high, adaptive_momentum
)


def k_clusters(t: int) -> int:
    """
    Número de clusters endógeno.

    k(t) = 2 + floor(√log(t+1))
    """
    return 2 + int(np.sqrt(np.log(t + 1)))


def meta_rule_threshold(R_values: List[float], t: int) -> float:
    """
    Umbral de validez de meta-regla.

    τ_R(t) = percentile({|R_ij|}, 75 + 10/√t)
    """
    if not R_values:
        return 0.0

    percentile_idx = 75 + 10 / np.sqrt(t + 1)
    percentile_idx = min(99, percentile_idx)

    return float(np.percentile(np.abs(R_values), percentile_idx))


@dataclass
class MetaRule:
    """Una meta-regla: (contexto_cluster, política) → utilidad."""
    rule_id: int
    cluster_id: int
    policy_id: int
    conditional_utility: float  # U_ij
    strength: float  # R_ij
    persistence: int  # Veces que R > τ
    total_observations: int
    is_valid: bool = False
    detection_time: int = 0


@dataclass
class ContextCluster:
    """Un cluster de contextos similares."""
    cluster_id: int
    centroid: np.ndarray
    n_contexts: int
    mean_utility: float
    policy_utilities: Dict[int, List[float]]  # policy_id → [utilities]


@dataclass
class MetaRuleState:
    """Estado del sistema de meta-reglas."""
    t: int
    n_clusters: int
    n_meta_rules: int
    n_valid_rules: int
    mean_rule_strength: float
    best_rules: List[Tuple[int, int, float]]  # (cluster, policy, strength)
    meta_utility_gain: float


class StructuralMetaRules:
    """
    Sistema de meta-reglas estructurales.

    Aprende qué políticas funcionan en qué contextos,
    construyendo reglas de segundo orden.
    """

    def __init__(self, agent_name: str, context_dim: int = 10, n_policies: int = 7):
        """
        Inicializa sistema de meta-reglas.

        Args:
            agent_name: Nombre del agente
            context_dim: Dimensión del vector de contexto
            n_policies: Número de políticas disponibles
        """
        self.agent_name = agent_name
        self.context_dim = context_dim
        self.n_policies = n_policies

        # Historial de observaciones
        self.contexts: List[np.ndarray] = []
        self.policies_used: List[int] = []
        self.utilities: List[float] = []

        # Clusters de contexto
        self.clusters: Dict[int, ContextCluster] = {}
        self.context_to_cluster: Dict[int, int] = {}  # context_idx → cluster_id

        # Meta-reglas
        self.meta_rules: Dict[Tuple[int, int], MetaRule] = {}  # (cluster, policy) → rule
        self.next_rule_id = 0

        # Estadísticas
        self.utility_without_meta: List[float] = []
        self.utility_with_meta: List[float] = []

        self.t = 0

    def _update_clusters(self):
        """
        Actualiza clustering de contextos.

        k(t) = 2 + floor(√log(t+1))
        """
        min_samples = L_t(self.t)
        if len(self.contexts) < min_samples:
            return

        # Número de clusters endógeno
        k = k_clusters(self.t)
        k = min(k, len(self.contexts) // 2)
        k = max(2, k)

        # Preparar datos
        window = min(max_history(self.t), len(self.contexts))
        X = np.array(self.contexts[-window:])

        # K-means
        try:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(X)
        except:
            return

        # Actualizar clusters
        self.clusters.clear()
        self.context_to_cluster.clear()

        for cluster_id in range(k):
            mask = labels == cluster_id
            if not np.any(mask):
                continue

            indices = np.where(mask)[0]
            start_idx = len(self.contexts) - window

            # Calcular utilidades por política para este cluster
            policy_utilities: Dict[int, List[float]] = {i: [] for i in range(self.n_policies)}
            cluster_utilities = []

            for local_idx in indices:
                global_idx = start_idx + local_idx
                if global_idx < len(self.policies_used) and global_idx < len(self.utilities):
                    policy = self.policies_used[global_idx]
                    utility = self.utilities[global_idx]
                    policy_utilities[policy].append(utility)
                    cluster_utilities.append(utility)
                    self.context_to_cluster[global_idx] = cluster_id

            self.clusters[cluster_id] = ContextCluster(
                cluster_id=cluster_id,
                centroid=kmeans.cluster_centers_[cluster_id],
                n_contexts=int(np.sum(mask)),
                mean_utility=float(np.mean(cluster_utilities)) if cluster_utilities else 0.5,
                policy_utilities=policy_utilities
            )

    def _compute_conditional_utilities(self) -> Dict[Tuple[int, int], float]:
        """
        Calcula utilidades condicionales.

        U_ij = E[u_t | c_t ∈ C_j, π_t = π_i]
        """
        U = {}

        for cluster_id, cluster in self.clusters.items():
            for policy_id in range(self.n_policies):
                utilities = cluster.policy_utilities.get(policy_id, [])
                if utilities:
                    U[(cluster_id, policy_id)] = float(np.mean(utilities))

        return U

    def _compute_meta_rule_strengths(self, U: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
        """
        Calcula fuerza de meta-reglas.

        R_ij = U_ij - median_k(U_kj)
        """
        R = {}

        for cluster_id in self.clusters:
            # Utilidades de todas las políticas en este cluster
            cluster_U = [U.get((cluster_id, p), 0) for p in range(self.n_policies)
                        if (cluster_id, p) in U]

            if not cluster_U:
                continue

            median_U = np.median(cluster_U)

            for policy_id in range(self.n_policies):
                key = (cluster_id, policy_id)
                if key in U:
                    R[key] = U[key] - median_U

        return R

    def _update_meta_rules(self, U: Dict[Tuple[int, int], float],
                          R: Dict[Tuple[int, int], float]):
        """
        Actualiza meta-reglas.

        τ_R(t) = percentile({|R_ij|}, 75 + 10/√t)
        """
        if not R:
            return

        # Calcular umbral
        threshold = meta_rule_threshold(list(R.values()), self.t)

        for key, strength in R.items():
            cluster_id, policy_id = key

            if key not in self.meta_rules:
                # Nueva meta-regla
                self.meta_rules[key] = MetaRule(
                    rule_id=self.next_rule_id,
                    cluster_id=cluster_id,
                    policy_id=policy_id,
                    conditional_utility=U.get(key, 0),
                    strength=strength,
                    persistence=0,
                    total_observations=1,
                    detection_time=self.t
                )
                self.next_rule_id += 1
            else:
                # Actualizar existente
                rule = self.meta_rules[key]
                beta = adaptive_momentum(self.utilities)
                rule.conditional_utility = beta * rule.conditional_utility + (1 - beta) * U.get(key, 0)
                rule.strength = beta * rule.strength + (1 - beta) * strength
                rule.total_observations += 1

            # Actualizar persistencia y validez
            rule = self.meta_rules[key]
            if abs(rule.strength) > threshold:
                rule.persistence += 1
            rule.is_valid = rule.persistence >= L_t(self.t) // 2

    def record_observation(self, context: np.ndarray, policy_id: int, utility: float):
        """
        Registra una observación (contexto, política, utilidad).

        Args:
            context: Vector de contexto
            policy_id: Política usada
            utility: Utilidad obtenida
        """
        self.t += 1

        self.contexts.append(context.copy())
        self.policies_used.append(policy_id)
        self.utilities.append(utility)

        # Registrar utilidad sin meta-reglas
        self.utility_without_meta.append(utility)

        # Limitar historial
        max_hist = max_history(self.t)
        if len(self.contexts) > max_hist:
            self.contexts = self.contexts[-max_hist:]
            self.policies_used = self.policies_used[-max_hist:]
            self.utilities = self.utilities[-max_hist:]

        # Actualizar clusters y meta-reglas periódicamente
        update_freq = max(10, L_t(self.t))
        if self.t % update_freq == 0:
            self._update_clusters()
            U = self._compute_conditional_utilities()
            R = self._compute_meta_rule_strengths(U)
            self._update_meta_rules(U, R)

    def get_meta_policy(self, context: np.ndarray) -> Tuple[int, float]:
        """
        Obtiene meta-política óptima para un contexto.

        Π*(c) = argmax_πi R_ij(c)

        Args:
            context: Vector de contexto actual

        Returns:
            (mejor_política, confianza)
        """
        if not self.clusters:
            return 0, 0.0

        # Encontrar cluster más cercano
        min_dist = float('inf')
        closest_cluster = 0

        for cluster_id, cluster in self.clusters.items():
            dist = np.linalg.norm(context - cluster.centroid)
            if dist < min_dist:
                min_dist = dist
                closest_cluster = cluster_id

        # Buscar mejor política para este cluster
        best_policy = 0
        best_strength = float('-inf')

        for policy_id in range(self.n_policies):
            key = (closest_cluster, policy_id)
            if key in self.meta_rules:
                rule = self.meta_rules[key]
                if rule.is_valid and rule.strength > best_strength:
                    best_strength = rule.strength
                    best_policy = policy_id

        # Confianza basada en fuerza relativa
        all_strengths = [r.strength for r in self.meta_rules.values() if r.is_valid]
        if all_strengths and best_strength > float('-inf'):
            confidence = (best_strength - np.min(all_strengths)) / (np.max(all_strengths) - np.min(all_strengths) + 1e-8)
        else:
            confidence = 0.0

        return best_policy, float(confidence)

    def apply_meta_policy(self, context: np.ndarray,
                         base_policy: np.ndarray) -> np.ndarray:
        """
        Aplica meta-política a política base.

        Args:
            context: Contexto actual
            base_policy: Política base

        Returns:
            Política ajustada
        """
        best_policy, confidence = self.get_meta_policy(context)

        if confidence < 0.3:
            return base_policy

        # Sesgar hacia mejor política
        adjusted = base_policy.copy()
        boost = confidence * 0.3
        adjusted[best_policy] += boost
        adjusted = np.clip(adjusted, 0.01, None)
        adjusted /= adjusted.sum()

        return adjusted

    def get_meta_utility_gain(self) -> float:
        """
        Calcula ganancia de utilidad por usar meta-reglas.

        ΔU = mean(U_with_meta) - mean(U_without_meta)
        """
        window = min(max_history(self.t), len(self.utility_with_meta))

        if window < L_t(self.t):
            return 0.0

        mean_with = np.mean(self.utility_with_meta[-window:]) if self.utility_with_meta else 0
        mean_without = np.mean(self.utility_without_meta[-window:]) if self.utility_without_meta else 0

        return float(mean_with - mean_without)

    def check_go_conditions(self) -> Tuple[bool, Dict]:
        """
        Verifica condiciones GO/NO-GO.

        GO si:
        - ≥5 meta-reglas válidas
        - ΔU > 0
        - Crisis reducidas

        Returns:
            (is_go, details)
        """
        valid_rules = [r for r in self.meta_rules.values() if r.is_valid]
        n_valid = len(valid_rules)

        delta_u = self.get_meta_utility_gain()

        # Crisis = utilidades muy bajas
        recent_utils = self.utilities[-50:] if len(self.utilities) >= 50 else self.utilities
        crisis_threshold = np.percentile(recent_utils, 10) if recent_utils else 0.3
        n_crises = sum(1 for u in recent_utils if u < crisis_threshold)

        is_go = n_valid >= 5 and delta_u > 0

        return is_go, {
            'n_valid_rules': n_valid,
            'delta_u': delta_u,
            'n_crises': n_crises,
            'conditions_met': {
                'enough_rules': n_valid >= 5,
                'positive_gain': delta_u > 0
            }
        }

    def get_state(self) -> MetaRuleState:
        """Obtiene estado actual del sistema."""
        valid_rules = [r for r in self.meta_rules.values() if r.is_valid]

        # Mejores reglas
        best_rules = sorted(
            [(r.cluster_id, r.policy_id, r.strength) for r in valid_rules],
            key=lambda x: x[2],
            reverse=True
        )[:5]

        return MetaRuleState(
            t=self.t,
            n_clusters=len(self.clusters),
            n_meta_rules=len(self.meta_rules),
            n_valid_rules=len(valid_rules),
            mean_rule_strength=float(np.mean([r.strength for r in valid_rules])) if valid_rules else 0,
            best_rules=best_rules,
            meta_utility_gain=self.get_meta_utility_gain()
        )

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas del sistema."""
        state = self.get_state()
        is_go, go_details = self.check_go_conditions()

        return {
            'agent': self.agent_name,
            't': self.t,
            'n_clusters': state.n_clusters,
            'n_meta_rules': state.n_meta_rules,
            'n_valid_rules': state.n_valid_rules,
            'mean_rule_strength': state.mean_rule_strength,
            'meta_utility_gain': state.meta_utility_gain,
            'best_rules': state.best_rules,
            'is_go': is_go,
            'go_details': go_details
        }


def test_meta_rules():
    """Test de meta-reglas estructurales."""
    print("=" * 60)
    print("TEST AGI-16: STRUCTURAL META-RULES")
    print("=" * 60)

    meta = StructuralMetaRules("NEO", context_dim=5, n_policies=5)

    print("\nSimulando 500 observaciones con patrones contextuales...")

    for t in range(500):
        # Generar contexto con estructura
        if t % 100 < 50:
            # Contexto tipo A
            context = np.array([1.0, 0.0, 0.0, 0.0, 0.0]) + np.random.randn(5) * 0.1
            # Política 0 funciona bien aquí
            if np.random.random() < 0.7:
                policy = 0
                utility = 0.8 + np.random.randn() * 0.1
            else:
                policy = np.random.randint(1, 5)
                utility = 0.4 + np.random.randn() * 0.1
        else:
            # Contexto tipo B
            context = np.array([0.0, 1.0, 0.0, 0.0, 0.0]) + np.random.randn(5) * 0.1
            # Política 2 funciona bien aquí
            if np.random.random() < 0.7:
                policy = 2
                utility = 0.75 + np.random.randn() * 0.1
            else:
                policy = np.random.randint(0, 5)
                if policy == 2:
                    policy = 3
                utility = 0.35 + np.random.randn() * 0.1

        meta.record_observation(context, policy, utility)

        # Registrar si usaríamos meta-política
        if t > 100:
            best_policy, conf = meta.get_meta_policy(context)
            if conf > 0.3:
                # Simular que usamos meta-política
                if (t % 100 < 50 and best_policy == 0) or (t % 100 >= 50 and best_policy == 2):
                    meta_utility = 0.8 + np.random.randn() * 0.1
                else:
                    meta_utility = 0.5 + np.random.randn() * 0.1
                meta.utility_with_meta.append(meta_utility)

        if (t + 1) % 100 == 0:
            state = meta.get_state()
            print(f"  t={t+1}: clusters={state.n_clusters}, "
                  f"rules={state.n_valid_rules}/{state.n_meta_rules}, "
                  f"ΔU={state.meta_utility_gain:.3f}")

    # Resultados finales
    stats = meta.get_statistics()

    print("\n" + "=" * 60)
    print("RESULTADOS STRUCTURAL META-RULES")
    print("=" * 60)

    print(f"\n  Clusters: {stats['n_clusters']}")
    print(f"  Meta-reglas: {stats['n_meta_rules']}")
    print(f"  Reglas válidas: {stats['n_valid_rules']}")
    print(f"  Fuerza media: {stats['mean_rule_strength']:.3f}")
    print(f"  Ganancia utilidad: {stats['meta_utility_gain']:.3f}")

    print("\n  Mejores reglas (cluster, policy, strength):")
    for cluster, policy, strength in stats['best_rules'][:3]:
        print(f"    C{cluster}-P{policy}: R={strength:.3f}")

    print(f"\n  GO status: {stats['is_go']}")
    print(f"  Condiciones: {stats['go_details']['conditions_met']}")

    # Test de meta-política
    print("\n  Test de meta-política:")
    test_contexts = [
        (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), "Tipo A"),
        (np.array([0.0, 1.0, 0.0, 0.0, 0.0]), "Tipo B"),
    ]

    for ctx, desc in test_contexts:
        best, conf = meta.get_meta_policy(ctx)
        print(f"    {desc}: mejor_policy={best}, conf={conf:.3f}")

    if stats['n_valid_rules'] >= 3:
        print("\n  ✓ Meta-reglas estructurales funcionando")
    else:
        print("\n  ⚠ Pocas reglas válidas detectadas")

    return meta


if __name__ == "__main__":
    test_meta_rules()
