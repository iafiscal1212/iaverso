"""
AGI-13: Structural Curiosity & Open-Endedness
=============================================

"Buscar sistemáticamente huecos en lo que ya sé hacer."

Densidad local:
    ρ_i = (1/k) Σ_{j∈N_k(i)} exp(-||e_i - e_j||² / σ²)
    k = ⌈√N_epi⌉
    σ² = var(||e_i - e_j||)

Curiosidad espacial:
    C_i^space = rank(1/ρ_i)

Curiosidad por yo alternativo:
    C_self = var_k(J_A^(k))

Índice de curiosidad global:
    C_t = α·rank(C_epi^space) + β·rank(C_self)
    α = 1/std(C_space), β = 1/std(C_self)

100% endógeno.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .agi_dynamic_constants import (
    L_t, max_history, adaptive_momentum, update_period
)


@dataclass
class CuriosityTarget:
    """Un objetivo de curiosidad."""
    target_id: int
    embedding: np.ndarray
    density: float
    curiosity_score: float
    times_explored: int = 0
    last_explored: int = 0


@dataclass
class CuriosityState:
    """Estado de curiosidad en un momento."""
    t: int
    spatial_curiosity: float
    self_curiosity: float
    total_curiosity: float
    exploration_direction: np.ndarray
    top_targets: List[int]


class StructuralCuriosity:
    """
    Sistema de curiosidad estructural.

    Define curiosidad endógena basada en:
    - Huecos en el espacio de episodios
    - Discrepancias entre yo actual y yos contrafactuales
    """

    def __init__(self, agent_name: str, embedding_dim: int = None):
        """
        Inicializa sistema de curiosidad.

        Args:
            agent_name: Nombre del agente
            embedding_dim: Dimensión de embeddings (None = adaptativa)
        """
        self.agent_name = agent_name
        self._base_embedding_dim = embedding_dim
        self.embedding_dim = embedding_dim if embedding_dim else 10

        # Embeddings de episodios
        self.episode_embeddings: List[np.ndarray] = []
        self.episode_ids: List[int] = []

        # Densidades calculadas
        self.densities: Dict[int, float] = {}

        # Historial de curiosidad
        self.spatial_curiosity_history: List[float] = []
        self.self_curiosity_history: List[float] = []

        # Contrafactuales (de AGI-11)
        self.counterfactual_J_variance: float = 0.0

        # Pesos endógenos
        self.alpha: float = 0.5
        self.beta: float = 0.5

        # Objetivos de curiosidad
        self.targets: Dict[int, CuriosityTarget] = {}
        self.next_target_id = 0

        self.t = 0

    def _compute_sigma(self) -> float:
        """
        Calcula σ² endógenamente.

        σ² = var(||e_i - e_j||)
        """
        min_samples = L_t(self.t)
        if len(self.episode_embeddings) < min_samples:
            return 1.0

        # Calcular todas las distancias con ventana adaptativa
        distances = []
        window = min(max_history(self.t), len(self.episode_embeddings))
        embeddings = np.array(self.episode_embeddings[-window:])

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                d = np.linalg.norm(embeddings[i] - embeddings[j])
                distances.append(d)

        if not distances:
            return 1.0

        return float(np.var(distances) + 1e-8)

    def _compute_density(self, embedding: np.ndarray) -> float:
        """
        Calcula densidad local.

        ρ_i = (1/k) Σ_{j∈N_k(i)} exp(-||e_i - e_j||² / σ²)
        k = ⌈√N_epi⌉
        """
        min_samples = L_t(self.t)
        if len(self.episode_embeddings) < min_samples:
            return 0.5

        # k = sqrt(n_episodios) - endógeno
        k = L_t(len(self.episode_embeddings))
        k = min(k, len(self.episode_embeddings))

        sigma2 = self._compute_sigma()

        # Calcular distancias a todos los episodios
        distances = []
        for e in self.episode_embeddings:
            d = np.linalg.norm(embedding - e)
            distances.append(d)

        # k vecinos más cercanos
        sorted_indices = np.argsort(distances)[:k]

        # Densidad
        density = 0.0
        for idx in sorted_indices:
            d = distances[idx]
            density += np.exp(-d**2 / sigma2)

        density /= k
        return float(density)

    def _compute_spatial_curiosity(self, embedding: np.ndarray) -> float:
        """
        Calcula curiosidad espacial.

        C_i^space = rank(1/ρ_i)
        """
        density = self._compute_density(embedding)

        if density < 1e-8:
            return 1.0  # Máxima curiosidad

        inv_density = 1.0 / density

        # Rankear contra historial
        min_samples = L_t(self.t)
        if len(self.densities) < min_samples:
            return 0.5

        inv_densities = [1.0 / (d + 1e-8) for d in self.densities.values()]
        rank = np.sum(np.array(inv_densities) <= inv_density) / len(inv_densities)

        return float(rank)

    def _update_weights(self):
        """
        Actualiza pesos endógenos.

        α = 1/std(C_space), β = 1/std(C_self)
        """
        min_samples = L_t(self.t)
        if len(self.spatial_curiosity_history) < min_samples:
            return

        window = min(max_history(self.t), len(self.spatial_curiosity_history))
        std_space = np.std(self.spatial_curiosity_history[-window:]) + 1e-8
        std_self = np.std(self.self_curiosity_history[-window:]) + 1e-8 if self.self_curiosity_history else 1.0

        self.alpha = 1.0 / std_space
        self.beta = 1.0 / std_self

        # Normalizar
        total = self.alpha + self.beta
        self.alpha /= total
        self.beta /= total

    def record_episode(self, episode_id: int, embedding: np.ndarray):
        """
        Registra un nuevo episodio.

        Args:
            episode_id: ID del episodio
            embedding: Embedding del episodio
        """
        self.t += 1

        self.episode_embeddings.append(embedding.copy())
        self.episode_ids.append(episode_id)

        # Limitar historial adaptativamente
        max_hist = max_history(self.t)
        if len(self.episode_embeddings) > max_hist:
            self.episode_embeddings = self.episode_embeddings[-max_hist:]
            self.episode_ids = self.episode_ids[-max_hist:]

        # Calcular densidad
        density = self._compute_density(embedding)
        self.densities[episode_id] = density

        # Limpiar densidades antiguas
        if len(self.densities) > max_hist:
            old_ids = list(self.densities.keys())[:-max_hist]
            for oid in old_ids:
                del self.densities[oid]

        # Crear objetivo de curiosidad si zona poco densa
        if density < np.median(list(self.densities.values())) if self.densities else 0.5:
            target = CuriosityTarget(
                target_id=self.next_target_id,
                embedding=embedding.copy(),
                density=density,
                curiosity_score=1.0 / (density + 1e-8)
            )
            self.targets[self.next_target_id] = target
            self.next_target_id += 1

            # Limitar objetivos adaptativamente
            max_targets = L_t(self.t)
            if len(self.targets) > max_targets:
                # Eliminar menos curiosos
                sorted_targets = sorted(self.targets.values(),
                                        key=lambda t: t.curiosity_score)
                for t in sorted_targets[:5]:
                    del self.targets[t.target_id]

    def update_from_counterfactual(self, J_variance: float):
        """
        Actualiza curiosidad desde análisis contrafactual.

        C_self = var_k(J_A^(k))

        Args:
            J_variance: Varianza de J scores de yos alternativos
        """
        self.counterfactual_J_variance = J_variance
        self.self_curiosity_history.append(J_variance)

        if len(self.self_curiosity_history) > 500:
            self.self_curiosity_history = self.self_curiosity_history[-500:]

    def get_curiosity_state(self, current_embedding: np.ndarray) -> CuriosityState:
        """
        Obtiene estado de curiosidad actual.

        C_t = α·rank(C_epi^space) + β·rank(C_self)

        Args:
            current_embedding: Embedding actual

        Returns:
            CuriosityState
        """
        # Actualizar pesos
        self._update_weights()

        # Curiosidad espacial
        spatial = self._compute_spatial_curiosity(current_embedding)
        self.spatial_curiosity_history.append(spatial)

        if len(self.spatial_curiosity_history) > 500:
            self.spatial_curiosity_history = self.spatial_curiosity_history[-500:]

        # Curiosidad por yo alternativo
        if self.self_curiosity_history:
            self_curiosity = np.sum(np.array(self.self_curiosity_history) <=
                                   self.counterfactual_J_variance) / len(self.self_curiosity_history)
        else:
            self_curiosity = 0.5

        # Curiosidad total
        total = self.alpha * spatial + self.beta * self_curiosity

        # Dirección de exploración (hacia zona menos densa)
        if self.targets:
            # Hacia objetivo más curioso
            best_target = max(self.targets.values(), key=lambda t: t.curiosity_score)
            direction = best_target.embedding - current_embedding
            direction = direction / (np.linalg.norm(direction) + 1e-8)
        else:
            direction = np.random.randn(self.embedding_dim)
            direction = direction / np.linalg.norm(direction)

        # Top objetivos
        top_targets = sorted(self.targets.keys(),
                            key=lambda tid: self.targets[tid].curiosity_score,
                            reverse=True)[:5]

        return CuriosityState(
            t=self.t,
            spatial_curiosity=float(spatial),
            self_curiosity=float(self_curiosity),
            total_curiosity=float(total),
            exploration_direction=direction,
            top_targets=top_targets
        )

    def get_exploration_score(self, trajectory_embedding: np.ndarray,
                             expected_delta_V: float) -> float:
        """
        Calcula score de una trayectoria balanceando curiosidad y valor.

        score = γ·rank(ΔV) + (1-γ)·rank(C̄_t)
        γ = 1/√(t+1)

        Args:
            trajectory_embedding: Embedding de la trayectoria
            expected_delta_V: Cambio esperado en valor

        Returns:
            Score de exploración
        """
        # γ decrece con el tiempo (más curiosidad al inicio)
        gamma = 1.0 / np.sqrt(self.t + 1)
        gamma = min(0.9, max(0.1, gamma))

        # Curiosidad de la trayectoria
        curiosity = self._compute_spatial_curiosity(trajectory_embedding)

        # Combinar
        score = gamma * expected_delta_V + (1 - gamma) * curiosity

        return float(score)

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas de curiosidad."""
        return {
            'agent': self.agent_name,
            't': self.t,
            'n_episodes': len(self.episode_embeddings),
            'n_targets': len(self.targets),
            'alpha': self.alpha,
            'beta': self.beta,
            'mean_spatial_curiosity': float(np.mean(self.spatial_curiosity_history[-50:]))
                if self.spatial_curiosity_history else 0,
            'mean_self_curiosity': float(np.mean(self.self_curiosity_history[-50:]))
                if self.self_curiosity_history else 0,
            'counterfactual_J_variance': self.counterfactual_J_variance,
            'top_target_scores': [self.targets[tid].curiosity_score
                                 for tid in sorted(self.targets.keys(),
                                                  key=lambda t: self.targets[t].curiosity_score,
                                                  reverse=True)[:3]]
        }


def test_curiosity():
    """Test de curiosidad estructural."""
    print("=" * 60)
    print("TEST AGI-13: STRUCTURAL CURIOSITY")
    print("=" * 60)

    curiosity = StructuralCuriosity("NEO", embedding_dim=5)

    print("\nSimulando 300 episodios con diferentes densidades...")

    for t in range(300):
        # Generar embeddings con clusters
        if t % 100 < 70:
            # Zona densa
            center = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            embedding = center + np.random.randn(5) * 0.2
        else:
            # Zona poco densa (exploratoria)
            center = np.array([2.0, 1.0, -1.0, 0.5, -0.5])
            embedding = center + np.random.randn(5) * 0.5

        curiosity.record_episode(t, embedding)

        # Simular contrafactual
        if t % 20 == 0:
            J_var = np.random.uniform(0.5, 3.0)
            curiosity.update_from_counterfactual(J_var)

        if (t + 1) % 50 == 0:
            current = np.random.randn(5) * 0.3
            state = curiosity.get_curiosity_state(current)
            print(f"  t={t+1}: spatial={state.spatial_curiosity:.3f}, "
                  f"self={state.self_curiosity:.3f}, "
                  f"total={state.total_curiosity:.3f}")

    # Resultados finales
    stats = curiosity.get_statistics()

    print("\n" + "=" * 60)
    print("RESULTADOS STRUCTURAL CURIOSITY")
    print("=" * 60)

    print(f"\n  Episodios: {stats['n_episodes']}")
    print(f"  Objetivos de exploración: {stats['n_targets']}")
    print(f"  Pesos: α={stats['alpha']:.3f}, β={stats['beta']:.3f}")
    print(f"  Curiosidad espacial media: {stats['mean_spatial_curiosity']:.3f}")
    print(f"  Curiosidad self media: {stats['mean_self_curiosity']:.3f}")

    # Probar score de exploración
    print("\n  Scores de exploración:")
    test_embeddings = [
        (np.array([0.0, 0.0, 0.0, 0.0, 0.0]), 0.5, "zona densa"),
        (np.array([2.0, 1.0, -1.0, 0.5, -0.5]), 0.3, "zona poco densa"),
        (np.array([5.0, 5.0, 5.0, 5.0, 5.0]), 0.1, "zona muy nueva")
    ]

    for emb, delta_V, desc in test_embeddings:
        score = curiosity.get_exploration_score(emb, delta_V)
        print(f"    {desc}: score={score:.3f}")

    if stats['n_targets'] > 0:
        print("\n  ✓ Curiosidad estructural funcionando")
    else:
        print("\n  ⚠️ No se detectaron objetivos de exploración")

    return curiosity


if __name__ == "__main__":
    test_curiosity()
