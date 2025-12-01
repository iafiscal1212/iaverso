"""
AGI-6: Skills Estructurales
===========================

Detecta secuencias útiles de acción interna y las convierte
en "habilidades" reutilizables que aumentan U y V.

Ventana: W = ⌈√T⌉
Secuencia: L = ⌈√W⌉
Score: ΔV̂ = (ΔV - μ) / σ
Cluster eigenvalores → número de skills
Skills = centroides de embeddings de subsecuencias
Uso: P(S_s) ∝ rank(ΔV̂_s)

100% endógeno.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from .agi_dynamic_constants import (
    L_t, max_history, similarity_threshold, adaptive_momentum,
    update_period, n_clusters_from_eigenvalues, kmeans_iterations
)


@dataclass
class Skill:
    """Una habilidad estructural aprendida."""
    skill_id: int
    centroid: np.ndarray  # Embedding del skill
    action_sequence: List[int]  # Secuencia de acciones típica
    mean_delta_V: float  # Mejora media de valor
    usage_count: int = 0
    success_rate: float = 0.5
    creation_time: int = 0
    last_used: int = 0


@dataclass
class SkillActivation:
    """Resultado de activar un skill."""
    skill: Skill
    match_score: float
    predicted_delta_V: float
    confidence: float


class StructuralSkills:
    """
    Sistema de skills estructurales.

    Detecta patrones de acción que producen buenos resultados
    y los convierte en habilidades reutilizables.
    """

    def __init__(self, agent_name: str, action_dim: int = 10):
        """
        Inicializa sistema de skills.

        Args:
            agent_name: Nombre del agente
            action_dim: Dimensión del espacio de acciones
        """
        self.agent_name = agent_name
        self.action_dim = action_dim

        # Historial de acciones y valores
        self.action_history: List[np.ndarray] = []
        self.value_history: List[float] = []

        # Skills aprendidos
        self.skills: Dict[int, Skill] = {}
        self.next_skill_id = 0

        # Matriz de subsecuencias para clustering
        self.subsequence_embeddings: List[np.ndarray] = []
        self.subsequence_delta_V: List[float] = []

        self.t = 0

    def _get_window_size(self) -> int:
        """Calcula tamaño de ventana: W = ⌈√T⌉"""
        return L_t(self.t)

    def _get_sequence_length(self) -> int:
        """Calcula longitud de secuencia: L = ⌈√W⌉"""
        W = self._get_window_size()
        return max(3, int(np.ceil(np.sqrt(W))))

    def _compute_delta_V_normalized(self, delta_V: float) -> float:
        """
        Normaliza delta_V: ΔV̂ = (ΔV - μ) / σ
        """
        min_samples = L_t(self.t)
        if len(self.value_history) < min_samples:
            return 0.0

        # Calcular deltas históricos
        deltas = np.diff(self.value_history)
        if len(deltas) < min_samples:
            return 0.0

        mu = np.mean(deltas)
        sigma = np.std(deltas) + 1e-8

        return (delta_V - mu) / sigma

    def _embed_subsequence(self, actions: List[np.ndarray]) -> np.ndarray:
        """
        Genera embedding de una subsecuencia de acciones.

        Usa estadísticas de la secuencia como features.
        """
        if not actions:
            return np.zeros(self.action_dim * 3)

        actions_arr = np.array(actions)

        # Features: media, std, transiciones
        mean = np.mean(actions_arr, axis=0)
        std = np.std(actions_arr, axis=0)

        # Transiciones (diferencias)
        if len(actions) > 1:
            diffs = np.diff(actions_arr, axis=0)
            transitions = np.mean(np.abs(diffs), axis=0)
        else:
            transitions = np.zeros(self.action_dim)

        return np.concatenate([mean, std, transitions])

    def _extract_subsequences(self) -> List[Tuple[np.ndarray, float]]:
        """
        Extrae subsecuencias y sus delta_V asociados.
        """
        min_samples = L_t(self.t)
        if len(self.action_history) < min_samples:
            return []

        L = self._get_sequence_length()
        subsequences = []

        for i in range(len(self.action_history) - L):
            # Subsecuencia de acciones
            subseq = self.action_history[i:i+L]

            # Delta V de la subsecuencia
            if i + L < len(self.value_history):
                delta_V = self.value_history[i+L] - self.value_history[i]
            else:
                continue

            embedding = self._embed_subsequence(subseq)
            delta_V_norm = self._compute_delta_V_normalized(delta_V)

            subsequences.append((embedding, delta_V_norm))

        return subsequences

    def _cluster_skills(self, embeddings: np.ndarray,
                       delta_Vs: np.ndarray) -> List[Skill]:
        """
        Agrupa subsecuencias en skills usando eigenvalores.

        Número de clusters = eigenvalores >= mediana
        """
        min_samples = L_t(self.t)
        if len(embeddings) < min_samples:
            return []

        # Matriz de covarianza
        cov = np.cov(embeddings.T)
        if cov.ndim == 0:
            return []

        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except:
            return []

        # Número de clusters = eigenvalores >= mediana (endógeno)
        n_clusters = n_clusters_from_eigenvalues(eigenvalues)

        # Simple k-means clustering
        skills = []
        if n_clusters > 0:
            # Inicializar centroides con k-means++
            centroids = self._kmeans_init(embeddings, n_clusters)

            # Iterar k-means (número adaptativo de iteraciones)
            n_iters = kmeans_iterations(len(embeddings))
            for _ in range(n_iters):
                # Asignar puntos a clusters
                assignments = []
                for emb in embeddings:
                    dists = [np.linalg.norm(emb - c) for c in centroids]
                    assignments.append(np.argmin(dists))

                # Actualizar centroides
                new_centroids = []
                for k in range(n_clusters):
                    mask = np.array(assignments) == k
                    if np.any(mask):
                        new_centroids.append(np.mean(embeddings[mask], axis=0))
                    else:
                        new_centroids.append(centroids[k])
                centroids = new_centroids

            # Crear skills
            for k in range(n_clusters):
                mask = np.array(assignments) == k
                if not np.any(mask):
                    continue

                cluster_delta_Vs = delta_Vs[mask]
                mean_delta_V = float(np.mean(cluster_delta_Vs))

                # Solo crear skill si tiene valor positivo
                if mean_delta_V > 0:
                    skill = Skill(
                        skill_id=self.next_skill_id,
                        centroid=centroids[k],
                        action_sequence=[],  # Se llena después
                        mean_delta_V=mean_delta_V,
                        creation_time=self.t
                    )
                    skills.append(skill)
                    self.next_skill_id += 1

        return skills

    def _kmeans_init(self, X: np.ndarray, k: int) -> List[np.ndarray]:
        """Inicialización k-means++."""
        centroids = [X[np.random.randint(len(X))]]

        for _ in range(k - 1):
            # Distancias al centroide más cercano
            dists = np.array([
                min(np.linalg.norm(x - c) for c in centroids)
                for x in X
            ])

            # Probabilidad proporcional a distancia²
            probs = dists ** 2
            probs = probs / probs.sum()

            # Seleccionar nuevo centroide
            idx = np.random.choice(len(X), p=probs)
            centroids.append(X[idx])

        return centroids

    def record_action(self, action: np.ndarray, value: float):
        """
        Registra una acción y su valor resultante.

        Args:
            action: Vector de acción tomada
            value: Valor resultante
        """
        self.t += 1

        self.action_history.append(action)
        self.value_history.append(value)

        # Limitar historial adaptativamente
        max_hist = max_history(self.t)
        if len(self.action_history) > max_hist:
            self.action_history = self.action_history[-max_hist:]
            self.value_history = self.value_history[-max_hist:]

        # Actualizar skills con período adaptativo
        period = update_period(self.value_history)
        if self.t % period == 0:
            self._update_skills()

    def _update_skills(self):
        """Actualiza el conjunto de skills."""
        subsequences = self._extract_subsequences()
        min_samples = L_t(self.t)
        if len(subsequences) < min_samples:
            return

        embeddings = np.array([s[0] for s in subsequences])
        delta_Vs = np.array([s[1] for s in subsequences])

        # Guardar para referencia
        self.subsequence_embeddings = list(embeddings)
        self.subsequence_delta_V = list(delta_Vs)

        # Clustering
        new_skills = self._cluster_skills(embeddings, delta_Vs)

        # Integrar nuevos skills
        # Calcular umbral de similaridad endógeno
        sim_history = []
        for skill in new_skills:
            for existing in self.skills.values():
                sim = 1.0 / (1.0 + np.linalg.norm(skill.centroid - existing.centroid))
                sim_history.append(sim)

        sim_thresh = similarity_threshold(sim_history) if sim_history else 0.5

        for skill in new_skills:
            # Verificar si ya existe uno similar
            is_new = True
            for existing_id, existing in self.skills.items():
                similarity = 1.0 / (1.0 + np.linalg.norm(skill.centroid - existing.centroid))
                if similarity > sim_thresh:
                    # Actualizar existente con momentum adaptativo
                    beta = adaptive_momentum(self.value_history)
                    existing.mean_delta_V = beta * existing.mean_delta_V + (1 - beta) * skill.mean_delta_V
                    is_new = False
                    break

            if is_new:
                self.skills[skill.skill_id] = skill

        # Eliminar skills poco útiles (umbrales endógenos)
        to_remove = []
        usage_counts = [s.usage_count for s in self.skills.values() if s.usage_count > 0]
        success_rates = [s.success_rate for s in self.skills.values()]

        usage_threshold = L_t(self.t)
        success_threshold = np.percentile(success_rates, 25) if success_rates else 0.3
        age_threshold = max_history(self.t) // 2

        for skill_id, skill in self.skills.items():
            if skill.usage_count > usage_threshold and skill.success_rate < success_threshold:
                to_remove.append(skill_id)
            elif self.t - skill.creation_time > age_threshold and skill.usage_count == 0:
                to_remove.append(skill_id)

        for skill_id in to_remove:
            del self.skills[skill_id]

    def match_skill(self, current_sequence: List[np.ndarray]) -> Optional[SkillActivation]:
        """
        Encuentra el skill que mejor coincide con la secuencia actual.

        Args:
            current_sequence: Secuencia de acciones reciente

        Returns:
            SkillActivation o None
        """
        if not self.skills or len(current_sequence) < 2:
            return None

        current_embedding = self._embed_subsequence(current_sequence)

        best_match = None
        best_score = 0.0

        for skill in self.skills.values():
            # Similitud coseno
            dot = np.dot(current_embedding, skill.centroid)
            norm = np.linalg.norm(current_embedding) * np.linalg.norm(skill.centroid)
            if norm > 0:
                similarity = dot / norm
            else:
                similarity = 0

            # Usar umbral endógeno basado en historial de similaridades
            if similarity > best_score:
                best_score = similarity
                best_match = skill

        if best_match:
            return SkillActivation(
                skill=best_match,
                match_score=best_score,
                predicted_delta_V=best_match.mean_delta_V,
                confidence=best_match.success_rate
            )

        return None

    def get_skill_probability(self, skill_id: int) -> float:
        """
        Calcula probabilidad de usar un skill.

        P(S_s) ∝ rank(ΔV̂_s)
        """
        if skill_id not in self.skills:
            return 0.0

        # Rankear por delta_V
        delta_Vs = [s.mean_delta_V for s in self.skills.values()]
        skill_ids = list(self.skills.keys())

        ranks = {}
        sorted_indices = np.argsort(delta_Vs)
        for rank_idx, idx in enumerate(sorted_indices):
            ranks[skill_ids[idx]] = rank_idx + 1

        total_rank = sum(ranks.values())
        return ranks[skill_id] / total_rank if total_rank > 0 else 0

    def report_skill_outcome(self, skill_id: int, success: bool, actual_delta_V: float):
        """
        Reporta el resultado de usar un skill.

        Args:
            skill_id: ID del skill usado
            success: Si fue exitoso
            actual_delta_V: Cambio real en valor
        """
        if skill_id not in self.skills:
            return

        skill = self.skills[skill_id]
        skill.usage_count += 1
        skill.last_used = self.t

        # Actualizar success rate
        alpha = 1.0 / skill.usage_count
        skill.success_rate = (1 - alpha) * skill.success_rate + alpha * float(success)

        # Actualizar predicción de delta_V con momentum adaptativo
        beta = adaptive_momentum(self.value_history)
        skill.mean_delta_V = beta * skill.mean_delta_V + (1 - beta) * actual_delta_V

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas de skills."""
        if not self.skills:
            return {
                'agent': self.agent_name,
                't': self.t,
                'n_skills': 0,
                'skills': []
            }

        skill_info = []
        for skill in self.skills.values():
            skill_info.append({
                'id': skill.skill_id,
                'mean_delta_V': skill.mean_delta_V,
                'usage_count': skill.usage_count,
                'success_rate': skill.success_rate,
                'probability': self.get_skill_probability(skill.skill_id)
            })

        return {
            'agent': self.agent_name,
            't': self.t,
            'n_skills': len(self.skills),
            'skills': sorted(skill_info, key=lambda x: x['mean_delta_V'], reverse=True),
            'mean_delta_V_all': float(np.mean([s['mean_delta_V'] for s in skill_info])) if skill_info else 0,
            'mean_success_rate': float(np.mean([s['success_rate'] for s in skill_info])) if skill_info else 0
        }


def test_skills():
    """Test de skills estructurales."""
    print("=" * 60)
    print("TEST AGI-6: SKILLS ESTRUCTURALES")
    print("=" * 60)

    skills = StructuralSkills("NEO", action_dim=5)

    # Simular 500 pasos con patrones de acción
    print("\nSimulando 500 pasos...")

    for t in range(500):
        # Generar acción con patrones
        if t % 50 < 25:
            # Patrón A: acciones que aumentan valor
            action = np.array([0.8, 0.2, 0.5, 0.1, 0.3]) + np.random.randn(5) * 0.1
            value = 0.5 + 0.01 * t + np.random.randn() * 0.05
        else:
            # Patrón B: acciones que mantienen valor
            action = np.array([0.2, 0.8, 0.3, 0.6, 0.4]) + np.random.randn(5) * 0.1
            value = 0.4 + 0.005 * t + np.random.randn() * 0.05

        skills.record_action(action, value)

        if (t + 1) % 100 == 0:
            stats = skills.get_statistics()
            print(f"  t={t+1}: {stats['n_skills']} skills detectados")

    # Resultados finales
    stats = skills.get_statistics()

    print("\n" + "=" * 60)
    print("RESULTADOS SKILLS")
    print("=" * 60)

    print(f"\n  Total skills: {stats['n_skills']}")
    print(f"  Mean delta_V: {stats['mean_delta_V_all']:.3f}")
    print(f"  Mean success rate: {stats['mean_success_rate']:.3f}")

    print("\n  Top 5 skills:")
    for skill in stats['skills'][:5]:
        print(f"    Skill {skill['id']}: delta_V={skill['mean_delta_V']:.3f}, "
              f"prob={skill['probability']:.3f}, used={skill['usage_count']}")

    if stats['n_skills'] > 0:
        print("\n  ✓ Skills estructurales emergiendo correctamente")
    else:
        print("\n  ⚠️ No se detectaron skills")

    return skills


if __name__ == "__main__":
    test_skills()
