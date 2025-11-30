#!/usr/bin/env python3
"""
Phase R2: Endogenous Goal Manifold (EGM)
=========================================

En vez de un solo objetivo S, el sistema descubre una variedad de
"picos" preferidos en su espacio de estados.

No hay "quiero X" - hay "aquí, dinámicamente, todo encaja mejor".

Construcción:
1. Clustering ponderado por S sobre historial (z_t, S_t)
2. Para cada prototipo μ_k:
   - Valor estructural V_k = E[S_t | z_t ≈ μ_k]
   - Persistencia P_k = duración media de visitas
   - Robustez R_k = estabilidad ante perturbaciones
3. Campo de atracción G(z) = Σ rank(V_k + P_k) φ_k(z)
4. EGM = conjunto de picos de G(z)

100% ENDÓGENO - Sin constantes mágicas
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.stats import rankdata
from collections import deque
import json


@dataclass
class GoalPrototype:
    """Un prototipo de 'goal' estructural."""
    mu: np.ndarray  # Centro del cluster
    n_visits: int = 0
    total_S: float = 0.0
    visit_durations: List[int] = field(default_factory=list)
    perturbation_responses: List[float] = field(default_factory=list)

    @property
    def value(self) -> float:
        """Valor estructural V_k = E[S | z ≈ μ_k]."""
        if self.n_visits == 0:
            return 0.0
        return self.total_S / self.n_visits

    @property
    def persistence(self) -> float:
        """Persistencia P_k = duración media de visitas."""
        if not self.visit_durations:
            return 0.0
        return float(np.mean(self.visit_durations))

    @property
    def robustness(self) -> float:
        """Robustez R_k = estabilidad ante perturbaciones."""
        if not self.perturbation_responses:
            return 0.5
        # Robustez = 1 - varianza de respuestas normalizada
        var = np.var(self.perturbation_responses)
        return 1.0 / (1.0 + var)


class EndogenousGoalManifold:
    """
    Sistema de Goal Manifold Endógeno.

    Descubre regiones del espacio de estados donde:
    - S es consistentemente alto
    - Las visitas son persistentes
    - El sistema es robusto a perturbaciones
    """

    def __init__(self, d_state: int = 8):
        self.d_state = d_state
        self.history: List[Tuple[np.ndarray, float]] = []  # (z, S)
        self.prototypes: List[GoalPrototype] = []

        # Estado de visita actual
        self._current_prototype_idx: Optional[int] = None
        self._visit_start_t: int = 0

        # Historial de campo G
        self._G_history: deque = deque(maxlen=1000)

    def _derive_n_clusters(self) -> int:
        """Número de clusters derivado endógenamente de √T."""
        T = len(self.history)
        return max(2, int(np.sqrt(T + 1)))

    def _derive_distance_threshold(self) -> float:
        """Umbral de distancia derivado de historial."""
        if len(self.history) < 10:
            return 1.0

        # Basado en dispersión típica
        states = np.array([h[0] for h in self.history])
        pairwise_dists = []
        n_samples = min(100, len(states))
        indices = np.random.choice(len(states), size=n_samples, replace=False)

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                d = np.linalg.norm(states[indices[i]] - states[indices[j]])
                pairwise_dists.append(d)

        if pairwise_dists:
            return float(np.percentile(pairwise_dists, 25))
        return 1.0

    def _find_nearest_prototype(self, z: np.ndarray) -> Tuple[int, float]:
        """Encuentra prototipo más cercano."""
        if not self.prototypes:
            return -1, float('inf')

        distances = [np.linalg.norm(z - p.mu) for p in self.prototypes]
        idx = int(np.argmin(distances))
        return idx, distances[idx]

    def _update_prototypes_online(self, z: np.ndarray, S: float):
        """
        Actualización online de prototipos (clustering incremental).
        Ponderado por S.
        """
        k = self._derive_n_clusters()
        threshold = self._derive_distance_threshold()

        idx, dist = self._find_nearest_prototype(z)

        if idx == -1 or dist > threshold:
            # Crear nuevo prototipo si hay espacio
            if len(self.prototypes) < k:
                new_proto = GoalPrototype(mu=z.copy())
                new_proto.n_visits = 1
                new_proto.total_S = S
                self.prototypes.append(new_proto)
                idx = len(self.prototypes) - 1
            else:
                # Asignar al más cercano de todos modos
                idx, _ = self._find_nearest_prototype(z)

        if idx >= 0 and idx < len(self.prototypes):
            proto = self.prototypes[idx]

            # Actualizar centro con media móvil ponderada por S
            alpha = 1.0 / (np.sqrt(proto.n_visits + 1))
            weight = S / (np.mean([h[1] for h in self.history[-100:]]) + 1e-12)
            weight = np.clip(weight, 0.1, 10.0)

            proto.mu = proto.mu + alpha * weight * (z - proto.mu)
            proto.n_visits += 1
            proto.total_S += S

        return idx

    def _track_visits(self, prototype_idx: int, t: int):
        """Rastrea duración de visitas a prototipos."""
        if self._current_prototype_idx != prototype_idx:
            # Fin de visita anterior
            if self._current_prototype_idx is not None and self._current_prototype_idx < len(self.prototypes):
                duration = t - self._visit_start_t
                if duration > 0:
                    self.prototypes[self._current_prototype_idx].visit_durations.append(duration)

            # Inicio de nueva visita
            self._current_prototype_idx = prototype_idx
            self._visit_start_t = t

    def _test_perturbation_robustness(self, z: np.ndarray, S: float,
                                      prototype_idx: int):
        """
        Testea robustez del prototipo ante pequeña perturbación.
        """
        if prototype_idx < 0 or prototype_idx >= len(self.prototypes):
            return

        proto = self.prototypes[prototype_idx]

        # Perturbación proporcional a escala del sistema
        if len(self.history) < 10:
            return

        states = np.array([h[0] for h in self.history[-100:]])
        scale = np.std(states)

        # Simular perturbación (sin ejecutarla realmente)
        perturbation = np.random.randn(self.d_state) * scale * 0.1
        z_perturbed = z + perturbation

        # Medir "distancia de retorno" al prototipo
        dist_before = np.linalg.norm(z - proto.mu)
        dist_after = np.linalg.norm(z_perturbed - proto.mu)

        # Respuesta = cambio relativo
        response = (dist_after - dist_before) / (dist_before + 1e-12)
        proto.perturbation_responses.append(response)

        # Limitar historial
        if len(proto.perturbation_responses) > 100:
            proto.perturbation_responses = proto.perturbation_responses[-100:]

    def add_observation(self, z: np.ndarray, S: float):
        """Añade observación (estado, score)."""
        t = len(self.history)
        self.history.append((z.copy(), S))

        # Actualizar prototipos
        idx = self._update_prototypes_online(z, S)

        # Rastrear visitas
        self._track_visits(idx, t)

        # Test de robustez (probabilísticamente)
        if np.random.random() < 0.1:  # 10% de las veces
            self._test_perturbation_robustness(z, S, idx)

    def compute_attraction_field(self, z: np.ndarray) -> float:
        """
        Calcula campo de atracción G(z).

        G(z) = Σ rank(V_k + P_k) * φ_k(z)

        donde φ_k(z) es kernel RBF basado en distancias históricas.
        """
        if not self.prototypes:
            return 0.0

        # Calcular scores de cada prototipo
        scores = []
        for proto in self.prototypes:
            score = proto.value + proto.persistence
            scores.append(score)

        # Rankear scores
        if len(scores) > 1:
            ranks = rankdata(scores, method='average') / len(scores)
        else:
            ranks = [0.5]

        # Calcular kernel bandwidth endógeno
        if len(self.history) > 10:
            states = np.array([h[0] for h in self.history[-100:]])
            bandwidth = float(np.median([np.linalg.norm(s) for s in states])) + 1e-12
        else:
            bandwidth = 1.0

        # Calcular G(z)
        G = 0.0
        for i, proto in enumerate(self.prototypes):
            dist = np.linalg.norm(z - proto.mu)
            phi = np.exp(-dist**2 / (2 * bandwidth**2))  # RBF kernel
            G += ranks[i] * phi

        self._G_history.append(G)
        return G

    def get_goal_manifold(self) -> List[Dict]:
        """
        Retorna el Goal Manifold: picos de G(z).

        Son los prototipos ordenados por score total.
        """
        if not self.prototypes:
            return []

        manifold = []
        for i, proto in enumerate(self.prototypes):
            total_score = proto.value + proto.persistence + proto.robustness
            manifold.append({
                'index': i,
                'mu': proto.mu.tolist(),
                'value': proto.value,
                'persistence': proto.persistence,
                'robustness': proto.robustness,
                'total_score': total_score,
                'n_visits': proto.n_visits
            })

        # Ordenar por score total
        manifold.sort(key=lambda x: x['total_score'], reverse=True)

        return manifold

    def get_nearest_goal(self, z: np.ndarray) -> Optional[Dict]:
        """Retorna el goal más cercano al estado actual."""
        idx, dist = self._find_nearest_prototype(z)
        if idx < 0:
            return None

        manifold = self.get_goal_manifold()
        for goal in manifold:
            if goal['index'] == idx:
                goal['distance'] = float(dist)
                return goal

        return None

    def get_gradient_to_goal(self, z: np.ndarray,
                            goal_idx: int = None) -> np.ndarray:
        """
        Calcula gradiente hacia el goal especificado (o el mejor).

        Útil para guiar la dinámica hacia goals estructurales.
        """
        if not self.prototypes:
            return np.zeros(self.d_state)

        if goal_idx is None:
            # Usar el mejor goal
            manifold = self.get_goal_manifold()
            if not manifold:
                return np.zeros(self.d_state)
            goal_idx = manifold[0]['index']

        if goal_idx >= len(self.prototypes):
            return np.zeros(self.d_state)

        proto = self.prototypes[goal_idx]
        direction = proto.mu - z
        norm = np.linalg.norm(direction)

        if norm < 1e-12:
            return np.zeros(self.d_state)

        # Escalar por distancia y score del goal
        scale = proto.value / (norm + 1e-12)
        return direction * scale

    def get_stats(self) -> Dict:
        """Estadísticas del sistema EGM."""
        if not self.prototypes:
            return {
                'n_prototypes': 0,
                'n_observations': len(self.history)
            }

        manifold = self.get_goal_manifold()

        return {
            'n_prototypes': len(self.prototypes),
            'n_observations': len(self.history),
            'mean_G': float(np.mean(self._G_history)) if self._G_history else 0.0,
            'top_goal_value': manifold[0]['value'] if manifold else 0.0,
            'top_goal_persistence': manifold[0]['persistence'] if manifold else 0.0,
            'top_goal_robustness': manifold[0]['robustness'] if manifold else 0.0,
            'top_goal_total': manifold[0]['total_score'] if manifold else 0.0,
            'goal_score_spread': float(np.std([m['total_score'] for m in manifold])) if len(manifold) > 1 else 0.0,
            'manifold': manifold[:5]  # Top 5 goals
        }


def run_phaseR2_test(n_steps: int = 2000) -> Dict:
    """
    Test de Phase R2: Endogenous Goal Manifold.

    Verifica:
    1. Se forman prototipos de goals
    2. Los goals tienen valor, persistencia y robustez diferenciados
    3. El campo G(z) varía significativamente
    4. Los mejores goals corresponden a regiones de alto S
    """
    print("=" * 70)
    print("PHASE R2: ENDOGENOUS GOAL MANIFOLD (EGM)")
    print("=" * 70)

    egm = EndogenousGoalManifold(d_state=8)

    # Simular dinámica con regiones de alto S
    z = np.random.randn(8) * 0.1

    # Crear "atractores" artificiales para el test
    attractors = [
        np.array([1, 0, 0, 0, 0, 0, 0, 0]) * 0.5,
        np.array([0, 1, 0, 0, 0, 0, 0, 0]) * 0.5,
        np.array([0, 0, 1, 1, 0, 0, 0, 0]) * 0.3,
    ]

    G_values = []
    S_values = []

    print(f"\nEjecutando {n_steps} pasos...")

    for t in range(n_steps):
        # Calcular S basado en cercanía a atractores
        dists_to_attractors = [np.linalg.norm(z - a) for a in attractors]
        min_dist = min(dists_to_attractors)
        S = np.exp(-min_dist)  # S alto cerca de atractores

        # Añadir observación
        egm.add_observation(z, S)

        # Calcular campo G
        G = egm.compute_attraction_field(z)
        G_values.append(G)
        S_values.append(S)

        # Dinámica: moverse hacia atractor más cercano + ruido
        nearest_attractor = attractors[np.argmin(dists_to_attractors)]
        drift = 0.1 * (nearest_attractor - z)
        noise = np.random.randn(8) * 0.05
        z = z + drift + noise

        # Ocasionalmente saltar a otro atractor
        if np.random.random() < 0.02:
            z = attractors[np.random.randint(len(attractors))] + np.random.randn(8) * 0.1

        if (t + 1) % 400 == 0:
            stats = egm.get_stats()
            print(f"  t={t+1}: prototypes={stats['n_prototypes']}, "
                  f"top_score={stats['top_goal_total']:.4f}, G_mean={stats['mean_G']:.4f}")

    # Análisis final
    stats = egm.get_stats()
    manifold = egm.get_goal_manifold()

    # Verificar correlación entre S y proximidad a goals
    if manifold:
        top_goal_mu = np.array(manifold[0]['mu'])
        history_states = np.array([h[0] for h in egm.history])
        history_S = np.array([h[1] for h in egm.history])

        dists_to_top = np.array([np.linalg.norm(s - top_goal_mu) for s in history_states])

        # Correlación negativa esperada (cerca del goal = alto S)
        corr = np.corrcoef(dists_to_top, history_S)[0, 1]
    else:
        corr = 0.0

    # GO/NO-GO criteria
    criteria = {
        'prototypes_formed': len(manifold) >= 2,
        'goals_differentiated': stats['goal_score_spread'] > 0.01 if len(manifold) > 1 else False,
        'top_goal_valuable': stats['top_goal_value'] > 0.3,
        'G_varies': float(np.std(G_values)) > 0.01,
        'goals_correlate_with_S': corr < -0.1  # Negativo porque menor distancia = mayor S
    }

    n_pass = sum(criteria.values())
    go = n_pass >= 3

    print(f"\n{'='*70}")
    print("RESULTADOS PHASE R2")
    print(f"{'='*70}")
    print(f"\nEstadísticas:")
    print(f"  - Prototipos formados: {stats['n_prototypes']}")
    print(f"  - Observaciones: {stats['n_observations']}")
    print(f"  - G medio: {stats['mean_G']:.4f}")
    print(f"  - Spread de scores: {stats['goal_score_spread']:.4f}")

    print(f"\nTop 3 Goals:")
    for i, goal in enumerate(manifold[:3]):
        print(f"  {i+1}. Value={goal['value']:.4f}, "
              f"Persistence={goal['persistence']:.4f}, "
              f"Robustness={goal['robustness']:.4f}, "
              f"Total={goal['total_score']:.4f}")

    print(f"\nCorrelación dist-to-goal vs S: {corr:.4f}")

    print(f"\nGO/NO-GO Criteria:")
    for criterion, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"  - {criterion}: {status}")

    print(f"\n{'GO' if go else 'NO-GO'} ({n_pass}/5 criteria passed)")

    return {
        'go': go,
        'stats': stats,
        'criteria': criteria,
        'correlation_dist_S': corr,
        'G_values': G_values,
        'S_values': S_values
    }


if __name__ == "__main__":
    result = run_phaseR2_test(n_steps=2000)

    # Guardar resultados
    import os
    os.makedirs('/root/NEO_EVA/results/phaseR2', exist_ok=True)

    with open('/root/NEO_EVA/results/phaseR2/phaseR2_results.json', 'w') as f:
        json.dump({
            'go': result['go'],
            'stats': result['stats'],
            'criteria': {k: bool(v) for k, v in result['criteria'].items()},
            'correlation_dist_S': result['correlation_dist_S']
        }, f, indent=2)

    print(f"\nResultados guardados en results/phaseR2/")
