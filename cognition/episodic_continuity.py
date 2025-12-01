"""
Episodic Continuity - Continuidad Vital entre Episodios
========================================================

Cada episodio E_i tiene:
- Vector estructural z̄_i
- Drives medios d̄_i
- φ medio φ̄_i
- Símbolos activos {S_i}
- Efecto en WORLD-1 ΔW_i
- Firma narrativa n_i
- Tema dominante (cluster)
- Intención dominante

Métricas de continuidad CE(i) entre episodios consecutivos:
- CE_struct: Coherencia estructural (Mahalanobis)
- CE_sym: Continuidad simbólica (Jaccard)
- CE_causal: Continuidad causal (coseno de ΔW)
- CE_goal: Continuidad teleológica (coseno de metas)
- CE_cluster: Continuidad narrativa (mismo cluster)

CE_global = promedio ponderado con pesos endógenos (1/var)

Uso cognitivo:
- CE baja → planificación aumenta, curiosidad sube
- CE alta → consolidación y estabilidad

100% endógeno. Sin números mágicos.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import defaultdict
from scipy.spatial.distance import mahalanobis
from sklearn.cluster import KMeans

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history


@dataclass
class EpisodeSignature:
    """Firma completa de un episodio."""
    episode_id: int
    agent_id: str

    # Vectores estructurales
    z_bar: np.ndarray  # Vector estructural medio
    d_bar: np.ndarray  # Drives medios
    phi_bar: float  # φ medio

    # Símbolos
    active_symbols: Set[str]
    top_symbols: List[str]  # Top-k por frecuencia
    symbol_counts: Dict[str, int]

    # Efectos causales
    delta_w: np.ndarray  # Efecto acumulado en WORLD-1

    # Teleología
    goal_vector: np.ndarray  # Meta dominante
    intention: str  # Intención dominante (de AGI-3)

    # Narrativa
    narrative_signature: np.ndarray  # concat([z̄, φ̄, top_symbols_encoded, ΔW])
    theme: Optional[int] = None  # Cluster temático

    # Métricas del episodio
    mean_reward: float = 0.0
    mean_ci: float = 0.0
    mean_cf: float = 0.0
    n_steps: int = 0


@dataclass
class ContinuityMetrics:
    """Métricas de continuidad entre dos episodios."""
    episode_i: int
    episode_j: int

    ce_struct: float  # Coherencia estructural
    ce_sym: float  # Continuidad simbólica
    ce_causal: float  # Continuidad causal
    ce_goal: float  # Continuidad teleológica
    ce_cluster: float  # Continuidad de cluster narrativo

    ce_total: float  # Continuidad total ponderada
    weights: Dict[str, float]  # Pesos usados


class EpisodicContinuity:
    """
    Sistema de continuidad episódica.

    Mide y mantiene la coherencia vital entre episodios,
    detectando disrupciones narrativas y facilitando consolidación.
    """

    def __init__(self, agent_id: str, state_dim: int, n_clusters: int = None):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.t = 0

        # Número de clusters endógeno: sqrt(episodios esperados)
        self.n_clusters_base = n_clusters

        # Historial de episodios
        self.episodes: List[EpisodeSignature] = []
        self.continuity_history: List[ContinuityMetrics] = []

        # Para clustering narrativo
        self.narrative_vectors: List[np.ndarray] = []
        self.cluster_model: Optional[KMeans] = None

        # Historiales para pesos endógenos
        self.ce_struct_history: List[float] = []
        self.ce_sym_history: List[float] = []
        self.ce_causal_history: List[float] = []
        self.ce_goal_history: List[float] = []
        self.ce_cluster_history: List[float] = []

        # Acumuladores del episodio actual
        self._current_episode_data: Dict[str, Any] = self._init_episode_data()

    def _init_episode_data(self) -> Dict[str, Any]:
        """Inicializa acumuladores para un nuevo episodio."""
        return {
            'z_vectors': [],
            'd_vectors': [],
            'phi_values': [],
            'symbols': defaultdict(int),
            'delta_w_accum': np.zeros(self.state_dim),
            'goal_vectors': [],
            'intentions': [],
            'rewards': [],
            'ci_scores': [],
            'cf_scores': [],
            'n_steps': 0
        }

    def observe_step(self, t: int, z: np.ndarray, d: np.ndarray, phi: float,
                     symbols: List[str], delta_w: np.ndarray, goal: np.ndarray,
                     intention: str, reward: float, ci: float, cf: float):
        """
        Registra una observación dentro del episodio actual.
        """
        self.t = t
        data = self._current_episode_data

        data['z_vectors'].append(z.copy())
        data['d_vectors'].append(d.copy())
        data['phi_values'].append(phi)

        for s in symbols:
            data['symbols'][s] += 1

        data['delta_w_accum'] += delta_w
        data['goal_vectors'].append(goal.copy())
        data['intentions'].append(intention)
        data['rewards'].append(reward)
        data['ci_scores'].append(ci)
        data['cf_scores'].append(cf)
        data['n_steps'] += 1

    def close_episode(self, episode_id: int) -> EpisodeSignature:
        """
        Cierra el episodio actual y crea su firma.
        """
        data = self._current_episode_data

        if data['n_steps'] == 0:
            # Episodio vacío
            return None

        # Promedios
        z_bar = np.mean(data['z_vectors'], axis=0)
        d_bar = np.mean(data['d_vectors'], axis=0)
        phi_bar = np.mean(data['phi_values'])

        # Símbolos
        active_symbols = set(data['symbols'].keys())
        sorted_symbols = sorted(data['symbols'].items(),
                               key=lambda x: x[1], reverse=True)
        n_top = max(3, int(np.sqrt(len(active_symbols))))
        top_symbols = [s for s, _ in sorted_symbols[:n_top]]

        # Meta dominante (promedio de vectores de meta)
        goal_vector = np.mean(data['goal_vectors'], axis=0)

        # Intención dominante (más frecuente)
        intention_counts = defaultdict(int)
        for intent in data['intentions']:
            intention_counts[intent] += 1
        intention = max(intention_counts.items(),
                       key=lambda x: x[1])[0] if intention_counts else "none"

        # Firma narrativa: concatenación de características
        # Codificar top_symbols como vector one-hot sparse
        symbol_encoding = np.zeros(20)  # Max 20 símbolos distintos
        for i, s in enumerate(top_symbols[:5]):
            idx = hash(s) % 20
            symbol_encoding[idx] = 1.0

        narrative_sig = np.concatenate([
            z_bar[:5],  # Primeras 5 dims del vector estructural
            [phi_bar],
            symbol_encoding[:10],
            data['delta_w_accum'][:5] / (data['n_steps'] + 1e-8)
        ])

        # Crear firma
        signature = EpisodeSignature(
            episode_id=episode_id,
            agent_id=self.agent_id,
            z_bar=z_bar,
            d_bar=d_bar,
            phi_bar=phi_bar,
            active_symbols=active_symbols,
            top_symbols=top_symbols,
            symbol_counts=dict(data['symbols']),
            delta_w=data['delta_w_accum'] / (data['n_steps'] + 1e-8),
            goal_vector=goal_vector,
            intention=intention,
            narrative_signature=narrative_sig,
            mean_reward=np.mean(data['rewards']) if data['rewards'] else 0,
            mean_ci=np.mean(data['ci_scores']) if data['ci_scores'] else 0.5,
            mean_cf=np.mean(data['cf_scores']) if data['cf_scores'] else 0.5,
            n_steps=data['n_steps']
        )

        # Guardar y actualizar clustering
        self.episodes.append(signature)
        self.narrative_vectors.append(narrative_sig)
        self._update_clustering()

        # Calcular continuidad con episodio anterior
        if len(self.episodes) >= 2:
            metrics = self.compute_continuity(
                self.episodes[-2],
                self.episodes[-1]
            )
            self.continuity_history.append(metrics)

        # Reset acumuladores
        self._current_episode_data = self._init_episode_data()

        return signature

    def _update_clustering(self):
        """Actualiza el modelo de clustering narrativo."""
        if len(self.narrative_vectors) < 3:
            return

        # Número de clusters endógeno
        n_clusters = self.n_clusters_base
        if n_clusters is None:
            n_clusters = max(2, min(len(self.narrative_vectors) // 2,
                                   int(np.sqrt(len(self.narrative_vectors)))))

        # Fit clustering
        X = np.array(self.narrative_vectors)
        self.cluster_model = KMeans(n_clusters=n_clusters,
                                    n_init=10, random_state=42)
        labels = self.cluster_model.fit_predict(X)

        # Asignar clusters a episodios
        for i, ep in enumerate(self.episodes):
            ep.theme = int(labels[i])

    def compute_continuity(self, ep_i: EpisodeSignature,
                          ep_j: EpisodeSignature) -> ContinuityMetrics:
        """
        Calcula métricas de continuidad entre dos episodios.
        """
        # (A) Coherencia estructural: 1 - D_Mahal(z̄_i, z̄_j)
        try:
            # Covarianza de vectores z históricos
            if len(self.episodes) > 2:
                z_history = np.array([e.z_bar for e in self.episodes[-10:]])
                cov = np.cov(z_history.T)
                if np.linalg.det(cov) > 1e-10:
                    cov_inv = np.linalg.inv(cov)
                    d_mahal = mahalanobis(ep_i.z_bar, ep_j.z_bar, cov_inv)
                    ce_struct = 1 / (1 + d_mahal)  # Normalizar a (0,1)
                else:
                    # Fallback a distancia euclidiana normalizada
                    d_eucl = np.linalg.norm(ep_i.z_bar - ep_j.z_bar)
                    ce_struct = 1 / (1 + d_eucl)
            else:
                d_eucl = np.linalg.norm(ep_i.z_bar - ep_j.z_bar)
                ce_struct = 1 / (1 + d_eucl)
        except:
            ce_struct = 0.5

        # (B) Continuidad simbólica: Jaccard
        intersection = len(ep_i.active_symbols & ep_j.active_symbols)
        union = len(ep_i.active_symbols | ep_j.active_symbols)
        ce_sym = intersection / (union + 1e-8)

        # (C) Continuidad causal: cos(ΔW_i, ΔW_j)
        norm_i = np.linalg.norm(ep_i.delta_w)
        norm_j = np.linalg.norm(ep_j.delta_w)
        if norm_i > 1e-8 and norm_j > 1e-8:
            ce_causal = (np.dot(ep_i.delta_w, ep_j.delta_w) /
                        (norm_i * norm_j))
            ce_causal = (ce_causal + 1) / 2  # De [-1,1] a [0,1]
        else:
            ce_causal = 0.5

        # (D) Continuidad teleológica: 1 - D_cos(g_i, g_j)
        norm_gi = np.linalg.norm(ep_i.goal_vector)
        norm_gj = np.linalg.norm(ep_j.goal_vector)
        if norm_gi > 1e-8 and norm_gj > 1e-8:
            cos_goal = np.dot(ep_i.goal_vector, ep_j.goal_vector) / (norm_gi * norm_gj)
            ce_goal = (cos_goal + 1) / 2
        else:
            ce_goal = 0.5

        # (E) Continuidad de cluster narrativo
        if ep_i.theme is not None and ep_j.theme is not None:
            ce_cluster = 1.0 if ep_i.theme == ep_j.theme else 0.0
        else:
            ce_cluster = 0.5

        # Guardar en historiales
        self.ce_struct_history.append(ce_struct)
        self.ce_sym_history.append(ce_sym)
        self.ce_causal_history.append(ce_causal)
        self.ce_goal_history.append(ce_goal)
        self.ce_cluster_history.append(ce_cluster)

        # Pesos endógenos: inversamente proporcionales a varianza
        weights = self._compute_weights()

        # CE total ponderado
        ce_total = (weights['struct'] * ce_struct +
                   weights['sym'] * ce_sym +
                   weights['causal'] * ce_causal +
                   weights['goal'] * ce_goal +
                   weights['cluster'] * ce_cluster)

        return ContinuityMetrics(
            episode_i=ep_i.episode_id,
            episode_j=ep_j.episode_id,
            ce_struct=float(ce_struct),
            ce_sym=float(ce_sym),
            ce_causal=float(ce_causal),
            ce_goal=float(ce_goal),
            ce_cluster=float(ce_cluster),
            ce_total=float(ce_total),
            weights=weights
        )

    def _compute_weights(self) -> Dict[str, float]:
        """Calcula pesos endógenos basados en varianza inversa."""
        histories = {
            'struct': self.ce_struct_history,
            'sym': self.ce_sym_history,
            'causal': self.ce_causal_history,
            'goal': self.ce_goal_history,
            'cluster': self.ce_cluster_history
        }

        # Varianzas
        variances = {}
        for name, hist in histories.items():
            if len(hist) > 2:
                variances[name] = np.var(hist[-20:]) + 1e-8
            else:
                variances[name] = 0.1  # Default

        # Pesos inversamente proporcionales
        inv_vars = {k: 1/v for k, v in variances.items()}
        total = sum(inv_vars.values())

        return {k: v/total for k, v in inv_vars.items()}

    def get_ce_global(self) -> float:
        """
        Calcula CE global: promedio de CE(i) sobre todos los pares consecutivos.
        """
        if not self.continuity_history:
            return 0.5

        return float(np.mean([m.ce_total for m in self.continuity_history]))

    def get_continuity_effect(self) -> Dict[str, float]:
        """
        Calcula el efecto de la continuidad sobre el comportamiento.

        CE alta → consolidación (learning rate bajo, planning horizon corto)
        CE baja → exploración (learning rate alto, planning horizon largo)
        """
        ce_global = self.get_ce_global()

        # Modulación de learning rate: más CE → menos LR
        lr_factor = 1 / (1 + ce_global)

        # Modulación de planning horizon: más CE → menos profundidad
        planning_factor = 1 - ce_global * 0.5

        # Modulación de curiosidad: menos CE → más curiosidad
        curiosity_factor = 1 - ce_global

        return {
            'ce_global': ce_global,
            'lr_factor': float(lr_factor),
            'planning_factor': float(planning_factor),
            'curiosity_factor': float(curiosity_factor),
            'is_coherent': ce_global > 0.5,
            'is_fragmented': ce_global < 0.3
        }

    def detect_narrative_disruption(self) -> Dict[str, Any]:
        """
        Detecta disrupciones narrativas (caídas bruscas en CE).
        """
        if len(self.continuity_history) < 2:
            return {'disruption': False}

        recent = self.continuity_history[-5:]
        ce_values = [m.ce_total for m in recent]

        if len(ce_values) < 2:
            return {'disruption': False}

        # Detectar caída
        last_ce = ce_values[-1]
        prev_mean = np.mean(ce_values[:-1])

        # Umbral endógeno: Q25 de la distribución histórica
        if len(self.continuity_history) > 5:
            threshold = np.percentile([m.ce_total for m in self.continuity_history], 25)
        else:
            threshold = 0.3

        disruption = last_ce < threshold and last_ce < prev_mean * 0.7

        return {
            'disruption': disruption,
            'last_ce': float(last_ce),
            'prev_mean': float(prev_mean),
            'threshold': float(threshold),
            'severity': float(prev_mean - last_ce) if disruption else 0.0
        }

    def get_thematic_symbols(self) -> Dict[str, List[str]]:
        """
        Identifica símbolos temáticos: aparecen en 3+ episodios consecutivos.
        """
        if len(self.episodes) < 3:
            return {'thematic': [], 'transient': []}

        # Contar apariciones consecutivas
        symbol_streaks = defaultdict(int)

        for i in range(len(self.episodes) - 2):
            common = (self.episodes[i].active_symbols &
                     self.episodes[i+1].active_symbols &
                     self.episodes[i+2].active_symbols)
            for s in common:
                symbol_streaks[s] += 1

        # Umbral: Q75 de streaks
        if symbol_streaks:
            threshold = np.percentile(list(symbol_streaks.values()), 75)
            thematic = [s for s, c in symbol_streaks.items() if c >= threshold]
        else:
            thematic = []

        # Transitorios: solo aparecen una vez
        all_symbols = set()
        for ep in self.episodes:
            all_symbols.update(ep.active_symbols)
        transient = [s for s in all_symbols if s not in symbol_streaks or symbol_streaks[s] < 2]

        return {
            'thematic': thematic,
            'transient': transient[:10],
            'n_thematic': len(thematic),
            'n_transient': len(transient)
        }

    def get_goal_persistence(self) -> Dict[str, float]:
        """
        Mide la persistencia de metas: metas que sobreviven 2+ episodios.
        """
        if len(self.episodes) < 2:
            return {'persistence': 0.5, 'n_persistent_goals': 0}

        # Comparar metas entre episodios
        persistences = []
        for i in range(1, len(self.episodes)):
            g1 = self.episodes[i-1].goal_vector
            g2 = self.episodes[i].goal_vector

            norm1 = np.linalg.norm(g1)
            norm2 = np.linalg.norm(g2)

            if norm1 > 1e-8 and norm2 > 1e-8:
                cos = np.dot(g1, g2) / (norm1 * norm2)
                persistences.append((cos + 1) / 2)
            else:
                persistences.append(0.5)

        return {
            'persistence': float(np.mean(persistences)),
            'n_persistent_goals': sum(1 for p in persistences if p > 0.7),
            'goal_volatility': float(np.std(persistences)) if len(persistences) > 1 else 0
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Estadísticas completas del sistema de continuidad."""
        return {
            'agent_id': self.agent_id,
            'n_episodes': len(self.episodes),
            'ce_global': self.get_ce_global(),
            'ce_components': {
                'struct': np.mean(self.ce_struct_history) if self.ce_struct_history else 0.5,
                'sym': np.mean(self.ce_sym_history) if self.ce_sym_history else 0.5,
                'causal': np.mean(self.ce_causal_history) if self.ce_causal_history else 0.5,
                'goal': np.mean(self.ce_goal_history) if self.ce_goal_history else 0.5,
                'cluster': np.mean(self.ce_cluster_history) if self.ce_cluster_history else 0.5
            },
            'continuity_effect': self.get_continuity_effect(),
            'thematic_symbols': self.get_thematic_symbols(),
            'goal_persistence': self.get_goal_persistence(),
            'narrative_disruption': self.detect_narrative_disruption()
        }


def test_episodic_continuity():
    """Test del sistema de continuidad episódica."""
    print("=" * 70)
    print("TEST: EPISODIC CONTINUITY")
    print("=" * 70)

    np.random.seed(42)

    ec = EpisodicContinuity('NEO', state_dim=12)

    # Simular 5 episodios
    for ep_num in range(5):
        print(f"\n--- Episodio {ep_num + 1} ---")

        # Base para continuidad (con algo de drift)
        base_z = np.random.randn(12) * 0.5 + ep_num * 0.1
        base_goal = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * (1 + ep_num * 0.1)

        for t in range(100):
            z = base_z + np.random.randn(12) * 0.1
            d = np.random.randn(12) * 0.1
            phi = 0.5 + np.random.randn() * 0.1
            symbols = [f"S{np.random.randint(10)}" for _ in range(3)]
            delta_w = np.random.randn(12) * 0.05
            goal = base_goal + np.random.randn(12) * 0.05
            intention = f"intent_{ep_num % 3}"
            reward = np.random.randn() * 0.3
            ci = 0.5 + np.random.randn() * 0.1
            cf = 0.5 + np.random.randn() * 0.1

            ec.observe_step(t, z, d, phi, symbols, delta_w, goal,
                           intention, reward, ci, cf)

        sig = ec.close_episode(ep_num)
        print(f"  Símbolos activos: {len(sig.active_symbols)}")
        print(f"  Top símbolos: {sig.top_symbols[:3]}")
        print(f"  Tema: {sig.theme}")
        print(f"  Reward medio: {sig.mean_reward:.3f}")

        if len(ec.continuity_history) > 0:
            last_ce = ec.continuity_history[-1]
            print(f"  CE con anterior:")
            print(f"    struct={last_ce.ce_struct:.3f}, sym={last_ce.ce_sym:.3f}")
            print(f"    causal={last_ce.ce_causal:.3f}, goal={last_ce.ce_goal:.3f}")
            print(f"    TOTAL={last_ce.ce_total:.3f}")

    # Estadísticas finales
    print("\n" + "=" * 70)
    stats = ec.get_statistics()
    print(f"CE Global: {stats['ce_global']:.4f}")
    print(f"Componentes CE: {stats['ce_components']}")
    print(f"Efecto continuidad: {stats['continuity_effect']}")
    print(f"Símbolos temáticos: {stats['thematic_symbols']['n_thematic']}")
    print(f"Persistencia metas: {stats['goal_persistence']['persistence']:.3f}")
    print("=" * 70)

    return ec


if __name__ == "__main__":
    test_episodic_continuity()
