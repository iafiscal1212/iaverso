"""
AGI-7: Generalización Cruz-Mundo
================================

Que conceptos/skills/metas funcionen no solo en un régimen,
sino en múltiples mundos (WORLD-1, WORLD-2, etc.).

Índice de generalización:
    Gen_s = 1 - var(g_s) / var_max

donde:
    g_s = (ΔV^(1), ..., ΔV^(R))

Uso:
    w_s = rank(score_s) + rank(Gen_s)

El agente favorece skills que:
- Ya le han funcionado
- Y funcionan en muchos regímenes

100% endógeno.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class WorldRegime(Enum):
    """Regímenes del mundo detectados endógenamente."""
    STABLE = "stable"
    VOLATILE = "volatile"
    GROWTH = "growth"
    DECLINE = "decline"
    CRISIS = "crisis"
    EXPLORATION = "exploration"
    UNKNOWN = "unknown"


@dataclass
class RegimePerformance:
    """Rendimiento de un skill/concepto en un régimen."""
    regime: WorldRegime
    delta_V_samples: List[float] = field(default_factory=list)
    mean_delta_V: float = 0.0
    std_delta_V: float = 0.0
    n_samples: int = 0


@dataclass
class GeneralizableItem:
    """Item (skill/concepto) con métricas de generalización."""
    item_id: int
    item_type: str  # 'skill', 'concept', 'goal'
    regime_performance: Dict[WorldRegime, RegimePerformance]
    generalization_index: float = 0.0
    combined_weight: float = 0.0
    total_score: float = 0.0


class CrossWorldGeneralization:
    """
    Sistema de generalización entre mundos/regímenes.

    Mide qué tan bien funcionan skills y conceptos
    en diferentes condiciones del mundo.
    """

    def __init__(self, agent_name: str):
        """
        Inicializa sistema de generalización.

        Args:
            agent_name: Nombre del agente
        """
        self.agent_name = agent_name

        # Items rastreados (skills, conceptos, metas)
        self.items: Dict[int, GeneralizableItem] = {}
        self.next_item_id = 0

        # Historial de regímenes
        self.regime_history: List[WorldRegime] = []
        self.current_regime: WorldRegime = WorldRegime.UNKNOWN

        # Historial para detección de régimen
        self.stability_history: List[float] = []
        self.growth_history: List[float] = []
        self.volatility_history: List[float] = []

        self.t = 0

    def _detect_regime(self, z: np.ndarray, phi: np.ndarray,
                       delta_V: float) -> WorldRegime:
        """
        Detecta el régimen actual del mundo endógenamente.

        Basado en estabilidad, crecimiento y volatilidad.
        """
        # Calcular métricas
        stability = 1.0 / (1.0 + np.var(z))
        growth = delta_V
        volatility = np.std(phi) if len(phi) > 0 else 0

        # Registrar
        self.stability_history.append(stability)
        self.growth_history.append(growth)
        self.volatility_history.append(volatility)

        # Limitar historial
        max_hist = 100
        if len(self.stability_history) > max_hist:
            self.stability_history = self.stability_history[-max_hist:]
            self.growth_history = self.growth_history[-max_hist:]
            self.volatility_history = self.volatility_history[-max_hist:]

        if len(self.stability_history) < 10:
            return WorldRegime.UNKNOWN

        # Percentiles para clasificación endógena
        stab_pctl = np.sum(np.array(self.stability_history) <= stability) / len(self.stability_history)
        growth_pctl = np.sum(np.array(self.growth_history) <= growth) / len(self.growth_history)
        vol_pctl = np.sum(np.array(self.volatility_history) <= volatility) / len(self.volatility_history)

        # Clasificar régimen
        if vol_pctl > 0.8:
            if growth_pctl < 0.3:
                return WorldRegime.CRISIS
            else:
                return WorldRegime.VOLATILE
        elif stab_pctl > 0.7:
            if growth_pctl > 0.6:
                return WorldRegime.GROWTH
            elif growth_pctl < 0.4:
                return WorldRegime.DECLINE
            else:
                return WorldRegime.STABLE
        elif growth_pctl > 0.7:
            return WorldRegime.EXPLORATION
        else:
            return WorldRegime.STABLE

    def register_item(self, item_type: str) -> int:
        """
        Registra un nuevo item para tracking.

        Args:
            item_type: Tipo ('skill', 'concept', 'goal')

        Returns:
            ID del item
        """
        item_id = self.next_item_id
        self.next_item_id += 1

        self.items[item_id] = GeneralizableItem(
            item_id=item_id,
            item_type=item_type,
            regime_performance={r: RegimePerformance(regime=r) for r in WorldRegime}
        )

        return item_id

    def record_performance(self, item_id: int, delta_V: float,
                          z: np.ndarray, phi: np.ndarray):
        """
        Registra rendimiento de un item en el régimen actual.

        Args:
            item_id: ID del item
            delta_V: Cambio en valor
            z: Estado actual de drives
            phi: Estado fenomenológico
        """
        self.t += 1

        if item_id not in self.items:
            return

        # Detectar régimen actual
        regime = self._detect_regime(z, phi, delta_V)
        self.current_regime = regime
        self.regime_history.append(regime)

        # Registrar rendimiento
        item = self.items[item_id]
        perf = item.regime_performance[regime]

        perf.delta_V_samples.append(delta_V)
        perf.n_samples += 1

        # Limitar muestras
        if len(perf.delta_V_samples) > 100:
            perf.delta_V_samples = perf.delta_V_samples[-100:]

        # Actualizar estadísticas
        perf.mean_delta_V = float(np.mean(perf.delta_V_samples))
        perf.std_delta_V = float(np.std(perf.delta_V_samples))

        # Recalcular generalización
        self._update_generalization(item_id)

    def _update_generalization(self, item_id: int):
        """
        Actualiza índice de generalización.

        Gen_s = 1 - var(g_s) / var_max
        donde g_s = (ΔV^(1), ..., ΔV^(R))
        """
        item = self.items[item_id]

        # Obtener rendimientos por régimen
        performances = []
        for regime, perf in item.regime_performance.items():
            if perf.n_samples > 0:
                performances.append(perf.mean_delta_V)

        if len(performances) < 2:
            item.generalization_index = 0.5
            return

        # Calcular varianza
        var_g = np.var(performances)

        # Varianza máxima endógena (basada en rango de valores)
        range_g = max(performances) - min(performances) if performances else 1.0
        var_max = (range_g ** 2) / 4  # Varianza máxima para distribución uniforme

        # Índice de generalización
        if var_max > 0:
            item.generalization_index = float(1.0 - min(1.0, var_g / var_max))
        else:
            item.generalization_index = 1.0

        # Calcular peso combinado
        # w_s = rank(score_s) + rank(Gen_s)
        self._update_combined_weights()

    def _update_combined_weights(self):
        """
        Actualiza pesos combinados de todos los items.

        w_s = rank(score_s) + rank(Gen_s)
        """
        if not self.items:
            return

        # Calcular scores (media de delta_V sobre todos los regímenes)
        scores = {}
        gen_indices = {}

        for item_id, item in self.items.items():
            mean_perf = []
            for perf in item.regime_performance.values():
                if perf.n_samples > 0:
                    mean_perf.append(perf.mean_delta_V)

            scores[item_id] = np.mean(mean_perf) if mean_perf else 0.0
            gen_indices[item_id] = item.generalization_index

        # Calcular ranks
        score_values = list(scores.values())
        gen_values = list(gen_indices.values())
        item_ids = list(scores.keys())

        score_ranks = {item_ids[i]: np.sum(np.array(score_values) <= score_values[i])
                      for i in range(len(item_ids))}
        gen_ranks = {item_ids[i]: np.sum(np.array(gen_values) <= gen_values[i])
                    for i in range(len(item_ids))}

        # Combinar
        for item_id in self.items:
            self.items[item_id].total_score = scores[item_id]
            self.items[item_id].combined_weight = float(
                score_ranks[item_id] + gen_ranks[item_id]
            )

    def get_item_preference(self, item_id: int) -> float:
        """
        Obtiene preferencia normalizada por un item.

        Basado en peso combinado.
        """
        if item_id not in self.items:
            return 0.0

        total_weight = sum(item.combined_weight for item in self.items.values())
        if total_weight == 0:
            return 1.0 / len(self.items)

        return self.items[item_id].combined_weight / total_weight

    def get_best_items(self, n: int = 5, item_type: Optional[str] = None) -> List[int]:
        """
        Obtiene los n mejores items por peso combinado.

        Args:
            n: Número de items
            item_type: Filtrar por tipo (opcional)

        Returns:
            Lista de IDs de items
        """
        filtered = [
            (item_id, item) for item_id, item in self.items.items()
            if item_type is None or item.item_type == item_type
        ]

        sorted_items = sorted(filtered, key=lambda x: x[1].combined_weight, reverse=True)
        return [item_id for item_id, _ in sorted_items[:n]]

    def get_regime_distribution(self) -> Dict[str, float]:
        """Obtiene distribución de regímenes visitados."""
        if not self.regime_history:
            return {}

        counts = {}
        for regime in self.regime_history:
            counts[regime.value] = counts.get(regime.value, 0) + 1

        total = len(self.regime_history)
        return {k: v / total for k, v in counts.items()}

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas de generalización."""
        if not self.items:
            return {
                'agent': self.agent_name,
                't': self.t,
                'n_items': 0,
                'current_regime': self.current_regime.value
            }

        gen_indices = [item.generalization_index for item in self.items.values()]
        scores = [item.total_score for item in self.items.values()]

        return {
            'agent': self.agent_name,
            't': self.t,
            'n_items': len(self.items),
            'current_regime': self.current_regime.value,
            'regime_distribution': self.get_regime_distribution(),
            'mean_generalization': float(np.mean(gen_indices)),
            'std_generalization': float(np.std(gen_indices)),
            'mean_score': float(np.mean(scores)),
            'best_items': self.get_best_items(3),
            'most_generalizable': self._get_most_generalizable(3)
        }

    def _get_most_generalizable(self, n: int) -> List[int]:
        """Obtiene los n items más generalizables."""
        sorted_items = sorted(
            self.items.items(),
            key=lambda x: x[1].generalization_index,
            reverse=True
        )
        return [item_id for item_id, _ in sorted_items[:n]]


def test_generalization():
    """Test de generalización cruz-mundo."""
    print("=" * 60)
    print("TEST AGI-7: GENERALIZACIÓN CRUZ-MUNDO")
    print("=" * 60)

    gen = CrossWorldGeneralization("NEO")

    # Registrar algunos items
    skill_ids = [gen.register_item('skill') for _ in range(5)]
    concept_ids = [gen.register_item('concept') for _ in range(3)]

    print(f"\nRegistrados {len(skill_ids)} skills y {len(concept_ids)} conceptos")
    print("Simulando 500 pasos en diferentes regímenes...")

    for t in range(500):
        # Simular diferentes regímenes
        if t < 100:
            # Régimen estable
            z = np.ones(6) / 6 + np.random.randn(6) * 0.01
            phi = np.array([0.5, 0.5, 0.5, 0.5, 0.5]) + np.random.randn(5) * 0.05
            base_delta_V = 0.1
        elif t < 200:
            # Régimen de crecimiento
            z = np.array([0.3, 0.1, 0.2, 0.2, 0.1, 0.1]) + np.random.randn(6) * 0.02
            phi = np.array([0.7, 0.6, 0.5, 0.4, 0.8]) + np.random.randn(5) * 0.1
            base_delta_V = 0.3
        elif t < 300:
            # Régimen volátil
            z = np.random.random(6)
            z = z / z.sum()
            phi = np.random.random(5) * 0.5
            base_delta_V = np.random.randn() * 0.2
        elif t < 400:
            # Régimen de crisis
            z = np.array([0.05, 0.05, 0.1, 0.1, 0.35, 0.35]) + np.random.randn(6) * 0.03
            phi = np.array([0.2, 0.3, 0.1, 0.2, 0.1]) + np.random.randn(5) * 0.15
            base_delta_V = -0.1
        else:
            # Régimen de exploración
            z = np.array([0.4, 0.3, 0.1, 0.1, 0.05, 0.05]) + np.random.randn(6) * 0.02
            phi = np.array([0.6, 0.8, 0.7, 0.5, 0.6]) + np.random.randn(5) * 0.1
            base_delta_V = 0.2

        z = np.clip(z, 0.01, None)
        z = z / z.sum()

        # Registrar rendimiento de cada skill
        for i, skill_id in enumerate(skill_ids):
            # Algunos skills funcionan mejor en ciertos regímenes
            if i == 0:  # Skill generalista
                delta_V = base_delta_V + np.random.randn() * 0.05
            elif i == 1:  # Skill especializado en estable
                delta_V = base_delta_V * (2.0 if t < 100 else 0.5) + np.random.randn() * 0.05
            elif i == 2:  # Skill especializado en crecimiento
                delta_V = base_delta_V * (2.0 if 100 <= t < 200 else 0.5) + np.random.randn() * 0.05
            else:  # Skills aleatorios
                delta_V = base_delta_V * np.random.uniform(0.5, 1.5) + np.random.randn() * 0.05

            gen.record_performance(skill_id, delta_V, z, phi)

        # Conceptos
        for concept_id in concept_ids:
            delta_V = base_delta_V * np.random.uniform(0.8, 1.2) + np.random.randn() * 0.05
            gen.record_performance(concept_id, delta_V, z, phi)

        if (t + 1) % 100 == 0:
            stats = gen.get_statistics()
            print(f"  t={t+1}: régimen={stats['current_regime']}, "
                  f"mean_gen={stats['mean_generalization']:.3f}")

    # Resultados finales
    stats = gen.get_statistics()

    print("\n" + "=" * 60)
    print("RESULTADOS GENERALIZACIÓN")
    print("=" * 60)

    print(f"\n  Items rastreados: {stats['n_items']}")
    print(f"  Generalización media: {stats['mean_generalization']:.3f}")
    print(f"  Score medio: {stats['mean_score']:.3f}")

    print("\n  Distribución de regímenes:")
    for regime, freq in stats['regime_distribution'].items():
        print(f"    {regime}: {freq*100:.1f}%")

    print("\n  Items más generalizables:")
    for item_id in stats['most_generalizable']:
        item = gen.items[item_id]
        print(f"    Item {item_id} ({item.item_type}): "
              f"gen={item.generalization_index:.3f}, "
              f"weight={item.combined_weight:.3f}")

    if stats['mean_generalization'] > 0.3:
        print("\n  ✓ Generalización cruz-mundo funcionando")
    else:
        print("\n  ⚠️ Baja generalización detectada")

    return gen


if __name__ == "__main__":
    test_generalization()
