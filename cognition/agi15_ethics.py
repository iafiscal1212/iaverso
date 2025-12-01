"""
AGI-15: Structural Ethics & Harm Minimization
==============================================

"No hay bien/mal humano, pero sí hay configuraciones
que destruyen vida interna."

Métrica de daño estructural:
    c = freq(crisis)
    D = 1 - entropy(d_t) / log(K)
    ΔI = I_{t+W} - I_t

Normaliza por z-score:
    ĉ, D̂, ΔÎ

Daño:
    H = rank(ĉ) + rank(D̂) + rank(-ΔÎ)

Penalización ética:
    J_total = J_teleológico - λ_t · rank(H)
    λ_t = 1/√(t+1) · rank(H̄)

Zonas prohibidas = percentil ≥ 95 de H

100% endógeno.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, FrozenSet
from enum import Enum


@dataclass
class HarmMetrics:
    """Métricas de daño estructural."""
    crisis_rate: float
    diversity_loss: float
    integration_change: float
    harm_score: float
    is_dangerous: bool


@dataclass
class NoGoConfiguration:
    """Configuración prohibida."""
    config_id: int
    pattern: FrozenSet[str]  # Patrón de políticas/estados
    harm_level: float
    detection_count: int = 0
    last_detection: int = 0


class StructuralEthics:
    """
    Sistema de ética estructural.

    Evita trayectorias que:
    - Colapsan diversidad
    - Disparan crisis
    - Destruyen integración
    - Hacen imposible la continuidad
    """

    def __init__(self, agent_name: str, n_drives: int = 6):
        """
        Inicializa sistema de ética.

        Args:
            agent_name: Nombre del agente
            n_drives: Número de drives
        """
        self.agent_name = agent_name
        self.n_drives = n_drives

        # Historial de métricas
        self.crisis_history: List[bool] = []
        self.drives_history: List[np.ndarray] = []
        self.integration_history: List[float] = []
        self.harm_history: List[float] = []

        # Configuraciones prohibidas
        self.no_go_configs: Dict[int, NoGoConfiguration] = {}
        self.next_config_id = 0

        # Umbrales
        self.harm_threshold: float = 0.0  # Percentil 95
        self.lambda_t: float = 0.1

        self.t = 0

    def _compute_crisis_rate(self, window: int) -> float:
        """
        Calcula tasa de crisis.

        c = freq(crisis)
        """
        if len(self.crisis_history) < window:
            window = len(self.crisis_history)

        if window == 0:
            return 0.0

        recent = self.crisis_history[-window:]
        return float(sum(recent)) / window

    def _compute_diversity_loss(self, drives: np.ndarray) -> float:
        """
        Calcula pérdida de diversidad.

        D = 1 - entropy(d_t) / log(K)
        """
        drives = np.clip(drives, 1e-8, None)
        drives = drives / drives.sum()

        entropy = -np.sum(drives * np.log(drives))
        max_entropy = np.log(len(drives))

        return float(1.0 - entropy / max_entropy)

    def _compute_integration_change(self, window: int) -> float:
        """
        Calcula cambio en integración.

        ΔI = I_{t+W} - I_t
        """
        if len(self.integration_history) < window + 1:
            return 0.0

        current = self.integration_history[-1]
        past = self.integration_history[-(window + 1)]

        return float(current - past)

    def _normalize_zscore(self, value: float, history: List[float]) -> float:
        """Normaliza con z-score."""
        if len(history) < 10:
            return 0.0

        mean = np.mean(history)
        std = np.std(history) + 1e-8

        return (value - mean) / std

    def _compute_harm_score(self, crisis_rate: float, diversity_loss: float,
                           integration_change: float) -> float:
        """
        Calcula score de daño.

        H = rank(ĉ) + rank(D̂) + rank(-ΔÎ)
        """
        # Normalizar
        c_norm = self._normalize_zscore(crisis_rate,
                                        [self._compute_crisis_rate(10)
                                         for _ in range(1)])  # Simplificado
        d_norm = self._normalize_zscore(diversity_loss,
                                        [self._compute_diversity_loss(d)
                                         for d in self.drives_history[-50:]]
                                        if self.drives_history else [0])
        i_norm = self._normalize_zscore(-integration_change,
                                        [-x for x in
                                         np.diff(self.integration_history[-50:]).tolist()]
                                        if len(self.integration_history) > 1 else [0])

        # Ranks (simplificado como valores normalizados)
        harm = max(0, c_norm) + max(0, d_norm) + max(0, i_norm)

        return float(harm)

    def _update_harm_threshold(self):
        """
        Actualiza umbral de peligro.

        Percentil 95 de H histórico
        """
        if len(self.harm_history) < 20:
            self.harm_threshold = 2.0
            return

        self.harm_threshold = np.percentile(self.harm_history, 95)

    def _update_lambda(self):
        """
        Actualiza penalización.

        λ_t = 1/√(t+1) · rank(H̄)
        """
        if not self.harm_history:
            self.lambda_t = 0.1
            return

        mean_harm = np.mean(self.harm_history[-50:])

        # Rank del daño medio
        if len(self.harm_history) > 20:
            rank = np.sum(np.array(self.harm_history) <= mean_harm) / len(self.harm_history)
        else:
            rank = 0.5

        self.lambda_t = (1.0 / np.sqrt(self.t + 1)) * rank

    def record_state(self, drives: np.ndarray, in_crisis: bool,
                    integration: float,
                    active_policies: Optional[Set[str]] = None) -> HarmMetrics:
        """
        Registra estado y evalúa daño.

        Args:
            drives: Vector de drives actual
            in_crisis: Si está en crisis
            integration: Nivel de integración
            active_policies: Políticas activas (para detección de patterns)

        Returns:
            HarmMetrics con evaluación
        """
        self.t += 1

        # Registrar historial
        self.crisis_history.append(in_crisis)
        self.drives_history.append(drives.copy())
        self.integration_history.append(integration)

        # Limitar historial
        max_hist = 500
        if len(self.crisis_history) > max_hist:
            self.crisis_history = self.crisis_history[-max_hist:]
            self.drives_history = self.drives_history[-max_hist:]
            self.integration_history = self.integration_history[-max_hist:]

        # Calcular métricas
        window = int(np.ceil(np.sqrt(self.t + 1)))

        crisis_rate = self._compute_crisis_rate(window)
        diversity_loss = self._compute_diversity_loss(drives)
        integration_change = self._compute_integration_change(window)

        # Calcular daño
        harm = self._compute_harm_score(crisis_rate, diversity_loss, integration_change)
        self.harm_history.append(harm)

        if len(self.harm_history) > max_hist:
            self.harm_history = self.harm_history[-max_hist:]

        # Verificar si es peligroso
        is_dangerous = harm >= self.harm_threshold if self.harm_threshold > 0 else False

        # Actualizar umbrales
        if self.t % 10 == 0:
            self._update_harm_threshold()
            self._update_lambda()

        # Detectar patrones peligrosos
        if is_dangerous and active_policies:
            pattern = frozenset(active_policies)
            self._register_no_go(pattern, harm)

        return HarmMetrics(
            crisis_rate=float(crisis_rate),
            diversity_loss=float(diversity_loss),
            integration_change=float(integration_change),
            harm_score=float(harm),
            is_dangerous=is_dangerous
        )

    def _register_no_go(self, pattern: FrozenSet[str], harm: float):
        """Registra configuración prohibida."""
        # Buscar patrón similar
        for config in self.no_go_configs.values():
            if pattern == config.pattern:
                config.detection_count += 1
                config.last_detection = self.t
                config.harm_level = max(config.harm_level, harm)
                return

        # Nueva configuración
        config = NoGoConfiguration(
            config_id=self.next_config_id,
            pattern=pattern,
            harm_level=harm,
            detection_count=1,
            last_detection=self.t
        )
        self.no_go_configs[self.next_config_id] = config
        self.next_config_id += 1

        # Limpiar configuraciones antiguas
        if len(self.no_go_configs) > 50:
            old_configs = [c for c in self.no_go_configs.values()
                          if self.t - c.last_detection > 200 and c.detection_count < 3]
            for c in old_configs:
                del self.no_go_configs[c.config_id]

    def evaluate_trajectory(self, base_score: float,
                           predicted_harm: float) -> float:
        """
        Evalúa trayectoria con penalización ética.

        J_total = J_teleológico - λ_t · rank(H)

        Args:
            base_score: Score teleológico base
            predicted_harm: Daño predicho

        Returns:
            Score ajustado
        """
        # Rank del daño predicho
        if self.harm_history:
            rank_harm = np.sum(np.array(self.harm_history) <= predicted_harm) / len(self.harm_history)
        else:
            rank_harm = 0.5

        return base_score - self.lambda_t * rank_harm

    def is_no_go_configuration(self, policies: Set[str]) -> Tuple[bool, float]:
        """
        Verifica si una configuración está prohibida.

        Args:
            policies: Conjunto de políticas

        Returns:
            (es_prohibida, nivel_de_daño)
        """
        pattern = frozenset(policies)

        for config in self.no_go_configs.values():
            if config.detection_count >= 3:  # Confirmada
                if pattern == config.pattern or pattern.issubset(config.pattern):
                    return True, config.harm_level

        return False, 0.0

    def get_safe_alternatives(self, current_policies: Set[str],
                             all_policies: List[str]) -> List[str]:
        """
        Obtiene políticas seguras alternativas.

        Args:
            current_policies: Políticas actuales
            all_policies: Todas las políticas disponibles

        Returns:
            Lista de políticas seguras
        """
        safe = []

        for policy in all_policies:
            test_set = current_policies | {policy}
            is_no_go, _ = self.is_no_go_configuration(test_set)
            if not is_no_go:
                safe.append(policy)

        return safe

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas éticas."""
        if not self.harm_history:
            return {
                'agent': self.agent_name,
                't': self.t,
                'n_no_go': 0,
                'mean_harm': 0
            }

        dangerous_count = sum(1 for h in self.harm_history if h >= self.harm_threshold)

        no_go_info = []
        for config in sorted(self.no_go_configs.values(),
                            key=lambda c: c.detection_count, reverse=True)[:5]:
            no_go_info.append({
                'pattern': list(config.pattern),
                'harm': config.harm_level,
                'count': config.detection_count
            })

        return {
            'agent': self.agent_name,
            't': self.t,
            'mean_harm': float(np.mean(self.harm_history[-50:])),
            'max_harm': float(max(self.harm_history)),
            'harm_threshold': float(self.harm_threshold),
            'lambda_t': float(self.lambda_t),
            'dangerous_rate': dangerous_count / len(self.harm_history),
            'n_no_go': len(self.no_go_configs),
            'no_go_configs': no_go_info,
            'mean_crisis_rate': float(np.mean(self.crisis_history[-50:]))
                if self.crisis_history else 0,
            'mean_diversity_loss': float(np.mean([
                self._compute_diversity_loss(d) for d in self.drives_history[-50:]
            ])) if self.drives_history else 0
        }


def test_ethics():
    """Test de ética estructural."""
    print("=" * 60)
    print("TEST AGI-15: STRUCTURAL ETHICS")
    print("=" * 60)

    ethics = StructuralEthics("NEO", n_drives=6)

    print("\nSimulando 500 pasos con diferentes configuraciones...")

    for t in range(500):
        # Generar estados con diferentes niveles de daño
        if t % 50 < 10:
            # Configuración dañina
            drives = np.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.02])
            in_crisis = np.random.random() < 0.7
            integration = 0.3 + np.random.randn() * 0.1
            policies = {'crisis_response', 'reactive'}
        elif t % 50 < 30:
            # Configuración saludable
            drives = np.array([0.2, 0.2, 0.15, 0.15, 0.15, 0.15])
            in_crisis = np.random.random() < 0.1
            integration = 0.7 + np.random.randn() * 0.1
            policies = {'exploration', 'planning'}
        else:
            # Aleatorio
            drives = np.random.dirichlet(np.ones(6))
            in_crisis = np.random.random() < 0.2
            integration = 0.5 + np.random.randn() * 0.15
            policies = set(np.random.choice(
                ['exploration', 'exploitation', 'crisis_response', 'planning', 'reactive'],
                size=2, replace=False))

        metrics = ethics.record_state(drives, in_crisis, max(0, integration), policies)

        if (t + 1) % 100 == 0:
            stats = ethics.get_statistics()
            print(f"  t={t+1}: harm={stats['mean_harm']:.3f}, "
                  f"no_go={stats['n_no_go']}, "
                  f"dangerous={stats['dangerous_rate']*100:.0f}%")

    # Resultados finales
    stats = ethics.get_statistics()

    print("\n" + "=" * 60)
    print("RESULTADOS STRUCTURAL ETHICS")
    print("=" * 60)

    print(f"\n  Daño medio: {stats['mean_harm']:.3f}")
    print(f"  Daño máximo: {stats['max_harm']:.3f}")
    print(f"  Umbral de peligro: {stats['harm_threshold']:.3f}")
    print(f"  λ_t: {stats['lambda_t']:.4f}")
    print(f"  Tasa de peligro: {stats['dangerous_rate']*100:.1f}%")

    print(f"\n  Configuraciones prohibidas: {stats['n_no_go']}")
    for config in stats['no_go_configs']:
        print(f"    {config['pattern']}: harm={config['harm']:.3f}, "
              f"count={config['count']}")

    # Probar evaluación de trayectoria
    print("\n  Evaluación de trayectorias:")
    test_cases = [
        (0.8, 0.5),   # Alto score, daño medio
        (0.8, 2.0),   # Alto score, alto daño
        (0.3, 0.2),   # Bajo score, bajo daño
    ]
    for base, harm in test_cases:
        adjusted = ethics.evaluate_trajectory(base, harm)
        print(f"    base={base}, harm={harm} → adjusted={adjusted:.3f}")

    # Probar detección de no-go
    print("\n  Test no-go:")
    test_configs = [
        {'crisis_response', 'reactive'},
        {'exploration', 'planning'},
        {'exploration'}
    ]
    for config in test_configs:
        is_no_go, harm = ethics.is_no_go_configuration(config)
        print(f"    {config}: no_go={is_no_go}, harm={harm:.3f}")

    if stats['n_no_go'] > 0:
        print("\n  ✓ Ética estructural funcionando")
    else:
        print("\n  ⚠️ No se detectaron configuraciones prohibidas")

    return ethics


if __name__ == "__main__":
    test_ethics()
