"""
AGI-18: Auto-Reconfiguración Reflexiva v2
==========================================

"Cambiar cómo funciono en respuesta a cómo funcioné."

Pesos de módulos:
    w_t = [w_t^SelfModel, w_t^ToM, w_t^Norms, ..., w_t^Meta]
    w_t^total = 1

Utilidad atribuida:
    U_m = corr(activación_m, r_t)

Gradiente de reconfiguración:
    Δw_m = η · (U_m - median(U))
    η = 1/√(t+1)

Nueva configuración:
    w_{t+1} = softmax(log(w_t) + Δw)

Entropía de configuración:
    H_conf = -Σ w_m log(w_m)

Restricciones:
    w_m ∈ [w_min, w_max]
    w_min = 0.05, w_max = 0.4 (endógenos de var(U))

v2 Improvements:
- CF and CI scores integrated as utility signals
- Modules with high CF/CI correlation get boosted weights
- Causal effectiveness becomes a driver

100% endógeno.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .agi_dynamic_constants import (
    L_t, max_history, adaptive_learning_rate, adaptive_momentum
)


def weight_bounds(U_history: List[float], t: int) -> Tuple[float, float]:
    """
    Calcula límites de pesos endógenos.

    w_min = 0.01 + 0.04 / √(var(U) + 1)
    w_max = 0.3 + 0.1 / √(t + 1)
    """
    if len(U_history) < 5:
        return 0.05, 0.4

    var_U = np.var(U_history[-100:])
    w_min = 0.01 + 0.04 / np.sqrt(var_U + 1)
    w_max = 0.3 + 0.1 / np.sqrt(t + 1)

    return float(np.clip(w_min, 0.01, 0.1)), float(np.clip(w_max, 0.25, 0.5))


@dataclass
class ModuleState:
    """Estado de un módulo cognitivo."""
    module_id: str
    weight: float
    activation_history: List[float] = field(default_factory=list)
    utility_correlation: float = 0.0
    gradient: float = 0.0
    n_updates: int = 0


@dataclass
class ReconfigurationState:
    """Estado de reconfiguración."""
    t: int
    module_weights: Dict[str, float]
    configuration_entropy: float
    total_reconfigurations: int
    most_weighted_module: str
    least_weighted_module: str
    is_stable: bool


class ReflectiveReconfiguration:
    """
    Sistema de auto-reconfiguración reflexiva.

    Ajusta los pesos de los módulos cognitivos
    basándose en su contribución a la utilidad.
    """

    # Módulos cognitivos del sistema
    DEFAULT_MODULES = [
        'self_model',      # AGI-4
        'theory_of_mind',  # AGI-5
        'planning',        # AGI-6
        'value',           # AGI-7
        'adaptation',      # AGI-8
        'equilibrium',     # AGI-10
        'norms',           # AGI-12
        'curiosity',       # AGI-13
        'uncertainty',     # AGI-14
        'ethics',          # AGI-15
        'meta_rules'       # AGI-16
    ]

    def __init__(self, agent_name: str, module_names: List[str] = None):
        """
        Inicializa sistema de reconfiguración.

        Args:
            agent_name: Nombre del agente
            module_names: Lista de módulos (usa default si None)
        """
        self.agent_name = agent_name
        self.module_names = module_names or self.DEFAULT_MODULES

        # Inicializar módulos con pesos uniformes
        n_modules = len(self.module_names)
        initial_weight = 1.0 / n_modules

        self.modules: Dict[str, ModuleState] = {
            name: ModuleState(
                module_id=name,
                weight=initial_weight
            )
            for name in self.module_names
        }

        # Historial de recompensas
        self.reward_history: List[float] = []

        # Historial de configuraciones
        self.weight_history: List[Dict[str, float]] = []

        # Contadores
        self.n_reconfigurations = 0
        self.stability_counter = 0

        # v2: CF/CI integration
        self.cf_score_history: List[float] = []
        self.ci_score_history: List[float] = []
        self.cf_ci_correlations: Dict[str, List[float]] = {
            name: [] for name in self.module_names
        }

        self.t = 0

    def _compute_utility_correlations(self):
        """
        Calcula correlación de cada módulo con utilidad.

        U_m = corr(activación_m, r_t)
        """
        min_samples = L_t(self.t)
        if len(self.reward_history) < min_samples:
            return

        window = min(max_history(self.t), len(self.reward_history))
        rewards = np.array(self.reward_history[-window:])

        for module in self.modules.values():
            if len(module.activation_history) < min_samples:
                continue

            activations = np.array(module.activation_history[-window:])

            # Alinear longitudes
            min_len = min(len(activations), len(rewards))
            if min_len < min_samples:
                continue

            activations = activations[-min_len:]
            r = rewards[-min_len:]

            # Correlación
            if np.std(activations) > 0 and np.std(r) > 0:
                corr = np.corrcoef(activations, r)[0, 1]
                module.utility_correlation = float(corr) if not np.isnan(corr) else 0.0

    def _compute_gradients(self):
        """
        Calcula gradientes de reconfiguración.

        Δw_m = η · (U_m - median(U)) + η_cf · CF_CI_corr_m

        v2: Adds boost from CF/CI correlation.
        """
        correlations = [m.utility_correlation for m in self.modules.values()]

        if not correlations:
            return

        median_corr = np.median(correlations)
        eta = adaptive_learning_rate(self.t, 1.0)

        for module in self.modules.values():
            # Base gradient from reward correlation
            base_gradient = eta * (module.utility_correlation - median_corr)

            # v2: CF/CI boost
            cf_ci_boost = 0.0
            if self.cf_ci_correlations[module.module_id]:
                # Mean CF/CI correlation for this module
                mean_cf_ci_corr = np.mean(self.cf_ci_correlations[module.module_id][-20:])
                # Boost scale: endogenous from variance of correlations
                all_cf_ci = [np.mean(corrs[-20:]) if corrs else 0
                            for corrs in self.cf_ci_correlations.values()]
                if len(all_cf_ci) > 1:
                    median_cf_ci = np.median(all_cf_ci)
                    # Boost if above median
                    cf_ci_boost = eta * 0.5 * (mean_cf_ci_corr - median_cf_ci)

            module.gradient = base_gradient + cf_ci_boost

    def _apply_reconfiguration(self):
        """
        Aplica reconfiguración.

        w_{t+1} = softmax(log(w_t) + Δw)
        """
        # Calcular límites endógenos
        w_min, w_max = weight_bounds(self.reward_history, self.t)

        # Log weights + gradientes
        log_weights = []
        for name in self.module_names:
            module = self.modules[name]
            log_w = np.log(module.weight + 1e-8)
            new_log_w = log_w + module.gradient
            log_weights.append(new_log_w)

        # Softmax
        log_weights = np.array(log_weights)
        max_log = np.max(log_weights)
        exp_weights = np.exp(log_weights - max_log)
        new_weights = exp_weights / np.sum(exp_weights)

        # Aplicar límites
        new_weights = np.clip(new_weights, w_min, w_max)
        new_weights /= np.sum(new_weights)

        # Detectar cambio significativo
        old_weights = np.array([self.modules[name].weight for name in self.module_names])
        weight_change = np.linalg.norm(new_weights - old_weights)

        # Threshold endógeno
        change_threshold = 0.01 + 0.05 / np.sqrt(self.t + 1)

        if weight_change > change_threshold:
            self.n_reconfigurations += 1
            self.stability_counter = 0
        else:
            self.stability_counter += 1

        # Actualizar pesos
        for i, name in enumerate(self.module_names):
            self.modules[name].weight = float(new_weights[i])
            self.modules[name].n_updates += 1

        # Guardar configuración
        self.weight_history.append({
            name: self.modules[name].weight for name in self.module_names
        })

        if len(self.weight_history) > max_history(self.t):
            self.weight_history = self.weight_history[-max_history(self.t):]

    def record_activations(
        self,
        module_activations: Dict[str, float],
        reward: float,
        cf_score: Optional[float] = None,
        ci_score: Optional[float] = None
    ):
        """
        Registra activaciones de módulos y recompensa.

        v2: Also records CF/CI scores for causal utility integration.

        Args:
            module_activations: {nombre_módulo: activación}
            reward: Recompensa obtenida
            cf_score: Optional CF score for this step
            ci_score: Optional CI score for this step
        """
        self.t += 1
        max_hist = max_history(self.t)

        # Registrar recompensa
        self.reward_history.append(reward)
        if len(self.reward_history) > max_hist:
            self.reward_history = self.reward_history[-max_hist:]

        # v2: Record CF/CI scores
        if cf_score is not None:
            self.cf_score_history.append(cf_score)
            if len(self.cf_score_history) > max_hist:
                self.cf_score_history = self.cf_score_history[-max_hist:]

        if ci_score is not None:
            self.ci_score_history.append(ci_score)
            if len(self.ci_score_history) > max_hist:
                self.ci_score_history = self.ci_score_history[-max_hist:]

        # Registrar activaciones
        for name, activation in module_activations.items():
            if name in self.modules:
                self.modules[name].activation_history.append(activation)
                if len(self.modules[name].activation_history) > max_hist:
                    self.modules[name].activation_history = \
                        self.modules[name].activation_history[-max_hist:]

        # Reconfigurar periódicamente
        update_freq = max(10, L_t(self.t))
        if self.t % update_freq == 0:
            self._compute_utility_correlations()
            self._compute_cf_ci_correlations()  # v2
            self._compute_gradients()
            self._apply_reconfiguration()

    def _compute_cf_ci_correlations(self):
        """
        v2: Compute correlation of each module with CF/CI scores.

        Modules that correlate with high CF/CI get boosted.
        """
        min_samples = L_t(self.t)

        # Need both CF and CI history
        if len(self.cf_score_history) < min_samples or len(self.ci_score_history) < min_samples:
            return

        window = min(max_history(self.t), len(self.cf_score_history))

        # Combined causal score: CF * CI
        cf_scores = np.array(self.cf_score_history[-window:])
        ci_scores = np.array(self.ci_score_history[-window:])
        causal_scores = cf_scores * ci_scores

        for module in self.modules.values():
            if len(module.activation_history) < min_samples:
                continue

            activations = np.array(module.activation_history[-window:])

            # Align lengths
            min_len = min(len(activations), len(causal_scores))
            if min_len < min_samples:
                continue

            activations = activations[-min_len:]
            cs = causal_scores[-min_len:]

            # Correlation with causal effectiveness
            if np.std(activations) > 0 and np.std(cs) > 0:
                corr = np.corrcoef(activations, cs)[0, 1]
                if not np.isnan(corr):
                    self.cf_ci_correlations[module.module_id].append(corr)

                    # Limit history
                    if len(self.cf_ci_correlations[module.module_id]) > 100:
                        self.cf_ci_correlations[module.module_id] = \
                            self.cf_ci_correlations[module.module_id][-100:]

    def get_module_weight(self, module_name: str) -> float:
        """
        Obtiene peso de un módulo.

        Args:
            module_name: Nombre del módulo

        Returns:
            Peso del módulo
        """
        if module_name in self.modules:
            return self.modules[module_name].weight
        return 1.0 / len(self.modules)

    def get_weighted_output(self, module_outputs: Dict[str, float]) -> float:
        """
        Calcula salida ponderada de módulos.

        output = Σ w_m · output_m

        Args:
            module_outputs: {nombre_módulo: salida}

        Returns:
            Salida ponderada
        """
        total = 0.0
        weight_sum = 0.0

        for name, output in module_outputs.items():
            if name in self.modules:
                weight = self.modules[name].weight
                total += weight * output
                weight_sum += weight

        if weight_sum > 0:
            return total / weight_sum
        return 0.0

    def get_configuration_entropy(self) -> float:
        """
        Calcula entropía de configuración.

        H_conf = -Σ w_m log(w_m)
        """
        weights = [m.weight for m in self.modules.values()]
        weights = np.array(weights)

        # Entropía
        entropy = -np.sum(weights * np.log(weights + 1e-8))

        # Normalizar por entropía máxima
        max_entropy = np.log(len(weights))

        return float(entropy / max_entropy)

    def is_configuration_stable(self) -> bool:
        """
        Verifica si configuración es estable.

        Estable si no ha cambiado significativamente en L_t pasos.
        """
        return self.stability_counter >= L_t(self.t)

    def get_state(self) -> ReconfigurationState:
        """Obtiene estado actual."""
        weights = {name: m.weight for name, m in self.modules.items()}

        most_weighted = max(weights, key=weights.get)
        least_weighted = min(weights, key=weights.get)

        return ReconfigurationState(
            t=self.t,
            module_weights=weights,
            configuration_entropy=self.get_configuration_entropy(),
            total_reconfigurations=self.n_reconfigurations,
            most_weighted_module=most_weighted,
            least_weighted_module=least_weighted,
            is_stable=self.is_configuration_stable()
        )

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas del sistema."""
        state = self.get_state()

        module_stats = {}
        for name, module in self.modules.items():
            module_stats[name] = {
                'weight': module.weight,
                'utility_correlation': module.utility_correlation,
                'gradient': module.gradient,
                'n_updates': module.n_updates
            }

        return {
            'agent': self.agent_name,
            't': self.t,
            'n_modules': len(self.modules),
            'configuration_entropy': state.configuration_entropy,
            'n_reconfigurations': state.total_reconfigurations,
            'is_stable': state.is_stable,
            'stability_counter': self.stability_counter,
            'most_weighted': state.most_weighted_module,
            'least_weighted': state.least_weighted_module,
            'modules': module_stats,
            'weight_bounds': weight_bounds(self.reward_history, self.t)
        }


def test_reconfiguration():
    """Test de auto-reconfiguración reflexiva."""
    print("=" * 60)
    print("TEST AGI-18: REFLECTIVE RECONFIGURATION")
    print("=" * 60)

    # Módulos simplificados para test
    modules = ['self_model', 'planning', 'norms', 'curiosity', 'meta_rules']
    reconfig = ReflectiveReconfiguration("NEO", modules)

    print(f"\nSimulando 500 pasos con {len(modules)} módulos...")

    for t in range(500):
        # Diferentes módulos contribuyen más en diferentes fases
        phase = (t // 100) % 3

        activations = {}
        for i, module in enumerate(modules):
            # Base activation
            base = 0.5 + np.random.randn() * 0.1

            # Algunos módulos más activos en ciertas fases
            if phase == 0 and module == 'planning':
                base += 0.3
            elif phase == 1 and module == 'norms':
                base += 0.25
            elif phase == 2 and module == 'curiosity':
                base += 0.2

            activations[module] = float(np.clip(base, 0, 1))

        # Recompensa correlacionada con ciertos módulos
        if phase == 0:
            reward = 0.5 + 0.3 * activations['planning'] + np.random.randn() * 0.1
        elif phase == 1:
            reward = 0.5 + 0.3 * activations['norms'] + np.random.randn() * 0.1
        else:
            reward = 0.5 + 0.3 * activations['curiosity'] + np.random.randn() * 0.1

        reward = float(np.clip(reward, 0, 1))
        reconfig.record_activations(activations, reward)

        if (t + 1) % 100 == 0:
            state = reconfig.get_state()
            print(f"  t={t+1}: entropy={state.configuration_entropy:.3f}, "
                  f"reconfigs={state.total_reconfigurations}, "
                  f"top={state.most_weighted_module}")

    # Resultados finales
    stats = reconfig.get_statistics()

    print("\n" + "=" * 60)
    print("RESULTADOS REFLECTIVE RECONFIGURATION")
    print("=" * 60)

    print(f"\n  Módulos: {stats['n_modules']}")
    print(f"  Reconfiguraciones: {stats['n_reconfigurations']}")
    print(f"  Entropía config: {stats['configuration_entropy']:.3f}")
    print(f"  Es estable: {stats['is_stable']}")
    print(f"  Límites de peso: {stats['weight_bounds']}")

    print("\n  Por módulo:")
    for name, mod_stats in stats['modules'].items():
        print(f"    {name}: w={mod_stats['weight']:.3f}, "
              f"corr={mod_stats['utility_correlation']:.3f}")

    print(f"\n  Más pesado: {stats['most_weighted']}")
    print(f"  Menos pesado: {stats['least_weighted']}")

    # Test de salida ponderada
    print("\n  Test de salida ponderada:")
    test_outputs = {m: np.random.uniform(0.3, 0.8) for m in modules}
    weighted = reconfig.get_weighted_output(test_outputs)
    print(f"    Salida ponderada: {weighted:.3f}")

    if stats['n_reconfigurations'] > 0:
        print("\n  ✓ Auto-reconfiguración funcionando")
    else:
        print("\n  ⚠ Sin reconfiguraciones detectadas")

    return reconfig


if __name__ == "__main__":
    test_reconfiguration()
