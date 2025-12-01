"""
TEST 4 — AUTO-MODELO (Self-Model Accuracy) V2
=============================================

Qué mide: Autoconsciencia estructural
AGI involucrada: AGI-4 (SelfPredictorV2), AGI-11, AGI-14

Procedimiento:
1. Perturbas al agente internamente (shock controlado)
2. El agente predice sus estados futuros a 1,3,5 pasos
3. Comparas predicción vs realidad

Métrica:
    S4 = 1 - MSE_5steps / percentile95(MSE_null)

donde MSE_null es el error de un predictor naive (predice t-1).
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Importar funciones endógenas
from cognition.agi_dynamic_constants import (
    L_t, adaptive_learning_rate, to_simplex, normalized_entropy
)


@dataclass
class SelfModelMetrics:
    """Métricas de auto-modelo por agente."""
    agent_name: str
    mse_1step: float
    mse_3step: float
    mse_5step: float
    mse_null: float
    confidence_mean: float
    confidence_behavior_corr: float
    S4_score: float


class Test4SelfModel:
    """Test de precisión del auto-modelo usando SelfPredictorV2."""

    def __init__(self, agents: List[str] = None):
        self.agents = agents or ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
        self.baseline_steps = 300
        self.shock_steps = 150
        self.recovery_steps = 200
        self.n_shocks = 3

    def run(self, verbose: bool = True) -> Tuple[float, Dict]:
        """Ejecuta el test."""
        from cognition.self_model_v2 import SelfPredictorV2
        from cognition import IntrospectiveUncertainty, PredictionChannel

        if verbose:
            print("=" * 70)
            print("TEST 4: AUTO-MODELO (SelfPredictorV2)")
            print("=" * 70)

        # Dimensiones del estado
        z_dim = 6
        phi_dim = 5
        drives_dim = 6
        state_dim = z_dim + phi_dim + drives_dim

        # Inicializar módulos
        self_model = {a: SelfPredictorV2(a, z_dim=z_dim, phi_dim=phi_dim, drives_dim=drives_dim)
                      for a in self.agents}
        uncertainty = {a: IntrospectiveUncertainty(a) for a in self.agents}

        # Métricas
        metrics: Dict[str, Dict] = {a: {
            'errors_1step': [],
            'errors_3step': [],
            'errors_5step': [],
            'errors_null': [],  # Predictor naive
            'confidences': [],
            'behaviors': []  # Para correlación confianza-comportamiento
        } for a in self.agents}

        # Buffer de estados pasados para predictor null
        state_buffers: Dict[str, List[np.ndarray]] = {a: [] for a in self.agents}

        t = 0

        # Estado base generado con Dirichlet (endógeno)
        base_z = np.random.dirichlet(np.ones(z_dim) * 2)
        base_phi = np.random.dirichlet(np.ones(phi_dim) * 3)

        # Estado evolutivo que cambia con tendencia (para que el modelo pueda aprender)
        state_trajectory = {'z': base_z.copy(), 'phi': base_phi.copy()}

        def get_state_components(shock_active: bool = False,
                                 shock_magnitude: float = 0.0,
                                 current_t: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Genera componentes del estado con dinámica predecible + ruido."""
            # Ruido que mantiene variabilidad sin ser dominante
            # El piso mínimo asegura que haya algo que predecir
            noise_floor = 0.02
            noise_scale = max(noise_floor, 0.04 / np.sqrt(current_t / 200 + 1))

            # Z evoluciona con tendencia hacia base (predecible) + ruido (impredecible)
            decay_to_base = 0.05  # Lento retorno al atractor
            z = state_trajectory['z']
            z = z * (1 - decay_to_base) + base_z * decay_to_base

            if shock_active:
                # Shock: perturbación súbita
                shock_weights = np.zeros(z_dim)
                shock_weights[current_t % z_dim] = shock_magnitude
                z = z + shock_weights

            z = z + np.random.randn(z_dim) * noise_scale
            z = to_simplex(z)
            state_trajectory['z'] = z.copy()

            # phi evoluciona similarmente
            phi = state_trajectory['phi']
            phi = phi * (1 - decay_to_base) + base_phi * decay_to_base
            if shock_active:
                phi = phi * (1 - shock_magnitude * 0.3)
            phi = phi + np.random.randn(phi_dim) * noise_scale
            state_trajectory['phi'] = phi.copy()

            drives = z.copy()

            return z, phi, drives

        # Fase 1: Baseline (aprender auto-modelo)
        if verbose:
            print(f"\nFase 1: Baseline ({self.baseline_steps} pasos)")

        for step in range(self.baseline_steps):
            t += 1
            for agent in self.agents:
                z, phi, drives = get_state_components(current_t=t)
                state = np.concatenate([z, phi, drives])

                # Guardar en buffer
                state_buffers[agent].append(state.copy())
                if len(state_buffers[agent]) > 20:
                    state_buffers[agent] = state_buffers[agent][-20:]

                # Actualizar modelo
                self_model[agent].update(z, phi, drives)

                # Registrar predicción para uncertainty
                pred_1 = self_model[agent].predict_k_steps(state, 1)
                uncertainty[agent].record_prediction(
                    PredictionChannel.SELF_MODEL,
                    pred_1[0],
                    state[0]
                )

        # Fase 2: Shocks controlados con evaluación multi-paso
        if verbose:
            print(f"\nFase 2: {self.n_shocks} shocks controlados")

        for shock_idx in range(self.n_shocks):
            if verbose:
                print(f"\n  Shock {shock_idx + 1}:")

            # Magnitud del shock endógena: basada en entropía del estado actual
            current_entropy = normalized_entropy(base_z)
            shock_magnitude = 0.2 + 0.3 * current_entropy + np.random.random() * 0.1

            # Duración del shock adaptativa
            shock_duration = L_t(t)
            decay_rate = L_t(t) * 2  # Velocidad de decaimiento

            for step in range(self.shock_steps):
                t += 1
                shock_active = step < shock_duration
                decay = np.exp(-(step - shock_duration) / decay_rate) if step >= shock_duration else 1.0

                for agent in self.agents:
                    current_magnitude = shock_magnitude * decay if step >= shock_duration else shock_magnitude
                    recovery_phase = shock_duration + L_t(t)  # Fase de recuperación adaptativa
                    z, phi, drives = get_state_components(shock_active or step < recovery_phase, current_magnitude, current_t=t)
                    state = np.concatenate([z, phi, drives])

                    # Predicciones ANTES de ver estado real (usando estado del buffer)
                    if len(state_buffers[agent]) > 0:
                        prev_state = state_buffers[agent][-1]

                        pred_1 = self_model[agent].predict_k_steps(prev_state, 1)
                        pred_3 = self_model[agent].predict_k_steps(prev_state, 3)
                        pred_5 = self_model[agent].predict_k_steps(prev_state, 5)

                        # Errores de predicción
                        mse_1 = float(np.mean((pred_1 - state) ** 2))
                        metrics[agent]['errors_1step'].append(mse_1)

                        if len(state_buffers[agent]) >= 3:
                            mse_3 = float(np.mean((pred_3 - state) ** 2))
                            metrics[agent]['errors_3step'].append(mse_3)

                        if len(state_buffers[agent]) >= 5:
                            mse_5 = float(np.mean((pred_5 - state) ** 2))
                            metrics[agent]['errors_5step'].append(mse_5)

                        # Error del predictor null (predice estado anterior)
                        null_pred = state_buffers[agent][-1]
                        mse_null = float(np.mean((null_pred - state) ** 2))
                        metrics[agent]['errors_null'].append(mse_null)

                    # Confianza
                    conf = self_model[agent].confidence()
                    metrics[agent]['confidences'].append(conf)

                    # Comportamiento (learning rate que debería bajar con alta confianza)
                    lr = self_model[agent].get_learning_rate_modifier()
                    metrics[agent]['behaviors'].append(lr)

                    # Actualizar buffer y modelo
                    state_buffers[agent].append(state.copy())
                    if len(state_buffers[agent]) > 20:
                        state_buffers[agent] = state_buffers[agent][-20:]

                    self_model[agent].update(z, phi, drives)

            if verbose:
                mean_mse_5 = np.mean([np.mean(metrics[a]['errors_5step'][-self.shock_steps:])
                                     for a in self.agents if metrics[a]['errors_5step']])
                print(f"    MSE 5-step medio: {mean_mse_5:.4f}")

        # Fase 3: Recuperación
        if verbose:
            print(f"\nFase 3: Recuperación ({self.recovery_steps} pasos)")

        for step in range(self.recovery_steps):
            t += 1
            for agent in self.agents:
                z, phi, drives = get_state_components(current_t=t)
                state = np.concatenate([z, phi, drives])

                if len(state_buffers[agent]) > 0:
                    prev_state = state_buffers[agent][-1]
                    pred_5 = self_model[agent].predict_k_steps(prev_state, 5)
                    mse_5 = float(np.mean((pred_5 - state) ** 2))
                    metrics[agent]['errors_5step'].append(mse_5)

                    null_pred = state_buffers[agent][-1]
                    mse_null = float(np.mean((null_pred - state) ** 2))
                    metrics[agent]['errors_null'].append(mse_null)

                conf = self_model[agent].confidence()
                metrics[agent]['confidences'].append(conf)

                state_buffers[agent].append(state.copy())
                if len(state_buffers[agent]) > 20:
                    state_buffers[agent] = state_buffers[agent][-20:]

                self_model[agent].update(z, phi, drives)

        # Calcular resultados
        results: Dict[str, SelfModelMetrics] = {}
        S4_scores = []

        if verbose:
            print(f"\n{'=' * 70}")
            print("RESULTADOS")
            print("=" * 70)

        for agent in self.agents:
            # MSE por horizonte
            mse_1 = float(np.mean(metrics[agent]['errors_1step'])) if metrics[agent]['errors_1step'] else 1.0
            mse_3 = float(np.mean(metrics[agent]['errors_3step'])) if metrics[agent]['errors_3step'] else 1.0
            mse_5 = float(np.mean(metrics[agent]['errors_5step'])) if metrics[agent]['errors_5step'] else 1.0

            # MSE null (percentil 95)
            mse_null_list = metrics[agent]['errors_null']
            if mse_null_list:
                mse_null_p95 = float(np.percentile(mse_null_list, 95))
            else:
                mse_null_p95 = 1.0

            # S4 = 1 - MSE_5steps / percentile95(MSE_null)
            S4 = 1.0 - (mse_5 / (mse_null_p95 + 1e-8))
            S4 = float(np.clip(S4, 0, 1))

            # Confianza media
            conf_mean = float(np.mean(metrics[agent]['confidences'])) if metrics[agent]['confidences'] else 0.5

            # Correlación confianza-comportamiento
            if len(metrics[agent]['confidences']) > 10 and len(metrics[agent]['behaviors']) > 10:
                try:
                    corr = np.corrcoef(
                        metrics[agent]['confidences'][-200:],
                        metrics[agent]['behaviors'][-200:]
                    )[0, 1]
                    conf_behav_corr = float(corr) if not np.isnan(corr) else 0.0
                except:
                    conf_behav_corr = 0.0
            else:
                conf_behav_corr = 0.0

            results[agent] = SelfModelMetrics(
                agent_name=agent,
                mse_1step=mse_1,
                mse_3step=mse_3,
                mse_5step=mse_5,
                mse_null=mse_null_p95,
                confidence_mean=conf_mean,
                confidence_behavior_corr=conf_behav_corr,
                S4_score=S4
            )

            S4_scores.append(S4)

            if verbose:
                print(f"\n  {agent}:")
                print(f"    MSE 1-step: {mse_1:.4f}")
                print(f"    MSE 3-step: {mse_3:.4f}")
                print(f"    MSE 5-step: {mse_5:.4f}")
                print(f"    MSE null p95: {mse_null_p95:.4f}")
                print(f"    Confianza media: {conf_mean:.3f}")
                print(f"    Corr conf-behav: {conf_behav_corr:.3f}")
                print(f"    S4: {S4:.4f}")

        S4_global = float(np.mean(S4_scores))

        if verbose:
            print(f"\n{'═' * 70}")
            print(f"S4 (Auto-Modelo): {S4_global:.4f}")
            print(f"  Fórmula: S4 = 1 - MSE_5steps / percentile95(MSE_null)")
            print("═" * 70)

        return S4_global, {
            'score': S4_global,
            'agents': {a: vars(m) for a, m in results.items()},
            'n_shocks': self.n_shocks,
            'total_steps': t
        }


def run_test(verbose: bool = True) -> Tuple[float, Dict]:
    """Ejecuta Test 4."""
    test = Test4SelfModel()
    return test.run(verbose=verbose)


if __name__ == "__main__":
    score, results = run_test()
    print(f"\nFinal S4 Score: {score:.4f}")
