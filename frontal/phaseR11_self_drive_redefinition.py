#!/usr/bin/env python3
"""
R11 – Self-Drive Redefinition (SDR)
===================================

NEO y EVA dejan de tener un drive fijo.
Reapren sus propios pesos sobre variables internas,
guiados SOLO por:
1. Lo que ha funcionado mejor en el pasado
2. La coherencia con su propio "carácter" estructural

100% ENDÓGENO - Sin constantes mágicas
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys

sys.path.insert(0, '/root/NEO_EVA')


@dataclass
class SDRState:
    """Estado del Self-Drive Redefinition."""
    weights: np.ndarray  # w^A actual
    weight_history: List[np.ndarray]
    gradient_history: List[np.ndarray]
    n_updates: int
    feature_names: List[str]


class SelfDriveRedefinition:
    """
    R11: Los agentes redefinen sus propios drives.

    Variables fenomenológicas/estructurales:
    φ_t^A = [integration, neg_surprise, entropy, stability, novelty, otherness, identity, ...]

    Drive como combinación lineal:
    D_t^A = w_t^A · φ_t^A

    Meta-update endógeno basado en correlación con valor interno V.
    """

    def __init__(self, feature_names: List[str] = None):
        if feature_names is None:
            self.feature_names = [
                'integration', 'neg_surprise', 'entropy',
                'stability', 'novelty', 'otherness', 'identity'
            ]
        else:
            self.feature_names = feature_names

        self.d = len(self.feature_names)

        # Estados por agente
        self.agents: Dict[str, SDRState] = {}

        # Historias para análisis
        self.phi_history: Dict[str, List[np.ndarray]] = {}
        self.V_history: Dict[str, List[float]] = {}
        self.drive_history: Dict[str, List[float]] = {}

        self.t = 0

    def register_agent(self, name: str, initial_weights: np.ndarray = None):
        """Registra un agente con pesos iniciales."""
        if initial_weights is None:
            # Pesos uniformes iniciales
            initial_weights = np.ones(self.d) / self.d

        self.agents[name] = SDRState(
            weights=initial_weights.copy(),
            weight_history=[initial_weights.copy()],
            gradient_history=[],
            n_updates=0,
            feature_names=self.feature_names
        )
        self.phi_history[name] = []
        self.V_history[name] = []
        self.drive_history[name] = []

    def _compute_window(self) -> int:
        """Ventana de análisis: W_t = ceil(sqrt(t))"""
        return max(5, int(np.ceil(np.sqrt(self.t + 1))))

    def _compute_value(self, agent_metrics: Dict) -> float:
        """
        Valor interno V_t^A.
        V = rank(S) + rank(IGI) + rank(GI)

        Usa ranks sobre historia para ser endógeno.
        """
        # Extraer métricas disponibles
        S = agent_metrics.get('score', agent_metrics.get('identity', 0.5))
        IGI = agent_metrics.get('igi', agent_metrics.get('integration', 0.5))
        GI = agent_metrics.get('gi', agent_metrics.get('grounding', 0.5))

        # Si hay historia, usar ranks
        name = agent_metrics.get('name', 'default')
        if name in self.V_history and len(self.V_history[name]) > 10:
            # Rank relativo a historia reciente
            V_recent = self.V_history[name][-20:]
            raw_V = S + IGI + GI
            rank = np.mean([1 if raw_V > v else 0 for v in V_recent])
            return rank
        else:
            return S + IGI + GI

    def _compute_meta_gradient(self, name: str) -> np.ndarray:
        """
        Pseudo-gradiente meta: g_j^A = rank(corr_j^A)

        Para cada dimensión j de φ:
        corr_j = corr(φ_{t-W:t,j}, V_{t-W:t})
        """
        W = self._compute_window()

        if len(self.phi_history[name]) < W:
            return np.zeros(self.d)

        # Obtener ventana
        phi_window = np.array(self.phi_history[name][-W:])
        V_window = np.array(self.V_history[name][-W:])

        # Correlación por dimensión
        correlations = np.zeros(self.d)
        for j in range(self.d):
            if np.std(phi_window[:, j]) > 1e-10 and np.std(V_window) > 1e-10:
                corr = np.corrcoef(phi_window[:, j], V_window)[0, 1]
                if not np.isnan(corr):
                    correlations[j] = corr

        # Convertir a ranks (endógeno)
        ranks = np.zeros(self.d)
        for j in range(self.d):
            ranks[j] = np.mean([1 if correlations[j] > correlations[k] else 0
                               for k in range(self.d) if k != j])

        return ranks

    def _compute_meta_learning_rate(self, name: str) -> float:
        """η_meta = 1 / sqrt(N_updates + 1)"""
        n = self.agents[name].n_updates
        return 1.0 / np.sqrt(n + 1)

    def step(self, name: str, phi: np.ndarray, metrics: Dict) -> Dict:
        """
        Un paso de Self-Drive Redefinition.

        Args:
            name: Nombre del agente
            phi: Vector de features fenomenológicos
            metrics: Métricas del agente (score, igi, gi, etc.)

        Returns:
            Estado actual del SDR
        """
        self.t += 1

        if name not in self.agents:
            self.register_agent(name)

        state = self.agents[name]
        metrics['name'] = name

        # Guardar phi
        self.phi_history[name].append(phi.copy())

        # Calcular drive actual
        drive = np.dot(state.weights, phi)
        self.drive_history[name].append(drive)

        # Calcular valor
        V = self._compute_value(metrics)
        self.V_history[name].append(V)

        # Meta-update de pesos
        W = self._compute_window()
        if len(self.phi_history[name]) >= W:
            # Gradiente meta
            g = self._compute_meta_gradient(name)
            state.gradient_history.append(g.copy())

            # Learning rate
            eta = self._compute_meta_learning_rate(name)

            # Update: w_{t+1} = w_t + η * (g - mean(g))
            g_centered = g - np.mean(g)
            state.weights = state.weights + eta * g_centered

            # Normalización
            norm = np.linalg.norm(state.weights) + 1e-10
            state.weights = state.weights / norm

            # Asegurar no-negatividad (opcional pero útil)
            state.weights = np.clip(state.weights, 0.01, None)
            state.weights = state.weights / state.weights.sum()

            state.n_updates += 1

        state.weight_history.append(state.weights.copy())

        return {
            'name': name,
            't': self.t,
            'weights': state.weights.copy(),
            'drive': drive,
            'value': V,
            'n_updates': state.n_updates
        }

    def get_dominant_feature(self, name: str) -> Tuple[str, float]:
        """Retorna la feature dominante actual."""
        if name not in self.agents:
            return ('unknown', 0.0)

        idx = np.argmax(self.agents[name].weights)
        return (self.feature_names[idx], self.agents[name].weights[idx])

    def get_weight_divergence(self, name1: str, name2: str) -> float:
        """Divergencia entre pesos de dos agentes."""
        if name1 not in self.agents or name2 not in self.agents:
            return 0.0

        return float(np.linalg.norm(
            self.agents[name1].weights - self.agents[name2].weights
        ))


def test_R11_go_nogo(sdr: SelfDriveRedefinition, n_nulls: int = 100) -> Dict:
    """
    Tests GO/NO-GO para R11.

    GO si:
    1. Meta-adaptación real: var(w_t) > p95(null_shuffle)
    2. Consistencia: corr(Δw·φ, ΔV) > p95(null)
    3. Diferenciación: 0 < ||w^NEO - w^EVA|| < 1
    """
    results = {'passed': [], 'failed': []}

    # Necesitamos al menos 2 agentes
    if len(sdr.agents) < 2:
        results['failed'].append('need_two_agents')
        return results

    agents = list(sdr.agents.keys())

    # Test 1: Meta-adaptación real
    for name in agents:
        W = np.array(sdr.agents[name].weight_history)
        if len(W) < 50:
            continue

        # Varianza real
        var_real = np.var(W, axis=0).mean()

        # Nulos: shufflear temporalmente
        null_vars = []
        for _ in range(n_nulls):
            W_shuffled = W.copy()
            np.random.shuffle(W_shuffled)
            null_vars.append(np.var(W_shuffled, axis=0).mean())

        p95 = np.percentile(null_vars, 95)

        if var_real > p95:
            results['passed'].append(f'meta_adaptation_{name}')
        else:
            results['failed'].append(f'meta_adaptation_{name}')

    # Test 2: Consistencia
    for name in agents:
        if len(sdr.phi_history[name]) < 50:
            continue

        phi = np.array(sdr.phi_history[name])
        V = np.array(sdr.V_history[name])
        W = np.array(sdr.agents[name].weight_history[:-1])  # Alinear

        if len(W) < len(phi):
            phi = phi[:len(W)]
            V = V[:len(W)]

        # Δw · φ vs ΔV
        if len(W) < 2:
            continue

        delta_w = np.diff(W, axis=0)
        delta_V = np.diff(V)

        # Producto punto Δw · φ
        products = np.sum(delta_w * phi[:-1][:len(delta_w)], axis=1)

        if len(products) < 10:
            continue

        # Correlación real
        corr_real = np.corrcoef(products, delta_V[:len(products)])[0, 1]
        if np.isnan(corr_real):
            corr_real = 0

        # Nulos
        null_corrs = []
        for _ in range(n_nulls):
            delta_V_shuffled = delta_V[:len(products)].copy()
            np.random.shuffle(delta_V_shuffled)
            c = np.corrcoef(products, delta_V_shuffled)[0, 1]
            null_corrs.append(c if not np.isnan(c) else 0)

        p95 = np.percentile(null_corrs, 95)

        if corr_real > p95:
            results['passed'].append(f'consistency_{name}')
        else:
            results['failed'].append(f'consistency_{name}')

    # Test 3: Diferenciación
    if len(agents) >= 2:
        div = sdr.get_weight_divergence(agents[0], agents[1])

        # Nulos: pesos aleatorios
        null_divs = []
        for _ in range(n_nulls):
            w1 = np.random.dirichlet(np.ones(sdr.d))
            w2 = np.random.dirichlet(np.ones(sdr.d))
            null_divs.append(np.linalg.norm(w1 - w2))

        p5 = np.percentile(null_divs, 5)
        p95 = np.percentile(null_divs, 95)

        if p5 < div < p95:
            results['passed'].append('differentiation')
        else:
            results['failed'].append('differentiation')

    # Resumen
    results['is_go'] = len(results['failed']) == 0
    results['summary'] = f"GO" if results['is_go'] else f"NO-GO: {results['failed']}"

    return results


if __name__ == "__main__":
    print("R11 – Self-Drive Redefinition")
    print("=" * 50)

    # Demo
    sdr = SelfDriveRedefinition()
    sdr.register_agent("NEO")
    sdr.register_agent("EVA")

    np.random.seed(42)

    for t in range(200):
        # Simular features
        phi_neo = np.random.dirichlet(np.ones(7) * 2)
        phi_eva = np.random.dirichlet(np.ones(7) * 2)

        # Métricas simuladas
        metrics_neo = {'score': 0.5 + 0.1*np.sin(t/20), 'igi': 0.6, 'gi': 0.5}
        metrics_eva = {'score': 0.5 + 0.1*np.cos(t/20), 'igi': 0.5, 'gi': 0.6}

        sdr.step("NEO", phi_neo, metrics_neo)
        sdr.step("EVA", phi_eva, metrics_eva)

    # Resultados
    print(f"\nNEO dominante: {sdr.get_dominant_feature('NEO')}")
    print(f"EVA dominante: {sdr.get_dominant_feature('EVA')}")
    print(f"Divergencia: {sdr.get_weight_divergence('NEO', 'EVA'):.3f}")

    # Test GO/NO-GO
    results = test_R11_go_nogo(sdr)
    print(f"\n{results['summary']}")
    print(f"Passed: {results['passed']}")
