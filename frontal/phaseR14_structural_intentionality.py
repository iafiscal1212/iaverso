#!/usr/bin/env python3
"""
R14 – Structural Intentionality (SI)
====================================

El sistema no solo tiene "drives", sino dirección consistente:
No solo "estar bien", sino "ir hacia allí".

Como campos vectoriales endógenos, nada de "quiero esto".

100% ENDÓGENO
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys

sys.path.insert(0, '/root/NEO_EVA')


@dataclass
class SIState:
    """Estado de Structural Intentionality."""
    intention: np.ndarray  # I_t^A - dirección intencional
    persistence: float  # pers^A
    alignment: float  # align^A
    sii: float  # Índice de intención estructural
    intention_history: List[np.ndarray]


class StructuralIntentionality:
    """
    R14: Intencionalidad estructural.

    Dirección intencional:
    I_t^A = normalize(E[z_{t+Δ}] - z_t)

    donde Δ = ceil(sqrt(t)) [horizonte crece con experiencia]

    Persistencia:
    pers^A = corr(I_t, I_{t-1})

    Alineación con drive:
    align^A = corr(I_t, grad_D(z_t))

    Índice de Intención Estructural:
    SII^A = rank(pers) + rank(align)
    """

    def __init__(self, dim: int = 6):
        self.dim = dim
        self.agents: Dict[str, SIState] = {}

        # Historias
        self.z_history: Dict[str, List[np.ndarray]] = {}
        self.intention_history: Dict[str, List[np.ndarray]] = {}
        self.persistence_history: Dict[str, List[float]] = {}
        self.alignment_history: Dict[str, List[float]] = {}
        self.sii_history: Dict[str, List[float]] = {}

        self.t = 0

    def register_agent(self, name: str):
        """Registra un agente."""
        self.agents[name] = SIState(
            intention=np.zeros(self.dim),
            persistence=0.5,
            alignment=0.5,
            sii=0.5,
            intention_history=[]
        )
        self.z_history[name] = []
        self.intention_history[name] = []
        self.persistence_history[name] = []
        self.alignment_history[name] = []
        self.sii_history[name] = []

    def _compute_horizon(self) -> int:
        """Horizonte: Δ = ceil(sqrt(t))"""
        return max(5, int(np.ceil(np.sqrt(self.t + 1))))

    def _predict_future_state(self, name: str) -> np.ndarray:
        """
        Predice estado futuro E[z_{t+Δ}].

        Usa extrapolación lineal basada en tendencia reciente.
        """
        if len(self.z_history[name]) < 10:
            return self.z_history[name][-1] if self.z_history[name] else np.zeros(self.dim)

        delta = self._compute_horizon()

        # Tendencia reciente
        recent = np.array(self.z_history[name][-min(20, len(self.z_history[name])):])

        # Ajuste lineal simple por dimensión
        predicted = np.zeros(self.dim)
        for d in range(self.dim):
            if len(recent) >= 2:
                # Pendiente promedio
                slopes = np.diff(recent[:, d])
                avg_slope = np.mean(slopes)
                predicted[d] = recent[-1, d] + avg_slope * delta
            else:
                predicted[d] = recent[-1, d]

        return np.clip(predicted, 0.01, 0.99)

    def _compute_intention(self, z_current: np.ndarray, z_predicted: np.ndarray) -> np.ndarray:
        """
        Dirección intencional:
        I = normalize(z_predicted - z_current)
        """
        diff = z_predicted - z_current
        norm = np.linalg.norm(diff)

        if norm < 1e-10:
            return np.zeros(self.dim)

        return diff / norm

    def _compute_persistence(self, name: str, I_current: np.ndarray) -> float:
        """
        Persistencia: corr(I_t, I_{t-1})
        """
        if len(self.intention_history[name]) < 1:
            return 0.5

        I_prev = self.intention_history[name][-1]

        # Producto punto normalizado (ya están normalizados)
        dot = np.dot(I_current, I_prev)

        # Convertir de [-1, 1] a [0, 1]
        return (dot + 1) / 2

    def _estimate_drive_gradient(self, name: str, w: np.ndarray) -> np.ndarray:
        """
        Estima gradiente del drive respecto a z.

        grad_D(z) ≈ w (para D = w · φ(z), si φ es lineal)

        Para casos no lineales, usamos diferencias finitas.
        """
        if len(self.z_history[name]) < 2:
            return w  # Aproximación lineal

        # Diferencias finitas
        z_curr = self.z_history[name][-1]
        z_prev = self.z_history[name][-2]

        dz = z_curr - z_prev
        norm_dz = np.linalg.norm(dz)

        if norm_dz < 1e-10:
            return w

        # Aproximar gradiente como dirección de cambio ponderada por w
        return w * np.sign(dz)

    def _compute_alignment(self, I: np.ndarray, grad_D: np.ndarray) -> float:
        """
        Alineación: corr(I, grad_D)
        """
        norm_I = np.linalg.norm(I)
        norm_grad = np.linalg.norm(grad_D)

        if norm_I < 1e-10 or norm_grad < 1e-10:
            return 0.5

        dot = np.dot(I, grad_D / norm_grad)

        # Convertir de [-1, 1] a [0, 1]
        return (dot + 1) / 2

    def _compute_sii(self, name: str, pers: float, align: float) -> float:
        """
        Índice de Intención Estructural:
        SII = rank(pers) + rank(align)
        """
        # Rank de persistencia
        if len(self.persistence_history[name]) > 10:
            recent_pers = self.persistence_history[name][-20:]
            pers_rank = np.mean([1 if pers > p else 0 for p in recent_pers])
        else:
            pers_rank = pers

        # Rank de alineación
        if len(self.alignment_history[name]) > 10:
            recent_align = self.alignment_history[name][-20:]
            align_rank = np.mean([1 if align > a else 0 for a in recent_align])
        else:
            align_rank = align

        return (pers_rank + align_rank) / 2  # Normalizar a [0, 1]

    def step(self, name: str, z: np.ndarray, w: np.ndarray) -> Dict:
        """
        Un paso de intencionalidad estructural.

        Args:
            name: Nombre del agente
            z: Estado interno actual
            w: Pesos del drive

        Returns:
            Información del paso
        """
        self.t += 1

        if name not in self.agents:
            self.register_agent(name)

        state = self.agents[name]

        # Guardar estado
        self.z_history[name].append(z.copy())

        # Predecir futuro
        z_predicted = self._predict_future_state(name)

        # Calcular intención
        I = self._compute_intention(z, z_predicted)
        state.intention = I

        # Persistencia
        pers = self._compute_persistence(name, I)
        state.persistence = pers
        self.persistence_history[name].append(pers)

        # Gradiente del drive
        grad_D = self._estimate_drive_gradient(name, w)

        # Alineación
        align = self._compute_alignment(I, grad_D)
        state.alignment = align
        self.alignment_history[name].append(align)

        # SII
        sii = self._compute_sii(name, pers, align)
        state.sii = sii
        self.sii_history[name].append(sii)

        # Guardar intención
        self.intention_history[name].append(I.copy())
        state.intention_history.append(I.copy())

        return {
            'name': name,
            't': self.t,
            'intention': I.copy(),
            'persistence': pers,
            'alignment': align,
            'sii': sii,
            'z_predicted': z_predicted
        }

    def get_intention_field(self, name: str, n_recent: int = 20) -> np.ndarray:
        """Retorna el campo de intención promedio reciente."""
        if name not in self.intention_history:
            return np.zeros(self.dim)

        recent = self.intention_history[name][-n_recent:]
        if not recent:
            return np.zeros(self.dim)

        return np.mean(recent, axis=0)


def test_R14_go_nogo(si: StructuralIntentionality,
                     identity_losses: Dict[str, List[int]] = None,
                     n_nulls: int = 100) -> Dict:
    """
    Tests GO/NO-GO para R14.

    GO si:
    1. pers(real) > p95(null) - direcciones barajadas
    2. align(real) > p95(null) - intención alineada con drive
    3. ΔSII alrededor de losses ≠ 0
    """
    results = {'passed': [], 'failed': []}

    for name, state in si.agents.items():
        # Test 1: Persistencia real vs nulo
        if len(si.persistence_history[name]) > 20:
            pers_real = np.mean(si.persistence_history[name][-20:])

            # Nulos: intenciones barajadas
            null_pers = []
            for _ in range(n_nulls):
                if len(si.intention_history[name]) > 1:
                    intentions = [I.copy() for I in si.intention_history[name]]
                    np.random.shuffle(intentions)
                    # Calcular persistencia shuffled
                    pers_shuffled = []
                    for i in range(1, len(intentions)):
                        dot = np.dot(intentions[i], intentions[i-1])
                        pers_shuffled.append((dot + 1) / 2)
                    null_pers.append(np.mean(pers_shuffled) if pers_shuffled else 0.5)

            if null_pers:
                p95 = np.percentile(null_pers, 95)
                if pers_real > p95:
                    results['passed'].append(f'persistence_{name}')
                else:
                    results['failed'].append(f'persistence_{name}')

        # Test 2: Alineación real vs nulo
        if len(si.alignment_history[name]) > 20:
            align_real = np.mean(si.alignment_history[name][-20:])

            # Nulos
            null_align = []
            for _ in range(n_nulls):
                null_align.append(np.random.uniform(0.3, 0.7))

            p95 = np.percentile(null_align, 95)
            if align_real > p95:
                results['passed'].append(f'alignment_{name}')
            else:
                results['failed'].append(f'alignment_{name}')

        # Test 3: SII cambia tras losses (si hay datos)
        if identity_losses and name in identity_losses:
            losses = identity_losses[name]
            if losses and len(si.sii_history[name]) > max(losses) + 10:
                delta_siis = []
                for loss_t in losses:
                    if loss_t > 10 and loss_t < len(si.sii_history[name]) - 10:
                        sii_before = np.mean(si.sii_history[name][loss_t-10:loss_t])
                        sii_after = np.mean(si.sii_history[name][loss_t:loss_t+10])
                        delta_siis.append(abs(sii_after - sii_before))

                if delta_siis:
                    delta_real = np.mean(delta_siis)

                    # Nulos
                    null_deltas = []
                    for _ in range(n_nulls):
                        random_t = np.random.randint(20, len(si.sii_history[name]) - 20)
                        sii_b = np.mean(si.sii_history[name][random_t-10:random_t])
                        sii_a = np.mean(si.sii_history[name][random_t:random_t+10])
                        null_deltas.append(abs(sii_a - sii_b))

                    p95 = np.percentile(null_deltas, 95)
                    if delta_real > p95:
                        results['passed'].append(f'sii_responds_to_loss_{name}')
                    else:
                        results['failed'].append(f'sii_responds_to_loss_{name}')

    results['is_go'] = len(results['failed']) == 0
    results['summary'] = "GO" if results['is_go'] else f"NO-GO: {results['failed']}"

    return results


if __name__ == "__main__":
    print("R14 – Structural Intentionality")
    print("=" * 50)

    si = StructuralIntentionality(dim=6)
    si.register_agent("NEO")
    si.register_agent("EVA")

    np.random.seed(42)

    # Simular con trayectoria dirigida
    z_neo = np.array([0.2, 0.2, 0.15, 0.15, 0.15, 0.15])
    z_eva = np.array([0.15, 0.15, 0.2, 0.2, 0.15, 0.15])

    w_neo = np.array([0.3, 0.25, 0.1, 0.1, 0.15, 0.1])
    w_eva = np.array([0.1, 0.1, 0.25, 0.3, 0.15, 0.1])

    identity_losses = {'NEO': [], 'EVA': []}

    for t in range(200):
        # Movimiento con dirección + ruido
        direction_neo = np.array([0.01, 0.01, -0.005, -0.005, 0, 0])
        direction_eva = np.array([-0.005, -0.005, 0.01, 0.01, 0, 0])

        z_neo = z_neo + direction_neo + np.random.randn(6) * 0.01
        z_eva = z_eva + direction_eva + np.random.randn(6) * 0.01

        z_neo = np.clip(z_neo, 0.05, 0.95)
        z_eva = np.clip(z_eva, 0.05, 0.95)
        z_neo = z_neo / z_neo.sum()
        z_eva = z_eva / z_eva.sum()

        # Simular identity loss ocasional
        if t in [50, 120]:
            identity_losses['NEO'].append(t)
            z_neo = np.random.dirichlet(np.ones(6))
        if t in [80, 150]:
            identity_losses['EVA'].append(t)
            z_eva = np.random.dirichlet(np.ones(6))

        info_neo = si.step("NEO", z_neo, w_neo)
        info_eva = si.step("EVA", z_eva, w_eva)

        if t % 50 == 0:
            print(f"\nt={t}")
            print(f"NEO: SII={info_neo['sii']:.3f}, pers={info_neo['persistence']:.3f}, "
                  f"align={info_neo['alignment']:.3f}")
            print(f"EVA: SII={info_eva['sii']:.3f}, pers={info_eva['persistence']:.3f}, "
                  f"align={info_eva['alignment']:.3f}")

    # Test
    results = test_R14_go_nogo(si, identity_losses)
    print(f"\n{results['summary']}")
    print(f"Passed: {results['passed']}")
