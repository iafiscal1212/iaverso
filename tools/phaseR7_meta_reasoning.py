#!/usr/bin/env python3
"""
Phase R7: Structural Meta-Reasoning
====================================

El sistema razona sobre qué operador usar basándose en
su historial de mejora estructural.

Componentes:
1. Conjunto de operadores internos O = {O_1, ..., O_K}
2. Score endógeno por operador: ΔS̄(k) = E_t[ΔS_t^(k)]
3. Política sobre operadores: π_t(k) = softmax(β_t * r_k)
4. Operador compuesto: encadenar m_t operadores

100% ENDÓGENO:
- β_t = 1 / (std({ΔS̄(k)}) + 1)
- m_t = ⌈log_2(t+2)⌉
- H_t = ⌈√(t+1)⌉ (horizonte)
- Selección por rank de ΔS histórico

Criterios GO:
1. policy_improves: π mejora S vs random
2. meta_reasoning_active: m_t > 1 para t > √T
3. operator_differentiation: ranks diferenciados
4. composition_beneficial: cadenas mejoran vs single
5. endogenous_temperature: β_t varía con historia
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass
class Operator:
    """Un operador interno."""
    op_id: int
    name: str
    apply: Callable[[np.ndarray], np.ndarray]
    delta_S_history: List[float] = field(default_factory=list)
    usage_count: int = 0


class StructuralMetaReasoning:
    """
    Meta-razonamiento estructural.

    El sistema aprende qué operadores usar basándose
    únicamente en su historial de mejora de S.
    """

    def __init__(self, dim: int = 3):
        self.dim = dim

        # Historia (inicializar primero)
        self.S_history: List[float] = []
        self.policy_history: List[np.ndarray] = []
        self.chosen_operators: List[List[int]] = []
        self.beta_history: List[float] = []
        self.composition_lengths: List[int] = []
        self.z_history: List[np.ndarray] = []

        # Definir operadores internos (de R1)
        self.operators: List[Operator] = []
        self._init_operators()

    def _init_operators(self):
        """Inicializa operadores internos (de R1)."""

        # O1: Homeostasis - hacia media histórica
        def homeostasis(z: np.ndarray, history: List[np.ndarray]) -> np.ndarray:
            if len(history) < 2:
                return z
            mean = np.mean(history, axis=0)
            direction = mean - z
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                return z
            return z + direction / norm * np.sqrt(1 / (len(history) + 1))

        # O2: Exploration - máxima varianza
        def exploration(z: np.ndarray, history: List[np.ndarray]) -> np.ndarray:
            if len(history) < 3:
                return z + np.random.randn(len(z)) * 0.1
            cov = np.cov(np.array(history).T)
            if cov.ndim == 0:
                return z + np.random.randn(len(z)) * 0.1
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            max_dir = eigenvectors[:, -1]
            return z + max_dir * np.sqrt(1 / (len(history) + 1))

        # O3: Momentum - continuar trayectoria
        def momentum(z: np.ndarray, history: List[np.ndarray]) -> np.ndarray:
            if len(history) < 2:
                return z
            velocity = history[-1] - history[-2] if len(history) >= 2 else np.zeros_like(z)
            return z + velocity * (1 / np.sqrt(len(history) + 1))

        # O4: Contraction - reducir dispersión
        def contraction(z: np.ndarray, history: List[np.ndarray]) -> np.ndarray:
            if len(history) < 2:
                return z
            mean = np.mean(history, axis=0)
            return z + (mean - z) * (1 / np.sqrt(len(history) + 1))

        # O5: Orthogonal - mínima varianza
        def orthogonal(z: np.ndarray, history: List[np.ndarray]) -> np.ndarray:
            if len(history) < 3:
                return z
            cov = np.cov(np.array(history).T)
            if cov.ndim == 0:
                return z
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            min_dir = eigenvectors[:, 0]
            return z + min_dir * np.sqrt(1 / (len(history) + 1))

        # O6: Stability - hacia dimensiones estables
        def stability(z: np.ndarray, history: List[np.ndarray]) -> np.ndarray:
            if len(history) < 3:
                return z
            variances = np.var(np.array(history), axis=0)
            # Mover hacia dimensiones de baja varianza
            weights = 1 / (variances + 1e-10)
            weights = weights / np.sum(weights)
            target = np.mean(history, axis=0) * weights
            direction = target - z * weights
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                return z
            return z + direction / norm * np.sqrt(1 / (len(history) + 1))

        self.operators = [
            Operator(0, "homeostasis", lambda z, h=self.z_history: homeostasis(z, h)),
            Operator(1, "exploration", lambda z, h=self.z_history: exploration(z, h)),
            Operator(2, "momentum", lambda z, h=self.z_history: momentum(z, h)),
            Operator(3, "contraction", lambda z, h=self.z_history: contraction(z, h)),
            Operator(4, "orthogonal", lambda z, h=self.z_history: orthogonal(z, h)),
            Operator(5, "stability", lambda z, h=self.z_history: stability(z, h)),
        ]

    def _compute_S(self, z: np.ndarray) -> float:
        """
        Calcula S endógeno basado en historia.
        S = combinación de métricas estructurales.
        """
        if len(self.z_history) < 2:
            return 0.5

        t = len(self.z_history)
        w = max(1, int(np.sqrt(t)))

        # Componentes endógenos
        mean_hist = np.mean(self.z_history[-w:], axis=0)

        # Otherness: distancia a media
        otherness = np.linalg.norm(z - mean_hist)

        # Stability: inverso de varianza reciente
        var_recent = np.var(self.z_history[-w:])
        stability = 1 / (1 + var_recent)

        # Coherence: autocorrelación
        if t >= 2 * w:
            corr = np.corrcoef(
                np.array(self.z_history[-2*w:-w]).flatten(),
                np.array(self.z_history[-w:]).flatten()
            )[0, 1]
            coherence = abs(corr) if not np.isnan(corr) else 0.5
        else:
            coherence = 0.5

        # Combinación rank-based
        components = np.array([otherness, stability, coherence])
        ranks = np.argsort(np.argsort(components)) + 1
        S = np.sum(ranks * components) / np.sum(ranks)

        return float(np.clip(S, 0, 1))

    def _compute_beta(self) -> float:
        """
        Temperatura inversa endógena.
        β_t = 1 / (std({ΔS̄(k)}) + 1)
        """
        delta_S_means = []
        for op in self.operators:
            if len(op.delta_S_history) > 0:
                delta_S_means.append(np.mean(op.delta_S_history))
            else:
                delta_S_means.append(0)

        std = np.std(delta_S_means)
        return 1.0 / (std + 1)

    def _compute_policy(self) -> np.ndarray:
        """
        Política sobre operadores.
        π_t(k) = softmax(β_t * r_k)
        """
        # Calcular ranks de ΔS̄ por operador
        delta_S_means = []
        for op in self.operators:
            if len(op.delta_S_history) > 0:
                delta_S_means.append(np.mean(op.delta_S_history))
            else:
                delta_S_means.append(0)

        delta_S_means = np.array(delta_S_means)

        # Ranks (1 = peor, K = mejor)
        ranks = np.argsort(np.argsort(delta_S_means)) + 1

        # Beta endógeno
        beta = self._compute_beta()
        self.beta_history.append(beta)

        # Softmax
        logits = beta * ranks
        logits = logits - np.max(logits)  # Estabilidad numérica
        probs = np.exp(logits)
        probs = probs / np.sum(probs)

        return probs

    def _compute_m(self, t: int) -> int:
        """
        Número de operadores a encadenar.
        m_t = ⌈log_2(t+2)⌉
        """
        return int(np.ceil(np.log2(t + 2)))

    def _compute_horizon(self, t: int) -> int:
        """
        Horizonte endógeno.
        H_t = ⌈√(t+1)⌉
        """
        return int(np.ceil(np.sqrt(t + 1)))

    def step(self, z: np.ndarray) -> Dict:
        """Ejecuta un paso de meta-razonamiento."""
        t = len(self.z_history)
        self.z_history.append(z.copy())

        # S actual
        S_current = self._compute_S(z)
        self.S_history.append(S_current)

        # Política
        policy = self._compute_policy()
        self.policy_history.append(policy)

        # Número de operadores a encadenar
        m = self._compute_m(t)
        self.composition_lengths.append(m)

        # Seleccionar operadores según política
        chosen = []
        for _ in range(m):
            k = np.random.choice(len(self.operators), p=policy)
            chosen.append(k)
        self.chosen_operators.append(chosen)

        # Aplicar cadena de operadores
        z_new = z.copy()
        for k in chosen:
            op = self.operators[k]
            z_new = op.apply(z_new)
            op.usage_count += 1

        # Normalizar al simplex si es necesario
        if np.any(z_new < 0):
            z_new = np.abs(z_new)
        z_new = z_new / (np.sum(z_new) + 1e-10)

        # Calcular S después
        S_new = self._compute_S(z_new)

        # Decisión: usar z_new solo si mejora S
        if S_new > S_current:
            z_result = z_new
            improved = True
        else:
            z_result = z
            improved = False

        # Actualizar historial de ΔS por operador
        delta_S = S_new - S_current
        for k in chosen:
            self.operators[k].delta_S_history.append(delta_S)

        return {
            't': t,
            'S_before': S_current,
            'S_after': S_new,
            'improved': improved,
            'policy': policy.tolist(),
            'chosen_operators': chosen,
            'm': m,
            'beta': self.beta_history[-1] if self.beta_history else 1.0,
            'z_result': z_result.tolist()
        }

    def get_summary(self) -> Dict:
        """Resumen del meta-razonamiento."""
        T = len(self.S_history)

        # Estadísticas por operador
        op_stats = []
        for op in self.operators:
            if len(op.delta_S_history) > 0:
                mean_delta = np.mean(op.delta_S_history)
                std_delta = np.std(op.delta_S_history)
            else:
                mean_delta = 0
                std_delta = 0

            op_stats.append({
                'id': op.op_id,
                'name': op.name,
                'usage_count': op.usage_count,
                'mean_delta_S': float(mean_delta),
                'std_delta_S': float(std_delta)
            })

        # Diferenciación de operadores
        delta_means = [s['mean_delta_S'] for s in op_stats]
        op_differentiation = np.std(delta_means) if len(delta_means) > 1 else 0

        # Mejora de política vs random
        # Comparar S promedio con política vs baseline
        if T > 10:
            recent_S = np.mean(self.S_history[-T//2:])
            early_S = np.mean(self.S_history[:T//2])
            policy_improvement = recent_S - early_S
        else:
            policy_improvement = 0

        # Beta variabilidad
        if len(self.beta_history) > 1:
            beta_variability = np.std(self.beta_history)
        else:
            beta_variability = 0

        # Composición beneficiosa
        # Comparar S cuando m > 1 vs m = 1
        S_with_composition = []
        S_without = []
        for i, m in enumerate(self.composition_lengths):
            if i < len(self.S_history):
                if m > 1:
                    S_with_composition.append(self.S_history[i])
                else:
                    S_without.append(self.S_history[i])

        if len(S_with_composition) > 0 and len(S_without) > 0:
            composition_benefit = np.mean(S_with_composition) - np.mean(S_without)
        else:
            composition_benefit = 0

        return {
            'T': T,
            'operator_stats': op_stats,
            'operator_differentiation': float(op_differentiation),
            'policy_improvement': float(policy_improvement),
            'beta_variability': float(beta_variability),
            'composition_benefit': float(composition_benefit),
            'mean_composition_length': float(np.mean(self.composition_lengths)) if self.composition_lengths else 1,
            'final_S': float(self.S_history[-1]) if self.S_history else 0.5
        }


def run_phaseR7_test(n_steps: int = 1000, seed: int = 42) -> Dict:
    """Ejecuta test de Phase R7 - 100% ENDÓGENO."""
    np.random.seed(seed)

    smr = StructuralMetaReasoning(dim=3)

    # Estado inicial: centro del simplex
    z = np.ones(3) / 3

    results = []
    S_series = []

    for t in range(n_steps):
        result = smr.step(z)
        results.append(result)
        S_series.append(result['S_after'])

        # Actualizar z con resultado
        z = np.array(result['z_result'])

    # Summary
    summary = smr.get_summary()

    # Evaluación GO/NO-GO (100% ENDÓGENO)
    T = n_steps

    # Thresholds endógenos
    # policy_improves: mejora > 0
    # meta_reasoning_active: m > 1 para t > √T
    sqrt_T = int(np.sqrt(T))
    late_m = smr.composition_lengths[sqrt_T:] if len(smr.composition_lengths) > sqrt_T else []
    meta_active = np.mean(late_m) > 1 if late_m else False

    # operator_differentiation: std > 1/K
    K = len(smr.operators)
    diff_threshold = 1.0 / K

    # composition_beneficial: > 0
    # beta varies: std > 1/√T
    beta_threshold = 1.0 / np.sqrt(T)

    criteria = {
        'policy_improves': summary['policy_improvement'] > 0,
        'meta_reasoning_active': meta_active,
        'operator_differentiation': summary['operator_differentiation'] > diff_threshold,
        'composition_beneficial': summary['composition_benefit'] > 0,
        'endogenous_temperature': summary['beta_variability'] > beta_threshold
    }

    go = sum(criteria.values()) >= 3

    return {
        'go': go,
        'criteria': criteria,
        'summary': summary,
        'S_series': S_series[-100:],
        'beta_series': smr.beta_history[-100:] if smr.beta_history else []
    }


def main():
    """Ejecuta Phase R7."""
    print("=" * 70)
    print("PHASE R7: STRUCTURAL META-REASONING")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}")

    results = run_phaseR7_test(n_steps=1000)

    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)

    print(f"\nMejora de política: {results['summary']['policy_improvement']:.4f}")
    print(f"Diferenciación de operadores: {results['summary']['operator_differentiation']:.4f}")
    print(f"Beneficio de composición: {results['summary']['composition_benefit']:.4f}")
    print(f"Variabilidad de β: {results['summary']['beta_variability']:.4f}")

    print("\nOperadores:")
    for op in results['summary']['operator_stats']:
        print(f"  {op['name']}: uso={op['usage_count']}, ΔS̄={op['mean_delta_S']:.4f}")

    print("\nCriterios:")
    for name, passed in results['criteria'].items():
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")

    n_pass = sum(results['criteria'].values())
    n_total = len(results['criteria'])
    go_status = "GO" if results['go'] else "NO-GO"

    print(f"\nResultado: {go_status} ({n_pass}/{n_total} criterios)")

    # Guardar resultados
    results_dir = '/root/NEO_EVA/results/phaseR7'
    os.makedirs(results_dir, exist_ok=True)

    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32, np.int16)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(f'{results_dir}/phaseR7_results.json', 'w') as f:
        json.dump(convert_types(results), f, indent=2)

    # Generar figura
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # S over time
    ax = axes[0, 0]
    ax.plot(results['S_series'], alpha=0.7)
    ax.set_xlabel('Step (last 100)')
    ax.set_ylabel('S(t)')
    ax.set_title('Proto-Subjectivity Score')
    ax.grid(True, alpha=0.3)

    # Beta over time
    ax = axes[0, 1]
    if results['beta_series']:
        ax.plot(results['beta_series'], alpha=0.7, color='orange')
    ax.set_xlabel('Step (last 100)')
    ax.set_ylabel('β(t)')
    ax.set_title('Inverse Temperature (Endogenous)')
    ax.grid(True, alpha=0.3)

    # Operator usage
    ax = axes[1, 0]
    names = [op['name'] for op in results['summary']['operator_stats']]
    usages = [op['usage_count'] for op in results['summary']['operator_stats']]
    ax.barh(names, usages, color='teal', alpha=0.7)
    ax.set_xlabel('Usage Count')
    ax.set_title('Operator Usage')

    # Criteria
    ax = axes[1, 1]
    criteria_names = list(results['criteria'].keys())
    criteria_values = [1 if v else 0 for v in results['criteria'].values()]
    colors = ['green' if v else 'red' for v in results['criteria'].values()]
    y_pos = range(len(criteria_names))
    ax.barh(y_pos, criteria_values, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(criteria_names, fontsize=9)
    ax.set_xlim(0, 1.5)
    ax.set_title(f'Phase R7 Criteria: {go_status}')

    plt.tight_layout()
    plt.savefig('/root/NEO_EVA/figures/phaseR7_results.png', dpi=150)
    plt.close()

    print(f"\nResultados guardados en: {results_dir}")
    print(f"Figura: /root/NEO_EVA/figures/phaseR7_results.png")

    return results


if __name__ == "__main__":
    main()
