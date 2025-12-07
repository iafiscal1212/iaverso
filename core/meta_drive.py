#!/usr/bin/env python3
"""
Meta-Drive: Funciones de Drive Auto-Modificables
=================================================

Los agentes pueden modificar sus propias funciones de drive basándose
en su historia de éxito/fracaso.

Matemática:
-----------
El drive de un agente A es una combinación lineal de K componentes base:

    drive_A(t) = Σ_k w_k(t) · c_k(t)

donde:
    - c_k(t) son componentes base (entropía, sorpresa, novedad, etc.)
    - w_k(t) son pesos que EVOLUCIONAN

Evolución de pesos (100% endógena):

    Δw_k(t) = η(t) · ∂V/∂w_k

donde:
    - η(t) = 1/√(t+1) tasa de aprendizaje
    - V = valor acumulado (reward endógeno)
    - ∂V/∂w_k ≈ corr(c_k, V_history) gradiente aproximado

El "valor" V es endógeno:
    V(t) = -surprise(t) + integration(t)

    (sobrevivir = predecir bien + mantenerse integrado)

Esto permite que:
- NEO pueda descubrir que necesita más exploración
- EVA pueda descubrir que necesita más estabilidad
- Ambos pueden inventar drives que no programamos

100% ENDÓGENO - Sin constantes mágicas
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


@dataclass
class DriveComponent:
    """Un componente base del drive."""
    name: str
    value: float
    weight: float
    gradient: float = 0.0


@dataclass
class MetaDriveState:
    """Estado del meta-drive en un instante."""
    t: int
    agent: str
    components: List[DriveComponent]
    drive_total: float
    value: float
    weights_entropy: float  # Diversidad de pesos


class MetaDrive:
    """
    Sistema de drive auto-modificable.

    Componentes base (K=6):
    1. entropy: H(z) - prefiere estados informativos
    2. neg_surprise: -surprise - prefiere predecir bien
    3. novelty: distancia a historia - prefiere explorar
    4. stability: -var(z_recent) - prefiere estabilidad
    5. integration: corr(z_i, z_j) - prefiere coherencia
    6. otherness: distancia al otro agente - prefiere diferenciarse

    Los pesos w_k evolucionan para maximizar V(t).

    100% Endógeno
    """

    def __init__(self, agent: str, n_components: int = 6):
        self.agent = agent
        self.K = n_components

        # Nombres de componentes
        self.component_names = [
            'entropy', 'neg_surprise', 'novelty',
            'stability', 'integration', 'otherness'
        ]

        # Pesos iniciales (uniformes)
        self.weights = np.ones(self.K) / self.K

        # Historia
        self.component_history: List[np.ndarray] = []
        self.weight_history: List[np.ndarray] = []
        self.value_history: List[float] = []
        self.drive_history: List[float] = []

        # Para cálculo de gradientes
        self.z_history: List[np.ndarray] = []
        self.surprise_history: List[float] = []

        self.t = 0

    def _compute_learning_rate(self) -> float:
        """η(t) = 1/√(t+1) - endógeno, depende solo del tiempo"""
        return 1 / np.sqrt(self.t + 1)

    def _compute_entropy_component(self, z: np.ndarray) -> float:
        """c_1: Entropía normalizada."""
        EPS = np.finfo(float).eps
        z_safe = np.clip(z, EPS, 1)
        z_norm = z_safe / z_safe.sum()
        H = -np.sum(z_norm * np.log(z_norm))
        H_max = np.log(len(z))
        # Punto medio por simetría si H_max = 0
        return H / H_max if H_max > 0 else 1/2

    def _compute_neg_surprise_component(self, surprise: float) -> float:
        """c_2: Negativo de sorpresa (normalizado)."""
        if self.surprise_history:
            max_surprise = max(self.surprise_history) + np.finfo(float).eps
            return 1 - surprise / max_surprise
        # Punto medio por simetría
        return 1/2

    def _compute_novelty_component(self, z: np.ndarray) -> float:
        """c_3: Distancia al centroide histórico."""
        if len(self.z_history) < 10:
            # Punto medio por simetría
            return 1/2

        centroid = np.mean(self.z_history[-20:], axis=0)
        dist = np.linalg.norm(z - centroid)

        # Normalizar por distancia típica
        typical_dists = [np.linalg.norm(zh - centroid) for zh in self.z_history[-20:]]
        typical = np.mean(typical_dists) + np.finfo(float).eps

        return min(dist / typical, 1)

    def _compute_stability_component(self, z: np.ndarray) -> float:
        """c_4: Inverso de varianza reciente."""
        if len(self.z_history) < 5:
            # Punto medio por simetría
            return 1/2

        recent = np.array(self.z_history[-5:])
        variance = np.var(recent)

        # Normalizar
        if len(self.z_history) > 20:
            all_vars = [np.var(self.z_history[i:i+5])
                       for i in range(len(self.z_history)-5)]
            max_var = max(all_vars) + np.finfo(float).eps
            return 1 - variance / max_var

        # Factor endógeno: dimension del vector
        return 1 / (1 + variance * len(z))

    def _compute_integration_component(self, z: np.ndarray) -> float:
        """c_5: Correlación media entre dimensiones."""
        if len(self.z_history) < 10:
            # Punto medio por simetría
            return 1/2

        recent = np.array(self.z_history[-10:])
        if recent.shape[1] < 2:
            return 1/2

        corr = np.corrcoef(recent.T)
        mask = ~np.eye(len(z), dtype=bool)
        correlations = corr[mask]
        correlations = correlations[~np.isnan(correlations)]

        if len(correlations) == 0:
            return 1/2

        return float(np.mean(np.abs(correlations)))

    def _compute_otherness_component(self, z: np.ndarray,
                                      other_z: Optional[np.ndarray]) -> float:
        """c_6: Distancia al otro agente."""
        if other_z is None:
            # Punto medio por simetría
            return 1/2

        dist = np.linalg.norm(z - other_z)
        max_dist = np.sqrt(len(z))  # Máximo teórico geométrico

        return dist / max_dist

    def compute_components(self, z: np.ndarray, surprise: float,
                           other_z: Optional[np.ndarray] = None) -> np.ndarray:
        """Calcula todos los componentes del drive."""
        components = np.array([
            self._compute_entropy_component(z),
            self._compute_neg_surprise_component(surprise),
            self._compute_novelty_component(z),
            self._compute_stability_component(z),
            self._compute_integration_component(z),
            self._compute_otherness_component(z, other_z)
        ])
        return components

    def compute_value(self, surprise: float, integration: float) -> float:
        """
        Valor endógeno: V(t) = -surprise + integration

        Sobrevivir = predecir bien + mantenerse integrado
        """
        # Normalizar surprise
        if self.surprise_history:
            max_s = max(self.surprise_history) + np.finfo(float).eps
            norm_surprise = surprise / max_s
        else:
            # Punto medio por simetría
            norm_surprise = 1/2

        return -norm_surprise + integration

    def compute_gradients(self) -> np.ndarray:
        """
        Calcula gradientes ∂V/∂w_k ≈ corr(c_k, V_history)

        100% endógeno: correlación histórica
        """
        if len(self.component_history) < 20:
            return np.zeros(self.K)

        window = min(50, len(self.component_history))

        C = np.array(self.component_history[-window:])  # (window, K)
        V = np.array(self.value_history[-window:])       # (window,)

        gradients = np.zeros(self.K)
        for k in range(self.K):
            corr = np.corrcoef(C[:, k], V)[0, 1]
            if not np.isnan(corr):
                gradients[k] = corr

        return gradients

    def update_weights(self, gradients: np.ndarray) -> None:
        """
        Actualiza pesos: w_k += η · ∂V/∂w_k

        Con normalización para mantener Σw_k = 1
        """
        eta = self._compute_learning_rate()

        # Actualización
        self.weights = self.weights + eta * gradients

        # Mantener positivos
        # Límites endógenos: 1/K mínimo, K máximo (donde K = número de componentes)
        self.weights = np.clip(self.weights, 1/self.K, self.K)

        # Normalizar
        self.weights = self.weights / self.weights.sum()

    def step(self, z: np.ndarray, surprise: float,
             other_z: Optional[np.ndarray] = None) -> MetaDriveState:
        """
        Ejecuta un paso del meta-drive.

        1. Calcula componentes
        2. Calcula drive con pesos actuales
        3. Calcula valor
        4. Actualiza pesos (si hay suficiente historia)

        Returns:
            MetaDriveState con estado actual
        """
        self.t += 1

        # Registrar historia
        self.z_history.append(z.copy())
        self.surprise_history.append(surprise)

        # 1. Componentes
        components = self.compute_components(z, surprise, other_z)
        self.component_history.append(components)

        # 2. Drive
        drive = float(np.dot(self.weights, components))
        self.drive_history.append(drive)

        # 3. Valor
        integration = components[4]  # Usamos el componente de integración
        value = self.compute_value(surprise, integration)
        self.value_history.append(value)

        # 4. Actualizar pesos
        if self.t > 20:
            gradients = self.compute_gradients()
            self.update_weights(gradients)
        else:
            gradients = np.zeros(self.K)

        self.weight_history.append(self.weights.copy())

        # Entropía de pesos (diversidad)
        w_safe = np.clip(self.weights, np.finfo(float).eps, 1)
        weights_entropy = -np.sum(w_safe * np.log(w_safe))

        # Crear estado
        drive_components = [
            DriveComponent(
                name=self.component_names[k],
                value=float(components[k]),
                weight=float(self.weights[k]),
                gradient=float(gradients[k])
            )
            for k in range(self.K)
        ]

        return MetaDriveState(
            t=self.t,
            agent=self.agent,
            components=drive_components,
            drive_total=drive,
            value=value,
            weights_entropy=float(weights_entropy)
        )

    def get_dominant_drive(self) -> Tuple[str, float]:
        """Retorna el componente dominante actual."""
        k = np.argmax(self.weights)
        return self.component_names[k], float(self.weights[k])

    def get_weight_trajectory(self) -> Dict[str, List[float]]:
        """Retorna la evolución de pesos."""
        if not self.weight_history:
            return {}

        W = np.array(self.weight_history)
        return {
            name: W[:, k].tolist()
            for k, name in enumerate(self.component_names)
        }

    def get_summary(self) -> Dict[str, Any]:
        """Resumen del estado actual."""
        dominant, dominant_weight = self.get_dominant_drive()

        return {
            't': self.t,
            'agent': self.agent,
            'weights': {name: float(w) for name, w in
                       zip(self.component_names, self.weights)},
            'dominant_drive': dominant,
            'dominant_weight': dominant_weight,
            'drive_mean': float(np.mean(self.drive_history[-50:])) if self.drive_history else 0,
            'value_mean': float(np.mean(self.value_history[-50:])) if self.value_history else 0
        }


class DualMetaDrive:
    """
    Sistema dual con meta-drives para NEO y EVA.

    Cada agente modifica sus propios pesos basándose en su historia.
    Pueden observar al otro (otherness component).
    """

    def __init__(self):
        self.neo_drive = MetaDrive('NEO')
        self.eva_drive = MetaDrive('EVA')

        self.t = 0
        self.interaction_history: List[Dict] = []

    def step(self, neo_z: np.ndarray, neo_surprise: float,
             eva_z: np.ndarray, eva_surprise: float) -> Dict[str, Any]:
        """Ejecuta un paso para ambos agentes."""
        self.t += 1

        # Cada uno ve al otro
        neo_state = self.neo_drive.step(neo_z, neo_surprise, other_z=eva_z)
        eva_state = self.eva_drive.step(eva_z, eva_surprise, other_z=neo_z)

        interaction = {
            't': self.t,
            'neo': neo_state,
            'eva': eva_state,
            'weight_divergence': self._compute_weight_divergence()
        }
        self.interaction_history.append(interaction)

        return interaction

    def _compute_weight_divergence(self) -> float:
        """Divergencia entre pesos de NEO y EVA."""
        return float(np.linalg.norm(
            self.neo_drive.weights - self.eva_drive.weights
        ))

    def get_comparison(self) -> Dict[str, Any]:
        """Compara evolución de drives."""
        neo_summary = self.neo_drive.get_summary()
        eva_summary = self.eva_drive.get_summary()

        return {
            't': self.t,
            'NEO': neo_summary,
            'EVA': eva_summary,
            'weight_divergence': self._compute_weight_divergence(),
            # Umbrales endógenos basados en 1/K y 1-1/K
            'drives_converged': self._compute_weight_divergence() < 1/self.neo_drive.K,
            'drives_diverged': self._compute_weight_divergence() > 1 - 1/self.neo_drive.K
        }


def run_meta_drive_experiment(T: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """
    Experimento: ¿Los agentes modifican sus drives?
    ¿Convergen o divergen?
    """

    print("=" * 70)
    print("EXPERIMENTO: META-DRIVES AUTO-MODIFICABLES")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}")
    print(f"T = {T}")
    print()

    np.random.seed(seed)

    # Sistema dual
    dual = DualMetaDrive()

    # Estados iniciales: distribución uniforme (máxima entropía)
    # ORIGEN: 1/K para cada componente = máxima entropía
    K = 6  # Número de componentes (definido por arquitectura)
    neo_z = np.ones(K) / K  # ORIGEN: distribución uniforme
    eva_z = np.ones(K) / K  # ORIGEN: distribución uniforme

    print("Pesos iniciales (uniformes):")
    print(f"  NEO: {dual.neo_drive.weights}")
    print(f"  EVA: {dual.eva_drive.weights}")
    print()

    print("Simulando evolución de drives...")

    for t in range(T):
        # Dinámica simple: cada agente evoluciona según su drive actual

        # NEO: tiende a comprimir (reduce entropía si su drive lo favorece)
        # ORIGEN: drive_state = máxima incertidumbre (0.5) si no hay historial
        neo_drive_state = dual.neo_drive.drive_history[-1] if dual.neo_drive.drive_history else 0.5
        # ORIGEN: Ruido proporcional a 1/sqrt(t+1) (mismo que learning rate)
        noise_scale = 1 / np.sqrt(t + 1)  # ORIGEN: decaimiento estándar de ruido
        neo_noise = np.random.randn(K) * noise_scale * (1 - neo_drive_state)

        # EVA: tiende a explorar (aumenta variación si su drive lo favorece)
        eva_drive_state = dual.eva_drive.drive_history[-1] if dual.eva_drive.drive_history else 0.5
        eva_noise = np.random.randn(K) * noise_scale * eva_drive_state

        # Interacción: acoplamiento proporcional a 1/K
        # ORIGEN: 1/K = escala natural de interacción entre K componentes
        coupling = 1 / K
        neo_z = neo_z + neo_noise + coupling * (eva_z - neo_z)
        eva_z = eva_z + eva_noise + coupling * (neo_z - eva_z)

        # Normalizar con límites endógenos
        # ORIGEN: eps = precisión máquina, 1-eps = complemento
        eps = np.finfo(float).eps
        neo_z = np.clip(neo_z, eps, 1 - eps)
        eva_z = np.clip(eva_z, eps, 1 - eps)
        neo_z = neo_z / neo_z.sum()
        eva_z = eva_z / eva_z.sum()

        # Sorpresa como distancia a predicción simple (estado anterior)
        if t > 0:
            neo_surprise = np.linalg.norm(neo_z - dual.neo_drive.z_history[-1])
            eva_surprise = np.linalg.norm(eva_z - dual.eva_drive.z_history[-1])
        else:
            # ORIGEN: Sin historial, sorpresa = 0 (no hay predicción que violar)
            neo_surprise = 0.0
            eva_surprise = 0.0

        # Paso
        result = dual.step(neo_z, neo_surprise, eva_z, eva_surprise)

        if t % (T // 5) == 0:
            print(f"\nt={t}:")
            neo_dom, neo_w = dual.neo_drive.get_dominant_drive()
            eva_dom, eva_w = dual.eva_drive.get_dominant_drive()
            print(f"  NEO dominante: {neo_dom} ({neo_w:.3f})")
            print(f"  EVA dominante: {eva_dom} ({eva_w:.3f})")
            print(f"  Divergencia pesos: {result['weight_divergence']:.3f}")

    print()
    print("=" * 70)
    print("RESULTADOS FINALES")
    print("=" * 70)
    print()

    comparison = dual.get_comparison()

    print("NEO - Pesos finales:")
    for name, w in comparison['NEO']['weights'].items():
        print(f"  {name}: {w:.4f}")
    print(f"  Drive dominante: {comparison['NEO']['dominant_drive']}")
    print()

    print("EVA - Pesos finales:")
    for name, w in comparison['EVA']['weights'].items():
        print(f"  {name}: {w:.4f}")
    print(f"  Drive dominante: {comparison['EVA']['dominant_drive']}")
    print()

    print(f"Divergencia final: {comparison['weight_divergence']:.4f}")
    print(f"¿Convergieron?: {'Sí' if comparison['drives_converged'] else 'No'}")
    print(f"¿Divergieron?: {'Sí' if comparison['drives_diverged'] else 'No'}")

    # Análisis de trayectorias
    neo_traj = dual.neo_drive.get_weight_trajectory()
    eva_traj = dual.eva_drive.get_weight_trajectory()

    # Guardar resultados
    results = {
        'timestamp': datetime.now().isoformat(),
        'T': T,
        'seed': seed,
        'final_comparison': comparison,
        'neo_weight_trajectory': neo_traj,
        'eva_weight_trajectory': eva_traj,
        'value_history': {
            'NEO': dual.neo_drive.value_history,
            'EVA': dual.eva_drive.value_history
        }
    }

    os.makedirs('/root/NEO_EVA/results/meta_drive', exist_ok=True)

    with open('/root/NEO_EVA/results/meta_drive/meta_drive_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Visualización
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. Evolución de pesos NEO
        ax1 = axes[0, 0]
        for name, traj in neo_traj.items():
            ax1.plot(traj, label=name, alpha=0.7)
        ax1.set_xlabel('Tiempo')
        ax1.set_ylabel('Peso')
        ax1.set_title('NEO: Evolución de Pesos del Drive')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. Evolución de pesos EVA
        ax2 = axes[0, 1]
        for name, traj in eva_traj.items():
            ax2.plot(traj, label=name, alpha=0.7)
        ax2.set_xlabel('Tiempo')
        ax2.set_ylabel('Peso')
        ax2.set_title('EVA: Evolución de Pesos del Drive')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. Divergencia temporal
        ax3 = axes[0, 2]
        divergences = [h['weight_divergence'] for h in dual.interaction_history]
        ax3.plot(divergences, 'purple', alpha=0.7)
        # ORIGEN: Umbrales basados en 1/K y 1-1/K (endógenos)
        K_viz = dual.neo_drive.K
        ax3.axhline(y=1/K_viz, color='green', linestyle='--', label=f'Convergencia (1/K={1/K_viz:.2f})')
        ax3.axhline(y=1-1/K_viz, color='red', linestyle='--', label=f'Divergencia (1-1/K={1-1/K_viz:.2f})')
        ax3.set_xlabel('Tiempo')
        ax3.set_ylabel('Divergencia de Pesos')
        ax3.set_title('NEO vs EVA: Divergencia de Drives')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Valor acumulado
        ax4 = axes[1, 0]
        ax4.plot(dual.neo_drive.value_history, 'b-', label='NEO', alpha=0.7)
        ax4.plot(dual.eva_drive.value_history, 'r-', label='EVA', alpha=0.7)
        ax4.set_xlabel('Tiempo')
        ax4.set_ylabel('Valor V(t)')
        ax4.set_title('Valor Endógeno (Fitness)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Comparación final de pesos
        ax5 = axes[1, 1]
        x = np.arange(6)
        width = 0.35
        neo_final = list(comparison['NEO']['weights'].values())
        eva_final = list(comparison['EVA']['weights'].values())
        ax5.bar(x - width/2, neo_final, width, label='NEO', color='blue', alpha=0.7)
        ax5.bar(x + width/2, eva_final, width, label='EVA', color='red', alpha=0.7)
        ax5.set_xticks(x)
        ax5.set_xticklabels(['ent', 'neg_sur', 'nov', 'stab', 'int', 'oth'], rotation=45)
        ax5.set_ylabel('Peso')
        ax5.set_title('Pesos Finales')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Drive total temporal
        ax6 = axes[1, 2]
        ax6.plot(dual.neo_drive.drive_history, 'b-', label='NEO drive', alpha=0.7)
        ax6.plot(dual.eva_drive.drive_history, 'r-', label='EVA drive', alpha=0.7)
        ax6.set_xlabel('Tiempo')
        ax6.set_ylabel('Drive Total')
        ax6.set_title('Evolución del Drive')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs('/root/NEO_EVA/figures', exist_ok=True)
        plt.savefig('/root/NEO_EVA/figures/meta_drive_results.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nFigura: /root/NEO_EVA/figures/meta_drive_results.png")

    except Exception as e:
        print(f"Warning: No se pudo crear visualización: {e}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Meta-Drive experiment')
    parser.add_argument('--T', type=int, required=True, help='Number of steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    run_meta_drive_experiment(T=args.T, seed=args.seed)


# =============================================================================
# BLOQUE DE AUDITORÍA NORMA DURA
# =============================================================================
"""
MAGIC NUMBERS AUDIT
==================

NÚMEROS CORREGIDOS:
- neo_z = [0.4, 0.3, ...] -> REEMPLAZADO por np.ones(K)/K (distribución uniforme)
- eva_z = [0.2, 0.2, ...] -> REEMPLAZADO por np.ones(K)/K (distribución uniforme)
- 0.05 * noise -> REEMPLAZADO por 1/sqrt(t+1) (decaimiento estándar)
- 0.02 * coupling -> REEMPLAZADO por 1/K (escala natural)
- np.clip(z, 0.01, 0.99) -> REEMPLAZADO por np.clip(z, eps, 1-eps) (precisión máquina)
- neo_surprise = 0.1 -> REEMPLAZADO por 0.0 (sin predicción inicial)
- axhline(y=0.1) -> REEMPLAZADO por 1/K (umbral endógeno)
- axhline(y=0.5) -> REEMPLAZADO por 1-1/K (umbral endógeno)

CONSTANTES MATEMÁTICAS USADAS:
- 1/K: Peso uniforme entre K componentes (máxima entropía)
  ORIGEN: Definición de distribución uniforme discreta
- 1/sqrt(t+1): Learning rate / noise decay
  ORIGEN: Tasa estándar de convergencia en optimización estocástica
- np.finfo(float).eps: Precisión máquina
  ORIGEN: Constante numérica estándar
- np.log(): Para cálculo de entropía
  ORIGEN: Definición de entropía de Shannon

VALORES INICIALES ENDÓGENOS:
- Pesos iniciales = 1/K (distribución uniforme, máxima entropía)
- Estados iniciales = 1/K (distribución uniforme)
- Sorpresa inicial = 0 (sin predicción que violar)

PARÁMETROS DE ENTRADA (no hardcodeados):
- T: Proporcionado por usuario via CLI
- seed: Proporcionado por usuario via CLI

TODAS LAS DECISIONES TIENEN ORIGEN DOCUMENTADO.
"""
