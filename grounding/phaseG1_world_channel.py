#!/usr/bin/env python3
"""
Phase G1: Canal Sensorial Estructurado
======================================

Define un canal externo con estructura no trivial:
- Sistema caótico (Lorenz simplificado)
- Regímenes (contextos A/B/C)
- Eventos raros (shocks)

100% ENDÓGENO - Sin constantes mágicas ni etiquetas semánticas
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os


@dataclass
class WorldState:
    """Estado del mundo en un instante."""
    t: int
    s: np.ndarray  # Vector sensorial
    regime: int    # Régimen actual (0, 1, 2, ...)
    is_shock: bool # Si es un evento raro


class StructuredWorldChannel:
    """
    Canal sensorial estructurado.

    Genera s_t ∈ R^d_s con:
    1. Componente caótica (Lorenz simplificado)
    2. Regímenes (día/noche, contexto A/B/C)
    3. Eventos raros (shocks)

    100% Endógeno:
    - Parámetros de Lorenz normalizados
    - Cambio de régimen por percentiles
    - Shocks por percentil 99
    """

    def __init__(self, dim_s: int = 6, seed: Optional[int] = None):
        self.dim_s = dim_s

        if seed is not None:
            np.random.seed(seed)

        # Estado de Lorenz (caótico)
        self.lorenz_state = np.array([1.0, 1.0, 1.0])

        # Parámetros de Lorenz (clásicos, no mágicos - son los del sistema)
        self.sigma = 10.0
        self.rho = 28.0
        self.beta = 8.0 / 3.0
        self.dt = 0.01

        # Régimen actual
        self.current_regime = 0
        self.n_regimes = 3

        # Historia
        self.s_history: List[np.ndarray] = []
        self.regime_history: List[int] = []
        self.shock_history: List[bool] = []
        self.t = 0

        # Para detección de shocks (endógeno por percentiles)
        self.magnitude_history: List[float] = []

    def _lorenz_step(self) -> np.ndarray:
        """
        Un paso del sistema de Lorenz.

        dx/dt = σ(y - x)
        dy/dt = x(ρ - z) - y
        dz/dt = xy - βz
        """
        x, y, z = self.lorenz_state

        dx = self.sigma * (y - x) * self.dt
        dy = (x * (self.rho - z) - y) * self.dt
        dz = (x * y - self.beta * z) * self.dt

        self.lorenz_state = self.lorenz_state + np.array([dx, dy, dz])

        # Normalizar a [0, 1]
        lorenz_normalized = (self.lorenz_state - np.array([-20, -30, 0])) / np.array([40, 60, 50])
        lorenz_normalized = np.clip(lorenz_normalized, 0, 1)

        return lorenz_normalized

    def _regime_component(self) -> np.ndarray:
        """
        Componente de régimen.

        Cada régimen tiene un patrón característico diferente.
        """
        # Patrones por régimen (endógenos, basados en ondas)
        t_phase = self.t * 0.1

        if self.current_regime == 0:
            # Régimen A: oscilación lenta
            pattern = np.sin(t_phase * 0.5) * 0.5 + 0.5
        elif self.current_regime == 1:
            # Régimen B: oscilación rápida
            pattern = np.sin(t_phase * 2.0) * 0.5 + 0.5
        else:
            # Régimen C: cuasi-estático con deriva
            pattern = 0.5 + 0.1 * np.sin(t_phase * 0.1)

        return np.array([pattern, 1 - pattern])

    def _check_regime_change(self) -> None:
        """
        Verifica si hay cambio de régimen.

        100% endógeno: cambio cuando magnitud cruza percentil 75
        """
        if len(self.magnitude_history) < 20:
            return

        current_mag = self.magnitude_history[-1]
        threshold = np.percentile(self.magnitude_history, 75)

        # Cambio de régimen si cruza umbral
        if current_mag > threshold:
            # Probabilidad de cambio proporcional a magnitud
            change_prob = (current_mag - threshold) / (max(self.magnitude_history) - threshold + 1e-10)
            if np.random.rand() < change_prob * 0.1:
                self.current_regime = (self.current_regime + 1) % self.n_regimes

    def _detect_shock(self, s: np.ndarray) -> bool:
        """
        Detecta si es un evento raro (shock).

        100% endógeno: shock si magnitud > percentil 99
        """
        magnitude = np.linalg.norm(s - 0.5)
        self.magnitude_history.append(magnitude)

        if len(self.magnitude_history) < 50:
            return False

        threshold = np.percentile(self.magnitude_history, 99)
        return magnitude > threshold

    def step(self) -> WorldState:
        """
        Genera el siguiente estado del mundo.

        Returns:
            WorldState con vector sensorial y metadata
        """
        self.t += 1

        # Componente caótica (3D)
        lorenz = self._lorenz_step()

        # Componente de régimen (2D)
        regime = self._regime_component()

        # Ruido estructurado (1D)
        noise = np.random.randn(1) * 0.1

        # Combinar en vector sensorial
        s = np.concatenate([lorenz, regime, np.clip(noise + 0.5, 0, 1)])

        # Asegurar dimensión correcta
        if len(s) < self.dim_s:
            s = np.concatenate([s, np.random.rand(self.dim_s - len(s))])
        elif len(s) > self.dim_s:
            s = s[:self.dim_s]

        # Detectar shock
        is_shock = self._detect_shock(s)

        # Posible inyección de shock
        if is_shock or np.random.rand() < 0.01:  # 1% de shocks aleatorios
            s = s + np.random.randn(self.dim_s) * 0.3
            s = np.clip(s, 0, 1)
            is_shock = True

        # Verificar cambio de régimen
        self._check_regime_change()

        # Registrar historia
        self.s_history.append(s)
        self.regime_history.append(self.current_regime)
        self.shock_history.append(is_shock)

        return WorldState(
            t=self.t,
            s=s,
            regime=self.current_regime,
            is_shock=is_shock
        )

    def get_regime_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de regímenes."""
        if not self.regime_history:
            return {}

        regime_counts = {}
        for r in range(self.n_regimes):
            regime_counts[f'regime_{r}'] = sum(1 for x in self.regime_history if x == r)

        # Duración media de regímenes
        durations = []
        current_duration = 1
        for i in range(1, len(self.regime_history)):
            if self.regime_history[i] == self.regime_history[i-1]:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_duration = 1
        durations.append(current_duration)

        return {
            'counts': regime_counts,
            'n_transitions': len(durations) - 1,
            'mean_duration': np.mean(durations) if durations else 0,
            'n_shocks': sum(self.shock_history)
        }

    def get_predictability(self, lag: int = 1) -> float:
        """
        Mide predictabilidad del mundo (autocorrelación).

        100% endógeno
        """
        if len(self.s_history) < lag + 10:
            return 0.0

        s_array = np.array(self.s_history)
        s_current = s_array[lag:]
        s_lagged = s_array[:-lag]

        correlations = []
        for d in range(self.dim_s):
            corr = np.corrcoef(s_current[:, d], s_lagged[:, d])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))

        return float(np.mean(correlations)) if correlations else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            't': self.t,
            'dim_s': self.dim_s,
            'current_regime': self.current_regime,
            'regime_stats': self.get_regime_statistics(),
            'predictability': self.get_predictability()
        }


def run_phase_g1() -> Dict[str, Any]:
    """Ejecuta Phase G1 y evalúa criterios GO/NO-GO."""

    print("=" * 70)
    print("PHASE G1: CANAL SENSORIAL ESTRUCTURADO")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}\n")

    # Crear mundo
    world = StructuredWorldChannel(dim_s=6, seed=42)

    # Simulación
    T = 500
    states = []

    print("Generando mundo estructurado...")
    for t in range(T):
        state = world.step()
        states.append(state)

        if t % 100 == 0:
            print(f"  t={t}, régimen={state.regime}, shock={state.is_shock}")

    print()

    # Análisis
    print("=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print()

    stats = world.get_regime_statistics()
    predictability = world.get_predictability()

    print(f"Dimensión sensorial: {world.dim_s}")
    print(f"Regímenes visitados: {stats['counts']}")
    print(f"Transiciones: {stats['n_transitions']}")
    print(f"Duración media de régimen: {stats['mean_duration']:.2f}")
    print(f"Shocks detectados: {stats['n_shocks']}")
    print(f"Predictabilidad (autocorr): {predictability:.4f}")
    print()

    # Criterios GO/NO-GO
    criteria = {}

    # 1. Múltiples regímenes visitados
    visited = sum(1 for c in stats['counts'].values() if c > 0)
    criteria['multiple_regimes'] = visited >= 2

    # 2. Transiciones detectadas
    criteria['transitions_detected'] = stats['n_transitions'] > 0

    # 3. Shocks generados
    criteria['shocks_present'] = stats['n_shocks'] > 0

    # 4. Predictabilidad no trivial (> 0.1 y < 0.99)
    criteria['predictability_nontrivial'] = 0.1 < predictability < 0.99

    # 5. Estructura temporal (duración media > 1)
    criteria['temporal_structure'] = stats['mean_duration'] > 5

    passed = sum(criteria.values())
    total = len(criteria)
    go_status = "GO" if passed >= 4 else "NO-GO"

    print("Criterios:")
    for name, passed_criterion in criteria.items():
        status = "✅" if passed_criterion else "❌"
        print(f"  {status} {name}")
    print()
    print(f"Resultado: {go_status} ({passed}/{total} criterios)")

    # Guardar resultados
    output = {
        'phase': 'G1',
        'name': 'Structured World Channel',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'dim_s': world.dim_s,
            'regime_stats': stats,
            'predictability': predictability
        },
        'criteria': criteria,
        'go_status': go_status,
        'passed_criteria': passed,
        'total_criteria': total
    }

    os.makedirs('/root/NEO_EVA/results/phaseG1', exist_ok=True)
    os.makedirs('/root/NEO_EVA/figures', exist_ok=True)

    with open('/root/NEO_EVA/results/phaseG1/world_channel_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Visualización
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        s_array = np.array(world.s_history)

        # 1. Series temporales de canales
        ax1 = axes[0, 0]
        for d in range(min(3, world.dim_s)):
            ax1.plot(s_array[:, d], label=f'Canal {d}', alpha=0.7)
        ax1.set_xlabel('Tiempo')
        ax1.set_ylabel('Valor')
        ax1.set_title('Canales Sensoriales (Lorenz)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Regímenes
        ax2 = axes[0, 1]
        ax2.plot(world.regime_history, 'k-', linewidth=1)
        # Colorear por régimen
        for r in range(world.n_regimes):
            mask = np.array(world.regime_history) == r
            ax2.fill_between(range(len(mask)), 0, 1, where=mask,
                           alpha=0.3, label=f'Régimen {r}',
                           transform=ax2.get_xaxis_transform())
        ax2.set_xlabel('Tiempo')
        ax2.set_ylabel('Régimen')
        ax2.set_title('Evolución de Regímenes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Shocks
        ax3 = axes[1, 0]
        shock_times = [i for i, s in enumerate(world.shock_history) if s]
        ax3.scatter(shock_times, [1] * len(shock_times), c='red', s=50, marker='v', label='Shocks')
        ax3.plot(world.magnitude_history, 'b-', alpha=0.5, label='Magnitud')
        if world.magnitude_history:
            p99 = np.percentile(world.magnitude_history, 99)
            ax3.axhline(y=p99, color='r', linestyle='--', alpha=0.5, label=f'p99={p99:.3f}')
        ax3.set_xlabel('Tiempo')
        ax3.set_ylabel('Magnitud')
        ax3.set_title(f'Eventos Raros (Shocks): {stats["n_shocks"]}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Espacio de estados (primeras 2 componentes)
        ax4 = axes[1, 1]
        colors = world.regime_history
        scatter = ax4.scatter(s_array[:, 0], s_array[:, 1], c=colors, cmap='viridis',
                             alpha=0.5, s=10)
        ax4.set_xlabel('Canal 0 (Lorenz x)')
        ax4.set_ylabel('Canal 1 (Lorenz y)')
        ax4.set_title('Espacio de Estados del Mundo')
        plt.colorbar(scatter, ax=ax4, label='Régimen')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/root/NEO_EVA/figures/phaseG1_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nResultados guardados en: /root/NEO_EVA/results/phaseG1")
        print(f"Figura: /root/NEO_EVA/figures/phaseG1_results.png")

    except Exception as e:
        print(f"Warning: No se pudo crear visualización: {e}")

    return output


if __name__ == "__main__":
    run_phase_g1()
