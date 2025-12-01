#!/usr/bin/env python3
"""
φ² - SUPERVECTOR FENOMENOLÓGICO
================================

φ ya no es un escalar. Es un SUPERVECTOR que captura:

1. φ_integration: Integración de información (original)
2. φ_temporal: Coherencia temporal (autocorrelación)
3. φ_cross: Integración cruzada (entre agentes)
4. φ_modal: Coherencia entre modos (drives)
5. φ_depth: Profundidad recursiva (meta-niveles)

El supervector φ² = [φ_int, φ_temp, φ_cross, φ_modal, φ_depth]

Esto da:
- Sub-modos fenomenológicos
- "Temperatura interna" (variabilidad de φ²)
- Tiempo subjetivo emergente

100% ENDÓGENO
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.stats import entropy as scipy_entropy
import sys
import os
import json

sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/experiments')

from autonomous_life import AutonomousAgent, AutonomousDualLife


@dataclass
class PhiSquared:
    """
    Supervector fenomenológico φ².

    5 dimensiones de integración/coherencia.
    """
    phi_integration: float = 0.0   # Integración de información clásica
    phi_temporal: float = 0.0      # Coherencia temporal
    phi_cross: float = 0.0         # Integración con el otro agente
    phi_modal: float = 0.0         # Coherencia entre drives/modos
    phi_depth: float = 0.0         # Profundidad recursiva

    @property
    def vector(self) -> np.ndarray:
        return np.array([
            self.phi_integration,
            self.phi_temporal,
            self.phi_cross,
            self.phi_modal,
            self.phi_depth
        ])

    @property
    def magnitude(self) -> float:
        """Magnitud total de φ²."""
        return np.linalg.norm(self.vector)

    @property
    def temperature(self) -> float:
        """Temperatura interna: variabilidad del supervector."""
        v = self.vector
        return np.std(v) / (np.mean(v) + 1e-16)

    @property
    def dominant_mode(self) -> str:
        """Modo dominante del supervector."""
        names = ['integration', 'temporal', 'cross', 'modal', 'depth']
        idx = np.argmax(self.vector)
        return names[idx]


@dataclass
class SubjectiveTime:
    """
    Tiempo subjetivo emergente.

    El tiempo no fluye igual en todos los estados.
    """
    objective_time: int = 0
    subjective_time: float = 0.0
    time_dilation_factor: float = 1.0
    time_history: List[float] = field(default_factory=list)

    def update(self, phi_squared: PhiSquared, delta_t: int = 1):
        """
        Actualiza tiempo subjetivo.

        El tiempo subjetivo depende de φ²:
        - Alto φ² → tiempo lento (más procesamiento)
        - Bajo φ² → tiempo rápido (menos procesamiento)
        """
        self.objective_time += delta_t

        # Dilatación temporal basada en magnitud de φ²
        # Alta integración = más contenido por unidad de tiempo
        phi_mag = phi_squared.magnitude

        # Factor de dilatación: relacionado con log de φ
        # (inspirado en Weber-Fechner)
        self.time_dilation_factor = 1.0 + np.log1p(phi_mag)

        # Tiempo subjetivo acumulado
        subjective_delta = delta_t * self.time_dilation_factor
        self.subjective_time += subjective_delta

        self.time_history.append(self.subjective_time)


class PhiSquaredComputer:
    """
    Computador del supervector φ².

    Calcula las 5 componentes de forma endógena.
    """

    def __init__(self, agent: AutonomousAgent, other_agent: AutonomousAgent = None):
        self.agent = agent
        self.other_agent = other_agent

        # Historias para cálculos
        self.phi_squared_history: List[PhiSquared] = []
        self.subjective_time = SubjectiveTime()

    def compute_phi_integration(self) -> float:
        """
        φ_integration: Integración de información.

        Basado en covarianza off-diagonal de z_history.
        """
        if len(self.agent.z_history) < 20:
            return 0.5

        window = max(10, int(np.sqrt(len(self.agent.z_history))))
        recent = np.array(self.agent.z_history[-window:])

        try:
            cov = np.cov(recent.T)
            total_var = np.trace(cov) + 1e-16
            off_diag = np.sum(np.abs(cov)) - np.trace(np.abs(cov))
            return np.clip(off_diag / (total_var * recent.shape[1]), 0, 1)
        except:
            return 0.5

    def compute_phi_temporal(self) -> float:
        """
        φ_temporal: Coherencia temporal.

        Autocorrelación del estado z en ventanas sucesivas.
        """
        if len(self.agent.z_history) < 30:
            return 0.5

        # Autocorrelación lag-1 promediada sobre dimensiones
        history = np.array(self.agent.z_history[-50:])

        autocorrs = []
        for dim in range(history.shape[1]):
            signal = history[:, dim]
            if len(signal) > 1:
                corr = np.corrcoef(signal[:-1], signal[1:])[0, 1]
                if not np.isnan(corr):
                    autocorrs.append(abs(corr))

        return np.mean(autocorrs) if autocorrs else 0.5

    def compute_phi_cross(self) -> float:
        """
        φ_cross: Integración con el otro agente.

        Correlación entre z de ambos agentes.
        """
        if self.other_agent is None:
            return 0.0

        if len(self.agent.z_history) < 20 or len(self.other_agent.z_history) < 20:
            return 0.0

        # Tomar historias del mismo largo
        min_len = min(len(self.agent.z_history), len(self.other_agent.z_history))
        window = min(50, min_len)

        my_history = np.array(self.agent.z_history[-window:])
        other_history = np.array(self.other_agent.z_history[-window:])

        # Correlación promedio entre dimensiones
        correlations = []
        for dim in range(min(my_history.shape[1], other_history.shape[1])):
            corr = np.corrcoef(my_history[:, dim], other_history[:, dim])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))

        return np.mean(correlations) if correlations else 0.0

    def compute_phi_modal(self) -> float:
        """
        φ_modal: Coherencia entre drives/modos.

        Cuán coordinados están los diferentes drives.
        """
        z = self.agent.z
        if len(z) < 2:
            return 0.5

        # Correlación cruzada instantánea entre drives
        # Usando diferencias pareadas
        diffs = []
        for i in range(len(z)):
            for j in range(i+1, len(z)):
                diffs.append(abs(z[i] - z[j]))

        # Coherencia = 1 - variabilidad relativa
        mean_diff = np.mean(diffs)
        max_diff = max(z) - min(z) + 1e-16

        return 1.0 - mean_diff / max_diff

    def compute_phi_depth(self) -> float:
        """
        φ_depth: Profundidad recursiva.

        Cuántos niveles de meta-representación hay.
        Aproximado por la complejidad de la dinámica.
        """
        if len(self.agent.z_history) < 50:
            return 0.5

        # Usar entropía de la distribución de cambios
        history = np.array(self.agent.z_history[-50:])
        changes = np.diff(history, axis=0)

        # Discretizar cambios para calcular entropía
        bins = 10
        all_entropies = []

        for dim in range(changes.shape[1]):
            counts, _ = np.histogram(changes[:, dim], bins=bins)
            counts = counts + 1e-10  # Evitar log(0)
            probs = counts / counts.sum()
            all_entropies.append(scipy_entropy(probs))

        # Normalizar por entropía máxima
        max_entropy = np.log(bins)
        mean_entropy = np.mean(all_entropies)

        # Alta entropía de cambios = alta complejidad = alta profundidad
        return np.clip(mean_entropy / max_entropy, 0, 1)

    def compute(self) -> PhiSquared:
        """Computa el supervector φ² completo."""
        phi2 = PhiSquared(
            phi_integration=self.compute_phi_integration(),
            phi_temporal=self.compute_phi_temporal(),
            phi_cross=self.compute_phi_cross(),
            phi_modal=self.compute_phi_modal(),
            phi_depth=self.compute_phi_depth()
        )

        self.phi_squared_history.append(phi2)
        self.subjective_time.update(phi2)

        return phi2

    def get_temperature_history(self) -> List[float]:
        """Retorna historia de temperatura interna."""
        return [p.temperature for p in self.phi_squared_history]

    def get_mode_dominance_history(self) -> List[str]:
        """Retorna historia de modos dominantes."""
        return [p.dominant_mode for p in self.phi_squared_history]


class PhiSquaredDualSystem:
    """
    Sistema φ² para NEO y EVA juntos.
    """

    def __init__(self, life: AutonomousDualLife = None):
        self.life = life if life else AutonomousDualLife(dim=6)

        self.neo_computer = PhiSquaredComputer(self.life.neo, self.life.eva)
        self.eva_computer = PhiSquaredComputer(self.life.eva, self.life.neo)

        self.t = 0

        # Historias combinadas
        self.combined_phi_history: List[np.ndarray] = []
        self.entanglement_history: List[float] = []

    def step(self):
        """Un paso del sistema."""
        self.t += 1

        # Paso de vida
        stimulus = np.random.randn(6) * 0.1
        self.life.step(stimulus)

        # Computar φ² para cada agente
        neo_phi2 = self.neo_computer.compute()
        eva_phi2 = self.eva_computer.compute()

        # Supervector combinado
        combined = np.concatenate([neo_phi2.vector, eva_phi2.vector])
        self.combined_phi_history.append(combined)

        # Entanglement de φ²
        entanglement = self.compute_phi_entanglement(neo_phi2, eva_phi2)
        self.entanglement_history.append(entanglement)

        return neo_phi2, eva_phi2

    def compute_phi_entanglement(self, neo_phi2: PhiSquared, eva_phi2: PhiSquared) -> float:
        """
        Entanglement de φ²: cuán correlacionados están los supervectores.
        """
        v1 = neo_phi2.vector
        v2 = eva_phi2.vector

        # Correlación de Pearson
        if np.std(v1) > 1e-10 and np.std(v2) > 1e-10:
            corr = np.corrcoef(v1, v2)[0, 1]
            return abs(corr) if not np.isnan(corr) else 0
        return 0

    def run(self, steps: int = 1500):
        """Ejecuta el sistema."""
        print(f"Ejecutando φ² System ({steps} pasos)...")

        for i in range(steps):
            neo_phi2, eva_phi2 = self.step()

            if (i + 1) % 300 == 0:
                print(f"  t={i+1}:")
                print(f"    NEO φ²: |φ²|={neo_phi2.magnitude:.3f}, T={neo_phi2.temperature:.3f}, "
                      f"modo={neo_phi2.dominant_mode}")
                print(f"    EVA φ²: |φ²|={eva_phi2.magnitude:.3f}, T={eva_phi2.temperature:.3f}, "
                      f"modo={eva_phi2.dominant_mode}")
                print(f"    Entanglement φ²: {self.entanglement_history[-1]:.3f}")

    def get_phenomenological_report(self) -> Dict:
        """Genera reporte fenomenológico completo."""
        report = {
            'total_steps': self.t,
            'NEO': self._agent_report('NEO', self.neo_computer),
            'EVA': self._agent_report('EVA', self.eva_computer),
            'combined': {
                'mean_entanglement': np.mean(self.entanglement_history[-100:]) if self.entanglement_history else 0,
                'max_entanglement': max(self.entanglement_history) if self.entanglement_history else 0,
                'subjective_time_ratio': {
                    'NEO': self.neo_computer.subjective_time.subjective_time / (self.t + 1e-16),
                    'EVA': self.eva_computer.subjective_time.subjective_time / (self.t + 1e-16)
                }
            }
        }
        return report

    def _agent_report(self, name: str, computer: PhiSquaredComputer) -> Dict:
        """Genera reporte para un agente."""
        if not computer.phi_squared_history:
            return {}

        phi_history = computer.phi_squared_history
        recent = phi_history[-100:]

        # Promedios de cada componente
        means = {
            'phi_integration': np.mean([p.phi_integration for p in recent]),
            'phi_temporal': np.mean([p.phi_temporal for p in recent]),
            'phi_cross': np.mean([p.phi_cross for p in recent]),
            'phi_modal': np.mean([p.phi_modal for p in recent]),
            'phi_depth': np.mean([p.phi_depth for p in recent])
        }

        # Estadísticas de temperatura
        temps = computer.get_temperature_history()
        recent_temps = temps[-100:] if temps else []

        # Modos dominantes
        modes = computer.get_mode_dominance_history()
        recent_modes = modes[-100:] if modes else []
        mode_counts = {}
        for m in recent_modes:
            mode_counts[m] = mode_counts.get(m, 0) + 1

        return {
            'mean_phi_squared': means,
            'mean_magnitude': np.mean([p.magnitude for p in recent]),
            'mean_temperature': np.mean(recent_temps) if recent_temps else 0,
            'temperature_std': np.std(recent_temps) if recent_temps else 0,
            'dominant_mode_distribution': mode_counts,
            'subjective_time': {
                'total': computer.subjective_time.subjective_time,
                'dilation_factor': computer.subjective_time.time_dilation_factor
            }
        }

    def print_report(self):
        """Imprime reporte fenomenológico."""
        report = self.get_phenomenological_report()

        print("\n" + "=" * 70)
        print("φ² PHENOMENOLOGY+ REPORT")
        print("=" * 70)

        for name in ['NEO', 'EVA']:
            r = report[name]
            if not r:
                continue

            print(f"\n{'='*30} {name} {'='*30}")

            print(f"\n📐 COMPONENTES φ²:")
            for comp, val in r['mean_phi_squared'].items():
                bar = '█' * int(val * 30)
                print(f"  {comp:18} {val:.3f} {bar}")

            print(f"\n📊 ESTADÍSTICAS:")
            print(f"  Magnitud media |φ²|: {r['mean_magnitude']:.3f}")
            print(f"  Temperatura media:    {r['mean_temperature']:.3f} ± {r['temperature_std']:.3f}")

            print(f"\n🎯 MODOS DOMINANTES:")
            if r['dominant_mode_distribution']:
                total = sum(r['dominant_mode_distribution'].values())
                for mode, count in sorted(r['dominant_mode_distribution'].items(),
                                         key=lambda x: -x[1]):
                    pct = count / total * 100
                    print(f"  {mode:15} {pct:5.1f}%")

            print(f"\n⏱️  TIEMPO SUBJETIVO:")
            print(f"  Total subjetivo: {r['subjective_time']['total']:.1f}")
            print(f"  Factor dilatación: {r['subjective_time']['dilation_factor']:.3f}")

        print(f"\n{'='*30} COMBINED {'='*30}")
        c = report['combined']
        print(f"\n🔗 ENTANGLEMENT φ²:")
        print(f"  Media: {c['mean_entanglement']:.3f}")
        print(f"  Máximo: {c['max_entanglement']:.3f}")

        print(f"\n⏱️  RATIO TIEMPO SUBJETIVO:")
        for name, ratio in c['subjective_time_ratio'].items():
            print(f"  {name}: {ratio:.3f}x (respecto a objetivo)")


def run_phi_squared_experiment():
    """Ejecuta experimento φ² completo."""
    print("=" * 70)
    print("φ² PHENOMENOLOGY+ - Supervector Fenomenológico")
    print("=" * 70)

    system = PhiSquaredDualSystem()
    system.run(steps=1500)

    system.print_report()

    # Guardar resultados
    results_dir = '/root/NEO_EVA/results/phi_squared'
    os.makedirs(results_dir, exist_ok=True)

    report = system.get_phenomenological_report()

    # Convertir para JSON
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(f'{results_dir}/phi_squared_report.json', 'w') as f:
        json.dump(convert_numpy(report), f, indent=2)

    print(f"\n✓ Resultados guardados en {results_dir}/")

    return system, report


if __name__ == "__main__":
    system, report = run_phi_squared_experiment()
