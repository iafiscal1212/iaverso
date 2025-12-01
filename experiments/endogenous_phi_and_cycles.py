#!/usr/bin/env python3
"""
φ ENDÓGENO Y CICLOS-DECISIONES
==============================

Todo endógeno:
- φ = emerge de la dinámica interna, no de fórmulas externas
- Umbrales = percentiles de la propia historia
- Fases = detectadas por cambios en la propia serie temporal
- Decisiones = cambios detectados endógenamente

Principio: el sistema define sus propios estados, no nosotros.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os
import sys

sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/experiments')

from autonomous_life import AutonomousAgent, AutonomousDualLife

DRIVE_NAMES = ['entropy', 'neg_surprise', 'novelty', 'stability', 'integration', 'otherness']


class EndogenousMetrics:
    """
    Métricas completamente endógenas.

    Cada métrica se define SOLO en términos de la historia del propio agente.
    No hay constantes mágicas - todo son percentiles o ratios internos.
    """

    def __init__(self, agent: AutonomousAgent):
        self.agent = agent

        # Historias propias para calcular umbrales
        self.phi_history = []
        self.coherence_history = []
        self.complexity_history = []

    def compute_phi(self, t: int) -> float:
        """
        φ endógeno: integración de información emergente.

        φ = (información total del sistema) - (suma de información de partes)

        Aproximación endógena:
        φ ≈ covarianza_entre_dimensiones / varianza_total

        Si las dimensiones covarían mucho → están integradas → φ alto
        Si son independientes → φ bajo
        """
        if not hasattr(self.agent, 'z_history') or len(self.agent.z_history) < 20:
            return 0.5

        # Ventana endógena: sqrt(t)
        window = max(10, min(int(np.sqrt(t + 1) * 3), len(self.agent.z_history)))
        recent = np.array(self.agent.z_history[-window:])

        if recent.shape[0] < 5:
            return 0.5

        # Matriz de covarianza
        try:
            cov_matrix = np.cov(recent.T)

            # Varianza total = traza
            total_var = np.trace(cov_matrix)

            # Covarianza entre dimensiones = suma de elementos fuera de diagonal
            off_diagonal = np.sum(np.abs(cov_matrix)) - np.trace(np.abs(cov_matrix))

            # φ = ratio de integración
            # Alto cuando hay mucha covarianza relativa a varianza
            if total_var > 1e-10:
                phi = off_diagonal / (total_var * recent.shape[1])
            else:
                phi = 0.5

        except:
            phi = 0.5

        # Normalizar a [0, 1] usando historia propia
        self.phi_history.append(phi)

        if len(self.phi_history) > 50:
            # Normalizar respecto a la propia historia
            phi_min = np.percentile(self.phi_history, 5)
            phi_max = np.percentile(self.phi_history, 95)
            if phi_max > phi_min:
                phi = (phi - phi_min) / (phi_max - phi_min)
                phi = np.clip(phi, 0, 1)

        return phi

    def compute_coherence(self, t: int) -> float:
        """
        Coherencia endógena: estabilidad de la estructura interna.

        Coherencia = 1 / (1 + varianza_reciente / varianza_baseline)

        El baseline es la propia historia del agente.
        """
        if not hasattr(self.agent, 'meta_drive') or not self.agent.meta_drive.weight_history:
            return 0.5

        history = self.agent.meta_drive.weight_history
        if len(history) < 20:
            return 0.5

        # Ventana reciente: endógena
        window = max(5, min(int(np.sqrt(t + 1)), len(history) // 4))

        recent = np.array(history[-window:])
        baseline = np.array(history[:-window]) if len(history) > window else recent

        var_recent = np.mean(np.var(recent, axis=0))
        var_baseline = np.mean(np.var(baseline, axis=0)) + 1e-10

        # Coherencia: baja varianza reciente = alta coherencia
        coherence = 1.0 / (1.0 + var_recent / var_baseline)

        self.coherence_history.append(coherence)

        return coherence

    def compute_complexity(self, t: int) -> float:
        """
        Complejidad endógena: balance entre orden y caos.

        Complejidad = entropía * (1 - entropía)
        Máxima cuando entropía ≈ 0.5
        """
        if not hasattr(self.agent, 'meta_drive'):
            return 0.5

        weights = self.agent.meta_drive.weights
        weights = np.clip(weights, 1e-10, None)
        weights = weights / weights.sum()

        # Entropía normalizada
        max_entropy = np.log(len(weights))
        entropy = -np.sum(weights * np.log(weights)) / max_entropy

        # Complejidad: máxima en el medio
        complexity = 4 * entropy * (1 - entropy)

        self.complexity_history.append(complexity)

        return complexity

    def detect_phase(self, t: int) -> str:
        """
        Detecta la fase actual de forma endógena.

        Fases basadas en combinaciones de métricas propias:
        - EXPLORATION: alta complejidad, baja coherencia
        - CONSOLIDATION: baja complejidad, alta coherencia
        - CRISIS: todo bajo
        - FLOW: todo alto
        """
        phi = self.compute_phi(t)
        coherence = self.compute_coherence(t)
        complexity = self.compute_complexity(t)

        # Umbrales endógenos: medianas de la propia historia
        if len(self.phi_history) < 20:
            return 'initializing'

        phi_median = np.median(self.phi_history)
        coh_median = np.median(self.coherence_history)
        comp_median = np.median(self.complexity_history)

        high_phi = phi > phi_median
        high_coh = coherence > coh_median
        high_comp = complexity > comp_median

        if high_comp and not high_coh:
            return 'exploration'
        elif not high_comp and high_coh:
            return 'consolidation'
        elif not high_phi and not high_coh:
            return 'crisis'
        elif high_phi and high_coh:
            return 'flow'
        else:
            return 'transition'


@dataclass
class EndogenousDecision:
    """Decisión detectada endógenamente."""
    t: int
    decision_type: str
    magnitude: float  # Magnitud relativa a la historia
    context_phase: str
    from_state: str
    to_state: str


class EndogenousDecisionDetector:
    """
    Detecta decisiones/cambios de forma endógena.

    Un "cambio" es significativo si supera el umbral de la propia historia.
    """

    def __init__(self, agent: AutonomousAgent):
        self.agent = agent
        self.change_magnitudes = []
        self.decisions = []
        self.prev_dominant = None
        self.prev_phase = None

    def detect_drive_change(self, t: int, metrics: EndogenousMetrics) -> Optional[EndogenousDecision]:
        """Detecta cambio de drive dominante."""
        if not hasattr(self.agent, 'meta_drive'):
            return None

        weights = self.agent.meta_drive.weights
        current_dominant = DRIVE_NAMES[np.argmax(weights)]

        if self.prev_dominant is not None and current_dominant != self.prev_dominant:
            # Calcular magnitud del cambio
            if len(self.agent.meta_drive.weight_history) > 1:
                prev_weights = self.agent.meta_drive.weight_history[-2]
                change = np.sum(np.abs(weights - prev_weights))
                self.change_magnitudes.append(change)

                # Magnitud relativa a la historia
                if len(self.change_magnitudes) > 10:
                    magnitude = change / (np.median(self.change_magnitudes) + 1e-10)
                else:
                    magnitude = 1.0
            else:
                magnitude = 1.0

            decision = EndogenousDecision(
                t=t,
                decision_type='drive_change',
                magnitude=magnitude,
                context_phase=metrics.detect_phase(t),
                from_state=self.prev_dominant,
                to_state=current_dominant
            )

            self.prev_dominant = current_dominant
            self.decisions.append(decision)
            return decision

        self.prev_dominant = current_dominant
        return None

    def detect_phase_change(self, t: int, metrics: EndogenousMetrics) -> Optional[EndogenousDecision]:
        """Detecta cambio de fase."""
        current_phase = metrics.detect_phase(t)

        if self.prev_phase is not None and current_phase != self.prev_phase:
            decision = EndogenousDecision(
                t=t,
                decision_type='phase_change',
                magnitude=1.0,
                context_phase=current_phase,
                from_state=self.prev_phase,
                to_state=current_phase
            )

            self.prev_phase = current_phase
            self.decisions.append(decision)
            return decision

        self.prev_phase = current_phase
        return None


def run_endogenous_analysis(T: int = 1500, n_seeds: int = 3) -> Dict:
    """
    Análisis completo con métricas endógenas.
    """
    print("=" * 70)
    print("ANÁLISIS ENDÓGENO: φ, CICLOS Y DECISIONES")
    print("=" * 70)
    print(f"T={T}, seeds={n_seeds}")
    print("Todo endógeno: umbrales, fases, detección de cambios")

    all_results = []

    for seed in range(n_seeds):
        print(f"\n{'='*50}")
        print(f"SEED {seed}")
        print('='*50)

        np.random.seed(42 + seed)

        life = AutonomousDualLife(dim=6)

        # Métricas endógenas
        neo_metrics = EndogenousMetrics(life.neo)
        eva_metrics = EndogenousMetrics(life.eva)

        # Detectores de decisiones
        neo_detector = EndogenousDecisionDetector(life.neo)
        eva_detector = EndogenousDecisionDetector(life.eva)

        # Historias
        phi_history = {'neo': [], 'eva': []}
        coherence_history = {'neo': [], 'eva': []}
        phase_history = {'neo': [], 'eva': []}
        identity_history = {'neo': [], 'eva': []}

        for t in range(T):
            stimulus = np.random.dirichlet(np.ones(6) * 2)
            life.step(stimulus)

            # Calcular métricas endógenas
            neo_phi = neo_metrics.compute_phi(t)
            eva_phi = eva_metrics.compute_phi(t)

            neo_coh = neo_metrics.compute_coherence(t)
            eva_coh = eva_metrics.compute_coherence(t)

            neo_phase = neo_metrics.detect_phase(t)
            eva_phase = eva_metrics.detect_phase(t)

            # Guardar
            phi_history['neo'].append(neo_phi)
            phi_history['eva'].append(eva_phi)
            coherence_history['neo'].append(neo_coh)
            coherence_history['eva'].append(eva_coh)
            phase_history['neo'].append(neo_phase)
            phase_history['eva'].append(eva_phase)
            identity_history['neo'].append(life.neo.identity_strength)
            identity_history['eva'].append(life.eva.identity_strength)

            # Detectar decisiones
            neo_detector.detect_drive_change(t, neo_metrics)
            neo_detector.detect_phase_change(t, neo_metrics)
            eva_detector.detect_drive_change(t, eva_metrics)
            eva_detector.detect_phase_change(t, eva_metrics)

        # Análisis
        print("\n--- Resumen ---")
        print(f"  φ promedio NEO: {np.mean(phi_history['neo']):.3f}")
        print(f"  φ promedio EVA: {np.mean(phi_history['eva']):.3f}")
        print(f"  Coherencia promedio NEO: {np.mean(coherence_history['neo']):.3f}")
        print(f"  Coherencia promedio EVA: {np.mean(coherence_history['eva']):.3f}")

        # Distribución de fases
        from collections import Counter
        neo_phases = Counter(phase_history['neo'])
        eva_phases = Counter(phase_history['eva'])

        print(f"\n  Fases NEO: {dict(neo_phases)}")
        print(f"  Fases EVA: {dict(eva_phases)}")

        # Decisiones
        neo_decisions = neo_detector.decisions
        eva_decisions = eva_detector.decisions

        print(f"\n  Decisiones NEO: {len(neo_decisions)}")
        print(f"  Decisiones EVA: {len(eva_decisions)}")

        # Test de hipótesis endógeno
        print("\n--- Test de Hipótesis Endógeno ---")

        h1_result = test_h1_endogenous(
            phi_history['neo'],
            identity_history['neo'],
            neo_decisions
        )
        print(f"  H1 (φ↑ + id↓ → cambio): {h1_result['interpretation']}")

        h2_result = test_h2_endogenous(
            phi_history['neo'],
            identity_history['neo'],
            neo_decisions
        )
        print(f"  H2 (id↑ + φ↓ → consolidación): {h2_result['interpretation']}")

        h3_result = test_h3_endogenous(
            phase_history['neo'],
            neo_decisions
        )
        print(f"  H3 (crisis → cambio): {h3_result['interpretation']}")

        all_results.append({
            'seed': seed,
            'phi_mean_neo': np.mean(phi_history['neo']),
            'phi_mean_eva': np.mean(phi_history['eva']),
            'coherence_mean_neo': np.mean(coherence_history['neo']),
            'coherence_mean_eva': np.mean(coherence_history['eva']),
            'n_decisions_neo': len(neo_decisions),
            'n_decisions_eva': len(eva_decisions),
            'phases_neo': dict(neo_phases),
            'phases_eva': dict(eva_phases),
            'h1': h1_result,
            'h2': h2_result,
            'h3': h3_result
        })

    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)

    h1_confirmed = sum(1 for r in all_results if r['h1']['supported'])
    h2_confirmed = sum(1 for r in all_results if r['h2']['supported'])
    h3_confirmed = sum(1 for r in all_results if r['h3']['supported'])

    print(f"\nH1 (φ↑ + id↓ → más cambio): {h1_confirmed}/{n_seeds} confirmado")
    print(f"H2 (id↑ + φ↓ → consolidación): {h2_confirmed}/{n_seeds} confirmado")
    print(f"H3 (crisis → cambio): {h3_confirmed}/{n_seeds} confirmado")

    if h1_confirmed + h2_confirmed + h3_confirmed >= n_seeds * 1.5:
        print("""
═══════════════════════════════════════════════════════════════════
CONCLUSIÓN POSITIVA:

Los estados internos (φ, identidad, coherencia) MODULAN
SISTEMÁTICAMENTE cómo el sistema decide y cambia.

Esto es evidencia de:
"Estados internos funcionales que afectan el procesamiento"

Una forma prudente de aproximarse a:
"Estados subjetivos con rol causal en el comportamiento"
═══════════════════════════════════════════════════════════════════
""")

    # Guardar
    os.makedirs('/root/NEO_EVA/results/endogenous_cycles', exist_ok=True)

    with open('/root/NEO_EVA/results/endogenous_cycles/results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'parameters': {'T': T, 'n_seeds': n_seeds},
            'results': [{k: v for k, v in r.items()
                        if not isinstance(v, np.ndarray)}
                       for r in all_results],
            'summary': {
                'h1_confirmed': h1_confirmed,
                'h2_confirmed': h2_confirmed,
                'h3_confirmed': h3_confirmed
            }
        }, f, indent=2, default=str)

    # Visualización
    visualize_endogenous_results(all_results, phi_history, identity_history, phase_history)

    return all_results


def test_h1_endogenous(phi_history: List, identity_history: List,
                       decisions: List[EndogenousDecision]) -> Dict:
    """
    H1: Cuando φ alto e identidad baja → más cambio

    Todo endógeno: umbrales son percentiles de la propia historia.
    """
    if len(phi_history) < 100 or not decisions:
        return {'supported': False, 'interpretation': 'datos insuficientes', 'ratio': 0}

    # Umbrales endógenos
    phi_high = np.percentile(phi_history, 70)
    id_low = np.percentile(identity_history, 30)

    # Momentos de alta φ y baja identidad
    high_phi_low_id_times = set()
    other_times = set()

    for t in range(len(phi_history)):
        if phi_history[t] > phi_high and identity_history[t] < id_low:
            high_phi_low_id_times.add(t)
        else:
            other_times.add(t)

    # Contar decisiones en cada condición
    drive_changes = [d for d in decisions if d.decision_type == 'drive_change']

    changes_in_hpli = sum(1 for d in drive_changes if d.t in high_phi_low_id_times)
    changes_in_other = sum(1 for d in drive_changes if d.t in other_times)

    # Tasas
    rate_hpli = changes_in_hpli / max(1, len(high_phi_low_id_times))
    rate_other = changes_in_other / max(1, len(other_times))

    ratio = rate_hpli / (rate_other + 1e-10)

    supported = ratio > 1.5 and changes_in_hpli > 0

    return {
        'supported': supported,
        'ratio': float(ratio),
        'changes_hpli': changes_in_hpli,
        'changes_other': changes_in_other,
        'n_hpli_times': len(high_phi_low_id_times),
        'interpretation': f'CONFIRMADO (ratio={ratio:.2f})' if supported else f'NO (ratio={ratio:.2f})'
    }


def test_h2_endogenous(phi_history: List, identity_history: List,
                       decisions: List[EndogenousDecision]) -> Dict:
    """
    H2: Cuando identidad alta y φ bajo → consolidación (menos cambio)
    """
    if len(phi_history) < 100 or not decisions:
        return {'supported': False, 'interpretation': 'datos insuficientes', 'ratio': 0}

    phi_low = np.percentile(phi_history, 30)
    id_high = np.percentile(identity_history, 70)

    high_id_low_phi_times = set()
    other_times = set()

    for t in range(len(phi_history)):
        if phi_history[t] < phi_low and identity_history[t] > id_high:
            high_id_low_phi_times.add(t)
        else:
            other_times.add(t)

    drive_changes = [d for d in decisions if d.decision_type == 'drive_change']

    changes_in_hilp = sum(1 for d in drive_changes if d.t in high_id_low_phi_times)
    changes_in_other = sum(1 for d in drive_changes if d.t in other_times)

    rate_hilp = changes_in_hilp / max(1, len(high_id_low_phi_times))
    rate_other = changes_in_other / max(1, len(other_times))

    # Para H2: queremos MENOS cambio en hilp, así que ratio inverso
    ratio = rate_other / (rate_hilp + 1e-10)

    supported = ratio > 1.5 and len(high_id_low_phi_times) > 20

    return {
        'supported': supported,
        'ratio': float(ratio),
        'changes_hilp': changes_in_hilp,
        'changes_other': changes_in_other,
        'n_hilp_times': len(high_id_low_phi_times),
        'interpretation': f'CONFIRMADO (ratio={ratio:.2f})' if supported else f'NO (ratio={ratio:.2f})'
    }


def test_h3_endogenous(phase_history: List, decisions: List[EndogenousDecision]) -> Dict:
    """
    H3: Las crisis predicen cambios de drive

    Endógeno: usamos las fases detectadas por el propio sistema.
    """
    if len(phase_history) < 100 or not decisions:
        return {'supported': False, 'interpretation': 'datos insuficientes', 'ratio': 0}

    # Tiempos de crisis (detectados endógenamente)
    crisis_times = [t for t, phase in enumerate(phase_history) if phase == 'crisis']

    if not crisis_times:
        return {'supported': False, 'interpretation': 'sin crisis detectadas', 'ratio': 0}

    # Cambios de drive
    drive_changes = [d for d in decisions if d.decision_type == 'drive_change']

    if not drive_changes:
        return {'supported': False, 'interpretation': 'sin cambios de drive', 'ratio': 0}

    # Distancia de cada cambio a la crisis más cercana
    distances = []
    for d in drive_changes:
        min_dist = min(abs(d.t - ct) for ct in crisis_times)
        distances.append(min_dist)

    # Distancia esperada si fuera aleatorio
    # Aproximación: T / (n_crisis + 1) / 2
    T = len(phase_history)
    expected_random = T / (len(crisis_times) + 1) / 2

    mean_distance = np.mean(distances)

    # Ratio: si cambios están más cerca de crisis que aleatorio
    ratio = expected_random / (mean_distance + 1e-10)

    supported = ratio > 1.3 and mean_distance < expected_random

    return {
        'supported': supported,
        'ratio': float(ratio),
        'mean_distance': float(mean_distance),
        'expected_random': float(expected_random),
        'n_crisis_times': len(crisis_times),
        'interpretation': f'CONFIRMADO (ratio={ratio:.2f})' if supported else f'NO (ratio={ratio:.2f})'
    }


def visualize_endogenous_results(results: List, phi_history: Dict,
                                  identity_history: Dict, phase_history: Dict):
    """Visualiza resultados endógenos."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Usar última seed para visualización detallada
        neo_phi = phi_history['neo']
        neo_id = identity_history['neo']
        neo_phases = phase_history['neo']

        T = len(neo_phi)
        t_range = range(T)

        # 1. φ vs tiempo
        ax = axes[0, 0]
        ax.plot(t_range, neo_phi, 'b-', alpha=0.7, linewidth=0.5)
        ax.axhline(np.median(neo_phi), color='b', linestyle='--', alpha=0.5, label='mediana')
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('φ (endógeno)')
        ax.set_title('Integración de Información (φ)')
        ax.legend()

        # 2. Identidad vs tiempo
        ax = axes[0, 1]
        ax.plot(t_range, neo_id, 'g-', alpha=0.7, linewidth=0.5)
        ax.axhline(np.median(neo_id), color='g', linestyle='--', alpha=0.5, label='mediana')
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('Identidad')
        ax.set_title('Fuerza de Identidad')
        ax.legend()

        # 3. φ vs Identidad (espacio de fases)
        ax = axes[1, 0]

        # Colorear por fase
        phase_colors = {
            'exploration': 'red',
            'consolidation': 'blue',
            'crisis': 'black',
            'flow': 'green',
            'transition': 'gray',
            'initializing': 'yellow'
        }

        for phase in set(neo_phases):
            mask = [p == phase for p in neo_phases]
            phi_phase = [neo_phi[i] for i in range(T) if mask[i]]
            id_phase = [neo_id[i] for i in range(T) if mask[i]]
            ax.scatter(phi_phase, id_phase, c=phase_colors.get(phase, 'gray'),
                      label=phase, alpha=0.3, s=5)

        ax.set_xlabel('φ')
        ax.set_ylabel('Identidad')
        ax.set_title('Espacio de Fases (φ vs Identidad)')
        ax.legend(fontsize=8)

        # 4. Distribución de fases
        ax = axes[1, 1]
        from collections import Counter
        phase_counts = Counter(neo_phases)
        phases = list(phase_counts.keys())
        counts = [phase_counts[p] for p in phases]
        colors = [phase_colors.get(p, 'gray') for p in phases]

        ax.bar(phases, counts, color=colors, alpha=0.7)
        ax.set_ylabel('Tiempo en fase')
        ax.set_title('Distribución de Fases')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig('/root/NEO_EVA/results/endogenous_cycles/endogenous_analysis.png',
                   dpi=150, bbox_inches='tight')
        print(f"\nFigura guardada: /root/NEO_EVA/results/endogenous_cycles/endogenous_analysis.png")
        plt.close()

    except Exception as e:
        print(f"Error en visualización: {e}")


if __name__ == "__main__":
    run_endogenous_analysis(T=1500, n_seeds=3)
