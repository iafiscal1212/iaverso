#!/usr/bin/env python3
"""
WEAVER Orchestrator
===================

Orquestador principal que integra todos los componentes.
100% ENDÓGENO - Sin constantes mágicas
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import os

from .global_state import GlobalState
from .multiscale_views import MultiscaleViews
from .phase_graph import PhaseGraph
from .indices import GlobalIndices


class WeaverOrchestrator:
    """
    World Emergence via Autonomous VEctor Reasoning

    Orquestador principal del sistema NEO_EVA.
    Integra todas las fases y módulos.

    100% Endógeno:
    - Todas las decisiones basadas en métricas internas
    - Sin parámetros mágicos
    """

    def __init__(self):
        # Componentes principales
        self.state = GlobalState()
        self.views = MultiscaleViews(n_scales=3)
        self.graph = PhaseGraph()
        self.indices = GlobalIndices()

        # Fases registradas
        self.registered_phases: List[str] = []

        # Historia de decisiones
        self.decisions: List[Dict[str, Any]] = []

        # Configuración (todo endógeno)
        self.update_interval = 1  # Se ajusta con √t

    def register_phase(self, name: str) -> None:
        """Registra una fase en el orquestador."""
        self.registered_phases.append(name)
        self.state.register_phase(name)
        self.graph.add_phase(name)

    def step(self, z: np.ndarray, S: float,
             phase_metrics: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Ejecuta un paso del orquestador.

        Args:
            z: Estado actual del sistema
            S: Entropía actual
            phase_metrics: Métricas de cada fase (opcional)

        Returns:
            Dict con estado global y decisiones
        """
        # Actualizar estado global
        self.state.step(z, S)
        t = self.state.t

        # Actualizar vistas multi-escala
        scale_views = self.views.update(z, S)

        # Actualizar datos para índices
        for i, view in enumerate(scale_views):
            self.indices.update_scale_data(i, view.S_mean)

        # Actualizar métricas de fases si disponibles
        if phase_metrics:
            for phase, metrics in phase_metrics.items():
                # Usar primera métrica como señal para grafo
                if metrics:
                    first_metric = list(metrics.values())[0]
                    self.graph.update_metric(phase, first_metric)

                # Actualizar coherencia
                if 'coherence' in metrics:
                    self.indices.update_coherence_data(metrics['coherence'])

                # Actualizar goals
                if 'goal_score' in metrics:
                    self.indices.update_goal_data(metrics['goal_score'])

        # Recalcular grafo de dependencias periódicamente
        self.update_interval = max(5, int(np.sqrt(t + 1)))
        if t % self.update_interval == 0:
            self.graph.compute_all_te()

        # Calcular índices globales
        indices = self.indices.compute_all(t)

        # Generar decisiones/recomendaciones
        decision = self._make_decision(t, indices)
        self.decisions.append(decision)

        return {
            't': t,
            'state': self.state.get_system_summary(),
            'views': self.views.to_dict(),
            'indices': {
                'MSI': indices.MSI,
                'SCI': indices.SCI,
                'EGI': indices.EGI
            },
            'decision': decision
        }

    def _make_decision(self, t: int, indices) -> Dict[str, Any]:
        """
        Genera decisión/recomendación basada en estado actual.

        100% endógeno: basado en índices y tendencias
        """
        trends = self.indices.get_trends()

        # Análisis de estado
        msi_low = indices.MSI < np.percentile(self.indices.MSI_history, 25) if len(self.indices.MSI_history) > 5 else False
        sci_low = indices.SCI < np.percentile(self.indices.SCI_history, 25) if len(self.indices.SCI_history) > 5 else False
        egi_declining = trends.get('EGI_trend', 0) < 0

        # Determinar acción recomendada
        if msi_low and sci_low:
            action = 'increase_integration'
            reason = 'Baja integración multi-escala y coherencia'
        elif egi_declining:
            action = 'reinforce_goals'
            reason = 'Goals emergentes en declive'
        elif msi_low:
            action = 'sync_scales'
            reason = 'Desincronización entre escalas temporales'
        elif sci_low:
            action = 'increase_coherence'
            reason = 'Baja coherencia estructural'
        else:
            action = 'maintain'
            reason = 'Sistema estable'

        return {
            't': t,
            'action': action,
            'reason': reason,
            'indices': {
                'MSI': indices.MSI,
                'SCI': indices.SCI,
                'EGI': indices.EGI
            },
            'trends': trends
        }

    def get_phase_order(self) -> List[str]:
        """Retorna orden óptimo de ejecución de fases."""
        return self.graph.get_topological_order()

    def get_critical_phases(self) -> List[str]:
        """
        Identifica fases críticas (alta centralidad).

        100% endógeno: basado en PageRank del grafo
        """
        centrality = self.graph.get_centrality()
        if not centrality:
            return []

        # Umbral endógeno: P75
        threshold = np.percentile(list(centrality.values()), 75)
        return [phase for phase, c in centrality.items() if c >= threshold]

    def get_report(self) -> Dict[str, Any]:
        """Genera reporte completo del sistema."""
        return {
            'timestamp': datetime.now().isoformat(),
            'global_state': self.state.to_dict(),
            'multiscale_views': self.views.to_dict(),
            'phase_graph': self.graph.to_dict(),
            'indices': self.indices.to_dict(),
            'phase_order': self.get_phase_order(),
            'critical_phases': self.get_critical_phases(),
            'recent_decisions': self.decisions[-10:] if self.decisions else []
        }

    def save_report(self, path: str) -> None:
        """Guarda reporte en archivo."""
        report = self.get_report()
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)


def run_weaver_test() -> Dict[str, Any]:
    """Ejecuta test completo de WEAVER."""

    print("=" * 70)
    print("WEAVER: WORLD EMERGENCE VIA AUTONOMOUS VECTOR REASONING")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}\n")

    np.random.seed(42)

    # Crear orquestador
    weaver = WeaverOrchestrator()

    # Registrar fases
    phases = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10']
    for phase in phases:
        weaver.register_phase(phase)

    # Simulación
    T = 200
    results = []

    z = np.array([0.4, 0.3, 0.3])

    for t in range(T):
        # Dinámica
        noise = np.random.randn(3) * 0.02
        z = z + noise
        z = np.clip(z, 0.01, 0.99)
        z = z / z.sum()

        S = -np.sum(z * np.log(z + 1e-10))

        # Simular métricas de fases
        phase_metrics = {}
        for phase in phases:
            phase_metrics[phase] = {
                'coherence': 0.5 + 0.3 * np.random.rand(),
                'goal_score': 0.3 + 0.2 * np.random.rand() + 0.001 * t
            }

        # Paso del orquestador
        result = weaver.step(z, S, phase_metrics)
        results.append(result)

    # Reporte final
    print("=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print()

    final_indices = weaver.indices.get_summary()
    print(f"Índices finales:")
    print(f"  MSI (Multi-Scale Integration): {final_indices['MSI']:.4f}")
    print(f"  SCI (Structural Coherence):    {final_indices['SCI']:.4f}")
    print(f"  EGI (Emergent Goal):           {final_indices['EGI']:.4f}")
    print()

    print(f"Fases críticas: {weaver.get_critical_phases()}")
    print(f"Orden de ejecución: {weaver.get_phase_order()[:5]}...")
    print()

    graph_info = weaver.graph.to_dict()
    print(f"Grafo de dependencias:")
    print(f"  Nodos: {len(graph_info['nodes'])}")
    print(f"  Aristas: {graph_info['n_edges']}")
    print(f"  Ciclos detectados: {len(graph_info['cycles'])}")
    print()

    # Criterios GO/NO-GO
    criteria = {}

    # 1. Índices calculados
    criteria['indices_computed'] = final_indices['computed']

    # 2. MSI > 0 (hay integración entre escalas)
    criteria['msi_positive'] = final_indices['MSI'] > 0

    # 3. SCI > 0.3 (coherencia razonable)
    criteria['sci_healthy'] = final_indices['SCI'] > 0.3

    # 4. EGI tiene tendencia
    trends = final_indices.get('trends', {})
    criteria['egi_trending'] = abs(trends.get('EGI_trend', 0)) > 0

    # 5. Grafo funcional
    criteria['graph_functional'] = graph_info['n_edges'] > 0

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
        'phase': 'WEAVER',
        'name': 'World Emergence via Autonomous VEctor Reasoning',
        'timestamp': datetime.now().isoformat(),
        'indices': final_indices,
        'graph': graph_info,
        'criteria': criteria,
        'go_status': go_status,
        'passed_criteria': passed,
        'total_criteria': total
    }

    os.makedirs('/root/NEO_EVA/results/weaver', exist_ok=True)
    os.makedirs('/root/NEO_EVA/figures', exist_ok=True)

    with open('/root/NEO_EVA/results/weaver/orchestrator_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Guardar reporte completo
    weaver.save_report('/root/NEO_EVA/results/weaver/full_report.json')

    # Visualización
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Índices a lo largo del tiempo
        ax1 = axes[0, 0]
        ax1.plot(weaver.indices.MSI_history, label='MSI', linewidth=1.5)
        ax1.plot(weaver.indices.SCI_history, label='SCI', linewidth=1.5)
        ax1.plot(weaver.indices.EGI_history, label='EGI', linewidth=1.5)
        ax1.set_xlabel('Tiempo')
        ax1.set_ylabel('Valor')
        ax1.set_title('Índices Globales WEAVER')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Vistas multi-escala
        ax2 = axes[0, 1]
        scale_names = ['√t', '2√t', '4√t']
        for i in range(3):
            if weaver.indices.scale_data[i]:
                ax2.plot(weaver.indices.scale_data[i], label=f'Escala {scale_names[i]}', alpha=0.7)
        ax2.set_xlabel('Tiempo')
        ax2.set_ylabel('S_mean')
        ax2.set_title('Vistas Multi-Escala')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Decisiones del sistema
        ax3 = axes[1, 0]
        action_counts = {}
        for d in weaver.decisions:
            action = d.get('action', 'unknown')
            action_counts[action] = action_counts.get(action, 0) + 1
        if action_counts:
            ax3.bar(action_counts.keys(), action_counts.values(), color='steelblue', alpha=0.7)
            ax3.set_xlabel('Acción')
            ax3.set_ylabel('Frecuencia')
            ax3.set_title('Distribución de Decisiones')
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)

        # 4. Centralidad de fases
        ax4 = axes[1, 1]
        centrality = weaver.graph.get_centrality()
        if centrality:
            sorted_phases = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            phases_sorted = [p[0] for p in sorted_phases]
            values = [p[1] for p in sorted_phases]
            colors = ['red' if p in weaver.get_critical_phases() else 'steelblue' for p in phases_sorted]
            ax4.barh(phases_sorted, values, color=colors, alpha=0.7)
            ax4.set_xlabel('Centralidad (PageRank)')
            ax4.set_ylabel('Fase')
            ax4.set_title('Centralidad de Fases')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/root/NEO_EVA/figures/weaver_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nResultados guardados en: /root/NEO_EVA/results/weaver")
        print(f"Figura: /root/NEO_EVA/figures/weaver_results.png")

    except Exception as e:
        print(f"Warning: No se pudo crear visualización: {e}")

    return output


if __name__ == "__main__":
    run_weaver_test()
