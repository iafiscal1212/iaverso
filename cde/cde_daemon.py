"""
CDE Daemon - Proceso Residente del CDE
======================================

Proceso que:
- Recibe señales
- Actualiza WORLD-X
- Emite informes
- Mantiene salud y ética

100% endógeno. Sin números mágicos.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cde.cde_observer import CDEObserver
from cde.cde_worldx import WorldX, WorldXState
from cde.cde_ethics import CDEEthics, EthicsEvaluation
from cde.cde_health import CDEHealth, HealthEvaluation, Intervention
from cde.cde_coherence import CDECoherence, CoherenceEvaluation
from cde.cde_report import CDEReportGenerator, CDEReport


@dataclass
class CDEState:
    """Estado completo del CDE."""
    t: int
    worldx: WorldXState
    ethics: EthicsEvaluation
    health: HealthEvaluation
    coherence: CoherenceEvaluation
    interventions: List[Intervention]
    report: CDEReport


class CDEDaemon:
    """
    Demonio del CDE.

    Proceso residente que:
    1. Observa el sistema
    2. Actualiza mundo interno
    3. Evalúa ética y salud
    4. Propone intervenciones
    5. Genera informes

    Todo endógeno.
    """

    def __init__(self):
        self.t = 0

        # Componentes
        self.observer = CDEObserver()
        self.worldx = WorldX()
        self.ethics = CDEEthics()
        self.health = CDEHealth()
        self.coherence = CDECoherence()
        self.reporter = CDEReportGenerator()

        # Estado actual
        self._current_state: Optional[CDEState] = None

        # Historial de estados
        self._state_history: List[CDEState] = []

    def observe_decision(
        self,
        module: str,
        decision_vector: np.ndarray,
        confidence: float
    ) -> None:
        """Registra una decisión."""
        self.observer.observe_decision(module, decision_vector, confidence)

    def observe_event(
        self,
        event_type: str,
        payload: Dict[str, Any]
    ) -> None:
        """Registra un evento."""
        self.observer.observe_event(event_type, payload)

    def observe_resources(
        self,
        cpu: float,
        ram: float,
        load: float,
        queue_length: int
    ) -> None:
        """Registra estado de recursos."""
        self.observer.observe_resources(cpu, ram, load, queue_length)

    def tick(self) -> CDEState:
        """
        Ejecuta un ciclo completo del CDE.

        Returns:
            Estado completo del CDE
        """
        self.t += 1

        # 1. Obtener observaciones
        self.observer.step()
        observation_vector = self.observer.get_observation_state()

        # 2. Actualizar mundo interno
        worldx_state = self.worldx.step(observation_vector)

        # 3. Evaluar ética
        ethics_eval = self.ethics.evaluate(worldx_state)

        # 4. Evaluar salud
        health_eval = self.health.evaluate_health(worldx_state, ethics_eval)

        # 5. Proponer intervenciones
        interventions = self.health.propose_intervention(
            worldx_state, health_eval, ethics_eval
        )

        # 6. Evaluar coherencia
        coherence_eval = self.coherence.compute_coherence(worldx_state)

        # 7. Generar informe
        report = self.reporter.generate(
            worldx_state, ethics_eval, health_eval,
            coherence_eval, interventions
        )

        # Construir estado
        state = CDEState(
            t=self.t,
            worldx=worldx_state,
            ethics=ethics_eval,
            health=health_eval,
            coherence=coherence_eval,
            interventions=interventions,
            report=report
        )

        self._current_state = state
        self._state_history.append(state)

        return state

    def get_report(self) -> Optional[CDEReport]:
        """Retorna último informe."""
        return self.reporter.get_latest_report()

    def get_report_json(self) -> str:
        """Retorna último informe en JSON."""
        report = self.get_report()
        if report:
            return self.reporter.to_json(report)
        return "{}"

    def get_state(self) -> Optional[CDEState]:
        """Retorna estado actual."""
        return self._current_state

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas del demonio."""
        return {
            't': self.t,
            'observer': self.observer.get_statistics(),
            'worldx': self.worldx.get_statistics(),
            'ethics': self.ethics.get_statistics(),
            'health': self.health.get_statistics(),
            'coherence': self.coherence.get_statistics(),
            'reporter': self.reporter.get_statistics()
        }

    def run_standalone(self, steps: int = 100) -> List[CDEReport]:
        """
        Ejecuta en modo standalone para testing.

        Genera datos sintéticos para probar el sistema.

        Args:
            steps: Número de pasos

        Returns:
            Lista de informes generados
        """
        reports = []

        for step in range(steps):
            # Generar datos sintéticos
            cpu = 0.3 + 0.2 * np.sin(step / 10) + np.random.randn() * 0.1
            ram = 0.4 + 0.1 * np.cos(step / 15) + np.random.randn() * 0.05
            load = 0.5 + 0.3 * np.sin(step / 20) + np.random.randn() * 0.1
            queue = max(0, int(5 + 10 * np.sin(step / 25) + np.random.randn() * 3))

            # Observar recursos
            self.observe_resources(
                cpu=float(np.clip(cpu, 0, 1)),
                ram=float(np.clip(ram, 0, 1)),
                load=float(np.clip(load, 0, 1)),
                queue_length=queue
            )

            # Simular decisiones ocasionales
            if step % 5 == 0:
                self.observe_decision(
                    module=f"module_{step % 3}",
                    decision_vector=np.random.randn(4),
                    confidence=0.5 + np.random.rand() * 0.5
                )

            # Simular eventos ocasionales
            if step % 10 == 0:
                self.observe_event(
                    event_type="test_event",
                    payload={"step": step}
                )

            # Tick
            state = self.tick()
            reports.append(state.report)

        return reports


# Punto de entrada para testing
if __name__ == "__main__":
    print("CDE Daemon - Test Run")
    print("=" * 50)

    daemon = CDEDaemon()
    reports = daemon.run_standalone(steps=50)

    print(f"\nGenerados {len(reports)} informes")
    print("\nÚltimo informe:")
    print(daemon.get_report_json())

    print("\nEstadísticas:")
    stats = daemon.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
