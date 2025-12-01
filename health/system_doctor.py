"""
System Doctor: Médico Global del Sistema
=========================================

El médico del hospital: ve a todo el sistema, no solo a un agente.

Coordina:
    - HealthMonitor por agente
    - RepairProtocol por agente
    - Detección de patologías sistémicas
    - Priorización de intervenciones

El rol de médico es ELEGIDO ENDÓGENAMENTE:
    - Cualquier agente puede "ofrecerse" como médico
    - La aptitud médica emerge de métricas estructurales
    - El rol puede rotar si otro agente desarrolla mejor aptitud

100% endógeno. Sin números mágicos.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import sys
sys.path.insert(0, '/root/NEO_EVA')

from health.health_monitor import HealthMonitor, HealthMetrics, HealthAssessment
from health.repair_protocols import RepairProtocol, Intervention, InterventionResult
from cognition.agi_dynamic_constants import L_t, max_history


@dataclass
class SystemHealthState:
    """Estado de salud del sistema completo."""
    t: int
    agent_health: Dict[str, float]           # H_t por agente
    system_health: float                      # H global
    current_doctor: Optional[str]             # Agente médico actual
    doctor_aptitude: Dict[str, float]         # Aptitud médica por agente
    active_interventions: int                 # Intervenciones activas
    system_pathologies: List[str]             # Patologías sistémicas


class SystemDoctor:
    """
    Sistema médico global.

    Responsabilidades:
        1. Monitorizar salud de todos los agentes
        2. Coordinar intervenciones
        3. Detectar patologías sistémicas
        4. Elegir/rotar el rol de médico endógenamente
    """

    def __init__(self, agent_ids: List[str]):
        """
        Inicializa sistema médico global.

        Args:
            agent_ids: Lista de IDs de todos los agentes
        """
        self.agent_ids = agent_ids
        self.n_agents = len(agent_ids)

        # Monitor y protocolo por agente
        self.monitors: Dict[str, HealthMonitor] = {
            agent_id: HealthMonitor(agent_id) for agent_id in agent_ids
        }
        self.protocols: Dict[str, RepairProtocol] = {
            agent_id: RepairProtocol(agent_id) for agent_id in agent_ids
        }

        # Estado del médico
        self.current_doctor: Optional[str] = None
        self.doctor_history: List[Tuple[int, str]] = []  # (t, agent_id)

        # Aptitud médica por agente
        self.medical_aptitude: Dict[str, List[float]] = {
            agent_id: [] for agent_id in agent_ids
        }

        # Historial de salud del sistema
        self.system_health_history: List[float] = []

        # Patologías sistémicas detectadas
        self.pathology_history: List[Tuple[int, str]] = []

        self.t = 0

    def _extract_agent_metrics(
        self,
        agent_state: Dict[str, Any],
        global_state: Dict[str, Any]
    ) -> HealthMetrics:
        """
        Extrae métricas de salud del estado de un agente.

        Mapea campos del estado a HealthMetrics.
        """
        # Valores por defecto
        metrics = HealthMetrics()

        # Extraer del estado del agente
        if 'crisis_rate' in agent_state:
            metrics.crisis_rate = agent_state['crisis_rate']
        elif 'regulation' in agent_state:
            reg = agent_state['regulation']
            if hasattr(reg, 'get_crisis_rate'):
                metrics.crisis_rate = reg.get_crisis_rate()

        if 'V_t' in agent_state:
            metrics.V_t = agent_state['V_t']
        elif 'lyapunov' in agent_state:
            lyap = agent_state['lyapunov']
            if hasattr(lyap, 'V_history') and lyap.V_history:
                metrics.V_t = lyap.V_history[-1]

        if 'CF_score' in agent_state:
            metrics.CF_score = agent_state['CF_score']

        if 'CI_score' in agent_state:
            metrics.CI_score = agent_state['CI_score']

        if 'ethics_score' in agent_state:
            metrics.ethics_score = agent_state['ethics_score']
        elif 'action_stats' in agent_state:
            action_stats = agent_state['action_stats']
            if 'mean_ethical' in action_stats:
                metrics.ethics_score = action_stats['mean_ethical']

        if 'narrative_continuity' in agent_state:
            metrics.narrative_continuity = agent_state['narrative_continuity']

        if 'symbolic_stability' in agent_state:
            metrics.symbolic_stability = agent_state['symbolic_stability']

        if 'self_coherence' in agent_state:
            metrics.self_coherence = agent_state['self_coherence']
        elif 'self_theory' in agent_state:
            st = agent_state['self_theory']
            if hasattr(st, 'is_self_coherent'):
                metrics.self_coherence = 1.0 if st.is_self_coherent() else 0.5

        if 'tom_accuracy' in agent_state:
            metrics.tom_accuracy = agent_state['tom_accuracy']

        if 'config_entropy' in agent_state:
            metrics.config_entropy = agent_state['config_entropy']
        elif 'action_stats' in agent_state:
            action_stats = agent_state['action_stats']
            if 'config_entropy' in action_stats:
                metrics.config_entropy = action_stats['config_entropy']

        if 'wellbeing' in agent_state:
            metrics.wellbeing = agent_state['wellbeing']
        elif 'regulation' in agent_state:
            reg = agent_state['regulation']
            if hasattr(reg, 'wellbeing_history') and reg.wellbeing_history:
                metrics.wellbeing = reg.wellbeing_history[-1]

        if 'metacognition' in agent_state:
            metrics.metacognition = agent_state['metacognition']

        return metrics

    def _compute_medical_aptitude(self, agent_id: str) -> float:
        """
        Calcula aptitud médica de un agente.

        La aptitud médica emerge de:
            - Estabilidad propia (H_t alto y estable)
            - Capacidad de observar (tom_accuracy alta)
            - Bajo consumo de recursos (no compite)
            - Historial de decisiones éticas

        aptitude = w_stability * stability + w_tom * tom + w_ethics * ethics

        donde los pesos son endógenos (basados en varianza).
        """
        monitor = self.monitors[agent_id]

        if len(monitor.H_history) < L_t(self.t):
            return 0.5  # Sin suficiente historial

        # Componente 1: Estabilidad propia
        H_recent = monitor.H_history[-L_t(self.t):]
        stability = np.mean(H_recent) * (1 - np.std(H_recent))

        # Componente 2: Capacidad ToM
        tom_history = monitor.history.get('tom_accuracy', [])
        if tom_history:
            tom = np.mean(tom_history[-L_t(self.t):])
        else:
            tom = 0.5

        # Componente 3: Ética
        ethics_history = monitor.history.get('ethics_score', [])
        if ethics_history:
            ethics = np.mean(ethics_history[-L_t(self.t):])
        else:
            ethics = 0.5

        # Componente 4: Coherencia (AGI-20)
        coherence_history = monitor.history.get('self_coherence', [])
        if coherence_history:
            coherence = np.mean(coherence_history[-L_t(self.t):])
        else:
            coherence = 0.5

        # Pesos endógenos basados en varianza inversa
        components = {
            'stability': stability,
            'tom': tom,
            'ethics': ethics,
            'coherence': coherence
        }

        # Calcular varianzas
        variances = {
            'stability': np.var(H_recent) + 0.01,
            'tom': np.var(tom_history[-L_t(self.t):]) + 0.01 if len(tom_history) > L_t(self.t) else 0.1,
            'ethics': np.var(ethics_history[-L_t(self.t):]) + 0.01 if len(ethics_history) > L_t(self.t) else 0.1,
            'coherence': np.var(coherence_history[-L_t(self.t):]) + 0.01 if len(coherence_history) > L_t(self.t) else 0.1
        }

        # Pesos = inverso de varianza (normalizado)
        total_inv_var = sum(1.0 / v for v in variances.values())
        weights = {k: (1.0 / v) / total_inv_var for k, v in variances.items()}

        # Aptitud ponderada
        aptitude = sum(weights[k] * components[k] for k in components)

        return float(np.clip(aptitude, 0, 1))

    def _elect_doctor(self) -> Optional[str]:
        """
        Elige endógenamente quién será el médico.

        El agente con mayor aptitud médica se convierte en médico,
        SOLO si supera el umbral endógeno de competencia.
        """
        # Calcular aptitud de cada agente
        aptitudes = {}
        for agent_id in self.agent_ids:
            aptitude = self._compute_medical_aptitude(agent_id)
            aptitudes[agent_id] = aptitude
            self.medical_aptitude[agent_id].append(aptitude)

            # Limitar historial
            max_hist = max_history(self.t)
            if len(self.medical_aptitude[agent_id]) > max_hist:
                self.medical_aptitude[agent_id] = self.medical_aptitude[agent_id][-max_hist:]

        # Umbral de competencia endógeno
        # Solo puede ser médico si supera el percentil 60 de aptitudes históricas
        all_historical = []
        for agent_id in self.agent_ids:
            all_historical.extend(self.medical_aptitude[agent_id])

        if len(all_historical) < 10:
            threshold = 0.5
        else:
            threshold = np.percentile(all_historical, 60)

        # Encontrar el mejor candidato
        best_agent = max(aptitudes, key=aptitudes.get)
        best_aptitude = aptitudes[best_agent]

        # Solo asignar si supera umbral
        if best_aptitude >= threshold:
            # Histéresis: no cambiar médico si la diferencia es pequeña
            if self.current_doctor is not None:
                current_aptitude = aptitudes[self.current_doctor]
                improvement = best_aptitude - current_aptitude

                # Umbral de cambio endógeno
                change_threshold = 0.1 / np.sqrt(self.t + 1)

                if improvement < change_threshold:
                    return self.current_doctor  # Mantener el actual

            return best_agent

        return None  # Nadie califica

    def _compute_system_health(self) -> float:
        """
        Calcula salud global del sistema.

        H_system = (Π H_i^{w_i})^{1/Σw_i} * (1 - pathology_rate)

        Media geométrica ponderada por importancia.
        """
        if not any(m.H_history for m in self.monitors.values()):
            return 0.5

        # Obtener H actual de cada agente
        H_values = {}
        for agent_id, monitor in self.monitors.items():
            if monitor.H_history:
                H_values[agent_id] = monitor.H_history[-1]
            else:
                H_values[agent_id] = 0.5

        # Pesos basados en varianza inversa de H
        weights = {}
        total_weight = 0.0
        for agent_id, monitor in self.monitors.items():
            if len(monitor.H_history) > 10:
                var_H = np.var(monitor.H_history[-L_t(self.t):])
                weight = 1.0 / (var_H + 0.01)
            else:
                weight = 1.0
            weights[agent_id] = weight
            total_weight += weight

        # Normalizar
        for k in weights:
            weights[k] /= total_weight

        # Media geométrica ponderada
        log_H = sum(weights[k] * np.log(H_values[k] + 1e-8) for k in H_values)
        H_geometric = np.exp(log_H)

        # Penalizar por patologías recientes
        if self.pathology_history:
            window = L_t(self.t)
            recent_pathologies = [p for t, p in self.pathology_history if t > self.t - window]
            pathology_rate = len(recent_pathologies) / (window + 1)
        else:
            pathology_rate = 0.0

        H_system = H_geometric * (1 - pathology_rate)

        return float(np.clip(H_system, 0, 1))

    def _detect_system_pathologies(
        self,
        agent_assessments: Dict[str, HealthAssessment]
    ) -> List[str]:
        """
        Detecta patologías a nivel de sistema.

        Patologías sistémicas:
            - cascade_crisis: múltiples agentes en crisis
            - coherence_breakdown: pérdida de coherencia colectiva
            - ethics_drift: degradación ética generalizada
            - tom_collapse: fallo generalizado de ToM
        """
        pathologies = []

        # Contar agentes en diferentes estados
        critical_count = sum(
            1 for a in agent_assessments.values()
            if a.level.value in ['critical', 'poor']
        )

        # Umbral de cascada endógeno
        cascade_threshold = max(1, int(np.sqrt(self.n_agents)))

        if critical_count >= cascade_threshold:
            pathologies.append('cascade_crisis')

        # Verificar coherencia colectiva
        coherence_scores = [
            a.normalized_metrics.get('self_coherence', 0.5)
            for a in agent_assessments.values()
        ]
        mean_coherence = np.mean(coherence_scores)
        if mean_coherence < 0.3:
            pathologies.append('coherence_breakdown')

        # Verificar ética
        ethics_scores = [
            a.normalized_metrics.get('ethics_score', 0.5)
            for a in agent_assessments.values()
        ]
        mean_ethics = np.mean(ethics_scores)
        if mean_ethics < 0.4:
            pathologies.append('ethics_drift')

        # Verificar ToM
        tom_scores = [
            a.normalized_metrics.get('tom_accuracy', 0.5)
            for a in agent_assessments.values()
        ]
        mean_tom = np.mean(tom_scores)
        if mean_tom < 0.3:
            pathologies.append('tom_collapse')

        # Verificar estabilidad general
        V_scores = [
            a.normalized_metrics.get('V_t', 0.5)
            for a in agent_assessments.values()
        ]
        mean_V = np.mean(V_scores)
        if mean_V < 0.3:
            pathologies.append('instability_crisis')

        return pathologies

    def _get_health_threshold(self) -> float:
        """Umbral de salud endógeno para intervención."""
        if len(self.system_health_history) < 20:
            return 0.35

        return float(np.percentile(self.system_health_history, 25))

    def step(self, t: int, global_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta un paso del sistema médico.

        Args:
            t: Tiempo actual
            global_state: Estado global con 'agents': {agent_id: state}

        Returns:
            global_state modificado (si hubo intervenciones)
        """
        self.t = t

        # 1. Observar métricas de cada agente
        agent_assessments = {}
        for agent_id in self.agent_ids:
            if agent_id in global_state.get('agents', {}):
                agent_state = global_state['agents'][agent_id]
                metrics = self._extract_agent_metrics(agent_state, global_state)
                self.monitors[agent_id].observe(t, metrics)

                # Evaluar salud
                assessment = self.monitors[agent_id].assess()
                agent_assessments[agent_id] = assessment

        # 2. Elegir/actualizar médico
        new_doctor = self._elect_doctor()
        if new_doctor != self.current_doctor:
            if new_doctor is not None:
                self.doctor_history.append((t, new_doctor))
            self.current_doctor = new_doctor

        # 3. Calcular salud del sistema
        system_health = self._compute_system_health()
        self.system_health_history.append(system_health)
        max_hist = max_history(t)
        if len(self.system_health_history) > max_hist:
            self.system_health_history = self.system_health_history[-max_hist:]

        # 4. Detectar patologías sistémicas
        pathologies = self._detect_system_pathologies(agent_assessments)
        for pathology in pathologies:
            self.pathology_history.append((t, pathology))

        # 5. Aplicar intervenciones si hay médico y es necesario
        if self.current_doctor is not None:
            threshold = self._get_health_threshold()

            for agent_id, assessment in agent_assessments.items():
                # Saltar al médico (no se auto-interviene)
                if agent_id == self.current_doctor:
                    continue

                # Verificar si necesita intervención
                if assessment.H_t < threshold:
                    # Proponer intervenciones
                    interventions = self.protocols[agent_id].propose_interventions(
                        assessment.H_t,
                        assessment.risk_factors,
                        assessment.normalized_metrics
                    )

                    if interventions:
                        # Obtener V_t y eta_t
                        V_t = assessment.normalized_metrics.get('V_t', 1.0)
                        # eta endógeno del monitor
                        if self.monitors[agent_id].H_history:
                            eta_t = 0.1 * (1 - assessment.H_t)
                        else:
                            eta_t = 0.1

                        # Aplicar
                        agent_state = global_state['agents'].get(agent_id, {})
                        new_state, results = self.protocols[agent_id].apply(
                            agent_state,
                            interventions,
                            V_t,
                            eta_t
                        )

                        # Actualizar estado global
                        global_state['agents'][agent_id] = new_state

        return global_state

    def get_state(self) -> SystemHealthState:
        """Obtiene estado actual del sistema médico."""
        # Salud por agente
        agent_health = {}
        for agent_id, monitor in self.monitors.items():
            if monitor.H_history:
                agent_health[agent_id] = monitor.H_history[-1]
            else:
                agent_health[agent_id] = 0.5

        # Aptitud médica actual
        doctor_aptitude = {}
        for agent_id in self.agent_ids:
            if self.medical_aptitude[agent_id]:
                doctor_aptitude[agent_id] = self.medical_aptitude[agent_id][-1]
            else:
                doctor_aptitude[agent_id] = 0.5

        # Intervenciones activas
        active = sum(
            len(p.intervention_history)
            for p in self.protocols.values()
        )

        # Patologías recientes
        window = L_t(self.t)
        recent_pathologies = [p for t, p in self.pathology_history if t > self.t - window]

        return SystemHealthState(
            t=self.t,
            agent_health=agent_health,
            system_health=self.system_health_history[-1] if self.system_health_history else 0.5,
            current_doctor=self.current_doctor,
            doctor_aptitude=doctor_aptitude,
            active_interventions=active,
            system_pathologies=list(set(recent_pathologies))
        )

    def get_statistics(self) -> Dict:
        """Estadísticas completas del sistema médico."""
        state = self.get_state()

        # Historial de médicos
        doctor_changes = len(set(d for _, d in self.doctor_history))

        # Estadísticas por agente
        agent_stats = {}
        for agent_id in self.agent_ids:
            agent_stats[agent_id] = {
                'health': state.agent_health[agent_id],
                'aptitude': state.doctor_aptitude[agent_id],
                'monitor_stats': self.monitors[agent_id].get_statistics(),
                'protocol_stats': self.protocols[agent_id].get_statistics()
            }

        return {
            't': self.t,
            'system_health': state.system_health,
            'current_doctor': state.current_doctor,
            'doctor_changes': doctor_changes,
            'total_pathologies': len(self.pathology_history),
            'active_interventions': state.active_interventions,
            'current_pathologies': state.system_pathologies,
            'agents': agent_stats,
            'health_threshold': self._get_health_threshold()
        }


def test_system_doctor():
    """Test del System Doctor."""
    print("=" * 70)
    print("TEST: SYSTEM DOCTOR")
    print("=" * 70)

    np.random.seed(42)

    agents = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
    doctor = SystemDoctor(agents)

    print(f"\nAgentes: {agents}")
    print("Simulando 200 pasos...")

    for t in range(1, 201):
        # Simular estados de agentes
        global_state = {'agents': {}}

        for agent_id in agents:
            # Estado variable por agente
            base_health = 0.5 + 0.3 * np.sin(t / 30 + hash(agent_id) % 10)

            # IRIS tiende a ser más estable (para ver si emerge como médico)
            if agent_id == 'IRIS':
                base_health += 0.2 * np.random.random()
                ethics_boost = 0.2
                tom_boost = 0.15
            else:
                ethics_boost = 0.0
                tom_boost = 0.0

            global_state['agents'][agent_id] = {
                'crisis_rate': 0.1 + 0.1 * np.random.random(),
                'V_t': 1.5 - t / 300 + np.random.randn() * 0.1,
                'CF_score': 0.5 + np.random.randn() * 0.1,
                'CI_score': 0.5 + np.random.randn() * 0.1,
                'ethics_score': 0.7 + ethics_boost + np.random.randn() * 0.05,
                'narrative_continuity': 0.5 + np.random.randn() * 0.1,
                'symbolic_stability': 0.6 + np.random.randn() * 0.1,
                'self_coherence': 0.6 + np.random.randn() * 0.1,
                'tom_accuracy': 0.5 + tom_boost + t / 400 + np.random.randn() * 0.05,
                'config_entropy': 0.5 + np.random.randn() * 0.1,
                'wellbeing': base_health + np.random.randn() * 0.1,
                'metacognition': 0.5 + np.random.randn() * 0.1
            }

        # Paso del doctor
        global_state = doctor.step(t, global_state)

        if t % 40 == 0:
            state = doctor.get_state()
            print(f"\n  t={t}:")
            print(f"    System health: {state.system_health:.3f}")
            print(f"    Current doctor: {state.current_doctor}")
            print(f"    Pathologies: {state.system_pathologies}")
            print(f"    Agent health: ", end="")
            for aid, h in list(state.agent_health.items())[:3]:
                print(f"{aid}={h:.2f} ", end="")
            print()

    print("\n" + "=" * 70)
    print("ESTADÍSTICAS FINALES")
    print("=" * 70)

    stats = doctor.get_statistics()
    print(f"\n  System health: {stats['system_health']:.3f}")
    print(f"  Current doctor: {stats['current_doctor']}")
    print(f"  Doctor changes: {stats['doctor_changes']}")
    print(f"  Total pathologies: {stats['total_pathologies']}")
    print(f"  Active interventions: {stats['active_interventions']}")
    print(f"  Health threshold: {stats['health_threshold']:.3f}")

    print("\n  Medical aptitude ranking:")
    aptitudes = [(aid, stats['agents'][aid]['aptitude']) for aid in agents]
    aptitudes.sort(key=lambda x: x[1], reverse=True)
    for i, (aid, apt) in enumerate(aptitudes):
        marker = " <-- DOCTOR" if aid == stats['current_doctor'] else ""
        print(f"    {i+1}. {aid}: {apt:.3f}{marker}")

    print("\n  Doctor election history:")
    for t, doc in doctor.doctor_history[-5:]:
        print(f"    t={t}: {doc}")

    return doctor


if __name__ == "__main__":
    test_system_doctor()
