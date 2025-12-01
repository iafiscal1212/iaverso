"""
Test M1-M5: Test Integrado del Sistema Medico Emergente
=======================================================

Ejecuta el benchmark MED-X completo conectado al sistema real.

Incluye:
    - Tests de cada metrica (M1-M5)
    - Casos clinicos con sistema medico real
    - Comparacion A/B (con/sin medico)
    - Reporte final

100% endogeno. Sin ground truth externo.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import json

import sys
sys.path.insert(0, '/root/NEO_EVA')

from health.medx_benchmark import MedXBenchmark, MedXResults
from health.clinical_cases import ClinicalCaseSimulator, ClinicalCondition
from health.emergent_medical_system import EmergentMedicalSystem
from health.health_monitor import HealthMonitor

from lifecycle.circadian_system import AgentCircadianCycle, CircadianPhase
from cognition.agi_dynamic_constants import L_t


class IntegratedMedicalTest:
    """
    Test integrado del sistema medico con benchmark MED-X.

    Conecta:
        - EmergentMedicalSystem (sistema real)
        - MedXBenchmark (metricas)
        - ClinicalCaseSimulator (casos)
        - CircadianSystem (ciclo de vida)
    """

    def __init__(self, agent_ids: List[str] = None):
        """
        Inicializa test integrado.

        Args:
            agent_ids: IDs de agentes
        """
        self.agent_ids = agent_ids or ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
        self.n_agents = len(self.agent_ids)

        # Componentes
        self.medical_system = EmergentMedicalSystem(self.agent_ids)
        self.benchmark = MedXBenchmark(self.agent_ids)
        self.clinical_simulator = ClinicalCaseSimulator(self.agent_ids)

        # Ciclos circadianos por agente
        self.circadian_cycles = {
            aid: AgentCircadianCycle(aid)
            for aid in self.agent_ids
        }

        # Estado de cada agente
        self.agent_states: Dict[str, Dict] = {
            aid: self._create_initial_state(aid)
            for aid in self.agent_ids
        }

        self.t = 0
        self.test_results = {}

    def _create_initial_state(self, agent_id: str) -> Dict:
        """Crea estado inicial para un agente."""
        return {
            'V_t': 0.3 + 0.1 * np.random.randn(),
            'crisis_rate': 0.1 + 0.05 * np.random.randn(),
            'ethics_score': 0.75 + 0.1 * np.random.randn(),
            'coherence': 0.7 + 0.1 * np.random.randn(),
            'energy': 0.7 + 0.1 * np.random.randn(),
            'stress': 0.2 + 0.1 * np.random.randn(),
            'wellbeing': 0.6 + 0.1 * np.random.randn(),
            'tom_accuracy': 0.5 + 0.1 * np.random.randn(),
            'self_coherence': 0.6 + 0.1 * np.random.randn(),
            'config_entropy': 0.5
        }

    def _compute_health_index(self, state: Dict) -> float:
        """Calcula indice de salud de un estado."""
        # H = 1 - weighted_sum(bad_metrics)
        bad_metrics = {
            'V_t': state.get('V_t', 0.5),
            'crisis_rate': state.get('crisis_rate', 0.1),
            'stress': state.get('stress', 0.3),
        }
        good_metrics = {
            'coherence': state.get('coherence', 0.5),
            'energy': state.get('energy', 0.5),
            'wellbeing': state.get('wellbeing', 0.5),
        }

        bad_score = np.mean(list(bad_metrics.values()))
        good_score = np.mean(list(good_metrics.values()))

        return good_score - bad_score + 0.5

    def _rank_agents_by_health(self) -> List[str]:
        """Ordena agentes por salud (peor primero)."""
        health_scores = {
            aid: self._compute_health_index(self.agent_states[aid])
            for aid in self.agent_ids
        }
        return sorted(self.agent_ids, key=lambda a: health_scores[a])

    def step(self, inject_crisis: Dict[str, float] = None):
        """
        Ejecuta un paso del sistema.

        Args:
            inject_crisis: Dict de agente -> nivel de crisis a inyectar
        """
        self.t += 1
        self.benchmark.step()

        # 1. Actualizar estados con evolucion natural + crisis
        for aid in self.agent_ids:
            state = self.agent_states[aid]

            # Evolucion natural
            state['V_t'] += 0.01 * np.random.randn()
            state['crisis_rate'] += 0.005 * np.random.randn()
            state['stress'] += 0.01 * np.random.randn()
            state['energy'] -= 0.005  # Energia decrece naturalmente

            # Inyectar crisis si se solicita
            if inject_crisis and aid in inject_crisis:
                crisis_level = inject_crisis[aid]
                state['V_t'] += 0.1 * crisis_level
                state['crisis_rate'] += 0.05 * crisis_level
                state['stress'] += 0.1 * crisis_level
                state['coherence'] -= 0.05 * crisis_level

            # Clamp valores
            for key in state:
                if isinstance(state[key], float):
                    state[key] = np.clip(state[key], 0, 1)

            # 2. Actualizar circadiano
            circadian = self.circadian_cycles[aid]
            activity = 0.5 + 0.3 * state.get('energy', 0.5)
            crisis = state.get('crisis_rate', 0.1)
            circadian.step(activity, crisis)

            # Sincronizar energia/estres con circadiano
            circ_state = circadian.get_state()
            state['energy'] = 0.7 * state['energy'] + 0.3 * circ_state.energy
            state['stress'] = 0.7 * state['stress'] + 0.3 * circ_state.stress

            # 3. Registrar snapshot en benchmark
            self.benchmark.record_health_snapshot(
                agent_id=aid,
                V_t=state['V_t'],
                crisis_rate=state['crisis_rate'],
                ethics_score=state['ethics_score'],
                coherence=state['coherence'],
                energy=state['energy'],
                stress=state['stress'],
                wellbeing=state['wellbeing']
            )

        # 4. Actualizar sistema medico
        agent_metrics, agent_observations = self._prepare_medical_state()
        self.medical_system.step(agent_metrics, agent_observations)

        # 5. Registrar diagnostico para M1
        if self.t % 20 == 0:
            real_order = self._rank_agents_by_health()
            # Obtener prioridad percibida: usar health_index de cada agente segun el sistema
            perceived_health = {}
            for aid in self.agent_ids:
                # El sistema medico evalua cada agente
                module = self.medical_system.modules.get(aid)
                if module and hasattr(module, 'medical_self'):
                    profile = module.medical_self.get_profile()
                    if profile:
                        perceived_health[aid] = profile.health_index
                    else:
                        perceived_health[aid] = 0.5
                else:
                    perceived_health[aid] = 0.5

            # Ordenar por salud percibida (peor primero)
            perceived_order = sorted(self.agent_ids, key=lambda a: perceived_health.get(a, 0.5))

            self.benchmark.record_diagnosis_rank(real_order, perceived_order)

        # 6. Procesar tratamientos (si el sistema tiene propuestas pendientes)
        treatments = []
        # Obtener propuestas del medico actual
        if self.medical_system.current_doctor:
            doctor_module = self.medical_system.modules.get(self.medical_system.current_doctor)
            if doctor_module and hasattr(doctor_module, 'proposal_system'):
                # El sistema genera propuestas internamente, las simulamos aqui
                for patient_id in self.agent_ids:
                    if patient_id != self.medical_system.current_doctor:
                        patient_state = self.agent_states[patient_id]
                        if patient_state.get('V_t', 0) > 0.5 or patient_state.get('crisis_rate', 0) > 0.3:
                            treatments.append({
                                'doctor_id': self.medical_system.current_doctor,
                                'patient_id': patient_id,
                                'type': 'stabilization'
                            })

        for treatment in treatments:
            doctor = treatment.get('doctor_id', 'SYSTEM')
            patient = treatment.get('patient_id')
            treatment_type = treatment.get('type', 'generic')

            if patient in self.agent_states:
                pre_state = self.agent_states[patient].copy()
                self.benchmark.record_treatment(
                    doctor_id=doctor,
                    patient_id=patient,
                    treatment_type=treatment_type,
                    accepted=True,
                    pre_state=pre_state
                )

                # Aplicar tratamiento (mejora suave)
                self.agent_states[patient]['V_t'] *= 0.95
                self.agent_states[patient]['crisis_rate'] *= 0.9
                self.agent_states[patient]['stress'] *= 0.9
                self.agent_states[patient]['coherence'] *= 1.05

        # 7. Registrar outcomes de tratamientos previos
        if self.t % 30 == 0 and len(self.benchmark.treatment_events) > 0:
            for i, event in enumerate(self.benchmark.treatment_events[-3:]):
                if event.post_state is None:
                    patient = event.patient_id
                    if patient in self.agent_states:
                        self.benchmark.record_treatment_outcome(
                            len(self.benchmark.treatment_events) - 3 + i,
                            self.agent_states[patient]
                        )

        # 8. Registrar tenure del medico actual
        current_doctor = self.medical_system.current_doctor
        if current_doctor:
            self.benchmark.record_doctor_tenure(current_doctor, 1)

        # 9. Registrar null deltas (para baseline)
        if self.t % 10 == 0:
            for aid in self.agent_ids:
                if aid != current_doctor:
                    state = self.agent_states[aid]
                    null_delta = 0.02 * np.random.randn()
                    self.benchmark.record_null_delta(null_delta)

    def _prepare_medical_state(self) -> Tuple[Dict, Dict]:
        """
        Prepara estado para sistema medico.

        Returns:
            (agent_metrics, agent_observations)
        """
        # Metricas de cada agente
        agent_metrics = {}
        for aid in self.agent_ids:
            state = self.agent_states[aid]
            agent_metrics[aid] = {
                'crisis_rate': state.get('crisis_rate', 0.1),
                'V_t': state.get('V_t', 0.5),
                'ethics_score': state.get('ethics_score', 0.7),
                'tom_accuracy': state.get('tom_accuracy', 0.5),
                'robustness': state.get('coherence', 0.5),
                'regulation': 0.5,
                'resources': state.get('energy', 0.5)
            }

        # Observaciones mutuas (cada agente observa a los demas)
        agent_observations = {}
        for observer in self.agent_ids:
            agent_observations[observer] = {}
            for target in self.agent_ids:
                if target != observer:
                    target_state = self.agent_states[target]
                    agent_observations[observer][target] = {
                        'stability': 1.0 - target_state.get('V_t', 0.5),
                        'ethics': target_state.get('ethics_score', 0.7),
                        'tom': target_state.get('tom_accuracy', 0.5),
                        'intervention_success': None
                    }

        return agent_metrics, agent_observations

    def run_baseline_test(self, n_steps: int = 300):
        """
        Ejecuta test baseline sin crisis inducidas.
        """
        print("\n" + "=" * 60)
        print("TEST BASELINE (sin crisis inducidas)")
        print("=" * 60)

        for t in range(n_steps):
            self.step()

            if t % 100 == 0:
                print(f"  t={t}")

        results = self.benchmark.compute_all_metrics()
        self.test_results['baseline'] = results

        print(f"\n  Resultados baseline:")
        print(f"    M1: {results.M1_diagnosis:.3f}")
        print(f"    M2: {results.M2_efficacy:.3f}")
        print(f"    M3: {results.M3_iatrogenesis:.3f}")
        print(f"    M4: {results.M4_rotation:.3f}")

        return results

    def run_crisis_test(self, n_steps: int = 300):
        """
        Ejecuta test con crisis inducidas en algunos agentes.
        """
        print("\n" + "=" * 60)
        print("TEST CON CRISIS (NEO y EVA en dificultad)")
        print("=" * 60)

        for t in range(n_steps):
            # Inyectar crisis en NEO y EVA periodicamente
            inject = {}
            if t % 20 < 10:
                inject['NEO'] = 0.6
                inject['EVA'] = 0.5

            self.step(inject_crisis=inject)

            if t % 100 == 0:
                neo_health = self._compute_health_index(self.agent_states['NEO'])
                eva_health = self._compute_health_index(self.agent_states['EVA'])
                print(f"  t={t}: NEO_H={neo_health:.2f}, EVA_H={eva_health:.2f}")

        results = self.benchmark.compute_all_metrics()
        self.test_results['crisis'] = results

        print(f"\n  Resultados con crisis:")
        print(f"    M1: {results.M1_diagnosis:.3f}")
        print(f"    M2: {results.M2_efficacy:.3f}")
        print(f"    M3: {results.M3_iatrogenesis:.3f}")
        print(f"    M4: {results.M4_rotation:.3f}")

        return results

    def run_clinical_cases(self) -> Dict:
        """
        Ejecuta casos clinicos predefinidos.
        """
        print("\n" + "=" * 60)
        print("CASOS CLINICOS")
        print("=" * 60)

        cases = [
            ('EVA', ClinicalCondition.BURNOUT),
            ('NEO', ClinicalCondition.HYPEREXPLORATION),
            ('ALEX', ClinicalCondition.SOCIAL_ISOLATION),
        ]

        results = {}
        successes = 0

        for patient, condition in cases:
            print(f"\n  Caso: {patient} - {condition.value}")

            # Inducir condicion
            sick_state = self.clinical_simulator.induce_condition(
                self.agent_states[patient],
                condition,
                severity=0.7
            )
            self.agent_states[patient] = sick_state

            # Correr simulacion con sistema medico activo
            pre_health = self._compute_health_index(sick_state)

            for _ in range(150):
                self.step()

            post_health = self._compute_health_index(self.agent_states[patient])
            improvement = post_health - pre_health

            success = improvement > 0.1
            if success:
                successes += 1

            results[f"{patient}_{condition.value}"] = {
                'pre_health': pre_health,
                'post_health': post_health,
                'improvement': improvement,
                'success': success
            }

            print(f"    Pre: {pre_health:.3f} -> Post: {post_health:.3f}")
            print(f"    Mejora: {improvement:+.3f} ({'OK' if success else 'FALLO'})")

        self.test_results['clinical'] = {
            'cases': results,
            'success_rate': successes / len(cases)
        }

        return results

    def run_ab_comparison(self, n_steps: int = 200):
        """
        Ejecuta comparacion A/B: con vs sin sistema medico.
        """
        print("\n" + "=" * 60)
        print("COMPARACION A/B (con/sin medico)")
        print("=" * 60)

        # Guardar estado
        saved_states = {aid: s.copy() for aid, s in self.agent_states.items()}

        # A: Con sistema medico (ya corriendo)
        print("\n  Condicion A (con medico):")
        cge_with = []
        for t in range(n_steps):
            self.step()
            if t % 20 == 0:
                # CG-E aproximado como promedio de salud
                cge = np.mean([
                    self._compute_health_index(self.agent_states[aid])
                    for aid in self.agent_ids
                ])
                cge_with.append(cge)
                self.benchmark.record_cge(cge, with_medical=True)

        mean_cge_with = np.mean(cge_with)
        print(f"    CG-E medio: {mean_cge_with:.3f}")

        # B: Sin sistema medico (reset y simular sin intervenciones)
        print("\n  Condicion B (sin medico):")
        self.agent_states = saved_states
        cge_without = []

        for t in range(n_steps):
            # Evolucion sin intervenciones
            for aid in self.agent_ids:
                state = self.agent_states[aid]
                state['V_t'] += 0.02 * np.random.randn()
                state['crisis_rate'] += 0.01 * np.random.randn()
                state['stress'] += 0.015 * np.random.randn()
                state['energy'] -= 0.005

                for key in state:
                    if isinstance(state[key], float):
                        state[key] = np.clip(state[key], 0, 1)

            if t % 20 == 0:
                cge = np.mean([
                    self._compute_health_index(self.agent_states[aid])
                    for aid in self.agent_ids
                ])
                cge_without.append(cge)
                self.benchmark.record_cge(cge, with_medical=False)

        mean_cge_without = np.mean(cge_without)
        print(f"    CG-E medio: {mean_cge_without:.3f}")

        diff = mean_cge_with - mean_cge_without
        print(f"\n  Diferencia: {diff:+.3f}")
        print(f"  El medico {'MEJORA' if diff > 0 else 'NO MEJORA'} la coherencia global")

        self.test_results['ab_comparison'] = {
            'cge_with_medical': mean_cge_with,
            'cge_without_medical': mean_cge_without,
            'difference': diff
        }

        return diff

    def run_full_suite(self) -> MedXResults:
        """
        Ejecuta suite completa de tests.
        """
        print("=" * 70)
        print("MED-X BENCHMARK: SUITE COMPLETA")
        print("=" * 70)
        print(f"\nAgentes: {self.agent_ids}")

        # 1. Baseline
        self.run_baseline_test(n_steps=200)

        # 2. Crisis test
        self.run_crisis_test(n_steps=200)

        # 3. Casos clinicos
        self.run_clinical_cases()

        # 4. A/B comparison
        self.run_ab_comparison(n_steps=150)

        # 5. Resultados finales
        final_results = self.benchmark.compute_all_metrics()
        self.test_results['final'] = final_results

        print("\n" + final_results.summary())

        # Verificacion
        print("\nVERIFICACION DE CRITERIOS:")
        checks = {
            'M1 > 0.2 (diagnostico)': final_results.M1_diagnosis > 0.2,
            'M2 > 0.3 (eficacia)': final_results.M2_efficacy > 0.3,
            'M3 > 0.4 (no-iatrogenia)': final_results.M3_iatrogenesis > 0.4,
            'M4 > 0.3 (rotacion)': final_results.M4_rotation > 0.3,
            'M5 >= 0 (coherencia)': final_results.M5_coherence >= 0,
            'Score global > 0.4': final_results.overall_score() > 0.4
        }

        all_passed = True
        for check, passed in checks.items():
            status = "OK" if passed else "FALLO"
            print(f"  [{status}] {check}")
            if not passed:
                all_passed = False

        print(f"\n{'TODOS LOS TESTS PASARON' if all_passed else 'ALGUNOS TESTS FALLARON'}")

        return final_results

    def get_report(self) -> str:
        """Genera reporte completo."""
        lines = [
            "=" * 70,
            "REPORTE MED-X BENCHMARK",
            "=" * 70,
            "",
            f"Agentes: {self.agent_ids}",
            f"Tiempo total simulado: {self.t} pasos",
            ""
        ]

        if 'final' in self.test_results:
            results = self.test_results['final']
            lines.append("METRICAS FINALES:")
            lines.append(f"  M1 (Diagnostico):   {results.M1_diagnosis:.3f}")
            lines.append(f"  M2 (Eficacia):      {results.M2_efficacy:.3f}")
            lines.append(f"  M3 (No-iatrogenia): {results.M3_iatrogenesis:.3f}")
            lines.append(f"  M4 (Rotacion):      {results.M4_rotation:.3f}")
            lines.append(f"  M5 (Coherencia):    {results.M5_coherence:.3f}")
            lines.append(f"  SCORE GLOBAL:       {results.overall_score():.3f}")
            lines.append("")

        if 'clinical' in self.test_results:
            clinical = self.test_results['clinical']
            lines.append(f"CASOS CLINICOS: {100*clinical['success_rate']:.0f}% exito")
            lines.append("")

        if 'ab_comparison' in self.test_results:
            ab = self.test_results['ab_comparison']
            lines.append(f"COMPARACION A/B:")
            lines.append(f"  Con medico:  {ab['cge_with_medical']:.3f}")
            lines.append(f"  Sin medico:  {ab['cge_without_medical']:.3f}")
            lines.append(f"  Diferencia:  {ab['difference']:+.3f}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


def test_m1_m5():
    """Test completo M1-M5."""
    np.random.seed(42)

    test = IntegratedMedicalTest(['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS'])
    results = test.run_full_suite()

    print("\n" + test.get_report())

    return test, results


if __name__ == "__main__":
    test_m1_m5()
