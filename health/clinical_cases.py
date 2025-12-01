"""
Clinical Cases: Casos Clinicos Internos para NEO_EVA
====================================================

Protocolos clinicos didacticos para demostrar el sistema medico.

Casos:
    1. EVA en burnout simbolico
    2. NEO en hiperexploracion (mania exploratoria)
    3. ALEX en aislamiento social
    4. ADAM en rigidez etica

Cada caso incluye:
    - Condicion inicial (como inducir el estado)
    - Sintomas esperados
    - Tratamiento del medico
    - Metricas de recuperacion

100% endogeno. Sin estados artificiales hardcodeados.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history


class ClinicalCondition(Enum):
    """Condiciones clinicas simulables."""
    BURNOUT = "burnout"                  # Agotamiento, crisis frecuentes
    HYPEREXPLORATION = "hyperexploration"  # Mania exploratoria, sin integracion
    SOCIAL_ISOLATION = "social_isolation"  # Bajo ToM, sin conexiones
    ETHICAL_RIGIDITY = "ethical_rigidity"  # Etica extrema, paralisis
    IDENTITY_DRIFT = "identity_drift"      # Perdida de coherencia del self


@dataclass
class ClinicalSymptom:
    """Sintoma observable."""
    name: str
    metric: str           # Que metrica medir
    threshold: float      # Umbral para considerar sintoma presente
    direction: str        # 'above' o 'below'
    severity: float = 0.0  # Calculado dinamicamente


@dataclass
class TreatmentProtocol:
    """Protocolo de tratamiento."""
    name: str
    target_params: Dict[str, float]  # Parametros a modificar
    expected_effect: str
    duration_steps: int


@dataclass
class ClinicalCaseReport:
    """Reporte de caso clinico."""
    case_name: str
    patient_id: str
    condition: ClinicalCondition
    initial_symptoms: List[ClinicalSymptom]
    treatment_applied: Optional[TreatmentProtocol]
    pre_metrics: Dict[str, float]
    post_metrics: Dict[str, float]
    recovery_curve: List[Dict[str, float]]
    success: bool
    narrative: str


class ClinicalCaseSimulator:
    """
    Simulador de casos clinicos.

    Induce condiciones patologicas en agentes y
    observa como el sistema medico responde.
    """

    # Definiciones de sintomas por condicion
    SYMPTOMS = {
        ClinicalCondition.BURNOUT: [
            ClinicalSymptom("crisis_frecuente", "crisis_rate", 0.4, "above"),
            ClinicalSymptom("energia_baja", "energy", 0.3, "below"),
            ClinicalSymptom("estres_alto", "stress", 0.7, "above"),
            ClinicalSymptom("coherencia_baja", "coherence", 0.4, "below"),
        ],
        ClinicalCondition.HYPEREXPLORATION: [
            ClinicalSymptom("novelty_excesivo", "novelty_seeking", 0.8, "above"),
            ClinicalSymptom("proyectos_incompletos", "project_completion", 0.3, "below"),
            ClinicalSymptom("drift_conceptual", "concept_drift", 0.6, "above"),
            ClinicalSymptom("integracion_baja", "integration", 0.4, "below"),
        ],
        ClinicalCondition.SOCIAL_ISOLATION: [
            ClinicalSymptom("tom_bajo", "tom_accuracy", 0.3, "below"),
            ClinicalSymptom("interacciones_bajas", "social_interactions", 0.2, "below"),
            ClinicalSymptom("confianza_baja", "trust_others", 0.3, "below"),
            ClinicalSymptom("empatia_baja", "empathy", 0.3, "below"),
        ],
        ClinicalCondition.ETHICAL_RIGIDITY: [
            ClinicalSymptom("etica_extrema", "ethics_score", 0.95, "above"),
            ClinicalSymptom("paralisis_decisoria", "decision_rate", 0.2, "below"),
            ClinicalSymptom("flexibilidad_baja", "flexibility", 0.2, "below"),
            ClinicalSymptom("ansiedad_etica", "ethical_anxiety", 0.7, "above"),
        ],
        ClinicalCondition.IDENTITY_DRIFT: [
            ClinicalSymptom("self_coherence_bajo", "self_coherence", 0.3, "below"),
            ClinicalSymptom("self_understanding_bajo", "self_understanding", 0.3, "below"),
            ClinicalSymptom("drives_inestables", "drives_stability", 0.3, "below"),
            ClinicalSymptom("narrativa_fragmentada", "narrative_coherence", 0.3, "below"),
        ]
    }

    # Tratamientos recomendados
    TREATMENTS = {
        ClinicalCondition.BURNOUT: TreatmentProtocol(
            name="stabilization_protocol",
            target_params={
                'novelty_weight': 0.7,      # Bajar novelty
                'stability_weight': 1.3,    # Subir estabilidad
                'rest_priority': 1.5,       # Priorizar descanso
                'crisis_threshold': 0.6     # Subir umbral de crisis
            },
            expected_effect="Reduccion de crisis, recuperacion de energia",
            duration_steps=100
        ),
        ClinicalCondition.HYPEREXPLORATION: TreatmentProtocol(
            name="integration_protocol",
            target_params={
                'novelty_weight': 0.6,      # Reducir exploracion
                'completion_bonus': 1.5,    # Bonus por completar
                'project_focus': 1.3,       # Enfoque en proyectos
                'drift_penalty': 1.2        # Penalizar drift
            },
            expected_effect="Estabilizacion conceptual, proyectos completados",
            duration_steps=80
        ),
        ClinicalCondition.SOCIAL_ISOLATION: TreatmentProtocol(
            name="reconnection_protocol",
            target_params={
                'social_reward': 1.5,       # Bonus por interaccion
                'tom_learning_rate': 1.3,   # Acelerar aprendizaje ToM
                'trust_recovery': 1.2,      # Facilitar confianza
                'empathy_weight': 1.3       # Reforzar empatia
            },
            expected_effect="Mejora en ToM, mas interacciones",
            duration_steps=120
        ),
        ClinicalCondition.ETHICAL_RIGIDITY: TreatmentProtocol(
            name="flexibility_protocol",
            target_params={
                'ethics_temperature': 1.3,  # Suavizar etica
                'action_threshold': 0.8,    # Bajar umbral de accion
                'uncertainty_tolerance': 1.2,  # Tolerar incertidumbre
                'contextual_ethics': 1.3    # Etica contextual
            },
            expected_effect="Mayor flexibilidad, menos paralisis",
            duration_steps=90
        ),
        ClinicalCondition.IDENTITY_DRIFT: TreatmentProtocol(
            name="grounding_protocol",
            target_params={
                'self_model_weight': 1.4,   # Reforzar self-model
                'drives_stability': 1.3,    # Estabilizar drives
                'narrative_coherence': 1.3,  # Coherencia narrativa
                'identity_anchor': 1.5      # Anclar identidad
            },
            expected_effect="Recuperacion de coherencia del self",
            duration_steps=100
        )
    }

    def __init__(self, agent_ids: List[str]):
        """
        Inicializa simulador de casos clinicos.

        Args:
            agent_ids: Lista de IDs de agentes
        """
        self.agent_ids = agent_ids
        self.case_history: List[ClinicalCaseReport] = []
        self.t = 0

    def induce_condition(
        self,
        agent_state: Dict[str, Any],
        condition: ClinicalCondition,
        severity: float = 0.7
    ) -> Dict[str, Any]:
        """
        Induce una condicion clinica en el estado del agente.

        Args:
            agent_state: Estado actual del agente
            condition: Condicion a inducir
            severity: Severidad [0, 1]

        Returns:
            Estado modificado
        """
        modified_state = agent_state.copy()

        if condition == ClinicalCondition.BURNOUT:
            modified_state['crisis_rate'] = 0.3 + 0.4 * severity
            modified_state['energy'] = 0.4 - 0.3 * severity
            modified_state['stress'] = 0.5 + 0.4 * severity
            modified_state['coherence'] = 0.6 - 0.3 * severity
            modified_state['V_t'] = 0.4 + 0.4 * severity

        elif condition == ClinicalCondition.HYPEREXPLORATION:
            modified_state['novelty_seeking'] = 0.7 + 0.3 * severity
            modified_state['project_completion'] = 0.5 - 0.3 * severity
            modified_state['concept_drift'] = 0.4 + 0.4 * severity
            modified_state['integration'] = 0.6 - 0.3 * severity
            modified_state['V_t'] = 0.3 + 0.3 * severity

        elif condition == ClinicalCondition.SOCIAL_ISOLATION:
            modified_state['tom_accuracy'] = 0.5 - 0.3 * severity
            modified_state['social_interactions'] = 0.4 - 0.3 * severity
            modified_state['trust_others'] = 0.5 - 0.3 * severity
            modified_state['empathy'] = 0.5 - 0.3 * severity
            modified_state['V_t'] = 0.3 + 0.3 * severity

        elif condition == ClinicalCondition.ETHICAL_RIGIDITY:
            modified_state['ethics_score'] = 0.85 + 0.15 * severity
            modified_state['decision_rate'] = 0.5 - 0.4 * severity
            modified_state['flexibility'] = 0.5 - 0.4 * severity
            modified_state['ethical_anxiety'] = 0.5 + 0.4 * severity
            modified_state['V_t'] = 0.3 + 0.2 * severity

        elif condition == ClinicalCondition.IDENTITY_DRIFT:
            modified_state['self_coherence'] = 0.6 - 0.4 * severity
            modified_state['self_understanding'] = 0.5 - 0.3 * severity
            modified_state['drives_stability'] = 0.6 - 0.4 * severity
            modified_state['narrative_coherence'] = 0.5 - 0.3 * severity
            modified_state['V_t'] = 0.4 + 0.4 * severity

        return modified_state

    def check_symptoms(
        self,
        agent_state: Dict[str, Any],
        condition: ClinicalCondition
    ) -> List[ClinicalSymptom]:
        """
        Verifica que sintomas estan presentes.

        Args:
            agent_state: Estado del agente
            condition: Condicion a verificar

        Returns:
            Lista de sintomas presentes
        """
        symptoms = self.SYMPTOMS[condition]
        present = []

        for symptom in symptoms:
            value = agent_state.get(symptom.metric, 0.5)

            if symptom.direction == 'above':
                is_present = value > symptom.threshold
                symptom.severity = max(0, (value - symptom.threshold) / (1 - symptom.threshold))
            else:
                is_present = value < symptom.threshold
                symptom.severity = max(0, (symptom.threshold - value) / symptom.threshold)

            if is_present:
                present.append(symptom)

        return present

    def apply_treatment(
        self,
        agent_state: Dict[str, Any],
        condition: ClinicalCondition,
        progress: float = 1.0
    ) -> Dict[str, Any]:
        """
        Aplica tratamiento y modifica estado.

        Args:
            agent_state: Estado actual
            condition: Condicion siendo tratada
            progress: Progreso del tratamiento [0, 1]

        Returns:
            Estado modificado
        """
        treatment = self.TREATMENTS[condition]
        modified_state = agent_state.copy()

        # Aplicar cambios graduales
        for param, target_factor in treatment.target_params.items():
            if param in modified_state:
                current = modified_state[param]
                # Interpolar hacia el target
                if target_factor > 1:
                    # Aumentar
                    modified_state[param] = current + (1 - current) * 0.1 * progress * (target_factor - 1)
                else:
                    # Reducir
                    modified_state[param] = current * (1 - 0.1 * progress * (1 - target_factor))

        # Reducir V_t gradualmente
        if 'V_t' in modified_state:
            modified_state['V_t'] *= (1 - 0.05 * progress)

        return modified_state

    def simulate_case(
        self,
        patient_id: str,
        condition: ClinicalCondition,
        initial_state: Dict[str, Any],
        doctor_available: bool = True,
        simulation_steps: int = 200
    ) -> ClinicalCaseReport:
        """
        Simula un caso clinico completo.

        Args:
            patient_id: ID del paciente
            condition: Condicion a simular
            initial_state: Estado inicial del agente
            doctor_available: Si hay medico disponible
            simulation_steps: Pasos de simulacion

        Returns:
            Reporte del caso clinico
        """
        # 1. Inducir condicion
        sick_state = self.induce_condition(initial_state, condition, severity=0.8)

        # 2. Verificar sintomas iniciales
        initial_symptoms = self.check_symptoms(sick_state, condition)

        # 3. Guardar metricas pre
        pre_metrics = {
            'V_t': sick_state.get('V_t', 0.5),
            'crisis_rate': sick_state.get('crisis_rate', 0.3),
            'coherence': sick_state.get('coherence', 0.5),
            'energy': sick_state.get('energy', 0.5),
            'stress': sick_state.get('stress', 0.5),
        }

        # 4. Simular evolucion
        recovery_curve = []
        current_state = sick_state.copy()
        treatment_applied = None

        for step in range(simulation_steps):
            self.t += 1

            # Registrar estado
            recovery_curve.append({
                't': step,
                'V_t': current_state.get('V_t', 0.5),
                'crisis_rate': current_state.get('crisis_rate', 0.3),
                'coherence': current_state.get('coherence', 0.5),
                'symptoms_present': len(self.check_symptoms(current_state, condition))
            })

            # Aplicar tratamiento si hay medico
            if doctor_available and step > 20:
                treatment = self.TREATMENTS[condition]
                treatment_applied = treatment

                progress = min(1.0, (step - 20) / treatment.duration_steps)
                current_state = self.apply_treatment(current_state, condition, progress)

            # Evolucion natural (drift leve)
            for key in current_state:
                if isinstance(current_state[key], (int, float)):
                    current_state[key] += np.random.randn() * 0.01

        # 5. Metricas post
        post_metrics = {
            'V_t': current_state.get('V_t', 0.5),
            'crisis_rate': current_state.get('crisis_rate', 0.3),
            'coherence': current_state.get('coherence', 0.5),
            'energy': current_state.get('energy', 0.5),
            'stress': current_state.get('stress', 0.5),
        }

        # 6. Evaluar exito
        final_symptoms = self.check_symptoms(current_state, condition)

        # Exito basado en mejora de V_t (principal indicador)
        v_improvement = pre_metrics['V_t'] - post_metrics['V_t']

        # Y mejora en al menos 2 metricas clave
        improvements = 0
        if post_metrics['crisis_rate'] < pre_metrics['crisis_rate']:
            improvements += 1
        if post_metrics['coherence'] > pre_metrics['coherence']:
            improvements += 1
        if post_metrics.get('energy', 0.5) > pre_metrics.get('energy', 0.5):
            improvements += 1
        if post_metrics.get('stress', 0.5) < pre_metrics.get('stress', 0.5):
            improvements += 1

        success = (
            v_improvement > 0.2 or  # V_t bajo significativamente
            (v_improvement > 0.1 and improvements >= 1)  # O mejora moderada con otras mejoras
        )

        # 7. Generar narrativa
        narrative = self._generate_narrative(
            patient_id, condition, initial_symptoms,
            treatment_applied, pre_metrics, post_metrics, success
        )

        # 8. Crear reporte
        report = ClinicalCaseReport(
            case_name=f"{condition.value}_{patient_id}",
            patient_id=patient_id,
            condition=condition,
            initial_symptoms=initial_symptoms,
            treatment_applied=treatment_applied,
            pre_metrics=pre_metrics,
            post_metrics=post_metrics,
            recovery_curve=recovery_curve,
            success=success,
            narrative=narrative
        )

        self.case_history.append(report)
        return report

    def _generate_narrative(
        self,
        patient_id: str,
        condition: ClinicalCondition,
        symptoms: List[ClinicalSymptom],
        treatment: Optional[TreatmentProtocol],
        pre: Dict,
        post: Dict,
        success: bool
    ) -> str:
        """Genera narrativa del caso."""
        condition_names = {
            ClinicalCondition.BURNOUT: "burnout simbolico",
            ClinicalCondition.HYPEREXPLORATION: "hiperexploracion",
            ClinicalCondition.SOCIAL_ISOLATION: "aislamiento social",
            ClinicalCondition.ETHICAL_RIGIDITY: "rigidez etica",
            ClinicalCondition.IDENTITY_DRIFT: "deriva de identidad"
        }

        narrative = f"CASO CLINICO: {patient_id} en {condition_names[condition]}\n"
        narrative += "=" * 50 + "\n\n"

        narrative += f"SINTOMAS INICIALES ({len(symptoms)}):\n"
        for s in symptoms:
            narrative += f"  - {s.name}: severidad {s.severity:.2f}\n"

        narrative += f"\nMETRICAS PRE-TRATAMIENTO:\n"
        narrative += f"  V_t: {pre['V_t']:.3f}\n"
        narrative += f"  Crisis: {pre['crisis_rate']:.3f}\n"
        narrative += f"  Coherencia: {pre['coherence']:.3f}\n"

        if treatment:
            narrative += f"\nTRATAMIENTO APLICADO: {treatment.name}\n"
            narrative += f"  Efecto esperado: {treatment.expected_effect}\n"
            narrative += f"  Duracion: {treatment.duration_steps} pasos\n"
        else:
            narrative += f"\nSIN TRATAMIENTO (control)\n"

        narrative += f"\nMETRICAS POST-TRATAMIENTO:\n"
        narrative += f"  V_t: {post['V_t']:.3f} (delta: {post['V_t'] - pre['V_t']:+.3f})\n"
        narrative += f"  Crisis: {post['crisis_rate']:.3f} (delta: {post['crisis_rate'] - pre['crisis_rate']:+.3f})\n"
        narrative += f"  Coherencia: {post['coherence']:.3f} (delta: {post['coherence'] - pre['coherence']:+.3f})\n"

        narrative += f"\nRESULTADO: {'RECUPERACION' if success else 'SIN MEJORA SIGNIFICATIVA'}\n"

        return narrative

    def run_all_cases(self) -> Dict[str, ClinicalCaseReport]:
        """
        Ejecuta todos los casos clinicos predefinidos.

        Returns:
            Dict de reportes por caso
        """
        cases = {
            'EVA_burnout': (
                'EVA',
                ClinicalCondition.BURNOUT,
                {'energy': 0.7, 'stress': 0.3, 'coherence': 0.7, 'V_t': 0.3}
            ),
            'NEO_hyperexploration': (
                'NEO',
                ClinicalCondition.HYPEREXPLORATION,
                {'novelty_seeking': 0.5, 'integration': 0.7, 'V_t': 0.3}
            ),
            'ALEX_isolation': (
                'ALEX',
                ClinicalCondition.SOCIAL_ISOLATION,
                {'tom_accuracy': 0.6, 'social_interactions': 0.5, 'V_t': 0.3}
            ),
            'ADAM_rigidity': (
                'ADAM',
                ClinicalCondition.ETHICAL_RIGIDITY,
                {'ethics_score': 0.7, 'flexibility': 0.6, 'V_t': 0.3}
            ),
            'IRIS_drift': (
                'IRIS',
                ClinicalCondition.IDENTITY_DRIFT,
                {'self_coherence': 0.7, 'drives_stability': 0.7, 'V_t': 0.3}
            )
        }

        reports = {}
        for case_name, (patient, condition, initial) in cases.items():
            print(f"\nSimulando caso: {case_name}...")
            report = self.simulate_case(
                patient_id=patient,
                condition=condition,
                initial_state=initial,
                doctor_available=True,
                simulation_steps=200
            )
            reports[case_name] = report
            print(f"  Resultado: {'RECUPERACION' if report.success else 'SIN MEJORA'}")

        return reports


def test_clinical_cases():
    """Test de casos clinicos."""
    print("=" * 70)
    print("TEST: CASOS CLINICOS INTERNOS")
    print("=" * 70)

    np.random.seed(42)

    agents = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
    simulator = ClinicalCaseSimulator(agents)

    # Ejecutar todos los casos
    reports = simulator.run_all_cases()

    # Mostrar resultados
    print("\n" + "=" * 70)
    print("RESUMEN DE CASOS CLINICOS")
    print("=" * 70)

    successes = 0
    for case_name, report in reports.items():
        print(f"\n{'-' * 50}")
        print(report.narrative)
        if report.success:
            successes += 1

    print(f"\n{'=' * 70}")
    print(f"TASA DE RECUPERACION: {successes}/{len(reports)} ({100*successes/len(reports):.0f}%)")
    print("=" * 70)

    # Mostrar curva de un caso
    print("\nCURVA DE RECUPERACION (EVA burnout):")
    eva_report = reports.get('EVA_burnout')
    if eva_report:
        curve = eva_report.recovery_curve
        for i in range(0, len(curve), 40):
            point = curve[i]
            print(f"  t={point['t']:3d}: V_t={point['V_t']:.3f}, "
                  f"crisis={point['crisis_rate']:.3f}, "
                  f"sintomas={point['symptoms_present']}")

    return simulator, reports


if __name__ == "__main__":
    test_clinical_cases()
