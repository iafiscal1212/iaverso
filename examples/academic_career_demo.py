#!/usr/bin/env python3
"""
DEMO: SISTEMA DE CARRERAS ACADEMICAS
=====================================

Demuestra como funciona el sistema de carreras academicas emergentes:

1. Agentes hacen tareas
2. Miden afinidad
3. Emerge especializacion
4. Progresan en niveles academicos (grado -> master -> doctorado)
5. Generan etiquetas emergentes

NORMA DURA EN ACCION:
- Sin "if accuracy > 0.8" -> usa percentil 80 del PROPIO historial
- Sin "is_physicist = True" -> etiquetas emergentes de metricas
- Seleccion de dominio por softmax de afinidades (no hardcodeada)
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from domains.specialization.career_integrated_engine import (
    CareerIntegratedEngine, ResearchDirector
)
from domains.specialization.academic_career import AcademicLevel


def demo_basic_usage():
    """Demo basico del sistema."""
    print("=" * 70)
    print("DEMO 1: USO BASICO DEL SISTEMA")
    print("=" * 70)

    engine = CareerIntegratedEngine(seed=42)

    agent = 'DEMO_AGENT'

    print("\n1. Agente solicita siguiente investigacion:")
    request = engine.request_next_research(agent)
    print(f"   Dominio seleccionado: {request['domain']}")
    print(f"   Nivel actual: {request['academic_level']}")
    print(f"   Tipo de tarea: {request['task_type']}")

    print("\n2. Sistema genera la tarea:")
    task = engine.generate_task(request)
    print(f"   Task ID: {task.task_id}")
    print(f"   Tiene ground truth: {task.has_ground_truth}")
    print(f"   Modo evaluacion: {task.evaluation_mode.value}")

    print("\n3. Agente resuelve y sistema registra:")
    # Simular solucion (usando oracle para demo)
    result = engine.submit_result(
        agent_id=agent,
        task=task,
        solution=task.oracle_solution
    )
    print(f"   Performance: {result['performance']:.3f}")
    print(f"   Exito: {result['succeeded']}")
    print(f"   Estado academico: {result['academic_status']}")

    print("\n4. Reporte academico:")
    report = engine.get_academic_report(agent)
    print(f"   Total tareas: {report['summary']['total_tasks']}")
    print(f"   Etiqueta emergente: {report['summary']['emergent_label']}")


def demo_specialization_emergence():
    """Demo de como emerge la especializacion."""
    print("\n" + "=" * 70)
    print("DEMO 2: EMERGENCIA DE ESPECIALIZACION")
    print("=" * 70)

    director = ResearchDirector(seed=123)

    # Tres agentes con diferentes "talentos naturales"
    # (simulados por diferentes niveles de ruido en sus soluciones)
    agents = ['LEIBNIZ', 'FEYNMAN', 'DARWIN']

    # Sesgos simulan "afinidad natural"
    # LEIBNIZ: mejor en matematicas
    # FEYNMAN: mejor en fisica
    # DARWIN: generalista
    biases = {
        'LEIBNIZ': {'mathematics': 0.1, 'physics': 0.3, 'medicine': 0.4},
        'FEYNMAN': {'mathematics': 0.3, 'physics': 0.1, 'medicine': 0.4},
        'DARWIN': {'mathematics': 0.25, 'physics': 0.25, 'medicine': 0.2},
    }

    def talent_solver(agent_id, task):
        """Solver que simula diferentes talentos."""
        if task.oracle_solution is None:
            return None

        # Ruido base
        noise = biases.get(agent_id, {}).get(task.domain, 0.3)

        sol = task.oracle_solution
        if isinstance(sol, dict):
            return {k: v * (1 + np.random.randn() * noise)
                    if isinstance(v, (int, float)) else v
                    for k, v in sol.items()}
        elif isinstance(sol, np.ndarray):
            return sol * (1 + np.random.randn() * noise)
        return sol

    print("\nIniciando sesion de investigacion...")
    director.start_session(agents)

    print("\nEjecutando 50 rondas de investigacion...")
    for round_num in range(50):
        results = director.run_research_round(solver_fn=talent_solver)

        # Reportar cada 10 rondas
        if (round_num + 1) % 10 == 0:
            print(f"\n--- Ronda {round_num + 1} ---")
            for r in results:
                status = r['academic_status']
                promo = " -> PROMOCION!" if status.get('promoted') else ""
                print(f"  {r['agent_id']}: {r['domain']}/{status['level']} "
                      f"perf={r['performance']:.2f}{promo}")

    print("\n" + "=" * 50)
    print("RESULTADOS FINALES")
    print("=" * 50)

    report = director.get_session_report()

    print(f"\nRondas totales: {report['n_rounds']}")

    print("\nETIQUETAS EMERGENTES:")
    for agent, label in report['labels'].items():
        print(f"  {agent}: {label}")

    print("\nRANKING DE ESPECIALIZACION (z-score):")
    for agent, z in report['specialization_ranking']:
        print(f"  {agent}: z = {z:.3f}")

    print("\nNIVELES ACADEMICOS POR DOMINIO:")
    for agent, ar in report['agent_reports'].items():
        career = ar.get('career', {})
        promo = career.get('promotion_status', {})
        levels = [f"{d}:{info['current_level']}" for d, info in promo.items()]
        print(f"  {agent}: {', '.join(levels)}")


def demo_curriculum_levels():
    """Demo de la estructura de niveles."""
    print("\n" + "=" * 70)
    print("DEMO 3: ESTRUCTURA DE CURRICULOS")
    print("=" * 70)

    from domains.specialization.academic_career import DomainCurriculum, AcademicLevel

    print("\n=== CURRICULO DE MATEMATICAS ===")
    math_curr = DomainCurriculum('mathematics')
    for level in [AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE, AcademicLevel.DOCTORAL]:
        tasks = math_curr.get_tasks_for_level(level)
        if tasks:
            print(f"\n{level.value.upper()}:")
            for t in tasks:
                print(f"  - {t.task_type}: {t.description}")
                params = t.difficulty_params
                if params:
                    print(f"    Params: {params}")

    print("\n=== CURRICULO DE FISICA ===")
    phys_curr = DomainCurriculum('physics')
    for level in [AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE, AcademicLevel.DOCTORAL]:
        tasks = phys_curr.get_tasks_for_level(level)
        if tasks:
            print(f"\n{level.value.upper()}:")
            for t in tasks:
                print(f"  - {t.task_type}: {t.description}")


def demo_promotion_criteria():
    """Demo del criterio de promocion endogeno."""
    print("\n" + "=" * 70)
    print("DEMO 4: CRITERIO DE PROMOCION ENDOGENO")
    print("=" * 70)

    from domains.specialization.academic_career import AcademicCareerEngine

    engine = AcademicCareerEngine(seed=456)

    agent = 'PROMOTION_DEMO'
    domain = 'mathematics'

    print("\nSimulando agente que MEJORA con el tiempo...")
    print("(Promocion requiere estar en percentil 80 del PROPIO historial)")

    print("\nFase 1: Rendimiento bajo consistente")
    for i in range(10):
        perf = 0.3 + np.random.uniform(-0.05, 0.05)
        result = engine.record_task_result(
            agent_id=agent,
            domain=domain,
            performance=perf,
            succeeded=False
        )
        if i % 3 == 0:
            print(f"  Tarea {i+1}: perf={perf:.3f}")

    can_promote, _, _ = engine.check_promotion(agent, domain)
    print(f"\n  Puede promocionar? {can_promote}")
    print("  (No, porque rendimiento reciente NO supera su propio percentil 80)")

    print("\nFase 2: Rendimiento MEJORA significativamente")
    for i in range(10):
        perf = 0.8 + np.random.uniform(-0.05, 0.05)
        result = engine.record_task_result(
            agent_id=agent,
            domain=domain,
            performance=perf,
            succeeded=True
        )
        if i % 3 == 0:
            print(f"  Tarea {i+11}: perf={perf:.3f}")

    can_promote, new_level, prov = engine.check_promotion(agent, domain)
    print(f"\n  Puede promocionar? {can_promote}")
    if can_promote:
        print(f"  Nuevo nivel: {new_level.value}")

    print("\nNOTA: El criterio NO es 'accuracy > 0.8'")
    print("      Es 'rendimiento reciente en percentil 80 de MI historial'")


def demo_agent_decides_research():
    """Demo de como el agente 'decide' que investigar."""
    print("\n" + "=" * 70)
    print("DEMO 5: AGENTE 'DECIDE' QUE INVESTIGAR")
    print("=" * 70)

    engine = CareerIntegratedEngine(seed=789)

    agent = 'AUTONOMOUS_RESEARCHER'

    print("\n1. Sin historial: seleccion casi uniforme")
    weights = engine.get_exploration_weights(agent)
    print(f"   Pesos de exploracion: {weights}")

    print("\n2. Despues de explorar matematicas...")
    for _ in range(5):
        request = engine.request_next_research(agent)
        request['domain'] = 'mathematics'  # Forzar para demo
        task = engine.generate_task(request)
        engine.submit_result(agent, task, task.oracle_solution)

    weights = engine.get_exploration_weights(agent)
    print(f"   Nuevos pesos: {weights}")

    print("\n3. Agente solicita siguiente investigacion:")
    for i in range(5):
        request = engine.request_next_research(agent)
        print(f"   Solicitud {i+1}: {request['domain']} / {request['task_type']}")

    print("\nNOTA: El dominio se selecciona por softmax sobre afinidades")
    print("      No hay 'if agent_name == GAUSS: domain = math'")


def main():
    """Ejecuta todos los demos."""
    print("\n" + "#" * 70)
    print("#  SISTEMA DE CARRERAS ACADEMICAS - DEMOS COMPLETOS")
    print("#" * 70)

    demo_basic_usage()
    demo_specialization_emergence()
    demo_curriculum_levels()
    demo_promotion_criteria()
    demo_agent_decides_research()

    print("\n" + "#" * 70)
    print("#  RESUMEN DE PRINCIPIOS NORMA DURA")
    print("#" * 70)

    print("""
PRINCIPIOS CUMPLIDOS:

1. SIN NUMEROS MAGICOS:
   - Promocion por percentil 80 (derivado de teoria normal)
   - No hay "if accuracy > 0.8"

2. UMBRALES ENDOGENOS:
   - Cada agente se compara consigo mismo
   - Percentil calculado sobre su propio historial

3. ETIQUETAS EMERGENTES:
   - No existe is_physicist = True
   - Etiquetas derivadas de: nivel academico * afinidad
   - Solo para logging/analisis humano

4. SELECCION DE INVESTIGACION:
   - El agente "decide" basado en softmax de afinidades
   - No hay if/else por nombre de agente

5. CURRICULO ESTRUCTURAL:
   - Niveles definidos por complejidad estructural
   - No por umbrales de rendimiento
   - UNDERGRADUATE: sistemas simples, ground truth
   - GRADUATE: complejidad media, ruido
   - DOCTORAL: sin ground truth, falsificacion

USO:
   from domains.specialization.career_integrated_engine import ResearchDirector

   director = ResearchDirector()
   director.start_session(['GAUSS', 'NEWTON', 'EULER'])

   for _ in range(100):
       director.run_research_round()

   report = director.get_session_report()
   print(report['labels'])  # Etiquetas emergentes
""")


if __name__ == "__main__":
    main()
