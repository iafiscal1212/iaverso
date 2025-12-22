#!/usr/bin/env python3
"""
Test del sistema IAVERSO consolidado.
"""
import sys
sys.path.insert(0, '/opt/iaverso')

from core.endolens import get_endolens
from core.neosynt import get_neosynt
from core.language import detect_language
from lab.genetic_lab import get_genetic_lab
from agents.gamma import get_gamma

def test_endolens():
    print("=== TEST ENDOLENS ===")
    endolens = get_endolens()
    
    state = endolens.process("Esta es una prueba de estructura")
    print(f"Firma: {state.signature}")
    print(f"E-series: {state.eseries.as_dict()}")
    print(f"Estabilidad: {state.stability:.3f}")
    print(f"Status: {state.status}")
    print(f"Invariantes: {state.invariants}")
    print(f"Tensiones: {state.tensions}")
    print()
    return state

def test_neosynt(state):
    print("=== TEST NEOSYNT ===")
    neosynt = get_neosynt()
    
    resolution = neosynt.resolve(state, target='stable')
    print(f"Status: {resolution.status}")
    print(f"Estabilidad final: {resolution.stability_score:.3f}")
    print(f"Operadores: {resolution.operators_applied}")
    print(f"Alternativas: {len(resolution.alternatives)}")
    print()
    return resolution

def test_language():
    print("=== TEST LANGUAGE ===")
    tests = [
        "Hola esto es español",
        "Hello this is english",
        "Olá isto é português",
        "Bonjour ceci est français",
        "こんにちは",
        "Привет",
    ]
    for text in tests:
        result = detect_language(text)
        print(f"\"{text[:30]}...\" -> {result.language} ({result.confidence:.0%}, {result.layer})")
    print()

def test_lab():
    print("=== TEST LABORATORIO GENÉTICO ===")
    lab = get_genetic_lab()
    
    # Crear cultivo
    culture = lab.create_culture(
        seed="ATCGATCGATCG",
        population_size=10,
        conditions={
            'mutation_rate': 0.2,
            'selection': 'fitness',
            'target': 'stability',
            'temperature': 1.0
        }
    )
    print(f"Cultivo creado: {culture.id}")
    print(f"Tamaño: {culture.size}")
    print(f"Fitness promedio: {culture.avg_fitness:.3f}")
    print(f"Mejor espécimen: {culture.best_specimen.id} (fitness={culture.best_specimen.fitness:.3f})")
    print()
    
    # Evolucionar
    print("Evolucionando 5 generaciones...")
    lab.evolve_culture(culture.id, generations=5)
    print(f"Nueva generación: {culture.generation}")
    print(f"Fitness promedio: {culture.avg_fitness:.3f}")
    print(f"Mejor espécimen: {culture.best_specimen.id} (fitness={culture.best_specimen.fitness:.3f})")
    print()
    
    # Experimento
    print("Proponiendo experimento...")
    experiment = lab.propose_experiment(culture)
    print(f"Experimento: {experiment.action}")
    print(f"Incertidumbre: {experiment.uncertainty:.3f}")
    
    result = lab.execute_experiment(experiment)
    print(f"Sorpresa: {result.surprise:.4f}")
    print(f"Éxito: {result.success}")
    print()
    
    return culture

def test_gamma(state, resolution):
    print("=== TEST GAMMA (NARRADOR) ===")
    gamma = get_gamma()
    
    narration = gamma.narrate(
        query="prueba de investigación estructural",
        state=state,
        resolution=resolution
    )
    
    print(gamma.format_full_narration(narration))
    print()

def main():
    print("\n" + "="*60)
    print("IAVERSO - Sistema Consolidado de Inferencia Activa")
    print("="*60 + "\n")
    
    # Tests
    state = test_endolens()
    resolution = test_neosynt(state)
    test_language()
    test_lab()
    test_gamma(state, resolution)
    
    print("="*60)
    print("TODOS LOS TESTS PASARON")
    print("="*60)

if __name__ == '__main__':
    main()
