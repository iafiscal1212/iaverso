#!/usr/bin/env python3
"""
TEST: Verificación de Unicidad de Métricas por Agente
=====================================================

Verifica que:
1. Ninguna métrica es idéntica entre agentes
2. Correlación(serie_i, serie_j) < 0.9999 para i ≠ j
3. Cada agente tiene variabilidad endógena propia

100% Endógeno - Sin números mágicos externos.
"""

import numpy as np
import sys
import pytest

sys.path.insert(0, '/root/NEO_EVA')

from core.agents import NEO, EVA, BaseAgent


class TestMetricUniqueness:
    """Tests para verificar que métricas son únicas por agente."""

    def setup_method(self):
        """Reset agent counter para tests independientes."""
        BaseAgent._agent_counter = 0

    def test_initial_state_different(self):
        """Verifica que estados iniciales son diferentes entre agentes."""
        agents = [NEO(dim_visible=6, dim_hidden=6) for _ in range(5)]

        # Extraer z_visible de cada agente
        z_visibles = [a.z_visible for a in agents]

        # Verificar que NO son idénticos
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                is_identical = np.array_equal(z_visibles[i], z_visibles[j])
                assert not is_identical, f"Agentes {i} y {j} tienen z_visible idéntico"

    def test_rng_independent(self):
        """Verifica que cada agente tiene su propio RNG."""
        agents = [NEO(dim_visible=6, dim_hidden=6) for _ in range(3)]

        # Generar números aleatorios de cada RNG
        random_numbers = [a._rng.uniform(0, 1, 10) for a in agents]

        # Verificar que NO son idénticos
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                is_identical = np.array_equal(random_numbers[i], random_numbers[j])
                assert not is_identical, f"Agentes {i} y {j} tienen RNG idéntico"

    def test_entropy_different_after_steps(self):
        """Verifica que entropía (S) es diferente entre agentes después de steps."""
        agents = [NEO(dim_visible=6, dim_hidden=6) for _ in range(5)]

        # Simular 100 pasos con estímulo compartido
        rng = np.random.default_rng(42)
        for _ in range(100):
            stimulus = rng.uniform(0, 1, 6)
            for agent in agents:
                agent.step(stimulus)

        # Extraer entropía final
        entropies = [a.get_state().S for a in agents]

        # Verificar que NO todas son idénticas
        unique_entropies = len(set([round(e, 10) for e in entropies]))
        assert unique_entropies > 1, "Todas las entropías son idénticas - artefacto detectado"

    def test_psi_norm_different(self):
        """Verifica que psi_norm es diferente entre agentes."""
        agents = [NEO(dim_visible=6, dim_hidden=6) for _ in range(5)]

        rng = np.random.default_rng(42)
        for _ in range(100):
            stimulus = rng.uniform(0, 1, 6)
            for agent in agents:
                agent.step(stimulus)

        # Calcular psi_norm
        psi_norms = [np.linalg.norm(a.get_state().z_visible) for a in agents]

        # Verificar diferencias
        unique_norms = len(set([round(p, 10) for p in psi_norms]))
        assert unique_norms > 1, "Todos los psi_norm son idénticos - artefacto detectado"

    def test_correlation_below_threshold(self):
        """Verifica que correlación entre series es < 0.9999."""
        agents = [NEO(dim_visible=6, dim_hidden=6) for _ in range(3)]

        # Simular y recolectar series
        rng = np.random.default_rng(42)
        series = {i: [] for i in range(len(agents))}

        for _ in range(200):
            stimulus = rng.uniform(0, 1, 6)
            for i, agent in enumerate(agents):
                response = agent.step(stimulus)
                series[i].append(response.surprise)

        # Verificar correlaciones
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                corr = np.corrcoef(series[i], series[j])[0, 1]
                assert corr < 0.9999, f"Correlación {i}-{j} = {corr} >= 0.9999 - artefacto"

    def test_neo_eva_different(self):
        """Verifica que NEO y EVA producen series diferentes."""
        neo = NEO(dim_visible=6, dim_hidden=6)
        eva = EVA(dim_visible=6, dim_hidden=6)

        rng = np.random.default_rng(42)
        neo_values = []
        eva_values = []

        for _ in range(200):
            stimulus = rng.uniform(0, 1, 6)
            neo_resp = neo.step(stimulus)
            eva_resp = eva.step(stimulus)
            neo_values.append(neo_resp.value)
            eva_values.append(eva_resp.value)

        # Correlación debe ser significativamente < 1
        corr = np.corrcoef(neo_values, eva_values)[0, 1]
        assert corr < 0.95, f"NEO-EVA correlación = {corr} - demasiado alta"

    def test_same_type_different_instances(self):
        """Verifica que múltiples instancias del mismo tipo son diferentes."""
        neos = [NEO(dim_visible=6, dim_hidden=6) for _ in range(3)]

        rng = np.random.default_rng(42)
        series = {i: [] for i in range(len(neos))}

        for _ in range(200):
            stimulus = rng.uniform(0, 1, 6)
            for i, neo in enumerate(neos):
                response = neo.step(stimulus)
                series[i].append(response.value)

        # Verificar que no son idénticas
        for i in range(len(neos)):
            for j in range(i + 1, len(neos)):
                max_diff = np.max(np.abs(np.array(series[i]) - np.array(series[j])))
                assert max_diff > 0.001, f"NEO {i} y {j} tienen series casi idénticas"

    def test_variability_positive(self):
        """Verifica que todas las métricas tienen std > 0."""
        agent = NEO(dim_visible=6, dim_hidden=6)

        rng = np.random.default_rng(42)
        values = []
        surprises = []
        entropies = []

        for _ in range(200):
            stimulus = rng.uniform(0, 1, 6)
            response = agent.step(stimulus)
            values.append(response.value)
            surprises.append(response.surprise)
            entropies.append(agent.get_state().S)

        assert np.std(values) > 0, "Value sin variabilidad"
        assert np.std(surprises) > 0, "Surprise sin variabilidad"
        assert np.std(entropies) > 0, "Entropy sin variabilidad"


class TestNoSharedArrays:
    """Tests para verificar que no hay arrays compartidos por referencia."""

    def setup_method(self):
        """Reset agent counter."""
        BaseAgent._agent_counter = 0

    def test_z_visible_not_shared(self):
        """Verifica que z_visible no es compartido entre agentes."""
        agent1 = NEO(dim_visible=6, dim_hidden=6)
        agent2 = NEO(dim_visible=6, dim_hidden=6)

        # Verificar que no son el mismo objeto
        assert agent1.z_visible is not agent2.z_visible, "z_visible compartido por referencia"

        # Modificar uno no debe afectar al otro
        original_agent2 = agent2.z_visible.copy()
        agent1.z_visible[0] = 999.0

        assert np.array_equal(agent2.z_visible, original_agent2), "Modificación afectó otro agente"

    def test_z_hidden_not_shared(self):
        """Verifica que z_hidden no es compartido entre agentes."""
        agent1 = NEO(dim_visible=6, dim_hidden=6)
        agent2 = NEO(dim_visible=6, dim_hidden=6)

        assert agent1.z_hidden is not agent2.z_hidden, "z_hidden compartido por referencia"

    def test_history_not_shared(self):
        """Verifica que historiales no son compartidos."""
        agent1 = NEO(dim_visible=6, dim_hidden=6)
        agent2 = NEO(dim_visible=6, dim_hidden=6)

        assert agent1.z_history is not agent2.z_history, "z_history compartido"
        assert agent1.S_history is not agent2.S_history, "S_history compartido"
        assert agent1.surprise_history is not agent2.surprise_history, "surprise_history compartido"

    def test_state_copy_independent(self):
        """Verifica que get_state() retorna copia independiente."""
        agent = NEO(dim_visible=6, dim_hidden=6)

        state1 = agent.get_state()
        state2 = agent.get_state()

        # Modificar state1 no debe afectar state2
        state1.z_visible[0] = 999.0

        assert state1.z_visible is not state2.z_visible, "Estados comparten z_visible"
        assert state2.z_visible[0] != 999.0, "Modificación de state1 afectó state2"


class TestEndogenousOnly:
    """Tests para verificar que todo es endógeno."""

    def setup_method(self):
        """Reset agent counter."""
        BaseAgent._agent_counter = 0

    def test_no_external_constants(self):
        """Verifica que no hay constantes mágicas externas en métricas."""
        agent = NEO(dim_visible=6, dim_hidden=6)

        # Las únicas constantes permitidas son fracciones simples y eps de máquina
        # Verificar que learning_rate es endógeno
        assert agent.learning_rate == 0.1, "learning_rate inicial no es 0.1"

        agent.step(np.ones(6) * 0.5)
        # learning_rate = 1/√(t+1) = 1/√2 ≈ 0.707
        expected_lr = 1.0 / np.sqrt(2)
        assert abs(agent.learning_rate - expected_lr) < 0.001, "learning_rate no es endógeno"

    def test_perturbation_scale_endogenous(self):
        """Verifica que la escala de perturbación es endógena (1/√dim)."""
        dim = 6
        expected_scale = 1.0 / np.sqrt(dim)

        # La perturbación debe estar en rango [-scale, scale]
        agents = [NEO(dim_visible=dim, dim_hidden=dim) for _ in range(100)]

        # Verificar que las perturbaciones están en el rango esperado
        base = np.ones(dim) / dim
        deviations = [np.max(np.abs(a.z_visible - base / base.sum())) for a in agents]
        max_deviation = max(deviations)

        # La desviación máxima debe ser aproximadamente la escala
        assert max_deviation < expected_scale * 2, "Perturbación fuera de rango endógeno"


def run_all_tests():
    """Ejecuta todos los tests manualmente."""
    print("=" * 70)
    print("TEST: VERIFICACIÓN DE UNICIDAD DE MÉTRICAS")
    print("=" * 70)

    # Reset counter
    BaseAgent._agent_counter = 0

    tests_passed = 0
    tests_failed = 0

    test_classes = [TestMetricUniqueness, TestNoSharedArrays, TestEndogenousOnly]

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    if hasattr(instance, 'setup_method'):
                        instance.setup_method()
                    getattr(instance, method_name)()
                    print(f"  ✓ {method_name}")
                    tests_passed += 1
                except AssertionError as e:
                    print(f"  ✗ {method_name}: {e}")
                    tests_failed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: Error - {e}")
                    tests_failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTADOS: {tests_passed} passed, {tests_failed} failed")
    print("=" * 70)

    return tests_failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
