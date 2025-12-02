#!/usr/bin/env python3
"""
TEST POST-FIX INDEPENDENCE
===========================

Tests automatizados para verificar que la corrección de artefactos
es permanente y reproducible.

Ejecutar con: pytest tests/test_post_fix_independence.py -v
"""

import numpy as np
import sys
import pytest

sys.path.insert(0, '/root/NEO_EVA')

from core.agents import NEO, EVA, BaseAgent
from omega.q_field import QField


class TestPostFixIndependence:
    """Tests de independencia post-fix."""

    def setup_method(self):
        """Reset para cada test."""
        BaseAgent._agent_counter = 0

    def test_series_not_identical_after_steps(self):
        """Verifica que series no son idénticas después de múltiples pasos."""
        agents = [NEO(dim_visible=6, dim_hidden=6) for _ in range(5)]

        rng = np.random.default_rng(42)
        series = {i: [] for i in range(len(agents))}

        for _ in range(200):
            stimulus = rng.uniform(0, 1, 6)
            for i, agent in enumerate(agents):
                response = agent.step(stimulus)
                series[i].append(np.linalg.norm(agent.z_visible))

        # Verificar que ningún par es idéntico
        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                is_identical = np.array_equal(series[i], series[j])
                assert not is_identical, f"Series {i} y {j} son idénticas - ARTEFACTO"

    def test_correlation_below_threshold(self):
        """Verifica correlación < 0.9999 entre agentes."""
        agents = [NEO(dim_visible=6, dim_hidden=6) for _ in range(3)]

        rng = np.random.default_rng(42)
        series = {i: [] for i in range(len(agents))}

        for _ in range(300):
            stimulus = rng.uniform(0, 1, 6)
            for i, agent in enumerate(agents):
                response = agent.step(stimulus)
                series[i].append(response.surprise)

        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                corr = np.corrcoef(series[i], series[j])[0, 1]
                assert corr < 0.9999, f"Correlación {i}-{j} = {corr:.4f} >= 0.9999"

    def test_entropy_different_between_agents(self):
        """Verifica que entropía difiere entre agentes."""
        agents = [NEO(dim_visible=6, dim_hidden=6) for _ in range(5)]

        rng = np.random.default_rng(42)
        for _ in range(100):
            stimulus = rng.uniform(0, 1, 6)
            for agent in agents:
                agent.step(stimulus)

        entropies = [a.get_state().S for a in agents]
        unique = len(set([round(e, 10) for e in entropies]))
        assert unique > 1, "Todas las entropías son idénticas"

    def test_q_coherence_per_agent(self):
        """Verifica que Q-Field retorna coherencia por agente."""
        q_field = QField()
        agents = [NEO(dim_visible=6, dim_hidden=6) for _ in range(3)]
        agent_names = ['A0', 'A1', 'A2']

        rng = np.random.default_rng(42)
        for _ in range(50):
            stimulus = rng.uniform(0, 1, 6)
            for name, agent in zip(agent_names, agents):
                agent.step(stimulus)
                state = agent.get_state()
                q_field.register_state(name, state.z_visible)

        stats = q_field.get_statistics()

        # Verificar que existen claves por agente
        for name in agent_names:
            assert f'{name}_coherence' in stats, f"Falta {name}_coherence en stats"

        # Verificar que no son todos iguales
        coherences = [stats[f'{name}_coherence'] for name in agent_names]
        unique = len(set([round(c, 10) for c in coherences]))
        assert unique > 1, "Todas las coherencias son iguales"

    def test_initial_state_uniqueness(self):
        """Verifica unicidad de estados iniciales."""
        agents = [NEO(dim_visible=6, dim_hidden=6) for _ in range(10)]

        states = [a.z_visible.copy() for a in agents]

        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                is_identical = np.array_equal(states[i], states[j])
                assert not is_identical, f"Estados iniciales {i} y {j} idénticos"

    def test_rng_produces_different_sequences(self):
        """Verifica que RNG individual produce secuencias diferentes."""
        agents = [NEO(dim_visible=6, dim_hidden=6) for _ in range(5)]

        sequences = [a._rng.uniform(0, 1, 20) for a in agents]

        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                is_identical = np.array_equal(sequences[i], sequences[j])
                assert not is_identical, f"RNG {i} y {j} producen secuencias idénticas"

    def test_neo_eva_different_dynamics(self):
        """Verifica que NEO y EVA tienen dinámicas diferentes."""
        neo = NEO(dim_visible=6, dim_hidden=6)
        eva = EVA(dim_visible=6, dim_hidden=6)

        rng = np.random.default_rng(42)
        neo_vals = []
        eva_vals = []

        for _ in range(200):
            stimulus = rng.uniform(0, 1, 6)
            neo_resp = neo.step(stimulus)
            eva_resp = eva.step(stimulus)
            neo_vals.append(neo_resp.value)
            eva_vals.append(eva_resp.value)

        corr = np.corrcoef(neo_vals, eva_vals)[0, 1]
        assert corr < 0.95, f"NEO-EVA correlación = {corr:.4f} demasiado alta"

    def test_multiple_seeds_produce_different_results(self):
        """Verifica que diferentes seeds producen resultados diferentes."""
        results = []

        for seed in [1, 2, 3, 4, 5]:
            BaseAgent._agent_counter = 0
            agents = [NEO(dim_visible=6, dim_hidden=6) for _ in range(3)]

            rng = np.random.default_rng(seed)
            series = []

            for _ in range(100):
                stimulus = rng.uniform(0, 1, 6)
                for agent in agents:
                    agent.step(stimulus)

            final_state = agents[0].z_visible.copy()
            results.append(final_state)

        # Verificar que los resultados son diferentes
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                is_identical = np.array_equal(results[i], results[j])
                assert not is_identical, f"Seeds {i+1} y {j+1} producen resultados idénticos"


class TestNoRegressions:
    """Tests para verificar que no hay regresiones."""

    def setup_method(self):
        BaseAgent._agent_counter = 0

    def test_agent_step_still_works(self):
        """Verifica que step() funciona correctamente."""
        agent = NEO(dim_visible=6, dim_hidden=6)

        response = agent.step(np.ones(6) * 0.5)

        assert hasattr(response, 'surprise')
        assert hasattr(response, 'value')
        assert not np.isnan(response.surprise)
        assert not np.isnan(response.value)

    def test_get_state_returns_valid_state(self):
        """Verifica que get_state() retorna estado válido."""
        agent = NEO(dim_visible=6, dim_hidden=6)
        agent.step(np.ones(6) * 0.5)

        state = agent.get_state()

        assert hasattr(state, 'z_visible')
        assert hasattr(state, 'S')
        assert len(state.z_visible) == 6
        assert not np.isnan(state.S)

    def test_qfield_registers_state(self):
        """Verifica que Q-Field registra estados."""
        q_field = QField()
        agent = NEO(dim_visible=6, dim_hidden=6)

        state = agent.get_state()
        q_state = q_field.register_state('test', state.z_visible)

        assert q_state is not None
        assert hasattr(q_state, 'coherence')
        assert 0 <= q_state.coherence <= 1


def run_all_post_fix_tests():
    """Ejecuta todos los tests manualmente."""
    print("=" * 70)
    print("TESTS POST-FIX INDEPENDENCE")
    print("=" * 70)

    BaseAgent._agent_counter = 0
    tests_passed = 0
    tests_failed = 0

    test_classes = [TestPostFixIndependence, TestNoRegressions]

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
    success = run_all_post_fix_tests()
    sys.exit(0 if success else 1)
