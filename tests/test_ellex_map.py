"""
Test ELLEX-MAP: Existential Life Layer Explorer
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np


def test_layer_emergence():
    """Test base layer classes."""
    print("\n=== Test Layer Emergence ===")
    from ellex_map.layer_emergence import LayerHistory, LayerType

    history = LayerHistory()

    # Add some values
    for i in range(20):
        history.add(0.5 + 0.1 * np.sin(i * 0.5), t=i)

    print(f"  History length: {len(history.values)}")
    print(f"  Variance: {history.get_variance():.3f}")
    print(f"  Trend: {history.get_trend():.3f}")
    print(f"  Stability: {history.get_stability():.3f}")

    assert len(history.values) == 20
    assert 0 <= history.get_stability() <= 1
    print("  [PASS] Layer emergence works")


def test_coherence_surface():
    """Test coherence layers."""
    print("\n=== Test Coherence Surface ===")
    from ellex_map.coherence_surface import (
        CognitiveCoherence,
        SymbolicCoherence,
        NarrativeCoherence,
        LifeCoherence,
        SocialCoherence
    )

    agent_id = "test_agent"

    # Test L1
    l1 = CognitiveCoherence(agent_id)
    for t in range(10):
        obs = {
            'cognitive_load': 0.5 + 0.1 * np.random.randn(),
            'attention_focus': 0.6 + 0.1 * np.random.randn(),
            'memory_coherence': 0.7 + 0.1 * np.random.randn()
        }
        state = l1.update(obs)
    print(f"  L1 Cognitive: {state.value:.3f}")

    # Test L2
    l2 = SymbolicCoherence(agent_id)
    for t in range(10):
        obs = {
            'active_concepts': {'love': 0.8, 'fear': 0.3, 'hope': 0.6},
            'concept_stability': 0.7,
            'connection_density': 0.5
        }
        state = l2.update(obs)
    print(f"  L2 Symbolic: {state.value:.3f}")

    # Test L3
    l3 = NarrativeCoherence(agent_id)
    for t in range(10):
        obs = {
            'arc_completeness': 0.6,
            'temporal_flow': 0.7,
            'episode_resonance': 0.5
        }
        state = l3.update(obs)
    print(f"  L3 Narrative: {state.value:.3f}")

    # Test L4
    l4 = LifeCoherence(agent_id)
    for t in range(10):
        obs = {
            'drives': {'survival': 0.7, 'curiosity': 0.8, 'connection': 0.6},
            'goals': {'explore': 0.7, 'learn': 0.8},
            'environment_fit': 0.6
        }
        state = l4.update(obs)
    print(f"  L4 Life: {state.value:.3f}")

    # Test L6
    l6 = SocialCoherence(agent_id)
    for t in range(10):
        obs = {
            'social_connections': ['agent_1', 'agent_2', 'user'],
            'interaction_quality': 0.7,
            'trust_levels': {'agent_1': 0.8, 'agent_2': 0.6}
        }
        state = l6.update(obs)
    print(f"  L6 Social: {state.value:.3f}")

    print("  [PASS] Coherence surfaces work")


def test_existential_tension():
    """Test tension layer."""
    print("\n=== Test Existential Tension ===")
    from ellex_map.existential_tension import ExistentialTension

    l7 = ExistentialTension("test_agent")

    zones_seen = set()
    for t in range(30):
        stress = 0.3 + 0.4 * np.sin(t * 0.3)
        obs = {
            'drives': [0.5, 0.6, 0.7, 0.4],
            'goals': [0.6, 0.5, 0.8],
            'stress': stress,
            'transitions': ['wake', 'liminal', 'rest'][t % 3:t % 3 + 1]
        }
        layer_state = l7.update(obs)
        tension_state = l7.get_tension_state()
        zones_seen.add(tension_state.zone)

    print(f"  L7 Tension: {layer_state.value:.3f}")
    print(f"  Zone: {tension_state.zone}")
    print(f"  Zones seen: {zones_seen}")
    print(f"  Drive variance: {tension_state.drive_variance:.4f}")
    print(f"  Stress: {tension_state.stress:.3f}")

    assert 0 <= layer_state.value <= 1
    print("  [PASS] Existential tension works")


def test_health_equilibrium():
    """Test health layer."""
    print("\n=== Test Health Equilibrium ===")
    from ellex_map.health_equilibrium import HealthEquilibrium

    l5 = HealthEquilibrium("test_agent")

    for t in range(20):
        obs = {
            'diagnosis_quality': 0.7 + 0.1 * np.random.randn(),
            'treatment_efficacy': 0.6 + 0.1 * np.random.randn(),
            'iatrogenesis_rate': 0.1 * np.random.rand(),
            'rotation_health': 0.8,
            'health_history': [0.6, 0.65, 0.7, 0.68, 0.72],
            'stress_events': [0.2, 0.3, 0.6, 0.2, 0.1]
        }
        layer_state = l5.update(obs)

    health_state = l5.get_health_state()
    print(f"  L5 Health: {layer_state.value:.3f}")
    print(f"  Is healthy: {health_state.is_healthy}")
    print(f"  Resilience: {health_state.resilience:.3f}")

    assert 0 <= layer_state.value <= 1
    print("  [PASS] Health equilibrium works")


def test_circadian_phase_space():
    """Test phase layer."""
    print("\n=== Test Circadian Phase Space ===")
    from ellex_map.circadian_phase_space import CircadianPhaseSpace

    l9 = CircadianPhaseSpace("test_agent")

    phases = ['wake', 'wake', 'wake', 'liminal', 'rest', 'rest',
              'dream', 'dream', 'liminal', 'wake']

    for i, phase in enumerate(phases * 3):
        obs = {
            'current_phase': phase,
            'phase_efficacy': 0.6 + 0.2 * np.random.rand(),
            'multiagent_sync': 0.7
        }
        layer_state = l9.update(obs)

    eq = l9.get_phase_equilibrium()
    print(f"  L9 Phase Equilibrium: {layer_state.value:.3f}")
    print(f"  Wake proportion: {eq.wake_proportion:.3f}")
    print(f"  Rest proportion: {eq.rest_proportion:.3f}")
    print(f"  Dream proportion: {eq.dream_proportion:.3f}")
    print(f"  Transition smoothness: {eq.transition_smoothness:.3f}")

    assert 0 <= layer_state.value <= 1
    print("  [PASS] Circadian phase space works")


def test_ellex_index():
    """Test ELLEX index."""
    print("\n=== Test ELLEX Index ===")
    from ellex_map.ellex_index import ELLEXIndex

    l10 = ELLEXIndex("test_agent")

    for t in range(30):
        obs = {
            'L1_cognitive': 0.6 + 0.1 * np.random.randn(),
            'L2_symbolic': 0.7 + 0.1 * np.random.randn(),
            'L3_narrative': 0.5 + 0.1 * np.random.randn(),
            'L4_life': 0.65 + 0.1 * np.random.randn(),
            'L5_health': 0.7 + 0.1 * np.random.randn(),
            'L6_social': 0.6 + 0.1 * np.random.randn(),
            'L7_tension': 0.5 + 0.15 * np.sin(t * 0.3),  # Oscillating tension
            'L8_identity': 0.75 + 0.05 * np.random.randn(),
            'L9_phase': 0.6 + 0.1 * np.random.randn()
        }
        layer_state = l10.update(obs)

    ellex_state = l10.get_ellex_state()
    print(f"  ELLEX Index: {ellex_state.ellex:.3f}")
    print(f"  Zone: {ellex_state.zone}")
    print(f"  Stability: {ellex_state.stability:.3f}")
    print(f"  Trend: {ellex_state.trend:.3f}")

    # Show layer weights
    print("  Layer weights:")
    for name, weight in sorted(ellex_state.layer_weights.items()):
        print(f"    {name}: {weight:.3f}")

    # Weakest layers
    weakest = l10.get_weakest_layers(3)
    print("  Weakest layers:")
    for name, val in weakest:
        print(f"    {name}: {val:.3f}")

    assert 0 <= ellex_state.ellex <= 1
    print("  [PASS] ELLEX index works")


def test_ellex_map():
    """Test full ELLEX map."""
    print("\n=== Test Full ELLEX Map ===")
    from ellex_map.ellex_map import ELLEXMap

    ellex = ELLEXMap("test_agent")

    for t in range(50):
        # Generate observations
        obs = {
            # Cognitive
            'cognitive_load': 0.5 + 0.1 * np.random.randn(),
            'attention_focus': 0.6 + 0.1 * np.random.randn(),
            'memory_coherence': 0.7 + 0.1 * np.random.randn(),

            # Symbolic
            'active_concepts': {
                'exploration': 0.7 + 0.1 * np.sin(t * 0.2),
                'connection': 0.6,
                'growth': 0.5 + 0.1 * np.cos(t * 0.1)
            },
            'connections': [('exploration', 'growth'), ('connection', 'growth')],

            # Narrative
            'new_episode': {
                'content': f'Episode {t}',
                'themes': ['growth', 'learning'] if t % 3 == 0 else ['exploration'],
                'valence': 0.5 + 0.3 * np.sin(t * 0.15),
                'significance': 0.6 + 0.2 * np.random.rand()
            },

            # Life
            'drives': {'survival': 0.7, 'curiosity': 0.8 + 0.1 * np.sin(t * 0.1)},
            'goals': {'learn': 0.7, 'connect': 0.6},
            'environment_fit': 0.6,

            # Health
            'diagnosis_quality': 0.7,
            'treatment_efficacy': 0.6,
            'iatrogenesis_rate': 0.05,
            'rotation_health': 0.8,

            # Social
            'social_connections': ['agent_1', 'agent_2'],
            'interaction_quality': 0.7,

            # Tension
            'stress': 0.3 + 0.2 * np.sin(t * 0.2),

            # Identity
            'core_values': {'integrity': 0.9, 'curiosity': 0.85},
            'narrative_element': 'growth' if t % 2 == 0 else 'learning',
            'behavior': 'explore' if t % 3 == 0 else 'reflect',

            # Phase
            'current_phase': ['wake', 'wake', 'liminal', 'rest', 'dream'][t % 5],
            'phase_efficacy': 0.6 + 0.2 * np.random.rand(),
            'multiagent_sync': 0.7
        }

        state = ellex.update(obs)

    print(f"  Final ELLEX: {state.ellex:.3f}")
    print(f"  Existential Zone: {state.existential_zone}")
    print(f"  Tension Zone: {state.tension_zone}")
    print(f"  Health Status: {state.health_status}")
    print(f"  Coherence Mean: {state.coherence_mean:.3f}")
    print(f"  Stability: {state.stability:.3f}")
    print(f"  Trend: {state.trend:+.3f}")

    # Layer summary
    print("\n  Layer Values:")
    summary = ellex.get_layer_summary()
    for name, value in sorted(summary.items()):
        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        print(f"    {name:<15} {bar} {value:.3f}")

    # Weakest and strongest
    print("\n  Weakest areas:")
    for name, value in ellex.get_weakest_areas(3):
        print(f"    {name}: {value:.3f}")

    print("\n  Strongest areas:")
    for name, value in ellex.get_strongest_areas(3):
        print(f"    {name}: {value:.3f}")

    assert 0 <= state.ellex <= 1
    print("\n  [PASS] Full ELLEX map works")


def test_visualizer():
    """Test ELLEX visualizer."""
    print("\n=== Test ELLEX Visualizer ===")
    from ellex_map.ellex_map import ELLEXMap
    from ellex_map.ellex_visualizer import ELLEXVisualizer

    ellex = ELLEXMap("visual_agent")

    # Run a few steps
    for t in range(20):
        obs = {
            'cognitive_load': 0.5,
            'attention_focus': 0.6,
            'memory_coherence': 0.7,
            'active_concepts': {'test': 0.7, 'visual': 0.6},
            'connections': [('test', 'visual')],
            'drives': {'curiosity': 0.8},
            'goals': {'visualize': 0.7},
            'stress': 0.3,
            'current_phase': 'wake',
            'phase_efficacy': 0.7,
            'multiagent_sync': 0.8,
            'core_values': {'clarity': 0.9},
            'narrative_element': 'testing',
            'behavior': 'observe'
        }
        state = ellex.update(obs)

    visualizer = ELLEXVisualizer(ellex)

    # Test ASCII radar
    print("\n  ASCII Radar:")
    radar = visualizer.generate_ascii_radar(state)
    print(radar)

    # Test tension-identity plot
    print("\n  Tension-Identity Plot:")
    ti_plot = visualizer.generate_tension_identity_plot(state)
    print(ti_plot)

    # Test lifecycle view
    print("\n  Lifecycle View:")
    lifecycle = visualizer.generate_lifecycle_view(state)
    print(lifecycle)

    # Test health dashboard
    print("\n  Health Dashboard:")
    health = visualizer.generate_health_dashboard(state)
    print(health)

    print("\n  [PASS] Visualizer works")


def test_full_report():
    """Test full report generation."""
    print("\n=== Test Full Report ===")
    from ellex_map.ellex_map import ELLEXMap
    from ellex_map.ellex_visualizer import ELLEXVisualizer

    ellex = ELLEXMap("report_agent")

    # Run enough steps for meaningful data
    for t in range(30):
        obs = {
            'cognitive_load': 0.5 + 0.1 * np.sin(t * 0.2),
            'attention_focus': 0.6 + 0.1 * np.cos(t * 0.15),
            'memory_coherence': 0.7,
            'active_concepts': {
                'alpha': 0.7,
                'beta': 0.6,
                'gamma': 0.5
            },
            'connections': [('alpha', 'beta'), ('beta', 'gamma')],
            'new_episode': {
                'content': f'Event {t}',
                'themes': ['alpha', 'beta'],
                'valence': 0.5 + 0.3 * np.sin(t * 0.1),
                'significance': 0.6
            },
            'drives': {'d1': 0.7, 'd2': 0.6},
            'goals': {'g1': 0.8},
            'stress': 0.3 + 0.2 * np.abs(np.sin(t * 0.3)),
            'current_phase': ['wake', 'liminal', 'rest', 'dream'][t % 4],
            'phase_efficacy': 0.7,
            'multiagent_sync': 0.75,
            'diagnosis_quality': 0.7,
            'treatment_efficacy': 0.65,
            'core_values': {'v1': 0.9, 'v2': 0.85},
            'narrative_element': 'story',
            'behavior': 'act'
        }
        state = ellex.update(obs)

    visualizer = ELLEXVisualizer(ellex)
    report = visualizer.generate_full_report(state)

    print(report)
    print("\n  [PASS] Full report works")


def run_all_tests():
    """Run all ELLEX-MAP tests."""
    print("=" * 60)
    print("        ELLEX-MAP TEST SUITE")
    print("=" * 60)

    test_layer_emergence()
    test_coherence_surface()
    test_existential_tension()
    test_health_equilibrium()
    test_circadian_phase_space()
    test_ellex_index()
    test_ellex_map()
    test_visualizer()
    test_full_report()

    print("\n" + "=" * 60)
    print("        ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
