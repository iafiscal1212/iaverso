#!/usr/bin/env python3
"""
Phase I2-Dual: IGI Separado por Agente
======================================

Calcula IGI_NEO(t) e IGI_EVA(t) de forma independiente:
- Integración interna de cada agente
- Integración ecológica (cómo se acopla con el otro)
- Tests GO separados contra nulos

100% ENDÓGENO - Sin constantes mágicas
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import sys

sys.path.insert(0, '/root/NEO_EVA/integration')
from phaseI1_subsystems import SubsystemDecomposition, Subsystem


@dataclass
class AgentIntegration:
    """Métricas de integración para un agente."""
    t: int
    I_int: float      # Integración interna
    I_eco: float      # Integración ecológica
    IGI: float        # Índice total
    in_episode: bool  # En episodio de alta integración


class DualAgentIGI:
    """
    Calcula IGI separado para NEO y EVA.

    Subconjuntos:
    - NEO: {NEO_vis, NEO_hid, workspace, drives}
    - EVA: {EVA_vis, EVA_hid, workspace, drives}

    Para cada agente A:
    - I_A_int(t) = mean_{i,j ∈ A, i≠j} rank(TE_ij(t))
    - I_A_eco(t) = mean_{i∈A, j∈B} rank(TE_ij(t) + TE_ji(t))
    - IGI_A(t) = rank(I_A_int) + rank(I_A_eco)

    100% Endógeno
    """

    def __init__(self, total_dim: int = 12):
        self.decomp = SubsystemDecomposition(total_dim=total_dim)

        # Índices de subsistemas por agente
        # NEO: NEO_vis(0), NEO_hid(1), workspace(4), drives(5)
        # EVA: EVA_vis(2), EVA_hid(3), workspace(4), drives(5)
        self.neo_indices = [0, 1, 4, 5]  # NEO_vis, NEO_hid, workspace, drives
        self.eva_indices = [2, 3, 4, 5]  # EVA_vis, EVA_hid, workspace, drives

        # Índices exclusivos (para integración ecológica)
        self.neo_exclusive = [0, 1]  # Solo NEO
        self.eva_exclusive = [2, 3]  # Solo EVA
        self.shared = [4, 5]         # Compartidos

        # Historia por agente
        self.neo_I_int_history: List[float] = []
        self.neo_I_eco_history: List[float] = []
        self.neo_IGI_history: List[float] = []

        self.eva_I_int_history: List[float] = []
        self.eva_I_eco_history: List[float] = []
        self.eva_IGI_history: List[float] = []

        # Episodios de alta integración
        self.neo_episodes: List[Dict] = []
        self.eva_episodes: List[Dict] = []
        self.neo_in_episode = False
        self.eva_in_episode = False
        self.neo_episode_start = 0
        self.eva_episode_start = 0

        # Historia de TE
        self.te_history: List[np.ndarray] = []
        self.z_history: List[np.ndarray] = []

        self.t = 0

    def _compute_internal_integration(self, te_matrix: np.ndarray,
                                       indices: List[int]) -> float:
        """
        Calcula integración interna de un agente.

        I_A_int = mean_{i,j ∈ A, i≠j} rank(TE_ij)

        100% endógeno: ranks sobre valores actuales
        """
        values = []
        for i in indices:
            for j in indices:
                if i != j:
                    values.append(te_matrix[i, j])

        if not values:
            return 0.0

        # Calcular ranks
        sorted_vals = np.sort(values)
        ranks = [np.searchsorted(sorted_vals, v) / len(sorted_vals) for v in values]

        return float(np.mean(ranks))

    def _compute_ecological_integration(self, te_matrix: np.ndarray,
                                         agent_indices: List[int],
                                         other_indices: List[int]) -> float:
        """
        Calcula integración ecológica (cómo se acopla con el otro agente).

        I_A_eco = mean_{i∈A, j∈B} rank(TE_ij + TE_ji)

        100% endógeno
        """
        values = []
        for i in agent_indices:
            for j in other_indices:
                if i != j:
                    # Suma bidireccional
                    te_sum = te_matrix[i, j] + te_matrix[j, i]
                    values.append(te_sum)

        if not values:
            return 0.0

        # Calcular ranks
        sorted_vals = np.sort(values)
        ranks = [np.searchsorted(sorted_vals, v) / len(sorted_vals) for v in values]

        return float(np.mean(ranks))

    def _compute_agent_igi(self, I_int: float, I_eco: float,
                           I_int_history: List[float],
                           I_eco_history: List[float]) -> float:
        """
        Calcula IGI de un agente.

        IGI_A = rank(I_A_int) + rank(I_A_eco)

        100% endógeno: ranks sobre historia
        """
        if len(I_int_history) < 2:
            return 1.0

        # Rank de I_int
        sorted_int = np.sort(I_int_history)
        rank_int = np.searchsorted(sorted_int, I_int) / len(sorted_int)

        # Rank de I_eco
        sorted_eco = np.sort(I_eco_history)
        rank_eco = np.searchsorted(sorted_eco, I_eco) / len(sorted_eco)

        return rank_int + rank_eco

    def _check_episode(self, IGI: float, IGI_history: List[float],
                        in_episode: bool, episode_start: int,
                        episodes: List[Dict], agent: str) -> Tuple[bool, int, Dict]:
        """
        Detecta episodios de alta integración para un agente.

        Alta integración: IGI >= q90 de historia
        """
        event = {}

        if len(IGI_history) < 10:
            return in_episode, episode_start, event

        threshold = np.percentile(IGI_history, 90)
        is_high = IGI >= threshold

        if is_high and not in_episode:
            # Inicio de episodio
            in_episode = True
            episode_start = self.t
            event = {'agent': agent, 'event': 'start', 't': self.t}

        elif not is_high and in_episode:
            # Fin de episodio
            in_episode = False
            duration = self.t - episode_start
            mean_igi = np.mean(IGI_history[episode_start:self.t]) if episode_start < len(IGI_history) else IGI

            episode = {
                'agent': agent,
                'start_t': episode_start,
                'end_t': self.t,
                'duration': duration,
                'mean_igi': float(mean_igi)
            }
            episodes.append(episode)
            event = {'agent': agent, 'event': 'end', 'episode': episode}

        return in_episode, episode_start, event

    def step(self, z: np.ndarray) -> Dict[str, Any]:
        """
        Ejecuta un paso de cálculo de IGI dual.

        Args:
            z: Estado global

        Returns:
            Dict con IGI_NEO, IGI_EVA y métricas
        """
        self.t += 1
        self.z_history.append(z.copy())

        # Paso de descomposición
        decomp_result = self.decomp.step(z)

        if not decomp_result['ready']:
            return {
                't': self.t,
                'ready': False,
                'IGI_NEO': None,
                'IGI_EVA': None
            }

        te_matrix = decomp_result['TE_matrix']
        self.te_history.append(te_matrix)

        # === NEO ===
        neo_I_int = self._compute_internal_integration(te_matrix, self.neo_indices)
        neo_I_eco = self._compute_ecological_integration(
            te_matrix, self.neo_exclusive, self.eva_exclusive + self.shared
        )
        self.neo_I_int_history.append(neo_I_int)
        self.neo_I_eco_history.append(neo_I_eco)

        neo_IGI = self._compute_agent_igi(
            neo_I_int, neo_I_eco,
            self.neo_I_int_history, self.neo_I_eco_history
        )
        self.neo_IGI_history.append(neo_IGI)

        # Episodios NEO
        self.neo_in_episode, self.neo_episode_start, neo_event = self._check_episode(
            neo_IGI, self.neo_IGI_history,
            self.neo_in_episode, self.neo_episode_start,
            self.neo_episodes, 'NEO'
        )

        # === EVA ===
        eva_I_int = self._compute_internal_integration(te_matrix, self.eva_indices)
        eva_I_eco = self._compute_ecological_integration(
            te_matrix, self.eva_exclusive, self.neo_exclusive + self.shared
        )
        self.eva_I_int_history.append(eva_I_int)
        self.eva_I_eco_history.append(eva_I_eco)

        eva_IGI = self._compute_agent_igi(
            eva_I_int, eva_I_eco,
            self.eva_I_int_history, self.eva_I_eco_history
        )
        self.eva_IGI_history.append(eva_IGI)

        # Episodios EVA
        self.eva_in_episode, self.eva_episode_start, eva_event = self._check_episode(
            eva_IGI, self.eva_IGI_history,
            self.eva_in_episode, self.eva_episode_start,
            self.eva_episodes, 'EVA'
        )

        return {
            't': self.t,
            'ready': True,
            'NEO': AgentIntegration(
                t=self.t,
                I_int=neo_I_int,
                I_eco=neo_I_eco,
                IGI=neo_IGI,
                in_episode=self.neo_in_episode
            ),
            'EVA': AgentIntegration(
                t=self.t,
                I_int=eva_I_int,
                I_eco=eva_I_eco,
                IGI=eva_IGI,
                in_episode=self.eva_in_episode
            ),
            'IGI_NEO': neo_IGI,
            'IGI_EVA': eva_IGI,
            'neo_event': neo_event,
            'eva_event': eva_event
        }

    def _generate_null_shuffle_agent(self, agent: str) -> List[float]:
        """
        Genera IGI nulo por shuffle para un agente.

        100% endógeno
        """
        if len(self.z_history) < 30:
            return []

        T = len(self.z_history)
        null_decomp = SubsystemDecomposition(total_dim=self.decomp.total_dim)

        # Shuffle por subsistema
        z_shuffled = []
        for t in range(T):
            z_new = np.zeros(self.decomp.total_dim)
            for subsys in self.decomp.subsystems:
                random_t = np.random.randint(0, T)
                z_new[subsys.start_idx:subsys.end_idx] = \
                    self.z_history[random_t][subsys.start_idx:subsys.end_idx]
            z_shuffled.append(z_new)

        # Calcular IGI null
        null_I_int = []
        null_I_eco = []
        null_IGI = []

        indices = self.neo_indices if agent == 'NEO' else self.eva_indices
        exclusive = self.neo_exclusive if agent == 'NEO' else self.eva_exclusive
        other = self.eva_exclusive if agent == 'NEO' else self.neo_exclusive

        for z in z_shuffled:
            result = null_decomp.step(z)
            if result['ready']:
                te = result['TE_matrix']

                I_int = self._compute_internal_integration(te, indices)
                I_eco = self._compute_ecological_integration(te, exclusive, other + self.shared)

                null_I_int.append(I_int)
                null_I_eco.append(I_eco)

                if len(null_I_int) > 1:
                    IGI = self._compute_agent_igi(I_int, I_eco, null_I_int, null_I_eco)
                    null_IGI.append(IGI)

        return null_IGI

    def certify_agent(self, agent: str, n_nulls: int = 5) -> Dict[str, Any]:
        """
        Certifica integración de un agente contra nulos.

        GO si:
        1. IGI_A_mean > p95(IGI_null)
        2. Var(IGI_A) > p95(Var_null) (estructura diferenciada)
        3. Correlación IGI_A con TE_A > 0

        100% endógeno
        """
        IGI_history = self.neo_IGI_history if agent == 'NEO' else self.eva_IGI_history
        episodes = self.neo_episodes if agent == 'NEO' else self.eva_episodes

        if len(IGI_history) < 50:
            return {'certified': False, 'reason': 'insufficient_data', 'agent': agent}

        IGI_mean = np.mean(IGI_history)
        IGI_var = np.var(IGI_history)

        # Generar nulos
        null_IGI = []
        null_vars = []

        for _ in range(n_nulls):
            null = self._generate_null_shuffle_agent(agent)
            if null:
                null_IGI.extend(null)
                null_vars.append(np.var(null))

        if not null_IGI:
            return {'certified': False, 'reason': 'null_generation_failed', 'agent': agent}

        # Test 1: IGI_mean > p95(null)
        p95_null = np.percentile(null_IGI, 95)
        test_mean = IGI_mean > p95_null

        # Test 2: Var(IGI) > p95(Var_null)
        if null_vars:
            p95_var = np.percentile(null_vars, 95)
            test_var = IGI_var > p95_var
        else:
            p95_var = 0
            test_var = False

        # Test 3: Correlación con TE total del agente
        if len(self.te_history) >= len(IGI_history):
            indices = self.neo_indices if agent == 'NEO' else self.eva_indices
            te_means = []
            for te in self.te_history[-len(IGI_history):]:
                te_agent = np.mean([te[i, j] for i in indices for j in indices if i != j])
                te_means.append(te_agent)

            if len(te_means) == len(IGI_history):
                corr = np.corrcoef(IGI_history, te_means)[0, 1]
                test_corr = corr > 0 if not np.isnan(corr) else False
            else:
                corr = 0
                test_corr = False
        else:
            corr = 0
            test_corr = False

        # Test 4: Episodios de alta integración
        T = len(IGI_history)
        min_duration = max(1, int(np.sqrt(T) / 4))
        long_episodes = [e for e in episodes if e['duration'] >= min_duration]
        K_required = max(1, int(np.sqrt(T) / 10))
        test_episodes = len(long_episodes) >= K_required

        certified = sum([test_mean, test_var, test_corr, test_episodes]) >= 3

        return {
            'agent': agent,
            'certified': certified,
            'IGI_mean': float(IGI_mean),
            'IGI_var': float(IGI_var),
            'p95_null': float(p95_null),
            'p95_var_null': float(p95_var),
            'test_mean': test_mean,
            'test_var': test_var,
            'corr_TE': float(corr) if not np.isnan(corr) else 0.0,
            'test_corr': test_corr,
            'n_long_episodes': len(long_episodes),
            'K_required': K_required,
            'test_episodes': test_episodes
        }

    def get_comparison(self) -> Dict[str, Any]:
        """Compara integración NEO vs EVA."""
        if not self.neo_IGI_history or not self.eva_IGI_history:
            return {'ready': False}

        n = min(len(self.neo_IGI_history), len(self.eva_IGI_history))

        neo_mean = np.mean(self.neo_IGI_history[-n:])
        eva_mean = np.mean(self.eva_IGI_history[-n:])

        # Correlación entre IGI_NEO e IGI_EVA
        corr = np.corrcoef(self.neo_IGI_history[-n:], self.eva_IGI_history[-n:])[0, 1]

        # Divergencia
        divergence = abs(neo_mean - eva_mean)

        return {
            'ready': True,
            't': self.t,
            'NEO': {
                'IGI_mean': float(neo_mean),
                'IGI_std': float(np.std(self.neo_IGI_history[-n:])),
                'I_int_mean': float(np.mean(self.neo_I_int_history[-n:])),
                'I_eco_mean': float(np.mean(self.neo_I_eco_history[-n:])),
                'n_episodes': len(self.neo_episodes)
            },
            'EVA': {
                'IGI_mean': float(eva_mean),
                'IGI_std': float(np.std(self.eva_IGI_history[-n:])),
                'I_int_mean': float(np.mean(self.eva_I_int_history[-n:])),
                'I_eco_mean': float(np.mean(self.eva_I_eco_history[-n:])),
                'n_episodes': len(self.eva_episodes)
            },
            'correlation': float(corr) if not np.isnan(corr) else 0.0,
            'divergence': float(divergence),
            'dominant': 'NEO' if neo_mean > eva_mean else 'EVA'
        }


def run_phase_i2_dual() -> Dict[str, Any]:
    """Ejecuta Phase I2-Dual y evalúa criterios GO/NO-GO."""

    print("=" * 70)
    print("PHASE I2-DUAL: IGI SEPARADO POR AGENTE")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}\n")

    np.random.seed(42)

    # Crear sistema
    igi_dual = DualAgentIGI(total_dim=12)

    # Simulación extendida
    T = 500

    z = np.random.rand(12)
    z = z / z.sum()

    print("Simulando dinámica...")
    for t in range(T):
        # Dinámica con estructura
        noise = np.random.randn(12) * 0.02

        # Acoplos estructurados
        if t > 0:
            # NEO: más integración interna
            z[0:2] += 0.08 * z[2:4]  # NEO_vis ← NEO_hid
            z[2:4] += 0.05 * z[0:2]  # NEO_hid ← NEO_vis

            # EVA: más integración ecológica
            z[4:6] += 0.1 * z[0:2]   # EVA_vis ← NEO_vis
            z[0:2] += 0.08 * z[4:6]  # NEO_vis ← EVA_vis

            # Workspace y drives
            z[8:10] += 0.05 * (z[0:2] + z[4:6])  # workspace ← ambos
            z[10:12] += 0.03 * z[8:10]            # drives ← workspace

        # Perturbaciones periódicas diferenciadas
        if t % 100 < 20:
            z[0:4] += 0.1 * np.random.randn(4) * 0.1  # Perturbar NEO
        if t % 100 >= 50 and t % 100 < 70:
            z[4:8] += 0.1 * np.random.randn(4) * 0.1  # Perturbar EVA

        z = z + noise
        z = np.clip(z, 0.01, 0.99)
        z = z / z.sum()

        result = igi_dual.step(z)

        if t % 100 == 0 and result['ready']:
            print(f"  t={t}: IGI_NEO={result['IGI_NEO']:.3f}, IGI_EVA={result['IGI_EVA']:.3f}")

    print()

    # Certificación por agente
    print("Certificando integración por agente...")
    cert_neo = igi_dual.certify_agent('NEO', n_nulls=5)
    cert_eva = igi_dual.certify_agent('EVA', n_nulls=5)

    print()
    print("=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print()

    comparison = igi_dual.get_comparison()

    print("NEO:")
    print(f"  IGI medio: {comparison['NEO']['IGI_mean']:.4f}")
    print(f"  I_int medio: {comparison['NEO']['I_int_mean']:.4f}")
    print(f"  I_eco medio: {comparison['NEO']['I_eco_mean']:.4f}")
    print(f"  Episodios: {comparison['NEO']['n_episodes']}")
    print(f"  Certificado: {'Sí' if cert_neo['certified'] else 'No'}")
    print()

    print("EVA:")
    print(f"  IGI medio: {comparison['EVA']['IGI_mean']:.4f}")
    print(f"  I_int medio: {comparison['EVA']['I_int_mean']:.4f}")
    print(f"  I_eco medio: {comparison['EVA']['I_eco_mean']:.4f}")
    print(f"  Episodios: {comparison['EVA']['n_episodes']}")
    print(f"  Certificado: {'Sí' if cert_eva['certified'] else 'No'}")
    print()

    print(f"Correlación NEO-EVA: {comparison['correlation']:.4f}")
    print(f"Divergencia: {comparison['divergence']:.4f}")
    print(f"Dominante: {comparison['dominant']}")
    print()

    # Criterios GO/NO-GO
    criteria = {}

    # 1. IGI calculados para ambos
    criteria['igi_computed'] = len(igi_dual.neo_IGI_history) > 0 and len(igi_dual.eva_IGI_history) > 0

    # 2. NEO certificado
    criteria['neo_certified'] = cert_neo['certified']

    # 3. EVA certificado
    criteria['eva_certified'] = cert_eva['certified']

    # 4. Integración diferenciada (divergencia > 0)
    criteria['differentiated'] = comparison['divergence'] > 0.05

    # 5. Correlación positiva (ambos responden a estímulos similares)
    criteria['correlated'] = comparison['correlation'] > 0

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
        'phase': 'I2-Dual',
        'name': 'Dual Agent IGI',
        'timestamp': datetime.now().isoformat(),
        'comparison': comparison,
        'certification_NEO': cert_neo,
        'certification_EVA': cert_eva,
        'criteria': criteria,
        'go_status': go_status,
        'passed_criteria': passed,
        'total_criteria': total
    }

    os.makedirs('/root/NEO_EVA/results/phaseI2_dual', exist_ok=True)

    with open('/root/NEO_EVA/results/phaseI2_dual/igi_dual_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Visualización
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. IGI temporal por agente
        ax1 = axes[0, 0]
        ax1.plot(igi_dual.neo_IGI_history, 'b-', label='IGI NEO', alpha=0.7)
        ax1.plot(igi_dual.eva_IGI_history, 'r-', label='IGI EVA', alpha=0.7)
        ax1.set_xlabel('Tiempo')
        ax1.set_ylabel('IGI')
        ax1.set_title('Índice de Integración Global por Agente')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. I_int vs I_eco por agente
        ax2 = axes[0, 1]
        ax2.scatter(igi_dual.neo_I_int_history, igi_dual.neo_I_eco_history,
                   c='blue', alpha=0.3, s=10, label='NEO')
        ax2.scatter(igi_dual.eva_I_int_history, igi_dual.eva_I_eco_history,
                   c='red', alpha=0.3, s=10, label='EVA')
        ax2.set_xlabel('I_int (Integración Interna)')
        ax2.set_ylabel('I_eco (Integración Ecológica)')
        ax2.set_title('Espacio Integración Interna vs Ecológica')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. I_int temporal
        ax3 = axes[1, 0]
        ax3.plot(igi_dual.neo_I_int_history, 'b-', label='NEO I_int', alpha=0.7)
        ax3.plot(igi_dual.eva_I_int_history, 'r-', label='EVA I_int', alpha=0.7)
        ax3.plot(igi_dual.neo_I_eco_history, 'b--', label='NEO I_eco', alpha=0.5)
        ax3.plot(igi_dual.eva_I_eco_history, 'r--', label='EVA I_eco', alpha=0.5)
        ax3.set_xlabel('Tiempo')
        ax3.set_ylabel('Valor')
        ax3.set_title('Componentes de Integración')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Diferencia IGI
        ax4 = axes[1, 1]
        n = min(len(igi_dual.neo_IGI_history), len(igi_dual.eva_IGI_history))
        diff = np.array(igi_dual.neo_IGI_history[-n:]) - np.array(igi_dual.eva_IGI_history[-n:])
        ax4.plot(diff, 'purple', alpha=0.7)
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.fill_between(range(len(diff)), 0, diff,
                        where=np.array(diff) > 0, color='blue', alpha=0.3, label='NEO > EVA')
        ax4.fill_between(range(len(diff)), 0, diff,
                        where=np.array(diff) < 0, color='red', alpha=0.3, label='EVA > NEO')
        ax4.set_xlabel('Tiempo')
        ax4.set_ylabel('IGI_NEO - IGI_EVA')
        ax4.set_title('Diferencia de Integración')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs('/root/NEO_EVA/figures', exist_ok=True)
        plt.savefig('/root/NEO_EVA/figures/phaseI2_dual_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nResultados guardados en: /root/NEO_EVA/results/phaseI2_dual")
        print(f"Figura: /root/NEO_EVA/figures/phaseI2_dual_results.png")

    except Exception as e:
        print(f"Warning: No se pudo crear visualización: {e}")

    return output


if __name__ == "__main__":
    run_phase_i2_dual()
