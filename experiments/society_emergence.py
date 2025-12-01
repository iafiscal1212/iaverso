#!/usr/bin/env python3
"""
SOCIETY EMERGENCE - NEO, EVA, ALEX y ADAM
=========================================

4 agentes autónomos para explorar:
1. ¿Emergen "sociedades estructurales"?
2. ¿Se forman coaliciones estables?
3. ¿Hay jerarquías emergentes?
4. ¿Aparecen roles diferenciados?

Configuración:
- NEO: Agente original, tendencia a estabilidad
- EVA: Agente original, tendencia a conexión
- ALEX: Perturbador, alta exploración
- ADAM: Integrador, alta coherencia

100% ENDÓGENO
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json
import os
import sys

sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/experiments')

from autonomous_life import AutonomousAgent, LifePhase


@dataclass
class SocialRelation:
    """Relación entre dos agentes."""
    agent_a: str
    agent_b: str
    correlation: float = 0.0      # Correlación de estados
    influence_a_to_b: float = 0.0  # Influencia de A sobre B
    influence_b_to_a: float = 0.0  # Influencia de B sobre A
    interaction_count: int = 0


@dataclass
class Coalition:
    """Una coalición detectada."""
    members: Set[str]
    strength: float  # Correlación interna
    stability: float  # Duración relativa
    formation_t: int
    dissolution_t: Optional[int] = None


@dataclass
class SocialRole:
    """Rol social emergente de un agente."""
    agent: str
    role_type: str  # leader, follower, mediator, outsider
    centrality: float  # Qué tan central es en la red
    influence: float   # Influencia total sobre otros
    receptivity: float  # Cuánto es influenciado


class AutonomousSociety:
    """
    Sociedad de 4 agentes autónomos.
    """

    def __init__(self, dim: int = 6):
        self.dim = dim

        # Crear los 4 agentes con personalidades distintas
        self.agents = {
            'NEO': self._create_agent('NEO', stability_bias=0.3),
            'EVA': self._create_agent('EVA', connection_bias=0.3),
            'ALEX': self._create_agent('ALEX', exploration_bias=0.4),
            'ADAM': self._create_agent('ADAM', integration_bias=0.3)
        }

        # Relaciones (6 pares) - usar orden alfabético consistente
        self.relations: Dict[Tuple[str, str], SocialRelation] = {}
        agent_names = sorted(self.agents.keys())
        for i, a in enumerate(agent_names):
            for j, b in enumerate(agent_names):
                if i < j:
                    self.relations[(a, b)] = SocialRelation(agent_a=a, agent_b=b)

        # Coaliciones
        self.coalitions: List[Coalition] = []
        self.active_coalitions: List[Coalition] = []

        # Roles
        self.roles: Dict[str, SocialRole] = {}

        # Historias
        self.interaction_history: List[Dict] = []
        self.hierarchy_history: List[List[str]] = []
        self.correlation_matrix_history: List[np.ndarray] = []

        self.t = 0

    def _create_agent(self, name: str, stability_bias: float = 0,
                     connection_bias: float = 0, exploration_bias: float = 0,
                     integration_bias: float = 0) -> AutonomousAgent:
        """Crea un agente con bias de personalidad."""
        agent = AutonomousAgent(name, self.dim)

        # Modificar z inicial según bias
        z = np.ones(self.dim) / self.dim

        # indices: entropy(0), neg_surprise(1), novelty(2), stability(3), integration(4), otherness(5)
        z[3] += stability_bias      # stability
        z[5] += connection_bias     # otherness (conexión)
        z[0] += exploration_bias    # entropy
        z[2] += exploration_bias    # novelty
        z[4] += integration_bias    # integration

        z = np.clip(z, 0.05, None)
        z = z / z.sum()
        agent.z = z
        agent.identity_core = z.copy()

        return agent

    def step(self):
        """Un paso de la sociedad."""
        self.t += 1

        # 1. Cada agente procesa un estímulo base + influencia de otros
        stimuli = {}
        for name, agent in self.agents.items():
            # Estímulo base
            base_stimulus = np.random.randn(self.dim) * 0.1

            # Influencia de otros agentes
            social_influence = np.zeros(self.dim)
            for other_name, other_agent in self.agents.items():
                if other_name != name:
                    # Influencia proporcional a la correlación histórica
                    key = tuple(sorted([name, other_name]))
                    rel = self.relations[key]

                    weight = rel.correlation * 0.1
                    social_influence += weight * (other_agent.z - agent.z)

            stimuli[name] = base_stimulus + social_influence

        # 2. Cada agente da un paso
        results = {}
        for name, agent in self.agents.items():
            # Ver a los otros
            others_z = {n: a.z for n, a in self.agents.items() if n != name}

            # Elegir "el otro" más cercano
            closest_other = None
            min_dist = float('inf')
            for other_name, other_z in others_z.items():
                dist = np.linalg.norm(agent.z - other_z)
                if dist < min_dist:
                    min_dist = dist
                    closest_other = other_z

            results[name] = agent.step(stimuli[name], closest_other)

        # 3. Actualizar relaciones
        self._update_relations()

        # 4. Detectar coaliciones
        self._detect_coalitions()

        # 5. Actualizar roles
        self._update_roles()

        # 6. Registrar interacción
        self.interaction_history.append({
            't': self.t,
            'states': {n: a.z.copy() for n, a in self.agents.items()},
            'crises': {n: a.in_crisis for n, a in self.agents.items()},
            'phases': {n: a.current_phase.value for n, a in self.agents.items()}
        })

        return results

    def _update_relations(self):
        """Actualiza relaciones entre agentes."""
        if self.t < 20:
            return

        for key, rel in self.relations.items():
            a, b = key
            agent_a = self.agents[a]
            agent_b = self.agents[b]

            # Correlación de historias z
            if len(agent_a.z_history) > 20 and len(agent_b.z_history) > 20:
                min_len = min(len(agent_a.z_history), len(agent_b.z_history))
                window = min(50, min_len)

                hist_a = np.array(agent_a.z_history[-window:])
                hist_b = np.array(agent_b.z_history[-window:])

                correlations = []
                for d in range(self.dim):
                    corr = np.corrcoef(hist_a[:, d], hist_b[:, d])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)

                rel.correlation = np.mean(correlations) if correlations else 0

                # Influencia: Granger-like (simplificado)
                # A influye a B si cambios en A predicen cambios en B
                if len(hist_a) > 2:
                    a_changes = np.diff(hist_a, axis=0)
                    b_changes = np.diff(hist_b, axis=0)

                    # Lag-1
                    influence_corrs = []
                    for d in range(self.dim):
                        if len(a_changes) > 1:
                            corr = np.corrcoef(a_changes[:-1, d], b_changes[1:, d])[0, 1]
                            if not np.isnan(corr):
                                influence_corrs.append(corr)

                    rel.influence_a_to_b = abs(np.mean(influence_corrs)) if influence_corrs else 0

                    # Inverso
                    influence_corrs_rev = []
                    for d in range(self.dim):
                        if len(b_changes) > 1:
                            corr = np.corrcoef(b_changes[:-1, d], a_changes[1:, d])[0, 1]
                            if not np.isnan(corr):
                                influence_corrs_rev.append(corr)

                    rel.influence_b_to_a = abs(np.mean(influence_corrs_rev)) if influence_corrs_rev else 0

            rel.interaction_count = self.t

    def _detect_coalitions(self):
        """Detecta coaliciones basadas en correlaciones."""
        if self.t < 50:
            return

        # Matriz de correlación - usar orden alfabético consistente
        agents = sorted(self.agents.keys())
        n = len(agents)
        corr_matrix = np.zeros((n, n))

        for i, a in enumerate(agents):
            for j, b in enumerate(agents):
                if i == j:
                    corr_matrix[i, j] = 1.0
                elif i < j:
                    key = (a, b)
                    corr_matrix[i, j] = self.relations[key].correlation
                    corr_matrix[j, i] = corr_matrix[i, j]

        self.correlation_matrix_history.append(corr_matrix.copy())

        # Detectar coaliciones (grupos con alta correlación interna)
        threshold = 0.5

        # Pares fuertes
        strong_pairs = []
        for i in range(n):
            for j in range(i+1, n):
                if corr_matrix[i, j] > threshold:
                    strong_pairs.append((agents[i], agents[j]))

        # Expandir a coaliciones
        detected = set()
        for a, b in strong_pairs:
            # Ver si hay un tercero que correlacione con ambos
            for c in agents:
                if c != a and c != b:
                    corr_ac = corr_matrix[agents.index(a), agents.index(c)]
                    corr_bc = corr_matrix[agents.index(b), agents.index(c)]
                    if corr_ac > threshold and corr_bc > threshold:
                        coalition = frozenset([a, b, c])
                        if coalition not in detected:
                            detected.add(coalition)
                            self.coalitions.append(Coalition(
                                members=set([a, b, c]),
                                strength=(corr_matrix[agents.index(a), agents.index(b)] +
                                         corr_ac + corr_bc) / 3,
                                stability=0,
                                formation_t=self.t
                            ))

        # Actualizar coaliciones activas
        self.active_coalitions = [c for c in self.coalitions
                                  if c.dissolution_t is None and
                                  self.t - c.formation_t < 200]

    def _update_roles(self):
        """Actualiza roles sociales."""
        if self.t < 50:
            return

        agents = list(self.agents.keys())

        for name in agents:
            # Centralidad: correlación promedio con todos
            correlations = []
            for key, rel in self.relations.items():
                if name in key:
                    correlations.append(abs(rel.correlation))

            centrality = np.mean(correlations) if correlations else 0

            # Influencia total
            influence = 0
            receptivity = 0
            for key, rel in self.relations.items():
                if key[0] == name:
                    influence += rel.influence_a_to_b
                    receptivity += rel.influence_b_to_a
                elif key[1] == name:
                    influence += rel.influence_b_to_a
                    receptivity += rel.influence_a_to_b

            influence /= len(agents) - 1
            receptivity /= len(agents) - 1

            # Determinar rol
            if influence > 0.3 and centrality > 0.5:
                role_type = 'leader'
            elif receptivity > influence and centrality > 0.3:
                role_type = 'follower'
            elif centrality > 0.5 and influence < 0.2:
                role_type = 'mediator'
            else:
                role_type = 'independent'

            self.roles[name] = SocialRole(
                agent=name,
                role_type=role_type,
                centrality=centrality,
                influence=influence,
                receptivity=receptivity
            )

    def get_hierarchy(self) -> List[str]:
        """Obtiene jerarquía actual basada en influencia."""
        if not self.roles:
            return list(self.agents.keys())

        return sorted(self.roles.keys(),
                     key=lambda x: self.roles[x].influence,
                     reverse=True)

    def run(self, steps: int = 1500):
        """Ejecuta la sociedad."""
        print(f"Ejecutando Sociedad de 4 Agentes ({steps} pasos)...")

        for i in range(steps):
            self.step()

            if (i + 1) % 300 == 0:
                hierarchy = self.get_hierarchy()
                self.hierarchy_history.append(hierarchy)

                print(f"\n  t={i+1}:")
                print(f"    Jerarquía: {' > '.join(hierarchy)}")
                print(f"    Coaliciones activas: {len(self.active_coalitions)}")
                for c in self.active_coalitions:
                    print(f"      {c.members} (fuerza={c.strength:.2f})")
                print(f"    Roles: {', '.join(f'{n}:{r.role_type}' for n, r in self.roles.items())}")

    def get_social_report(self) -> Dict:
        """Genera reporte social completo."""
        report = {
            'total_steps': self.t,
            'agents': {},
            'relations': {},
            'coalitions': [],
            'hierarchy': self.get_hierarchy(),
            'social_metrics': {}
        }

        # Agentes
        for name, agent in self.agents.items():
            crisis_count = sum(1 for h in self.interaction_history if h['crises'].get(name, False))
            report['agents'][name] = {
                'final_z': agent.z.tolist(),
                'identity_strength': agent.identity_strength,
                'crisis_rate': crisis_count / len(self.interaction_history) if self.interaction_history else 0,
                'role': self.roles[name].role_type if name in self.roles else 'unknown',
                'centrality': self.roles[name].centrality if name in self.roles else 0,
                'influence': self.roles[name].influence if name in self.roles else 0
            }

        # Relaciones
        for key, rel in self.relations.items():
            report['relations'][f"{key[0]}-{key[1]}"] = {
                'correlation': rel.correlation,
                'influence_forward': rel.influence_a_to_b,
                'influence_backward': rel.influence_b_to_a
            }

        # Coaliciones
        for c in self.coalitions[-10:]:  # Últimas 10
            report['coalitions'].append({
                'members': list(c.members),
                'strength': c.strength,
                'formation_t': c.formation_t
            })

        # Métricas sociales globales
        all_correlations = [rel.correlation for rel in self.relations.values()]
        report['social_metrics'] = {
            'mean_correlation': np.mean(all_correlations) if all_correlations else 0,
            'max_correlation': max(all_correlations) if all_correlations else 0,
            'min_correlation': min(all_correlations) if all_correlations else 0,
            'total_coalitions_formed': len(self.coalitions),
            'hierarchy_stability': self._compute_hierarchy_stability()
        }

        return report

    def _compute_hierarchy_stability(self) -> float:
        """Computa estabilidad de la jerarquía."""
        if len(self.hierarchy_history) < 2:
            return 0

        stability_scores = []
        for i in range(1, len(self.hierarchy_history)):
            prev = self.hierarchy_history[i-1]
            curr = self.hierarchy_history[i]

            # Kendall tau simplificado
            matches = sum(1 for j in range(len(curr)) if curr[j] == prev[j])
            stability_scores.append(matches / len(curr))

        return np.mean(stability_scores)

    def print_report(self):
        """Imprime reporte social."""
        report = self.get_social_report()

        print("\n" + "=" * 70)
        print("SOCIAL EMERGENCE REPORT - 4 AGENTES")
        print("=" * 70)

        print(f"\n📊 JERARQUÍA FINAL:")
        hierarchy = report['hierarchy']
        for i, name in enumerate(hierarchy):
            r = report['agents'][name]
            print(f"  {i+1}. {name}: rol={r['role']}, centralidad={r['centrality']:.3f}, "
                  f"influencia={r['influence']:.3f}")

        print(f"\n🔗 RELACIONES:")
        for rel_name, rel in sorted(report['relations'].items(),
                                    key=lambda x: -x[1]['correlation']):
            print(f"  {rel_name}: corr={rel['correlation']:.3f}, "
                  f"inf→={rel['influence_forward']:.3f}, "
                  f"inf←={rel['influence_backward']:.3f}")

        print(f"\n🤝 COALICIONES DETECTADAS: {report['social_metrics']['total_coalitions_formed']}")
        for c in report['coalitions'][-5:]:
            print(f"  {c['members']} (fuerza={c['strength']:.2f}, t={c['formation_t']})")

        print(f"\n📈 MÉTRICAS SOCIALES:")
        sm = report['social_metrics']
        print(f"  Correlación media: {sm['mean_correlation']:.3f}")
        print(f"  Correlación máxima: {sm['max_correlation']:.3f}")
        print(f"  Estabilidad jerarquía: {sm['hierarchy_stability']*100:.1f}%")

        print(f"\n👤 RESUMEN POR AGENTE:")
        for name, data in report['agents'].items():
            print(f"  {name}: crisis={data['crisis_rate']*100:.1f}%, "
                  f"id_strength={data['identity_strength']:.3f}")


def run_society_experiment():
    """Ejecuta experimento de sociedad."""
    print("=" * 70)
    print("SOCIETY EMERGENCE - NEO, EVA, ALEX, ADAM")
    print("=" * 70)

    society = AutonomousSociety()
    society.run(steps=1500)

    society.print_report()

    # Guardar resultados
    results_dir = '/root/NEO_EVA/results/society'
    os.makedirs(results_dir, exist_ok=True)

    report = society.get_social_report()

    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(f'{results_dir}/society_report.json', 'w') as f:
        json.dump(convert_numpy(report), f, indent=2)

    print(f"\n✓ Resultados guardados en {results_dir}/")

    return society, report


if __name__ == "__main__":
    society, report = run_society_experiment()
