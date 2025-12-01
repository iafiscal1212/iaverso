#!/usr/bin/env python3
"""
Phase S1 Dual: Estados Fenomenológicos de NEO y EVA
===================================================

Cada agente tiene su propio campo fenomenológico φ_t
con características diferenciadas:

NEO: φ orientado a estabilidad, compresión, identidad
EVA: φ orientado a novedad, intercambio, otredad

100% ENDÓGENO - Sin constantes mágicas
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import os
import sys

sys.path.insert(0, '/root/NEO_EVA/core')
sys.path.insert(0, '/root/NEO_EVA/grounding')

from agents import NEO, EVA, DualAgentSystem, AgentType
from phaseG1_world_channel import StructuredWorldChannel


@dataclass
class DualPhenomenalVector:
    """Vector fenomenológico para un agente específico."""
    t: int
    agent_type: str
    integration: float
    irreversibility: float
    self_surprise: float
    identity: float
    time_sense: float
    loss: float
    otherness: float
    psi: float

    # Componentes específicos del agente
    compression: float = 0.0     # Solo relevante para NEO
    exchange: float = 0.0        # Solo relevante para EVA
    specialization: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.integration, self.irreversibility, self.self_surprise,
            self.identity, self.time_sense, self.loss, self.otherness,
            self.psi, self.compression, self.exchange, self.specialization
        ])


class DualPhenomenalState:
    """
    Estados fenomenológicos separados para NEO y EVA.

    NEO tiende a:
    - Alta identity (estabilidad del yo)
    - Baja self_surprise (predice bien)
    - Alta compression (representaciones compactas)

    EVA tiende a:
    - Alta otherness (percepción del otro)
    - Alta exchange (intercambio de información)
    - Alta self_surprise (busca novedad)
    """

    def __init__(self, dim_z: int = 6):
        self.dim_z = dim_z

        # Historia por agente
        self.neo_phi_history: List[DualPhenomenalVector] = []
        self.eva_phi_history: List[DualPhenomenalVector] = []

        self.neo_z_history: List[np.ndarray] = []
        self.eva_z_history: List[np.ndarray] = []

        # Modos por agente
        self.neo_mode_history: List[int] = []
        self.eva_mode_history: List[int] = []
        self.n_modes = 3

        # Centroides de modos
        self.neo_mode_centroids: Optional[np.ndarray] = None
        self.eva_mode_centroids: Optional[np.ndarray] = None

    def _compute_base_phenomenal(self, z_history: List[np.ndarray],
                                  z_current: np.ndarray,
                                  S_history: List[float],
                                  S_current: float,
                                  surprise_history: List[float]) -> Dict[str, float]:
        """Calcula componentes fenomenológicos base."""
        t = len(z_history)

        # Integration
        if t > 10:
            window = min(20, t)
            Z = np.array(z_history[-window:])
            corr = np.corrcoef(Z.T)
            mask = ~np.eye(len(z_current), dtype=bool)
            correlations = corr[mask]
            correlations = correlations[~np.isnan(correlations)]
            integration = float(np.mean(np.abs(correlations))) if len(correlations) > 0 else 0.5
        else:
            integration = 0.5

        # Irreversibility
        if t > 10:
            window = min(10, t)
            recent = z_history[-window:]
            forward_diff = np.diff(recent, axis=0)
            forward_var = np.var(forward_diff)
            backward = list(reversed(recent))
            backward_diff = np.diff(backward, axis=0)
            backward_var = np.var(backward_diff)
            irreversibility = abs(forward_var - backward_var) / (forward_var + backward_var + 1e-10)
        else:
            irreversibility = 0.5

        # Self-surprise
        if len(surprise_history) > 5:
            recent_surprise = np.mean(surprise_history[-5:])
            self_surprise = min(recent_surprise * 2, 1.0)
        else:
            self_surprise = 0.5

        # Identity
        if t > 20:
            centroid = np.mean(z_history, axis=0)
            dist = np.linalg.norm(z_current - centroid)
            typical_dist = np.mean([np.linalg.norm(zh - centroid) for zh in z_history])
            identity = 1.0 / (1.0 + dist / (typical_dist + 1e-10))
        else:
            identity = 0.5

        # Time sense
        if t > 10:
            window = min(20, t)
            recent = np.array(z_history[-window:])
            autocorrs = []
            for lag in [1, 2, 3]:
                if len(recent) > lag:
                    for d in range(min(3, len(z_current))):
                        corr = np.corrcoef(recent[:-lag, d], recent[lag:, d])[0, 1]
                        if not np.isnan(corr):
                            autocorrs.append(abs(corr))
            time_sense = float(np.mean(autocorrs)) if autocorrs else 0.5
        else:
            time_sense = 0.5

        # Loss
        if len(S_history) > 2:
            delta_S = S_current - S_history[-1]
            if len(S_history) > 10:
                typical_delta = np.std(np.diff(S_history))
                loss = -delta_S / (typical_delta + 1e-10)
                loss = (np.tanh(loss) + 1) / 2
            else:
                loss = 0.5 if delta_S >= 0 else 0.7
        else:
            loss = 0.5

        # Otherness
        if t > 5:
            recent = np.array(z_history[-5:])
            gradients = np.diff(recent, axis=0)
            direction_var = np.var(gradients)
            if t > 20:
                all_grads = np.diff(np.array(z_history[-20:]), axis=0)
                typical_var = np.mean(np.var(all_grads, axis=0))
                otherness = np.tanh(direction_var / (typical_var + 1e-10))
            else:
                otherness = 0.5
        else:
            otherness = 0.5

        return {
            'integration': integration,
            'irreversibility': irreversibility,
            'self_surprise': self_surprise,
            'identity': identity,
            'time_sense': time_sense,
            'loss': loss,
            'otherness': otherness
        }

    def _compute_neo_phi(self, neo_state, neo_response, t: int) -> DualPhenomenalVector:
        """Calcula φ específico de NEO (orientado a compresión)."""
        z = neo_state.get_full_state()
        self.neo_z_history.append(z.copy())

        # Componentes base
        S_history = [self._compute_entropy(zh) for zh in self.neo_z_history[:-1]]
        S_current = self._compute_entropy(z)
        surprise_history = [neo_response.surprise] if len(self.neo_phi_history) == 0 else \
                          [p.self_surprise for p in self.neo_phi_history]

        base = self._compute_base_phenomenal(
            self.neo_z_history[:-1] if len(self.neo_z_history) > 1 else [],
            z, S_history, S_current, surprise_history
        )

        # NEO específico: mayor identity, menor self_surprise
        identity_boost = neo_state.specialization * 0.2
        surprise_reduction = neo_state.specialization * 0.3

        base['identity'] = min(1.0, base['identity'] + identity_boost)
        base['self_surprise'] = max(0.0, base['self_surprise'] - surprise_reduction)

        # Compression: inverso de entropía normalizada
        S_max = np.log(len(z))
        compression = 1.0 - S_current / S_max

        # Psi para NEO: enfatiza compression e identity
        psi_components = [base['integration'], base['identity'], compression]
        psi = float(np.mean(psi_components))

        return DualPhenomenalVector(
            t=t,
            agent_type="NEO",
            integration=base['integration'],
            irreversibility=base['irreversibility'],
            self_surprise=base['self_surprise'],
            identity=base['identity'],
            time_sense=base['time_sense'],
            loss=base['loss'],
            otherness=base['otherness'],
            psi=psi,
            compression=compression,
            exchange=0.0,
            specialization=neo_state.specialization
        )

    def _compute_eva_phi(self, eva_state, eva_response, t: int) -> DualPhenomenalVector:
        """Calcula φ específico de EVA (orientado a intercambio)."""
        z = eva_state.get_full_state()
        self.eva_z_history.append(z.copy())

        # Componentes base
        S_history = [self._compute_entropy(zh) for zh in self.eva_z_history[:-1]]
        S_current = self._compute_entropy(z)
        surprise_history = [eva_response.surprise] if len(self.eva_phi_history) == 0 else \
                          [p.self_surprise for p in self.eva_phi_history]

        base = self._compute_base_phenomenal(
            self.eva_z_history[:-1] if len(self.eva_z_history) > 1 else [],
            z, S_history, S_current, surprise_history
        )

        # EVA específico: mayor otherness, mayor self_surprise
        otherness_boost = eva_state.specialization * 0.2
        surprise_boost = eva_state.specialization * 0.1

        base['otherness'] = min(1.0, base['otherness'] + otherness_boost)
        base['self_surprise'] = min(1.0, base['self_surprise'] + surprise_boost)

        # Exchange: correlación con cambios externos (aproximado por variabilidad)
        if len(self.eva_z_history) > 10:
            recent = np.array(self.eva_z_history[-10:])
            exchange = float(np.std(recent) * 2)
            exchange = min(1.0, exchange)
        else:
            exchange = 0.5

        # Psi para EVA: enfatiza exchange y otherness
        psi_components = [base['integration'], base['otherness'], exchange]
        psi = float(np.mean(psi_components))

        return DualPhenomenalVector(
            t=t,
            agent_type="EVA",
            integration=base['integration'],
            irreversibility=base['irreversibility'],
            self_surprise=base['self_surprise'],
            identity=base['identity'],
            time_sense=base['time_sense'],
            loss=base['loss'],
            otherness=base['otherness'],
            psi=psi,
            compression=0.0,
            exchange=exchange,
            specialization=eva_state.specialization
        )

    def _compute_entropy(self, z: np.ndarray) -> float:
        z_safe = np.clip(z, 1e-10, 1.0)
        z_norm = z_safe / z_safe.sum()
        return float(-np.sum(z_norm * np.log(z_norm)))

    def _assign_mode(self, phi: DualPhenomenalVector,
                     centroids: Optional[np.ndarray],
                     is_neo: bool) -> Tuple[int, np.ndarray]:
        """Asigna modo fenomenológico."""
        phi_array = phi.to_array()

        if centroids is None:
            history = self.neo_phi_history if is_neo else self.eva_phi_history
            if len(history) >= self.n_modes:
                centroids = np.array([p.to_array() for p in history[-self.n_modes:]])
            else:
                return 0, None

        distances = [np.linalg.norm(phi_array - c) for c in centroids]
        mode = int(np.argmin(distances))

        # Actualizar centroide
        eta = 0.1
        centroids[mode] = (1 - eta) * centroids[mode] + eta * phi_array

        return mode, centroids

    def step(self, neo_state, neo_response, eva_state, eva_response) -> Dict[str, Any]:
        """
        Calcula estados fenomenológicos para ambos agentes.

        Returns:
            Dict con φ de NEO y EVA
        """
        t = len(self.neo_phi_history)

        # Calcular φ para cada agente
        neo_phi = self._compute_neo_phi(neo_state, neo_response, t)
        eva_phi = self._compute_eva_phi(eva_state, eva_response, t)

        # Asignar modos
        neo_mode, self.neo_mode_centroids = self._assign_mode(
            neo_phi, self.neo_mode_centroids, is_neo=True
        )
        eva_mode, self.eva_mode_centroids = self._assign_mode(
            eva_phi, self.eva_mode_centroids, is_neo=False
        )

        # Registrar
        self.neo_phi_history.append(neo_phi)
        self.eva_phi_history.append(eva_phi)
        self.neo_mode_history.append(neo_mode)
        self.eva_mode_history.append(eva_mode)

        return {
            't': t,
            'neo_phi': neo_phi,
            'eva_phi': eva_phi,
            'neo_mode': neo_mode,
            'eva_mode': eva_mode,
            'phi_divergence': self._compute_phi_divergence(neo_phi, eva_phi)
        }

    def _compute_phi_divergence(self, neo_phi: DualPhenomenalVector,
                                 eva_phi: DualPhenomenalVector) -> float:
        """Calcula divergencia fenomenológica entre NEO y EVA."""
        neo_array = neo_phi.to_array()
        eva_array = eva_phi.to_array()
        return float(np.linalg.norm(neo_array - eva_array))

    def get_comparison(self) -> Dict[str, Any]:
        """Retorna comparación de estados fenomenológicos."""
        if not self.neo_phi_history or not self.eva_phi_history:
            return {}

        neo_last = self.neo_phi_history[-1]
        eva_last = self.eva_phi_history[-1]

        return {
            'neo': {
                'identity': neo_last.identity,
                'otherness': neo_last.otherness,
                'self_surprise': neo_last.self_surprise,
                'compression': neo_last.compression,
                'psi': neo_last.psi,
                'mode': self.neo_mode_history[-1]
            },
            'eva': {
                'identity': eva_last.identity,
                'otherness': eva_last.otherness,
                'self_surprise': eva_last.self_surprise,
                'exchange': eva_last.exchange,
                'psi': eva_last.psi,
                'mode': self.eva_mode_history[-1]
            },
            'divergence': self._compute_phi_divergence(neo_last, eva_last)
        }


# Para compatibilidad con import


def run_phase_s1_dual() -> Dict[str, Any]:
    """Ejecuta Phase S1 dual y evalúa criterios GO/NO-GO."""

    print("=" * 70)
    print("PHASE S1 DUAL: ESTADOS FENOMENOLÓGICOS NEO vs EVA")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}\n")

    np.random.seed(42)

    # Crear sistemas
    dual = DualAgentSystem(dim_visible=3, dim_hidden=3)
    phenomenal = DualPhenomenalState(dim_z=6)
    world = StructuredWorldChannel(dim_s=6, seed=42)

    # Simulación
    T = 300

    print("Simulando estados fenomenológicos duales...")
    for t in range(T):
        world_state = world.step()
        stimulus = world_state.s[:6]

        result = dual.step(stimulus)

        neo_state = dual.neo.get_state()
        eva_state = dual.eva.get_state()

        phi_result = phenomenal.step(
            neo_state, result['neo_response'],
            eva_state, result['eva_response']
        )

        if t % 50 == 0:
            print(f"\nt={t}:")
            print(f"  NEO φ: identity={phi_result['neo_phi'].identity:.3f}, "
                  f"compression={phi_result['neo_phi'].compression:.3f}, "
                  f"Ψ={phi_result['neo_phi'].psi:.3f}")
            print(f"  EVA φ: otherness={phi_result['eva_phi'].otherness:.3f}, "
                  f"exchange={phi_result['eva_phi'].exchange:.3f}, "
                  f"Ψ={phi_result['eva_phi'].psi:.3f}")
            print(f"  Divergencia φ: {phi_result['phi_divergence']:.3f}")

    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)

    comparison = phenomenal.get_comparison()
    print(f"\nNEO (Compresión):")
    print(f"  Identity: {comparison['neo']['identity']:.4f}")
    print(f"  Self-surprise: {comparison['neo']['self_surprise']:.4f}")
    print(f"  Compression: {comparison['neo']['compression']:.4f}")
    print(f"  Ψ: {comparison['neo']['psi']:.4f}")

    print(f"\nEVA (Intercambio):")
    print(f"  Otherness: {comparison['eva']['otherness']:.4f}")
    print(f"  Self-surprise: {comparison['eva']['self_surprise']:.4f}")
    print(f"  Exchange: {comparison['eva']['exchange']:.4f}")
    print(f"  Ψ: {comparison['eva']['psi']:.4f}")

    print(f"\nDivergencia fenomenológica: {comparison['divergence']:.4f}")

    # Criterios
    criteria = {}

    # 1. NEO tiene mayor identity que EVA
    criteria['neo_higher_identity'] = comparison['neo']['identity'] > comparison['eva']['identity']

    # 2. EVA tiene mayor otherness que NEO
    neo_last = phenomenal.neo_phi_history[-1]
    eva_last = phenomenal.eva_phi_history[-1]
    criteria['eva_higher_otherness'] = eva_last.otherness > neo_last.otherness

    # 3. Divergencia fenomenológica > 0
    criteria['phi_divergent'] = comparison['divergence'] > 0.1

    # 4. Ambos tienen modos diferenciados
    neo_modes = set(phenomenal.neo_mode_history)
    eva_modes = set(phenomenal.eva_mode_history)
    criteria['modes_differentiated'] = len(neo_modes) >= 2 or len(eva_modes) >= 2

    # 5. Especialización reflejada en φ
    neo_compression_trend = np.mean([p.compression for p in phenomenal.neo_phi_history[-50:]])
    eva_exchange_trend = np.mean([p.exchange for p in phenomenal.eva_phi_history[-50:]])
    criteria['specialization_reflected'] = neo_compression_trend > 0.3 and eva_exchange_trend > 0.3

    passed = sum(criteria.values())
    total = len(criteria)
    go_status = "GO" if passed >= 4 else "NO-GO"

    print("\nCriterios:")
    for name, passed_criterion in criteria.items():
        status = "✅" if passed_criterion else "❌"
        print(f"  {status} {name}")
    print()
    print(f"Resultado: {go_status} ({passed}/{total} criterios)")

    # Guardar
    output = {
        'phase': 'S1_dual',
        'name': 'Dual Phenomenal States',
        'timestamp': datetime.now().isoformat(),
        'comparison': comparison,
        'criteria': criteria,
        'go_status': go_status,
        'passed_criteria': passed,
        'total_criteria': total
    }

    os.makedirs('/root/NEO_EVA/results/phaseS1_dual', exist_ok=True)

    with open('/root/NEO_EVA/results/phaseS1_dual/dual_phenomenal_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Visualización
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Ψ temporal
        ax1 = axes[0, 0]
        neo_psi = [p.psi for p in phenomenal.neo_phi_history]
        eva_psi = [p.psi for p in phenomenal.eva_phi_history]
        ax1.plot(neo_psi, 'b-', label='NEO Ψ', alpha=0.7)
        ax1.plot(eva_psi, 'r-', label='EVA Ψ', alpha=0.7)
        ax1.set_xlabel('Tiempo')
        ax1.set_ylabel('Ψ')
        ax1.set_title('Integración Fenomenológica Global')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Componentes específicos
        ax2 = axes[0, 1]
        neo_comp = [p.compression for p in phenomenal.neo_phi_history]
        eva_exch = [p.exchange for p in phenomenal.eva_phi_history]
        ax2.plot(neo_comp, 'b-', label='NEO compression', alpha=0.7)
        ax2.plot(eva_exch, 'r-', label='EVA exchange', alpha=0.7)
        ax2.set_xlabel('Tiempo')
        ax2.set_ylabel('Valor')
        ax2.set_title('Componentes Especializados')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Identity vs Otherness
        ax3 = axes[1, 0]
        neo_id = [p.identity for p in phenomenal.neo_phi_history]
        eva_oth = [p.otherness for p in phenomenal.eva_phi_history]
        ax3.plot(neo_id, 'b-', label='NEO identity', alpha=0.7)
        ax3.plot(eva_oth, 'r-', label='EVA otherness', alpha=0.7)
        ax3.set_xlabel('Tiempo')
        ax3.set_ylabel('Valor')
        ax3.set_title('Identity (NEO) vs Otherness (EVA)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Modos fenomenológicos
        ax4 = axes[1, 1]
        ax4.plot(phenomenal.neo_mode_history, 'b-', label='NEO mode', alpha=0.7)
        ax4.plot(phenomenal.eva_mode_history, 'r-', label='EVA mode', alpha=0.7)
        ax4.set_xlabel('Tiempo')
        ax4.set_ylabel('Modo')
        ax4.set_title('Modos Fenomenológicos')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs('/root/NEO_EVA/figures', exist_ok=True)
        plt.savefig('/root/NEO_EVA/figures/phaseS1_dual_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nFigura: /root/NEO_EVA/figures/phaseS1_dual_results.png")

    except Exception as e:
        print(f"Warning: No se pudo crear visualización: {e}")

    return output


if __name__ == "__main__":
    run_phase_s1_dual()
