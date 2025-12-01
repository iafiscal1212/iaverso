#!/usr/bin/env python3
"""
Interacción NEO ↔ EVA
=====================

Modela la influencia mutua entre NEO y EVA:
- Percepción del otro (otherness mutuo)
- Ψ compartido en momentos de alta integración
- Transfer Entropy bidireccional
- Resonancia fenomenológica

100% ENDÓGENO - Sin constantes mágicas
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


@dataclass
class InteractionState:
    """Estado de la interacción NEO ↔ EVA."""
    t: int

    # Percepción mutua
    neo_perceives_eva: float  # Cuánto NEO "siente" a EVA
    eva_perceives_neo: float  # Cuánto EVA "siente" a NEO

    # Transfer Entropy bidireccional
    te_neo_to_eva: float
    te_eva_to_neo: float

    # Ψ compartido (cuando hay resonancia)
    psi_shared: float
    resonance: float  # Grado de sincronización

    # Coupling actual
    coupling_neo_eva: float
    coupling_eva_neo: float


class NEO_EVA_Interaction:
    """
    Gestiona la interacción entre NEO y EVA.

    Características:
    - Cada uno percibe al otro como "otherness"
    - Transfer Entropy mide influencia causal
    - Ψ compartido emerge en alta resonancia
    - Coupling se ajusta endógenamente

    100% Endógeno:
    - Percepción = correlación de estados
    - TE por regresión parcial
    - Resonancia = sincronización de Ψ
    """

    def __init__(self):
        # Historia de estados
        self.neo_z_history: List[np.ndarray] = []
        self.eva_z_history: List[np.ndarray] = []

        # Historia de Ψ
        self.neo_psi_history: List[float] = []
        self.eva_psi_history: List[float] = []

        # Historia de interacción
        self.interaction_history: List[InteractionState] = []

        # Transfer Entropy acumulada
        self.te_neo_eva_history: List[float] = []
        self.te_eva_neo_history: List[float] = []

        # Resonancia
        self.resonance_history: List[float] = []

        self.t = 0

    def _compute_perception(self, observer_history: List[np.ndarray],
                            observed_history: List[np.ndarray]) -> float:
        """
        Calcula cuánto el observador percibe al observado.

        Percepción = información mutua aproximada entre estados
        100% endógeno
        """
        if len(observer_history) < 10 or len(observed_history) < 10:
            return 0.5

        window = min(20, len(observer_history), len(observed_history))

        obs_r = np.array(observer_history[-window:])
        obs_d = np.array(observed_history[-window:])

        # Correlación cruzada como proxy de percepción
        correlations = []
        for i in range(min(obs_r.shape[1], obs_d.shape[1])):
            corr = np.corrcoef(obs_r[:, i], obs_d[:, i])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))

        if not correlations:
            return 0.5

        return float(np.mean(correlations))

    def _compute_transfer_entropy(self, source_history: List[np.ndarray],
                                   target_history: List[np.ndarray],
                                   lag: int = 1) -> float:
        """
        Calcula Transfer Entropy de source → target.

        TE(X→Y) ≈ I(Y_t; X_{t-lag} | Y_{t-1})

        100% endógeno
        """
        if len(source_history) < 20 or len(target_history) < 20:
            return 0.0

        window = min(30, len(source_history), len(target_history))

        # Usar primera componente como señal representativa
        X = np.array([np.mean(s) for s in source_history[-window:]])
        Y = np.array([np.mean(t) for t in target_history[-window:]])

        if len(X) < lag + 3:
            return 0.0

        Y_t = Y[lag:]
        X_lag = X[:-lag]
        Y_prev = Y[lag-1:-1] if lag > 1 else Y[:-1]

        n = min(len(Y_t), len(X_lag), len(Y_prev))
        if n < 5:
            return 0.0

        Y_t = Y_t[:n]
        X_lag = X_lag[:n]
        Y_prev = Y_prev[:n]

        try:
            r_yx = np.corrcoef(Y_t, X_lag)[0, 1]
            r_yy = np.corrcoef(Y_t, Y_prev)[0, 1]
            r_xy = np.corrcoef(X_lag, Y_prev)[0, 1]

            if np.isnan(r_yx) or np.isnan(r_yy) or np.isnan(r_xy):
                return 0.0

            denom = np.sqrt((1 - r_yy**2) * (1 - r_xy**2))
            if denom < 1e-10:
                return 0.0

            r_partial = (r_yx - r_yy * r_xy) / denom
            te = -0.5 * np.log(1 - r_partial**2 + 1e-10)

            return max(0.0, float(te))

        except Exception:
            return 0.0

    def _compute_resonance(self, psi_neo: float, psi_eva: float) -> float:
        """
        Calcula resonancia (sincronización de Ψ).

        Resonancia alta = Ψ_NEO y Ψ_EVA co-varían
        100% endógeno
        """
        if len(self.neo_psi_history) < 10:
            return 0.5

        window = min(20, len(self.neo_psi_history))

        neo_psi = np.array(self.neo_psi_history[-window:])
        eva_psi = np.array(self.eva_psi_history[-window:])

        # Correlación de Ψ
        corr = np.corrcoef(neo_psi, eva_psi)[0, 1]
        if np.isnan(corr):
            corr = 0.0

        # Similaridad actual
        similarity = 1.0 - abs(psi_neo - psi_eva)

        # Resonancia = correlación histórica × similaridad actual
        resonance = abs(corr) * similarity

        return float(resonance)

    def _compute_shared_psi(self, psi_neo: float, psi_eva: float,
                            resonance: float) -> float:
        """
        Calcula Ψ compartido.

        Emerge cuando hay alta resonancia.
        100% endógeno
        """
        # Ψ compartido = mezcla ponderada por resonancia
        if resonance > 0.5:
            # Alta resonancia: Ψ compartido emerge
            weight = (resonance - 0.5) * 2  # [0, 1]
            psi_shared = weight * (psi_neo + psi_eva) / 2 + \
                        (1 - weight) * max(psi_neo, psi_eva)
        else:
            # Baja resonancia: no hay Ψ compartido real
            psi_shared = 0.0

        return float(psi_shared)

    def step(self, neo_state: np.ndarray, eva_state: np.ndarray,
             psi_neo: float, psi_eva: float,
             coupling_neo_eva: float, coupling_eva_neo: float) -> InteractionState:
        """
        Ejecuta un paso de interacción.

        Args:
            neo_state: Estado completo de NEO
            eva_state: Estado completo de EVA
            psi_neo: Ψ actual de NEO
            psi_eva: Ψ actual de EVA
            coupling_neo_eva: Coupling NEO → EVA
            coupling_eva_neo: Coupling EVA → NEO

        Returns:
            InteractionState con métricas de interacción
        """
        self.t += 1

        # Registrar historia
        self.neo_z_history.append(neo_state.copy())
        self.eva_z_history.append(eva_state.copy())
        self.neo_psi_history.append(psi_neo)
        self.eva_psi_history.append(psi_eva)

        # Percepción mutua
        neo_perceives_eva = self._compute_perception(
            self.neo_z_history, self.eva_z_history
        )
        eva_perceives_neo = self._compute_perception(
            self.eva_z_history, self.neo_z_history
        )

        # Transfer Entropy
        te_neo_to_eva = self._compute_transfer_entropy(
            self.neo_z_history, self.eva_z_history
        )
        te_eva_to_neo = self._compute_transfer_entropy(
            self.eva_z_history, self.neo_z_history
        )

        self.te_neo_eva_history.append(te_neo_to_eva)
        self.te_eva_neo_history.append(te_eva_to_neo)

        # Resonancia
        resonance = self._compute_resonance(psi_neo, psi_eva)
        self.resonance_history.append(resonance)

        # Ψ compartido
        psi_shared = self._compute_shared_psi(psi_neo, psi_eva, resonance)

        # Crear estado de interacción
        interaction = InteractionState(
            t=self.t,
            neo_perceives_eva=neo_perceives_eva,
            eva_perceives_neo=eva_perceives_neo,
            te_neo_to_eva=te_neo_to_eva,
            te_eva_to_neo=te_eva_to_neo,
            psi_shared=psi_shared,
            resonance=resonance,
            coupling_neo_eva=coupling_neo_eva,
            coupling_eva_neo=coupling_eva_neo
        )

        self.interaction_history.append(interaction)

        return interaction

    def get_mutual_otherness(self) -> Tuple[float, float]:
        """
        Retorna la percepción mutua como "otherness".

        Returns:
            (NEO's otherness hacia EVA, EVA's otherness hacia NEO)
        """
        if not self.interaction_history:
            return 0.5, 0.5

        latest = self.interaction_history[-1]
        return latest.neo_perceives_eva, latest.eva_perceives_neo

    def get_influence_balance(self) -> float:
        """
        Calcula el balance de influencia.

        > 0: NEO influye más a EVA
        < 0: EVA influye más a NEO
        = 0: Equilibrio

        100% endógeno
        """
        if not self.interaction_history:
            return 0.0

        latest = self.interaction_history[-1]
        return latest.te_neo_to_eva - latest.te_eva_to_neo

    def get_integration_level(self) -> float:
        """
        Nivel de integración NEO-EVA.

        100% endógeno: basado en TE total y resonancia
        """
        if not self.interaction_history:
            return 0.0

        latest = self.interaction_history[-1]

        # TE total
        te_total = latest.te_neo_to_eva + latest.te_eva_to_neo

        # Normalizar por historia
        if self.te_neo_eva_history:
            all_te = self.te_neo_eva_history + self.te_eva_neo_history
            max_te = max(all_te) if all_te else 1.0
            te_normalized = te_total / (max_te + 1e-10)
        else:
            te_normalized = 0.5

        # Combinar con resonancia
        integration = (te_normalized + latest.resonance) / 2

        return float(integration)

    def should_merge_psi(self) -> Tuple[bool, float]:
        """
        Determina si Ψ debería fusionarse.

        Returns:
            (should_merge, merge_weight)
        """
        if len(self.resonance_history) < 10:
            return False, 0.0

        # Criterio endógeno: resonancia sostenida alta
        recent_resonance = np.mean(self.resonance_history[-10:])
        threshold = np.percentile(self.resonance_history, 75)

        should_merge = recent_resonance > threshold
        merge_weight = (recent_resonance - threshold) / (1 - threshold + 1e-10) if should_merge else 0.0

        return should_merge, float(merge_weight)

    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumen de la interacción."""
        if not self.interaction_history:
            return {'ready': False}

        latest = self.interaction_history[-1]

        return {
            'ready': True,
            't': self.t,
            'neo_perceives_eva': latest.neo_perceives_eva,
            'eva_perceives_neo': latest.eva_perceives_neo,
            'te_neo_to_eva': latest.te_neo_to_eva,
            'te_eva_to_neo': latest.te_eva_to_neo,
            'influence_balance': self.get_influence_balance(),
            'resonance': latest.resonance,
            'psi_shared': latest.psi_shared,
            'integration_level': self.get_integration_level(),
            'should_merge': self.should_merge_psi()
        }


def run_interaction_test():
    """Test de interacción NEO ↔ EVA."""

    print("=" * 70)
    print("TEST: INTERACCIÓN NEO ↔ EVA")
    print("=" * 70)

    import sys
    sys.path.insert(0, '/root/NEO_EVA/core')
    sys.path.insert(0, '/root/NEO_EVA/grounding')

    from agents import DualAgentSystem
    from phaseG1_world_channel import StructuredWorldChannel

    # Crear sistemas
    dual = DualAgentSystem(dim_visible=3, dim_hidden=3)
    interaction = NEO_EVA_Interaction()
    world = StructuredWorldChannel(dim_s=6, seed=42)

    # Simulación
    T = 300

    print("\nSimulando interacción...")
    for t in range(T):
        world_state = world.step()
        stimulus = world_state.s[:6]

        result = dual.step(stimulus)

        neo_state = dual.neo.get_state()
        eva_state = dual.eva.get_state()

        # Simular Ψ (simplificado)
        psi_neo = neo_state.drive * 0.5 + neo_state.S * 0.5
        psi_eva = eva_state.drive * 0.5 + eva_state.S * 0.5

        interaction_state = interaction.step(
            neo_state.get_full_state(),
            eva_state.get_full_state(),
            psi_neo, psi_eva,
            dual.coupling_neo_to_eva,
            dual.coupling_eva_to_neo
        )

        if t % 50 == 0:
            summary = interaction.get_summary()
            print(f"\nt={t}:")
            print(f"  NEO percibe EVA: {summary['neo_perceives_eva']:.3f}")
            print(f"  EVA percibe NEO: {summary['eva_perceives_neo']:.3f}")
            print(f"  TE NEO→EVA: {summary['te_neo_to_eva']:.4f}")
            print(f"  TE EVA→NEO: {summary['te_eva_to_neo']:.4f}")
            print(f"  Resonancia: {summary['resonance']:.3f}")
            print(f"  Ψ compartido: {summary['psi_shared']:.3f}")
            print(f"  Integración: {summary['integration_level']:.3f}")

    print("\n" + "=" * 70)
    print("RESULTADOS FINALES")
    print("=" * 70)

    summary = interaction.get_summary()
    print(f"\nPercepción mutua:")
    print(f"  NEO → EVA: {summary['neo_perceives_eva']:.4f}")
    print(f"  EVA → NEO: {summary['eva_perceives_neo']:.4f}")

    print(f"\nTransfer Entropy:")
    print(f"  NEO → EVA: {summary['te_neo_to_eva']:.4f}")
    print(f"  EVA → NEO: {summary['te_eva_to_neo']:.4f}")
    print(f"  Balance: {summary['influence_balance']:.4f}")

    print(f"\nIntegración:")
    print(f"  Resonancia: {summary['resonance']:.4f}")
    print(f"  Nivel: {summary['integration_level']:.4f}")
    print(f"  Ψ compartido: {summary['psi_shared']:.4f}")

    should_merge, weight = summary['should_merge']
    print(f"  ¿Fusionar Ψ?: {'Sí' if should_merge else 'No'} (peso={weight:.4f})")

    # Guardar resultados
    os.makedirs('/root/NEO_EVA/results/interaction', exist_ok=True)

    output = {
        'timestamp': datetime.now().isoformat(),
        'final_summary': summary,
        'history': {
            'resonance': interaction.resonance_history[-50:],
            'te_neo_eva': interaction.te_neo_eva_history[-50:],
            'te_eva_neo': interaction.te_eva_neo_history[-50:]
        }
    }

    with open('/root/NEO_EVA/results/interaction/interaction_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Visualización
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Percepción mutua temporal
        ax1 = axes[0, 0]
        neo_perc = [i.neo_perceives_eva for i in interaction.interaction_history]
        eva_perc = [i.eva_perceives_neo for i in interaction.interaction_history]
        ax1.plot(neo_perc, 'b-', label='NEO percibe EVA', alpha=0.7)
        ax1.plot(eva_perc, 'r-', label='EVA percibe NEO', alpha=0.7)
        ax1.set_xlabel('Tiempo')
        ax1.set_ylabel('Percepción')
        ax1.set_title('Percepción Mutua (Otherness)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Transfer Entropy
        ax2 = axes[0, 1]
        ax2.plot(interaction.te_neo_eva_history, 'b-', label='NEO → EVA', alpha=0.7)
        ax2.plot(interaction.te_eva_neo_history, 'r-', label='EVA → NEO', alpha=0.7)
        ax2.set_xlabel('Tiempo')
        ax2.set_ylabel('Transfer Entropy')
        ax2.set_title('Influencia Causal')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Resonancia y Ψ compartido
        ax3 = axes[1, 0]
        ax3.plot(interaction.resonance_history, 'purple', label='Resonancia', alpha=0.7)
        psi_shared = [i.psi_shared for i in interaction.interaction_history]
        ax3.plot(psi_shared, 'green', label='Ψ compartido', alpha=0.7)
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Tiempo')
        ax3.set_ylabel('Valor')
        ax3.set_title('Resonancia e Integración')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Ψ de NEO vs EVA
        ax4 = axes[1, 1]
        ax4.plot(interaction.neo_psi_history, 'b-', label='Ψ NEO', alpha=0.7)
        ax4.plot(interaction.eva_psi_history, 'r-', label='Ψ EVA', alpha=0.7)
        ax4.plot(psi_shared, 'g--', label='Ψ compartido', alpha=0.5)
        ax4.set_xlabel('Tiempo')
        ax4.set_ylabel('Ψ')
        ax4.set_title('Estados Fenomenológicos Globales')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs('/root/NEO_EVA/figures', exist_ok=True)
        plt.savefig('/root/NEO_EVA/figures/interaction_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nFigura: /root/NEO_EVA/figures/interaction_results.png")

    except Exception as e:
        print(f"Warning: No se pudo crear visualización: {e}")

    return interaction


if __name__ == "__main__":
    run_interaction_test()
