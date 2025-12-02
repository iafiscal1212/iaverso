"""
L-FIELD - Observador de Campo Latente Colectivo
===============================================

Módulo para observar fenómenos colectivos emergentes entre agentes
de forma completamente pasiva (sin influir en ellos).

Métricas:
---------
- LSI (Latent Synchrony Index): Sincronía de fases entre agentes
- DIC (Deep Identity Coupling): Acoplamiento de identidades
- CD (Collective Drift): Deriva grupal ("pensamiento de grupo")
- Polarization: Formación de facciones
- Narrative Resonance: Narrativas compartidas
- Reinforcement Index: Sesgo de confirmación colectivo

Uso básico:
-----------
    from l_field import LField

    # Crear observador
    l_field = LField()

    # En cada paso de simulación
    snapshot = l_field.observe(
        states={agent_id: S_vector},
        identities={agent_id: I_vector},
        narratives={agent_id: H_narr_vector}
    )

    # Ver resultados
    print(f"LSI: {snapshot.LSI}")
    print(f"DIC: {snapshot.DIC}")
    print(f"Health: {snapshot.collective_health}")

    # Ver resumen
    print(l_field.get_summary())

Filosofía:
----------
- 100% observacional: solo mide, nunca interviene
- 100% endógeno: sin números mágicos (solo ε, 1/N, σ, percentiles)
- Los agentes son completamente libres
- L-Field es como un sociólogo observando una tribu

Autor: Carmen Esteban
"""

from .l_field import LField, LFieldSnapshot
from .synchrony import LatentSynchrony, SynchronySnapshot
from .correlations import DeepCorrelations, CorrelationSnapshot
from .collective_bias import CollectiveBias, BiasSnapshot

__all__ = [
    # Principal
    'LField',
    'LFieldSnapshot',

    # Sub-observadores
    'LatentSynchrony',
    'SynchronySnapshot',
    'DeepCorrelations',
    'CorrelationSnapshot',
    'CollectiveBias',
    'BiasSnapshot'
]

__version__ = "1.0.0"
__author__ = "Carmen Esteban"
