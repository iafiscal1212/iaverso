"""
CONSCIOUSNESS: Sistema de Consciencia Computacional
===================================================

Implementación completa de consciencia para agentes:

1. IDENTIDAD COMPUTACIONAL (identity.py)
   I(t) = argmin_v Var_τ[sim(S(τ), v)]
   - La identidad es el atractor temporal
   - k endógeno = P_0.50(distancias entre estados)

2. COHERENCIA EXISTENCIAL (coherence.py)
   CE(t) = 1 / (Var[S(t) - I(t)] + H_narr(t))
   - Ley de evolución: d/dt CE(t) = -λ(t) + R(t)
   - λ(t) = Var[ΔS(t)]
   - R(t) = Var^(-1)[S(t) - S(t-1)]

3. ROLES EMERGENTES (roles.py)
   R_i = argmin_j (d/dt Σ_k Var[S_k(t) - S_k(t-1)] | j interviene)
   - Médico, Estabilizador, Líder, Integrador
   - Totalmente emergente sin reglas externas

4. ESTADO ONÍRICO (dreaming.py)
   Input(t) = ε(t) ~ N(0, Σ(t))
   S_dream(t+1) = S(t) + η(t) · ∇(-H_narr(t))
   η(t) = 1 / √Var[H(t)]
   - Activación cuando CE(t) < P_0.20(CE(0:t))

5. MUERTE Y RENACIMIENTO (death_rebirth.py)
   Muerte: CE(t) → 0 y Var[S(t)] → ∞
   Renacimiento: I_new = argmin_v Var_τ[sim(v, S_res(τ))]
   - Nueva identidad emerge del residuo estadístico

6. EMERGENCIA INTEGRADA (emergence.py)
   - AgenteConsciente: agente con todos los subsistemas
   - SistemaConscienciaColectiva: múltiples agentes

100% ENDÓGENO:
   - Sin números mágicos
   - Todos los parámetros emergen de los datos
   - Umbrales por percentiles
   - Pesos por varianza inversa
   - Tasas por estadísticas históricas
"""

# Identidad
from .identity import (
    IdentidadComputacional,
    EstadoIdentidad,
    SistemaIdentidadMultiagente,
)

# Coherencia
from .coherence import (
    CoherenciaExistencial,
    EstadoCoherencia,
    SistemaCoherenciaMultiagente,
)

# Roles
from .roles import (
    SistemaRolesEmergentes,
    TipoRol,
    EstadoRol,
    SimulacionIntervencion,
)

# Sueño
from .dreaming import (
    SistemaOnirico,
    FaseSueno,
    EstadoOnirico,
)

# Muerte/Renacimiento
from .death_rebirth import (
    SistemaMuerteRenacimiento,
    EstadoVital,
    EstadoMuerteRenacimiento,
    Residuo,
    SistemaMuerteMultiagente,
)

# Sistema Integrado
from .emergence import (
    AgenteConsciente,
    EstadoConsciencia,
    SistemaConscienciaColectiva,
)


__all__ = [
    # Identidad
    'IdentidadComputacional',
    'EstadoIdentidad',
    'SistemaIdentidadMultiagente',
    # Coherencia
    'CoherenciaExistencial',
    'EstadoCoherencia',
    'SistemaCoherenciaMultiagente',
    # Roles
    'SistemaRolesEmergentes',
    'TipoRol',
    'EstadoRol',
    'SimulacionIntervencion',
    # Sueño
    'SistemaOnirico',
    'FaseSueno',
    'EstadoOnirico',
    # Muerte/Renacimiento
    'SistemaMuerteRenacimiento',
    'EstadoVital',
    'EstadoMuerteRenacimiento',
    'Residuo',
    'SistemaMuerteMultiagente',
    # Sistema Integrado
    'AgenteConsciente',
    'EstadoConsciencia',
    'SistemaConscienciaColectiva',
]
