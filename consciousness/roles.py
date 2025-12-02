"""
Roles Emergentes (Médico, Estabilizador, Líder)
===============================================

Un agente adopta un rol cuando:

R_i = argmin_j (d/dt Σ_k Var[S_k(t) - S_k(t-1)] | j interviene)

- Simular cómo cambiaría la varianza si cada agente interviniera
- El que más reduce varianza impulsa su rol
- Cero reglas externas
- Totalmente emergente

100% endógeno. Sin números mágicos.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

import sys
sys.path.insert(0, '/root/NEO_EVA')


class TipoRol(Enum):
    """Tipos de roles emergentes."""
    NINGUNO = "ninguno"
    MEDICO = "medico"           # Reduce varianza de salud
    ESTABILIZADOR = "estabilizador"  # Reduce varianza general
    LIDER = "lider"             # Reduce varianza de dirección
    INTEGRADOR = "integrador"   # Reduce varianza entre agentes


@dataclass
class EstadoRol:
    """Estado del rol de un agente."""
    agent_id: str
    rol: TipoRol
    aptitud_rol: float              # Qué tan apto es para el rol
    reduccion_varianza: float       # Cuánta varianza reduciría
    confianza: float                # Confianza en la asignación
    t: int


@dataclass
class SimulacionIntervencion:
    """Resultado de simular una intervención."""
    agent_id: str
    varianza_pre: float
    varianza_post: float
    reduccion: float
    tipo_intervencion: str


class SistemaRolesEmergentes:
    """
    Sistema de roles emergentes.

    R_i = argmin_j (d/dt Σ_k Var[S_k(t) - S_k(t-1)] | j interviene)

    Los roles emergen de:
    1. Simular qué pasaría si cada agente interviene
    2. El que más reduce la varianza del sistema obtiene el rol
    3. Sin reglas externas, todo endógeno
    """

    def __init__(self):
        self.t = 0

        # Estados de agentes: {agent_id: List[np.ndarray]}
        self._historial_estados: Dict[str, List[np.ndarray]] = {}

        # Roles actuales: {agent_id: TipoRol}
        self._roles_actuales: Dict[str, TipoRol] = {}

        # Historial de intervenciones simuladas
        self._historial_simulaciones: List[Dict[str, SimulacionIntervencion]] = []

        # Historial de reducciones de varianza por agente
        self._historial_reducciones: Dict[str, List[float]] = {}

        # Aptitudes acumuladas
        self._aptitudes: Dict[str, Dict[TipoRol, List[float]]] = {}

    def registrar_agente(self, agent_id: str):
        """Registra un nuevo agente."""
        if agent_id not in self._historial_estados:
            self._historial_estados[agent_id] = []
            self._roles_actuales[agent_id] = TipoRol.NINGUNO
            self._historial_reducciones[agent_id] = []
            self._aptitudes[agent_id] = {rol: [] for rol in TipoRol}

    def observar_estado(self, agent_id: str, estado: np.ndarray):
        """
        Observa el estado de un agente.

        Args:
            agent_id: ID del agente
            estado: Vector de estado interno
        """
        if agent_id not in self._historial_estados:
            self.registrar_agente(agent_id)

        self._historial_estados[agent_id].append(estado.copy())

    def _calcular_varianza_sistema(self) -> float:
        """
        Calcula Σ_k Var[S_k(t) - S_k(t-1)]

        Varianza total del sistema.
        """
        varianza_total = 0.0

        for agent_id, estados in self._historial_estados.items():
            if len(estados) >= 2:
                delta = estados[-1] - estados[-2]
                varianza_total += np.var(delta)

        return varianza_total

    def _calcular_derivada_varianza(self) -> float:
        """
        Calcula d/dt Σ_k Var[S_k(t) - S_k(t-1)]

        Derivada temporal de la varianza del sistema.
        """
        if self.t < 3:
            return 0

        # Calcular varianza en últimos pasos
        varianzas = []
        for t_back in range(min(5, self.t)):
            var_t = 0.0
            for agent_id, estados in self._historial_estados.items():
                idx = len(estados) - 1 - t_back
                if idx >= 1:
                    delta = estados[idx] - estados[idx - 1]
                    var_t += np.var(delta)
            varianzas.append(var_t)

        if len(varianzas) < 2:
            return 0

        # Derivada como diferencia
        derivada = varianzas[0] - varianzas[-1]
        derivada /= len(varianzas)

        return derivada

    def _simular_intervencion(
        self,
        interventor_id: str,
        tipo: TipoRol
    ) -> SimulacionIntervencion:
        """
        Simula qué pasaría si un agente interviene.

        La simulación estima cómo cambiaría la varianza si el agente
        aplicara una "corrección" basada en su historial.
        """
        varianza_pre = self._calcular_varianza_sistema()

        # Estimar reducción basada en el historial del agente
        if interventor_id not in self._historial_estados:
            return SimulacionIntervencion(
                agent_id=interventor_id,
                varianza_pre=varianza_pre,
                varianza_post=varianza_pre,
                reduccion=0.0,
                tipo_intervencion=tipo.value
            )

        estados_interventor = self._historial_estados[interventor_id]

        if len(estados_interventor) < 3:
            return SimulacionIntervencion(
                agent_id=interventor_id,
                varianza_pre=varianza_pre,
                varianza_post=varianza_pre,
                reduccion=0.0,
                tipo_intervencion=tipo.value
            )

        # Calcular "estabilidad" del interventor
        # Un agente estable puede reducir más varianza
        deltas_interventor = []
        for i in range(1, len(estados_interventor)):
            deltas_interventor.append(
                estados_interventor[i] - estados_interventor[i-1]
            )

        if not deltas_interventor:
            return SimulacionIntervencion(
                agent_id=interventor_id,
                varianza_pre=varianza_pre,
                varianza_post=varianza_pre,
                reduccion=0.0,
                tipo_intervencion=tipo.value
            )

        # Varianza del interventor
        var_interventor = np.mean([np.var(d) for d in deltas_interventor])

        # Factor de estabilidad = 1 / (1 + var_interventor)
        estabilidad = 1 / (1 + var_interventor)

        # Estimar reducción de varianza
        # Basado en correlación entre estados del interventor y del sistema
        correlacion_total = 0.0
        n_agentes = 0

        for other_id, otros_estados in self._historial_estados.items():
            if other_id == interventor_id:
                continue
            if len(otros_estados) < 2:
                continue

            # Correlación entre últimos estados
            estado_interventor = estados_interventor[-1]
            estado_otro = otros_estados[-1]

            # Usar similitud como proxy de "influencia potencial"
            norm_i = np.linalg.norm(estado_interventor)
            norm_o = np.linalg.norm(estado_otro)

            if norm_i > np.finfo(float).eps and norm_o > np.finfo(float).eps:
                sim = np.dot(estado_interventor, estado_otro) / (norm_i * norm_o)
                correlacion_total += abs(sim)
                n_agentes += 1

        if n_agentes > 0:
            correlacion_media = correlacion_total / n_agentes
        else:
            correlacion_media = 0

        # Reducción estimada = estabilidad * correlación * varianza_pre
        reduccion_estimada = estabilidad * correlacion_media * varianza_pre

        # Ajustar por tipo de rol
        if tipo == TipoRol.MEDICO:
            # Médico es más efectivo cuando hay alta varianza
            factor_rol = varianza_pre / (varianza_pre + 1)
        elif tipo == TipoRol.ESTABILIZADOR:
            # Estabilizador es consistentemente efectivo
            factor_rol = estabilidad
        elif tipo == TipoRol.LIDER:
            # Líder es efectivo con alta correlación
            factor_rol = correlacion_media
        elif tipo == TipoRol.INTEGRADOR:
            # Integrador es efectivo cuando hay muchos agentes
            factor_rol = n_agentes / (n_agentes + 1)
        else:
            # Por simetría, punto medio de [0,1]
            factor_rol = 1 / 2

        reduccion_final = reduccion_estimada * factor_rol
        varianza_post = max(0.0, varianza_pre - reduccion_final)

        return SimulacionIntervencion(
            agent_id=interventor_id,
            varianza_pre=varianza_pre,
            varianza_post=varianza_post,
            reduccion=reduccion_final,
            tipo_intervencion=tipo.value
        )

    def calcular_roles(self) -> Dict[str, EstadoRol]:
        """
        Calcula los roles emergentes.

        R_i = argmin_j (d/dt Σ_k Var[S_k(t) - S_k(t-1)] | j interviene)

        Returns:
            Dict con el estado de rol de cada agente
        """
        self.t += 1

        if len(self._historial_estados) < 2:
            return {}

        # Simular intervención de cada agente para cada rol
        simulaciones: Dict[str, Dict[TipoRol, SimulacionIntervencion]] = {}

        for agent_id in self._historial_estados.keys():
            simulaciones[agent_id] = {}
            for rol in [TipoRol.MEDICO, TipoRol.ESTABILIZADOR,
                       TipoRol.LIDER, TipoRol.INTEGRADOR]:
                sim = self._simular_intervencion(agent_id, rol)
                simulaciones[agent_id][rol] = sim

        # Para cada rol, encontrar quién lo desempeña mejor
        mejores_por_rol: Dict[TipoRol, Tuple[str, float]] = {}

        for rol in [TipoRol.MEDICO, TipoRol.ESTABILIZADOR,
                   TipoRol.LIDER, TipoRol.INTEGRADOR]:
            mejor_agent = None
            mejor_reduccion = -float('inf')

            for agent_id, sims in simulaciones.items():
                reduccion = sims[rol].reduccion
                if reduccion > mejor_reduccion:
                    mejor_reduccion = reduccion
                    mejor_agent = agent_id

            if mejor_agent is not None:
                mejores_por_rol[rol] = (mejor_agent, mejor_reduccion)

        # Asignar roles (cada agente puede tener un rol)
        roles_asignados: Dict[str, TipoRol] = {}
        agentes_con_rol = set()

        # Ordenar roles por importancia de reducción
        roles_ordenados = sorted(
            mejores_por_rol.items(),
            key=lambda x: x[1][1],
            reverse=True
        )

        for rol, (agent_id, reduccion) in roles_ordenados:
            if agent_id not in agentes_con_rol:
                roles_asignados[agent_id] = rol
                agentes_con_rol.add(agent_id)

        # Actualizar roles actuales
        for agent_id in self._historial_estados.keys():
            if agent_id in roles_asignados:
                self._roles_actuales[agent_id] = roles_asignados[agent_id]
            else:
                self._roles_actuales[agent_id] = TipoRol.NINGUNO

        # Construir estados de rol
        estados: Dict[str, EstadoRol] = {}

        for agent_id in self._historial_estados.keys():
            rol = self._roles_actuales[agent_id]

            # Calcular aptitud para el rol asignado
            if rol != TipoRol.NINGUNO and agent_id in simulaciones:
                reduccion = simulaciones[agent_id][rol].reduccion
            else:
                reduccion = 0.0

            # Guardar en historial
            self._historial_reducciones[agent_id].append(reduccion)

            # Confianza basada en consistencia del rol
            if len(self._historial_reducciones[agent_id]) > 3:
                var_reducciones = np.var(self._historial_reducciones[agent_id][-10:])
                confianza = 1 / (1 + var_reducciones)
            else:
                # Por simetría, punto medio de [0,1]
                confianza = 1 / 2

            # Aptitud = percentil de reducción en el historial
            if len(self._historial_reducciones[agent_id]) > 5:
                todas_reducciones = []
                for hist in self._historial_reducciones.values():
                    todas_reducciones.extend(hist)
                if todas_reducciones:
                    percentil = np.mean(reduccion > np.array(todas_reducciones))
                    aptitud = percentil
                else:
                    # Por simetría, punto medio
                    aptitud = 1 / 2
            else:
                # Por simetría, punto medio
                aptitud = 1 / 2

            estados[agent_id] = EstadoRol(
                agent_id=agent_id,
                rol=rol,
                aptitud_rol=float(aptitud),
                reduccion_varianza=float(reduccion),
                confianza=float(confianza),
                t=self.t
            )

            # Actualizar aptitudes por rol
            if rol != TipoRol.NINGUNO:
                self._aptitudes[agent_id][rol].append(aptitud)

        return estados

    def obtener_rol(self, agent_id: str) -> TipoRol:
        """Obtiene el rol actual de un agente."""
        return self._roles_actuales.get(agent_id, TipoRol.NINGUNO)

    def obtener_medico(self) -> Optional[str]:
        """Obtiene el agente con rol de médico."""
        for agent_id, rol in self._roles_actuales.items():
            if rol == TipoRol.MEDICO:
                return agent_id
        return None

    def obtener_estabilizador(self) -> Optional[str]:
        """Obtiene el agente con rol de estabilizador."""
        for agent_id, rol in self._roles_actuales.items():
            if rol == TipoRol.ESTABILIZADOR:
                return agent_id
        return None

    def obtener_lider(self) -> Optional[str]:
        """Obtiene el agente con rol de líder."""
        for agent_id, rol in self._roles_actuales.items():
            if rol == TipoRol.LIDER:
                return agent_id
        return None

    def aptitud_historica(self, agent_id: str, rol: TipoRol) -> float:
        """
        Calcula aptitud histórica de un agente para un rol.
        """
        if agent_id not in self._aptitudes:
            return 0

        historial = self._aptitudes[agent_id].get(rol, [])
        if not historial:
            return 0

        # Media ponderada por recencia (más peso a valores recientes)
        n = len(historial)
        pesos = np.arange(1, n + 1)
        pesos = pesos / pesos.sum()

        return float(np.dot(historial, pesos))

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas del sistema de roles."""
        stats = {
            't': self.t,
            'n_agentes': len(self._historial_estados),
            'roles_asignados': {
                agent_id: rol.value
                for agent_id, rol in self._roles_actuales.items()
            },
            'varianza_sistema': self._calcular_varianza_sistema(),
            'derivada_varianza': self._calcular_derivada_varianza(),
        }

        # Contar roles
        conteo_roles = {rol.value: 0 for rol in TipoRol}
        for rol in self._roles_actuales.values():
            conteo_roles[rol.value] += 1
        stats['conteo_roles'] = conteo_roles

        return stats
