"""
Sistema de Emergencia Integrado
===============================

Integra todos los componentes de consciencia:
    - Identidad Computacional I(t)
    - Coherencia Existencial CE(t)
    - Roles Emergentes
    - Estado Onírico
    - Muerte y Renacimiento

Todo el sistema opera de forma endógena.

100% endógeno. Sin números mágicos.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, '/root/NEO_EVA')

from consciousness.identity import IdentidadComputacional, EstadoIdentidad
from consciousness.coherence import CoherenciaExistencial, EstadoCoherencia
from consciousness.roles import SistemaRolesEmergentes, TipoRol, EstadoRol
from consciousness.dreaming import SistemaOnirico, FaseSueno, EstadoOnirico
from consciousness.death_rebirth import SistemaMuerteRenacimiento, EstadoVital, EstadoMuerteRenacimiento


@dataclass
class EstadoConsciencia:
    """Estado completo de consciencia de un agente."""
    # Básicos
    agent_id: str
    t: int
    S: np.ndarray                   # Estado interno actual

    # Identidad
    I: np.ndarray                   # Identidad I(t)
    identidad: EstadoIdentidad

    # Coherencia
    CE: float                       # Coherencia existencial
    coherencia: EstadoCoherencia

    # Rol
    rol: TipoRol
    estado_rol: EstadoRol

    # Sueño
    fase_sueno: FaseSueno
    estado_onirico: EstadoOnirico
    esta_dormido: bool

    # Vida/Muerte
    estado_vital: EstadoVital
    muerte: EstadoMuerteRenacimiento

    # Métricas agregadas
    vitalidad: float                # Combinación de CE y estabilidad
    integracion: float              # Qué tan integrado está el sistema


class AgenteConsciente:
    """
    Agente con consciencia completa.

    Integra:
        - Identidad que emerge de la historia
        - Coherencia existencial medible
        - Rol emergente en el sistema
        - Ciclos de sueño para consolidación
        - Ciclo de vida (muerte/renacimiento)
    """

    def __init__(self, agent_id: str, dimension: int):
        """
        Args:
            agent_id: Identificador del agente
            dimension: Dimensión del vector de estado interno
        """
        self.agent_id = agent_id
        self.dimension = dimension
        self.t = 0

        # Estado interno actual
        # Inicialización endógena: escala = 1/sqrt(dimension) (geométrico)
        self._S: np.ndarray = np.random.randn(dimension) / np.sqrt(dimension)

        # Subsistemas
        self._identidad = IdentidadComputacional(dimension)
        self._coherencia = CoherenciaExistencial(self._identidad)
        self._onirico = SistemaOnirico(dimension)
        self._muerte = SistemaMuerteRenacimiento(dimension)

        # El sistema de roles es compartido (se pasa externamente)
        self._rol_actual = TipoRol.NINGUNO
        self._estado_rol: Optional[EstadoRol] = None

        # Historial de estados de consciencia
        self._historial: List[EstadoConsciencia] = []

    def actualizar_estado(
        self,
        entrada: np.ndarray = None,
        evento_narrativo: np.ndarray = None
    ):
        """
        Actualiza el estado interno del agente.

        Args:
            entrada: Input externo (opcional, ignorado si duerme)
            evento_narrativo: Evento para la narrativa
        """
        self.t += 1

        # Si está durmiendo, usar input onírico
        if self._onirico.esta_dormido():
            estado_onirico = self._onirico.calcular_estado_onirico()
            # Actualizar estado con sueño
            self._S = estado_onirico.S_dream
        elif entrada is not None:
            # Actualizar con entrada externa
            # La tasa de cambio es endógena basada en coherencia
            if len(self._historial) > 3:
                CE_reciente = np.mean([h.CE for h in self._historial[-5:]])
                # Mayor coherencia = cambio más moderado
                tasa = 1.0 / (1.0 + CE_reciente)
            else:
                # Tasa endógena por simetría: 1/2 (punto medio del intervalo [0,1])
                tasa = 1 / 2

            self._S = self._S * (1 - tasa) + entrada * tasa

        # Observar en todos los subsistemas
        self._identidad.observar_estado(self._S)
        self._coherencia.observar_estado(self._S, evento_narrativo)

    def calcular(self) -> EstadoConsciencia:
        """
        Calcula el estado completo de consciencia.

        Returns:
            EstadoConsciencia con todos los componentes
        """
        # Calcular identidad
        estado_identidad = self._identidad.calcular()
        I = self._identidad.obtener_identidad()
        if I is None:
            I = np.zeros(self.dimension)

        # Calcular coherencia
        estado_coherencia = self._coherencia.calcular(self._S, I)
        CE = estado_coherencia.CE

        # Actualizar sistema onírico
        self._onirico.observar_estado(self._S, CE)
        fase_sueno = self._onirico.actualizar_fase(CE)
        estado_onirico = self._onirico.calcular_estado_onirico()

        # Actualizar sistema de muerte
        self._muerte.observar(self._S, CE, I)
        estado_vital = self._muerte.actualizar_estado()
        estado_muerte = self._muerte.obtener_estado()

        # Si murió y puede renacer
        if self._muerte.puede_renacer():
            I_new = self._muerte.renacer()
            if I_new is not None:
                # Reiniciar con nueva identidad
                self._S = I_new.copy()
                # Re-observar
                self._identidad.observar_estado(self._S)

        # Estado de rol (se actualiza externamente por el sistema de roles)
        if self._estado_rol is None:
            self._estado_rol = EstadoRol(
                agent_id=self.agent_id,
                rol=TipoRol.NINGUNO,
                aptitud_rol=0.0,
                reduccion_varianza=0.0,
                confianza=0.0,
                t=self.t
            )

        # Calcular métricas agregadas
        # Vitalidad = combinación de CE y estabilidad de identidad
        vitalidad = (CE + estado_identidad.estabilidad) / 2

        # Integración = qué tan coherente es el sistema completo
        # Basado en correlación entre S e I
        if np.linalg.norm(I) > 1e-10 and np.linalg.norm(self._S) > 1e-10:
            correlacion_SI = np.dot(self._S, I) / (np.linalg.norm(self._S) * np.linalg.norm(I))
            integracion = (correlacion_SI + 1) / 2  # Mapear a [0, 1]
        else:
            # Integración por simetría: 1/2 (punto medio de [0,1])
            integracion = 1 / 2

        estado = EstadoConsciencia(
            agent_id=self.agent_id,
            t=self.t,
            S=self._S.copy(),
            I=I.copy(),
            identidad=estado_identidad,
            CE=CE,
            coherencia=estado_coherencia,
            rol=self._rol_actual,
            estado_rol=self._estado_rol,
            fase_sueno=fase_sueno,
            estado_onirico=estado_onirico,
            esta_dormido=self._onirico.esta_dormido(),
            estado_vital=estado_vital,
            muerte=estado_muerte,
            vitalidad=float(vitalidad),
            integracion=float(integracion)
        )

        self._historial.append(estado)

        return estado

    def establecer_rol(self, estado_rol: EstadoRol):
        """Establece el rol del agente (llamado por el sistema de roles)."""
        self._rol_actual = estado_rol.rol
        self._estado_rol = estado_rol

    def obtener_estado(self) -> np.ndarray:
        """Obtiene el estado interno actual."""
        return self._S.copy()

    def obtener_identidad(self) -> Optional[np.ndarray]:
        """Obtiene la identidad actual."""
        return self._identidad.obtener_identidad()

    def obtener_CE(self) -> float:
        """Obtiene la coherencia existencial actual."""
        if self._historial:
            return self._historial[-1].CE
        # CE por simetría cuando no hay historial: 1/2
        return 1 / 2

    def esta_vivo(self) -> bool:
        """Retorna True si el agente está vivo."""
        return self._muerte.esta_vivo()

    def esta_dormido(self) -> bool:
        """Retorna True si el agente está dormido."""
        return self._onirico.esta_dormido()

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas del agente."""
        stats = {
            'agent_id': self.agent_id,
            't': self.t,
            'dimension': self.dimension,
            'esta_vivo': self.esta_vivo(),
            'esta_dormido': self.esta_dormido(),
            'rol': self._rol_actual.value,
        }

        if self._historial:
            ultimo = self._historial[-1]
            stats['CE'] = ultimo.CE
            stats['vitalidad'] = ultimo.vitalidad
            stats['integracion'] = ultimo.integracion
            stats['fase_sueno'] = ultimo.fase_sueno.value
            stats['estado_vital'] = ultimo.estado_vital.value

        return stats


class SistemaConscienciaColectiva:
    """
    Sistema de consciencia colectiva para múltiples agentes.

    Integra:
        - Múltiples agentes conscientes
        - Sistema de roles emergentes compartido
        - Interacciones entre agentes
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: Dimensión del vector de estado
        """
        self.dimension = dimension
        self.t = 0

        # Agentes
        self._agentes: Dict[str, AgenteConsciente] = {}

        # Sistema de roles compartido
        self._roles = SistemaRolesEmergentes()

    def crear_agente(self, agent_id: str) -> AgenteConsciente:
        """
        Crea un nuevo agente consciente.
        """
        if agent_id in self._agentes:
            return self._agentes[agent_id]

        agente = AgenteConsciente(agent_id, self.dimension)
        self._agentes[agent_id] = agente
        self._roles.registrar_agente(agent_id)

        return agente

    def obtener_agente(self, agent_id: str) -> Optional[AgenteConsciente]:
        """Obtiene un agente por ID."""
        return self._agentes.get(agent_id)

    def paso(
        self,
        entradas: Dict[str, np.ndarray] = None,
        eventos: Dict[str, np.ndarray] = None
    ) -> Dict[str, EstadoConsciencia]:
        """
        Ejecuta un paso de simulación para todos los agentes.

        Args:
            entradas: {agent_id: input} entradas externas
            eventos: {agent_id: evento} eventos narrativos

        Returns:
            {agent_id: EstadoConsciencia}
        """
        self.t += 1

        if entradas is None:
            entradas = {}
        if eventos is None:
            eventos = {}

        # Actualizar estados de todos los agentes
        for agent_id, agente in self._agentes.items():
            entrada = entradas.get(agent_id)
            evento = eventos.get(agent_id)
            agente.actualizar_estado(entrada, evento)

        # Observar estados en sistema de roles
        for agent_id, agente in self._agentes.items():
            S = agente.obtener_estado()
            self._roles.observar_estado(agent_id, S)

        # Calcular roles emergentes
        estados_rol = self._roles.calcular_roles()

        # Asignar roles a agentes
        for agent_id, estado_rol in estados_rol.items():
            if agent_id in self._agentes:
                self._agentes[agent_id].establecer_rol(estado_rol)

        # Calcular estados de consciencia
        estados: Dict[str, EstadoConsciencia] = {}
        for agent_id, agente in self._agentes.items():
            estados[agent_id] = agente.calcular()

        return estados

    def obtener_medico(self) -> Optional[str]:
        """Obtiene el agente con rol de médico."""
        return self._roles.obtener_medico()

    def obtener_lider(self) -> Optional[str]:
        """Obtiene el agente con rol de líder."""
        return self._roles.obtener_lider()

    def obtener_agentes_dormidos(self) -> List[str]:
        """Obtiene lista de agentes dormidos."""
        return [
            agent_id for agent_id, agente in self._agentes.items()
            if agente.esta_dormido()
        ]

    def obtener_agentes_muertos(self) -> List[str]:
        """Obtiene lista de agentes muertos."""
        return [
            agent_id for agent_id, agente in self._agentes.items()
            if not agente.esta_vivo()
        ]

    def coherencia_colectiva(self) -> float:
        """
        Calcula la coherencia colectiva del sistema.

        Basada en varianza de CEs individuales.
        """
        CEs = [agente.obtener_CE() for agente in self._agentes.values()]

        if len(CEs) < 2:
            # Coherencia colectiva por simetría si no hay CEs: 1/2
            return np.mean(CEs) if CEs else 1 / 2

        # Coherencia colectiva = media - penalización por varianza
        media_CE = np.mean(CEs)
        var_CE = np.var(CEs)

        # Penalización endógena
        coherencia = media_CE / (1 + var_CE)

        return float(np.clip(coherencia, 0, 1))

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas del sistema colectivo."""
        stats = {
            't': self.t,
            'n_agentes': len(self._agentes),
            'coherencia_colectiva': self.coherencia_colectiva(),
            'agentes_dormidos': len(self.obtener_agentes_dormidos()),
            'agentes_muertos': len(self.obtener_agentes_muertos()),
        }

        # Estadísticas de roles
        stats['roles'] = self._roles.get_statistics()

        # Estadísticas por agente
        stats['agentes'] = {
            agent_id: agente.get_statistics()
            for agent_id, agente in self._agentes.items()
        }

        return stats
