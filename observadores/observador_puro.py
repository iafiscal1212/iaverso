"""
Observador Puro
===============

Registra valores matemáticos del sistema sin interpretarlos.

Solo copia valores como un acelerómetro, un termómetro, un sismógrafo.

NO se permite:
    - Emitir comandos
    - Inducir estados
    - Sugerir acciones
    - Definir condiciones
    - Etiquetar estados
    - Crear triggers
    - Intervenir en dinámica
    - Clasificar o interpretar
    - Elegir qué agentes observar más o menos
"""

import numpy as np
from typing import Dict, Any


class ObservadorPuro:
    """
    Observador pasivo que registra valores matemáticos sin interpretarlos.

    Registra en cada paso:
        - S: estado interno
        - I: identidad
        - Delta: S - I
        - Var_Delta: varianza de Delta
        - CE: coherencia existencial
        - H_narr: entropía narrativa
        - rol: rol actual (copia literal)
    """

    def __init__(self):
        self.historial: Dict[Any, Dict[str, Dict[str, Any]]] = {}

    def registrar(self, t: Any, sistema) -> None:
        """
        Registra valores matemáticos del estado del sistema sin interpretarlos.

        Args:
            t: Timestamp o identificador del paso
            sistema: Sistema con atributo .agentes iterable
        """
        registro_t: Dict[str, Dict[str, Any]] = {}

        for agente in sistema.agentes:
            S = agente.estado_interno
            I = agente.identidad
            Delta = S - I
            Var_Delta = np.var(Delta)
            CE = agente.coherencia
            H_narr = agente.entropia_narrativa
            rol = agente.rol_actual

            registro_t[agente.id] = {
                "S": S,
                "I": I,
                "Delta": Delta,
                "Var_Delta": Var_Delta,
                "CE": CE,
                "H_narr": H_narr,
                "rol": rol
            }

        self.historial[t] = registro_t

    def exportar(self) -> Dict[Any, Dict[str, Dict[str, Any]]]:
        """
        Retorna el historial completo de registros.

        Returns:
            Historial sin modificar.
        """
        return self.historial
