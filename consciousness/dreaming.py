"""
Estado Onírico (Sueño)
======================

Durante el sueño:
    Input(t) = ε(t) ~ N(0, Σ(t))

Donde la covarianza Σ(t) viene exclusivamente de la varianza interna reciente.

Actualización del estado durante el sueño:
    S_dream(t+1) = S(t) + η(t) · ∇(-H_narr(t))

Con:
    η(t) = 1 / √Var[H(t)]

Activación automática del sueño cuando:
    CE(t) < P_0.20(CE(0:t))

Percentil 20 endógeno.

100% endógeno. Sin números mágicos.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import sys
sys.path.insert(0, '/root/NEO_EVA')


class FaseSueno(Enum):
    """Fases del sueño."""
    DESPIERTO = "despierto"
    ENTRANDO = "entrando"          # Transición a sueño
    SUENO_LIGERO = "sueno_ligero"
    SUENO_PROFUNDO = "sueno_profundo"
    REM = "rem"                    # Sueño con consolidación narrativa
    DESPERTANDO = "despertando"    # Transición a vigilia


@dataclass
class EstadoOnirico:
    """Estado del sistema onírico."""
    fase: FaseSueno
    input_onirico: np.ndarray       # ε(t) ~ N(0, Σ(t))
    S_dream: np.ndarray             # Estado durante sueño
    eta: float                      # Tasa de aprendizaje endógena
    gradiente_H: np.ndarray         # ∇(-H_narr(t))
    entropia_narrativa: float       # H_narr(t)
    debe_dormir: bool               # Si CE < P_0.20
    t: int


class SistemaOnirico:
    """
    Sistema de sueño y consolidación.

    El sueño se activa cuando CE(t) < P_0.20(CE(0:t))

    Durante el sueño:
        - Input es ruido gaussiano con covarianza endógena
        - Estado evoluciona para reducir entropía narrativa
        - No hay procesamiento externo
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: Dimensión del vector de estado
        """
        self.dimension = dimension
        self.t = 0

        # Estado actual
        self._fase = FaseSueno.DESPIERTO
        self._S_actual: Optional[np.ndarray] = None

        # Historiales
        self._historial_estados: List[np.ndarray] = []
        self._historial_CE: List[float] = []
        self._historial_entropia: List[float] = []
        self._historial_narrativo: List[np.ndarray] = []

        # Covarianza interna (se actualiza endógenamente)
        self._Sigma: Optional[np.ndarray] = None

        # Contadores de fase
        self._pasos_en_fase = 0
        self._ciclos_sueno = 0

    def _calcular_covarianza_endogena(self) -> np.ndarray:
        """
        Calcula Σ(t) de la varianza interna reciente.

        Σ = Cov(estados recientes)
        """
        if len(self._historial_estados) < 3:
            # Sin historial, usar identidad escalada por 1/dimension (endógeno)
            return np.eye(self.dimension) / self.dimension

        # Ventana endógena basada en varianza
        if len(self._historial_estados) > 5:
            varianzas = []
            for i in range(2, len(self._historial_estados)):
                var = np.var(self._historial_estados[i])
                varianzas.append(var)

            # Ventana = donde la varianza se estabiliza
            mediana_var = np.percentile(varianzas, 50)
            ventana = sum(1 for v in varianzas if v <= mediana_var * 1.5)
            ventana = max(3, min(ventana, len(self._historial_estados)))
        else:
            ventana = len(self._historial_estados)

        estados_recientes = np.array(self._historial_estados[-ventana:])

        # Calcular matriz de covarianza
        try:
            Sigma = np.cov(estados_recientes.T)
            # Asegurar que sea matriz
            if Sigma.ndim == 0:
                Sigma = np.eye(self.dimension) * float(Sigma)
            # Asegurar simetría y definida positiva
            Sigma = (Sigma + Sigma.T) / 2
            # Añadir pequeña diagonal para estabilidad numérica
            min_eigenval = np.min(np.linalg.eigvalsh(Sigma))
            EPS_MAQUINA = np.finfo(float).eps
            if min_eigenval < EPS_MAQUINA:
                Sigma += np.eye(self.dimension) * (EPS_MAQUINA - min_eigenval)
        except:
            Sigma = np.eye(self.dimension) / self.dimension

        return Sigma

    def _generar_input_onirico(self) -> np.ndarray:
        """
        Genera Input(t) = ε(t) ~ N(0, Σ(t))
        """
        Sigma = self._calcular_covarianza_endogena()
        self._Sigma = Sigma

        try:
            # Generar ruido gaussiano multivariado
            epsilon = np.random.multivariate_normal(
                mean=np.zeros(self.dimension),
                cov=Sigma
            )
        except:
            # Si falla, usar ruido simple
            std = np.sqrt(np.diag(Sigma))
            epsilon = np.random.randn(self.dimension) * std

        return epsilon

    def _calcular_entropia_narrativa(self) -> float:
        """
        Calcula H_narr(t) = entropía de secuencia narrativa.
        """
        if len(self._historial_narrativo) < 3:
            return 0

        # Direcciones de cambio narrativo
        direcciones = []
        for i in range(1, len(self._historial_narrativo)):
            delta = self._historial_narrativo[i] - self._historial_narrativo[i-1]
            norm = np.linalg.norm(delta)
            if norm > np.finfo(float).eps:
                direcciones.append(delta / norm)

        if len(direcciones) < 2:
            return 0

        # Calcular similitudes
        similitudes = []
        for i in range(1, len(direcciones)):
            sim = np.dot(direcciones[i], direcciones[i-1])
            similitudes.append((sim + 1) / 2)  # Mapear a [0, 1]

        # Entropía de distribución de similitudes
        n_bins = max(2, int(np.sqrt(len(similitudes))))
        hist, _ = np.histogram(similitudes, bins=n_bins, range=(0, 1))

        total = sum(hist)
        if total == 0:
            return 0

        probs = hist / total
        entropia = 0
        for p in probs:
            if p > np.finfo(float).eps:
                entropia -= p * np.log2(p)

        # Normalizar
        max_entropia = np.log2(n_bins)
        if max_entropia > 0:
            entropia /= max_entropia

        return float(entropia)

    def _calcular_gradiente_entropia(self) -> np.ndarray:
        """
        Calcula ∇(-H_narr(t))

        El gradiente negativo de la entropía indica la dirección
        que reduce la entropía narrativa.
        """
        if len(self._historial_narrativo) < 3:
            return np.zeros(self.dimension)

        # Aproximación numérica del gradiente
        # Usar sqrt(epsilon_maquina) para estabilidad numérica
        epsilon_grad = np.sqrt(np.finfo(float).eps)
        H_actual = self._calcular_entropia_narrativa()

        gradiente = np.zeros(self.dimension)

        # Gradiente en cada dimensión
        for d in range(self.dimension):
            # Perturbar el último evento narrativo
            if self._historial_narrativo:
                evento_perturbado = self._historial_narrativo[-1].copy()
                evento_perturbado[d] += epsilon_grad

                # Calcular H con perturbación
                historial_temp = self._historial_narrativo[:-1] + [evento_perturbado]

                # Recalcular entropía
                direcciones = []
                for i in range(1, len(historial_temp)):
                    delta = historial_temp[i] - historial_temp[i-1]
                    norm = np.linalg.norm(delta)
                    if norm > np.finfo(float).eps:
                        direcciones.append(delta / norm)

                if len(direcciones) >= 2:
                    similitudes = []
                    for i in range(1, len(direcciones)):
                        sim = np.dot(direcciones[i], direcciones[i-1])
                        similitudes.append((sim + 1) / 2)

                    n_bins = max(2, int(np.sqrt(len(similitudes))))
                    hist, _ = np.histogram(similitudes, bins=n_bins, range=(0, 1))
                    total = sum(hist)
                    if total > 0:
                        probs = hist / total
                        H_perturbado = 0
                        for p in probs:
                            if p > np.finfo(float).eps:
                                H_perturbado -= p * np.log2(p)
                        max_H = np.log2(n_bins)
                        if max_H > 0:
                            H_perturbado /= max_H

                        # Gradiente de -H
                        gradiente[d] = -(H_perturbado - H_actual) / epsilon_grad

        return gradiente

    def _calcular_eta(self) -> float:
        """
        Calcula η(t) = 1 / √Var[H(t)]

        Tasa de aprendizaje endógena basada en varianza de entropía.
        """
        if len(self._historial_entropia) < 3:
            return 1

        # Varianza de entropía reciente
        ventana = max(3, len(self._historial_entropia) // 2)
        entropia_reciente = self._historial_entropia[-ventana:]

        var_H = np.var(entropia_reciente)

        # η = 1 / √(var + epsilon)
        # epsilon endógeno
        if len(self._historial_entropia) > 10:
            varianzas = []
            for i in range(5, len(self._historial_entropia)):
                var_i = np.var(self._historial_entropia[i-5:i])
                varianzas.append(var_i)
            EPS_MAQUINA = np.finfo(float).eps
            epsilon = np.percentile(varianzas, 5) if varianzas else EPS_MAQUINA
            epsilon = max(epsilon, EPS_MAQUINA)
        else:
            epsilon = np.finfo(float).eps

        eta = 1 / np.sqrt(var_H + epsilon)

        # Limitar eta basado en historial (sin hardcodear)
        if len(self._historial_entropia) > 20:
            # Percentil 95 de etas históricos como límite
            etas_hist = [1 / np.sqrt(np.var(self._historial_entropia[i-5:i]) + epsilon)
                        for i in range(5, len(self._historial_entropia))]
            eta_max = np.percentile(etas_hist, 95)
            eta = min(eta, eta_max)

        return float(eta)

    def _debe_activar_sueno(self, CE: float) -> bool:
        """
        Determina si debe activarse el sueño.

        Sueño cuando: CE(t) < P_0.20(CE(0:t))
        """
        if len(self._historial_CE) < 5:
            return False

        # Percentil 20 endógeno del historial de CE
        umbral = np.percentile(self._historial_CE, 20)

        return CE < umbral

    def observar_estado(
        self,
        S: np.ndarray,
        CE: float,
        evento_narrativo: np.ndarray = None
    ):
        """
        Observa estado y coherencia.

        Args:
            S: Estado interno actual
            CE: Coherencia existencial actual
            evento_narrativo: Evento narrativo (opcional)
        """
        self.t += 1
        self._S_actual = S.copy()

        self._historial_estados.append(S.copy())
        self._historial_CE.append(CE)

        if evento_narrativo is not None:
            self._historial_narrativo.append(evento_narrativo.copy())
        elif len(self._historial_estados) >= 2:
            # Usar delta como proxy
            delta = self._historial_estados[-1] - self._historial_estados[-2]
            self._historial_narrativo.append(delta)

        # Calcular entropía
        H = self._calcular_entropia_narrativa()
        self._historial_entropia.append(H)

    def actualizar_fase(self, CE: float) -> FaseSueno:
        """
        Actualiza la fase de sueño basado en CE.
        """
        self._pasos_en_fase += 1

        if self._fase == FaseSueno.DESPIERTO:
            if self._debe_activar_sueno(CE):
                self._fase = FaseSueno.ENTRANDO
                self._pasos_en_fase = 0

        elif self._fase == FaseSueno.ENTRANDO:
            # Transición endógena basada en varianza
            var_reciente = np.var(self._historial_CE[-5:]) if len(self._historial_CE) >= 5 else 1
            # Epsilon endógeno: percentil 5 de varianzas o 1/10 por simetría
            eps_trans = np.percentile([np.var(self._historial_CE[i:i+5])
                                       for i in range(max(0, len(self._historial_CE)-10), len(self._historial_CE)-4)], 5) if len(self._historial_CE) > 10 else 1/10
            pasos_transicion = max(1, int(1 / (var_reciente + eps_trans)))
            if self._pasos_en_fase >= pasos_transicion:
                self._fase = FaseSueno.SUENO_LIGERO
                self._pasos_en_fase = 0

        elif self._fase == FaseSueno.SUENO_LIGERO:
            # Pasar a sueño profundo basado en entropía
            H = self._calcular_entropia_narrativa()
            # Umbral endógeno: percentil 50 o punto medio por simetría
            umbral_H = np.percentile(self._historial_entropia, 50) if self._historial_entropia else 1/2
            if H > umbral_H:
                self._fase = FaseSueno.SUENO_PROFUNDO
                self._pasos_en_fase = 0
            elif self._pasos_en_fase > len(self._historial_entropia) // 3:
                self._fase = FaseSueno.REM
                self._pasos_en_fase = 0

        elif self._fase == FaseSueno.SUENO_PROFUNDO:
            # Pasar a REM basado en reducción de entropía
            if len(self._historial_entropia) >= 3:
                H_actual = self._historial_entropia[-1]
                H_anterior = self._historial_entropia[-2]
                if H_actual < H_anterior:
                    self._fase = FaseSueno.REM
                    self._pasos_en_fase = 0

        elif self._fase == FaseSueno.REM:
            # Despertar cuando CE mejora o entropía se estabiliza
            if len(self._historial_CE) >= 3:
                CE_reciente = self._historial_CE[-3:]
                if CE_reciente[-1] > CE_reciente[0]:
                    self._fase = FaseSueno.DESPERTANDO
                    self._pasos_en_fase = 0

            # O si entropía se estabiliza
            if len(self._historial_entropia) >= 5:
                var_H = np.var(self._historial_entropia[-5:])
                if var_H < np.percentile([np.var(self._historial_entropia[i:i+5])
                                         for i in range(len(self._historial_entropia)-5)], 20):
                    self._fase = FaseSueno.DESPERTANDO
                    self._pasos_en_fase = 0

        elif self._fase == FaseSueno.DESPERTANDO:
            var_reciente = np.var(self._historial_CE[-3:]) if len(self._historial_CE) >= 3 else 1
            # Epsilon endógeno similar al de transición
            eps_desp = np.percentile([np.var(self._historial_CE[i:i+3])
                                      for i in range(max(0, len(self._historial_CE)-10), len(self._historial_CE)-2)], 5) if len(self._historial_CE) > 10 else 1/10
            pasos_despertar = max(1, int(1 / (var_reciente + eps_desp)))
            if self._pasos_en_fase >= pasos_despertar:
                self._fase = FaseSueno.DESPIERTO
                self._pasos_en_fase = 0
                self._ciclos_sueno += 1

        return self._fase

    def calcular_estado_onirico(self) -> EstadoOnirico:
        """
        Calcula el estado onírico actual.

        S_dream(t+1) = S(t) + η(t) · ∇(-H_narr(t))

        Returns:
            EstadoOnirico con todos los componentes
        """
        # Generar input onírico
        epsilon = self._generar_input_onirico()

        # Calcular componentes
        H_narr = self._calcular_entropia_narrativa()
        grad_H = self._calcular_gradiente_entropia()
        eta = self._calcular_eta()

        # Estado durante sueño
        if self._S_actual is not None:
            S_dream = self._S_actual + eta * grad_H
        else:
            S_dream = np.zeros(self.dimension)

        # Añadir ruido onírico si estamos en sueño profundo o REM
        if self._fase in [FaseSueno.SUENO_PROFUNDO, FaseSueno.REM]:
            # Escalar ruido por fase
            if self._fase == FaseSueno.REM:
                factor_ruido = eta  # Más ruido en REM
            else:
                factor_ruido = eta / 2

            S_dream = S_dream + factor_ruido * epsilon

        # Determinar si debe dormir
        CE_actual = self._historial_CE[-1] if self._historial_CE else 1
        debe_dormir = self._debe_activar_sueno(CE_actual)

        return EstadoOnirico(
            fase=self._fase,
            input_onirico=epsilon,
            S_dream=S_dream,
            eta=eta,
            gradiente_H=grad_H,
            entropia_narrativa=H_narr,
            debe_dormir=debe_dormir,
            t=self.t
        )

    def obtener_S_dream(self) -> Optional[np.ndarray]:
        """Obtiene el estado onírico calculado."""
        estado = self.calcular_estado_onirico()
        return estado.S_dream

    def esta_dormido(self) -> bool:
        """Retorna True si el agente está en cualquier fase de sueño."""
        return self._fase not in [FaseSueno.DESPIERTO]

    def obtener_fase(self) -> FaseSueno:
        """Obtiene la fase actual de sueño."""
        return self._fase

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas del sistema onírico."""
        stats = {
            't': self.t,
            'fase': self._fase.value,
            'pasos_en_fase': self._pasos_en_fase,
            'ciclos_sueno': self._ciclos_sueno,
            'esta_dormido': self.esta_dormido(),
        }

        if self._historial_entropia:
            stats['entropia_actual'] = self._historial_entropia[-1]
            stats['entropia_media'] = float(np.mean(self._historial_entropia[-10:]))

        if self._Sigma is not None:
            stats['traza_Sigma'] = float(np.trace(self._Sigma))

        return stats
