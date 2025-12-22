"""
EndoLens Core - Percepción Estructural

NORMA DURA:
- Solo calcula estructura matemática
- No interpreta significado
- No juzga valor
- Emite firma, el humano lee

Rol en Inferencia Activa: PERCEPCIÓN
"¿Qué estado estructural observo?"
"""

import math
import hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Signature:
    """Firma estructural EndoLens."""
    attractor: int      # A: coordenada del atractor
    energy: int         # E: nivel de energía (0-9)
    drift: int          # Δ: varianza/dispersión
    raw: str            # Representación string: ⟨A33·E2·Δ19⟩
    
    def __str__(self):
        return self.raw


@dataclass 
class ESeries:
    """E-series: métricas estructurales."""
    E0: float  # Persistencia (qué tanto se mantiene)
    E1: float  # Variabilidad (qué tanto cambia)
    E2: float  # Tendencia (hacia dónde va)
    E3: float  # Complejidad (cuántas relaciones)
    
    def as_dict(self) -> Dict:
        return {'E0': self.E0, 'E1': self.E1, 'E2': self.E2, 'E3': self.E3}
    
    def stability_score(self) -> float:
        """Calcula estabilidad. No interpreta, solo calcula."""
        # Alta persistencia (E0 bajo) + baja variabilidad (E1 bajo) = estable
        stability = 1.0 - (self.E0 * 0.3 + self.E1 * 0.4 + self.E2 * 0.1 + self.E3 * 0.2)
        return max(0.0, min(1.0, stability))


@dataclass
class StructuralState:
    """Estado estructural completo."""
    signature: Signature
    eseries: ESeries
    stability: float
    status: str  # 'stable', 'dynamic', 'unstable'
    invariants: List[str]
    tensions: List[str]
    

class EndoLensCore:
    """
    Motor de percepción estructural.
    
    Transforma input → firma estructural.
    No sabe qué significa el input.
    No sabe qué significa la firma.
    Solo transforma.
    """
    
    def __init__(self):
        self.operators = ['T_equiv', 'T_norm', 'T_vec']
        
    def process(self, text: str) -> StructuralState:
        """
        Procesa texto y retorna estado estructural.
        
        NORMA DURA: No interpreta, solo transforma.
        """
        # 1. Calcular gematría base
        gematria = self._calculate_gematria(text)
        
        # 2. Calcular E-series
        eseries = self._calculate_eseries(text, gematria)
        
        # 3. Generar firma
        signature = self._generate_signature(gematria, eseries)
        
        # 4. Detectar invariantes (propiedades que no cambian bajo transformación)
        invariants = self._detect_invariants(text, gematria)
        
        # 5. Detectar tensiones (inconsistencias estructurales)
        tensions = self._detect_tensions(eseries)
        
        # 6. Calcular estabilidad
        stability = eseries.stability_score()
        
        # 7. Determinar status (sin juicio, solo umbral)
        if stability >= 0.7:
            status = 'stable'
        elif stability >= 0.4:
            status = 'dynamic'
        else:
            status = 'unstable'
        
        return StructuralState(
            signature=signature,
            eseries=eseries,
            stability=stability,
            status=status,
            invariants=invariants,
            tensions=tensions
        )
    
    def _calculate_gematria(self, text: str) -> int:
        """Suma gematrica del texto."""
        total = 0
        for char in text.lower():
            if char.isalpha():
                total += ord(char) - ord('a') + 1
            elif char.isdigit():
                total += int(char)
        return total
    
    def _calculate_eseries(self, text: str, gematria: int) -> ESeries:
        """Calcula E-series desde el texto."""
        if not text:
            return ESeries(E0=0, E1=0, E2=0, E3=0)
        
        chars = list(text.lower())
        n = len(chars)
        
        # E0: Persistencia - ratio de caracteres repetidos
        unique = len(set(chars))
        E0 = 1.0 - (unique / n) if n > 0 else 0
        
        # E1: Variabilidad - entropía normalizada
        from collections import Counter
        freq = Counter(chars)
        entropy = 0
        for count in freq.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)
        max_entropy = math.log2(n) if n > 1 else 1
        E1 = entropy / max_entropy if max_entropy > 0 else 0
        
        # E2: Tendencia - diferencia entre primera y segunda mitad
        mid = n // 2
        first_half = sum(ord(c) for c in chars[:mid]) if mid > 0 else 0
        second_half = sum(ord(c) for c in chars[mid:]) if mid < n else 0
        E2 = abs(first_half - second_half) / (first_half + second_half + 1)
        E2 = min(1.0, E2)
        
        # E3: Complejidad - basada en bigramas únicos
        bigrams = set()
        for i in range(len(chars) - 1):
            bigrams.add(chars[i] + chars[i+1])
        max_bigrams = n - 1 if n > 1 else 1
        E3 = len(bigrams) / max_bigrams if max_bigrams > 0 else 0
        
        return ESeries(E0=round(E0, 3), E1=round(E1, 3), E2=round(E2, 3), E3=round(E3, 3))
    
    def _generate_signature(self, gematria: int, eseries: ESeries) -> Signature:
        """Genera firma estructural."""
        # Attractor: basado en gematría mod 100
        attractor = gematria % 100
        
        # Energy: basado en E1 (variabilidad)
        energy = int(eseries.E1 * 9)
        
        # Drift: basado en E3 (complejidad) * 100
        drift = int(eseries.E3 * 100)
        
        raw = f"⟨A{attractor}·E{energy}·Δ{drift}⟩"
        
        return Signature(
            attractor=attractor,
            energy=energy,
            drift=drift,
            raw=raw
        )
    
    def _detect_invariants(self, text: str, gematria: int) -> List[str]:
        """
        Detecta propiedades invariantes bajo transformación.
        
        NORMA DURA: Solo reporta qué se mantiene, no por qué importa.
        """
        invariants = []
        
        # Invariante: suma gematrica (siempre se preserva bajo T_equiv)
        invariants.append(f"gematric_sum:{gematria}")
        
        # Invariante: paridad
        parity = 'even' if gematria % 2 == 0 else 'odd'
        invariants.append(f"parity:{parity}")
        
        # Invariante: longitud
        invariants.append(f"length:{len(text)}")
        
        # Invariante: hash estructural (primeros 8 chars del sha256)
        struct_hash = hashlib.sha256(text.encode()).hexdigest()[:8]
        invariants.append(f"struct_hash:{struct_hash}")
        
        return invariants
    
    def _detect_tensions(self, eseries: ESeries) -> List[str]:
        """
        Detecta tensiones estructurales.
        
        NORMA DURA: Solo reporta inconsistencias, no las juzga.
        """
        tensions = []
        
        # Tensión: alta variabilidad con alta persistencia (contradictorio)
        if eseries.E0 > 0.5 and eseries.E1 > 0.5:
            tensions.append("high_persistence_high_variability")
        
        # Tensión: alta complejidad con baja variabilidad (potencialmente atrapado)
        if eseries.E3 > 0.6 and eseries.E1 < 0.2:
            tensions.append("complex_but_static")
        
        # Tensión: tendencia fuerte (E2 alto)
        if eseries.E2 > 0.3:
            tensions.append("strong_directional_tendency")
        
        return tensions
    
    def compare(self, state1: StructuralState, state2: StructuralState) -> Dict:
        """
        Compara dos estados estructurales.
        
        NORMA DURA: Solo reporta diferencias numéricas.
        """
        return {
            'stability_delta': state2.stability - state1.stability,
            'E0_delta': state2.eseries.E0 - state1.eseries.E0,
            'E1_delta': state2.eseries.E1 - state1.eseries.E1,
            'E2_delta': state2.eseries.E2 - state1.eseries.E2,
            'E3_delta': state2.eseries.E3 - state1.eseries.E3,
            'drift_delta': state2.signature.drift - state1.signature.drift,
            'same_attractor': state1.signature.attractor == state2.signature.attractor
        }


# Singleton para uso global
_instance = None

def get_endolens() -> EndoLensCore:
    global _instance
    if _instance is None:
        _instance = EndoLensCore()
    return _instance
