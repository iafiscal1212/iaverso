"""
Language Detector - Versión Consolidada

NORMA DURA:
- Solo detecta idioma por patrones matemáticos
- No usa ML externo ni embeddings
- Basado en caracteres únicos, bigramas, palabras función

Este archivo consolida las 36+ versiones anteriores en UNA sola.
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LanguageResult:
    """Resultado de detección de idioma."""
    language: str       # Código: ES, EN, PT, FR, DE, IT, etc.
    confidence: float   # 0.0 - 1.0
    layer: str          # ALPHA (absoluto), BETA (estadístico), GAMMA (inferido)
    discriminator: str  # Qué lo detectó


class LanguageDetector:
    """
    Detector de idiomas NORMA DURA.
    
    Arquitectura de 3 capas:
    - ALPHA: Discriminadores absolutos (ñ, ç, ß, scripts)
    - BETA: Estadísticos (palabras función, bigramas)
    - GAMMA: Inferencia (si todo falla)
    """
    
    # === CAPA ALPHA: Discriminadores absolutos ===
    ABSOLUTE_MARKERS = {
        'ñ': 'ES',
        'ç': 'PT',  # También FR, pero PT más común
        'ß': 'DE',
        'ø': 'DA',  # Danés/Noruego
        'å': 'SV',  # Sueco/Noruego
        'ü': 'DE',  # También TR
        'ö': 'DE',  # También SV, FI
        'ã': 'PT',
        'õ': 'PT',
    }
    
    # Scripts no latinos
    SCRIPT_MARKERS = {
        'cyrillic': 'RU',   # Ruso por defecto
        'arabic': 'AR',
        'hebrew': 'HE', 
        'chinese': 'ZH',
        'japanese': 'JA',
        'korean': 'KO',
        'greek': 'EL',
        'thai': 'TH',
        'devanagari': 'HI',
    }
    
    # === CAPA BETA: Palabras función ===
    FUNCTION_WORDS = {
        'ES': ['el', 'la', 'los', 'las', 'de', 'en', 'que', 'es', 'un', 'una', 'por', 'con', 'para', 'como', 'pero', 'este', 'esta', 'ese', 'esa', 'del', 'al'],
        'EN': ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'to', 'in', 'for', 'on', 'with', 'that', 'this', 'it', 'from', 'by', 'as', 'at'],
        'PT': ['o', 'a', 'os', 'as', 'de', 'em', 'que', 'um', 'uma', 'por', 'com', 'para', 'como', 'mas', 'este', 'esta', 'esse', 'essa', 'do', 'da', 'no', 'na', 'ao'],
        'FR': ['le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'est', 'en', 'que', 'qui', 'pour', 'dans', 'ce', 'cette', 'sur', 'avec', 'pas', 'je', 'tu', 'il', 'nous'],
        'DE': ['der', 'die', 'das', 'ein', 'eine', 'und', 'ist', 'in', 'zu', 'den', 'mit', 'von', 'auf', 'für', 'nicht', 'ich', 'du', 'er', 'sie', 'wir'],
        'IT': ['il', 'la', 'lo', 'i', 'gli', 'le', 'di', 'a', 'da', 'in', 'con', 'su', 'per', 'che', 'non', 'un', 'una', 'sono', 'come', 'questo', 'questa'],
    }
    
    # Bigramas discriminativos
    DISCRIMINATIVE_BIGRAMS = {
        'ES': ['ue', 'ió', 'ón', 'ño', 'qu', 'gu'],
        'EN': ['th', 'ng', 'ed', 'ly', 'tion', 'ight'],
        'PT': ['ão', 'nh', 'lh', 'çã', 'ões'],
        'FR': ['ou', 'ai', 'eau', 'oi', 'aux'],
        'DE': ['ch', 'sch', 'ei', 'ie', 'ung'],
        'IT': ['zz', 'cc', 'gg', 'gli', 'gn'],
    }
    
    def detect(self, text: str) -> LanguageResult:
        """
        Detecta idioma del texto.
        
        Prioridad:
        1. ALPHA - discriminadores absolutos
        2. BETA - estadísticos
        3. GAMMA - inferencia por defecto
        """
        if not text or len(text.strip()) < 2:
            return LanguageResult('ES', 0.1, 'GAMMA', 'default_short')
        
        text_lower = text.lower()
        
        # === CAPA ALPHA ===
        # Detectar script no latino
        script = self._detect_script(text)
        if script:
            lang = self.SCRIPT_MARKERS.get(script, 'UNK')
            return LanguageResult(lang, 0.95, 'ALPHA', f'script:{script}')
        
        # Detectar caracteres únicos
        for char, lang in self.ABSOLUTE_MARKERS.items():
            if char in text_lower:
                return LanguageResult(lang, 0.90, 'ALPHA', f'char:{char}')
        
        # === CAPA BETA ===
        # Contar palabras función
        words = set(text_lower.split())
        scores = {}
        
        for lang, func_words in self.FUNCTION_WORDS.items():
            matches = len(words.intersection(set(func_words)))
            if matches > 0:
                scores[lang] = matches
        
        if scores:
            best_lang = max(scores, key=scores.get)
            confidence = min(0.85, 0.3 + scores[best_lang] * 0.1)
            
            # Discriminador PT vs ES (muy similares)
            if best_lang in ['ES', 'PT'] and scores.get('ES', 0) > 0 and scores.get('PT', 0) > 0:
                # Buscar discriminadores específicos
                if any(w in words for w in ['não', 'você', 'isso', 'muito', 'também']):
                    return LanguageResult('PT', 0.85, 'BETA', 'pt_specific_words')
                if any(w in words for w in ['pero', 'muy', 'también', 'esto', 'ese']):
                    return LanguageResult('ES', 0.85, 'BETA', 'es_specific_words')
            
            return LanguageResult(best_lang, confidence, 'BETA', f'function_words:{scores[best_lang]}')
        
        # Contar bigramas
        bigram_scores = {}
        for lang, bigrams in self.DISCRIMINATIVE_BIGRAMS.items():
            count = sum(1 for bg in bigrams if bg in text_lower)
            if count > 0:
                bigram_scores[lang] = count
        
        if bigram_scores:
            best_lang = max(bigram_scores, key=bigram_scores.get)
            return LanguageResult(best_lang, 0.6, 'BETA', f'bigrams:{bigram_scores[best_lang]}')
        
        # === CAPA GAMMA ===
        # Por defecto, inferir del contexto o usar ES
        return LanguageResult('ES', 0.3, 'GAMMA', 'default_fallback')
    
    def _detect_script(self, text: str) -> Optional[str]:
        """Detecta script no latino."""
        for char in text:
            code = ord(char)
            
            # Cirílico
            if 0x0400 <= code <= 0x04FF:
                return 'cyrillic'
            # Árabe
            if 0x0600 <= code <= 0x06FF:
                return 'arabic'
            # Hebreo
            if 0x0590 <= code <= 0x05FF:
                return 'hebrew'
            # Chino (CJK)
            if 0x4E00 <= code <= 0x9FFF:
                return 'chinese'
            # Hiragana/Katakana (Japonés)
            if 0x3040 <= code <= 0x30FF:
                return 'japanese'
            # Hangul (Coreano)
            if 0xAC00 <= code <= 0xD7AF:
                return 'korean'
            # Griego
            if 0x0370 <= code <= 0x03FF:
                return 'greek'
            # Thai
            if 0x0E00 <= code <= 0x0E7F:
                return 'thai'
            # Devanagari
            if 0x0900 <= code <= 0x097F:
                return 'devanagari'
        
        return None


# Singleton
_instance = None

def get_language_detector() -> LanguageDetector:
    global _instance
    if _instance is None:
        _instance = LanguageDetector()
    return _instance


def detect_language(text: str) -> LanguageResult:
    """Función de conveniencia."""
    return get_language_detector().detect(text)
