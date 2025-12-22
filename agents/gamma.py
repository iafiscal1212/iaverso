"""
Gamma - Narrador Estructural

NORMA DURA:
- Gamma EXPLICA el proceso, no da veredictos
- Gamma NARRA lo que observa, paso a paso
- Gamma DESCRIBE opciones, no elige "la mejor"
- Gamma ABRE preguntas, no las cierra

Gamma es como un científico que narra su observación:
"Estoy viendo que E3 es 0.8. Eso indica alta complejidad estructural.
Si aplico T_reduce, predigo que bajará a 0.5.
Puedo probar eso, o puedo explorar otras direcciones.
¿Qué camino tomamos?"

NO es:
"La complejidad es demasiado alta. Debemos reducirla.
Esto demuestra que el sistema es inestable."

Rol: NARRADOR de Inferencia Activa
"Veo esto... puedo hacer esto... si hago esto espero esto..."
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import sys
sys.path.insert(0, '/opt/iaverso')

from core.endolens import get_endolens, StructuralState
from core.neosynt import get_neosynt, Resolution
from lab.genetic_lab import get_genetic_lab, Culture, Experiment, ExperimentResult


@dataclass
class Observation:
    """Una observación narrada."""
    what: str           # Qué veo
    numbers: Dict       # Los números
    context: str        # Contexto de la observación


@dataclass
class Option:
    """Una opción disponible (no "recomendación")."""
    action: str         # Qué puedo hacer
    prediction: str     # Qué espero que pase
    uncertainty: str    # Qué no sé


@dataclass
class Narration:
    """Narración completa de un ciclo de investigación."""
    opening: str        # Situación inicial
    observations: List[str]  # Lo que veo
    options: List[Option]    # Lo que puedo hacer
    questions: List[str]     # Preguntas abiertas
    next_steps: str     # Posibles caminos (sin elegir)


class GammaInvestigator:
    """
    Narrador estructural.
    
    Gamma narra el proceso de investigación como si estuviera
    pensando en voz alta. No da conclusiones, abre conversación.
    """
    
    def __init__(self):
        self.endolens = get_endolens()
        self.neosynt = get_neosynt()
        self.lab = get_genetic_lab()
        self.history: List[Narration] = []
        self.cycle = 0
        
    def narrate(
        self,
        query: str,
        state: StructuralState,
        resolution: Resolution = None
    ) -> Narration:
        """
        Narra lo que está pasando.
        
        No concluye. Explica y abre opciones.
        """
        self.cycle += 1
        
        # === APERTURA ===
        opening = self._narrate_opening(query, state)
        
        # === OBSERVACIONES ===
        observations = self._narrate_observations(state, resolution)
        
        # === OPCIONES ===
        options = self._narrate_options(state)
        
        # === PREGUNTAS ===
        questions = self._narrate_questions(state, resolution)
        
        # === SIGUIENTE ===
        next_steps = self._narrate_next_steps(state, options)
        
        narration = Narration(
            opening=opening,
            observations=observations,
            options=options,
            questions=questions,
            next_steps=next_steps
        )
        
        self.history.append(narration)
        return narration
    
    def _narrate_opening(self, query: str, state: StructuralState) -> str:
        """Narra la situación inicial."""
        return f"""Ciclo {self.cycle}. Estoy analizando: "{query}"

Veo una estructura con firma {state.signature}.
El estado es {state.status} con estabilidad {state.stability:.1%}.
"""
    
    def _narrate_observations(self, state: StructuralState, resolution: Resolution) -> List[str]:
        """Narra las observaciones."""
        obs = []
        
        # E-series
        e = state.eseries
        obs.append(f"La E-series muestra: E0={e.E0:.3f}, E1={e.E1:.3f}, E2={e.E2:.3f}, E3={e.E3:.3f}")
        
        # Interpretar E-series (sin juicio)
        if e.E0 < 0.1:
            obs.append("E0 bajo indica que la estructura tiene alta persistencia - se mantiene a través de variaciones.")
        elif e.E0 > 0.5:
            obs.append("E0 alto indica baja persistencia - la estructura cambia fácilmente.")
        
        if e.E1 > 0.5:
            obs.append("E1 alto indica alta variabilidad - hay mucha diversidad interna.")
        elif e.E1 < 0.2:
            obs.append("E1 bajo indica baja variabilidad - la estructura es más homogénea.")
        
        if e.E3 > 0.6:
            obs.append("E3 alto indica alta complejidad - hay muchas relaciones entre componentes.")
        elif e.E3 < 0.3:
            obs.append("E3 bajo indica baja complejidad - la estructura es más simple.")
        
        # Invariantes
        if state.invariants:
            obs.append(f"Detecté {len(state.invariants)} invariantes - propiedades que no cambian bajo transformación:")
            for inv in state.invariants[:3]:  # Mostrar hasta 3
                obs.append(f"  - {inv}")
        
        # Tensiones
        if state.tensions:
            obs.append(f"Observo {len(state.tensions)} tensiones estructurales:")
            for t in state.tensions:
                if t == 'high_persistence_high_variability':
                    obs.append("  - Alta persistencia con alta variabilidad: esto es contradictorio, podría indicar una estructura en transición.")
                elif t == 'complex_but_static':
                    obs.append("  - Complejidad alta pero estática: hay muchas relaciones pero poca dinámica.")
                elif t == 'strong_directional_tendency':
                    obs.append("  - Tendencia direccional fuerte: la estructura parece estar yendo hacia algun lado.")
        
        # Resolución (si hay)
        if resolution:
            obs.append(f"\nNeoSynt intentó resolver hacia un estado más estable.")
            obs.append(f"El resultado fue: {resolution.status}")
            if resolution.alternatives:
                obs.append(f"Generó {len(resolution.alternatives)} alternativas posibles.")
        
        return obs
    
    def _narrate_options(self, state: StructuralState) -> List[Option]:
        """Narra las opciones disponibles."""
        options = []
        
        # Opción 1: Estabilizar
        options.append(Option(
            action="Aplicar T_stabilize (intentar estabilizar)",
            prediction=f"Espero que la estabilidad suba de {state.stability:.1%} hacia ~{min(1, state.stability + 0.2):.1%}",
            uncertainty="No sé si funcionará - la estructura podría resistir el cambio."
        ))
        
        # Opción 2: Reducir complejidad
        if state.eseries.E3 > 0.4:
            options.append(Option(
                action="Aplicar T_reduce (reducir complejidad)",
                prediction=f"Espero que E3 baje de {state.eseries.E3:.3f} hacia ~{state.eseries.E3 * 0.7:.3f}",
                uncertainty="Reducir complejidad podría perder información estructural."
            ))
        
        # Opción 3: Explorar
        options.append(Option(
            action="Generar alternativas con el laboratorio genético",
            prediction="Obtendré 5-10 variantes de esta estructura para comparar.",
            uncertainty="Las variantes podrían ser muy diferentes o muy similares."
        ))
        
        # Opción 4: Evolucionar
        options.append(Option(
            action="Crear un cultivo y dejarlo evolucionar",
            prediction="Después de N generaciones, veré qué estructuras sobreviven.",
            uncertainty="El proceso es estocástico - cada ejecución puede dar resultados diferentes."
        ))
        
        # Opción 5: No hacer nada
        options.append(Option(
            action="Observar sin intervenir",
            prediction="Mantengo el estado actual sin modificarlo.",
            uncertainty="Quizás hay algo que estoy pasando por alto."
        ))
        
        return options
    
    def _narrate_questions(self, state: StructuralState, resolution: Resolution) -> List[str]:
        """Formula preguntas abiertas (no respuestas)."""
        questions = []
        
        questions.append("¿Qué camino exploramos primero?")
        
        if state.tensions:
            questions.append("Las tensiones que observo, ¿son un problema o una característica?")
        
        if state.stability < 0.5:
            questions.append("¿Buscamos estabilidad, o la inestabilidad nos dice algo?")        
        if state.eseries.E3 > 0.6:
            questions.append("¿La complejidad es inherente o es ruido que podemos reducir?")
        
        if resolution and resolution.alternatives:
            questions.append("Hay alternativas disponibles. ¿Las exploramos?")
        
        questions.append("¿Hay algo más que debería mirar?")
        
        return questions
    
    def _narrate_next_steps(self, state: StructuralState, options: List[Option]) -> str:
        """Narra posibles caminos sin elegir."""
        return f"""
Tengo {len(options)} opciones disponibles.
Puedo intentar modificar la estructura, explorar alternativas, o simplemente observar más.
No sé cuál es "mejor" - eso depende de qué estemos buscando.

Si me dices qué dirección tomar, ejecuto la acción y te cuento qué pasó.
O puedo explorar automáticamente siguiendo criterios estructurales (no semánticos).
"""
    
    def execute_and_narrate(
        self,
        option_index: int,
        current_state: StructuralState
    ) -> str:
        """
        Ejecuta una opción y narra el resultado.
        """
        narration = []
        
        narration.append(f"Ejecutando opción {option_index + 1}...\n")
        
        # Crear cultivo
        culture = self.lab.create_culture(
            seed="investigation_seed",
            population_size=10
        )
        
        narration.append(f"Creé un cultivo con {culture.size} especímenes.")
        narration.append(f"El mejor tiene fitness {culture.best_specimen.fitness:.3f}\n")
        
        # Proponer experimento
        experiment = self.lab.propose_experiment(culture)
        narration.append(f"El laboratorio propone: {experiment.action}")
        narration.append(f"Predicción: estabilidad esperada = {experiment.prediction.expected_stability:.3f}")
        narration.append(f"Incertidumbre: {experiment.uncertainty:.1%}\n")
        
        # Ejecutar
        result = self.lab.execute_experiment(experiment)
        
        narration.append("Resultado:")
        narration.append(f"  - Sorpresa: {result.surprise:.4f} (menor = predicción más acertada)")
        narration.append(f"  - Éxito: {'Sí' if result.success else 'No'}")
        narration.append(f"  - Estabilidad observada: {result.observed.stability:.3f}")
        narration.append(f"  - Nueva firma: {result.observed.signature}\n")
        
        if result.surprise < 0.1:
            narration.append("La predicción fue bastante precisa. El modelo parece entender bien esta estructura.")
        elif result.surprise > 0.3:
            narration.append("La predicción falló significativamente. Hay algo que el modelo no captura.")
            narration.append("Esto es interesante - podría indicar comportamiento no lineal.")
        
        narration.append("\n¿Qué hacemos ahora?")
        
        return '\n'.join(narration)
    
    def format_full_narration(self, narration: Narration) -> str:
        """Formatea la narración completa para mostrar."""
        lines = []
        
        lines.append(narration.opening)
        lines.append("\n--- QUÉ OBSERVO ---\n")
        for obs in narration.observations:
            lines.append(obs)
        
        lines.append("\n--- QUÉ PUEDO HACER ---\n")
        for i, opt in enumerate(narration.options):
            lines.append(f"{i+1}. {opt.action}")
            lines.append(f"   Predicción: {opt.prediction}")
            lines.append(f"   Incertidumbre: {opt.uncertainty}\n")
        
        lines.append("--- PREGUNTAS ABIERTAS ---\n")
        for q in narration.questions:
            lines.append(f"• {q}")
        
        lines.append("\n--- SIGUIENTE ---")
        lines.append(narration.next_steps)
        
        return '\n'.join(lines)


# Singleton
_instance = None

def get_gamma() -> GammaInvestigator:
    global _instance
    if _instance is None:
        _instance = GammaInvestigator()
    return _instance
