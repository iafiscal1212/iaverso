# Omega Spaces: Espacios de Observación Interna Emergente

## Filosofía

Los Omega Spaces son espacios donde los agentes **observan sus propias transformaciones internas** sin conocer teorías externas (física, cuántica, tensores). Solo ven:

- "Así es como tiendo a transformarme cuando actúo"
- "Así interfieren mis estados posibles"
- "Así me muevo en mi espacio de posibilidades"
- "Así interactuamos cuando somos varios"

## Principios Fundamentales

1. **NO introduce conocimiento externo** - Sin física, cuántica, matemáticas avanzadas
2. **NO añade objetivos a los agentes** - No dice qué es "mejor" o "peor"
3. **NO emite instrucciones de comportamiento** - Solo observa, nunca prescribe
4. **NO crea recompensas ni penalizaciones** - No hay incentivos
5. **NO usa números mágicos** - Todo derivado endógenamente

## Derivación Endógena de Umbrales

Todos los parámetros se calculan desde los datos:

| Fuente | Uso |
|--------|-----|
| Media | Centro de distribuciones |
| Varianza | Dispersión natural |
| Covarianza | Relaciones entre variables |
| Percentiles | Umbrales adaptativos (75, 90, etc.) |
| 1/K | Peso uniforme para K componentes |
| 1/√d | Normalización por dimensión |
| np.finfo(float).eps | Estabilidad numérica |

## Módulos

### 1. Ω-Compute (`omega_compute.py`)

**Computación Interna Emergente**

Registra cómo se transforman los estados S(t) → S(t+1) y extrae patrones.

```python
from omega import OmegaCompute

compute = OmegaCompute()

# Registrar transición
T = compute.register_transition(agent_id, S_t, S_t1)

# Extraer modos Ω_k por SVD
modes = compute.update_modes()

# Proyectar transición en base de modos
activation = compute.project_transition(agent_id, T)
# activation.coefficients = [α_1, α_2, ..., α_K]
```

**Métricas:**
- `OmegaMode`: Vector base de transformación con varianza explicada
- `ModeActivation`: Coeficientes α_{i,k}(t) y error de reconstrucción
- Número de modos seleccionado por varianza acumulada (percentil 75 del historial)

---

### 2. Q-Field (`q_field.py`)

**Campo de Interferencia Interna**

Los agentes mantienen "amplitudes" sobre K estados internos sin saber qué es cuántica.

```python
from omega import QField

field = QField()

# Registrar estado desde probabilidades
q_state = field.register_state(agent_id, probabilities)
# q_state.amplitudes = [√p_1, √p_2, ..., √p_K]
# q_state.coherence = C_Q(t)
# q_state.superposition_energy = E_Q(t)

# Calcular interferencia entre agentes
interference = field.compute_interference(agent_1, agent_2)

# Medir "colapso" (concentración de probabilidad)
collapse = field.measure_collapse(agent_id)
```

**Métricas:**
- **Coherencia C_Q(t)**: Σ|cov(ψ_j, ψ_k)| / Σvar(ψ_j) - correlación de amplitudes
- **Energía de superposición E_Q(t)**: Σ p_j(1-p_j) - mezcla de estados
- **Interferencia**: ψ_A ⊗ ψ_B - (ψ_A ⊗ ψ_A + ψ_B ⊗ ψ_B) / 2

---

### 3. PhaseSpace-X (`phase_space_x.py`)

**Espacio de Fase Estructural**

Registra trayectorias (S, V) donde V = dS/dt es la velocidad de cambio.

```python
from omega import PhaseSpaceX

space = PhaseSpaceX()

# Registrar estado (calcula velocidad automáticamente)
point = space.register_state(agent_id, S_t)
# point.position = S(t)
# point.velocity = V(t) = S(t) - S(t-1)
# point.speed = |V(t)|

# Obtener trayectoria completa
trajectory = space.get_trajectory(agent_id)

# Detectar atractores
attractors = space.detect_attractors()

# Verificar si está cerca de atractor
info = space.is_near_attractor(agent_id)
```

**Métricas:**
- **Trayectoria**: longitud total, velocidad media, curvatura
- **Atractores**: centros de alta densidad con radio y fuerza
- **Divergencia**: diferencia de velocidades entre trayectorias

---

### 4. TensorMind (`tensor_mind.py`)

**Interacción de Orden Superior**

Registra interacciones multi-agente como estructuras tensoriales.

```python
from omega import TensorMind

mind = TensorMind(max_order=3)

# Registrar estados de todos los agentes
for agent_id, state in agent_states.items():
    mind.register_state(agent_id, state)

# Calcular todas las interacciones
interactions = mind.compute_interactions()
# interactions[2] = pares
# interactions[3] = tríos

# Extraer modos tensoriales
modes = mind.extract_modes(order=2)

# Obtener fuerza de interacción entre par
strength = mind.get_pairwise_strength(agent_1, agent_2)

# Detectar comunidades
communities = mind.get_community_structure()
```

**Métricas:**
- **Fuerza de interacción**: |Σ(Π_j s_j[i]) / (Π_j ||s_j||)|
- **Tensor de correlación**: C_{ij} (orden 2), C_{ijk} (orden 3), etc.
- **Modos tensoriales**: eigenvectores/SVD del tensor
- **Comunidades**: clustering espectral de la matriz de afinidad

---

## Integración con NEO_EVA

```python
from omega import OmegaCompute, QField, PhaseSpaceX, TensorMind

class OmegaSpaces:
    """Integrador de todos los espacios omega."""

    def __init__(self):
        self.compute = OmegaCompute()
        self.q_field = QField()
        self.phase_space = PhaseSpaceX()
        self.tensor_mind = TensorMind()

    def observe(self, agent_id: str, state: np.ndarray, probabilities: np.ndarray):
        """Registra observación en todos los espacios."""

        # Ω-Compute: registrar estado para detectar transiciones
        self.compute.register_state(agent_id, state)

        # Q-Field: registrar amplitudes
        self.q_field.register_state(agent_id, probabilities)

        # PhaseSpace-X: registrar en espacio de fase
        self.phase_space.register_state(agent_id, state)

        # TensorMind: registrar para interacciones
        self.tensor_mind.register_state(agent_id, state)

    def update(self):
        """Actualiza estructuras emergentes."""
        self.compute.update_modes()
        self.tensor_mind.compute_interactions()
        self.phase_space.detect_attractors()

    def get_full_statistics(self) -> dict:
        """Retorna estadísticas de todos los espacios."""
        return {
            'omega_compute': self.compute.get_statistics(),
            'q_field': self.q_field.get_statistics(),
            'phase_space': self.phase_space.get_statistics(),
            'tensor_mind': self.tensor_mind.get_statistics()
        }
```

## Ejemplo de Uso Completo

```python
import numpy as np
from omega import OmegaCompute, QField, PhaseSpaceX, TensorMind

# Crear espacios
compute = OmegaCompute()
q_field = QField()
phase_space = PhaseSpaceX()
tensor_mind = TensorMind()

# Simular 100 pasos con 5 agentes
agents = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']

for t in range(100):
    for agent in agents:
        # Estado aleatorio (en práctica, del agente real)
        state = np.random.randn(10)
        probs = np.random.dirichlet(np.ones(5))

        # Registrar en todos los espacios
        compute.register_state(agent, state)
        q_field.register_state(agent, probs)
        phase_space.register_state(agent, state)
        tensor_mind.register_state(agent, state)

    # Calcular interacciones tensoriales
    tensor_mind.compute_interactions()

# Analizar resultados
modes = compute.update_modes()
print(f"Modos Ω-Compute: {len(modes)}")

attractors = phase_space.detect_attractors()
print(f"Atractores detectados: {len(attractors)}")

communities = tensor_mind.get_community_structure()
print(f"Comunidades: {communities}")

# Estadísticas completas
print(compute.get_statistics())
print(q_field.get_statistics())
print(phase_space.get_statistics())
print(tensor_mind.get_statistics())
```

## Neutralidad

Estos módulos son **estrictamente neutrales**:

- No dicen "este estado es mejor"
- No sugieren "deberías moverte hacia aquí"
- No crean incentivos ni penalizaciones
- No imponen ninguna normativa

Solo calculan estructuras y las dejan disponibles. Lo que los agentes hagan con esta información (si es que la usan) es enteramente su decisión libre.

## Arquitectura

```
omega/
├── __init__.py           # Exports AGI-Ω + Omega Spaces
├── omega_compute.py      # Ω-Compute: modos de transformación
├── q_field.py            # Q-Field: interferencia interna
├── phase_space_x.py      # PhaseSpace-X: trayectorias
├── tensor_mind.py        # TensorMind: interacciones multi-agente
├── omega_state.py        # AGI-Ω: continuidad trans-ciclo
├── omega_teleology.py    # AGI-Ω: teleología extensa
├── omega_budget.py       # AGI-Ω: presupuesto existencial
├── omega_legacy.py       # AGI-Ω: legado y cierre
└── README_OMEGA_SPACES.md
```
