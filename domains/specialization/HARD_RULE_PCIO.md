# PRINCIPIO DE CAUSALIDAD INTERNA OBLIGATORIA (PCIO)

## NORMA DURA SUPREMA

---

## Definicion

Toda transicion del sistema --tension, dominio, tarea, nivel, promocion, etiqueta emergente-- debe derivarse **exclusivamente** de metricas internas, medidas objetivamente dentro del estado del sistema y **sin intervencion externa**.

---

## VIOLA PCIO cualquier cosa que use:

- Nombres de agentes
- Roles predefinidos
- Reglas heuristicas
- if/else externos
- Preferencias humanas
- Decisiones manuales
- Pesos no derivados de datos internos
- "Intuiciones", "atajos" o logica fuera de metricas

---

## Condiciones Obligatorias

1. **Cada decision debe incluir su `source_metrics` en el YAML.**
   - Toda decision registrada debe tener trazabilidad completa.

2. **Cada transicion debe ser explicable solo desde las metricas internas.**
   - No puede haber factores ocultos o externos.

3. **Cada test debe fallar si aparece cualquier factor exogeno.**
   - Los tests son guardianes de la endogeneidad.

4. **Toda decision debe ser reproducible si el estado interno es identico.**
   - Determinismo controlado.

5. **Toda variacion solo puede provenir del ruido interno controlado por el seed.**
   - El unico azar permitido es el RNG con seed explicito.

---

## Validacion Automatica

### Campos Requeridos en Decisiones

```yaml
decision:
  type: tension_selection | domain_resolution | task_assignment | promotion
  timestamp: ISO8601
  source_metrics:
    - metric_name: tension_intensity_L2
      value: 0.847
      origin: FROM_DATA
    - metric_name: persistence
      value: 0.92
      origin: FROM_DATA
    - metric_name: percentile_rank
      value: 85.3
      origin: FROM_MATH
  derived_from:
    - internal_state_vector
    - tension_accumulator
  external_factors: []  # DEBE estar vacio
  reproducible: true
  seed: 42
```

### Campos Prohibidos

```yaml
# VIOLACION PCIO - NO PERMITIDO
decision:
  agent_name: "EVA"           # PROHIBIDO
  role: "explorer"            # PROHIBIDO
  human_preference: true      # PROHIBIDO
  heuristic_override: true    # PROHIBIDO
  manual_weight: 0.8          # PROHIBIDO
```

---

## Metricas Internas Validas

| Metrica | Descripcion | Origen |
|---------|-------------|--------|
| `tension_intensity_L2` | Norma L2 del vector de tension | FROM_DATA |
| `persistence` | Duracion temporal de la tension | FROM_DATA |
| `delta_intensity` | Cambio de intensidad entre rondas | FROM_DATA |
| `percentile_rank` | Posicion en distribucion | FROM_MATH |
| `performance` | Resultado de tarea ejecutada | FROM_DATA |
| `domain_affinity` | Afinidad calculada tension-dominio | FROM_DATA |
| `task_difficulty` | Dificultad derivada del nivel | FROM_THEORY |

---

## Origenes Permitidos

- **FROM_DATA**: Derivado de observaciones del sistema
- **FROM_MATH**: Calculado mediante formulas matematicas puras
- **FROM_THEORY**: Constante teorica documentada (ej: b-value ~ 1.0)

---

## Tests de Validacion

Cada test PCIO debe verificar:

```python
def test_pcio_compliance(decision):
    # 1. source_metrics presente
    assert 'source_metrics' in decision
    assert len(decision['source_metrics']) > 0

    # 2. Solo metricas internas
    assert decision.uses_only_internal_metrics()

    # 3. Sin factores externos
    assert decision.has_no_external_factors()
    assert decision.get('external_factors', []) == []

    # 4. Reproducibilidad
    assert decision.get('reproducible', False) == True

    # 5. Seed explicito si hay variacion
    if decision.has_variation():
        assert 'seed' in decision
```

---

## Violaciones y Consecuencias

| Violacion | Severidad | Accion |
|-----------|-----------|--------|
| Sin `source_metrics` | CRITICA | Test FAIL, abort |
| Campo externo presente | CRITICA | Test FAIL, abort |
| Dependencia de nombre | ALTA | Test FAIL |
| Rama determinista externa | ALTA | Test FAIL |
| Etiqueta usada para decidir | MEDIA | Warning + review |
| Peso no derivado | CRITICA | Test FAIL, abort |

---

## Flujo de Decision Valido PCIO

```
Estado Interno
     |
     v
Metricas Objetivas (tension, performance, etc.)
     |
     v
Calculo Matematico (percentiles, rankings, etc.)
     |
     v
Decision Derivada (con source_metrics completo)
     |
     v
Registro en YAML (trazabilidad total)
     |
     v
Validacion PCIO (tests automaticos)
```

---

## Ejemplos de Decisiones Validas

### Seleccion de Tension (VALIDO)

```yaml
decision:
  type: tension_selection
  selected: empirical_gap
  source_metrics:
    - metric_name: intensity_L2
      value: 0.89
      origin: FROM_DATA
    - metric_name: persistence
      value: 0.95
      origin: FROM_DATA
    - metric_name: percentile_rank
      value: 92.3
      origin: FROM_MATH
  derived_from:
    - tension_accumulator_state
  external_factors: []
  reproducible: true
  seed: 42
```

### Resolucion de Dominio (VALIDO)

```yaml
decision:
  type: domain_resolution
  selected: cosmology
  candidates: [physics, cosmology, medicine]
  source_metrics:
    - metric_name: affinity_score_physics
      value: 0.72
      origin: FROM_DATA
    - metric_name: affinity_score_cosmology
      value: 0.88
      origin: FROM_DATA
    - metric_name: affinity_score_medicine
      value: 0.65
      origin: FROM_DATA
  derived_from:
    - domain_mapping_matrix
    - tension_vector
  external_factors: []
  reproducible: true
```

---

## Auditoria PCIO

Para auditar sesiones:

```python
from domains.specialization.pcio_validator import run_pcio_audit

results = run_pcio_audit(
    logs_path="/root/NEO_EVA/logs/observation/sessions",
    session_range=(1, 2000)
)

print(f"Decisions audited: {results['total_decisions']}")
print(f"PCIO compliant: {results['compliant']}")
print(f"Violations found: {results['violations']}")
```

---

## Mantra PCIO

> "Si no esta en las metricas internas, no existe para el sistema."
> "Si no tiene source_metrics, no es una decision valida."
> "Si usa algo externo, viola PCIO y debe fallar."

---

*Norma Dura Suprema - Principio de Causalidad Interna Obligatoria*
*Version 1.0 - NEO_EVA Investigation*
