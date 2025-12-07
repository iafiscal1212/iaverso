# NORMA DURA SUPREMA: PRINCIPIO DE CAUSALIDAD INTERNA OBLIGATORIA (PCIO)

## Version 1.0 - NEO_EVA Investigation

---

## 1. Definicion Formal

El **Principio de Causalidad Interna Obligatoria (PCIO)** establece que:

> Toda transicion del sistema --tension, dominio, tarea, nivel, promocion, etiqueta emergente-- debe derivarse **exclusivamente** de metricas internas, medidas objetivamente dentro del estado del sistema y **sin intervencion externa**.

---

## 2. Axiomas PCIO

### Axioma 1: Clausura Causal
Ninguna decision del sistema puede depender de informacion que no este contenida en el estado interno medible.

### Axioma 2: Trazabilidad Total
Toda decision debe poder reconstruirse a partir de sus `source_metrics` documentadas.

### Axioma 3: Reproducibilidad Determinista
Dado el mismo estado interno y seed, la misma decision debe producirse siempre.

### Axioma 4: Ausencia de Agencia Externa
El sistema no puede ser influenciado por preferencias, roles, nombres, o cualquier etiqueta que no sea una metrica interna.

---

## 3. Violaciones PCIO

### 3.1 Viola PCIO cualquier cosa que use:

| Elemento Prohibido | Razon |
|--------------------|-------|
| Nombres de agentes | Introduce identidad externa |
| Roles predefinidos | Sesgo de comportamiento externo |
| Reglas heuristicas | Logica no derivada de datos |
| if/else externos | Ramas no basadas en metricas |
| Preferencias humanas | Intervencion externa directa |
| Decisiones manuales | Violacion de autonomia |
| Pesos no derivados | Parametros no justificados |
| "Intuiciones" | Logica no formalizable |
| "Atajos" | Bypass de metricas |

### 3.2 Ejemplos de Violaciones

```python
# VIOLACION: Usar nombre de agente
if agent.name == "EVA":
    behavior = "exploratory"  # PROHIBIDO

# VIOLACION: Preferencia humana
weight = user_config.get('preferred_domain_weight')  # PROHIBIDO

# VIOLACION: Heuristica fija
if tension_count > 5:  # PROHIBIDO - umbral no derivado
    select_domain("physics")

# VIOLACION: Peso manual
domain_score = base_score * 0.8  # PROHIBIDO - 0.8 no derivado
```

### 3.3 Ejemplos Correctos

```python
# CORRECTO: Derivado de datos
threshold = np.percentile(tensions, 75)  # FROM_MATH
if tension_value > threshold:
    process_high_tension()

# CORRECTO: Metrica interna
domain_score = affinity_matrix[tension_idx, domain_idx]  # FROM_DATA

# CORRECTO: Teoria documentada
b_value = 1.0  # FROM_THEORY: Gutenberg-Richter
```

---

## 4. Condiciones Obligatorias

### 4.1 Estructura de Decision Valida

Toda decision DEBE incluir:

```yaml
decision:
  type: string  # tension_selection | domain_resolution | task_assignment | promotion
  timestamp: ISO8601
  source_metrics:
    - metric_name: string
      value: number
      origin: FROM_DATA | FROM_MATH | FROM_THEORY
  derived_from:
    - internal_state_component
  external_factors: []  # SIEMPRE vacio
  reproducible: true
  seed: integer  # Si hay variacion estocastica
```

### 4.2 Origenes Permitidos

| Origen | Descripcion | Ejemplo |
|--------|-------------|---------|
| `FROM_DATA` | Derivado de observaciones del sistema | `performance: 0.85` |
| `FROM_MATH` | Calculado mediante formulas puras | `percentile: 92.3` |
| `FROM_THEORY` | Constante teorica documentada | `b_value: 1.0` |

### 4.3 Campos Prohibidos

Los siguientes campos **nunca** deben aparecer en una decision:

- `agent_name`
- `role`
- `human_preference`
- `heuristic_override`
- `manual_weight`
- `external_input`
- `user_preference`
- `predefined_role`
- `hardcoded_value`

---

## 5. Validacion Automatica

### 5.1 Tests Obligatorios

Cada decision debe pasar:

```python
def validate_pcio(decision):
    # 1. source_metrics presente y no vacio
    assert 'source_metrics' in decision
    assert len(decision['source_metrics']) > 0

    # 2. Solo metricas internas
    for metric in decision['source_metrics']:
        assert metric['origin'] in ['FROM_DATA', 'FROM_MATH', 'FROM_THEORY']

    # 3. Sin factores externos
    assert decision.get('external_factors', []) == []

    # 4. Sin campos prohibidos
    for field in FORBIDDEN_FIELDS:
        assert field not in decision

    # 5. Reproducibilidad
    assert decision.get('reproducible', False) == True

    return True
```

### 5.2 Auditoria de Sesiones

```python
from domains.specialization.pcio_validator import run_pcio_audit

results = run_pcio_audit(
    logs_path="/root/NEO_EVA/logs/observation/sessions",
    session_range=(1, 2000)
)

assert results['violations'] == 0, f"PCIO violations: {results['violations']}"
```

---

## 6. Flujo de Decision PCIO

```
+-------------------+
|  Estado Interno   |
|  (tensiones,      |
|   performance,    |
|   acumuladores)   |
+--------+----------+
         |
         v
+--------+----------+
|  Metricas         |
|  Objetivas        |
|  (L2, persistence,|
|   percentiles)    |
+--------+----------+
         |
         v
+--------+----------+
|  Calculo          |
|  Matematico       |
|  (rankings,       |
|   afinidades)     |
+--------+----------+
         |
         v
+--------+----------+
|  Decision         |
|  Derivada         |
|  (source_metrics  |
|   completo)       |
+--------+----------+
         |
         v
+--------+----------+
|  Registro YAML    |
|  (trazabilidad)   |
+--------+----------+
         |
         v
+--------+----------+
|  Validacion PCIO  |
|  (tests auto)     |
+-------------------+
```

---

## 7. Metricas Internas Validas

| Metrica | Tipo | Formula/Origen |
|---------|------|----------------|
| `tension_intensity_L2` | FROM_DATA | `sqrt(sum(v^2))` |
| `persistence` | FROM_DATA | Duracion temporal |
| `delta_intensity` | FROM_DATA | `I(t) - I(t-1)` |
| `percentile_rank` | FROM_MATH | Posicion en CDF |
| `performance` | FROM_DATA | Resultado de tarea |
| `domain_affinity` | FROM_DATA | Matriz aprendida |
| `task_difficulty` | FROM_THEORY | Nivel academico |
| `promotion_threshold` | FROM_MATH | Percentil 90 |

---

## 8. Consecuencias de Violacion

| Severidad | Tipo de Violacion | Accion |
|-----------|-------------------|--------|
| CRITICA | Sin source_metrics | Test FAIL, Abort |
| CRITICA | Campo externo | Test FAIL, Abort |
| CRITICA | Peso manual | Test FAIL, Abort |
| ALTA | Dependencia de nombre | Test FAIL |
| ALTA | Rama externa | Test FAIL |
| MEDIA | Etiqueta en decision | Warning + Review |

---

## 9. Integracion con Tests Existentes

### 9.1 test_endogenous_hard_fail.py

```python
def test_decision_has_source_metrics(decision):
    assert 'source_metrics' in decision
    assert decision.uses_only_internal_metrics()
    assert decision.has_no_external_factors()
```

### 9.2 test_tension_hard_rules.py

```python
def test_tension_pcio_compliant(tension):
    assert tension.origin in ['FROM_DATA', 'FROM_MATH']
    assert not tension.has_external_dependency()
```

### 9.3 test_tera_hard_fail.py

```python
def test_tera_decisions_pcio(tera_result):
    decision = tera_result.to_decision()
    assert PCIOValidator().validate(decision)
```

---

## 10. Mantra PCIO

> **"Si no esta en las metricas internas, no existe para el sistema."**

> **"Si no tiene source_metrics, no es una decision valida."**

> **"Si usa algo externo, viola PCIO y debe fallar."**

> **"La unica variacion permitida viene del seed."**

> **"Toda decision debe ser explicable solo desde el estado interno."**

---

## 11. Referencias

- `domains/specialization/HARD_RULE_PCIO.md` - Definicion tecnica
- `tests/test_PCIO_hard_fail.py` - Tests de validacion
- `tests/test_endogenous_hard_fail.py` - Tests de endogeneidad
- `tests/test_tension_hard_rules.py` - Tests de tensiones
- `tests/test_tera_hard_fail.py` - Tests del nucleo TERA

---

*Norma Dura Suprema - Principio de Causalidad Interna Obligatoria*
*NEO_EVA Investigation - Version 1.0*
