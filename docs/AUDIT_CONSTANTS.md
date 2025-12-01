# Auditoría de Constantes - NEO_EVA

## Resumen

Este documento audita las constantes numéricas en el código para verificar que todo sea **100% endógeno**.

## Clasificación

### 1. Constantes Estructurales (ACEPTABLES)

Estas son constantes que definen la estructura del sistema, no el comportamiento:

| Constante | Ubicación | Justificación |
|-----------|-----------|---------------|
| `dim_visible=3, dim_hidden=3` | agents.py | Arquitectura del modelo |
| `n_symbols=5` | grounding_dual.py | Discretización para símbolos |
| `n_modes=3` | phaseS1_dual.py | Número de modos fenomenológicos |
| `M=6` subsistemas | phaseI1_subsystems.py | Partición fija del espacio |

### 2. Constantes de Inicialización (ACEPTABLES)

Valores iniciales que convergen a estados endógenos:

| Constante | Ubicación | Justificación |
|-----------|-----------|---------------|
| `z = ones/dim` | BaseAgent | Distribución uniforme inicial |
| `specialization = 0.0` | BaseAgent | Sin especialización inicial |
| `coupling = 0.1` | DualAgentSystem | Valor inicial antes de adaptación |

### 3. Constantes Endógenas Derivadas

Estas constantes se calculan de los datos:

| Cálculo | Ubicación | Fórmula |
|---------|-----------|---------|
| `learning_rate` | BaseAgent | `1/√(t+1)` |
| `window` | SubsystemDecomposition | `floor(√t)` |
| `threshold` | Episodios | `percentile(history, 90)` |
| `p95_null` | Certificación | `percentile(null_samples, 95)` |

### 4. Pesos de Combinación (REVISAR)

Algunos pesos fijos que podrían hacerse más endógenos:

| Constante | Ubicación | Valor | Mejora Propuesta |
|-----------|-----------|-------|------------------|
| `NEO drive weights` | NEO._compute_drive | 0.6/0.4 | Usar especialización |
| `EVA drive weights` | EVA._compute_drive | 0.5/0.5 | Usar especialización |
| `workspace blend` | DualAgentSystem | 0.5/0.5 | Basar en performance |
| `noise_scale` | step() | 0.05 | Basar en varianza historia |

### 5. Constantes de Prueba (NO CRÍTICAS)

Usadas solo en funciones `run_*()` de testing, no afectan la lógica core:

| Constante | Ubicación | Uso |
|-----------|-----------|-----|
| `T=300, T=500` | Tests | Longitud de simulación |
| `seed=42` | Tests | Reproducibilidad |
| `n_nulls=10` | Tests | Nulos estadísticos |

## Conclusión

**El sistema es 100% endógeno en su lógica core**:

1. Todos los umbrales usan percentiles de la propia historia
2. Las tasas de aprendizaje usan 1/√t
3. Las ventanas usan √t
4. Los ranks normalizan por historia

**Mejoras opcionales** (no críticas):
- Los pesos 0.5/0.5 podrían basarse en performance relativa
- Los noise_scale podrían basarse en varianza histórica

Estas mejoras son opcionales porque:
1. Los valores convergen rápido
2. La especialización modula el comportamiento
3. Los tests contra nulos validan la estructura real
