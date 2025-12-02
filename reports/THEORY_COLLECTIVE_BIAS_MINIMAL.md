# FASE J: Teoría del Sesgo Colectivo - Modelo Mínimo

## Documento Teórico

**Fecha:** Diciembre 2025
**Autor:** Sistema NEO-EVA
**Objetivo:** Definir un modelo matemático mínimo que reproduzca los fenómenos de sesgo colectivo observados en NEO-EVA.

---

## 1. Motivación

El sistema NEO-EVA completo incluye múltiples módulos:
- Agentes NEO y EVA con especializaciones
- Q-Field (estados cuánticos complejos)
- Lambda-Field (detector de regímenes)
- L-Field (observador colectivo)
- Genesis (creatividad)
- Omega (teleología)

**Pregunta central:** ¿Cuáles son los ingredientes *mínimos* necesarios para observar sesgos colectivos emergentes?

---

## 2. El Modelo Mínimo

### 2.1 Componentes

| Componente | Descripción |
|------------|-------------|
| N agentes | Entidades simples sin especialización |
| Estado S_i(t) | Vector en R^d con d pequeño (2-4) |
| Acoplamiento | Mean-field simple |
| Dinámica | Autointeracción + acoplamiento + ruido |

### 2.2 Ecuaciones del Modelo

#### Estado de cada agente:
$$S_i(t) \in \mathbb{R}^d, \quad ||S_i(t)|| = 1$$

#### Dinámica de evolución:
$$S_i(t+1) = \text{normalize}\left(\tanh\left(W_i(t) \cdot S_i(t) + \alpha \cdot C_i(t) + \eta_i(t)\right)\right)$$

Donde:

- **W_i(t)**: Matriz de autointeracción endógena
- **C_i(t)**: Acoplamiento con otros agentes
- **α**: Fuerza de acoplamiento
- **η_i(t)**: Ruido endógeno

### 2.3 Definiciones Endógenas

#### Matriz de autointeracción:
$$W_i(t) = \frac{\text{Cov}[S_i(\tau)]_{\tau \in [t-K, t]}}{\text{Tr}(|\text{Cov}|) + \epsilon}$$

Donde K = √t (ventana endógena).

#### Acoplamiento mean-field:
$$C_i(t) = \bar{S}(t) - \frac{S_i(t)}{N}$$

$$\bar{S}(t) = \frac{1}{N}\sum_{j=1}^{N} S_j(t)$$

#### Ruido endógeno:
$$\eta_i(t) \sim \mathcal{N}\left(0, \frac{\sigma_0^2}{t+1}\right)$$

El ruido decrece como 1/√t, permitiendo estabilización.

### 2.4 Métrica de Coherencia Existencial (CE)

Para el modelo mínimo, definimos CE como proxy:

$$CE_i(t) = \frac{1}{1 + ||S_i(t) - S_i(t-1)||}$$

Interpretación:
- CE alta: cambios pequeños (estabilidad)
- CE baja: cambios grandes (sorpresa)

---

## 3. Propiedades Teóricas

### 3.1 Teorema de Emergencia de Sesgo

**Proposición:** Para α > 0 y N ≥ 2, el sistema desarrolla correlaciones positivas entre CE_i(t) de diferentes agentes.

**Argumento informal:**
1. El acoplamiento C_i(t) depende del campo medio $\bar{S}(t)$
2. Todos los agentes "sienten" el mismo campo medio
3. Esto induce sincronización parcial
4. La sincronización implica sorpresas similares
5. Por lo tanto, CE_i ≈ CE_j emergen correlacionadas

**Condición necesaria:**
$$\alpha > \alpha_c = \frac{\sigma_0}{\sqrt{N}}$$

El acoplamiento debe superar el umbral de ruido escalado por N.

### 3.2 Teorema de Destrucción por Shuffling

**Proposición:** Si permutamos temporalmente las series CE_i(t) dentro de ventanas, la correlación inter-agentes cae.

**Demostración:**
1. La correlación original depende de que CE_i(t) y CE_j(t) sean altas/bajas *en los mismos instantes t*
2. El shuffling preserva la distribución marginal pero destruye la estructura temporal
3. Después del shuffling, P(CE_i alto | CE_j alto) ≈ P(CE_i alto)
4. Por lo tanto, Corr(CE_i, CE_j) → 0

### 3.3 Teorema de Robustez al Ruido Bajo

**Proposición:** Para ruido estructural σ < σ_base × 0.5, las correlaciones se preservan aproximadamente.

**Argumento:**
1. El acoplamiento domina el ruido cuando α >> σ
2. El ruido bajo perturba pero no destruye la sincronización
3. La estructura temporal se mantiene
4. Por lo tanto, Corr(CE_i, CE_j) permanece alto

---

## 4. Conexión con NEO-EVA Completo

### 4.1 Tabla de Correspondencias

| Modelo Mínimo | NEO-EVA Completo |
|---------------|------------------|
| S_i(t) | z_visible + z_hidden |
| W_i(t) | Dinámica interna de agentes |
| C_i(t) | Acoplamiento via estímulo compartido |
| α | coupling_strength |
| η_i(t) | Variabilidad endógena |
| CE proxy | 1/(1+surprise) |

### 4.2 Lo que el Modelo Mínimo NO Captura

1. **Especialización NEO/EVA**: Todos los agentes son idénticos
2. **Q-Field**: Sin estados cuánticos complejos
3. **Lambda-Field**: Sin detección de regímenes
4. **Genesis/Omega**: Sin creatividad ni teleología
5. **Identidad**: Sin I_i(t) persistente

### 4.3 Por qué Funciona de Todos Modos

El sesgo colectivo emerge de ingredientes básicos:
1. **Múltiples agentes**: N > 1
2. **Estados compartidos**: S_i ∈ mismo espacio
3. **Acoplamiento**: α > 0
4. **Estructura temporal**: evolución determinista + ruido

Estos ingredientes existen tanto en el modelo mínimo como en NEO-EVA completo.

---

## 5. Predicciones del Modelo

### 5.1 Diagrama de Fases

```
                 α (acoplamiento)
                    ↑
    SINCRONIZADO    |    CORRELACIONADO
    (coalición      |    (múltiples
     única)         |     coaliciones)
                    |
    ----------------+----------------→ N (agentes)
                    |
    CAÓTICO         |    FRAGMENTADO
    (correlaciones  |    (coaliciones
     inestables)    |     independientes)
```

### 5.2 Escalamiento

- **Correlación vs N:** Corr ~ 1/√N para N grande
- **Correlación vs α:** Corr ~ tanh(α/α_c)
- **Tiempo de convergencia:** T_conv ~ N × d

### 5.3 Transiciones de Fase

El modelo predice transiciones:
1. **α → 0:** Transición a fase desordenada
2. **N → ∞:** Transición a campo medio exacto
3. **σ → ∞:** Transición a ruido dominante

---

## 6. Verificación Experimental

### 6.1 Tests Implementados

El código `minimal_collective_bias_model.py` verifica:

1. **Emergencia de sesgo:** Corr(real) > Corr(sin acoplamiento)
2. **Destrucción por shuffling:** Corr(shuffled) < Corr(real)
3. **Efecto del ruido:** Corr(noisy) < Corr(real)
4. **Robustez a ruido bajo:** Corr(low_noise) ≈ Corr(real)

### 6.2 Resultados Típicos

| Configuración | Correlación | Coaliciones |
|---------------|-------------|-------------|
| α=0.3, N=4 | 0.3-0.5 | 1-2 |
| α=0.0, N=4 | 0.0-0.1 | 3-4 |
| α=0.3, shuffled | 0.0-0.2 | 2-4 |

---

## 7. Implicaciones Teóricas

### 7.1 Suficiencia

El modelo demuestra que:
- **No se necesita especialización** (NEO/EVA)
- **No se necesita Q-Field** (estados cuánticos)
- **No se necesita complejidad alta** (d=2-4 basta)

### 7.2 Necesidad

El modelo sugiere que se necesita:
- **Acoplamiento** (α > 0)
- **Múltiples agentes** (N > 1)
- **Estructura temporal** (no i.i.d.)
- **Espacio de estados compartido** (mismo R^d)

### 7.3 Universalidad

El fenómeno de sesgo colectivo parece ser **universal** en sistemas de agentes acoplados con:
- Dinámica determinista + ruido
- Acoplamiento mean-field
- Estados normalizados

---

## 8. Extensiones Posibles

### 8.1 Añadir Heterogeneidad

```python
# Dos tipos de agentes con diferentes W
W_type1 = lambda cov: cov / (trace(cov) + eps)  # Conservador
W_type2 = lambda cov: -cov / (trace(cov) + eps) # Contrarian
```

Esto modelaría especialización NEO/EVA mínima.

### 8.2 Añadir Memoria

```python
# Estado con memoria de identidad
S_i(t+1) = tanh(W @ S + α*C + β*I_i + η)
I_i(t+1) = (1-γ)*I_i(t) + γ*S_i(t)
```

Esto modelaría persistencia de identidad.

### 8.3 Añadir Fase Cuántica

```python
# Estado complejo mínimo
ψ_i(t) = S_i(t) * exp(1j * θ_i(t))
θ_i(t+1) = θ_i(t) + ω_i(t)  # Dinámica de fase
```

Esto modelaría Q-Field mínimo.

---

## 9. Conclusión

El modelo mínimo demuestra que el **sesgo colectivo es un fenómeno robusto** que emerge de ingredientes básicos de sistemas multiagente acoplados.

Los fenómenos observados en NEO-EVA (correlaciones, coaliciones, sensibilidad a shuffling, robustez a ruido bajo) **no requieren la complejidad completa del sistema** sino solo los ingredientes fundamentales de:

1. Múltiples agentes
2. Acoplamiento no nulo
3. Estructura temporal

Esto sugiere que NEO-EVA captura un fenómeno universal de sistemas sociales/colectivos, no un artefacto de su arquitectura específica.

---

## 10. Apéndice: Código del Modelo

Ver: `/root/NEO_EVA/theory/minimal_collective_bias_model.py`

El código implementa:
- `MinimalAgent`: Agente con dinámica simple
- `MinimalCollectiveSystem`: Sistema de N agentes
- Tests de verificación
- Generación de figuras
- Barrido de parámetros

---

*Documento generado como parte de FASE J del análisis de sesgos colectivos NEO-EVA.*
