# AUDITORÍA NORMA DURA - NEO_EVA
**Fecha**: 2025-12-06 (actualizado)
**Estado**: CORRECCIÓN COMPLETADA ✅

## Archivos Corregidos (NORMA DURA COMPLIANT):
- `core/endogenous_constants.py`
- `core/explorer_agent.py`
- `core/meta_drive.py`
- `core/personality_calibration.py` (NUEVO)
- `scripts/agents_free_exploration.py`
- `scripts/agents_solve_paradoxes.py` (NUEVO)
- `worlds/complete_being.py` (~100 violaciones corregidas)
- `subjectivity/phaseS2_self_report_dual.py`

## NORMA DURA (Recordatorio)
> "Ningún número entra al código sin poder explicar de qué distribución de los datos sale"

---

## RESUMEN EJECUTIVO

| Categoría | Archivos afectados | Violaciones estimadas |
|-----------|-------------------|----------------------|
| core/ | 5+ | ~50 |
| scripts/ | 10+ | ~80 |
| worlds/ | 2 | ~100+ |
| subjectivity/ | 4 | ~20 |
| **TOTAL** | **21+** | **250+** |

---

## VIOLACIONES CRÍTICAS POR ARCHIVO

### 1. `worlds/complete_being.py` (MÁS CRÍTICO - ~100 violaciones)

```python
# PROHIBIDO - Umbrales de personalidad sin origen
if dna['sociability'] > 0.7:      # ¿Por qué 0.7?
elif dna['sociability'] < 0.4:    # ¿Por qué 0.4?
if dna['baseline_anxiety'] > 0.3: # ¿Por qué 0.3?
if dna['empathy'] > 0.8:          # ¿Por qué 0.8?
if dna['curiosity'] > 0.8:        # ¿Por qué 0.8?
if dna['risk_tolerance'] > 0.6:   # ¿Por qué 0.6?
if dna['resilience'] > 0.8:       # ¿Por qué 0.8?

# PROHIBIDO - Probabilidades sin justificación
if np.random.random() > 0.15:     # ¿Por qué 15%?
if np.random.random() < 0.1:      # ¿Por qué 10%?
if np.random.random() < 0.3:      # ¿Por qué 30%?

# PROHIBIDO - Umbrales emocionales arbitrarios
if emotions[dominant] < 0.2:      # ¿Por qué 0.2?
if body.pleasure > 0.3:           # ¿Por qué 0.3?
if body.pain > 0.5:               # ¿Por qué 0.5?
if self.emotions.anxiety > 0.5:   # ¿Por qué 0.5?
if self.emotions.fear > 0.6:      # ¿Por qué 0.6?
```

**CORRECCIÓN PROPUESTA**:
```python
# ORIGEN: Todos los umbrales deben calcularse de distribuciones observadas
personality_thresholds = EndogenousThresholds()

# Observar población durante calibración
for being in calibration_population:
    personality_thresholds.observe('sociability', being.dna['sociability'])

# Usar percentiles
sociability_high = personality_thresholds.get('sociability', 'high')  # p90
sociability_low = personality_thresholds.get('sociability', 'low')    # p10

if dna['sociability'] > sociability_high:  # ORIGEN: percentil 90 de población
    ...
```

---

### 2. `core/meta_drive.py` (~50 violaciones)

```python
# PROHIBIDO - Estados iniciales arbitrarios
neo_z = np.array([0.4, 0.3, 0.2, 0.05, 0.03, 0.02])  # ¿De dónde sale?
eva_z = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])     # ¿De dónde sale?

# PROHIBIDO - Tasas de ruido sin justificación
neo_noise = np.random.randn(6) * 0.05 * (1 - neo_drive_state)  # ¿Por qué 0.05?

# PROHIBIDO - Tasas de acoplamiento
neo_z = neo_z + neo_noise + 0.02 * (eva_z - neo_z)  # ¿Por qué 0.02?

# PROHIBIDO - Clips arbitrarios
neo_z = np.clip(neo_z, 0.01, 0.99)  # ¿Por qué 0.01 y 0.99?

# PROHIBIDO - Sorpresa inicial
neo_surprise = 0.1  # ¿Por qué 0.1?

# PROHIBIDO - Líneas de referencia en gráficos
ax3.axhline(y=0.1, color='green', linestyle='--')  # ¿Por qué 0.1?
ax3.axhline(y=0.5, color='red', linestyle='--')    # ¿Por qué 0.5?
```

**CORRECCIÓN PROPUESTA**:
```python
# ORIGEN: Estados iniciales = distribución uniforme normalizada
n_components = 6
neo_z = np.ones(n_components) / n_components  # ORIGEN: máxima entropía inicial

# ORIGEN: Ruido basado en precisión máquina o std de observaciones
eps = np.finfo(float).eps  # ORIGEN: precisión máquina
noise_scale = np.std(observations) if observations else eps

# ORIGEN: Acoplamiento de autocorrelación observada
coupling = estimate_autocorr_decay(history)  # ORIGEN: primer lag con acf < 1/e
```

---

### 3. `core/explorer_agent.py` (~30 violaciones)

```python
# PROHIBIDO - Umbral de confirmación de hipótesis
if h.confidence > 0.8 and h.success_rate > 0.7 and h.total_tests >= 10:

# PROHIBIDO - Scores fijos
uncertainty_score = 1.0 / (1.0 + len(related_hypotheses))
domain_score = 1.5 if pref in var else 0.5

# PROHIBIDO - Lag máximo fijo
max_lag: int = 12  # ¿Por qué 12?
```

**CORRECCIÓN PROPUESTA**:
```python
# ORIGEN: Umbral de confianza = percentil 90 de confianzas históricas
confidence_threshold = np.percentile(all_confidences, 90)

# ORIGEN: Umbral de éxito = percentil 75 de tasas de éxito
success_threshold = np.percentile(all_success_rates, 75)

# ORIGEN: Tests mínimos = MIN_SAMPLES_FOR_STATISTICS
min_tests = MATHEMATICAL_CONSTANTS['MIN_SAMPLES_FOR_STATISTICS']

# ORIGEN: max_lag = primer lag donde autocorrelación < 1/e
acf = estimate_acf(series)
max_lag = int(np.argmax(acf < np.exp(-1)))  # ORIGEN: tiempo de decorrelación
```

---

### 4. `scripts/agents_free_exploration.py` (~15 violaciones)

```python
# PROHIBIDO - Palabras iniciales hardcodeadas
starters = ['universe', 'matter', 'energy', 'time', 'space',
           'life', 'star', 'atom', 'wave', 'force']  # ELEGIDAS POR MÍ

# PROHIBIDO - Límites de extracción
return concepts[:20]           # ¿Por qué 20?
concepts[:10]                  # ¿Por qué 10?
facts[:5]                      # ¿Por qué 5?
self.interests[-30:]           # ¿Por qué 30?
self.facts_learned[-50:]       # ¿Por qué 50?

# PROHIBIDO - Rango de curiosidad
curiosity = 0.5 + random.random() * 0.5  # ¿Por qué 0.5-1.0?

# PROHIBIDO - Umbral de similitud
if 0.9 < ratio < 1.1:  # ¿Por qué 10%?
```

**CORRECCIÓN PROPUESTA**:
```python
# ORIGEN: Palabras iniciales de API externa (trending topics, random dictionary)
starters = fetch_random_words_from_api(n=10)  # ORIGEN: external API

# ORIGEN: Límites basados en memoria disponible o percentiles
memory_limit = estimate_working_memory_size()  # ORIGEN: hardware constraint

# ORIGEN: Curiosidad de distribución uniforme [0, 1]
curiosity = random.random()  # ORIGEN: U(0,1), sin sesgo

# ORIGEN: Umbral de similitud = IQR de ratios observados
similarity_threshold = 1.5 * estimate_iqr(observed_ratios)  # ORIGEN: Tukey fence
```

---

### 5. `scripts/agents_solve_paradoxes.py` (~20 violaciones)

```python
# PROHIBIDO - Confianzas de hipótesis inventadas
neo.hypothesize("...", 0.75)   # ¿Por qué 0.75?
neo.hypothesize("...", 0.4)    # ¿Por qué 0.4?
neo.hypothesize("...", 0.6)    # ¿Por qué 0.6?
neo.hypothesize("...", 0.85)   # ¿Por qué 0.85?

# PROHIBIDO - Umbrales de detección
abs(mean2 - mean1) / mean1 > 0.1  # ¿Por qué 0.1?
if corr > 0.5:                     # ¿Por qué 0.5?
```

---

### 6. `subjectivity/*.py` (~20 violaciones)

```python
# PROHIBIDO - Umbrales de criterios
test_result.AUC_ext > 0.3           # ¿Por qué 0.3?
comparison['divergence'] > 0.1      # ¿Por qué 0.1?
neo_compression_trend > 0.3         # ¿Por qué 0.3?
np.std(phi_history) > 0.01          # ¿Por qué 0.01?
abs(corr) > 0.1                     # ¿Por qué 0.1?
```

---

## CONSTANTES MATEMÁTICAS PERMITIDAS

Las siguientes constantes están **PERMITIDAS** por tener justificación estándar:

| Constante | Valor | Justificación |
|-----------|-------|---------------|
| `TUKEY_FENCE` | 1.5 | Definición estándar de Tukey para outliers |
| `IQR_EXTREME_OUTLIER` | 3.0 | Q1 - 3*IQR, Q3 + 3*IQR |
| `STANDARD_DEVIATIONS_95` | 1.96 | 95% intervalo de confianza |
| `STANDARD_DEVIATIONS_99` | 2.576 | 99% intervalo de confianza |
| `MIN_SAMPLES_FOR_STATISTICS` | 5 | Mínimo para std confiable |
| `MIN_SAMPLES_FOR_PERCENTILES` | 10 | Mínimo para percentiles |
| `np.finfo(float).eps` | ~2.2e-16 | Precisión máquina |
| `np.e` | 2.718... | Constante de Euler |
| `np.pi` | 3.14159... | Pi |
| `1/e` | 0.368... | Tiempo de decorrelación estándar |

---

## PLAN DE CORRECCIÓN

### Prioridad 1 (Crítico - Afecta resultados científicos)
1. [ ] `core/explorer_agent.py` - Umbrales de confirmación de hipótesis
2. [ ] `core/meta_drive.py` - Estados iniciales y tasas de acoplamiento
3. [ ] `scripts/agents_free_exploration.py` - Condiciones iniciales

### Prioridad 2 (Alto - Afecta comportamiento de agentes)
4. [ ] `worlds/complete_being.py` - Umbrales de personalidad
5. [ ] `worlds/complete_being.py` - Probabilidades de eventos
6. [ ] `subjectivity/*.py` - Criterios de validación

### Prioridad 3 (Medio - Afecta visualización)
7. [ ] Líneas de referencia en gráficos
8. [ ] Límites de clips

---

## BLOQUE DE AUDITORÍA ESTÁNDAR

Cada archivo corregido debe terminar con:

```python
"""
MAGIC NUMBERS AUDIT
==================
- threshold_X: ORIGEN: percentil Y de variable Z (endógeno)
- constant_A: ORIGEN: np.finfo(float).eps (constante numérica estándar)
- alpha: HEURÍSTICO, NO NORMA DURA. PENDIENTE: derivar de bootstrap

TODAS LAS CONSTANTES TIENEN ORIGEN DOCUMENTADO.
"""
```

---

## CONCLUSIÓN

**ESTADO ACTUAL**: El proyecto tiene ~250 violaciones de NORMA DURA.

**ACCIÓN REQUERIDA**: Refactorizar progresivamente, empezando por archivos de Prioridad 1.

**COMPROMISO**: Todo código nuevo cumplirá NORMA DURA desde su creación.
