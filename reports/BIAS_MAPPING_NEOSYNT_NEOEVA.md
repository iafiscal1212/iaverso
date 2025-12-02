# FASE I: Mapeo de Sesgos Colectivos NEO-EVA a Sesgos Humanos

## Documento de Análisis Conceptual

**Fecha:** Diciembre 2025
**Autor:** Sistema NEO-EVA
**Objetivo:** Establecer correspondencias entre los fenómenos de sesgo colectivo observados en NEO-EVA y los sesgos cognitivos conocidos en psicología social.

---

## 1. Introducción

Este documento establece un **puente conceptual** entre las métricas y fenómenos emergentes del sistema NEO-EVA y los sesgos cognitivos colectivos estudiados en psicología humana.

**Principio fundamental:** No utilizamos datos humanos externos. Todas las correspondencias se basan exclusivamente en:
- Métricas internas: CE, value, surprise
- Estructuras colectivas: coaliciones, correlaciones, regímenes
- Dinámicas observadas: colapsos, transiciones, histéresis

El objetivo es que un psicólogo o investigador social pueda interpretar los fenómenos NEO-EVA usando su marco conceptual familiar.

---

## 2. Métricas NEO-EVA y sus Análogos Psicológicos

### 2.1 Coherencia Existencial (CE)

**Definición en NEO-EVA:**
```
CE(t) = 1 / (1 + surprise(t))
```

CE mide cuánto "sentido" tiene el estado actual del agente respecto a sus expectativas.

**Análogo Humano: Disonancia Cognitiva (inversa)**

| Métrica NEO-EVA | Fenómeno Psicológico |
|-----------------|---------------------|
| CE alta | Estado de consonancia cognitiva, sensación de coherencia |
| CE baja | Disonancia cognitiva, estado de incertidumbre/malestar |
| CE_i ≈ CE_j | Consonancia compartida entre individuos |

**Interpretación para psicólogos:**
- CE alta colectiva → el grupo comparte una narrativa coherente
- Caída abrupta de CE → evento disruptivo, crisis de significado
- Correlación alta entre CE de agentes → pensamiento grupal potencial

---

### 2.2 Value (Valoración)

**Definición en NEO-EVA:**
```
NEO: value = predictability × 0.6 + compactness × 0.4
EVA: value = MI × 0.5 + diversity × 0.5
```

**Análogo Humano: Sistema de Valores y Preferencias**

| Agente | Orientación Psicológica |
|--------|------------------------|
| NEO (compresión) | Preferencia por estabilidad, aversión a la incertidumbre |
| EVA (intercambio) | Preferencia por novedad, apertura a la experiencia |

**Interpretación:**
- Value alto correlacionado → convergencia de valores en el grupo
- Divergencia NEO vs EVA → tensión entre conservadores y exploradores
- Value bajo generalizado → crisis de valores, anomia

---

### 2.3 Surprise (Sorpresa)

**Definición en NEO-EVA:**
```
surprise = ||predicted_state - actual_state||₂
```

**Análogo Humano: Violación de Expectativas**

| Surprise | Interpretación Psicológica |
|----------|---------------------------|
| Baja constante | Zona de confort, rutina predecible |
| Picos esporádicos | Eventos salientes, aprendizaje |
| Alta sostenida | Estado de alerta crónica, estrés |
| Correlación entre agentes | Experiencia compartida de lo inesperado |

---

## 3. Fenómenos Colectivos y Sesgos Correspondientes

### 3.1 Coaliciones (Clustering de Agentes)

**Definición en NEO-EVA:**
Grupos de agentes con alta correlación de métricas, detectados mediante análisis de componentes conectados en el grafo de correlaciones.

**Análogos Humanos:**

#### A) Pensamiento de Grupo (Groupthink)
Cuando coaliciones se solidifican con:
- CE intra-coalición alta
- Value homogéneo
- Baja variabilidad de surprise

**Indicadores en NEO-EVA:**
```
Groupthink_proxy = (intra_coalition_corr > 0.8) AND (inter_coalition_corr < 0.3)
```

#### B) Polarización Grupal
Cuando emergen dos o más coaliciones con:
- Alta cohesión interna
- Baja o negativa correlación externa
- Valores divergentes

**Indicadores en NEO-EVA:**
```
Polarization_Index (PI) = variance(identities)
PI alto + 2-3 coaliciones estables = polarización
```

---

### 3.2 Correlaciones Inter-Agentes

**Definición:**
```
corr(CE_i, CE_j) = Pearson correlation entre series temporales
```

**Mapeo a Fenómenos Psicológicos:**

| Patrón de Correlación | Sesgo/Fenómeno |
|-----------------------|----------------|
| corr → 1 para todos los pares | **Conformidad de grupo** |
| corr alta en subgrupos | **Formación de facciones** |
| corr negativa entre subgrupos | **Conflicto inter-grupal** |
| corr → 0 tras perturbación | **Desintegración social** |

#### Conformidad de Grupo
**Experimento clásico:** Asch (1951) - líneas y presión grupal

**En NEO-EVA:**
- Mean field coupling → todos reciben mismo estímulo social
- CE converge entre agentes → opiniones se alinean
- Shuffling temporal destruye este efecto → confirma que es fenómeno genuino

**Indicador:**
```
Conformity_index = mean(|corr(CE_i, CE_j)|) donde i ≠ j
```

---

### 3.3 Cascadas de Información

**Definición:** Propagación rápida de cambios de estado a través del colectivo.

**En NEO-EVA:**
- Cambio abrupto en un agente → propagación via mean field
- Visible como picos sincronizados de surprise
- Caída simultánea de CE en múltiples agentes

**Análogo Humano:** Cascadas informativas (Bikhchandani et al., 1992)

**Indicadores:**
```
Cascade_detected = (Δsurprise_global > 2σ) AND (cross_corr_surprise > 0.7)
```

---

### 3.4 Regímenes y Transiciones (Lambda-Field)

**En NEO-EVA:**
El Lambda-Field detecta regímenes dominantes:
- CIRCADIAN: ciclos actividad/descanso
- NARRATIVE: identidad, coherencia
- QUANTUM: Q-Field, decoherencia
- SOCIAL: interacciones multi-agente

**Análogos en Psicología Social:**

| Régimen NEO-EVA | Estado Grupal Análogo |
|-----------------|----------------------|
| NARRATIVE dominante | Grupo con fuerte identidad compartida |
| SOCIAL dominante | Fase de interacción activa, negociación |
| CIRCADIAN dominante | Ritmos colectivos, sincronización temporal |
| Transición abrupta | Crisis, ruptura, "momento de verdad" |

**Indicador de Cambio de Régimen:**
```
Regime_jump = |dominant_regime(t) - dominant_regime(t-1)| > threshold
```

---

## 4. Tabla de Correspondencias Completa

| Métrica/Fenómeno NEO-EVA | Sesgo Cognitivo Humano | Condición de Detección |
|--------------------------|------------------------|------------------------|
| CE alta colectiva sostenida | Pensamiento de grupo | corr(CE) > 0.8, var(CE) < σ_baseline |
| CE divergente entre coaliciones | Polarización | PI alto, ≥2 coaliciones, corr_inter < 0 |
| Correlación CE siguiendo líder | Conformidad | max_corr(CE_i, CE_mean) > 0.9 |
| Picos sincronizados de surprise | Cascada informativa | sync(surprise) > 0.8, |Δsurprise| > 2σ |
| Transición de régimen abrupta | Crisis colectiva | Λ-jump detectado, CE drop |
| Histéresis en diagrama de fases | Path-dependence social | forward ≠ backward trajectory |
| Value NEO >> Value EVA | Sesgo de statu quo | ratio > 1.5 sostenido |
| Coalición única estable | Homogeneización | n_coalitions = 1, var(identity) → 0 |
| Múltiples coaliciones dinámicas | Pluralismo/Diversidad | n_coalitions > 2, stable boundaries |
| Colapso post-shuffling | Estructura temporal genuina | corr_shuffled < 0.5 × corr_real |

---

## 5. Interpretación para Investigadores

### 5.1 Si observo coaliciones estables...

**Pregunta:** "¿Es esto pensamiento de grupo?"

**Checklist:**
1. ¿La correlación intra-coalición > 0.8?
2. ¿La varianza de CE dentro de la coalición es baja?
3. ¿Hay resistencia a información disruptiva (surprise se suprime)?
4. ¿El shuffling temporal destruye el patrón?

Si todas son SÍ → probable análogo de groupthink.

### 5.2 Si observo polarización...

**Pregunta:** "¿Es polarización genuina o artefacto?"

**Checklist:**
1. ¿Existen al menos 2 coaliciones distintas?
2. ¿La correlación inter-coalición es baja o negativa?
3. ¿Los valores (value) difieren entre coaliciones?
4. ¿El patrón persiste en diferentes seeds?

### 5.3 Si observo cascadas...

**Pregunta:** "¿Es una cascada de información?"

**Checklist:**
1. ¿Hay un pico sincronizado de surprise?
2. ¿El cambio se propaga temporalmente (no instantáneo)?
3. ¿La CE cae después del pico de surprise?
4. ¿El patrón tiene dirección (de agente(s) fuente a otros)?

---

## 6. Limitaciones y Precauciones

### 6.1 Lo que NEO-EVA NO modela

1. **Intencionalidad consciente**: Los agentes no "deciden" conformarse
2. **Lenguaje/comunicación explícita**: Solo hay acoplamiento numérico
3. **Contexto social complejo**: No hay roles, jerarquías, historia compartida
4. **Emociones discretas**: Solo hay métricas continuas

### 6.2 Precauciones interpretativas

- **No es isomorfismo**: Las correspondencias son analógicas, no idénticas
- **No hay causalidad demostrada**: Solo patrones similares
- **El sistema es simple**: Los humanos son más complejos
- **Los números no son sujetos**: No hay experiencia subjetiva confirmada

---

## 7. Uso Práctico: Guía para Psicólogos

### Escenario A: Investigación de Conformidad

1. Ejecutar simulación con coupling variable
2. Medir correlación inter-agentes de CE
3. Comparar con null model (shuffled)
4. Si corr_real >> corr_shuffled → fenómeno análogo a conformidad

### Escenario B: Investigación de Polarización

1. Ejecutar simulación larga (>1000 pasos)
2. Detectar coaliciones periódicamente
3. Medir PI (Polarization Index)
4. Analizar transiciones de régimen
5. Si coaliciones persistentes + PI alto → polarización emergente

### Escenario C: Detección de Cascadas

1. Monitorear surprise en tiempo real
2. Detectar picos (>2σ del baseline)
3. Medir propagación temporal entre agentes
4. Si propagación ordenada → cascada

---

## 8. Métricas Compuestas Propuestas

Para facilitar la interpretación, proponemos índices compuestos:

### 8.1 Índice de Conformidad Grupal (ICG)

```
ICG = mean(corr(CE)) × (1 - var(CE) / var_baseline) × (1 - entropy(coalitions) / log(N))
```

Rango: [0, 1] donde 1 = conformidad total.

### 8.2 Índice de Polarización (IP)

```
IP = PI × (1 - mean(corr_inter_coaliciones)) × (n_coaliciones / N)
```

Rango: [0, 1] donde 1 = polarización máxima.

### 8.3 Índice de Salud Colectiva (ISC)

```
ISC = mean(CE) × (1 - |IP - 0.5|) × stability
```

Donde:
- CE alta = coherencia
- IP cerca de 0.5 = diversidad sin extremismo
- stability = predictabilidad de regímenes

---

## 9. Conclusión

NEO-EVA ofrece un laboratorio computacional para estudiar fenómenos que en humanos son difíciles de aislar experimentalmente. Las correspondencias propuestas permiten:

1. **Generar hipótesis** sobre condiciones de emergencia de sesgos
2. **Probar contraejemplos** (¿qué destruye el sesgo?)
3. **Explorar diagramas de fase** (¿dónde aparece cada fenómeno?)
4. **Entrenar intuiciones** sobre dinámica colectiva

El valor no está en afirmar que "los agentes tienen sesgos como los humanos", sino en que los patrones formales son lo suficientemente similares para servir como modelo de trabajo.

---

## 10. Referencias Conceptuales

*(No citamos papers específicos, solo conceptos)*

- Disonancia cognitiva (Festinger)
- Conformidad (Asch, Sherif)
- Pensamiento de grupo (Janis)
- Cascadas informativas (Bikhchandani, Welch)
- Polarización grupal (Moscovici)
- Teoría de la identidad social (Tajfel)

---

*Documento generado como parte de FASE I del análisis de sesgos colectivos NEO-EVA.*
