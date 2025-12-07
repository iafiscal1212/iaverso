# Auditoría de Números Mágicos - NEO_EVA

## Norma Dura:
**"Ningún número entra al código sin poder explicar de qué distribución de los datos sale"**

---

## 🔴 VIOLACIONES ACTUALES (deben corregirse)

### 1. Scoring con número mágico 20
```python
# Archivo: agents_truly_endogenous.py:279
score = max(0, 100 - z_score * 20)
```
**PROBLEMA**: El `20` es arbitrario. ¿Por qué 20 y no 15 o 25?

**CORRECCIÓN REQUERIDA**: El factor debe derivarse de los datos. Por ejemplo:
- Usar IQR de la distribución aprendida
- Usar std de la distribución observada
- El agente debe calcular su propio "peso" de penalización

### 2. Default std como 20% de media
```python
# Archivo: agents_truly_endogenous.py:273
std_learned = np.std(learned_temps) if len(learned_temps) > 1 else mean_learned * 0.2
```
**PROBLEMA**: El `0.2` (20%) es arbitrario.

**CORRECCIÓN REQUERIDA**: Si no hay suficientes datos para std, el agente no debería poder evaluar (return None).

### 3. Score por defecto de 50
```python
# Archivo: agents_truly_endogenous.py:281
score = 50
```
**PROBLEMA**: ¿Por qué 50? Es un número arbitrario.

**CORRECCIÓN REQUERIDA**: Si no puede calcular score, debería retornar `None` o "indeterminado".

### 4. Filtro de temperaturas 50-1000
```python
# Archivo: agents_truly_endogenous.py:268
learned_temps = [f['value'] for f in temp_facts if 50 < f['value'] < 1000]
```
**PROBLEMA**: Los límites 50 y 1000 son arbitrarios.

**CORRECCIÓN REQUERIDA**: No filtrar. Usar todos los valores extraídos, o que el agente decida qué es outlier basándose en estadísticas (ej: ±3σ).

### 5. Curiosidad por defecto 0.5
```python
# Archivo: agents_truly_endogenous.py:93
curiosity = self.personality.get('curiosity', 0.5)
```
**PROBLEMA**: El 0.5 es arbitrario.

**JUSTIFICACIÓN POSIBLE**: Es el punto medio de una escala 0-1. Esto SÍ es justificable matemáticamente como "neutro" en una distribución uniforme.

**VEREDICTO**: ⚠️ Aceptable pero documentar.

### 6. Límite de contexto 200 caracteres
```python
# Archivo: real_knowledge_source.py:317 (extractor)
context = sentence.strip()[:200]
```
**PROBLEMA**: ¿Por qué 200?

**CORRECCIÓN**: Este es un límite técnico de display, no afecta la ciencia. Pero debería documentarse.

---

## 🟡 NÚMEROS TÉCNICOS (aceptables con documentación)

### Límites de API
- `timeout=30` - Límite técnico de red
- `limit=5`, `limit=10` - Límite de resultados de API
- `size=3` - Límite de papers a buscar

**VEREDICTO**: Estos no afectan la ciencia, solo la cantidad de datos.

### Escala de scores 0-100
- `score = max(0, 100 - ...)`

**VEREDICTO**: Es una escala arbitraria pero estándar. Lo importante es la comparación relativa, no el número absoluto.

---

## 🟢 NÚMEROS DERIVADOS DE DATOS (correctos)

### Estadísticas de distribución
```python
mean_val = np.mean(values)  # ✓ Viene de datos
std_val = np.std(values)    # ✓ Viene de datos
min_val = min(values)       # ✓ Viene de datos
max_val = max(values)       # ✓ Viene de datos
```

### Percentiles
```python
q1, q3 = vals.quantile([0.25, 0.75])  # ✓ Definición matemática estándar
iqr = q3 - q1                          # ✓ Definición matemática
```

---

## 📋 PLAN DE CORRECCIÓN

### Prioridad Alta:
1. Eliminar el factor `20` en scoring - usar IQR o std calculada
2. No usar `0.2` como default std - retornar None si insuficientes datos
3. No usar `50` como score default - retornar None

### Prioridad Media:
4. Eliminar filtro `50 < x < 1000` - usar detección de outliers estadística

### Documentar:
5. Curiosidad 0.5 - punto medio de escala uniforme
6. Límites técnicos de API - no afectan ciencia

---

## IMPLEMENTACIÓN CORRECTA

```python
def evaluate_temperature(self, planet_temp, learned_temps):
    """
    Evaluación sin números mágicos.

    Todos los umbrales vienen de los datos.
    """
    if len(learned_temps) < 5:
        return {
            'can_evaluate': False,
            'reason': 'Insuficientes datos aprendidos'
        }

    # Calcular estadísticas de la distribución aprendida
    mean_learned = np.mean(learned_temps)
    std_learned = np.std(learned_temps)
    q1 = np.percentile(learned_temps, 25)
    q3 = np.percentile(learned_temps, 75)
    iqr = q3 - q1

    # Detectar outliers con criterio estadístico (Tukey)
    # NO es un número mágico, es una definición matemática
    lower_bound = q1 - 1.5 * iqr  # Tukey fence
    upper_bound = q3 + 1.5 * iqr  # Tukey fence

    # Calcular z-score
    if std_learned > 0:
        z_score = abs(planet_temp - mean_learned) / std_learned
    else:
        return {'can_evaluate': False, 'reason': 'std = 0'}

    # Score basado en distribución normal
    # Prob de estar a ≤z desviaciones = scipy.stats.norm.cdf(z)
    from scipy.stats import norm
    prob_closer = 2 * (1 - norm.cdf(z_score))  # Two-tailed
    score = 100 * prob_closer  # Ahora viene de la distribución

    return {
        'can_evaluate': True,
        'score': score,
        'justification': {
            'mean_learned': mean_learned,
            'std_learned': std_learned,
            'z_score': z_score,
            'probability': prob_closer,
            'n_samples': len(learned_temps),
        }
    }
```

---

## CHECKLIST FINAL

Antes de publicar, verificar que CADA número:

- [ ] Viene de `np.mean()`, `np.std()`, `np.percentile()` de datos
- [ ] O es una constante matemática definida (π, e, 1.5 para Tukey)
- [ ] O es un límite técnico documentado (timeout, API limits)
- [ ] O se retorna None/indeterminado en lugar de asumir

**Si no puedes explicar de qué distribución sale, está prohibido.**
