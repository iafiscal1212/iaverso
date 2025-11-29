# Pre-registro: NEO↔EVA Sistema Endógeno v2.0

**Fecha de pre-registro**: 2025-11-29
**Investigadores**: Sistema NEO↔EVA
**Versión**: v2.0-endogenous

---

## 1. Hipótesis

### H1: Eliminación de Constantes Arbitrarias
El sistema v2 no contiene ninguna constante numérica que afecte la dinámica
(excepto tolerancias numéricas ≤1e-10 y constantes geométricas derivables).

**Criterio de éxito**: Auditoría estática = 0 violaciones.

### H2: Escalado Endógeno
Todos los parámetros escalan correctamente con T (tamaño de historia):
- τ, η ∝ 1/√T
- σ_noise ∝ 1/√T
- Límites OU ∝ cuantiles históricos

**Criterio de éxito**: Auditoría dinámica = PASS en test de varianza.

### H3: Acoplamiento Causal
El acoplamiento κ_t se deriva 100% de estadísticas observadas (urgencia,
autovalores, confianza, CV) sin factores arbitrarios.

**Criterio de éxito**: Δr(coupled - ablation) > 0 indica causalidad.

### H4: Gate por Cuantiles Puros
El gate de actualización usa únicamente cuantiles históricos (ρ_p95, IQR_p75)
sin factores multiplicativos.

**Criterio de éxito**: Código verificado por auditor estático.

---

## 2. Métricas Primarias

| Métrica | Definición | Umbral GO |
|---------|------------|-----------|
| Violaciones estáticas | Constantes hardcodeadas detectadas | = 0 |
| Tests dinámicos | Invariantes de escalado verificados | 2/2 PASS |
| Δr (coupled - ablation) | Diferencia de correlación | > 0 |
| MI significativa | p-value información mutua | < 0.05 |

---

## 3. Métricas Secundarias

| Métrica | Definición |
|---------|------------|
| Activaciones coupling | % de ciclos con κ > 0 |
| Varianza total | Var(I) integrada |
| Cuantiles τ | Distribución p50/p75/p99 |
| Límites OU | Rango adaptativo medio |

---

## 4. Criterios GO/NO-GO

### GO (Proceder con publicación)
- [ ] Auditoría estática: 0 violaciones
- [ ] Auditoría dinámica: 2/2 tests PASS
- [ ] Auditoría κ: Sin constantes mágicas
- [ ] Δr(coupled - ablation) > 0
- [ ] Reproducibilidad: Hashes verificados

### NO-GO (Requiere revisión)
- Cualquier violación en auditoría estática
- Fallo en tests de invariancia
- κ usando valores no derivados de datos
- Ablación indistinguible del acoplado

---

## 5. Análisis Planificado

### 5.1 Comparación v1 vs v2
- Correlación Pearson NEO↔EVA por componente
- Información Mutua con bootstrap
- Activaciones de coupling

### 5.2 Direccionalidad
- Transfer Entropy TE(NEO→EVA) y TE(EVA→NEO)
- Lags 1..L (L = primer cero de ACF)
- Nulos por phase randomization

### 5.3 Ablaciones
- `no_bus`: Sin comunicación por BUS
- `kappa=0`: κ forzado a cero
- `no_recall_eva`: EVA sin memoria de NEO

---

## 6. Datos y Código

### Repositorio
- Código: `tools/phase6_coupled_system_v2.py`
- Auditor: `tools/endogeneity_auditor.py`
- Scripts: `repro/run_endogenous.sh`

### Datos de salida
- Series temporales: `results/phase6_v2_*.json`
- Reporte auditoría: `results/endogeneity_audit.md`
- Hashes: `repro/results_hashes.sha256`

---

## 7. Declaración de Intenciones

Este pre-registro documenta las hipótesis y criterios de éxito
**antes** de analizar los resultados finales. Cualquier desviación
del plan será documentada y justificada.

El principio rector es: **"Si no sale de la historia, no entra en la dinámica"**.

---

_Firmado digitalmente por el sistema de auditoría_
_Hash del pre-registro: [calcular al commit]_
