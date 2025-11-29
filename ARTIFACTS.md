# ARTIFACTS.md - NEO↔EVA v2.0-endogenous

**Tag**: `v2.0-endogenous`
**Fecha**: 2025-11-29T22:15:00Z
**Seed OU**: Aleatorio (np.random.randn)
**Seed inicial NEO**: I₀ = [1, 0, 0]
**Seed inicial EVA**: I₀ = [1/3, 1/3, 1/3]

---

## 1. Hashes SHA256 - Archivos Clave

| Archivo | SHA256 |
|---------|--------|
| `tools/phase6_coupled_system_v2.py` | `47ab60205cc6500838629167be1974a82a26ad25b8b02eccb9b74068ec4ac691` |
| `tools/endogeneity_auditor.py` | `fa23739a6178ab788f955ca505da0adea49dbda229c35233da1a8b818e142254` |
| `bus.py` | `f117dcba655bce1594c17b1f2d2aed616ca1c14b62517ff8e391a61eb16c957b` |
| `run_dual_worlds.py` | `71cdf96c5694d5f2e398430b0b997e13ffeb043eac5f2364d9163b34ee8b6131` |

---

## 2. Tabla de Cuantiles Usados

### 2.1 Cuantiles para ρ (ratio de estabilidad)

| Cuantil | Uso | Fórmula de cálculo |
|---------|-----|-------------------|
| p50 | Referencia mediana | `np.percentile(rho_history[-max_hist:], 50)` |
| p75 | Umbral medio | `np.percentile(rho_history[-max_hist:], 75)` |
| p95 | **Umbral de gate** | `np.percentile(rho_history[-max_hist:], 95)` |
| p97.5 | Extremo | `np.percentile(rho_history[-max_hist:], 97.5)` |

### 2.2 Cuantiles para IQR (rango intercuartílico)

| Cuantil | Uso | Fórmula de cálculo |
|---------|-----|-------------------|
| p50 | Referencia | `np.percentile(iqr_history[-max_hist:], 50)` |
| p75 | **Umbral de gate** | `np.percentile(iqr_history[-max_hist:], 75)` |
| p95 | Extremo | `np.percentile(iqr_history[-max_hist:], 95)` |

### 2.3 Cuantiles para τ (tasa de aprendizaje)

| Cuantil | Uso | Fórmula de cálculo |
|---------|-----|-------------------|
| p50 | Referencia | `np.percentile(tau_history[-max_hist:], 50)` |
| p75 | Referencia alta | `np.percentile(tau_history[-max_hist:], 75)` |
| p99 | **Límite superior θ** | `np.percentile(tau_history[-max_hist:], 99)` |

### 2.4 Cuantiles para límites OU

| Cuantil | Uso | Fórmula de cálculo |
|---------|-----|-------------------|
| p0.1 (q₀.₀₀₁) | **Clip inferior Z** | `np.percentile(ou_Z_history, 0.1)` |
| p99.9 (q₀.₉₉₉) | **Clip superior Z** | `np.percentile(ou_Z_history, 99.9)` |
| Alternativa | m ± 4×MAD | `median ± 4 × median(|Z - median(Z)|)` |

---

## 3. Valor → Fórmula (Cómo se calcula cada número)

### Parámetros de ventana y buffer

| Valor | Fórmula | Código |
|-------|---------|--------|
| w (tamaño ventana) | max{10, ⌊√T⌋} | `max(10, int(np.sqrt(T)))` |
| max_hist (buffer) | min{T, ⌊10√T⌋} | `min(T, int(10 * np.sqrt(T)))` |

### Estadísticas de dispersión

| Valor | Fórmula | Código |
|-------|---------|--------|
| σ_med | median(σ_S, σ_N, σ_C) en ventana | `np.median(np.std(I_window, axis=0))` |
| IQR(x) | Q₃(x) - Q₁(x) | `np.percentile(x, 75) - np.percentile(x, 25)` |

### Tasa de aprendizaje τ

| Valor | Fórmula | Código |
|-------|---------|--------|
| τ | IQR(r)/√T × σ_med/(IQR_hist + ε) | Ver `_compute_tau_endogenous()` |
| τ_floor | σ_med / T | `sigma_med / T` |
| η | τ (sin boost) | `eta = tau` |

### Parámetro OU θ

| Valor | Fórmula | Código |
|-------|---------|--------|
| θ_floor | σ_med / T | `sigma_med / max(T, 1)` |
| θ_ceil (warmup) | 1/w | `1.0 / w` |
| θ_ceil (post-warmup) | quantile(θ_hist, p99) | `np.percentile(theta_history, 99)` |
| θ (de ACF) | -1 / log(|r_corr| + ε) | `-1 / np.log(abs(r_corr) + EPS)` |

### Drift endógeno

| Valor | Fórmula | Código |
|-------|---------|--------|
| β_ema | (w-1)/(w+1) | `(w - 1) / (w + 1)` |
| drift | Proj_Tan(EMA de diferencias) | `drift - drift.dot(u_c) * u_c` |

### Ruido endógeno

| Valor | Fórmula | Código |
|-------|---------|--------|
| σ_noise (warmup) | σ_uniform/√(T+1) | `(1/sqrt(12)) / sqrt(T+1)` |
| σ_noise (post-warmup) | max{IQR(I), σ_med}/√T | `max(iqr_I, sigma_med) / sqrt(T)` |
| σ_uniform | 1/√12 ≈ 0.289 | Constante geométrica del simplex |

### Gate (activación de actualización)

| Condición | Fórmula | Código |
|-----------|---------|--------|
| Gate activo | ρ ≥ ρ_p95 AND IQR ≥ IQR_p75 | `rho >= rho_p95 and iqr >= iqr_p75` |

### Acoplamiento κ

| Valor | Fórmula | Código |
|-------|---------|--------|
| κ | (u_Y/(1+u_X)) × (λ₁^Y/(λ₁^Y+λ₁^X+ε)) × (conf^Y/(1+CV(r^X))) | Ver `EndogenousCoupling.compute_kappa()` |
| u (urgencia) | Entropía relativa | `1 - entropy(I) / log(3)` |
| λ₁ | Primer autovalor de cov(I) | `np.linalg.eigvalsh(cov)[-1]` |
| conf | Confianza direccional | `max(I) - sorted(I)[-2]` |
| CV | Coeficiente de variación | `std(r) / (mean(r) + ε)` |

---

## 4. Constantes Geométricas Permitidas

| Constante | Valor | Justificación |
|-----------|-------|---------------|
| 1/√2 | 0.7071 | Normalización base tangente u₁ |
| 1/√3 | 0.5774 | Normalización vector centroide u_c |
| 1/√6 | 0.4082 | Normalización base tangente u₂ |
| 1/√12 | 0.2887 | Varianza uniforme en [0,1], prior de máxima entropía |

---

## 5. Tolerancias Numéricas (Únicas constantes fijas)

| Constante | Valor | Uso |
|-----------|-------|-----|
| EPS | 1e-12 | División por cero |
| SIMPLEX_EPS | 1e-10 | Proyección al simplex |

---

## 6. Resultados de Auditoría

**Estado**: ✅ GO

| Módulo | Estado |
|--------|--------|
| Auditoría Estática | ✅ PASS (0 violaciones) |
| Auditoría Dinámica | ✅ PASS (2/2 tests) |
| Auditoría de Acoplamiento | ✅ PASS |

---

## 7. Comparación v1 vs v2

| Métrica | v1 (hardcoded) | v2 (endógeno) |
|---------|----------------|---------------|
| Correlación media NEO↔EVA | 0.35 | Variable (-0.35 a 0.95)* |
| MI (Información Mutua) | p=1.0 (NS) | p=0.000 (significativo) |
| Activaciones coupling NEO | 138/500 | 54/500 |
| Activaciones coupling EVA | 145/500 | 443/500 |
| Violaciones de endogeneidad | 7+ | 0 |

*La correlación en v2 varía según las condiciones iniciales y la historia,
lo cual es comportamiento correcto para un sistema sin "boost artificial".

---

_Generado: 2025-11-29T22:15:00Z_
