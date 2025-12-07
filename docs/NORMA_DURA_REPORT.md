# NORMA DURA - Reporte de Violaciones

**Generado:** 2025-12-06T08:51:00.706890

> "Ningún número entra al código sin poder explicar de qué distribución sale"

## Resumen Ejecutivo

| Métrica | Valor |
|---------|-------|
| Total violaciones | 2656 |
| 🔴 Zona Estricta | 2353 |
| 🟡 Zona Sandbox | 303 |
| Archivos afectados (estricta) | 197 |
| Archivos afectados (sandbox) | 11 |

## 🔴 Top 20 Archivos - ZONA ESTRICTA

Estos archivos **DEBEN** ser corregidos antes de publicación.

| # | Archivo | Violaciones | Alta | Media | Baja |
|---|---------|-------------|------|-------|------|
| 1 | `audit_phases1_25_complete.py` | 62 | 18 | 0 | 44 |
| 2 | `audit_phases26_40_complete.py` | 56 | 25 | 0 | 31 |
| 3 | `phase6_coupled_system.py` | 40 | 31 | 3 | 6 |
| 4 | `knowledge_library.py` | 39 | 39 | 0 | 0 |
| 5 | `phase7_consent_autocouple.py` | 36 | 33 | 0 | 3 |
| 6 | `common.py` | 36 | 1 | 0 | 35 |
| 7 | `phase8_social_potentiated.py` | 35 | 29 | 1 | 5 |
| 8 | `phase6_coupled_system_v2.py` | 34 | 24 | 0 | 10 |
| 9 | `soft_hook.py` | 32 | 26 | 6 | 0 |
| 10 | `agents.py` | 31 | 20 | 7 | 4 |
| 11 | `physics_lab.py` | 31 | 13 | 1 | 17 |
| 12 | `neo_phase4_patched_server.py` | 31 | 27 | 1 | 3 |
| 13 | `explorer_agent_v2.py` | 30 | 24 | 1 | 5 |
| 14 | `post_fix_audit_complete.py` | 28 | 20 | 0 | 8 |
| 15 | `cosmos_fetcher.py` | 27 | 19 | 0 | 8 |
| 16 | `phase8_voluntary_consent.py` | 27 | 22 | 1 | 4 |
| 17 | `phase12_full_robustness.py` | 27 | 18 | 2 | 7 |
| 18 | `persistent_goals.py` | 26 | 24 | 0 | 2 |
| 19 | `phase6_iwvi_analysis.py` | 26 | 17 | 0 | 9 |
| 20 | `phase20_structural_veto.py` | 25 | 14 | 0 | 11 |
| ... | _177 archivos más_ | | | | |

### Detalles de Top 5

#### 1. `audit_phases1_25_complete.py`

**Ruta:** `/root/NEO_EVA/tools/audit_phases1_25_complete.py`
**Violaciones:** 62

| Línea | Magic Number | Severidad |
|-------|--------------|-----------|
| 27 | `15` | 🟢 low |
| 27 | `15` | 🟢 low |
| 28 | `16` | 🟢 low |
| 28 | `16` | 🟢 low |
| 29 | `17` | 🟢 low |
| 29 | `17` | 🟢 low |
| 30 | `18` | 🟢 low |
| 30 | `18` | 🟢 low |
| 31 | `19` | 🟢 low |
| 31 | `19` | 🟢 low |
| ... | _52 más_ | |

#### 2. `audit_phases26_40_complete.py`

**Ruta:** `/root/NEO_EVA/tools/audit_phases26_40_complete.py`
**Violaciones:** 56

| Línea | Magic Number | Severidad |
|-------|--------------|-----------|
| 35 | `26` | 🟢 low |
| 36 | `27` | 🟢 low |
| 37 | `28` | 🟢 low |
| 38 | `29` | 🟢 low |
| 40 | `31` | 🟢 low |
| 41 | `32` | 🟢 low |
| 42 | `33` | 🟢 low |
| 43 | `34` | 🟢 low |
| 44 | `35` | 🟢 low |
| 45 | `36` | 🟢 low |
| ... | _46 más_ | |

#### 3. `phase6_coupled_system.py`

**Ruta:** `/root/NEO_EVA/tools/phase6_coupled_system.py`
**Violaciones:** 40

| Línea | Magic Number | Severidad |
|-------|--------------|-----------|
| 260 | `= 0.1` | 🔴 high |
| 299 | `=0.0` | 🔴 high |
| 300 | `=0.1` | 🔴 high |
| 301 | `=0.5` | 🔴 high |
| 302 | `=0.1` | 🔴 high |
| 322 | `= 0.0` | 🔴 high |
| 330 | `= 0.1` | 🔴 high |
| 337 | `= 0.1` | 🔴 high |
| 347 | `= 0.5` | 🔴 high |
| 370 | `0.90` | 🔴 high |
| ... | _30 más_ | |

#### 4. `knowledge_library.py`

**Ruta:** `/root/NEO_EVA/research/knowledge_library.py`
**Violaciones:** 39

| Línea | Magic Number | Severidad |
|-------|--------------|-----------|
| 54 | `0.3` | 🔴 high |
| 76 | `0.6` | 🔴 high |
| 92 | `0.3` | 🔴 high |
| 103 | `0.4` | 🔴 high |
| 119 | `0.4` | 🔴 high |
| 130 | `0.6` | 🔴 high |
| 146 | `0.3` | 🔴 high |
| 157 | `0.4` | 🔴 high |
| 202 | `0.3` | 🔴 high |
| 202 | `= 0.3` | 🔴 high |
| ... | _29 más_ | |

#### 5. `phase7_consent_autocouple.py`

**Ruta:** `/root/NEO_EVA/tools/phase7_consent_autocouple.py`
**Violaciones:** 36

| Línea | Magic Number | Severidad |
|-------|--------------|-----------|
| 59 | `= 0.0` | 🔴 high |
| 60 | `= 0.0` | 🔴 high |
| 89 | `= 0.5` | 🔴 high |
| 93 | `> 0.5` | 🔴 high |
| 201 | `= 0.5` | 🔴 high |
| 224 | `= 0.0` | 🔴 high |
| 242 | `= 0.5` | 🔴 high |
| 345 | `= 0.0` | 🔴 high |
| 377 | `= 0.5` | 🔴 high |
| 457 | `12` | 🟢 low |
| ... | _26 más_ | |

## 🟡 Top 10 Archivos - ZONA SANDBOX

Estos archivos tienen warnings pero **no bloquean** el test.

| # | Archivo | Warnings |
|---|---------|----------|
| 1 | `complete_being.py` | 112 |
| 2 | `living_world_daemon.py` | 44 |
| 3 | `living_world.py` | 38 |
| 4 | `phaseS1_dual.py` | 35 |
| 5 | `cosmic_phenomena.py` | 24 |
| 6 | `test_complete_beings.py` | 16 |
| 7 | `phaseS1_phenomenal_state.py` | 11 |
| 8 | `phaseS2_self_report.py` | 10 |
| 9 | `phaseS2_self_report_dual.py` | 9 |
| 10 | `test_living_world.py` | 3 |

## Guía de Corrección

### Para números decimales (0.3, 0.7, etc.):
```python
# ANTES (prohibido):
if confidence > 0.7:

# DESPUÉS (correcto):
from core.norma_dura_config import CONSTANTS
if confidence > CONSTANTS.PERCENTILE_75:  # ORIGEN: percentil 75 de U(0,1)
```

### Para umbrales derivados de datos:
```python
# ANTES (prohibido):
threshold = 1.5

# DESPUÉS (correcto):
threshold = np.percentile(data, 75)  # ORIGEN: percentil 75 de datos observados
```

### Para valores iniciales:
```python
# ANTES (prohibido):
initial_value = 0.5

# DESPUÉS (correcto):
initial_value = 0.5  # ORIGEN: máxima incertidumbre en escala [0,1]
```

---

*Generado automáticamente por `scripts/norma_dura_report.py`*