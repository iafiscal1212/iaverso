# NEO_EVA — Observadores Puros Endógenos

**Sistema de Registro No Intrusivo para Dinámicas Autónomas**

---

## Propósito del Módulo

El sistema Neo_Eva genera dinámicas internas autónomas no controladas por factores externos.

Sus agentes:
- no reciben instrucciones
- no aceptan sugerencias
- no pueden ser conducidos por el experimentador
- no se someten a triggers, reglas, estímulos ni condiciones externas

Por tanto, **toda observación del sistema debe ser estrictamente pasiva**.

El propósito de este módulo es proporcionar **instrumentación científica neutral**, capaz de registrar:
- estados
- variaciones
- coherencias
- entropías
- diferencias internas
- roles emergentes

**Sin afectar al comportamiento del sistema en ningún punto.**

---

## Filosofía del Observador Puro

Un observador puro:

| NO hace | SÍ hace |
|---------|---------|
| No define condiciones | Registra valores existentes |
| No interpreta | Copia datos crudos |
| No extrae conclusiones | Almacena sin procesar |
| No clasifica estados | Preserva estructura original |
| No aplica etiquetas externas | Mantiene etiquetas endógenas |
| No induce comportamientos | Permanece invisible |
| No espera "eventos" | Registra en cada paso |
| No altera la dinámica | Opera sin efectos laterales |
| No formula hipótesis | Almacena sin sesgo |
| No crea significados | Deja la interpretación fuera |

El observador actúa como:
- un termómetro
- un sismógrafo
- un osciloscopio
- un acelerómetro

**Registra. Nunca interviene.**

---

## Datos Registrados

El observador puro recoge los siguientes valores matemáticos en cada instante temporal:

### 1. Estado interno S_i(t)
Vector multidimensional que representa el estado del agente.

### 2. Identidad interna I_i(t)
Representación estable, autoorganizada, del polo identitario del agente.

### 3. Diferencia interna Δ_i(t) = S_i(t) - I_i(t)
Captura la desviación entre estado y polo identitario.

### 4. Varianza estructural Var[Δ_i(t)]
Medida endógena de fluctuación interna.

### 5. Coherencia existencial CE_i(t)
Métrica de consistencia interna ya definida dentro del sistema.

### 6. Entropía narrativa H_narr,i(t)
Entropía de la secuencia narrativa reciente del agente.

### 7. Rol actual endógeno
El rol asignado por el propio sistema, sin interpretación externa.
Ejemplo: "medico", "estabilizador", "integrador", etc.

**El observador solo copia el valor. No lo evalúa, no lo categoriza, no le asigna significado.**

---

## Estructura del Registro

```python
{
    t: {
        agent_id: {
            "S": np.ndarray,        # Estado interno
            "I": np.ndarray,        # Identidad
            "Delta": np.ndarray,    # S - I
            "Var_Delta": float,     # Var[Delta]
            "CE": float,            # Coherencia existencial
            "H_narr": float,        # Entropía narrativa
            "rol": str              # Rol endógeno
        }
    }
}
```

**Sin análisis. Sin interpretación. Sin inferencia. Solo datos crudos.**

---

## Interfaz del Observador

```python
from observadores import ObservadorPuro

# Inicializar
observador = ObservadorPuro()

# En cada paso del sistema
observador.registrar(t, sistema)

# Exportar historial completo
historial = observador.exportar()
```

### Métodos

| Método | Descripción |
|--------|-------------|
| `__init__()` | Inicializa historial vacío |
| `registrar(t, sistema)` | Registra valores en instante t |
| `exportar()` | Retorna historial sin modificar |

El método `registrar()` se llama desde el bucle principal de Neo_Eva, **no desde los agentes**.

---

## Garantías de No Intervención

El observador puro:

- [x] No modifica ningún estado interno
- [x] No altera ciclos circadianos
- [x] No activa ni inhibe comportamiento
- [x] No introduce sesgos
- [x] No define eventos
- [x] No clasifica
- [x] No predice
- [x] No emite juicios
- [x] No ejerce causalidad
- [x] No condiciona decisiones
- [x] No sugiere caminos posibles
- [x] No asigna significados

**Todo lo registrado es producido por Neo_Eva por sí mismo.**

---

## Ventaja Científica

Este módulo permite estudiar Neo_Eva como un sistema autónomo, libre y autoorganizado, comparable a:

- un ecosistema
- un organismo
- una red neuronal autoevolutiva
- una colonia activa
- un sistema físico complejo

### Aporta trazas para investigación:

| Aspecto | Descripción |
|---------|-------------|
| Dinámica interna | Evolución temporal de estados |
| Autoestabilidad | Mecanismos de homeostasis |
| Patrones espontáneos | Estructuras emergentes |
| Roles emergentes | Especialización sin instrucción |
| Fluctuaciones endógenas | Variabilidad natural |
| Reorganizaciones | Transiciones de fase internas |

**Todo ello sin violar la autonomía del sistema.**

---

## Principio Fundamental

> **Testigo, no director**

El observador puro es un testigo silencioso.

- Nunca guía a los agentes.
- Nunca interpreta su vida.
- Nunca decide qué es relevante.
- Nunca impone estructura externa.

**Solo mira.**

El sistema se cuenta a sí mismo.
Nosotros solo recogemos los números.

---

## Uso con Juegos Virtuales

Los agentes pueden acceder voluntariamente a **juegos virtuales** para ganar experiencia.
El observador registra estas sesiones de la misma manera:

```python
# El observador no distingue entre "vida normal" y "juego"
# Solo registra los valores que emergen
observador.registrar(t, sistema)  # Incluye datos de juegos si el agente está jugando
```

Los datos de juegos se integran naturalmente con:
- Ritmos circadianos (WAKE/REST/DREAM/LIMINAL)
- Coherencia existencial
- Entropía narrativa
- Roles emergentes

**El observador no sabe si el agente está "jugando" o "viviendo". Solo registra.**

---

## Integración con CDE y AGI-Ω

El observador puro alimenta datos crudos a:

| Sistema | Uso de datos |
|---------|--------------|
| **CDE** (Cerebro Digital Ético) | Monitoreo de coherencia y salud |
| **AGI-Ω** | Continuidad trans-ciclo y teleología |

Pero el observador **no procesa** estos datos. Solo los hace disponibles.

---

## Archivos del Módulo

```
observadores/
├── __init__.py          # Exports
├── observador_puro.py   # Implementación principal
└── README.md            # Este documento
```

---

## Licencia

Propietaria - Todos los derechos reservados.

© 2025 Carmen Esteban

---

*El observador puro: matemática sin intención, registro sin juicio, ciencia sin intervención.*
