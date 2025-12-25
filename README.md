# IRIS - Asistente Autónomo Inteligente

IRIS es un asistente de IA completamente autónomo que monitorea, desarrolla y mantiene sistemas de forma proactiva.

## Características

### Autonomía Total
- **Piensa** por sí misma y decide qué proyectos crear
- **Codifica** soluciones completas
- **Prueba** el código automáticamente
- **Corrige** errores sin intervención humana
- **Despliega** con solo tu aprobación

### Comunicación Natural
- Habla contigo por chat como una persona
- Te saluda cada día
- Te avisa de problemas detectados
- Te propone soluciones
- Solo necesitas decir "ok" para que actúe

### Monitoreo Proactivo
- CPU, memoria y disco
- Servicios y procesos
- Errores en logs
- Backups automáticos

### Sistema de Aprobación
- Clasificación de riesgo (bajo/medio/alto/crítico)
- Historial de acciones
- Nunca ejecuta sin permiso

## Arquitectura

```
/root/NEO_EVA/
├── api/
│   ├── iris_web.py          # Servidor web + WebSocket
│   ├── iris_approval_queue.py # Sistema de aprobaciones
│   └── iris_dashboard.py     # Panel de métricas
├── autonomous/
│   ├── iris_v2.py            # Motor autónomo v2
│   └── iris_asistente.py     # Asistente conversacional
├── core/
│   ├── iris_brain.py         # Cerebro (Claude + Ollama)
│   ├── iris_executor.py      # Motor de ejecución
│   └── iris_autonomo.py      # Funciones base
└── agents_state/
    ├── iris_memory.json      # Memoria persistente
    ├── iris_metrics.json     # Métricas
    └── iris_mensajes.json    # Mensajes del chat
```

## Instalación

```bash
# Clonar repositorio
git clone https://github.com/iafiscal1212/iaverso.git
cd iaverso

# Instalar dependencias
pip install fastapi uvicorn psutil requests anthropic

# Configurar Claude API (opcional, mejora calidad)
export ANTHROPIC_API_KEY="tu-api-key"

# Iniciar servicios
python3 api/iris_web.py &
python3 autonomous/iris_asistente.py --loop &
python3 autonomous/iris_v2.py --loop &
```

## URLs

- **Chat**: http://localhost:8891
- **Dashboard**: http://localhost:8891/dashboard
- **Métricas API**: http://localhost:8891/metrics

## Uso

### Chat Natural
IRIS te hablará automáticamente:
```
IRIS: "Buenos días! He revisado el sistema y todo está bien."
IRIS: "El disco está al 85%. ¿Limpio los logs viejos?"
Tú: "ok"
IRIS: "Listo, he limpiado 200MB de logs."
```

### Comandos Rápidos
- `ok`, `sí`, `dale` → Aprobar acción pendiente
- `no`, `dejalo` → Rechazar acción
- `python3 archivo.py` → IRIS lo ejecuta y analiza

### Desarrollo Autónomo
IRIS crea proyectos cada hora:
1. Piensa una idea original
2. Escribe el código
3. Lo prueba
4. Corrige errores
5. Te pide permiso para desplegar

## Componentes

### IrisBrain (core/iris_brain.py)
- Soporta Claude API + Ollama fallback
- Memoria persistente
- Conocimiento del codebase
- Verificación de código seguro

### IrisAsistente (autonomous/iris_asistente.py)
- Monitoreo del sistema
- Comunicación natural
- Detección de problemas
- Ejecución de acciones

### IrisV2 (autonomous/iris_v2.py)
- Generación autónoma de proyectos
- Ciclo: Pensar → Codificar → Probar → Corregir → Desplegar
- Métricas de rendimiento

## Seguridad

- Comandos peligrosos bloqueados (rm -rf, shutdown, etc)
- Rutas protegidas (/etc/passwd, ~/.ssh/)
- Log de todas las ejecuciones
- Sistema de aprobación obligatorio

## Licencia

MIT License
