# IAverso - Sistema de Simulación Científica NORMA DURA

Sistema instrumental de procesamiento que conecta con fuentes de datos científicos reales para simulación estructural.

## Principios NORMA DURA

- **Solo estructura, no semántica**: El sistema transforma datos, no interpreta significados
- **Solo lectura**: No modifica fuentes externas, no genera predicciones
- **Sin trading**: Los datos financieros son solo para simulación estructural
- **Sin diagnósticos**: Los datos genómicos/médicos son solo informativos

## Arquitectura

```
iaverso/
├── api/           # Servidor API (aiohttp)
│   └── server.py  # 50+ endpoints unificados
├── core/          # Núcleo EndoLens + NeoSynt
│   ├── endolens.py
│   ├── neosynt.py
│   └── language.py
├── sim/           # Simuladores y clientes de datos
│   ├── base_simulator.py
│   ├── crypto_simulator.py    # Lab Cripto
│   ├── binance_client.py      # Datos Binance en vivo
│   ├── genomic_clients.py     # 7 fuentes genómicas
│   ├── physics_clients.py     # 4 fuentes físicas
│   └── math_clients.py        # 3 fuentes matemáticas
├── lab/           # Laboratorios de simulación
│   ├── genetic_lab.py
│   └── simulator.py
├── agents/        # Agentes Alpha, Beta, Gamma
└── tests/         # Tests del sistema
```

## Fuentes de Datos Reales (14 fuentes)

### Cripto (Binance)
- Precios en tiempo real
- Volumen 24h
- Datos de mercado

### Genómico (7 fuentes)
| Fuente | Tipo | Datos |
|--------|------|-------|
| GEO | Expresión | RNA-seq, microarrays |
| 1000 Genomes | Variantes | Variación poblacional |
| GTEx | Tejidos | Expresión por tejido |
| Orphanet | Enfermedades | Genes raros |
| ENCODE | Regulación | Epigenética |
| ArrayExpress | Expresión | Datos europeos |
| Allen Brain | Cerebro | Expresión cerebral |

### Física (4 fuentes)
| Fuente | Tipo | Datos |
|--------|------|-------|
| CERN | Partículas | Colisiones LHC |
| NASA DONKI | Solar | Fulguraciones, CMEs |
| USGS | Sismos | Terremotos globales |
| NOAA | Clima espacial | Índices geomagnéticos |

### Matemáticas (3 fuentes)
| Fuente | Tipo | Datos |
|--------|------|-------|
| OEIS | Secuencias | 370,000+ secuencias |
| LMFDB | Álgebra | Curvas elípticas |
| Santa Fe | Complejidad | Sistemas complejos |

## API Endpoints

Base: `https://iaverso.eu/api/iaverso/`

### Health
- `GET /health` - Estado del sistema

### Cripto
- `GET /sim/crypto/realtime` - Estado dimensional en vivo
- `GET /sim/crypto/price/{symbol}` - Precio (ej: BTCUSDT)
- `GET /sim/crypto/markets` - Lista de mercados

### Genómico
- `GET /sim/genetic/sources` - Lista de fuentes
- `GET /sim/genetic/sources/ping` - Verificar conexiones
- `GET /sim/genetic/realtime` - Estado dimensional

### Física
- `GET /sim/physics/sources` - Lista de fuentes
- `GET /sim/physics/earthquakes` - Sismos recientes
- `GET /sim/physics/solar` - Actividad solar

### Matemáticas
- `GET /sim/math/sources` - Lista de fuentes
- `GET /sim/math/sequence/{id}` - Buscar secuencia OEIS
- `POST /sim/math/search` - Buscar por términos

## Dimensiones Abstractas

Cada dominio mapea datos reales a 8 dimensiones normalizadas [0,1]:

**Cripto**: liquidity, volatility, concentration, latency, sentiment, leverage, network_load, regulatory_pressure

**Genómico**: expression_level, variability, tissue_specificity, regulatory_complexity, mutation_load, pathway_connectivity, conservation, disease_association

**Física**: energy_density, field_intensity, temporal_stability, spatial_coherence, event_frequency, magnitude_distribution, correlation_strength, anomaly_index

**Matemáticas**: pattern_density, sequence_regularity, structural_complexity, symmetry_index, growth_rate, periodicity, correlation_depth, novelty_index

## Instalación

```bash
pip install aiohttp numpy
python api/server.py
```

## Licencia

Propietario - Carmen Estévez / IAverso

## Disclaimer

Este sistema NO genera:
- Señales de trading
- Diagnósticos médicos
- Predicciones de ningún tipo

Solo transforma datos estructuralmente para simulación científica.
