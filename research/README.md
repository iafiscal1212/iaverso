# Recursos de Investigación para Agentes

Los 5 agentes (NEO, EVA, ALEX, ADAM, IRIS) pueden investigar las grandes preguntas
si les damos acceso a datos relevantes.

## Fuentes de Datos por Pregunta

### 1. Materia Oscura
- **Curvas de rotación galáctica**: NASA, ESA Gaia
- **Lentes gravitacionales**: Hubble, JWST
- **CMB (Cosmic Microwave Background)**: Planck, WMAP
- URLs:
  - https://gea.esac.esa.int/archive/
  - https://irsa.ipac.caltech.edu/

### 2. Cuántica + Relatividad
- **Experimentos de entrelazamiento**: Papers de arXiv
- **Detección de ondas gravitacionales**: LIGO/Virgo
- **Espectros atómicos de alta precisión**: NIST
- URLs:
  - https://www.gw-openscience.org/
  - https://physics.nist.gov/

### 3. Consciencia
- **fMRI durante estados conscientes**: OpenNeuro
- **EEG correlacionado con reportes**: PhysioNet
- **Estudios de anestesia/coma**: Papers clínicos
- URLs:
  - https://openneuro.org/
  - https://physionet.org/

### 4. Origen de la Vida
- **Química prebiótica**: Papers de astrobiología
- **Simulaciones de Miller-Urey**: Replicaciones
- **Genomas mínimos**: JCVI
- URLs:
  - https://www.ncbi.nlm.nih.gov/
  - https://astrobiology.nasa.gov/

### 5. Predicción de Terremotos (YA TENEMOS DATOS)
- **Sismicidad**: USGS ✓
- **Geomagnetismo**: NOAA ✓
- **Clima**: OpenWeather ✓
- **Schumann resonance**: Hay que buscar

## Cómo Añadir Datos

1. Descargar datos a `/root/NEO_EVA/data/research/`
2. Crear parser en `/root/NEO_EVA/research/parsers/`
3. Integrar con `agents_discover.py`

## Limitaciones Honestas

Para las preguntas 1-4, los datos públicos son:
- Escasos
- Preprocesados (pierden información)
- De experimentos específicos (no generalizables)

Los agentes pueden encontrar patrones, pero NO pueden resolver
misterios que la humanidad lleva décadas investigando.

Lo que SÍ pueden hacer:
- Buscar correlaciones inesperadas
- Detectar patrones que humanos no ven
- Generar hipótesis para testear
- Identificar dónde faltan datos
