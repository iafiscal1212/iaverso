#!/usr/bin/env python3
"""
IAVERSO API Server v4.0 - Laboratorios In Silico

Arquitectura de Inferencia Activa:
- ALPHA (Perceptor): Percibe estructuras - modo observacional
- BETA (Explorador): Conecta elementos - modo exploratorio
- GAMMA (Narrador): Integra Alpha+Beta, narra sin juzgar

TRES MODOS DE OPERACIÓN:
- CEREBRO: Investigación con Alpha → Beta → Gamma
- LAB GENÉTICO: Simulador de biología teórica in silico
- LAB CRIPTO: Simulador de dinámicas cripto in silico

⚠️ LAB CRIPTO:
- NO es trading, NO es inversión
- NO ejecuta órdenes reales
- Es SIMULACIÓN ESTRUCTURAL in silico
"""

import sys
sys.path.insert(0, '/opt/iaverso')

import json
from aiohttp import web
from datetime import datetime

from core.endolens import get_endolens
from core.neosynt import get_neosynt
from core.language import detect_language
from lab.simulator import get_simulator
from sim.crypto_simulator import get_crypto_simulator
from sim.binance_client import get_binance_client
from sim.genomic_clients import get_genomic_client
from sim.physics_clients import get_physics_client
from sim.math_clients import get_math_client
from agents.alpha import get_alpha
from agents.beta import get_beta
from agents.gamma import get_gamma


class IAVersoAPI:
    def __init__(self):
        self.endolens = get_endolens()
        self.neosynt = get_neosynt()

        # Simuladores
        self.genetic_sim = get_simulator()      # Lab genético existente
        self.crypto_sim = get_crypto_simulator()  # Lab cripto nuevo
        self.binance = get_binance_client()  # Cliente Binance para datos reales
        self.genomic = get_genomic_client()  # Cliente genómico unificado
        self.physics = get_physics_client()  # Cliente física unificado
        self.math = get_math_client()  # Cliente matemáticas unificado

        # Agentes NORMA DURA
        self.alpha = get_alpha()  # Perceptor
        self.beta = get_beta()    # Explorador
        self.gamma = get_gamma()  # Narrador

        self.app = web.Application()
        self._setup_routes()

    def _setup_routes(self):
        # Health y info
        self.app.router.add_get('/health', self.health)
        self.app.router.add_get('/modes', self.list_modes)
        self.app.router.add_get('/agents', self.list_agents)
        self.app.router.add_get('/domains', self.list_domains)

        # Core (compartido)
        self.app.router.add_post('/analyze', self.analyze)
        self.app.router.add_post('/resolve', self.resolve)
        self.app.router.add_post('/detect-language', self.detect_lang)

        # === AGENTES NORMA DURA ===
        self.app.router.add_post('/alpha/perceive', self.alpha_perceive)
        self.app.router.add_post('/beta/explore', self.beta_explore)
        self.app.router.add_post('/gamma/narrate', self.gamma_narrate)

        # === LAB GENÉTICO (existente) ===
        self.app.router.add_post('/sim/genetic/init_population', self.genetic_init_population)
        self.app.router.add_post('/sim/genetic/apply_operator', self.genetic_apply_operator)
        self.app.router.add_post('/sim/genetic/run_scenario', self.genetic_run_scenario)
        self.app.router.add_get('/sim/genetic/populations', self.genetic_list_populations)
        self.app.router.add_get('/sim/genetic/operators', self.genetic_list_operators)

        # === DATOS REALES GENÓMICOS (solo lectura) ===
        self.app.router.add_get('/sim/genetic/sources', self.genetic_list_sources)
        self.app.router.add_get('/sim/genetic/sources/ping', self.genetic_ping_sources)
        self.app.router.add_get('/sim/genetic/realtime', self.genetic_realtime_state)
        self.app.router.add_post('/sim/genetic/init_from_realtime', self.genetic_init_from_realtime)
        self.app.router.add_post('/sim/genetic/gene_search', self.genetic_gene_search)
        self.app.router.add_post('/sim/genetic/disease_genes', self.genetic_disease_genes)

        # === LAB CRIPTO (nuevo) ===
        self.app.router.add_post('/sim/crypto/init_population', self.crypto_init_population)
        self.app.router.add_post('/sim/crypto/apply_operator', self.crypto_apply_operator)
        self.app.router.add_post('/sim/crypto/run_scenario', self.crypto_run_scenario)
        self.app.router.add_post('/sim/crypto/stress_test', self.crypto_stress_test)
        self.app.router.add_post('/sim/crypto/fragility', self.crypto_fragility)
        self.app.router.add_get('/sim/crypto/populations', self.crypto_list_populations)
        self.app.router.add_get('/sim/crypto/operators', self.crypto_list_operators)
        self.app.router.add_get('/sim/crypto/markets', self.crypto_list_markets)

        # === DATOS REALES BINANCE (solo lectura) ===
        self.app.router.add_get('/sim/crypto/realtime', self.crypto_realtime_state)
        self.app.router.add_get('/sim/crypto/price/{symbol}', self.crypto_price)
        self.app.router.add_post('/sim/crypto/init_from_realtime', self.crypto_init_from_realtime)
        self.app.router.add_get('/sim/crypto/binance/ping', self.crypto_binance_ping)

        # === DATOS REALES FÍSICA ===
        self.app.router.add_get('/sim/physics/sources', self.physics_list_sources)
        self.app.router.add_get('/sim/physics/sources/ping', self.physics_ping_sources)
        self.app.router.add_get('/sim/physics/realtime', self.physics_realtime_state)
        self.app.router.add_get('/sim/physics/earthquakes', self.physics_earthquakes)
        self.app.router.add_get('/sim/physics/solar', self.physics_solar_activity)
        self.app.router.add_get('/sim/physics/space-weather', self.physics_space_weather)

        # === DATOS REALES MATEMÁTICAS ===
        self.app.router.add_get('/sim/math/sources', self.math_list_sources)
        self.app.router.add_get('/sim/math/sources/ping', self.math_ping_sources)
        self.app.router.add_get('/sim/math/realtime', self.math_realtime_state)
        self.app.router.add_get('/sim/math/sequence/{seq_id}', self.math_get_sequence)
        self.app.router.add_post('/sim/math/search', self.math_search)

        # === MODO CEREBRO (Alpha → Beta → Gamma) ===
        self.app.router.add_post('/cerebro/investigate', self.investigate)
        self.app.router.add_post('/cerebro/execute', self.execute_option)

        # Ciclo completo unificado
        self.app.router.add_post('/cycle', self.full_cycle)

    # ==========================================================================
    # HEALTH & INFO
    # ==========================================================================

    async def health(self, request):
        return web.json_response({
            'status': 'ok',
            'service': 'iaverso',
            'version': '4.0.0',
            'modes': ['cerebro', 'genetic', 'crypto'],
            'components': {
                'endolens': 'active',
                'neosynt': 'active',
                'genetic_sim': 'active',
                'crypto_sim': 'active',
                'alpha': 'active',
                'beta': 'active',
                'gamma': 'active'
            },
            'timestamp': datetime.now().isoformat()
        })

    async def list_modes(self, request):
        """Lista los modos disponibles para el frontend."""
        return web.json_response({
            'modes': [
                {
                    'id': 'cerebro',
                    'name': 'Cerebro Investigador',
                    'description': 'Alpha percibe, Beta explora, Gamma narra. Sin juicios.',
                    'icon': '🧠',
                    'agents': ['alpha', 'beta', 'gamma'],
                    'domain': None
                },
                {
                    'id': 'genetic',
                    'name': 'Lab Genético',
                    'description': 'Simulador de biología teórica. Poblaciones, operadores, escenarios in silico.',
                    'icon': '🧬',
                    'agents': ['alpha', 'beta', 'gamma'],
                    'domain': 'genetic'
                },
                {
                    'id': 'crypto',
                    'name': 'Lab Cripto',
                    'description': 'Simulador de dinámicas cripto in silico. NO es trading ni inversión.',
                    'icon': '₿',
                    'agents': ['alpha', 'beta', 'gamma'],
                    'domain': 'crypto',
                    'disclaimer': 'Simulación estructural únicamente. Sin conexión a exchanges ni dinero real.'
                }
            ]
        })

    async def list_domains(self, request):
        """Lista los dominios de simulación disponibles."""
        return web.json_response({
            'domains': [
                {
                    'id': 'genetic',
                    'name': 'Biología Teórica',
                    'description': 'Poblaciones abstractas, mutaciones, selección estructural',
                    'operators': len(self.genetic_sim._operators) if hasattr(self.genetic_sim, '_operators') else 8,
                    'icon': '🧬'
                },
                {
                    'id': 'crypto',
                    'name': 'Dinámicas Cripto',
                    'description': 'Estados de mercado abstractos, perturbaciones estructurales, fragilidad',
                    'operators': len(self.crypto_sim._operators),
                    'icon': '₿',
                    'disclaimer': 'In silico. Sin trading real.'
                }
            ]
        })

    async def list_agents(self, request):
        """Lista los agentes NORMA DURA."""
        return web.json_response({
            'agents': [
                {
                    'id': 'alpha',
                    'name': 'Perceptor Estructural',
                    'role': 'PERCIBE',
                    'mode': 'observacional',
                    'does': ['observa', 'describe', 'señala patrones'],
                    'does_not': ['pondera', 'concluye', 'prioriza'],
                    'icon': '👁️'
                },
                {
                    'id': 'beta',
                    'name': 'Explorador Relacional',
                    'role': 'EXPLORA',
                    'mode': 'exploratorio',
                    'does': ['conecta', 'propone relaciones', 'abre hipótesis'],
                    'does_not': ['sintetiza', 'decide relevancia', 'cierra preguntas'],
                    'icon': '🔗'
                },
                {
                    'id': 'gamma',
                    'name': 'Narrador',
                    'role': 'NARRA',
                    'mode': 'metacognitivo',
                    'does': ['integra', 'verbaliza incertidumbre', 'abre caminos'],
                    'does_not': ['juzga', 'concluye', 'aconseja'],
                    'icon': '📖'
                }
            ],
            'norma_dura': {
                'principle': 'Solo estructura, no semántica',
                'agents_are': 'Observadores, no jueces',
                'output': 'Preguntas abiertas, no conclusiones'
            }
        })

    # ==========================================================================
    # CORE (compartido)
    # ==========================================================================

    async def analyze(self, request):
        data = await request.json()
        text = data.get('text', '')

        state = self.endolens.process(text)
        lang = detect_language(text)

        return web.json_response({
            'signature': str(state.signature),
            'eseries': state.eseries.as_dict(),
            'stability': state.stability,
            'status': state.status,
            'invariants': state.invariants,
            'tensions': state.tensions,
            'language': {
                'code': lang.language,
                'confidence': lang.confidence,
                'layer': lang.layer
            }
        })

    async def resolve(self, request):
        data = await request.json()
        text = data.get('text', '')
        target = data.get('target', 'stable')

        state = self.endolens.process(text)
        resolution = self.neosynt.resolve(state, target)

        return web.json_response({
            'status': resolution.status,
            'stability_score': resolution.stability_score,
            'operators_applied': resolution.operators_applied,
            'alternatives_count': len(resolution.alternatives),
            'trace': resolution.trace
        })

    async def detect_lang(self, request):
        data = await request.json()
        text = data.get('text', '')
        result = detect_language(text)

        return web.json_response({
            'language': result.language,
            'confidence': result.confidence,
            'layer': result.layer,
            'discriminator': result.discriminator
        })

    # ==========================================================================
    # AGENTES NORMA DURA
    # ==========================================================================

    async def alpha_perceive(self, request):
        """Alpha PERCIBE - modo observacional."""
        data = await request.json()
        text = data.get('text', '')
        domain = data.get('domain', None)

        perception = self.alpha.perceive(text)

        response = {
            'agent': 'alpha',
            'role': 'perceptor',
            'mode': 'observacional',
            'perception': self.alpha.to_dict(perception),
            'description': self.alpha.describe(perception),
            'norma_dura': 'Observo, no pondero. Describo, no concluyo.'
        }

        if domain:
            response['domain'] = domain

        return web.json_response(response)

    async def beta_explore(self, request):
        """Beta EXPLORA - modo exploratorio."""
        data = await request.json()
        perception_data = data.get('perception', {})
        context = data.get('context', '')
        domain = data.get('domain', None)

        if not perception_data and context:
            perception = self.alpha.perceive(context)
            perception_data = self.alpha.to_dict(perception)

        exploration = self.beta.explore(perception_data, context)

        response = {
            'agent': 'beta',
            'role': 'explorador',
            'mode': 'exploratorio',
            'exploration': self.beta.to_dict(exploration),
            'description': self.beta.describe(exploration),
            'norma_dura': 'Conecto, no sintetizo. Abro hipótesis, no cierro.'
        }

        if domain:
            response['domain'] = domain

        return web.json_response(response)

    async def gamma_narrate(self, request):
        """Gamma NARRA - modo metacognitivo."""
        data = await request.json()
        query = data.get('query', '')
        domain = data.get('domain', None)

        state = self.endolens.process(query)
        resolution = self.neosynt.resolve(state)

        narration = self.gamma.narrate(query, state, resolution)
        formatted = self.gamma.format_full_narration(narration)

        response = {
            'agent': 'gamma',
            'role': 'narrador',
            'mode': 'metacognitivo',
            'narration': formatted,
            'questions': narration.questions,
            'options_count': len(narration.options),
            'norma_dura': 'Narro sin juzgar. Abro caminos, no decido.'
        }

        if domain:
            response['domain'] = domain

        return web.json_response(response)

    # ==========================================================================
    # LAB GENÉTICO
    # ==========================================================================

    async def genetic_init_population(self, request):
        """Inicializa población genética."""
        data = await request.json()
        seed = data.get('seed', '')
        size = data.get('size', 20)
        parameters = data.get('parameters', {})

        population = self.genetic_sim.init_population(seed, size, parameters)

        return web.json_response({
            'domain': 'genetic',
            'population_id': population.id,
            'size': population.size,
            'generation': population.generation,
            'avg_fitness': population.avg_fitness,
            'signature': getattr(population, 'signature', 'N/A')
        })

    async def genetic_apply_operator(self, request):
        """Aplica operador genético."""
        data = await request.json()
        population_id = data.get('population_id')
        operator = data.get('operator', 'O_point')
        iterations = data.get('iterations', 1)

        try:
            population = self.genetic_sim.apply_operator(population_id, operator, iterations)

            return web.json_response({
                'domain': 'genetic',
                'population_id': population.id,
                'generation': population.generation,
                'operator_applied': operator,
                'iterations': iterations,
                'avg_fitness': population.avg_fitness,
                'signature': getattr(population, 'signature', 'N/A')
            })
        except ValueError as e:
            return web.json_response({'error': str(e)}, status=404)

    async def genetic_run_scenario(self, request):
        """Ejecuta escenario genético."""
        data = await request.json()
        population_id = data.get('population_id')

        population = self.genetic_sim.populations.get(population_id)
        if not population:
            return web.json_response({'error': 'Population not found'}, status=404)

        scenario = self.genetic_sim.propose_scenario(population)
        result = self.genetic_sim.run_scenario(scenario)

        return web.json_response({
            'domain': 'genetic',
            'scenario_id': scenario.id,
            'operator': scenario.operator,
            'surprise': result.surprise,
            'success': result.success,
            'trace': result.trace
        })

    async def genetic_list_populations(self, request):
        """Lista poblaciones genéticas."""
        pops = self.genetic_sim.list_populations()
        return web.json_response({'domain': 'genetic', 'populations': pops})

    async def genetic_list_operators(self, request):
        """Lista operadores genéticos."""
        ops = self.genetic_sim.list_operators() if hasattr(self.genetic_sim, 'list_operators') else []
        return web.json_response({
            'domain': 'genetic',
            'operators': [
                {'id': 'O_point', 'name': 'Modificación puntual'},
                {'id': 'O_region', 'name': 'Modificación regional'},
                {'id': 'O_combine', 'name': 'Combinación'},
                {'id': 'O_reduce', 'name': 'Reducción'},
                {'id': 'O_expand', 'name': 'Expansión'},
                {'id': 'O_invert', 'name': 'Inversión'},
                {'id': 'O_duplicate', 'name': 'Duplicación'},
                {'id': 'O_permute', 'name': 'Permutación'},
            ]
        })

    # ==========================================================================
    # DATOS REALES GENÓMICOS
    # ==========================================================================

    async def genetic_list_sources(self, request):
        """Lista todas las fuentes de datos genómicos disponibles."""
        sources = self.genomic.list_sources()
        return web.json_response({
            'domain': 'genetic',
            'disclaimer': 'Fuentes de datos abiertos. Solo lectura. NO genera diagnósticos.',
            'sources': sources
        })

    async def genetic_ping_sources(self, request):
        """Verifica conexión con todas las fuentes genómicas."""
        try:
            pings = await self.genomic.ping_all()
            return web.json_response({
                'domain': 'genetic',
                'sources': pings,
                'connected': sum(1 for v in pings.values() if v),
                'total': len(pings)
            })
        except Exception as e:
            return web.json_response({
                'error': str(e)
            }, status=500)

    async def genetic_realtime_state(self, request):
        """
        Obtiene estado genético abstracto basado en datos reales.

        Combina múltiples fuentes (GEO, GTEx, ENCODE, etc.)
        ⚠️ Solo lectura. NO genera diagnósticos.
        """
        try:
            # Parámetros opcionales
            genes = request.query.get('genes', 'TP53,BRCA1,EGFR')
            genes = [g.strip() for g in genes.split(',')]
            disease = request.query.get('disease', None)
            sources = request.query.get('sources', None)
            if sources:
                sources = [s.strip() for s in sources.split(',')]

            state = await self.genomic.get_unified_state(
                genes=genes,
                disease=disease,
                sources=sources
            )

            return web.json_response({
                'domain': 'genetic',
                'disclaimer': 'Datos de fuentes públicas transformados a dimensiones abstractas. NO es diagnóstico.',
                'state': state.to_dict(),
                'dimensions': {
                    'expression_level': {'value': state.expression_level, 'description': 'Nivel de expresión normalizado'},
                    'variability': {'value': state.variability, 'description': 'Variabilidad entre muestras'},
                    'tissue_specificity': {'value': state.tissue_specificity, 'description': 'Especificidad tisular'},
                    'regulatory_complexity': {'value': state.regulatory_complexity, 'description': 'Complejidad regulatoria'},
                    'mutation_load': {'value': state.mutation_load, 'description': 'Carga mutacional'},
                    'pathway_connectivity': {'value': state.pathway_connectivity, 'description': 'Conectividad en rutas'},
                    'conservation': {'value': state.conservation, 'description': 'Conservación evolutiva'},
                    'disease_association': {'value': state.disease_association, 'description': 'Asociación a enfermedad'}
                },
                'query': {
                    'genes': genes,
                    'disease': disease,
                    'sources': sources or 'all'
                }
            })
        except Exception as e:
            return web.json_response({
                'error': str(e),
                'disclaimer': 'Error obteniendo datos.'
            }, status=500)

    async def genetic_init_from_realtime(self, request):
        """
        Inicializa población genética usando datos reales como base.

        Obtiene estado de múltiples fuentes y lo usa como semilla.
        ⚠️ La simulación posterior es in silico.
        """
        data = await request.json()
        size = data.get('size', 20)
        genes = data.get('genes', ['TP53', 'BRCA1', 'EGFR'])
        disease = data.get('disease', None)
        sources = data.get('sources', None)
        variance = data.get('variance', 0.1)

        try:
            # Obtener estado real
            real_state = await self.genomic.get_unified_state(
                genes=genes,
                disease=disease,
                sources=sources
            )

            # Usar estado real como base para la población
            parameters = {
                'base_state': real_state.to_dict(),
                'variance': variance,
                'source': 'genomic_realtime'
            }

            # Crear semilla descriptiva
            seed = f"realtime_{'_'.join(genes[:3])}_{real_state.timestamp.strftime('%Y%m%d_%H%M')}"

            population = self.genetic_sim.init_population(seed, size, parameters)

            return web.json_response({
                'domain': 'genetic',
                'disclaimer': 'Población inicializada con datos reales. Simulación posterior es in silico. NO es diagnóstico.',
                'population_id': population.id,
                'size': population.size,
                'generation': population.generation,
                'base_state': real_state.to_dict(),
                'sources_used': real_state.source,
                'genes_queried': genes,
                'disease_context': disease,
                'variance_applied': variance,
                'avg_fitness': population.avg_fitness,
                'signature': getattr(population, 'signature', 'N/A')
            })
        except Exception as e:
            return web.json_response({
                'error': str(e),
                'disclaimer': 'Error obteniendo datos reales.',
                'fallback': True
            }, status=500)

    async def genetic_gene_search(self, request):
        """
        Busca información de un gen en múltiples fuentes.

        ⚠️ Solo informativo. NO genera diagnósticos.
        """
        data = await request.json()
        gene = data.get('gene', 'TP53')
        sources = data.get('sources', ['GEO', '1000Genomes', 'GTEx'])

        results = {}

        try:
            if 'GEO' in sources:
                geo_result = await self.genomic.geo.search_gene_expression(gene)
                results['GEO'] = geo_result

            if '1000Genomes' in sources:
                genomes_result = await self.genomic.genomes1k.get_gene_variants(gene)
                results['1000Genomes'] = genomes_result

            if 'GTEx' in sources:
                gtex_result = await self.genomic.gtex.get_gene_expression(gene)
                results['GTEx'] = gtex_result

            if 'Orphanet' in sources:
                orphanet_result = await self.genomic.orphanet.get_gene_phenotypes(gene)
                results['Orphanet'] = orphanet_result

            if 'ENCODE' in sources:
                encode_result = await self.genomic.encode.get_gene_regulation(gene)
                results['ENCODE'] = encode_result

            if 'AllenBrain' in sources:
                allen_result = await self.genomic.allen.search_genes(gene)
                results['AllenBrain'] = allen_result

            return web.json_response({
                'domain': 'genetic',
                'disclaimer': 'Datos de fuentes públicas. Solo informativo. NO es diagnóstico.',
                'gene': gene,
                'sources_queried': list(results.keys()),
                'results': results
            })
        except Exception as e:
            return web.json_response({
                'error': str(e),
                'gene': gene
            }, status=500)

    async def genetic_disease_genes(self, request):
        """
        Busca genes asociados a un término de enfermedad.

        Usa datos de Orphanet y otras fuentes.
        ⚠️ Solo informativo. NO genera diagnósticos.
        """
        data = await request.json()
        disease_term = data.get('disease', 'muscular dystrophy')

        try:
            result = await self.genomic.orphanet.search_disease_genes(disease_term)

            return web.json_response({
                'domain': 'genetic',
                'disclaimer': 'Datos de fuentes públicas. Solo informativo. NO es diagnóstico médico.',
                'disease_term': disease_term,
                'genes_found': result['genes_found'],
                'genes': result['genes'],
                'source': 'Orphanet/Ensembl'
            })
        except Exception as e:
            return web.json_response({
                'error': str(e),
                'disease_term': disease_term
            }, status=500)

    # ==========================================================================
    # DATOS REALES FÍSICA
    # ==========================================================================

    async def physics_list_sources(self, request):
        """Lista todas las fuentes de datos físicos."""
        sources = self.physics.list_sources()
        return web.json_response({
            'domain': 'physics',
            'disclaimer': 'Datos físicos reales. Solo lectura. NO genera predicciones.',
            'sources': sources
        })

    async def physics_ping_sources(self, request):
        """Verifica conexión con todas las fuentes físicas."""
        try:
            pings = await self.physics.ping_all()
            return web.json_response({
                'domain': 'physics',
                'sources': pings,
                'connected': sum(1 for v in pings.values() if v),
                'total': len(pings)
            })
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def physics_realtime_state(self, request):
        """
        Obtiene estado físico abstracto combinando múltiples fuentes.

        ⚠️ Solo lectura. NO genera predicciones físicas.
        """
        try:
            sources = request.query.get('sources', None)
            if sources:
                sources = [s.strip() for s in sources.split(',')]

            state = await self.physics.get_unified_state(sources)

            return web.json_response({
                'domain': 'physics',
                'disclaimer': 'Datos reales transformados a dimensiones abstractas. NO es predicción.',
                'state': state.to_dict(),
                'dimensions': {
                    'energy_density': {'value': state.energy_density, 'description': 'Densidad energética normalizada'},
                    'field_intensity': {'value': state.field_intensity, 'description': 'Intensidad de campo'},
                    'temporal_stability': {'value': state.temporal_stability, 'description': 'Estabilidad temporal'},
                    'spatial_coherence': {'value': state.spatial_coherence, 'description': 'Coherencia espacial'},
                    'event_frequency': {'value': state.event_frequency, 'description': 'Frecuencia de eventos'},
                    'magnitude_distribution': {'value': state.magnitude_distribution, 'description': 'Distribución de magnitudes'},
                    'correlation_strength': {'value': state.correlation_strength, 'description': 'Fuerza de correlación'},
                    'anomaly_index': {'value': state.anomaly_index, 'description': 'Índice de anomalía'}
                }
            })
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def physics_earthquakes(self, request):
        """
        Obtiene sismos recientes de USGS.

        ⚠️ Solo informativo. NO genera predicciones sísmicas.
        """
        try:
            days = int(request.query.get('days', 7))
            min_mag = float(request.query.get('min_magnitude', 4.0))

            result = await self.physics.usgs.get_recent_earthquakes(
                min_magnitude=min_mag,
                days=days
            )

            return web.json_response({
                'domain': 'physics',
                'source': 'USGS',
                'disclaimer': 'Datos sísmicos reales. Solo informativo. NO es predicción.',
                **result
            })
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def physics_solar_activity(self, request):
        """
        Obtiene actividad solar de NASA.

        ⚠️ Solo informativo. NO genera predicciones.
        """
        try:
            days = int(request.query.get('days', 30))

            flares = await self.physics.nasa.get_solar_flares(days)
            storms = await self.physics.nasa.get_geomagnetic_storms(days)
            cmes = await self.physics.nasa.get_coronal_mass_ejections(days)

            return web.json_response({
                'domain': 'physics',
                'source': 'NASA_DONKI',
                'disclaimer': 'Datos de actividad solar. Solo informativo.',
                'period_days': days,
                'solar_flares': flares,
                'geomagnetic_storms': storms,
                'coronal_mass_ejections': cmes,
                'summary': {
                    'total_flares': flares['count'],
                    'total_storms': storms['count'],
                    'total_cmes': cmes['count']
                }
            })
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def physics_space_weather(self, request):
        """
        Obtiene clima espacial de NOAA.

        ⚠️ Solo informativo. NO genera predicciones.
        """
        try:
            solar_wind = await self.physics.noaa.get_solar_wind()
            kp_index = await self.physics.noaa.get_geomagnetic_indices()

            return web.json_response({
                'domain': 'physics',
                'source': 'NOAA',
                'disclaimer': 'Datos de clima espacial. Solo informativo.',
                'solar_wind': solar_wind,
                'geomagnetic_index': kp_index
            })
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    # ==========================================================================
    # DATOS REALES MATEMÁTICAS
    # ==========================================================================

    async def math_list_sources(self, request):
        """Lista todas las fuentes de datos matemáticos."""
        sources = self.math.list_sources()
        return web.json_response({
            'domain': 'math',
            'disclaimer': 'Estructuras matemáticas verificadas. Solo lectura.',
            'sources': sources
        })

    async def math_ping_sources(self, request):
        """Verifica conexión con todas las fuentes matemáticas."""
        try:
            pings = await self.math.ping_all()
            return web.json_response({
                'domain': 'math',
                'sources': pings,
                'connected': sum(1 for v in pings.values() if v),
                'total': len(pings)
            })
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def math_realtime_state(self, request):
        """
        Obtiene estado matemático abstracto combinando múltiples fuentes.

        ⚠️ Solo lectura. Estructuras matemáticas puras.
        """
        try:
            sources = request.query.get('sources', None)
            if sources:
                sources = [s.strip() for s in sources.split(',')]

            state = await self.math.get_unified_state(sources)

            return web.json_response({
                'domain': 'math',
                'disclaimer': 'Estructuras matemáticas transformadas a dimensiones abstractas.',
                'state': state.to_dict(),
                'dimensions': {
                    'pattern_density': {'value': state.pattern_density, 'description': 'Densidad de patrones'},
                    'sequence_regularity': {'value': state.sequence_regularity, 'description': 'Regularidad de secuencia'},
                    'structural_complexity': {'value': state.structural_complexity, 'description': 'Complejidad estructural'},
                    'symmetry_index': {'value': state.symmetry_index, 'description': 'Índice de simetría'},
                    'growth_rate': {'value': state.growth_rate, 'description': 'Tasa de crecimiento'},
                    'periodicity': {'value': state.periodicity, 'description': 'Periodicidad'},
                    'correlation_depth': {'value': state.correlation_depth, 'description': 'Profundidad de correlación'},
                    'novelty_index': {'value': state.novelty_index, 'description': 'Índice de novedad'}
                }
            })
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def math_get_sequence(self, request):
        """
        Obtiene una secuencia específica de OEIS.

        ⚠️ Datos matemáticos verificados.
        """
        seq_id = request.match_info.get('seq_id', 'A000045')

        try:
            result = await self.math.oeis.get_sequence(seq_id)

            return web.json_response({
                'domain': 'math',
                'source': 'OEIS',
                'disclaimer': 'Secuencia matemática verificada.',
                **result
            })
        except Exception as e:
            return web.json_response({'error': str(e), 'seq_id': seq_id}, status=500)

    async def math_search(self, request):
        """
        Busca secuencias o estructuras matemáticas.

        ⚠️ Solo lectura. Estructuras verificadas.
        """
        data = await request.json()
        query = data.get('query', '')
        source = data.get('source', 'OEIS')

        try:
            if source == 'OEIS':
                result = await self.math.oeis.search_sequences(query)
            else:
                result = {'error': f'Source {source} not supported for search'}

            return web.json_response({
                'domain': 'math',
                'source': source,
                'disclaimer': 'Resultados de búsqueda matemática.',
                **result
            })
        except Exception as e:
            return web.json_response({'error': str(e), 'query': query}, status=500)

    # ==========================================================================
    # LAB CRIPTO
    # ==========================================================================

    async def crypto_init_population(self, request):
        """
        Inicializa población de estados de mercado abstractos.

        ⚠️ Simulación in silico. NO es trading.
        """
        data = await request.json()
        seed = data.get('seed', '')
        size = data.get('size', 20)
        parameters = data.get('parameters', {})

        population = self.crypto_sim.init_population(seed, size, parameters)

        return web.json_response({
            'domain': 'crypto',
            'disclaimer': 'Simulación estructural in silico. NO es trading ni inversión.',
            'population_id': population.id,
            'size': population.size,
            'generation': population.generation,
            'avg_fitness': population.avg_fitness,
            'signature': population.signature,
            'sample_elements': [
                {
                    'id': e.id,
                    'data': e.data,
                    'fitness': e.fitness
                }
                for e in population.elements[:3]
            ]
        })

    async def crypto_apply_operator(self, request):
        """
        Aplica operador estructural (perturbación).

        Los operadores son PERTURBACIONES del sistema, no acciones humanas.
        """
        data = await request.json()
        population_id = data.get('population_id')
        operator = data.get('operator', 'O_volatility_spike')
        iterations = data.get('iterations', 1)
        parameters = data.get('parameters', {})

        try:
            population = self.crypto_sim.apply_operator(
                population_id, operator, iterations, parameters
            )

            return web.json_response({
                'domain': 'crypto',
                'disclaimer': 'Perturbación estructural simulada. NO es predicción.',
                'population_id': population.id,
                'generation': population.generation,
                'operator_applied': operator,
                'iterations': iterations,
                'avg_fitness': population.avg_fitness,
                'signature': population.signature,
                'history': population.history[-3:]
            })
        except ValueError as e:
            return web.json_response({'error': str(e)}, status=404)

    async def crypto_run_scenario(self, request):
        """
        Ejecuta escenario in silico.

        Mide sorpresa estructural, no predice comportamiento real.
        """
        data = await request.json()
        population_id = data.get('population_id')
        operator = data.get('operator', None)

        population = self.crypto_sim.get_population(population_id)
        if not population:
            return web.json_response({'error': 'Population not found'}, status=404)

        scenario = self.crypto_sim.propose_scenario(population, operator)
        result = self.crypto_sim.run_scenario(scenario)

        return web.json_response({
            'domain': 'crypto',
            'disclaimer': 'Escenario in silico. Mide coherencia estructural, no predice mercados.',
            'scenario_id': scenario.id,
            'operator': scenario.operator,
            'prediction': {
                'expected_stability': scenario.prediction.expected_stability,
                'confidence': scenario.prediction.confidence
            },
            'result': {
                'observed_stability': result.observed.stability if result.observed else None,
                'surprise': result.surprise,
                'success': result.success
            },
            'trace': result.trace,
            'model_update': result.model_update
        })

    async def crypto_stress_test(self, request):
        """
        Test de estrés estructural.

        Aplica secuencia de perturbaciones y mide degradación.
        NO es predicción de comportamiento real.
        """
        data = await request.json()
        population_id = data.get('population_id')
        operators = data.get('operators', ['O_volatility_spike', 'O_liquidity_drain'])
        iterations = data.get('iterations', 5)

        population = self.crypto_sim.get_population(population_id)
        if not population:
            return web.json_response({'error': 'Population not found'}, status=404)

        result = self.crypto_sim.stress_test(population, operators, iterations)

        return web.json_response({
            'domain': 'crypto',
            'disclaimer': 'Test de robustez estructural in silico. NO es predicción.',
            'population_id': population_id,
            **result
        })

    async def crypto_fragility(self, request):
        """
        Análisis de fragilidad estructural.

        Detecta combinaciones estructuralmente frágiles.
        NO es análisis de riesgo de inversión.
        """
        data = await request.json()
        population_id = data.get('population_id')

        population = self.crypto_sim.get_population(population_id)
        if not population:
            return web.json_response({'error': 'Population not found'}, status=404)

        result = self.crypto_sim.analyze_fragility(population)

        return web.json_response({
            'domain': 'crypto',
            'disclaimer': 'Análisis estructural in silico. NO es evaluación de riesgo financiero.',
            'population_id': population_id,
            **result
        })

    async def crypto_list_populations(self, request):
        """Lista poblaciones cripto activas."""
        pops = self.crypto_sim.list_populations()
        return web.json_response({
            'domain': 'crypto',
            'populations': pops
        })

    async def crypto_list_operators(self, request):
        """Lista operadores cripto disponibles."""
        ops = [
            {
                'id': spec.id,
                'name': spec.name,
                'description': spec.description,
                'parameters': spec.parameters
            }
            for spec in self.crypto_sim.list_operators()
        ]
        return web.json_response({
            'domain': 'crypto',
            'disclaimer': 'Operadores como perturbaciones estructurales, no acciones de trading.',
            'operators': ops
        })

    async def crypto_list_markets(self, request):
        """
        Lista variables del estado de mercado abstracto.

        Estas NO son métricas reales de mercado.
        Son dimensiones estructurales abstractas.
        """
        return web.json_response({
            'domain': 'crypto',
            'disclaimer': 'Variables estructurales abstractas, no métricas de mercado reales.',
            'variables': [
                {'id': 'liquidity', 'name': 'Liquidez', 'description': 'Disponibilidad de flujo (0-1)'},
                {'id': 'volatility', 'name': 'Volatilidad', 'description': 'Variabilidad estructural (0-1)'},
                {'id': 'concentration', 'name': 'Concentración', 'description': 'Distribución de nodos (0-1)'},
                {'id': 'latency', 'name': 'Latencia', 'description': 'Tiempo de propagación (0-1)'},
                {'id': 'sentiment', 'name': 'Momentum', 'description': 'Momentum agregado (0-1)'},
                {'id': 'leverage', 'name': 'Apalancamiento', 'description': 'Apalancamiento sistémico (0-1)'},
                {'id': 'network_load', 'name': 'Carga de red', 'description': 'Carga de red (0-1)'},
                {'id': 'regulatory_pressure', 'name': 'Presión regulatoria', 'description': 'Presión regulatoria (0-1)'}
            ]
        })

    # ==========================================================================
    # DATOS REALES BINANCE (solo lectura)
    # ==========================================================================

    async def crypto_binance_ping(self, request):
        """Verifica conexión con Binance."""
        try:
            is_connected = await self.binance.ping()
            return web.json_response({
                'status': 'connected' if is_connected else 'disconnected',
                'source': 'binance',
                'disclaimer': 'Solo lectura. NO ejecuta trades.'
            })
        except Exception as e:
            return web.json_response({
                'status': 'error',
                'error': str(e)
            }, status=500)

    async def crypto_realtime_state(self, request):
        """
        Obtiene estado abstracto del mercado basado en datos reales.

        Transforma datos de Binance a dimensiones estructurales 0-1.
        ⚠️ Solo lectura. NO es señal de trading.
        """
        try:
            symbols = request.query.get('symbols', None)
            if symbols:
                symbols = symbols.split(',')

            abstract_state = await self.binance.get_abstract_state(symbols)

            return web.json_response({
                'domain': 'crypto',
                'source': 'binance',
                'disclaimer': 'Datos transformados a dimensiones abstractas. NO es señal de trading.',
                'state': abstract_state.to_dict(),
                'dimensions': {
                    'liquidity': {'value': abstract_state.liquidity, 'description': 'Volumen 24h normalizado'},
                    'volatility': {'value': abstract_state.volatility, 'description': 'Variación high-low normalizada'},
                    'concentration': {'value': abstract_state.concentration, 'description': 'Ratio top coins'},
                    'latency': {'value': abstract_state.latency, 'description': 'Inverso frecuencia trades'},
                    'sentiment': {'value': abstract_state.sentiment, 'description': 'Cambio 24h normalizado'},
                    'leverage': {'value': abstract_state.leverage, 'description': 'Estimación apalancamiento'},
                    'network_load': {'value': abstract_state.network_load, 'description': 'Trades normalizados'},
                    'regulatory_pressure': {'value': abstract_state.regulatory_pressure, 'description': 'Valor base (no disponible via API)'}
                }
            })
        except Exception as e:
            return web.json_response({
                'error': str(e),
                'disclaimer': 'Error obteniendo datos. Verifique conexión.'
            }, status=500)

    async def crypto_price(self, request):
        """
        Obtiene precio actual de un símbolo.

        ⚠️ Solo informativo. NO es señal de trading.
        """
        symbol = request.match_info.get('symbol', 'BTCUSDT').upper()

        try:
            price = await self.binance.get_price(symbol)

            return web.json_response({
                'domain': 'crypto',
                'source': 'binance',
                'disclaimer': 'Solo informativo. NO es señal de trading.',
                'symbol': symbol,
                'price': price,
                'currency': 'USDT'
            })
        except Exception as e:
            return web.json_response({
                'error': str(e),
                'symbol': symbol
            }, status=500)

    async def crypto_init_from_realtime(self, request):
        """
        Inicializa población usando datos reales como base.

        Obtiene estado actual de Binance y lo usa como semilla
        para la población de simulación.

        ⚠️ La simulación posterior es in silico.
        Los datos iniciales son reales pero la evolución es simulada.
        """
        data = await request.json()
        size = data.get('size', 20)
        symbols = data.get('symbols', None)
        variance = data.get('variance', 0.1)  # Varianza para generar población

        try:
            # Obtener estado real
            real_state = await self.binance.get_abstract_state(symbols)

            # Usar estado real como base para la población
            parameters = {
                'base_state': real_state.to_dict(),
                'variance': variance,
                'source': 'binance_realtime'
            }

            # Crear semilla descriptiva
            seed = f"realtime_{real_state.timestamp.strftime('%Y%m%d_%H%M')}"

            population = self.crypto_sim.init_population(seed, size, parameters)

            return web.json_response({
                'domain': 'crypto',
                'source': 'binance',
                'disclaimer': 'Población inicializada con datos reales. Simulación posterior es in silico.',
                'population_id': population.id,
                'size': population.size,
                'generation': population.generation,
                'base_state': real_state.to_dict(),
                'variance_applied': variance,
                'avg_fitness': population.avg_fitness,
                'signature': population.signature,
                'sample_elements': [
                    {
                        'id': e.id,
                        'data': e.data,
                        'fitness': e.fitness
                    }
                    for e in population.elements[:3]
                ]
            })
        except Exception as e:
            return web.json_response({
                'error': str(e),
                'disclaimer': 'Error obteniendo datos reales. Usando datos simulados.',
                'fallback': True
            }, status=500)

    # ==========================================================================
    # CEREBRO (Alpha → Beta → Gamma)
    # ==========================================================================

    async def investigate(self, request):
        """
        Investigación completa: Alpha → Beta → Gamma
        """
        data = await request.json()
        query = data.get('query', '')
        domain = data.get('domain', None)

        # 1. ALPHA percibe (modo observacional)
        perception = self.alpha.perceive(query)
        perception_dict = self.alpha.to_dict(perception)

        # 2. BETA explora (modo exploratorio)
        exploration = self.beta.explore(perception_dict, query)
        exploration_dict = self.beta.to_dict(exploration)

        # 3. GAMMA narra (modo metacognitivo)
        state = self.endolens.process(query)
        resolution = self.neosynt.resolve(state)
        narration = self.gamma.narrate(query, state, resolution)
        formatted = self.gamma.format_full_narration(narration)

        response = {
            'mode': 'cerebro',
            'domain': domain,
            'cycle': len(self.gamma.history),
            'alpha': {
                'role': 'percepción (observacional)',
                'entities': len(perception_dict.get('entities', [])),
                'tensions': len(perception_dict.get('tensions', [])),
                'patterns': len(perception_dict.get('patterns', [])),
                'description': self.alpha.describe(perception)
            },
            'beta': {
                'role': 'exploración (relacional)',
                'connections': len(exploration_dict.get('connections', [])),
                'hypotheses': len(exploration_dict.get('hypotheses', [])),
                'open_questions': exploration_dict.get('open_questions', []),
                'description': self.beta.describe(exploration)
            },
            'gamma': {
                'role': 'narración (metacognitivo)',
                'narration': formatted,
                'questions': narration.questions
            },
            'state': {
                'signature': str(state.signature),
                'stability': state.stability,
                'eseries': state.eseries.as_dict()
            }
        }

        return web.json_response(response)

    async def execute_option(self, request):
        """Ejecuta una opción de Gamma."""
        data = await request.json()
        option_index = data.get('option', 0)
        query = data.get('query', '')

        state = self.endolens.process(query)
        result = self.gamma.execute_and_narrate(option_index, state)

        return web.json_response({
            'mode': 'cerebro',
            'narration': result,
            'cycle': len(self.gamma.history)
        })

    # ==========================================================================
    # CICLO UNIFICADO
    # ==========================================================================

    async def full_cycle(self, request):
        """
        Ciclo completo unificado.

        Soporta:
        - domain: 'genetic' | 'crypto' | None
        - mode: 'cerebro' | 'sim'
        - alpha/beta: pesos de exploración (0-1)
        """
        data = await request.json()
        query = data.get('input', data.get('query', ''))
        domain = data.get('domain', None)
        mode = data.get('mode', 'cerebro')
        alpha_weight = data.get('alpha', 0.5)
        beta_weight = data.get('beta', 0.5)

        # Normalizar pesos
        total = alpha_weight + beta_weight
        if total > 0:
            alpha_weight /= total
            beta_weight /= total

        response = {
            'domain': domain,
            'mode': mode,
            'weights': {
                'alpha': alpha_weight,
                'beta': beta_weight,
                'note': 'Pesos por defecto usados' if data.get('alpha') is None else 'Pesos personalizados'
            }
        }

        # 1. PERCEPCIÓN (Alpha + EndoLens)
        perception = self.alpha.perceive(query)
        state = self.endolens.process(query)

        response['perception'] = {
            'alpha': {
                'entities': len(self.alpha.to_dict(perception).get('entities', [])),
                'tensions': len(self.alpha.to_dict(perception).get('tensions', []))
            },
            'endolens': {
                'signature': str(state.signature),
                'stability': state.stability,
                'eseries': state.eseries.as_dict()
            }
        }

        # 2. EXPLORACIÓN (Beta)
        exploration = self.beta.explore(self.alpha.to_dict(perception), query)
        response['exploration'] = {
            'connections': len(self.beta.to_dict(exploration).get('connections', [])),
            'hypotheses': len(self.beta.to_dict(exploration).get('hypotheses', [])),
            'open_questions': self.beta.to_dict(exploration).get('open_questions', [])
        }

        # 3. ACCIÓN según dominio y modo
        if mode == 'sim' and domain:
            sim = self.crypto_sim if domain == 'crypto' else self.genetic_sim

            # Iniciar población
            population = sim.init_population(query, size=10)

            # Proponer escenario
            scenario = sim.propose_scenario(population)
            result = sim.run_scenario(scenario)

            response['simulation'] = {
                'population_id': population.id,
                'scenario_id': scenario.id,
                'operator': scenario.operator,
                'surprise': result.surprise,
                'success': result.success,
                'observed_stability': result.observed.stability if result.observed else None
            }

            if domain == 'crypto':
                response['simulation']['disclaimer'] = 'Simulación in silico. NO es predicción de mercado.'
                fragility = sim.analyze_fragility(population)
                response['simulation']['fragility'] = fragility

        # 4. NARRACIÓN (Gamma)
        resolution = self.neosynt.resolve(state)
        narration = self.gamma.narrate(query, state, resolution)

        response['narration'] = self.gamma.format_full_narration(narration)
        response['questions'] = narration.questions
        response['cycle'] = len(self.gamma.history)

        # Gamma debe mencionar si no se pasaron pesos
        if data.get('alpha') is None:
            response['gamma_note'] = 'Pesos alpha/beta no especificados. Usando valores por defecto (0.5/0.5).'

        return web.json_response(response)


def create_app():
    api = IAVersoAPI()
    return api.app


if __name__ == '__main__':
    app = create_app()
    print("=" * 60)
    print("IAVERSO API Server v4.0 - Laboratorios In Silico")
    print("=" * 60)
    print("MODOS:")
    print("  🧠 CEREBRO  - Investigación con Alpha → Beta → Gamma")
    print("  🧬 GENÉTICO - Simulador biología teórica in silico")
    print("  ₿  CRIPTO   - Simulador dinámicas cripto in silico")
    print("-" * 60)
    print("AGENTES NORMA DURA:")
    print("  ALPHA: Percibe, no pondera")
    print("  BETA:  Conecta, no sintetiza")
    print("  GAMMA: Narra, no juzga")
    print("-" * 60)
    print("⚠️  LAB CRIPTO: Solo simulación. NO es trading.")
    print("=" * 60)
    print("Escuchando en 0.0.0.0:8800")
    web.run_app(app, host='0.0.0.0', port=8800)
