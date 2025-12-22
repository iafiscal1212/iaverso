"""
Genomic Data Clients - Obtención de datos genómicos reales

NORMA DURA:
- Solo OBTIENE datos, no interpreta clínicamente
- Transforma datos a dimensiones estructurales abstractas
- NO genera diagnósticos ni recomendaciones médicas
- Los datos se usan para simulación in silico

Fuentes de acceso abierto:
- GEO (Gene Expression Omnibus): Expresión génica
- 1000 Genomes: Variación genética poblacional
- GTEx: Expresión por tejido
- Orphanet: Genes-enfermedades raras
- ENCODE: Regulación epigenética
- ArrayExpress: Expresión génica (Europa)
- Allen Brain Atlas: Expresión cerebral

Dimensiones abstractas derivadas:
- expression_level: Nivel de expresión normalizado
- variability: Variabilidad entre muestras
- tissue_specificity: Especificidad tisular
- regulatory_complexity: Complejidad regulatoria
- mutation_load: Carga mutacional
- pathway_connectivity: Conectividad en rutas
- conservation: Conservación evolutiva
- disease_association: Asociación a enfermedad
"""

import aiohttp
import asyncio
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import re


@dataclass
class GeneticState:
    """Estado genético abstracto derivado de datos reales."""
    expression_level: float      # 0-1: Nivel de expresión
    variability: float           # 0-1: Variabilidad entre muestras
    tissue_specificity: float    # 0-1: Especificidad tisular
    regulatory_complexity: float # 0-1: Complejidad regulatoria
    mutation_load: float         # 0-1: Carga mutacional
    pathway_connectivity: float  # 0-1: Conectividad en rutas
    conservation: float          # 0-1: Conservación evolutiva
    disease_association: float   # 0-1: Asociación a enfermedad

    source: str = ""
    genes: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            'expression_level': self.expression_level,
            'variability': self.variability,
            'tissue_specificity': self.tissue_specificity,
            'regulatory_complexity': self.regulatory_complexity,
            'mutation_load': self.mutation_load,
            'pathway_connectivity': self.pathway_connectivity,
            'conservation': self.conservation,
            'disease_association': self.disease_association,
            'source': self.source,
            'genes': self.genes[:10],  # Limitar para respuesta
            'timestamp': self.timestamp.isoformat()
        }


# =============================================================================
# GEO CLIENT - Gene Expression Omnibus
# =============================================================================

class GEOClient:
    """
    Cliente para Gene Expression Omnibus (NCBI).

    Acceso: Totalmente abierto
    Datos: Expresión génica (RNA-seq, microarrays)
    Uso: Perfiles de estabilidad/inestabilidad génica

    ⚠️ Solo lectura. NO genera diagnósticos.
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def search_datasets(self, query: str, max_results: int = 10) -> List[Dict]:
        """Busca datasets en GEO."""
        session = await self._get_session()

        params = {
            'db': 'gds',
            'term': query,
            'retmax': max_results,
            'retmode': 'json'
        }

        async with session.get(f"{self.BASE_URL}/esearch.fcgi", params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                ids = data.get('esearchresult', {}).get('idlist', [])
                return [{'id': id, 'database': 'GEO'} for id in ids]
            return []

    async def get_dataset_info(self, dataset_id: str) -> Dict:
        """Obtiene información de un dataset específico."""
        session = await self._get_session()

        params = {
            'db': 'gds',
            'id': dataset_id,
            'retmode': 'json'
        }

        async with session.get(f"{self.BASE_URL}/esummary.fcgi", params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                result = data.get('result', {})
                if dataset_id in result:
                    info = result[dataset_id]
                    return {
                        'id': dataset_id,
                        'title': info.get('title', ''),
                        'summary': info.get('summary', ''),
                        'organism': info.get('taxon', ''),
                        'samples': info.get('n_samples', 0),
                        'platform': info.get('gpl', ''),
                        'type': info.get('gdstype', '')
                    }
            return {}

    async def search_gene_expression(self, gene: str, disease: str = None) -> Dict:
        """Busca expresión de un gen, opcionalmente en contexto de enfermedad."""
        query = f"{gene}[Gene] AND Homo sapiens[Organism]"
        if disease:
            query += f" AND {disease}"

        datasets = await self.search_datasets(query, max_results=5)

        return {
            'gene': gene,
            'disease': disease,
            'datasets_found': len(datasets),
            'datasets': datasets
        }

    async def get_abstract_state(self, genes: List[str] = None, disease: str = None) -> GeneticState:
        """
        Genera estado abstracto basado en búsqueda en GEO.

        Transforma resultados a dimensiones estructurales.
        """
        genes = genes or ['TP53', 'BRCA1', 'EGFR']  # Genes comunes de estudio

        total_datasets = 0
        disease_datasets = 0

        for gene in genes[:5]:  # Limitar consultas
            result = await self.search_gene_expression(gene, disease)
            total_datasets += result['datasets_found']
            if disease:
                disease_datasets += result['datasets_found']

        # Transformar a dimensiones abstractas
        # Basado en disponibilidad de datos (más datos = más estudiado)
        expression_level = min(1.0, total_datasets / 50)  # Normalizar
        variability = 0.5 + (0.3 if disease else 0)  # Enfermedades = más variabilidad
        disease_association = min(1.0, disease_datasets / 20) if disease else 0.3

        return GeneticState(
            expression_level=round(expression_level, 3),
            variability=round(variability, 3),
            tissue_specificity=0.5,  # Requiere análisis más profundo
            regulatory_complexity=0.5,
            mutation_load=0.4 if disease else 0.2,
            pathway_connectivity=0.6,
            conservation=0.7,  # Genes humanos conservados
            disease_association=round(disease_association, 3),
            source='GEO',
            genes=genes
        )

    async def ping(self) -> bool:
        """Verifica conexión con GEO."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.BASE_URL}/einfo.fcgi?db=gds&retmode=json") as resp:
                return resp.status == 200
        except:
            return False


# =============================================================================
# 1000 GENOMES CLIENT
# =============================================================================

class ThousandGenomesClient:
    """
    Cliente para 1000 Genomes Project.

    Acceso: Totalmente abierto
    Datos: Variación genética poblacional
    Uso: Baseline genético, diversidad, simulaciones

    ⚠️ Solo lectura. NO genera diagnósticos.
    """

    # API REST de Ensembl para 1000 Genomes
    BASE_URL = "https://rest.ensembl.org"

    # Poblaciones del proyecto
    POPULATIONS = [
        'AFR',  # Africana
        'AMR',  # Americana
        'EAS',  # Este asiático
        'EUR',  # Europea
        'SAS'   # Sur asiático
    ]

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={'Content-Type': 'application/json'}
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_variant_info(self, variant_id: str) -> Dict:
        """Obtiene información de una variante específica."""
        session = await self._get_session()

        url = f"{self.BASE_URL}/variation/human/{variant_id}?pops=1"

        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                return {
                    'id': variant_id,
                    'source': data.get('source', ''),
                    'mappings': len(data.get('mappings', [])),
                    'populations': data.get('populations', [])
                }
            return {}

    async def get_gene_variants(self, gene: str) -> Dict:
        """Obtiene variantes conocidas para un gen."""
        session = await self._get_session()

        # Primero obtener el ID del gen
        url = f"{self.BASE_URL}/lookup/symbol/homo_sapiens/{gene}"

        async with session.get(url) as resp:
            if resp.status == 200:
                gene_data = await resp.json()
                gene_id = gene_data.get('id', '')

                # Obtener variantes en la región del gen
                region = f"{gene_data.get('seq_region_name')}:{gene_data.get('start')}-{gene_data.get('end')}"

                return {
                    'gene': gene,
                    'gene_id': gene_id,
                    'region': region,
                    'biotype': gene_data.get('biotype', ''),
                    'description': gene_data.get('description', '')
                }
            return {'gene': gene, 'error': 'Gene not found'}

    async def get_population_frequencies(self, variant_id: str) -> Dict:
        """Obtiene frecuencias alélicas por población."""
        session = await self._get_session()

        url = f"{self.BASE_URL}/variation/human/{variant_id}?pops=1"

        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                populations = data.get('populations', [])

                # Filtrar poblaciones de 1000 Genomes
                freqs = {}
                for pop in populations:
                    if '1000GENOMES' in pop.get('population', ''):
                        pop_name = pop['population'].split(':')[-1] if ':' in pop['population'] else pop['population']
                        freqs[pop_name] = pop.get('frequency', 0)

                return {
                    'variant': variant_id,
                    'frequencies': freqs,
                    'populations_found': len(freqs)
                }
            return {}

    async def get_abstract_state(self, genes: List[str] = None) -> GeneticState:
        """
        Genera estado abstracto basado en datos de 1000 Genomes.
        """
        genes = genes or ['TP53', 'BRCA1', 'APOE']

        gene_info = []
        for gene in genes[:3]:
            info = await self.get_gene_variants(gene)
            if 'error' not in info:
                gene_info.append(info)

        # Calcular dimensiones abstractas basadas en la información
        n_genes = len(gene_info)

        return GeneticState(
            expression_level=0.5,  # No aplicable directamente
            variability=0.6,  # Alta variabilidad en poblaciones
            tissue_specificity=0.4,
            regulatory_complexity=0.5,
            mutation_load=0.3,  # Población sana
            pathway_connectivity=0.5,
            conservation=0.8,  # Genes conservados
            disease_association=0.2,  # Población sana
            source='1000Genomes',
            genes=genes
        )

    async def ping(self) -> bool:
        """Verifica conexión con Ensembl/1000 Genomes."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.BASE_URL}/info/ping") as resp:
                return resp.status == 200
        except:
            return False


# =============================================================================
# GTEx CLIENT - Genotype-Tissue Expression
# =============================================================================

class GTExClient:
    """
    Cliente para GTEx Portal.

    Acceso: Abierto (API pública)
    Datos: Expresión génica por tejido
    Uso: Contexto tisular, eQTLs

    ⚠️ Solo lectura. NO genera diagnósticos.
    """

    BASE_URL = "https://gtexportal.org/api/v2"

    # Tejidos principales
    TISSUES = [
        'Brain', 'Heart', 'Liver', 'Lung', 'Kidney',
        'Muscle', 'Adipose', 'Blood', 'Skin', 'Colon'
    ]

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_gene_expression(self, gene: str) -> Dict:
        """Obtiene expresión de un gen en diferentes tejidos."""
        session = await self._get_session()

        url = f"{self.BASE_URL}/expression/geneExpression"
        params = {
            'geneSymbol': gene,
            'datasetId': 'gtex_v8'
        }

        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        'gene': gene,
                        'tissues': len(data.get('data', [])),
                        'expression_data': data.get('data', [])[:10]
                    }
        except:
            pass

        return {'gene': gene, 'tissues': 0, 'expression_data': []}

    async def get_tissue_info(self) -> List[Dict]:
        """Obtiene información de tejidos disponibles."""
        session = await self._get_session()

        url = f"{self.BASE_URL}/dataset/tissueSiteDetail"

        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('data', [])[:20]
        except:
            pass

        return []

    async def get_abstract_state(self, genes: List[str] = None) -> GeneticState:
        """
        Genera estado abstracto basado en GTEx.
        """
        genes = genes or ['TP53', 'GAPDH', 'ACTB']

        expression_data = []
        for gene in genes[:3]:
            data = await self.get_gene_expression(gene)
            expression_data.append(data)

        # Calcular especificidad tisular
        total_tissues = sum(d['tissues'] for d in expression_data)
        tissue_specificity = min(1.0, 1.0 - (total_tissues / 150))  # Menos tejidos = más específico

        return GeneticState(
            expression_level=0.6,
            variability=0.4,
            tissue_specificity=round(tissue_specificity, 3),
            regulatory_complexity=0.6,
            mutation_load=0.3,
            pathway_connectivity=0.5,
            conservation=0.7,
            disease_association=0.3,
            source='GTEx',
            genes=genes
        )

    async def ping(self) -> bool:
        """Verifica conexión con GTEx."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.BASE_URL}/dataset/tissueSiteDetail") as resp:
                return resp.status == 200
        except:
            return False


# =============================================================================
# ORPHANET CLIENT - Enfermedades Raras
# =============================================================================

class OrphanetClient:
    """
    Cliente para Orphanet.

    Acceso: Abierto
    Datos: Genes asociados a enfermedades raras
    Uso: Mapas estructurales gen→patología

    ⚠️ Solo lectura. NO genera diagnósticos.
    """

    # Orphanet no tiene API REST pública directa, usamos datos de Ensembl
    BASE_URL = "https://rest.ensembl.org"

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={'Content-Type': 'application/json'}
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_gene_phenotypes(self, gene: str) -> Dict:
        """Obtiene fenotipos asociados a un gen."""
        session = await self._get_session()

        url = f"{self.BASE_URL}/phenotype/gene/homo_sapiens/{gene}"

        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    # Filtrar fenotipos de Orphanet
                    orphanet_phenotypes = [
                        p for p in data
                        if 'orphanet' in p.get('source', '').lower()
                    ]

                    return {
                        'gene': gene,
                        'total_phenotypes': len(data),
                        'orphanet_phenotypes': len(orphanet_phenotypes),
                        'phenotypes': [
                            {
                                'description': p.get('description', ''),
                                'source': p.get('source', '')
                            }
                            for p in data[:10]
                        ]
                    }
        except:
            pass

        return {'gene': gene, 'total_phenotypes': 0, 'phenotypes': []}

    async def search_disease_genes(self, disease_term: str) -> Dict:
        """Busca genes asociados a un término de enfermedad."""
        session = await self._get_session()

        url = f"{self.BASE_URL}/phenotype/term/homo_sapiens/{disease_term}"

        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    genes = list(set(
                        g.get('gene_symbol', '')
                        for item in data
                        for g in item.get('genes', [])
                        if g.get('gene_symbol')
                    ))

                    return {
                        'disease_term': disease_term,
                        'genes_found': len(genes),
                        'genes': genes[:20]
                    }
        except:
            pass

        return {'disease_term': disease_term, 'genes_found': 0, 'genes': []}

    async def get_abstract_state(self, genes: List[str] = None, disease: str = None) -> GeneticState:
        """
        Genera estado abstracto basado en Orphanet/fenotipos.
        """
        genes = genes or ['CFTR', 'DMD', 'FBN1']  # Genes de enfermedades raras comunes

        total_phenotypes = 0
        orphanet_count = 0

        for gene in genes[:3]:
            data = await self.get_gene_phenotypes(gene)
            total_phenotypes += data['total_phenotypes']
            orphanet_count += data.get('orphanet_phenotypes', 0)

        # Más fenotipos = mayor asociación a enfermedad
        disease_association = min(1.0, total_phenotypes / 30)

        return GeneticState(
            expression_level=0.5,
            variability=0.6,  # Enfermedades raras = alta variabilidad
            tissue_specificity=0.7,  # Suelen afectar tejidos específicos
            regulatory_complexity=0.6,
            mutation_load=0.7,  # Alto para genes de enfermedad
            pathway_connectivity=0.5,
            conservation=0.8,  # Genes importantes = conservados
            disease_association=round(disease_association, 3),
            source='Orphanet',
            genes=genes
        )

    async def ping(self) -> bool:
        """Verifica conexión con Ensembl (para datos Orphanet)."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.BASE_URL}/info/ping") as resp:
                return resp.status == 200
        except:
            return False


# =============================================================================
# ENCODE CLIENT - Regulación Epigenética
# =============================================================================

class ENCODEClient:
    """
    Cliente para ENCODE Project.

    Acceso: Abierto
    Datos: Regulación génica, epigenética
    Uso: Inferencia regulatoria

    ⚠️ Solo lectura. NO genera diagnósticos.
    """

    BASE_URL = "https://www.encodeproject.org"

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={'Accept': 'application/json'}
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def search_experiments(self, query: str, assay_type: str = None) -> Dict:
        """Busca experimentos en ENCODE."""
        session = await self._get_session()

        params = {
            'type': 'Experiment',
            'searchTerm': query,
            'format': 'json',
            'limit': 10
        }

        if assay_type:
            params['assay_title'] = assay_type

        url = f"{self.BASE_URL}/search/"

        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    return {
                        'query': query,
                        'total': data.get('@graph', []),
                        'experiments': [
                            {
                                'accession': exp.get('accession', ''),
                                'assay': exp.get('assay_title', ''),
                                'target': exp.get('target', {}).get('label', '') if exp.get('target') else '',
                                'biosample': exp.get('biosample_ontology', {}).get('term_name', '') if exp.get('biosample_ontology') else ''
                            }
                            for exp in data.get('@graph', [])[:10]
                        ]
                    }
        except:
            pass

        return {'query': query, 'total': 0, 'experiments': []}

    async def get_gene_regulation(self, gene: str) -> Dict:
        """Obtiene datos de regulación para un gen."""
        # Buscar ChIP-seq y otros ensayos regulatorios
        chip_data = await self.search_experiments(gene, assay_type='ChIP-seq')

        return {
            'gene': gene,
            'chip_seq_experiments': len(chip_data.get('experiments', [])),
            'regulatory_data': chip_data.get('experiments', [])
        }

    async def get_abstract_state(self, genes: List[str] = None) -> GeneticState:
        """
        Genera estado abstracto basado en ENCODE.
        """
        genes = genes or ['MYC', 'TP53', 'CTCF']  # Genes regulatorios importantes

        total_experiments = 0

        for gene in genes[:3]:
            data = await self.get_gene_regulation(gene)
            total_experiments += data['chip_seq_experiments']

        # Más experimentos = mejor caracterizado regulatoriamente
        regulatory_complexity = min(1.0, total_experiments / 30)

        return GeneticState(
            expression_level=0.5,
            variability=0.4,
            tissue_specificity=0.5,
            regulatory_complexity=round(regulatory_complexity, 3),
            mutation_load=0.3,
            pathway_connectivity=0.7,  # Genes regulatorios = alta conectividad
            conservation=0.8,
            disease_association=0.4,
            source='ENCODE',
            genes=genes
        )

    async def ping(self) -> bool:
        """Verifica conexión con ENCODE."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.BASE_URL}/search/?type=Experiment&limit=1&format=json") as resp:
                return resp.status == 200
        except:
            return False


# =============================================================================
# ARRAYEXPRESS CLIENT
# =============================================================================

class ArrayExpressClient:
    """
    Cliente para ArrayExpress (EMBL-EBI).

    Acceso: Abierto
    Datos: Expresión génica (similar a GEO, Europa)
    Uso: Replicación y contraste de estudios

    ⚠️ Solo lectura. NO genera diagnósticos.
    """

    BASE_URL = "https://www.ebi.ac.uk/biostudies/api/v1"

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def search_studies(self, query: str, max_results: int = 10) -> Dict:
        """Busca estudios en ArrayExpress/BioStudies."""
        session = await self._get_session()

        params = {
            'query': query,
            'pageSize': max_results
        }

        url = f"{self.BASE_URL}/search"

        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    return {
                        'query': query,
                        'total_hits': data.get('totalHits', 0),
                        'studies': [
                            {
                                'accession': hit.get('accession', ''),
                                'title': hit.get('title', ''),
                                'type': hit.get('type', '')
                            }
                            for hit in data.get('hits', [])[:10]
                        ]
                    }
        except:
            pass

        return {'query': query, 'total_hits': 0, 'studies': []}

    async def get_abstract_state(self, genes: List[str] = None, disease: str = None) -> GeneticState:
        """
        Genera estado abstracto basado en ArrayExpress.
        """
        genes = genes or ['TP53', 'BRCA1', 'EGFR']

        query = ' OR '.join(genes[:3])
        if disease:
            query += f' AND {disease}'

        results = await self.search_studies(query)

        # Normalizar basado en hits
        expression_level = min(1.0, results['total_hits'] / 100)

        return GeneticState(
            expression_level=round(expression_level, 3),
            variability=0.5,
            tissue_specificity=0.5,
            regulatory_complexity=0.5,
            mutation_load=0.4 if disease else 0.2,
            pathway_connectivity=0.5,
            conservation=0.7,
            disease_association=0.5 if disease else 0.3,
            source='ArrayExpress',
            genes=genes
        )

    async def ping(self) -> bool:
        """Verifica conexión con ArrayExpress/BioStudies."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.BASE_URL}/search?query=test&pageSize=1") as resp:
                return resp.status == 200
        except:
            return False


# =============================================================================
# ALLEN BRAIN ATLAS CLIENT
# =============================================================================

class AllenBrainClient:
    """
    Cliente para Allen Brain Atlas.

    Acceso: Abierto
    Datos: Expresión génica cerebral
    Uso: Modelos teóricos de regulación neural

    ⚠️ Solo lectura. NO genera diagnósticos.
    """

    BASE_URL = "https://api.brain-map.org/api/v2"

    # Estructuras cerebrales principales
    STRUCTURES = [
        'Cerebral cortex', 'Hippocampus', 'Amygdala',
        'Thalamus', 'Hypothalamus', 'Cerebellum',
        'Basal ganglia', 'Brain stem'
    ]

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def search_genes(self, gene: str) -> Dict:
        """Busca un gen en Allen Brain Atlas."""
        session = await self._get_session()

        url = f"{self.BASE_URL}/data/Gene/query.json"
        params = {
            'criteria': f"[acronym$eq'{gene}']",
            'include': 'organism'
        }

        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    if data.get('success') and data.get('msg'):
                        genes = data['msg']
                        return {
                            'gene': gene,
                            'found': len(genes) > 0,
                            'data': [
                                {
                                    'id': g.get('id'),
                                    'name': g.get('name', ''),
                                    'acronym': g.get('acronym', ''),
                                    'organism': g.get('organism', {}).get('name', '')
                                }
                                for g in genes[:5]
                            ]
                        }
        except:
            pass

        return {'gene': gene, 'found': False, 'data': []}

    async def get_brain_structures(self) -> List[Dict]:
        """Obtiene lista de estructuras cerebrales."""
        session = await self._get_session()

        url = f"{self.BASE_URL}/data/Structure/query.json"
        params = {
            'criteria': "[ontology_id$eq1]",  # Human Brain Atlas
            'num_rows': 20
        }

        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    if data.get('success'):
                        return [
                            {
                                'id': s.get('id'),
                                'name': s.get('name', ''),
                                'acronym': s.get('acronym', '')
                            }
                            for s in data.get('msg', [])[:20]
                        ]
        except:
            pass

        return []

    async def get_abstract_state(self, genes: List[str] = None) -> GeneticState:
        """
        Genera estado abstracto basado en Allen Brain Atlas.
        """
        genes = genes or ['SLC17A7', 'GAD1', 'GFAP']  # Genes neuronales

        found_count = 0

        for gene in genes[:3]:
            result = await self.search_genes(gene)
            if result['found']:
                found_count += 1

        return GeneticState(
            expression_level=0.6,
            variability=0.5,
            tissue_specificity=0.9,  # Muy específico (cerebro)
            regulatory_complexity=0.7,  # Cerebro = alta complejidad
            mutation_load=0.3,
            pathway_connectivity=0.8,  # Redes neuronales
            conservation=0.8,
            disease_association=0.4,  # Neuro = muchas enfermedades
            source='AllenBrain',
            genes=genes
        )

    async def ping(self) -> bool:
        """Verifica conexión con Allen Brain Atlas."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.BASE_URL}/data/Gene/query.json?num_rows=1") as resp:
                return resp.status == 200
        except:
            return False


# =============================================================================
# UNIFIED CLIENT - Acceso unificado a todas las fuentes
# =============================================================================

class GenomicDataClient:
    """
    Cliente unificado para todas las fuentes genómicas.

    Combina datos de:
    - GEO (expresión)
    - 1000 Genomes (variantes)
    - GTEx (tejidos)
    - Orphanet (enfermedades raras)
    - ENCODE (regulación)
    - ArrayExpress (expresión Europa)
    - Allen Brain (cerebro)

    ⚠️ Solo lectura. NO genera diagnósticos.
    """

    def __init__(self):
        self.geo = GEOClient()
        self.genomes1k = ThousandGenomesClient()
        self.gtex = GTExClient()
        self.orphanet = OrphanetClient()
        self.encode = ENCODEClient()
        self.arrayexpress = ArrayExpressClient()
        self.allen = AllenBrainClient()

        self._sources = {
            'GEO': self.geo,
            '1000Genomes': self.genomes1k,
            'GTEx': self.gtex,
            'Orphanet': self.orphanet,
            'ENCODE': self.encode,
            'ArrayExpress': self.arrayexpress,
            'AllenBrain': self.allen
        }

    async def close(self):
        """Cierra todas las sesiones."""
        for client in self._sources.values():
            await client.close()

    async def ping_all(self) -> Dict[str, bool]:
        """Verifica conexión con todas las fuentes."""
        results = {}
        for name, client in self._sources.items():
            try:
                results[name] = await client.ping()
            except:
                results[name] = False
        return results

    async def get_unified_state(
        self,
        genes: List[str] = None,
        disease: str = None,
        sources: List[str] = None
    ) -> GeneticState:
        """
        Obtiene estado unificado combinando múltiples fuentes.

        Promedia las dimensiones de cada fuente consultada.
        """
        genes = genes or ['TP53', 'BRCA1', 'EGFR']
        sources = sources or list(self._sources.keys())

        states = []

        for source_name in sources:
            if source_name in self._sources:
                try:
                    client = self._sources[source_name]
                    if hasattr(client, 'get_abstract_state'):
                        if source_name in ['GEO', 'ArrayExpress', 'Orphanet']:
                            state = await client.get_abstract_state(genes, disease)
                        else:
                            state = await client.get_abstract_state(genes)
                        states.append(state)
                except Exception as e:
                    print(f"Error with {source_name}: {e}")
                    continue

        if not states:
            # Estado por defecto si todo falla
            return GeneticState(
                expression_level=0.5,
                variability=0.5,
                tissue_specificity=0.5,
                regulatory_complexity=0.5,
                mutation_load=0.3,
                pathway_connectivity=0.5,
                conservation=0.7,
                disease_association=0.3,
                source='default',
                genes=genes
            )

        # Promediar estados
        n = len(states)
        return GeneticState(
            expression_level=round(sum(s.expression_level for s in states) / n, 3),
            variability=round(sum(s.variability for s in states) / n, 3),
            tissue_specificity=round(sum(s.tissue_specificity for s in states) / n, 3),
            regulatory_complexity=round(sum(s.regulatory_complexity for s in states) / n, 3),
            mutation_load=round(sum(s.mutation_load for s in states) / n, 3),
            pathway_connectivity=round(sum(s.pathway_connectivity for s in states) / n, 3),
            conservation=round(sum(s.conservation for s in states) / n, 3),
            disease_association=round(sum(s.disease_association for s in states) / n, 3),
            source=','.join(s.source for s in states),
            genes=genes
        )

    def list_sources(self) -> List[Dict]:
        """Lista todas las fuentes disponibles."""
        return [
            {
                'id': 'GEO',
                'name': 'Gene Expression Omnibus',
                'type': 'expression',
                'access': 'open',
                'description': 'Expresión génica (RNA-seq, microarrays)'
            },
            {
                'id': '1000Genomes',
                'name': '1000 Genomes Project',
                'type': 'variants',
                'access': 'open',
                'description': 'Variación genética poblacional'
            },
            {
                'id': 'GTEx',
                'name': 'Genotype-Tissue Expression',
                'type': 'tissue_expression',
                'access': 'open',
                'description': 'Expresión por tejido, eQTLs'
            },
            {
                'id': 'Orphanet',
                'name': 'Orphanet',
                'type': 'rare_diseases',
                'access': 'open',
                'description': 'Genes asociados a enfermedades raras'
            },
            {
                'id': 'ENCODE',
                'name': 'ENCODE Project',
                'type': 'regulation',
                'access': 'open',
                'description': 'Regulación génica, epigenética'
            },
            {
                'id': 'ArrayExpress',
                'name': 'ArrayExpress',
                'type': 'expression',
                'access': 'open',
                'description': 'Expresión génica (Europa)'
            },
            {
                'id': 'AllenBrain',
                'name': 'Allen Brain Atlas',
                'type': 'brain_expression',
                'access': 'open',
                'description': 'Expresión génica cerebral'
            }
        ]


# =============================================================================
# SINGLETON
# =============================================================================

_genomic_client: Optional[GenomicDataClient] = None

def get_genomic_client() -> GenomicDataClient:
    global _genomic_client
    if _genomic_client is None:
        _genomic_client = GenomicDataClient()
    return _genomic_client
