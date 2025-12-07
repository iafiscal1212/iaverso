"""
API CONNECTOR - Conector Genérico para APIs
============================================

Conecta a APIs externas y convierte respuestas a estructuras matemáticas.
NO conoce la semántica de los endpoints.

NORMA DURA:
- Sin URLs hardcodeadas (se pasan como configuración)
- Procedencia documentada
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
import json
import hashlib
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from provenance import Provenance, ProvenanceType, get_provenance_logger


@dataclass
class APIResponse:
    """
    Respuesta de API convertida a estructura matemática.

    Solo números y estructura, sin semántica.
    """
    data: Any                       # Datos crudos
    arrays: Dict[str, np.ndarray]   # Arrays extraídos
    timestamp: str
    source_hash: str                # Hash del endpoint (no la URL)


class APIConnector:
    """
    Conector genérico para APIs.

    RESPONSABILIDADES:
    - Recibir configuración de endpoints (externamente)
    - Hacer requests y parsear respuestas
    - Convertir a arrays numéricos
    - Documentar procedencia

    NO HACE:
    - Definir qué APIs usar (lo define la humana)
    - Interpretar qué significan los datos
    - Hardcodear URLs
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Configuración externa con endpoints
                    La humana define qué APIs usar
        """
        self.config = config or {}
        self.logger = get_provenance_logger()
        self._request_cache: Dict[str, APIResponse] = {}

    def _hash_endpoint(self, endpoint_config: Dict) -> str:
        """Genera hash anónimo de un endpoint."""
        config_str = json.dumps(endpoint_config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]

    def _extract_arrays(
        self,
        data: Any,
        path: str = ""
    ) -> Dict[str, np.ndarray]:
        """
        Extrae arrays numéricos de estructura JSON.

        Recursivamente busca listas de números.
        """
        arrays = {}

        if isinstance(data, list):
            # Verificar si es lista de números
            if all(isinstance(x, (int, float)) for x in data):
                arrays[path or "root"] = np.array(data)
            # Si es lista de dicts, buscar recursivamente
            elif all(isinstance(x, dict) for x in data):
                # Transponer: convertir lista de dicts a dict de listas
                if data:
                    keys = data[0].keys()
                    for key in keys:
                        values = [item.get(key) for item in data]
                        if all(isinstance(v, (int, float)) for v in values if v is not None):
                            arr_path = f"{path}.{key}" if path else key
                            # Reemplazar None con NaN
                            values = [v if v is not None else np.nan for v in values]
                            arrays[arr_path] = np.array(values)

        elif isinstance(data, dict):
            for key, value in data.items():
                sub_path = f"{path}.{key}" if path else key
                sub_arrays = self._extract_arrays(value, sub_path)
                arrays.update(sub_arrays)

        return arrays

    def fetch_mock(
        self,
        endpoint_config: Dict,
        mock_data: Any
    ) -> APIResponse:
        """
        Simula fetch con datos mock.

        Para testing sin conexión real.
        """
        source_hash = self._hash_endpoint(endpoint_config)
        arrays = self._extract_arrays(mock_data)

        self.logger.log_from_data(
            value=f"mock_fetch[{len(arrays)} arrays]",
            source=f"endpoint_hash:{source_hash}",
            dataset="mock_api",
            context="APIConnector.fetch_mock"
        )

        return APIResponse(
            data=mock_data,
            arrays=arrays,
            timestamp=datetime.now().isoformat(),
            source_hash=source_hash
        )

    def fetch_from_file(
        self,
        path: Path,
        endpoint_label: str = ""
    ) -> APIResponse:
        """
        Carga datos de archivo JSON (simula respuesta API).

        Útil cuando la humana descarga datos manualmente.
        """
        with open(path, 'r') as f:
            data = json.load(f)

        source_hash = f"file_{hash(str(path)) % 10000:04d}"
        arrays = self._extract_arrays(data)

        self.logger.log_from_data(
            value=f"file_fetch[{len(arrays)} arrays]",
            source=f"file_hash:{source_hash}",
            dataset=endpoint_label or "json_file",
            context="APIConnector.fetch_from_file"
        )

        return APIResponse(
            data=data,
            arrays=arrays,
            timestamp=datetime.now().isoformat(),
            source_hash=source_hash
        )

    def fetch_live(
        self,
        url: str,
        headers: Optional[Dict] = None,
        params: Optional[Dict] = None,
        method: str = "GET"
    ) -> APIResponse:
        """
        Fetch real a una API.

        NOTA: La URL viene de configuración externa.
        Este código NO define qué URL usar.
        """
        try:
            import requests
        except ImportError:
            raise ImportError("Instalar requests: pip install requests")

        # Hash de la URL (no guardar URL completa por seguridad)
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]

        if method.upper() == "GET":
            response = requests.get(url, headers=headers, params=params)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, json=params)
        else:
            raise ValueError(f"Método {method} no soportado")

        response.raise_for_status()
        data = response.json()

        arrays = self._extract_arrays(data)

        self.logger.log_from_data(
            value=f"live_fetch[{len(arrays)} arrays]",
            source=f"url_hash:{url_hash}",
            dataset="live_api",
            context="APIConnector.fetch_live"
        )

        return APIResponse(
            data=data,
            arrays=arrays,
            timestamp=datetime.now().isoformat(),
            source_hash=url_hash
        )

    def to_timeseries(
        self,
        response: APIResponse,
        time_path: str,
        value_path: str
    ) -> Tuple[np.ndarray, np.ndarray, Provenance]:
        """
        Convierte respuesta a serie temporal.

        Args:
            response: Respuesta de API
            time_path: Path del array de timestamps
            value_path: Path del array de valores

        Returns:
            (t, values, provenance)
        """
        t = response.arrays.get(time_path)
        values = response.arrays.get(value_path)

        if t is None or values is None:
            available = list(response.arrays.keys())
            raise KeyError(
                f"Paths no encontrados. Disponibles: {available}"
            )

        # Asegurar mismo tamaño
        min_len = min(len(t), len(values))
        t = t[:min_len]
        values = values[:min_len]

        prov = self.logger.log_from_data(
            value=f"api_timeseries[{len(t)}]",
            source=f"endpoint:{response.source_hash}",
            dataset=f"{time_path}/{value_path}",
            context="APIConnector.to_timeseries"
        )

        return t, values, prov

    def list_available_arrays(
        self,
        response: APIResponse
    ) -> List[str]:
        """Lista arrays disponibles en una respuesta."""
        return list(response.arrays.keys())


# =============================================================================
# CONFIGURACIÓN DE ENDPOINTS (ejemplo - la humana define)
# =============================================================================

def create_endpoint_config(
    label: str,
    source_type: str = "file",
    path: Optional[str] = None,
    url: Optional[str] = None
) -> Dict:
    """
    Crea configuración de endpoint.

    La humana usa esto para definir qué datos cargar.
    El código NO conoce la semántica.

    Args:
        label: Etiqueta anónima (e.g., "source_01")
        source_type: "file" o "api"
        path: Ruta a archivo (si file)
        url: URL (si api) - la humana la proporciona

    Returns:
        Configuración de endpoint
    """
    return {
        'label': label,
        'type': source_type,
        'path': path,
        'url': url,
        'created': datetime.now().isoformat(),
    }
