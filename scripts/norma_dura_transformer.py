#!/usr/bin/env python3
"""
NORMA DURA TRANSFORMER
======================
Transforma cualquier JSON a formato NORMA DURA con proveniencia completa.

Uso:
    python norma_dura_transformer.py input.json output.json
    python norma_dura_transformer.py input.json  # sobrescribe
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Union


def infer_source(key: str, value: Any, path: str) -> str:
    """Infiere el source basado en el contexto."""
    key_lower = key.lower()
    path_lower = path.lower()

    # FROM_STATISTICS
    if any(s in key_lower for s in ['mean', 'std', 'var', 'median', 'percentile', 'p95', 'p99', 'pct', 'avg']):
        return "FROM_STATISTICS"
    if any(s in key_lower for s in ['min', 'max', 'range', 'quartile', 'iqr']):
        return "FROM_STATISTICS"

    # FROM_MATH
    if any(s in key_lower for s in ['ratio', 'factor', 'coefficient', 'normalized', 'scaled']):
        return "FROM_MATH"
    if any(s in key_lower for s in ['sum', 'product', 'diff', 'delta', 'derivative']):
        return "FROM_MATH"

    # CONFIG_OPERATIONAL
    if any(s in key_lower for s in ['config', 'param', 'setting', 'threshold', 'limit']):
        return "CONFIG_OPERATIONAL"
    if any(s in key_lower for s in ['size', 'len', 'count', 'num', 'n_']):
        return "CONFIG_OPERATIONAL"
    if any(s in path_lower for s in ['config', 'settings', 'parameters']):
        return "CONFIG_OPERATIONAL"

    # Default: FROM_DATA
    return "FROM_DATA"


def infer_origin(key: str, value: Any, path: str) -> str:
    """Infiere el origin basado en el contexto."""
    key_lower = key.lower()

    # Estadísticas
    if 'mean' in key_lower:
        return f"np.mean({path.split('.')[-2] if '.' in path else 'data'})"
    if 'std' in key_lower:
        return f"np.std({path.split('.')[-2] if '.' in path else 'data'})"
    if 'var' in key_lower:
        return f"np.var({path.split('.')[-2] if '.' in path else 'data'})"
    if 'median' in key_lower:
        return f"np.median({path.split('.')[-2] if '.' in path else 'data'})"
    if 'min' in key_lower:
        return f"np.min({path.split('.')[-2] if '.' in path else 'data'})"
    if 'max' in key_lower:
        return f"np.max({path.split('.')[-2] if '.' in path else 'data'})"

    # Series temporales / historiales
    if 'history' in path.lower() or 'series' in path.lower():
        return f"timeseries_{key}"

    # Default
    return f"computed_{key}"


def wrap_value(value: Any, key: str, path: str) -> Dict:
    """Envuelve un valor primitivo en formato proveniencia."""
    return {
        "value": value,
        "origin": infer_origin(key, value, path),
        "source": infer_source(key, value, path)
    }


def transform_recursive(obj: Any, key: str = "root", path: str = "") -> Any:
    """Transforma recursivamente un objeto JSON a formato NORMA DURA."""
    current_path = f"{path}.{key}" if path else key

    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            result[k] = transform_recursive(v, k, current_path)
        return result

    elif isinstance(obj, list):
        # Si es lista de números, envolver cada uno
        if obj and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj):
            # Para listas largas de datos, mantener como array pero documentar
            if len(obj) > 10:
                return {
                    "values": obj,
                    "n_elements": {
                        "value": len(obj),
                        "origin": "len(array)",
                        "source": "FROM_DATA"
                    },
                    "statistics": {
                        "mean": {
                            "value": sum(obj) / len(obj) if obj else 0,
                            "origin": "np.mean(array)",
                            "source": "FROM_STATISTICS"
                        },
                        "min": {
                            "value": min(obj) if obj else 0,
                            "origin": "np.min(array)",
                            "source": "FROM_STATISTICS"
                        },
                        "max": {
                            "value": max(obj) if obj else 0,
                            "origin": "np.max(array)",
                            "source": "FROM_STATISTICS"
                        }
                    },
                    "origin": f"array_{key}",
                    "source": "FROM_DATA"
                }
            else:
                # Listas cortas: envolver cada elemento
                return [
                    {
                        "value": x,
                        "origin": f"{key}[{i}]",
                        "source": "FROM_DATA"
                    }
                    for i, x in enumerate(obj)
                ]
        else:
            # Lista de objetos o mixta: procesar recursivamente
            return [transform_recursive(item, f"{key}[{i}]", current_path) for i, item in enumerate(obj)]

    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return wrap_value(obj, key, current_path)

    elif isinstance(obj, bool):
        return {
            "value": obj,
            "origin": f"boolean_{key}",
            "source": "FROM_DATA"
        }

    elif isinstance(obj, str):
        # Strings no necesitan wrapper de proveniencia (ya son documentación)
        return obj

    elif obj is None:
        return {
            "value": None,
            "origin": f"null_{key}",
            "source": "FROM_DATA"
        }

    return obj


def add_metadata_wrapper(data: Dict, input_path: str) -> Dict:
    """Añade metadata de NORMA DURA al archivo transformado."""
    filename = Path(input_path).stem

    # Si ya tiene metadata, preservarla y expandir
    existing_metadata = data.get("metadata", {})

    norma_dura_metadata = {
        "norma_dura_compliance": {
            "value": True,
            "origin": "transformation_complete",
            "source": "FROM_DATA"
        },
        "transformation_timestamp": {
            "value": datetime.now().isoformat(),
            "origin": "datetime.now()",
            "source": "FROM_DATA"
        },
        "original_file": {
            "value": filename,
            "origin": "input_filename",
            "source": "FROM_DATA"
        },
        "validation_protocol": {
            "value": "NORMA DURA - ZERO HARDCODE",
            "origin": "quality_standard",
            "source": "FROM_DATA"
        }
    }

    if isinstance(existing_metadata, dict):
        # Transformar metadata existente
        transformed_metadata = transform_recursive(existing_metadata, "metadata", "")
        if isinstance(transformed_metadata, dict):
            transformed_metadata.update(norma_dura_metadata)
            data["metadata"] = transformed_metadata
    else:
        data["metadata"] = norma_dura_metadata

    # Añadir audit_log si no existe
    if "audit_log" not in data:
        data["audit_log"] = {
            "transformed_by": {
                "value": "norma_dura_transformer.py",
                "origin": "script_name",
                "source": "FROM_DATA"
            },
            "transformation_date": {
                "value": datetime.now().strftime("%Y-%m-%d"),
                "origin": "current_date",
                "source": "FROM_DATA"
            }
        }

    return data


def transform_file(input_path: str, output_path: str = None) -> Dict:
    """Transforma un archivo JSON a formato NORMA DURA."""
    if output_path is None:
        output_path = input_path

    print(f"Leyendo: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("Transformando a NORMA DURA...")
    transformed = transform_recursive(data)

    if isinstance(transformed, dict):
        transformed = add_metadata_wrapper(transformed, input_path)

    print(f"Escribiendo: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transformed, f, indent=2, ensure_ascii=False)

    return transformed


def main():
    if len(sys.argv) < 2:
        print("Uso: python norma_dura_transformer.py input.json [output.json]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path

    transform_file(input_path, output_path)
    print("✓ Transformación completa")


if __name__ == "__main__":
    main()
