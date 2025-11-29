#!/bin/bash
# ==============================================================================
# run_endogenous.sh - Ejecución completa del sistema NEO↔EVA v2 endógeno
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$ROOT_DIR/results"
REPRO_DIR="$ROOT_DIR/repro"

echo "======================================================================"
echo "NEO↔EVA v2.0-endogenous - Ejecución Completa"
echo "======================================================================"
echo "Directorio raíz: $ROOT_DIR"
echo "Fecha: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# 1. Auditoría de endogeneidad
echo "[1/5] Ejecutando auditoría de endogeneidad..."
cd "$ROOT_DIR"
python3 tools/endogeneity_auditor.py --file tools/phase6_coupled_system_v2.py --full
echo ""

# 2. Corrida principal (500 ciclos)
echo "[2/5] Ejecutando corrida principal (500 ciclos)..."
python3 tools/phase6_coupled_system_v2.py --cycles 500 --output "$RESULTS_DIR/phase6_v2"
echo ""

# 3. Ablación (sin acoplamiento)
echo "[3/5] Ejecutando ablación (sin acoplamiento)..."
python3 tools/phase6_coupled_system_v2.py --cycles 500 --no-coupling --output "$RESULTS_DIR/phase6_v2_ablation"
echo ""

# 4. Corrida larga (2000 ciclos para estabilidad)
echo "[4/5] Ejecutando corrida larga (2000 ciclos)..."
python3 tools/phase6_coupled_system_v2.py --cycles 2000 --output "$RESULTS_DIR/phase6_v2_long"
echo ""

# 5. Generar hashes de resultados
echo "[5/5] Generando hashes de verificación..."
sha256sum "$RESULTS_DIR"/phase6_v2*.json > "$REPRO_DIR/results_hashes.sha256"
echo "Hashes guardados en: $REPRO_DIR/results_hashes.sha256"
echo ""

echo "======================================================================"
echo "Ejecución completa finalizada"
echo "======================================================================"
echo ""
echo "Archivos generados:"
ls -la "$RESULTS_DIR"/phase6_v2*.json
echo ""
echo "Para verificar integridad:"
echo "  cd $ROOT_DIR && sha256sum -c repro/results_hashes.sha256"
