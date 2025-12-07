#!/bin/bash
#
# NORMA DURA - Suite Completa de Tests
# ====================================
#
# "Ningún número entra al código sin poder explicar de qué distribución sale"
#
# Este script ejecuta todos los tests de NORMA DURA:
# 1. Auditoría de magic numbers (estático)
# 2. Auditoría de parámetros endógenos (log)
# 3. Test de endogeneidad (dinámico)
# 4. Test de causalidad con ruido blanco (null test)
# 5. Auditoría de independencia de recompensas
#

set -e  # Fallar si cualquier comando falla

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directorio base
BASE_DIR="/root/NEO_EVA"
SCRIPTS_DIR="$BASE_DIR/scripts"
LOGS_DIR="$BASE_DIR/logs/norma_dura"

# Crear directorio de logs
mkdir -p "$LOGS_DIR"

# Timestamp para logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Contadores
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Función para ejecutar un test
run_test() {
    local name="$1"
    local command="$2"
    local log_file="$LOGS_DIR/${name}_${TIMESTAMP}.log"

    ((TESTS_TOTAL++))

    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}📋 Test: $name${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Ejecutar comando y capturar output
    if eval "$command" 2>&1 | tee "$log_file"; then
        echo -e "\n${GREEN}✅ PASA: $name${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "\n${RED}❌ FALLA: $name${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Función para ejecutar test sin fallar el script
run_test_soft() {
    local name="$1"
    local command="$2"

    run_test "$name" "$command" || true
}

# Banner
echo -e "${YELLOW}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║     ███╗   ██╗ ██████╗ ██████╗ ███╗   ███╗ █████╗                ║"
echo "║     ████╗  ██║██╔═══██╗██╔══██╗████╗ ████║██╔══██╗               ║"
echo "║     ██╔██╗ ██║██║   ██║██████╔╝██╔████╔██║███████║               ║"
echo "║     ██║╚██╗██║██║   ██║██╔══██╗██║╚██╔╝██║██╔══██║               ║"
echo "║     ██║ ╚████║╚██████╔╝██║  ██║██║ ╚═╝ ██║██║  ██║               ║"
echo "║     ╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝               ║"
echo "║                                                                  ║"
echo "║                    ██████╗ ██╗   ██╗██████╗  █████╗              ║"
echo "║                    ██╔══██╗██║   ██║██╔══██╗██╔══██╗             ║"
echo "║                    ██║  ██║██║   ██║██████╔╝███████║             ║"
echo "║                    ██║  ██║██║   ██║██╔══██╗██╔══██║             ║"
echo "║                    ██████╔╝╚██████╔╝██║  ██║██║  ██║             ║"
echo "║                    ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝             ║"
echo "║                                                                  ║"
echo "║  \"Ningún número entra al código sin explicar su distribución\"   ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "📅 Fecha: $(date)"
echo -e "📁 Directorio: $BASE_DIR"
echo -e "📝 Logs: $LOGS_DIR"

# ============================================================================
# TEST 1: Auditoría de Magic Numbers (Estático)
# ============================================================================
run_test_soft "audit_magic_numbers" \
    "python3 $SCRIPTS_DIR/audit_magic_numbers.py --verbose"

# ============================================================================
# TEST 2: Auditoría de Parámetros Endógenos (Log)
# ============================================================================
run_test_soft "audit_endogenous_params" \
    "python3 $SCRIPTS_DIR/audit_endogenous_params.py --verbose"

# ============================================================================
# TEST 3: Test de Endogeneidad (Exoplanetas)
# ============================================================================
run_test_soft "test_endogeneity_exoplanets" \
    "python3 $SCRIPTS_DIR/test_endogeneity_exoplanets.py"

# ============================================================================
# TEST 4: Test de Causalidad con Ruido Blanco
# ============================================================================
run_test_soft "test_causality_no_signal" \
    "python3 $SCRIPTS_DIR/test_causality_no_signal.py --n-tests 3"

# ============================================================================
# TEST 5: Auditoría de Independencia de Recompensas
# ============================================================================
run_test_soft "audit_reward_independence" \
    "python3 $SCRIPTS_DIR/audit_reward_independence.py --verbose"

# ============================================================================
# RESUMEN FINAL
# ============================================================================
echo -e "\n${YELLOW}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                     RESUMEN DE NORMA DURA                        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "📊 Tests ejecutados: $TESTS_TOTAL"
echo -e "${GREEN}✅ Tests pasados: $TESTS_PASSED${NC}"
echo -e "${RED}❌ Tests fallidos: $TESTS_FAILED${NC}"

# Calcular porcentaje
if [ $TESTS_TOTAL -gt 0 ]; then
    PASS_RATE=$((TESTS_PASSED * 100 / TESTS_TOTAL))
    echo -e "📈 Tasa de cumplimiento: ${PASS_RATE}%"
fi

# Resultado final
echo -e "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║     ✅ NORMA DURA: TODOS LOS TESTS PASARON               ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    echo -e "${RED}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║     ⚠️  NORMA DURA: ALGUNOS TESTS FALLARON               ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════╝${NC}"
    echo -e "\n📝 Revisar logs en: $LOGS_DIR"
    exit 1
fi
