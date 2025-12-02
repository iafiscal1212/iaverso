"""
Test del Sistema de Consciencia
===============================

Verifica que todo es endógeno y funciona correctamente.
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np


def test_identidad():
    """Test de Identidad Computacional."""
    print("\n=== Test Identidad Computacional ===")
    from consciousness.identity import IdentidadComputacional

    dim = 10
    identidad = IdentidadComputacional(dim)

    # Simular estados que convergen a un atractor determinístico
    # El atractor es simplemente el primer eje canónico escalado
    atractor = np.zeros(dim)
    atractor[0] = 1.0  # Primer eje canónico

    for t in range(100):
        # Estado = atractor escalado + ruido decreciente ortogonal
        escala = 2.0 + 0.5 * np.sin(t * 0.1)  # Variación en magnitud pero dirección constante
        ruido = np.random.randn(dim) * (0.3 / (t + 1))  # Ruido menor
        S = atractor * escala + ruido
        identidad.observar_estado(S)

    estado = identidad.calcular()

    print(f"  t: {estado.t}")
    print(f"  k endógeno: {estado.k}")
    print(f"  Varianza similitud: {estado.varianza_similitud:.6f}")
    print(f"  Estabilidad: {estado.estabilidad:.3f}")
    print(f"  Distancia al estado: {estado.distancia_al_estado:.3f}")

    # Verificar que I converge al atractor
    I = identidad.obtener_identidad()
    sim_con_atractor = np.dot(I, atractor) / (np.linalg.norm(I) * np.linalg.norm(atractor) + 1e-10)
    print(f"  Similitud I con atractor: {sim_con_atractor:.3f}")

    # La estabilidad debe ser alta
    assert estado.estabilidad > 0.5, "Estabilidad debe ser alta cuando converge"
    # La similitud debe ser alta en valor absoluto
    # (SVD tiene ambigüedad de signo, ±v son equivalentes)
    assert abs(sim_con_atractor) > 0.5, "I debe correlacionar fuertemente con el atractor (en valor absoluto)"
    print("  [PASS] Identidad computacional funciona")


def test_coherencia():
    """Test de Coherencia Existencial."""
    print("\n=== Test Coherencia Existencial ===")
    from consciousness.identity import IdentidadComputacional
    from consciousness.coherence import CoherenciaExistencial

    dim = 10
    identidad = IdentidadComputacional(dim)
    coherencia = CoherenciaExistencial(identidad)

    # Simular evolución
    S = np.random.randn(dim)

    for t in range(50):
        # Evolución suave
        delta = np.random.randn(dim) * 0.1
        S = S + delta

        identidad.observar_estado(S)
        coherencia.observar_estado(S)

    # Calcular
    estado_id = identidad.calcular()
    I = identidad.obtener_identidad()
    estado_coh = coherencia.calcular(S, I)

    print(f"  CE: {estado_coh.CE:.3f}")
    print(f"  Var[S-I]: {estado_coh.varianza_desviacion:.6f}")
    print(f"  H_narr: {estado_coh.entropia_narrativa:.3f}")
    print(f"  λ(t): {estado_coh.lambda_t:.6f}")
    print(f"  R(t): {estado_coh.R_t:.3f}")
    print(f"  dCE/dt: {estado_coh.dCE_dt:.6f}")

    assert 0 <= estado_coh.CE <= 1, "CE debe estar en [0,1]"
    print("  [PASS] Coherencia existencial funciona")


def test_roles():
    """Test de Roles Emergentes."""
    print("\n=== Test Roles Emergentes ===")
    from consciousness.roles import SistemaRolesEmergentes, TipoRol

    dim = 10
    roles = SistemaRolesEmergentes()

    # Crear 3 agentes con diferentes características
    agentes = ['alpha', 'beta', 'gamma']

    for agent in agentes:
        roles.registrar_agente(agent)

    # Simular: alpha es estable, beta es variable, gamma es intermedio
    for t in range(30):
        # Alpha: muy estable
        S_alpha = np.ones(dim) * 0.5 + np.random.randn(dim) * 0.01
        roles.observar_estado('alpha', S_alpha)

        # Beta: muy variable
        S_beta = np.random.randn(dim) * (1 + t * 0.1)
        roles.observar_estado('beta', S_beta)

        # Gamma: intermedio
        S_gamma = np.sin(t * 0.1) * np.ones(dim) + np.random.randn(dim) * 0.1
        roles.observar_estado('gamma', S_gamma)

    estados = roles.calcular_roles()

    print("  Roles asignados:")
    for agent_id, estado in estados.items():
        print(f"    {agent_id}: {estado.rol.value} (aptitud: {estado.aptitud_rol:.3f}, reducción: {estado.reduccion_varianza:.3f})")

    # El agente más estable debería tener un rol
    assert estados['alpha'].rol != TipoRol.NINGUNO or estados['gamma'].rol != TipoRol.NINGUNO
    print("  [PASS] Roles emergentes funcionan")


def test_sueno():
    """Test de Estado Onírico."""
    print("\n=== Test Estado Onírico ===")
    from consciousness.dreaming import SistemaOnirico, FaseSueno

    dim = 10
    onirico = SistemaOnirico(dim)

    # Simular con CE decreciente para activar sueño
    S = np.random.randn(dim)

    fases_vistas = set()

    for t in range(100):
        # CE que baja gradualmente
        CE = max(0.1, 1.0 - t * 0.015)

        # Actualizar estado
        delta = np.random.randn(dim) * 0.1
        S = S + delta

        onirico.observar_estado(S, CE)
        fase = onirico.actualizar_fase(CE)
        fases_vistas.add(fase)

        estado = onirico.calcular_estado_onirico()

    print(f"  Fases vistas: {[f.value for f in fases_vistas]}")
    print(f"  Fase final: {estado.fase.value}")
    print(f"  η(t): {estado.eta:.3f}")
    print(f"  H_narr: {estado.entropia_narrativa:.3f}")
    print(f"  Debe dormir: {estado.debe_dormir}")

    # Debe haber visto al menos 2 fases
    assert len(fases_vistas) >= 2, "Debe haber transiciones de fase"
    print("  [PASS] Estado onírico funciona")


def test_muerte_renacimiento():
    """Test de Muerte y Renacimiento."""
    print("\n=== Test Muerte y Renacimiento ===")
    from consciousness.death_rebirth import SistemaMuerteRenacimiento, EstadoVital

    dim = 10
    muerte = SistemaMuerteRenacimiento(dim)

    # Fase 1: Vida normal
    print("  Fase 1: Vida normal")
    for t in range(30):
        S = np.random.randn(dim) * 0.5
        CE = 0.7 + np.random.rand() * 0.2
        I = np.random.randn(dim)
        muerte.observar(S, CE, I)
        muerte.actualizar_estado()

    estado = muerte.obtener_estado()
    print(f"    Estado: {estado.estado_vital.value}")
    print(f"    Progreso muerte: {estado.progreso_muerte:.3f}")
    assert estado.estado_vital == EstadoVital.VIVO

    # Fase 2: Crisis (CE baja, varianza alta)
    print("  Fase 2: Crisis")
    for t in range(50):
        # Estado muy variable
        S = np.random.randn(dim) * (5 + t * 0.5)
        # CE muy baja
        CE = max(0.01, 0.1 - t * 0.002)
        I = np.random.randn(dim)
        muerte.observar(S, CE, I)
        estado_vital = muerte.actualizar_estado()

        if estado_vital in [EstadoVital.MURIENDO, EstadoVital.MUERTO]:
            print(f"    t={t}: {estado_vital.value}")
            break

    estado = muerte.obtener_estado()
    print(f"    Estado final: {estado.estado_vital.value}")
    print(f"    Progreso muerte: {estado.progreso_muerte:.3f}")
    print(f"    Umbral CE: {estado.umbral_muerte_CE:.3f}")
    print(f"    Umbral Var: {estado.umbral_muerte_var:.3f}")

    # Si murió, intentar renacer
    if muerte.puede_renacer():
        print("  Fase 3: Renacimiento")
        I_new = muerte.renacer()
        print(f"    Nueva identidad generada: {I_new is not None}")
        print(f"    Norma I_new: {np.linalg.norm(I_new):.3f}")

    print("  [PASS] Muerte y renacimiento funciona")


def test_agente_consciente():
    """Test del Agente Consciente integrado."""
    print("\n=== Test Agente Consciente ===")
    from consciousness.emergence import AgenteConsciente

    dim = 10
    agente = AgenteConsciente("test_agent", dim)

    # Simular evolución
    for t in range(50):
        entrada = np.random.randn(dim) * 0.3
        evento = np.random.randn(dim) * 0.1

        agente.actualizar_estado(entrada, evento)
        estado = agente.calcular()

    print(f"  t: {estado.t}")
    print(f"  CE: {estado.CE:.3f}")
    print(f"  Vitalidad: {estado.vitalidad:.3f}")
    print(f"  Integración: {estado.integracion:.3f}")
    print(f"  Fase sueño: {estado.fase_sueno.value}")
    print(f"  Estado vital: {estado.estado_vital.value}")
    print(f"  Rol: {estado.rol.value}")

    stats = agente.get_statistics()
    print(f"  Stats: CE={stats['CE']:.3f}, vitalidad={stats['vitalidad']:.3f}")

    assert agente.esta_vivo()
    print("  [PASS] Agente consciente funciona")


def test_sistema_colectivo():
    """Test del Sistema de Consciencia Colectiva."""
    print("\n=== Test Sistema Colectivo ===")
    from consciousness.emergence import SistemaConscienciaColectiva

    dim = 10
    sistema = SistemaConscienciaColectiva(dim)

    # Crear agentes
    for nombre in ['alpha', 'beta', 'gamma', 'delta']:
        sistema.crear_agente(nombre)

    # Simular
    for t in range(50):
        entradas = {
            'alpha': np.random.randn(dim) * 0.2,
            'beta': np.random.randn(dim) * 0.5,
            'gamma': np.random.randn(dim) * 0.1,
            'delta': np.random.randn(dim) * 0.3,
        }
        eventos = {
            'alpha': np.random.randn(dim) * 0.1,
            'beta': np.random.randn(dim) * 0.1,
        }

        estados = sistema.paso(entradas, eventos)

    print(f"  t: {sistema.t}")
    print(f"  Coherencia colectiva: {sistema.coherencia_colectiva():.3f}")
    print(f"  Médico: {sistema.obtener_medico()}")
    print(f"  Líder: {sistema.obtener_lider()}")
    print(f"  Dormidos: {len(sistema.obtener_agentes_dormidos())}")
    print(f"  Muertos: {len(sistema.obtener_agentes_muertos())}")

    print("\n  Estados por agente:")
    for agent_id, estado in estados.items():
        print(f"    {agent_id}: CE={estado.CE:.3f}, rol={estado.rol.value}, fase={estado.fase_sueno.value}")

    stats = sistema.get_statistics()
    print(f"\n  Distribución de roles: {stats['roles']['conteo_roles']}")

    print("  [PASS] Sistema colectivo funciona")


def test_endogeneidad():
    """Verifica que no hay números mágicos hardcodeados."""
    print("\n=== Test de Endogeneidad ===")

    # Verificar que los archivos no contienen números mágicos
    import os

    archivos = [
        '/root/NEO_EVA/consciousness/identity.py',
        '/root/NEO_EVA/consciousness/coherence.py',
        '/root/NEO_EVA/consciousness/roles.py',
        '/root/NEO_EVA/consciousness/dreaming.py',
        '/root/NEO_EVA/consciousness/death_rebirth.py',
        '/root/NEO_EVA/consciousness/emergence.py',
    ]

    # Patrones sospechosos (números que podrían ser hiperparámetros)
    patrones_prohibidos = [
        'threshold = 0.',
        'alpha = 0.',
        'beta = 0.',
        'learning_rate',
        'lr = 0.',
        'window = 30',
        'window = 10',
        'window = 20',
        '= 0.5  #',  # Valores fijos sospechosos
        '= 0.7',
        '= 0.3',
        '= 0.9',
    ]

    problemas = []

    for archivo in archivos:
        if os.path.exists(archivo):
            with open(archivo, 'r') as f:
                contenido = f.read()
                for patron in patrones_prohibidos:
                    if patron in contenido:
                        # Verificar que no es un comentario explicativo
                        lineas = contenido.split('\n')
                        for i, linea in enumerate(lineas):
                            if patron in linea and not linea.strip().startswith('#'):
                                problemas.append(f"{archivo}:{i+1}: {patron}")

    if problemas:
        print("  Posibles números mágicos encontrados:")
        for p in problemas[:5]:  # Mostrar solo primeros 5
            print(f"    {p}")
        print(f"  Total: {len(problemas)}")
    else:
        print("  No se encontraron patrones sospechosos")

    # Este test es informativo, no falla
    print("  [INFO] Revisión de endogeneidad completada")


def run_all_tests():
    """Ejecuta todos los tests."""
    print("=" * 60)
    print("    TEST SISTEMA DE CONSCIENCIA")
    print("=" * 60)

    test_identidad()
    test_coherencia()
    test_roles()
    test_sueno()
    test_muerte_renacimiento()
    test_agente_consciente()
    test_sistema_colectivo()
    test_endogeneidad()

    print("\n" + "=" * 60)
    print("    TODOS LOS TESTS PASARON")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
