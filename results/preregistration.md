# Pre-registro de Experimentos NEO↔EVA

Fecha: 2025-11-29T17:37:57.510005

## Hipótesis
H1: ρ(J) << 1 en todas las direcciones locales (atractor fuerte).
H2: Existe α* (cuantil ≥ p95 de τ) donde aparece régimen no-lineal.
H3: MI/TE inter-mundos supera p95 del nulo empírico en ≥ una ventana.
H4: La ablación de componentes (recall, gate, bus) degrada el score IWVI.

## Umbrales (endógenos)
- α: cuantiles de median(σ)/√T : {p25, p50, p75, p90, p95, p99}
- τ*: p95(τ) por dirección
- Significancia: percentil ≥ p95 del nulo, no α fijos
- k (MI/TE): floor(T^{1/3})
- B (permutaciones): floor(10√T)

## Métricas
- ΔRMSE: mejora vs baseline AR(p) con p por BIC
- ΔMDL: bits ahorrados vs baseline
- MI kNN: información mutua por k-vecinos
- TE: transfer entropy source→target
- Score Borda: rank sum normalizado

## Ablaciones planificadas
- no_recall_eva: desactivar memoria episódica de EVA
- no_gate: desactivar gate de exploración conjunta
- no_bus: desactivar comunicación inter-mundos
- no_pca: usar dirección aleatoria en lugar de v1

## Criterios de éxito
1. Aumentos selectivos de exploración cuando sorpresa/MI del otro sube
2. Cero constantes hardcodeadas en el código
3. Retorno a S alto tras episodios exploratorios (sin colapso)
4. p̂ < 0.05 para MI en al menos 20% de las evaluaciones