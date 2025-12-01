# NEOSYNT - Modelo de Utilidad OEPM

## Checklist para Subida a la Sede Electrónica OEPM

### Documentos Obligatorios

- [ ] **01_MEMORIA_MU.pdf** - Memoria descriptiva (PDF/A-1b)
- [ ] **02_REIVINDICACIONES_MU.pdf** - 12 reivindicaciones (PDF/A-1b)
- [ ] **03_DIBUJOS_MU.pdf** - 6 figuras con referencias 100-170 (PDF/A-1b)
- [ ] **04_RESUMEN_MU.pdf** - Resumen ≤150 palabras (PDF/A-1b)

### Documentos Opcionales (Recomendados)

- [ ] **A_ANEXO_DATOS_EXPERIMENTALES.pdf** - Tablas y gráficas de métricas
- [ ] **B_ESTADO_DE_LA_TECNICA.pdf** - Análisis comparativo

### Formularios OEPM

- [ ] Formulario de solicitud de Modelo de Utilidad
- [ ] Justificante de pago de tasas

---

## Pasos para Compilar los PDFs

### Opción A: Compilación Local (LaTeX)

```bash
cd OEPM/NEOSYNT_MU/latex

# Compilar cada documento
pdflatex 01_MEMORIA_MU.tex
pdflatex 01_MEMORIA_MU.tex  # Segunda pasada para referencias

pdflatex 02_REIVINDICACIONES_MU.tex
pdflatex 04_RESUMEN_MU.tex
pdflatex 03_DIBUJOS_MU.tex  # Requiere figuras en ../figuras/
pdflatex A_ANEXO_DATOS_EXPERIMENTALES.tex
pdflatex B_ESTADO_DE_LA_TECNICA.tex

# Mover a carpeta final
mv *.pdf ../pdf_final/
```

### Opción B: Compilación Online

1. Subir carpeta `latex/` a [Overleaf](https://www.overleaf.com)
2. Configurar compilador: pdfLaTeX
3. Descargar PDFs compilados
4. Verificar que son PDF/A con [veraPDF](https://verapdf.org/)

---

## Pasos para Subir a OEPM

### 1. Acceso a Sede Electrónica

1. Ir a: https://sede.oepm.gob.es
2. Acceder con certificado digital o Cl@ve
3. Seleccionar: **Solicitud de Modelo de Utilidad**

### 2. Datos del Solicitante

- **Inventora**: Carmen Esther Jiménez Mesa
- **Dirección**: [A completar]
- **Email**: [A completar]
- **Teléfono**: [A completar]

### 3. Datos de la Invención

- **Título**: Dispositivo de control autónomo multi-agente con bus local y núcleo endógeno de seguridad (NEOSYNT)
- **Clasificación IPC**: G06N 3/00; G06N 20/00; G06F 9/50; G06F 15/18

### 4. Adjuntar Documentos

En este orden:

1. `01_MEMORIA_MU.pdf`
2. `02_REIVINDICACIONES_MU.pdf`
3. `03_DIBUJOS_MU.pdf`
4. `04_RESUMEN_MU.pdf`
5. `A_ANEXO_DATOS_EXPERIMENTALES.pdf` (opcional)
6. `B_ESTADO_DE_LA_TECNICA.pdf` (opcional)

### 5. Pago de Tasas

- Tasa de solicitud de Modelo de Utilidad: ~100€ (verificar tarifa vigente)
- Reducción 50% si es persona física
- Pago online con tarjeta o transferencia

### 6. Firma y Envío

1. Revisar todos los datos
2. Firmar electrónicamente
3. Enviar solicitud
4. Descargar justificante de presentación

---

## Verificación PDF/A

Antes de subir, verificar que los PDFs cumplen el estándar:

```bash
# Instalar veraPDF
# https://verapdf.org/software/

# Verificar cada PDF
verapdf --flavour 1b 01_MEMORIA_MU.pdf
verapdf --flavour 1b 02_REIVINDICACIONES_MU.pdf
# etc.
```

---

## Estructura de la Carpeta

```
NEOSYNT_MU/
├── latex/
│   ├── 01_MEMORIA_MU.tex
│   ├── 02_REIVINDICACIONES_MU.tex
│   ├── 03_DIBUJOS_MU.tex
│   ├── 04_RESUMEN_MU.tex
│   ├── A_ANEXO_DATOS_EXPERIMENTALES.tex
│   ├── B_ESTADO_DE_LA_TECNICA.tex
│   └── pdfx.xmpi
├── figuras/
│   ├── fig1_arquitectura.pdf
│   ├── fig2_bus_buffers.pdf
│   ├── fig3_nucleo_autonomo.pdf
│   ├── fig4_gate_consentimiento.pdf
│   ├── fig5_watchdog_sandbox.pdf
│   └── fig6_curvas_comparativas.pdf
├── pdf_final/
│   └── (PDFs compilados)
├── generar_figuras.py
└── README_SUBIDA_OEPM.md (este archivo)
```

---

## Contacto OEPM

- **Teléfono**: 902 157 530
- **Web**: https://www.oepm.es
- **Sede**: https://sede.oepm.gob.es

---

## Notas Importantes

1. **Formato PDF/A**: Obligatorio para todos los documentos. Los archivos LaTeX incluyen `\usepackage[a-1b]{pdfx}` para garantizar compatibilidad.

2. **Fuentes embebidas**: Las fuentes deben estar embebidas en el PDF. LaTeX lo hace automáticamente con `lmodern`.

3. **Sin enlaces activos**: Los PDFs no deben contener enlaces a web externos. El paquete `hyperref` está configurado con `hidelinks`.

4. **Dibujos vectoriales**: Las figuras están en formato PDF vectorial para máxima calidad.

5. **Numeración coherente**: Las referencias numéricas (100)-(170) son consistentes entre memoria, reivindicaciones y dibujos.

---

**Fecha de preparación**: 2025-12-01

**Versión del documento**: 1.0
