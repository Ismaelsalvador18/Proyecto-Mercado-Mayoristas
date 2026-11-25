# Plan de desarrollo (v3, ejecutado)

**Estudio:** Integración espacial de los mercados mayoristas de alimentos en el Perú,
2012–2023: traspaso de precios y valor predictivo de la referencia de Lima.

**Estado:** pipeline implementado y ejecutado de inicio a fin.

---

## 1. Pregunta e hipótesis

**Pregunta:** ¿cuánto y con qué rezago responde el precio mayorista departamental al de Lima, y qué
valor predictivo tiene esa relación?

**Sub-preguntas:**
1. ¿Qué tan integrados están los mercados?
2. ¿Cuál es el traspaso de corto plazo y la vida media?
3. ¿El traspaso es simétrico? (exploratorio)
4. ¿La relación tiene valor predictivo fuera de muestra?

**Hipótesis:**
- H1: traspaso incompleto (β < 1) y heterogéneo por producto.
- H2: vida media mayor en procesados (aceite, arroz) que en frescos (cebolla, papa).
- H3: la integración cae con la distancia; la zona aporta poco por sí sola.
- H4: el valor predictivo se concentra en frescos con traspaso alto.

---

## 2. Datos

| Elemento | Valor |
|---|---|
| Fuente | SISAP–MIDAGRI, módulo Precios de ciudades |
| Extracción | 24 consultas, 24 departamentos con datos |
| Crudo | 102 808 filas, 9 categorías armonizadas, 1997–2026 |
| Panel | 43 921 filas departamento-producto-mes |
| Limpio | 43 855 filas (66 atípicos, 0,15 %) |
| Ventana densa | 2012–2023 (85,7 % de las celdas) |
| Núcleo | papa, cebolla, zanahoria, arroz, aceite, huevos |

Detalle en `docs/diccionario_datos.md`.

---

## 3. Estructura del pipeline

```
scripts/
├── config.py                     rutas, productos, ventanas, hiperparámetros
├── catalogo.py                   catálogo y validación de 24 departamentos
├── modelos.py                    features, modelos y métricas compartidas
├── pipeline/
│   ├── 01_descargar_sisap.py
│   ├── 02_construir_panel.py
│   ├── 03_limpiar_panel.py
│   ├── 04_ecm.py
│   ├── 05_torneo_modelos.py
│   └── 06_robustez.py
├── analisis/
│   ├── 01.1_auditoria_fuente.py
│   ├── 02.1_auditoria_panel.py
│   ├── 03.1_verificacion_limpieza.py
│   ├── 03.2_descripcion_datos.py
│   ├── 03.3_estacionariedad.py
│   ├── 03.4_seleccion_rezagos.py
│   ├── 03.5_estructura_mercados.py
│   ├── 04.1_diagnostico_ecm.py
│   ├── 05.1_representacion_modelos.py
│   ├── 05.2_evaluacion_modelos.py
│   └── 06.1_robustez_visual.py
├── informe/
│   ├── 07_resultados_informe.py
│   └── 08_generar_informe.py     (generador editorial local, ignorado)
└── validar_fase1.py ... validar_fase8.py
```

La capa `pipeline` produce los modelos; `analisis` justifica cada decisión; `informe` numera y
formatea los artefactos. Las subnumeraciones indican el orden: `03.2` corre después de `03.1` y
antes de `03.3`.

Cada fase escribe un manifiesto en `outputs/_manifests/` y llama a un validador. El pipeline se
detiene si la fuente, el panel, la limpieza, el EDA, el ECM, los modelos o los artefactos dejan de
ser coherentes con la fase anterior. Los gates se ejecutan desde `scripts/ejecutar_todo.sh`.

---

## 4. Correcciones metodológicas aplicadas (Opción A)

1. **Estacionariedad:** ADF con constante (principal), ADF con constante y tendencia (robustez)
   y KPSS por producto y departamento. Se reporta el
   orden de integración y se evita afirmar cointegración universal.
2. **Rezagos:** q = 0…4 por producto; se elige por BIC. Resultado: q = 0 en los seis productos.
3. **ECM en niveles:** errores HAC (Newey–West), vida media con IC bootstrap (500 réplicas).
4. **Cointegración:** Engle–Granger como evidencia complementaria.
5. **Panel departamental:** mismo q por producto, 24 departamentos, resumen y mapa de vida media.
6. **Torneo:** nowcast sobre cambios (estacionarios) y pronóstico sobre niveles con caminata
   aleatoria.
7. **Representación de modelos:** coeficientes, camino de Ridge, árbol de RF, árbol sustituto de
   LightGBM, importancias, dependencia parcial y curvas de aprendizaje.
8. **Robustez:** q0/q1/q2, sin shock, ventana 2008 y variedad alternativa.

---

## 5. Artefactos de resultados

| Artefacto | Script |
|---|---|
| Flujo KDD | `07` |
| Tabla del dataset | `03.2` + `07` |
| Auditoría de panel | `02.1` + `07` |
| Verificación de limpieza | `03.1` + `07` |
| Descriptivos y campanas | `03.2` + `07` |
| Estacionariedad y rezagos | `03.3`, `03.4` + `07` |
| Correlaciones y geografía | `03.5` + `07` |
| ECM y mapa de vida media | `04` + `07` |
| Diagnóstico del ECM | `04.1` |
| Representación de modelos | `05.1` |
| Evaluación y curvas | `05.2` |
| Robustez | `06`, `06.1` |
| Protocolo y software | `07` |

---

## 6. Orden de ejecución

```bash
.venv/bin/python scripts/pipeline/01_descargar_sisap.py
.venv/bin/python scripts/analisis/01.1_auditoria_fuente.py
.venv/bin/python scripts/pipeline/02_construir_panel.py
.venv/bin/python scripts/analisis/02.1_auditoria_panel.py
.venv/bin/python scripts/pipeline/03_limpiar_panel.py
.venv/bin/python scripts/analisis/03.1_verificacion_limpieza.py
.venv/bin/python scripts/analisis/03.2_descripcion_datos.py
.venv/bin/python scripts/analisis/03.3_estacionariedad.py
.venv/bin/python scripts/analisis/03.4_seleccion_rezagos.py
.venv/bin/python scripts/analisis/03.5_estructura_mercados.py
.venv/bin/python scripts/pipeline/04_ecm.py
.venv/bin/python scripts/analisis/04.1_diagnostico_ecm.py
.venv/bin/python scripts/pipeline/05_torneo_modelos.py
.venv/bin/python scripts/analisis/05.1_representacion_modelos.py
.venv/bin/python scripts/analisis/05.2_evaluacion_modelos.py
.venv/bin/python scripts/pipeline/06_robustez.py
.venv/bin/python scripts/analisis/06.1_robustez_visual.py
.venv/bin/python scripts/informe/07_resultados_informe.py
```

---

## 7. Criterios de éxito

| Criterio | Umbral | Resultado |
|---|---|---|
| ECM | γ significativo en ≥ 5 de 6 productos | 6 de 6 en Puno |
| Heterogeneidad | rango de vida media ≥ 3× | ~3,5× (1,9 a 6,6 meses) |
| Geografía | distancia significativa; zona evaluada | distancia p = 0,022; zona p = 0,196 |
| Torneo | ranking en productos con ≥ 60 entrenamientos y ≥ 20 evaluaciones; ganador con R² > 0 y acierto > 55 % en ≥ 2 productos | Desarrollo rankea 4 productos elegibles; Random Forest: 3 de 6 con R² > 0 y 3 con acierto > 55 % en holdout |
| Pronóstico | mejora estable en ≥ 2 productos | Ridge y OLS mejoran en papa, cebolla y arroz; ambos pierden en zanahoria, aceite y huevos |

---

## 8. Riesgos

| Riesgo | Mitigación |
|---|---|
| Orden de integración mixto | ECM en niveles; cointegración como evidencia |
| q = 0 por BIC | Reportar q1/q2 como sensibilidad |
| HAC y bootstrap amplían intervalos | Se reportan; arroz pierde significancia sin el shock y el aceite duplica su vida media |
| Sin volúmenes | Variable omitida declarada |
| Cambios de unidad | Normalización y auditoría (313 de 532 series fuente departamento-variedad) |
| n pequeño en modelos | Curvas de aprendizaje; Random Forest queda primero en el ranking de nowcast de cuatro productos elegibles y Ridge/OLS mejoran varios pronósticos |

---

## 9. Entregables

- Dataset final y auditoría.
- Pipeline reproducible y validado (30 scripts versionados, incluidos 8 validadores de fase).
- 25 figuras de artefactos (24 numeradas) y 26 tablas de artefactos (22 numeradas; incluye matriz de diseño, predicciones y matrices de confusión).
- Manifiestos de integridad por fase en `outputs/_manifests/`.
- Dos cuadernos: resultados ejecutivos y guía detallada.
- `docs/metodologia.md`, `docs/referencias.md`, `docs/diccionario_datos.md`.
- README con ejecución de inicio a fin.
