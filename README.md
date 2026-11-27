# Análisis exploratorio y modelos predictivos de precios mayoristas en el Perú

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-3.1-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![statsmodels](https://img.shields.io/badge/statsmodels-0.15-4C72B0)](https://www.statsmodels.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

Proyecto de **EDA y aprendizaje supervisado** sobre precios mayoristas: construcción del dataset,
limpieza, análisis exploratorio, ingeniería de variables y comparación de modelos para estimar el
cambio del mes corriente y el precio del mes siguiente.

**Preguntas:**

1. ¿Cómo se distribuyen, se cubren y se relacionan los precios mayoristas de Lima y los departamentos?
2. ¿Qué modelo estima mejor el cambio del mes corriente y cuál generaliza mejor al pronosticar un mes?

**Hipótesis:**

- **H1:** el cambio de precio de Lima del mismo mes contiene información útil para estimar el cambio
  de precio en los departamentos.
- **H2:** el desempeño predictivo es heterogéneo por producto y por horizonte; un modelo bueno para
  el mes corriente no necesariamente sirve para pronosticar el mes siguiente.

### Glosario

| Término en el texto | Nombre interno | Significado |
|---|---|---|
| Estimación del mes corriente | `nowcast` | Estimar el cambio del mes actual con información del mismo mes |
| Validación | `development` | Período 2018–2020 para elegir el modelo |
| Prueba final | `holdout` | Período 2021–octubre 2026, usado una sola vez |
| Pronóstico a un mes | `forecast` | Estimar el precio del mes siguiente |
| Referencia | `baseline` | Predicción simple contra la que se compara el modelo |

Los nombres internos se conservan en archivos y columnas para no romper el código.

---

## Hallazgos principales

| Hallazgo | Evidencia |
|---|---|
| El dataset es heterogéneo y no balanceado | 24 departamentos, 216 series y 43 855 observaciones limpias; cobertura 85,7 % en 2012–2023 |
| La relación con Lima varía por producto y departamento | Papa: Ica 0,85 y Huánuco 0,77; Tumbes −0,03 y Loreto −0,01; promedio de 0,71 (Ica) a 0,12 (Loreto) |
| La distancia aporta, pero no explica todo | Distancia p = 0,022; zona natural p = 0,196; juntas explican 54 % de la variación |
| Hay valor predictivo contemporáneo | Random Forest: R² = 0,270 en papa, 0,218 en cebolla y 0,080 en zanahoria; acierto 69,5 / 64,4 / 52,9 % |
| El pronóstico a un mes es específico | Ridge y OLS mejoran el MAE en papa, cebolla y arroz; ambos empeoran en zanahoria, aceite y huevos |
| Lectura direccional | LightGBM: F1 = 0,662 y exactitud = 0,622; OLS y Ridge: recall = 0,660; ROC-AUC cercano a 0,63 |
| Clasificación de la dirección | Árbol de Decisión gini: exactitud = 58,5 % y F1 = 0,667; LightGBM: ROC-AUC = 0,621, pero recall = 0,303 |

**Conclusión:** la referencia de Lima aporta información predictiva contemporánea, pero el
desempeño depende del producto y del horizonte. No hay un modelo único para las seis categorías.

---

## Datos

| Elemento | Valor |
|---|---|
| Fuente | SISAP–MIDAGRI, módulo *Precios de ciudades* |
| Catálogo | `data/reference/departamentos.csv` (24 departamentos) |
| URL | `sistemas.midagri.gob.pe/sisap/portal2/ciudades` |
| Cobertura nominal | 1997–2026 |
| Cobertura efectiva | 2002–noviembre de 2026; ventana densa 2012–2023 |
| Unidad analítica | serie departamento-categoría por mes |
| Departamentos | 24 (unidad de análisis) |
| Categorías | 9 productos armonizados |
| Datos crudos | 102 808 filas |
| Panel construido | 43 921 filas |
| Panel limpio | 43 855 filas; 66 atípicos descartados (0,15 %) |
| Cobertura 2012–2023 | 85,7 % de las celdas departamento-categoría |
| Cobertura 2024–noviembre de 2026 | 66,5 % |

La fuente contiene **variedades comerciales**. Para el análisis se selecciona una variedad y una
unidad dominante por celda, y se armonizan en **9 categorías**. Las 532 series fuente
departamento-variedad se convierten en 216 series analíticas departamento-categoría.

Se trabaja con **24 departamentos**. SISAP reporta por separado los mercados provinciales de
Andahuaylas, Chota y Jaén, que pertenecen a Apurímac y Cajamarca; se excluyen para no duplicar la
unidad departamental. Callao tampoco se cuenta porque es provincia constitucional y no devuelve
registros.

**Categorías del núcleo:** papa, cebolla, zanahoria, arroz, aceite y huevos. Tomate, limón y azúcar
se usan en el análisis descriptivo.

Más detalle en [`docs/diccionario_datos.md`](docs/diccionario_datos.md).

---

## Ventanas temporales y tamaño de muestra

No hay tres estudios distintos: hay una sola fuente y tres cortes temporales.

| Ventana | Uso |
|---|---|
| 1997–2026 | Fuente completa disponible |
| 2012–2023 | Ventana densa para EDA y estimaciones |
| 2012–2017 | Entrenamiento de los modelos |
| 2018–2020 | Validación para elegir el modelo |
| 2021–octubre 2026 | Prueba final, una sola vez |
| Noviembre 2026 | Fuera de la prueba por corte prudencial |

Una **observación mensual** es un mes con datos completos para ese producto después de construir
log-precio, cambios y rezagos. Por eso el `N` no es el número de meses calendario: los faltantes y
los rezagos eliminan filas.

| Producto | Tarea | Entrenamiento 2012–2017 | Validación 2018–2020 | Prueba final 2021–2026 |
|---|---|---:|---:|---:|
| Papa | cambio del mes corriente | 67 | 28 | 60 |
| Papa | pronóstico a un mes | 69 | 29 | 62 |
| Cebolla | cambio del mes corriente | 67 | 17 | 60 |
| Cebolla | pronóstico a un mes | 69 | 19 | 62 |
| Zanahoria | cambio del mes corriente | 52 | 17 | 52 |
| Zanahoria | pronóstico a un mes | 57 | 19 | 53 |
| Arroz | cambio del mes corriente | 68 | 27 | 31 |
| Arroz | pronóstico a un mes | 71 | 26 | 36 |
| Aceite | cambio del mes corriente | 68 | 32 | 24 |
| Aceite | pronóstico a un mes | 71 | 33 | 24 |
| Huevos | cambio del mes corriente | 67 | 32 | 38 |
| Huevos | pronóstico a un mes | 69 | 33 | 38 |

La tabla completa se genera en `outputs/analisis/06_modelos/n_etapas.csv`.
El `N` cambia por los meses faltantes, las series truncadas, los productos sin dato en algún mes y
los rezagos que consumen las primeras observaciones. No es un error de conteo.

---

## Metodología

1. **Selección y auditoría:** descarga SISAP, revisión de nulos, duplicados, unidades y atípicos.
2. **Transformación:** normalización a S/ kg, log-precio, cambios mensuales y rezagos.
3. **EDA:** cobertura, distribuciones, correlaciones y geografía básica.
4. **Variables:** construcción de la matriz X/y con diferencias, rezagos y estacionalidad.
5. **Modelos:** OLS, Ridge y cuatro familias de árboles (Árbol de Decisión, Random Forest, XGBoost
   y LightGBM); validación 2018–2020 y prueba final enero 2021–octubre 2026.
6. **Evaluación:** R², MAE, mejora frente a la referencia, acierto direccional y matriz de confusión.

El protocolo distingue la **estimación del mes corriente** (cambio del mismo mes) del
**pronóstico a un mes** (precio del mes siguiente). El modelo se selecciona por rango promedio de R²
en **validación**, no por el mejor resultado de un solo producto.

## Enfoque computacional

El proyecto está pensado como un ejercicio de **análisis exploratorio, estructuras de datos y
modelos**:

- el panel se mantiene en formato tidy: una fila por departamento-categoría-mes;
- cada producto se transforma con una tabla pivote (`departamento × fecha`);
- el calendario mensual se repone antes de calcular diferencias y rezagos;
- la matriz de diseño se construye con `diff` y `shift`, sin mezclar meses faltantes;
- la validación es temporal: entrenamiento, validación y prueba final;
- se comparan cuatro familias de modelos y se evalúan con R², MAE y una lectura direccional.

La matriz de diseño, las predicciones de la prueba final y las matrices de confusión se guardan como
artefactos reproducibles en `outputs/analisis/06_modelos/`.

### Del panel a la matriz de diseño

Para la estimación del mes corriente de cada producto, la variable objetivo es el cambio logarítmico del precio de Puno
en el mes `t`. Los predictores son:

| Predictor | Construcción |
|---|---|
| `dl0` | cambio logarítmico de Lima en `t` |
| `dl1`–`dl3` | cambios de Lima en `t−1`, `t−2`, `t−3` |
| `dp1`, `dp2` | cambios propios de Puno en `t−1`, `t−2` |
| `sin1`, `cos1` | armónicos anuales para estacionalidad |

El modelo se entrena con observaciones de 2012–2017, se evalúa en validación 2018–2020 y se
prueba una sola vez en la prueba final 2021–octubre de 2026. La muestra de la matriz está en
`outputs/analisis/06_modelos/matriz_diseno.csv` y las predicciones completas en
`outputs/analisis/06_modelos/predicciones_holdout.csv`.

### Evaluación por dirección

Además de R² y MAE, se convierte el signo del cambio en una clasificación binaria para evaluar
aciertos, falsos positivos y falsos negativos. Los meses con cambio exactamente cero se excluyen de
esa lectura porque no tienen dirección. LightGBM acierta más el signo en la prueba final (62,2 % frente a
58,0 %), pero eso no cambia al ganador: la selección usa R² en validación y el acierto ignora la
magnitud del error. Las matrices agregadas están en
`outputs/analisis/06_modelos/matriz_confusion_modelos.csv` y por producto en
`matriz_confusion_producto.csv`.

---

## Estructura del proyecto

```text
proyecto-mercados-mayoristas/
├── assets/                          figura KDD de referencia
├── data/
│   ├── raw/                         respuestas HTML de la fuente (no versionadas)
│   ├── reference/                   catálogo de los 24 departamentos
│   └── processed/                   panel crudo, panel limpio y auditorías
├── docs/
│   ├── plan_desarrollo.md           decisiones y fases del proyecto
│   ├── metodologia.md               métodos, ecuaciones y supuestos
│   ├── diccionario_datos.md         variables, cobertura y linaje
│   └── referencias.md               bibliografía
├── notebooks/
│   ├── 01_resultados.ipynb          recorrido ejecutivo de resultados
│   └── 02_guia_detallada.ipynb      guía paso a paso
├── outputs/                         resultados intermedios, tablas y figuras
├── scripts/
│   ├── config.py                    rutas, ventanas y parámetros
│   ├── catalogo.py                  validación de los 24 departamentos
│   ├── modelos.py                   variables y métricas compartidas
│   ├── pipeline/                    datos y modelos
│   ├── analisis/                    EDA, estructura y evaluación
│   ├── clasificacion/               clasificadores de dirección
│   ├── informe/                     tablas, figuras y generación del informe
│   ├── validar_fase*.py             validadores de fase
│   └── ...
├── extras/econometria/              material técnico opcional, fuera del pipeline
├── requirements.txt
└── README.md
```

---

## Instalación

Requiere Python 3.11 o superior.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Uso

### Pipeline completo

```bash
bash scripts/ejecutar_todo.sh
```

El pipeline descarga los datos si no existen en caché, construye el panel, ejecuta los análisis y
genera los resultados en `outputs/`.

### Ejecución por etapas

```bash
# Datos y panel
.venv/bin/python scripts/pipeline/01_descargar_sisap.py
.venv/bin/python scripts/analisis/01.1_auditoria_fuente.py
.venv/bin/python scripts/pipeline/02_construir_panel.py
.venv/bin/python scripts/analisis/02.1_auditoria_panel.py

# Limpieza, EDA y estructura
.venv/bin/python scripts/pipeline/03_limpiar_panel.py
.venv/bin/python scripts/analisis/03.1_verificacion_limpieza.py
.venv/bin/python scripts/analisis/03.2_descripcion_datos.py
.venv/bin/python scripts/analisis/03.5_estructura_mercados.py

# Torneo de modelos y tamaño de muestra
.venv/bin/python scripts/pipeline/05_torneo_modelos.py
.venv/bin/python scripts/analisis/05.1_representacion_modelos.py
.venv/bin/python scripts/analisis/05.2_evaluacion_modelos.py
.venv/bin/python scripts/analisis/05.3_n_etapas.py

# Clasificación de la dirección
.venv/bin/python scripts/clasificacion/01_clasificadores_direccion.py

# Artefactos de resultados
.venv/bin/python scripts/informe/07_resultados_informe.py
```

### Cuaderno ejecutivo

```bash
.venv/bin/python -m ipykernel install --user --name mercados --display-name "Python (mercados)"
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=mercados notebooks/01_resultados.ipynb
```

Si `jupyter` no está en el `PATH`, usa el binario de tu instalación. Por ejemplo:

```bash
~/jupyter-env/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=mercados notebooks/01_resultados.ipynb
```

---

## Resultados y artefactos

- `data/processed/`: panel limpio y auditorías.
- `outputs/analisis/`: resultados intermedios por fase.
- `outputs/`: tablas y figuras listas para consulta.
- `outputs/analisis/06_modelos/`: matriz de diseño, predicciones de la prueba final y matrices de confusión.
- `outputs/_manifests/`: manifiestos y hashes de cada fase.
- `notebooks/01_resultados.ipynb`: recorrido ejecutivo con tablas y figuras.
- `notebooks/02_guia_detallada.ipynb`: guía paso a paso, también versionada.
- `informe.pdf`: informe final con 11 tablas, 10 figuras, 12 ecuaciones y un anexo con la matriz completa.
- `informe.docx`: fuente editable del informe.

El índice de tablas y figuras se genera dentro de `outputs/`. El informe se regenera al final del
pipeline y también puede generarse por separado:

```bash
.venv/bin/python scripts/informe/08_generar_informe.py
soffice --headless --convert-to pdf --outdir . informe.docx
```

El generador produce el `.docx`; la conversión a PDF requiere LibreOffice.

---

## Reproducibilidad

- Semillas fijas en los modelos (`random_state=0`).
- Versiones de software registradas en los artefactos de `outputs/`.
- Descarga con caché: si `data/raw/*.html` existe, no se consulta el portal.
- Pipeline completo: aproximadamente 10–15 minutos, dependiendo de la descarga y del torneo.
- `scripts/ejecutar_todo.sh` ejecuta los validadores de fase; si una etapa no cuadra con la anterior, se detiene.

La fuente se consulta con `scripts/pipeline/01_descargar_sisap.py`. Los HTML crudos no se versionan.

---

## Limitaciones

- El módulo de ciudades no publica volúmenes de abastecimiento.
- Los cambios de presentación y unidad introducen ruido de medición; se normalizan a kilogramos.
- Algunas series regionales terminan antes de 2026.
- No hay marca de fecha de publicación: se habla de valor predictivo contemporáneo, no de un
  estimación operativa del mes corriente.
- Los precios son mayoristas, no minoristas.
- El tamaño de muestra es limitado para modelos no lineales; los resultados se interpretan por
  producto.

---

## Referencias y documentación

- [`docs/metodologia.md`](docs/metodologia.md): método, ventanas y evaluación.
- [`docs/diccionario_datos.md`](docs/diccionario_datos.md): variables, cobertura y tratamiento de datos.
- [`docs/plan_desarrollo.md`](docs/plan_desarrollo.md): decisiones y fases.
- [`docs/referencias.md`](docs/referencias.md): bibliografía.

---

## Licencia y uso

Proyecto con fines académicos y de investigación. Si reutilizas el código, los datos o los resultados,
cita la fuente original (SISAP–MIDAGRI) y este repositorio.
