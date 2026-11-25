# Integración espacial de mercados mayoristas de alimentos en el Perú

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-3.1-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![statsmodels](https://img.shields.io/badge/statsmodels-0.15-4C72B0)](https://www.statsmodels.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

Proyecto de **ciencia de datos y EDA** aplicado a la transmisión de precios mayoristas desde Lima
hacia los departamentos del Perú. Combina un modelo de corrección de error (ECM), pruebas de
estacionariedad, análisis geográfico, ingeniería de variables y un torneo de modelos de nowcast y
pronóstico.

**Pregunta:** ¿cuánto y con qué rezago responde el precio mayorista departamental al de Lima, y qué valor
predictivo tiene esa relación?

---

## Hallazgos principales

| Hallazgo | Evidencia |
|---|---|
| El traspaso es incompleto y heterogéneo | Cebolla: 0,51 inmediato y 2,1 meses de vida media; papa: 0,31 y 1,9; zanahoria: 0,18 y 2,8; aceite: 0,24 y 6,6 |
| La integración cae con la distancia | Papa: Ica 0,85 y Huánuco 0,77; Tumbes −0,03 y Loreto −0,01. Promedio entre seis categorías: 0,71 (Ica) a 0,12 (Loreto); distancia p = 0,022 |
| La zona natural no basta | Kruskal–Wallis p = 0,196; distancia y zona explican 54 % de la variación |
| Hay valor predictivo contemporáneo | Random Forest: R² = 0,270 en papa, 0,218 en cebolla y 0,080 en zanahoria; acierto 69,5 / 64,4 / 52,9 % |
| El pronóstico a un mes es específico | Ridge y OLS mejoran el MAE en papa, cebolla y arroz; ambos empeoran en zanahoria, aceite y huevos |
| El orden de integración es mixto | Cebolla I(0); arroz y huevos I(1); aceite mixto en Puno e I(1) en Lima; papa y zanahoria mixtos |
| El BIC selecciona q = 0 | En las seis categorías del núcleo |

**Conclusión:** la referencia de Lima tiene valor contemporáneo y específico por producto. El
pronóstico a un mes mejora en algunos mercados, pero no existe un modelo único para las seis
categorías.

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

## Metodología

1. **Selección y auditoría:** descarga SISAP, revisión de nulos, duplicados, unidades y atípicos.
2. **Transformación:** normalización a S/ kg, log-precio, cambios mensuales y rezagos.
3. **Estacionariedad:** ADF con constante y con tendencia, KPSS y ADF sobre la primera diferencia.
4. **ECM:** traspaso contemporáneo, velocidad de ajuste, vida media, errores HAC y bootstrap.
5. **Torneo:** OLS, Ridge, Random Forest y LightGBM; desarrollo 2018–2020 y holdout enero
   2021–octubre 2026.
6. **Geografía:** correlación con Lima, distancia y zona natural.

El protocolo de evaluación distingue **nowcast** (cambio del mes corriente) de **pronóstico**
(precio del mes siguiente). La selección del modelo se hace por rango promedio de R² en desarrollo,
no por el mejor resultado de un solo producto.

## Enfoque computacional

El proyecto está pensado también como un ejercicio de **análisis exploratorio y estructura de
datos**, no solo como una estimación econométrica:

- el panel se mantiene en formato tidy: una fila por departamento-categoría-mes;
- cada producto se transforma con una tabla pivote (`departamento × fecha`);
- el calendario mensual se repone antes de calcular diferencias y rezagos;
- la matriz de diseño se construye con `diff` y `shift`, sin mezclar meses faltantes;
- la validación es temporal: entrenamiento, desarrollo y holdout;
- se comparan cuatro familias de modelos y se evalúan con R², MAE y una lectura direccional.

La matriz de diseño, las predicciones del holdout y las matrices de confusión se guardan como
artefactos reproducibles en `outputs/analisis/06_modelos/`.

### Del panel a la matriz de diseño

Para el nowcast de cada producto, la variable objetivo es el cambio logarítmico del precio de Puno
en el mes `t`. Los predictores son:

| Predictor | Construcción |
|---|---|
| `dl0` | cambio logarítmico de Lima en `t` |
| `dl1`–`dl3` | cambios de Lima en `t−1`, `t−2`, `t−3` |
| `dp1`, `dp2` | cambios propios de Puno en `t−1`, `t−2` |
| `sin1`, `cos1` | armónicos anuales para estacionalidad |

El modelo se entrena con observaciones anteriores a 2018, se evalúa en desarrollo 2018–2020 y se
prueba una sola vez en el holdout 2021–octubre de 2026. La muestra de la matriz está en
`outputs/analisis/06_modelos/matriz_diseno.csv` y las predicciones completas en
`outputs/analisis/06_modelos/predicciones_holdout.csv`.

### Evaluación por dirección

Además de R² y MAE, se convierte el signo del cambio en una clasificación binaria para evaluar
aciertos, falsos positivos y falsos negativos. Los meses con cambio exactamente cero se excluyen de
esa lectura porque no tienen dirección. LightGBM acierta más el signo en el holdout (62,2 % frente a
58,0 %), pero eso no cambia al ganador: la selección usa R² en desarrollo y el acierto ignora la
magnitud del error. Las matrices agregadas están en
`outputs/analisis/06_modelos/matriz_confusion_modelos.csv` y por producto en
`matriz_confusion_producto.csv`.

---

## Estructura del proyecto

```text
proyecto-mercados-mayoristas/
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
│   ├── ecm.py                       construcción y estimación del ECM
│   ├── pipeline/                    datos y modelos
│   ├── analisis/                    auditorías y diagnóstico
│   ├── validar_fase1.py ... 8.py    gates de integridad por fase
│   └── ...                          artefactos de resultados
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

# Limpieza y descripción
.venv/bin/python scripts/pipeline/03_limpiar_panel.py
.venv/bin/python scripts/analisis/03.1_verificacion_limpieza.py
.venv/bin/python scripts/analisis/03.2_descripcion_datos.py
.venv/bin/python scripts/analisis/03.3_estacionariedad.py
.venv/bin/python scripts/analisis/03.4_seleccion_rezagos.py
.venv/bin/python scripts/analisis/03.5_estructura_mercados.py

# ECM y diagnóstico
.venv/bin/python scripts/pipeline/04_ecm.py
.venv/bin/python scripts/analisis/04.1_diagnostico_ecm.py

# Torneo de modelos
.venv/bin/python scripts/pipeline/05_torneo_modelos.py
.venv/bin/python scripts/analisis/05.1_representacion_modelos.py
.venv/bin/python scripts/analisis/05.2_evaluacion_modelos.py

# Robustez
.venv/bin/python scripts/pipeline/06_robustez.py
.venv/bin/python scripts/analisis/06.1_robustez_visual.py

# Los artefactos de resultados se generan con el pipeline completo.
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
- `outputs/analisis/06_modelos/`: matriz de diseño, predicciones del holdout y matrices de confusión.
- `outputs/_manifests/`: manifiestos y hashes de cada fase.
- `notebooks/01_resultados.ipynb`: recorrido ejecutivo con tablas y figuras.
- `notebooks/02_guia_detallada.ipynb`: guía paso a paso, también versionada.

El índice de tablas y figuras se genera dentro de `outputs/`.

---

## Reproducibilidad

- Semillas fijas en los modelos (`random_state=0`).
- Versiones de software registradas en los artefactos de `outputs/`.
- Descarga con caché: si `data/raw/*.html` existe, no se consulta el portal.
- Pipeline completo: aproximadamente 10–15 minutos, dependiendo de la descarga y del torneo.
- `scripts/ejecutar_todo.sh` ejecuta los gates 1–8; si una fase no cuadra con la anterior, se detiene.

La fuente se consulta con `scripts/pipeline/01_descargar_sisap.py`. Los HTML crudos no se versionan.

---

## Limitaciones

- El módulo de ciudades no publica volúmenes de abastecimiento.
- Los cambios de presentación y unidad introducen ruido de medición; se normalizan a kilogramos.
- Algunas series regionales terminan antes de 2026.
- No hay marca de fecha de publicación: se habla de valor predictivo contemporáneo, no de un
  nowcast operativo.
- El shock de 2021–2023 afecta la robustez de algunos productos, especialmente el arroz.
- Los precios son mayoristas, no minoristas.
- El tamaño de muestra es limitado para modelos no lineales; los resultados se interpretan por
  producto.

---

## Referencias y documentación

- [`docs/metodologia.md`](docs/metodologia.md): ecuaciones, protocolo y supuestos.
- [`docs/diccionario_datos.md`](docs/diccionario_datos.md): variables, cobertura y tratamiento de datos.
- [`docs/plan_desarrollo.md`](docs/plan_desarrollo.md): decisiones y fases.
- [`docs/referencias.md`](docs/referencias.md): bibliografía.

---

## Licencia y uso

Proyecto con fines académicos y de investigación. Si reutilizas el código, los datos o los resultados,
cita la fuente original (SISAP–MIDAGRI) y este repositorio.
