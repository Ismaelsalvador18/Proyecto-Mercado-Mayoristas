# Metodología

## 1. Marco KDD

El estudio sigue las cinco fases del proceso de descubrimiento de conocimiento en bases de datos
(Fayyad et al., 1996):

1. **Selección:** SISAP–MIDAGRI, precios mayoristas de 24 departamentos y 9 categorías armonizadas.
2. **Preprocesamiento:** auditoría de nulos, duplicados, unidades y atípicos.
3. **Transformación:** normalización a S/ kg, log-precio, cambios, rezagos y estacionalidad.
4. **Minería:** modelo de corrección de error y torneo de modelos.
5. **Evaluación:** inferencia con HAC y bootstrap; pronóstico con R², MAE, mejora relativa, acierto
   direccional y Diebold–Mariano.

## 2. Datos

| Elemento | Valor |
|---|---|
| Fuente | SISAP, módulo Precios de ciudades |
| Unidad | departamento × producto × mes |
| Crudo | 102 808 filas |
| Panel | 43 921 filas |
| Limpio | 43 855 filas |
| Ventana nominal | 1997–2026 |
| Ventana densa | 2012–2023 |

La cobertura no es un panel balanceado: 81 de 216 series terminan antes de noviembre de 2026 y
17 empiezan después de 2012. Se declara la muestra efectiva por producto.

La unidad transversal es el **departamento**. SISAP reporta por separado Andahuaylas, Chota y Jaén;
las dos últimas son provincias de Cajamarca y la primera de Apurímac. Se excluyen para no tratar una
provincia como si fuera un departamento. Callao tampoco entra: es provincia constitucional y no
devuelve registros. El resultado son 24 departamentos comparables.

La unidad de análisis es una **serie departamento-categoría por mes** (24 departamentos × 9 categorías =
216 series potenciales). En el origen hay 532 series departamento-variedad; la construcción del panel
selecciona una variedad y unidad dominante para cada combinación departamento-categoría. “Variedad” se
reserva para las presentaciones comerciales de la fuente; “categoría” es el producto armonizado
que se analiza.

### 2.1 Variables

| Variable | Tipo | Unidad | Rol |
|---|---|---|---|
| `fecha` | fecha | mes | índice |
| `region` | categórica | 24 departamentos | unidad transversal |
| `categoria` | categórica | 9 categorías | unidad de producto armonizado |
| `precio_kg` | continua | S/ kg | variable de análisis |
| `log_precio` | continua | log S/ kg | estacionariedad |
| `cambio_log` | continua | log-diferencia | ECM y modelos |

### 2.2 Normalización y atípicos

El SISAP publica en kilogramos, arrobas, sacos, cajas y otras presentaciones. Para cada celda
departamento-producto se elige la unidad dominante y se divide por su equivalencia en kg. El 58,8 % de
las series departamento-variedad (313 de 532) cambia de unidad al menos una vez; por eso la normalización es obligatoria.

Se descartan valores con cociente > 4 frente a la mediana móvil de ±3 meses, o > 5 frente a la
mediana nacional del mismo producto y mes. Total: 66 casos (0,15 %). No se imputa: se marca como
faltante.

### 2.3 Transformación logarítmica

El log reduce la asimetría en 9 de 9 productos (ver `transformacion_log.csv`). Para dos meses
consecutivos, `Δlog(P) = log(P_t/P_{t-1})`; el cambio porcentual exacto es
`100 × (exp(Δlog(P)) − 1)`, y para cambios pequeños se aproxima por `100 × Δlog(P)`. Por ejemplo,
un aumento de 100 a 105 equivale exactamente a 5 % y a una diferencia logarítmica de 0,0488,
aproximadamente 4,88 %. El log reduce la asimetría observada, pero no garantiza normalidad ni
estabiliza necesariamente la varianza.

## 3. Estacionariedad

Se aplican ADF con constante (especificación principal), ADF con constante y tendencia (robustez)
y KPSS sobre log-precio. Si ADF y KPSS discrepan en niveles, se revisa además ADF en la primera
diferencia. Regla de clasificación aplicada en el código:

- ADF en niveles rechaza raíz unitaria (`p < 0,05`) y KPSS no rechaza estacionariedad → I(0).
- ADF en niveles no rechaza y KPSS rechaza; ADF en primera diferencia rechaza → I(1).
- Si las pruebas discrepan y la primera diferencia tampoco rechaza raíz unitaria → mixto.
- Los demás casos ambiguos se reportan como mixtos.

Resultado: cebolla es I(0); arroz y huevos son I(1); aceite es mixto en Puno e I(1) en Lima; papa y zanahoria son mixtos. Por eso
se habla de **modelo de corrección de error en niveles**, no de cointegración universal.

## 4. Modelo de corrección de error

Para cada producto y departamento:

```
Δp(t) = α + Σ(k=0..q) βk Δp_Lima(t−k) + γ [p(t−1) − p_Lima(t−1)] + ε(t)
```

- `β0`: traspaso inmediato.
- `γ`: velocidad de ajuste; debe estar entre −2 y 0.
- Vida media: `ln(0,5) / ln(1+γ)`.
- `q`: seleccionado por BIC entre 0 y 4. Resultado: q = 0 en los seis productos.
- Errores estándar HAC (Newey–West, 4 rezagos).
- Intervalo de confianza de la vida media por bootstrap de pares (500 réplicas).
- Engle–Granger como evidencia complementaria de equilibrio de largo plazo.

La asimetría (subidas vs. bajadas) se estima con `Δp_Lima` positivo y negativo; se reporta como
exploratoria porque ningún producto alcanza significancia al 5 %.

Para H2, “frescos” designa papa, cebolla y zanahoria; “procesados o almacenables”, aceite y arroz.
Huevos se mantiene como categoría intermedia y no se usa para forzar la dicotomía.

## 5. Torneo de modelos

### 5.1 Objetivos

- **Nowcast:** cambio del mes corriente; se justifica porque los cambios son estacionarios.
- **Pronóstico:** nivel del mes siguiente; se compara con la caminata aleatoria.

### 5.2 Modelos

| Modelo | Familia | Hiperparámetros |
|---|---|---|
| OLS | lineal | sin hiperparámetros |
| Ridge | lineal regularizado | α = 1, estandarización |
| Random Forest | no lineal, bagging | 300 árboles, hoja mínima 3 |
| LightGBM | no lineal, boosting | 300 árboles, lr 0,05, 7 hojas |

El cuarteto del ingeniero (árbol, RF, XGBoost, LightGBM) se reporta en anexo y no compite.

### 5.3 Variables

Nowcast: `Δlog Lima` (0–3 rezagos), `Δlog Puno` (1–2 rezagos), armónicos anuales.
Pronóstico: `log Puno` y `log Lima` (0–1 rezagos), mediana de otros departamentos, armónicos.

### 5.4 Protocolo

| Elemento | Definición |
|---|---|
| Desarrollo (evaluación) | enero 2018–diciembre 2020 (36 meses calendario) |
| Entrenamiento para desarrollo | observaciones anteriores a enero de 2018 |
| Holdout | enero 2021–octubre 2026, una sola evaluación |
| Elegibilidad del ranking de nowcast | ≥ 60 observaciones de entrenamiento y ≥ 20 de evaluación por producto |
| Baseline nowcast | cambio cero |
| Baseline pronóstico | caminata aleatoria |
| Métrica principal nowcast | R² |
| Métrica principal pronóstico | MAE |
| Métricas secundarias | MAE, mejora relativa, acierto direccional |
| Selección | rango promedio de R² en desarrollo |
| Contraste | Diebold–Mariano con corrección HLN |
| Compuerta | R² > 0 y acierto > 55 % en ≥ 2 productos |

Con esa regla, la ventana de desarrollo permite rankear papa, arroz, aceite y huevos; cebolla tiene
17 observaciones válidas de evaluación y zanahoria 52 de entrenamiento y 17 de evaluación, así que
ambas quedan fuera del ranking de desarrollo. Los seis productos sí tienen holdout suficiente. La
curva de aprendizaje de cebolla usa 33, 46, 56 o 67 observaciones mensuales de entrenamiento
anteriores a 2018 y evalúa en sus 17 observaciones válidas de 2018–2020. El ranking se debe leer
como ganador entre cuatro productos elegibles, no como una comparación equilibrada de seis.

### 5.5 Matriz de diseño

Para cada producto y departamento, el nowcast se construye así:

1. Se hace una tabla pivote de `precio_kg` con fechas en filas y departamentos en columnas.
2. Se repone el calendario mensual completo para no convertir un hueco en un mes consecutivo.
3. Se calcula `y = Δlog(Puno)` y los predictores `dl0–dl3`, `dp1–dp2`, `sin1` y `cos1`.
4. Se eliminan las filas sin historia suficiente.
5. Se separan entrenamiento (antes de 2018), desarrollo (2018–2020) y holdout (2021–octubre de 2026).

La matriz completa y una muestra legible se guardan en `matriz_diseno.csv` y
`matriz_diseno_muestra.csv`; las predicciones del holdout se guardan en `predicciones_holdout.csv`.
El objetivo se mantiene como variable continua; la matriz de confusión se obtiene después,
clasificando el signo del cambio observado y del predicho.

### 5.6 Evaluación direccional

Para cada modelo se construye una matriz 2×2 en el holdout:

| | Predice baja | Predice sube |
|---|---:|---:|
| Real baja | TN | FP |
| Real sube | FN | TP |

Los meses con cambio observado exactamente cero se excluyen porque no tienen dirección. Esta matriz
no reemplaza R² ni MAE: muestra los errores de signo que las métricas agregadas pueden ocultar.

Por eso LightGBM puede tener mayor acierto agregado en el holdout (62,2 % frente a 58,0 % de Random
Forest) y no ser el ganador. El acierto solo mira el signo: un acierto con error de 0,1 % pesa lo
mismo que uno con error de 5 %. La selección se hizo en desarrollo con R², donde Random Forest tiene
rango promedio 1,50 y acierto medio 46,9 %, frente a 4,00 y 42,8 % de LightGBM. En el holdout,
Random Forest conserva menor MAE en cinco de seis productos y R² positivo en papa, cebolla y
zanahoria, mientras LightGBM tiene R² muy negativo en arroz (−4,04) y zanahoria (−0,28). Elegir al
ganador según el holdout sería seleccionar sobre el conjunto de prueba; la matriz de confusión se
usa como diagnóstico complementario, no como criterio de selección.

## 6. Métricas

- **R²:** proporción de varianza explicada. En niveles es engañoso; se usa en cambios.
- **MAE:** error absoluto medio, en unidades de la variable.
- **Mejora relativa:** reducción porcentual del MAE frente a la caminata aleatoria.
- **Acierto direccional:** porcentaje de signos correctos entre meses con cambio observado distinto de cero. Los meses sin cambio no tienen dirección y no se cuentan como acierto ni como error.
- **Diebold–Mariano:** igualdad de precisión predictiva; pérdida cuadrática; h = 1; corrección
  de Harvey, Leybourne y Newbold.

## 7. Supuestos y validez

1. **Linealidad local:** el ECM supone relación lineal en logaritmos. En el torneo, Random Forest
   queda primero en el ranking de nowcast de los cuatro productos elegibles; Ridge/OLS mejoran el pronóstico a un mes en varios productos.
   No se generaliza un ganador a todos los objetivos.
2. **Sin volúmenes:** el módulo de ciudades no publica abastecimiento; hay variable omitida.
3. **Cambios de unidad:** normalizados, pero queda ruido de medición.
4. **Serie truncada:** 81 de 216 series terminan antes de noviembre de 2026.
5. **Quiebre 2021–2023:** se reporta robustez sin ese periodo; el arroz pierde significancia y la vida media del aceite sube de 6,6 a 10,1 meses.
6. **Rezago de publicación:** no verificable; se habla de valor predictivo contemporáneo, no de
   nowcast operativo.
7. **n pequeño:** el ECM base de Puno usa 114–139 observaciones mensuales. Por separado, la curva
   de aprendizaje de cebolla usa 33, 46, 56 y 67 observaciones de entrenamiento previas a 2018;
   muestra cómo cambia el rendimiento en ese producto, no una ley general.

El panel incluye registros de noviembre de 2026, pero el código fija octubre como corte del holdout.
Como no se verificó la fecha de cierre/publicación de noviembre, se usa octubre por prudencia; los
registros de noviembre quedan en el panel y no se califican como erróneos o incompletos.

## 8. Reproducibilidad

- Semillas fijas (`random_state=0`).
- Versiones en `tabla_17_software.csv`.
- Pipeline de 30 scripts versionados con orden explícito, incluidos 8 validadores de fase.
- Datos crudos en caché; datos procesados versionados.

## Anexo. Formulación matemática

Las ecuaciones se numeran igual que en los resultados. Se incluyen para que la implementación y el análisis sean verificables.

**Retorno logarítmico (1).**

$$
r_t = \ln(P_t) - \ln(P_{t-1})
$$

**Correlación de Pearson (2).**

$$
\rho_{XY} = \frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}
{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2 \sum_{i=1}^{n}(y_i-\bar{y})^2}}
$$

**ADF (3).**

$$
\Delta y_t = \alpha + \gamma y_{t-1} + \sum_{i=1}^{p}\delta_i \Delta y_{t-i} + \varepsilon_t,
\quad H_0:\gamma=0
$$

La tendencia determinista `βt` se agrega únicamente en la especificación ADF de robustez.

**KPSS (4).**

$$
KPSS = \frac{1}{T^2}\sum_{t=1}^{T}\frac{S_t^2}{\hat{\sigma}^2},
\quad S_t=\sum_{i=1}^{t}\hat{u}_i
$$

**AIC (5) y BIC (6).**

$$
AIC=-2\ln L+2k,\qquad BIC=-2\ln L+k\ln n
$$

**ECM (7).**

$$
\Delta p_{r,t} = \alpha + \sum_{k=0}^{q}\beta_k \Delta p_{L,t-k}
+ \gamma\left(p_{r,t-1}-p_{L,t-1}\right)+\varepsilon_t
$$

**Vida media (8).**

$$
HL=\frac{\ln(0.5)}{\ln(1+\gamma)}
$$

**Engle–Granger (9).**

$$
p_{r,t}=a+b\,p_{L,t}+u_t;\qquad \text{ADF sobre } \hat{u}_t
$$

**OLS (10) y Ridge (11).**

$$
\hat{\beta}=\arg\min_{\beta}\sum_{i=1}^{n}(y_i-x_i'\beta)^2
$$

$$
\hat{\beta}=\arg\min_{\beta}\sum_{i=1}^{n}(y_i-x_i'\beta)^2
+\lambda\sum_{j=1}^{p}\beta_j^2
$$

**Random Forest (12) y LightGBM (13).**

$$
\hat{f}(x)=\frac{1}{B}\sum_{b=1}^{B}T_b(x)
$$

$$
F_m(x)=F_{m-1}(x)+\nu h_m(x)
$$

**Métricas (14)–(18).**

$$
R^2=1-\frac{\sum_i(y_i-\hat{y}_i)^2}{\sum_i(y_i-\bar{y})^2}
$$

$$
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

$$
Mejora=100\left(1-\frac{MAE_{modelo}}{MAE_{RW}}\right)
$$

$$
DA=\frac{1}{n}\sum_t \mathbf{1}[y_t\neq y_{t-1}]\,\mathbf{1}\left[\operatorname{sign}(\hat{y}_t-y_{t-1})
=\operatorname{sign}(y_t-y_{t-1})\right]
$$

$$
DM=\frac{\bar{d}}{\sqrt{\hat{V}(d)/n}},\qquad d_t=e_{1t}^2-e_{2t}^2
$$

**Intervalo bootstrap (19).**

$$
IC(95\,\%)=[q(0.025;\theta^*),\,q(0.975;\theta^*)]
$$

**Símbolos.** \(P_t\): precio; \(y_t\): log-precio; \(r_t\): retorno; \(x_i\): predictores; \(\beta\): coeficientes; \(\gamma\): velocidad de ajuste; \(\lambda\): penalización Ridge; \(T_b\): árbol; \(B\): número de árboles; \(F_m\): ensemble de boosting; \(\nu\): tasa de aprendizaje; \(h_m\): árbol de corrección; \(e_{1t},e_{2t}\): errores de dos modelos; \(\bar{d}\): media de diferencias de pérdida; \(\theta^*\): vida media en una réplica bootstrap.
