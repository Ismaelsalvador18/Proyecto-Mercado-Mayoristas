# Diccionario de datos

## Fuente

| Campo | Valor |
|---|---|
| Nombre | SISAP – Sistema de Información de Abastecimiento y Precios |
| Institución | Ministerio de Desarrollo Agrario y Riego (MIDAGRI) |
| Módulo | Precios de ciudades |
| URL | `http://sistemas.midagri.gob.pe/sisap/portal2/ciudades` |
| Cobertura nominal | 1997–2026 |
| Frecuencia | mensual |
| Unidad | departamento × producto × mes |
| Moneda | soles corrientes (S/) |

El portal no ofrece una descarga única. La consulta se reproduce con una petición GET a
`/resumenes/filtrar` con `ajax=true`, departamento por departamento. Ver `scripts/pipeline/01_descargar_sisap.py`.

En la fuente, una **variedad** es una presentación comercial del producto. El panel final armoniza
esas variedades en 9 **categorías** y toma como unidad una serie departamento-categoría por mes
(216 combinaciones posibles). Las 532 series departamento-variedad de origen no son 532 series de
análisis. Se excluyen los mercados provinciales Andahuaylas, Chota y Jaén, y Callao, para mantener
24 departamentos como unidad transversal.

## Archivos

| Archivo | Filas | Contenido |
|---|---:|---|
| `data/reference/departamentos.csv` | 24 | catálogo oficial ISO/SISAP |
| `data/raw/ciudades_<código>.html` | 24 usados | respuesta cruda por departamento |
| `data/processed/precios_ciudades_largo.csv` | 102 808 | extracción de 24 departamentos |
| `data/processed/panel_precios.csv` | 43 921 | panel tidy normalizado |
| `data/processed/panel_precios_limpio.csv` | 43 855 | panel final sin atípicos |
| `outputs/informe/tablas/` | — | tablas numeradas del informe |
| `outputs/informe/figuras/` | — | figuras numeradas del informe |
| `data/processed/atipicos_panel.csv` | 66 | valores descartados |
| `data/processed/cobertura_panel.csv` | 216 | cobertura por celda (auditoría) |
| `data/processed/unidades_panel.csv` | 216 | unidad dominante por celda |

## Variables del panel

| Variable | Tipo | Descripción |
|---|---|---|
| `fecha` | fecha | primer día del mes |
| `anio` | entero | año |
| `mes` | entero | mes (1–12) |
| `departamento_id` | texto | código ISO 3166-2:PE |
| `departamento` | texto | nombre oficial con tildes |
| `region` | texto | alias ASCII del departamento |
| `categoria` | texto | producto comercial |
| `precio_kg` | decimal | precio promedio mayorista en S/ por kg |
| `n_registros` | entero | registros promediados en la celda |

## Variables derivadas en el análisis

| Variable | Cálculo | Uso |
|---|---|---|
| `log_precio` | `ln(precio_kg)` | transformar escala y analizar cambios relativos; reduce la asimetría observada |
| `cambio_log` | `Δ log_precio` | ECM y modelos |
| `rezago_k` | `log_precio` desplazado k meses | predictores |
| `mediana_otras` | mediana de otros departamentos en el mismo mes | señal de red |
| `distancia_lima_km` | distancia haversine capital departamental–Lima | heterogeneidad |
| `zona` | costa / sierra / selva | heterogeneidad |

## Normalización de unidades

El SISAP publica el precio en la presentación de cada mercado (kilogramo, arroba, saco, caja).
Para cada celda departamento-producto se elige la unidad dominante en el tiempo y se divide el valor
por su equivalencia en kilogramos. Sin este paso, comparar Puno con Lima sería comparar arrobas
con kilos.

## Regla de atípicos

Se descartan valores incompatibles con el entorno del precio, con dos criterios:

1. **Local:** precio / mediana móvil de ±3 meses > 4 o < 0,25.
2. **Nacional:** precio / mediana del mismo producto y mes entre departamentos > 5 o < 0,2.

Se marcan como faltantes; no se imputan. Total: 66 casos (0,15 %). El detalle está en
`data/processed/atipicos_panel.csv`.

## Cobertura

| Periodo | Departamentos | Productos | Densidad de celdas |
|---|---:|---:|---:|
| 1997–1999 | 1 (Lima) | 2–3 | 0,8 % |
| 2000–2001 | 1–22 | 3–7 | 0,9–2,3 % |
| 2002–2007 | 23 | 7–8 | 24–31 % |
| 2008–2011 | 24 | 9 | 44–80 % |
| 2012–2023 | 24 | 9 | 85,7 % |
| 2024–noviembre 2026 | 24 | 7–9 | 66,5 % |

La densidad se calcula respecto del panel completo posible de 24 departamentos × 9 categorías × meses
del periodo. La cobertura observada llega a noviembre de 2026; el holdout predictivo termina en
octubre de 2026, según el corte configurado en `scripts/config.py`. Como no se verificó la fecha de
cierre/publicación de noviembre, se usa octubre por prudencia; noviembre permanece en el panel y
no se declara erróneo ni incompleto.

## Productos

| Categoría | Variedad usada | Meses en Puno | Primer mes |
|---|---|---:|---|
| Papa | Papa canchan | 168 | 2012-03 |
| Cebolla | Cebolla cabeza roja | 160 | 2012-03 |
| Zanahoria | Zanahoria | 151 | 2012-03 |
| Arroz | Arroz superior | 162 | 2010-08 |
| Aceite | Aceite clásico botella 1 L | 182 | 2008-10 |
| Huevos | Huevos rosados | 147 | 2012-03 |
| Tomate | Tomate marzano | 182 | 2008-10 |
| Limón | Limón sutil bolsa | 93 | 2000-01 |
| Azúcar | Azúcar rubia nacional | 213 | 2008-10 |

**Papa amarilla** no es una décima categoría del panel principal. Se usa solo como variedad
alternativa en robustez: tiene 6 meses de cruce con Lima y no permite estimar el ECM.

## Limitaciones

1. El módulo de ciudades no publica volúmenes de abastecimiento.
2. Cambios de unidad y de nombre de variedad a lo largo del tiempo.
3. Series truncadas: tomate en Lima termina en 2014; varias series de Puno en 2023–2024.
4. No hay marca de fecha de publicación: no se puede afirmar un nowcast operativo.
5. Precios mayoristas, no minoristas.
