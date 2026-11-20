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
| Unidad | región × producto × mes |
| Moneda | soles corrientes (S/) |

El portal no ofrece una descarga única. La consulta se reproduce con una petición GET a
`/resumenes/filtrar` con `ajax=true`, región por región. Ver `scripts/pipeline/01_descargar_sisap.py`.

En la fuente, una **variedad** es una presentación comercial del producto. El panel final armoniza
esas variedades en 9 **categorías** y toma como unidad una serie región-categoría por mes (243
combinaciones posibles). Las 599 series región-variedad de origen no son 599 series de análisis.

## Archivos

| Archivo | Filas | Contenido |
|---|---:|---|
| `data/raw/ciudades_<código>.html` | 28 | respuesta cruda por región |
| `data/processed/precios_ciudades_largo.csv` | 115 294 | extracción completa |
| `data/processed/panel_precios.csv` | 49 113 | panel tidy normalizado |
| `data/processed/panel_precios_limpio.csv` | 49 041 | panel final sin atípicos |
| `outputs/informe/tablas/` | — | tablas numeradas del informe |
| `outputs/informe/figuras/` | — | figuras numeradas del informe |
| `data/processed/atipicos_panel.csv` | 72 | valores descartados |
| `data/processed/cobertura_panel.csv` | 243 | cobertura por celda (auditoría) |
| `data/processed/unidades_panel.csv` | 243 | unidad dominante por celda |

## Variables del panel

| Variable | Tipo | Descripción |
|---|---|---|
| `fecha` | fecha | primer día del mes |
| `anio` | entero | año |
| `mes` | entero | mes (1–12) |
| `region` | texto | departamento o provincia de mercado |
| `categoria` | texto | producto comercial |
| `precio_kg` | decimal | precio promedio mayorista en S/ por kg |
| `n_registros` | entero | registros promediados en la celda |

## Variables derivadas en el análisis

| Variable | Cálculo | Uso |
|---|---|---|
| `log_precio` | `ln(precio_kg)` | transformar escala y analizar cambios relativos; reduce la asimetría observada |
| `cambio_log` | `Δ log_precio` | ECM y modelos |
| `rezago_k` | `log_precio` desplazado k meses | predictores |
| `mediana_otras` | mediana de otras regiones en el mismo mes | señal de red |
| `distancia_lima_km` | distancia haversine capital regional–Lima | heterogeneidad |
| `zona` | costa / sierra / selva | heterogeneidad |

## Normalización de unidades

El SISAP publica el precio en la presentación de cada mercado (kilogramo, arroba, saco, caja).
Para cada celda región-producto se elige la unidad dominante en el tiempo y se divide el valor
por su equivalencia en kilogramos. Sin este paso, comparar Puno con Lima sería comparar arrobas
con kilos.

## Regla de atípicos

Se descartan valores incompatibles con el entorno del precio, con dos criterios:

1. **Local:** precio / mediana móvil de ±3 meses > 4 o < 0,25.
2. **Nacional:** precio / mediana del mismo producto y mes entre regiones > 5 o < 0,2.

Se marcan como faltantes; no se imputan. Total: 72 casos (0,15 %). El detalle está en
`data/processed/atipicos_panel.csv`.

## Cobertura

| Periodo | Regiones | Productos | Densidad de celdas |
|---|---:|---:|---:|
| 1997–1999 | 1 (Lima) | 2–3 | 0,8 % |
| 2000–2001 | 1–24 | 3–7 | 0,9–3,4 % |
| 2002–2007 | 25 | 7–8 | 24–33 % |
| 2008–2011 | 27 | 9 | 44–80 % |
| 2012–2023 | 27 | 9 | 85,5 % |
| 2024–noviembre 2026 | 27 | 7–9 | 65,8 % |

La densidad se calcula respecto del panel completo posible de 27 regiones × 9 categorías × meses
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
