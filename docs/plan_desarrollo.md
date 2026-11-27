# Plan de desarrollo

**Proyecto:** Análisis exploratorio y modelos predictivos de precios mayoristas en el Perú.

**Objetivo:** construir un dataset reproducible, explorar su estructura, construir variables y
comparar modelos para estimar el cambio del mes corriente y el precio del mes siguiente.

---

## 1. Preguntas e hipótesis

**Preguntas**

1. ¿Cómo se distribuyen, se cubren y se relacionan los precios mayoristas de Lima y los departamentos?
2. ¿Qué modelo estima mejor el cambio del mes corriente y cuál generaliza mejor al pronosticar un mes?

**Hipótesis**

- **H1:** el cambio de precio de Lima del mismo mes contiene información útil para estimar el cambio
  de precio en los departamentos.
- **H2:** el desempeño predictivo es heterogéneo por producto y por horizonte; un modelo bueno para
  el mes corriente no necesariamente sirve para pronosticar el mes siguiente.

El proyecto es de EDA y modelos. No busca demostrar causalidad ni una tesis de integración
estructural.

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

### Ventanas temporales

| Ventana | Uso |
|---|---|
| 1997–2026 | Fuente completa |
| 2012–2023 | EDA y ventana densa |
| 2012–2017 | Entrenamiento de los modelos |
| 2018–2020 | Validación |
| 2021–octubre 2026 | Prueba final |
| Noviembre 2026 | Fuera de la prueba |

El `N` de cada producto y etapa está en `outputs/analisis/06_modelos/n_etapas.csv`.

---

## 3. Estructura del pipeline

```text
scripts/
├── config.py                     rutas, productos, ventanas, hiperparámetros
├── catalogo.py                   catálogo y validación de 24 departamentos
├── modelos.py                    features, modelos y métricas compartidas
├── pipeline/
│   ├── 01_descargar_sisap.py
│   ├── 02_construir_panel.py
│   ├── 03_limpiar_panel.py
│   └── 05_torneo_modelos.py
├── analisis/
│   ├── 01.1_auditoria_fuente.py
│   ├── 02.1_auditoria_panel.py
│   ├── 03.1_verificacion_limpieza.py
│   ├── 03.2_descripcion_datos.py
│   ├── 03.5_estructura_mercados.py
│   ├── 05.1_representacion_modelos.py
│   ├── 05.2_evaluacion_modelos.py
│   └── 05.3_n_etapas.py
├── clasificacion/
│   └── 01_clasificadores_direccion.py
├── informe/
│   ├── 07_resultados_informe.py
│   └── 08_generar_informe.py
└── validar_fase1.py ... validar_fase8.py
```

El material econométrico retirado se conserva, fuera del flujo principal, en
`extras/econometria/`.

Cada fase escribe un manifiesto en `outputs/_manifests/` y llama a un validador. El pipeline se
detiene si la fuente, el panel, la limpieza, el EDA o los modelos dejan de ser coherentes con la
fase anterior.

---

## 4. Decisiones metodológicas

1. **Unidad:** 24 departamentos oficiales; se excluyen Callao y los mercados provinciales.
2. **Limpieza:** unidades normalizadas a S/ kg; atípicos marcados y excluidos, no imputados.
3. **Transformación:** log-precio, cambios, rezagos y armónicos anuales.
4. **Variables:** Lima contemporánea y rezagada, cambios propios rezagados y estacionalidad.
5. **Modelos:** OLS, Ridge y cuatro familias de árboles (Árbol de Decisión, Random Forest, XGBoost y LightGBM).
6. **Selección:** rango promedio de R² en validación, no en la prueba final.
7. **Evaluación:** R², MAE, mejora frente a la referencia, acierto direccional y matriz de confusión.
8. **Integridad:** gates por fase y manifiestos con hashes.

---

## 5. Artefactos

| Artefacto | Script |
|---|---|
| Flujo KDD | `07` |
| Tabla del dataset | `03.2` + `07` |
| Auditoría de panel | `02.1` + `07` |
| Verificación de limpieza | `03.1` + `07` |
| Descriptivos y distribuciones | `03.2` + `07` |
| Correlaciones y geografía | `03.5` + `07` |
| Matriz X/y | `05` + `07` |
| Ranking y prueba final | `05` + `07` |
| Pronóstico a un mes | `05` + `07` |
| Matriz de confusión | `05` + `07` |
| Tamaño de muestra por etapa | `05.3` |
| Representación de modelos | `05.1` |
| Curvas de aprendizaje | `05.2` |

---

## 6. Criterios de éxito

| Criterio | Resultado esperado |
|---|---|
| Cobertura | ≥ 80 % en la ventana densa |
| EDA | tablas y figuras reproducibles |
| Modelos | cuatro familias comparadas |
| Selección | ranking en validación, prueba final intacta |
| Evaluación | R², MAE, acierto y confusión |
| Integridad | gates 1–4, 6–7 aprobados |

---

## 7. Riesgos y límites

| Riesgo | Mitigación |
|---|---|
| Panel no balanceado | declarar `N` por producto y etapa |
| Datos mensuales | no prometer estimación operativa |
| Pocas observaciones | modelos simples y validación temporal |
| Tabla construida por el equipo | documentar variedad, unidad y limpieza |
| Cambios de unidad | normalización y auditoría |
| Pronóstico difícil | comparar con caminata aleatoria |

---

## 8. Entregables

- Dataset final y auditoría.
- Pipeline reproducible con validadores.
- 16 figuras y 17 tablas de resultados.
- Dos cuadernos: resultados ejecutivos y guía.
- README, metodología, diccionario y este plan.
- Informe final de 11 tablas, 11 figuras, 12 ecuaciones, referencias y un anexo con la matriz completa.
