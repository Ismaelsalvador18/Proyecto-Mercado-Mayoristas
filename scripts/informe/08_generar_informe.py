#!/usr/bin/env python3
"""Genera el informe final de EDA y modelos en formato Word.

El documento se apoya en las tablas y figuras generadas por
07_resultados_informe.py. Las ecuaciones se renderizan con matplotlib y se
insertan como imágenes para que se vean igual en Word y en PDF.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent.parent
INFORME = RAIZ / "outputs" / "informe"
TAB = INFORME / "tablas"
FIG = INFORME / "figuras"
AN = RAIZ / "outputs" / "analisis"
FORMULAS = Path(tempfile.mkdtemp(prefix="formulas_informe_"))
plt.rcParams["mathtext.fontset"] = "stix"


def cargar(nombre: str) -> pd.DataFrame:
    return pd.read_csv(TAB / nombre)


def p(doc: Document, texto: str, negrita: bool = False, cursiva: bool = False) -> None:
    par = doc.add_paragraph()
    run = par.add_run(texto)
    run.bold = negrita
    run.italic = cursiva
    run.font.size = Pt(11)
    par.paragraph_format.space_after = Pt(6)
    par.paragraph_format.line_spacing = 1.15


def vineta(doc: Document, texto: str) -> None:
    par = doc.add_paragraph(style="List Bullet")
    run = par.add_run(texto)
    run.font.size = Pt(10.5)
    par.paragraph_format.space_after = Pt(3)


def salto(doc: Document) -> None:
    doc.add_page_break()


def formula(doc: Document, latex: str, numero: int, ancho: float = 6.2) -> None:
    """Inserta una ecuación numerada como imagen centrada."""
    ruta = FORMULAS / f"ecuacion_{numero:02d}.png"
    fig = plt.figure(figsize=(10, 1.2))
    fig.text(0.5, 0.5, f"${latex}\\qquad ({numero})$",
             ha="center", va="center", fontsize=12)
    fig.savefig(ruta, dpi=300, bbox_inches="tight", pad_inches=0.03,
                facecolor="white")
    plt.close(fig)
    with Image.open(ruta) as imagen:
        ancho_px = imagen.size[0]
    ancho_in = min(ancho, ancho_px / 300)
    doc.add_picture(str(ruta), width=Inches(ancho_in))
    par = doc.paragraphs[-1]
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_before = Pt(4)
    par.paragraph_format.space_after = Pt(5)


def formula_multilinea(doc: Document, lineas: list[str], numero: int,
                      ancho: float = 6.2) -> None:
    """Inserta una ecuación larga en dos líneas, con numeración lateral."""
    ruta = FORMULAS / f"ecuacion_{numero:02d}.png"
    fig = plt.figure(figsize=(10, 1.4))
    for i, linea in enumerate(lineas):
        y = 0.68 if len(lineas) == 2 else 0.5
        if len(lineas) == 2 and i == 1:
            y = 0.28
        fig.text(0.48, y, f"${linea}$", ha="center", va="center", fontsize=12)
    fig.text(0.98, 0.5, f"({numero})", ha="right", va="center", fontsize=11)
    fig.savefig(ruta, dpi=300, bbox_inches="tight", pad_inches=0.03,
                facecolor="white")
    plt.close(fig)
    with Image.open(ruta) as imagen:
        ancho_px = imagen.size[0]
    ancho_in = min(ancho, ancho_px / 300)
    doc.add_picture(str(ruta), width=Inches(ancho_in))
    par = doc.paragraphs[-1]
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_before = Pt(4)
    par.paragraph_format.space_after = Pt(5)


def sombrear_celda(celda, color: str = "D9EAD3") -> None:
    tcPr = celda._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def tabla(
    doc: Document,
    df: pd.DataFrame,
    numero: int | str,
    titulo: str,
    columnas: list[str] | None = None,
    decimales: dict[str, int] | None = None,
    anchos: list[float] | None = None,
    font: float = 8,
    nota: str | None = None,
    mantener_junta: bool = True,
) -> None:
    decimales = decimales or {}
    if columnas:
        df = df[columnas].copy()
    cap = doc.add_paragraph()
    run = cap.add_run(f"Tabla {numero}. ")
    run.bold = True
    run.font.size = Pt(9)
    cap.add_run(titulo).font.size = Pt(9)
    cap.paragraph_format.space_before = Pt(5)
    cap.paragraph_format.space_after = Pt(3)
    cap.paragraph_format.keep_with_next = True

    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    hdr = t.rows[0].cells
    for j, col in enumerate(df.columns):
        hdr[j].text = str(col)
        for par in hdr[j].paragraphs:
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in par.runs:
                run.bold = True
                run.font.size = Pt(font)
        sombrear_celda(hdr[j])

    for _, fila in df.iterrows():
        celdas = t.add_row().cells
        for j, col in enumerate(df.columns):
            valor = fila[col]
            if pd.isna(valor):
                texto = "—"
            elif col in decimales:
                dec = decimales[col]
                try:
                    texto = f"{float(valor):.{dec}f}".replace(".", ",")
                except Exception:
                    texto = str(valor)
            elif isinstance(valor, float) and valor.is_integer():
                texto = f"{int(valor)}"
            else:
                texto = str(valor)
            celdas[j].text = texto
            for par in celdas[j].paragraphs:
                par.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
                for run in par.runs:
                    run.font.size = Pt(font)

    if anchos:
        for j, ancho in enumerate(anchos):
            for row in t.rows:
                row.cells[j].width = Inches(ancho)

    # Repite el encabezado si la tabla ocupa más de una página.
    trPr = t.rows[0]._tr.get_or_add_trPr()
    if trPr.find(qn("w:tblHeader")) is None:
        trPr.append(OxmlElement("w:tblHeader"))

    # Evita que una fila se corte entre dos páginas.
    for i, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr()
        if trPr.find(qn("w:cantSplit")) is None:
            trPr.append(OxmlElement("w:cantSplit"))
        if mantener_junta and i < len(t.rows) - 1:
            for celda in row.cells:
                for par in celda.paragraphs:
                    par.paragraph_format.keep_with_next = True

    if nota:
        for celda in t.rows[-1].cells:
            for par in celda.paragraphs:
                par.paragraph_format.keep_with_next = True
        par = doc.add_paragraph()
        run = par.add_run(nota)
        run.font.size = Pt(8)
        run.italic = True
        par.paragraph_format.space_before = Pt(2)
        par.paragraph_format.space_after = Pt(6)
    else:
        doc.add_paragraph()


def figura(doc: Document, nombre: str, numero: int, titulo: str,
           ancho: float = 6.2, carpeta: Path = FIG) -> None:
    ruta = carpeta / nombre
    if not ruta.exists():
        raise FileNotFoundError(f"No existe la figura: {ruta}")
    doc.add_picture(str(ruta), width=Inches(ancho))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.keep_with_next = True
    cap = doc.add_paragraph()
    run = cap.add_run(f"Figura {numero}. {titulo}")
    run.font.size = Pt(8)
    run.italic = True
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(8)


def configurar(doc: Document) -> None:
    sec = doc.sections[0]
    sec.top_margin = Inches(0.8)
    sec.bottom_margin = Inches(0.8)
    sec.left_margin = Inches(0.9)
    sec.right_margin = Inches(0.9)

    estilo = doc.styles["Normal"]
    estilo.font.name = "Times New Roman"
    estilo.font.size = Pt(11)
    estilo.paragraph_format.line_spacing = 1.15

    for level in [1, 2, 3]:
        st = doc.styles[f"Heading {level}"]
        st.font.name = "Times New Roman"
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.font.bold = True


def portada(doc: Document) -> None:
    for _ in range(5):
        doc.add_paragraph()
    t = doc.add_paragraph()
    run = t.add_run("Precios mayoristas de alimentos en el Perú: análisis exploratorio, modelos matemáticos y validación predictiva")
    run.bold = True
    run.font.size = Pt(19)
    run.font.name = "Times New Roman"
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER

    s = doc.add_paragraph()
    run = s.add_run("Del EDA a la comparación de modelos: 2012–2023, validación 2018–2020 y prueba final 2021–octubre de 2026")
    run.italic = True
    run.font.size = Pt(12)
    s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s.paragraph_format.space_after = Pt(30)

    for texto in ["[Nombre del estudiante]", "[Institución / curso]", "[Docente]"]:
        par = doc.add_paragraph()
        run = par.add_run(texto)
        run.font.size = Pt(12)
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_after = Pt(4)

    par = doc.add_paragraph()
    run = par.add_run("Noviembre de 2026")
    run.font.size = Pt(12)
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_before = Pt(30)


def resumen(doc: Document) -> None:
    doc.add_heading("Resumen", level=1)
    p(doc,
      "Este trabajo analiza los precios mayoristas de alimentos en el Perú y compara seis modelos "
      "de regresión: OLS, Ridge, Árbol de Decisión, Random Forest, XGBoost y LightGBM. A partir del "
      "SISAP–MIDAGRI se construyó un panel "
      "de 24 departamentos y 9 categorías, con 43 855 observaciones limpias; se normalizaron "
      "unidades, se detectaron atípicos y se construyeron cambios logarítmicos, rezagos y armónicos. "
      "La estimación contemporánea y el pronóstico a un mes se evaluaron con entrenamiento "
      "2012–2017, validación 2018–2020 y prueba final 2021–octubre de 2026. La relación con Lima es "
      "heterogénea: la distancia aporta (p = 0,022) y el modelo con distancia y zona explica 54 % de "
      "la variación. Random Forest obtiene el mejor ordenamiento contemporáneo, con R² positivo en "
      "papa, cebolla y zanahoria; Ridge y OLS son mejores para pronosticar un mes en papa, cebolla y "
      "arroz. El torneo de regresión incluye seis modelos y se complementa con una lectura "
      "direccional del signo de la predicción, no con un clasificador aparte. "
      "Conclusiones: Lima aporta información predictiva parcial y el desempeño depende del producto "
      "y del horizonte, por lo que no conviene un modelo único.")
    p(doc, "Palabras clave: precios mayoristas, modelos basados en árboles, Random Forest, XGBoost, validación temporal.",
      cursiva=True)

    doc.add_heading("Abstract", level=1)
    p(doc,
      "This paper studies wholesale food prices in Peru and compares six regression models: OLS, "
      "Ridge, Decision Tree, Random Forest, XGBoost, and LightGBM. Methodology: using SISAP–MIDAGRI "
      "data, we built a panel of 24 "
      "departments and 9 categories with 43,855 clean observations; units were normalized, outliers "
      "were flagged, and log changes, lags, and annual harmonics were constructed. Contemporary "
      "estimation and one-month forecasting were evaluated with 2012–2017 training, 2018–2020 "
      "validation, and a 2021–October 2026 final test. Results: the relationship with Lima is "
      "heterogeneous; distance is significant (p = 0.022) and the distance-plus-zone model explains "
      "54% of the variation. "
      "Random Forest ranks first in contemporary estimation, with positive R² in potato, onion, and "
      "carrot; Ridge and OLS perform better for one-month forecasts in potato, onion, and rice. "
      "The regression tournament includes six models and is complemented with a directional reading "
      "of the prediction sign, not with a separate classifier. "
      "Conclusions: Lima provides partial predictive information and performance depends on product "
      "and horizon, so a single model is not appropriate.")
    p(doc, "Keywords: wholesale prices, tree-based models, Random Forest, XGBoost, temporal validation.",
      cursiva=True)


def introduccion(doc: Document) -> None:
    doc.add_heading("1. Introducción", level=1)
    p(doc,
      "Los precios mayoristas de alimentos conectan la producción con el consumo y son un insumo "
      "directo para el bienestar de los hogares. Su comportamiento espacial se estudia con modelos "
      "de integración de mercados, que relacionan precios de lugares distintos a través de costos "
      "de transferencia y arbitraje (Ravallion, 1986; Fackler y Goodwin, 2001).")
    p(doc,
      "En un país con geografía fragmentada como el Perú, esa relación no puede suponerse uniforme. "
      "Cuando los costos de transferencia son altos, dos mercados pueden mostrar precios distintos "
      "incluso si existe comercio potencial (Baulch, 1997); además, la transmisión puede ser "
      "asimétrica y depender del producto y de la dirección del ajuste (Meyer y von "
      "Cramon-Taubadel, 2004). Por eso el proyecto no impone un coeficiente único: describe la "
      "heterogeneidad y la evalúa con modelos.")
    p(doc,
      "Este proyecto no busca demostrar causalidad ni una tesis de integración estructural. Busca "
      "describir los datos, construir variables, formular los modelos y comparar su capacidad "
      "predictiva con validación temporal. La pregunta práctica es si la referencia de Lima mejora "
      "la estimación del cambio departamental y si esa mejora se mantiene al pronosticar un mes.")

    doc.add_heading("1.1. Preguntas", level=2)
    vineta(doc, "¿Cómo se distribuyen, se cubren y se relacionan los precios mayoristas de Lima y los departamentos?")
    vineta(doc, "¿Qué modelo estima mejor el cambio del mes corriente y cuál generaliza mejor al pronosticar un mes?")

    doc.add_heading("1.2. Hipótesis", level=2)
    vineta(doc, "H1. El cambio de precio de Lima del mismo mes contiene información útil para estimar el cambio de precio en los departamentos.")
    vineta(doc, "H2. El desempeño predictivo es heterogéneo por producto y por horizonte; un modelo bueno para el mes corriente no necesariamente sirve para pronosticar el mes siguiente.")

    doc.add_heading("1.3. Objetivo y alcance", level=2)
    p(doc,
      "Construir un conjunto de datos auditado, describir su estructura, formular las ecuaciones "
      "de transformación y de los modelos, construir la matriz de predictores y comparar seis "
      "modelos de regresión con R², MAE, acierto de signo y matrices de confusión. La selección se hace en "
      "validación; la prueba final se mantiene intacta hasta el cierre.")
    p(doc,
      "El alcance es predictivo y asociativo: no se identifican efectos causales ni se modela el "
      "equilibrio de mercado. La unidad de análisis es el departamento, no el mercado minorista, y "
      "el horizonte es mensual. Estas decisiones acotan la interpretación de los resultados.")

    doc.add_heading("1.4. Organización del documento", level=2)
    p(doc,
      "La sección 2 presenta el flujo KDD y el diseño temporal. La sección 3 describe la fuente, "
      "la normalización y la limpieza. La sección 4 analiza cobertura, distribuciones, correlaciones "
      "y geografía, y cierra con la transición al modelado. La sección 5 construye las variables y "
      "las ventanas. La sección 6 formula los modelos y las métricas. La sección 7 interpreta los "
      "resultados y contrasta las hipótesis. La sección 8 resume conclusiones y límites.")


def metodologia(doc: Document) -> None:
    doc.add_heading("2. Metodología", level=1)
    doc.add_heading("2.1. Enfoque KDD", level=2)
    p(doc,
      "El flujo sigue las cinco fases del descubrimiento de conocimiento en bases de datos "
      "(Fayyad et al., 1996): selección, preprocesamiento, transformación, minería y evaluación. "
      "El análisis exploratorio se apoya en la inspección visual y descriptiva de distribuciones, "
      "cobertura y relaciones antes de modelar (Tukey, 1977). Cada fase deja un artefacto "
      "verificable y la siguiente consume ese artefacto.")
    figura(doc, "01_flujo_kdd.png", 1,
           "Fases del proceso KDD aplicado al estudio.", ancho=6.2)
    p(doc,
      "La selección descarga los datos de SISAP. El preprocesamiento audita nulos, duplicados, "
      "unidades y atípicos. La transformación normaliza a S/ kg, aplica logaritmos y construye "
      "cambios, rezagos y armónicos. La minería compara seis modelos de regresión. La evaluación "
      "usa R², MAE, mejora frente a una referencia y acierto de signo.")

    doc.add_heading("2.2. Diseño temporal y control de filtración", level=2)
    p(doc,
      "La validación es temporal: el pasado entrena, el período medio elige y el tramo final se "
      "reserva para una prueba única. Este diseño evita mezclar información futura con información "
      "pasada y sigue la recomendación de evaluar modelos de series de tiempo con cortes temporales "
      "(Bergmeir y Benítez, 2012).")
    p(doc,
      "El entrenamiento usa 2012–2017; la validación, 2018–2020; y la prueba final, 2021–octubre "
      "de 2026. Ningún rezago, media móvil o parámetro se calcula con observaciones posteriores al "
      "corte. La matriz se construye antes de recortar la ventana para que un mes faltante no se "
      "trate como si fuera el mes anterior. La prueba final no se usa para elegir el modelo.")

    doc.add_heading("2.3. Reproducibilidad y trazabilidad", level=2)
    p(doc,
      "Todas las etapas tienen semillas fijas, un catálogo único de departamentos y validadores que "
      "comparan los artefactos con la fase anterior. Los modelos se ajustan con el mismo conjunto "
      "de predictores y las mismas ventanas. Los resultados se guardan en tablas y figuras "
      "numeradas, de modo que el informe pueda regenerarse sin reescribir cifras a mano.")


def datos(doc: Document) -> None:
    doc.add_heading("3. Datos", level=1)
    doc.add_heading("3.1. Fuente y unidad de análisis", level=2)
    p(doc,
      "La fuente es el módulo Precios de ciudades del Sistema de Información de Abastecimiento y "
      "Precios (SISAP), del Ministerio de Desarrollo Agrario y Riego (MIDAGRI, 2026). El portal "
      "publica precios mayoristas mensuales por departamento y producto desde 1997. La unidad es "
      "departamento × producto × mes.")
    variables = cargar("tabla_01_variables.csv")
    tabla(doc, variables, 1, "Variables del panel de análisis.",
          anchos=[1.0, 0.9, 1.1, 1.8, 1.4],
          nota="Categoría es el producto armonizado; las variedades de la fuente no se cuentan como categorías.")

    doc.add_heading("3.2. Normalización de unidades y transformación logarítmica", level=2)
    p(doc,
      "El SISAP publica el precio en presentaciones distintas: kilogramo, arroba, saco, caja u "
      "otra unidad comercial. Para cada celda departamento-producto se elige la presentación "
      "dominante y se divide el precio reportado por su equivalencia en kilogramos. La operación "
      "se resume en la ecuación 1.")
    formula(doc,
            r"P_{i,t}=\frac{v_{i,t}}{q_{i,t}},\qquad p_{i,t}=\ln(P_{i,t})",
            1, ancho=4.8)
    p(doc,
      "donde v(i,t) es el precio reportado en la presentación de la fuente, q(i,t) su "
      "equivalencia en kilogramos, P(i,t) el precio en soles por kilogramo y p(i,t) su "
      "logaritmo. La normalización evita comparar arrobas con kilos; el logaritmo reduce la "
      "asimetría y hace comparables los cambios relativos.")
    p(doc,
      "La unidad transversal es el departamento. Se usan los 24 departamentos oficiales del país. "
      "SISAP también publica Andahuaylas, Chota, Jaén y Callao; se excluyen porque los tres "
      "primeros son provincias de Apurímac y Cajamarca, y Callao es provincia constitucional. "
      "Incorporarlos duplicaría la unidad de análisis y mezclaría mercados con departamentos.")

    doc.add_heading("3.3. Auditoría y limpieza", level=2)
    auditoria = cargar("tabla_02_auditoria.csv")
    auditoria["valor"] = auditoria["valor"].astype(str).str.replace(".", ",", regex=False)
    tabla(doc, auditoria, 2, "Auditoría del panel.",
          anchos=[3.0, 1.8],
          nota="La cobertura bruta se calcula contra 24 departamentos × 9 categorías × meses del período. La cobertura limpia de 2012–2023 es 85,7 %.")
    p(doc,
      "El panel construido tiene 43 921 filas y no presenta nulos ni duplicados en su clave. La "
      "auditoría bruta reporta 85,9 % de cobertura en 2012–2023; después de descartar 66 valores "
      "(0,15 %), la matriz limpia conserva 43 855 filas y 85,7 % de cobertura. Los descartes "
      "corresponden a errores de punto decimal o precios incompatibles con la mediana local o "
      "nacional. No se imputaron valores: se excluyeron y se reportaron.")
    figura(doc, "23_verificacion_limpieza.png", 2,
           "Efecto de la limpieza sobre la distribución del precio de la cebolla.", ancho=5.8)
    p(doc,
      "El corte de la prueba final es octubre de 2026. Noviembre permanece en el panel porque la "
      "fuente ya lo publica, pero no se evalúa: no se verificó su fecha de cierre y usarlo sería "
      "incorporar un mes potencialmente incompleto.")


def eda(doc: Document) -> None:
    doc.add_heading("4. Análisis exploratorio", level=1)
    doc.add_heading("4.1. Cobertura y truncamiento", level=2)
    p(doc,
      "La cobertura no es un panel balanceado. En 2000 hay 22 de los 24 departamentos y 7 de las "
      "9 categorías; la cobertura se vuelve densa a partir de 2012. La ventana 2012–2023 tiene "
      "85,7 % de celdas completas. Las series truncadas y los meses faltantes se declaran y no se "
      "rellenan.")
    figura(doc, "02_cobertura.png", 3,
           "Cobertura anual del panel analítico.", ancho=5.8)

    doc.add_heading("4.2. Distribución y transformación logarítmica", level=2)
    p(doc,
      "Los precios son asimétricos a la derecha. La Tabla 3 resume media, mediana, dispersión y "
      "forma de la distribución para los nueve productos.")
    descriptivos = cargar("tabla_03_descriptivos.csv")
    tabla(doc, descriptivos, 3, "Estadísticos descriptivos del precio por producto.",
          decimales={"n": 0, "media": 2, "desv": 2, "min": 2, "p25": 2, "mediana": 2,
                     "p75": 2, "max": 2, "asimetria": 2, "curtosis": 2},
          anchos=[0.9, 0.45, 0.55, 0.5, 0.45, 0.45, 0.55, 0.45, 0.45, 0.55, 0.55],
          font=7)
    p(doc,
      "La Figura 4 muestra las distribuciones en la escala original. La Figura 5 repite el mismo "
      "ejercicio con el logaritmo del precio. La comparación es intencional: en limón, zanahoria y "
      "cebolla, las colas derechas de la Figura 4 se comprimen en la Figura 5 y las distribuciones "
      "se acercan a una forma más simétrica. Por eso los modelos no trabajan con el nivel del "
      "precio, sino con cambios logarítmicos.")
    figura(doc, "03_campanas_precio.png", 4,
           "Distribución del precio por producto en la escala original.", ancho=6.1)
    figura(doc, "04_campanas_log.png", 5,
           "Distribución del logaritmo del precio por producto.", ancho=6.1)

    doc.add_heading("4.3. Correlación de cambios con Lima", level=2)
    p(doc,
      "Para medir la relación mensual entre Lima y cada departamento se usa la correlación de "
      "Pearson sobre los cambios logarítmicos. La ecuación 2 define el cambio y la ecuación 3 la "
      "correlación entre Lima y el departamento j.")
    formula(doc,
            r"\Delta p_{i,t}=p_{i,t}-p_{i,t-1}",
            2, ancho=3.2)
    formula(doc,
            r"\rho_{j}=\frac{\sum_{t}(\Delta p_{L,t}-\overline{\Delta p_{L}})(\Delta p_{j,t}-\overline{\Delta p_{j}})}{\sqrt{\sum_{t}(\Delta p_{L,t}-\overline{\Delta p_{L}})^{2}\sum_{t}(\Delta p_{j,t}-\overline{\Delta p_{j}})^{2}}}",
            3, ancho=6.2)
    p(doc,
      "donde Δp(L,t) es el cambio de Lima en el mes t, Δp(j,t) el cambio "
      "del departamento j y las barras indican el promedio de la ventana. La correlación es "
      "descriptiva: no identifica causalidad ni corrige por otras variables.")
    p(doc,
      "La relación entre departamentos no es uniforme. Para papa, Lima se mueve con Ica (0,85) y "
      "Huánuco (0,77), y casi nada con Tumbes (−0,03) o Loreto (−0,01). La Figura 6 muestra la "
      "correlación con Lima por producto y departamento. El patrón no se repite entre productos: "
      "una misma región puede estar integrada en papa y desconectada en aceite.")
    figura(doc, "07_correlacion_lima.png", 6,
           "Correlación de cambios mensuales con Lima por producto y departamento.", ancho=6.1)

    doc.add_heading("4.4. Geografía de la integración", level=2)
    p(doc,
      "Para resumir la integración de cada departamento se promedian sus correlaciones con Lima "
      "en los seis productos del núcleo. El promedio se calcula en la escala z de Fisher, que "
      "evita que correlaciones extremas dominen la media, y luego se devuelve a la escala de "
      "correlación con la ecuación 4.")
    formula(doc,
            r"z_{j}=\mathrm{artanh}(\rho_{j}),\qquad r_{j}=\tanh(\bar{z}_{j})",
            4, ancho=4.8)
    p(doc,
      "donde ρ(j) es la correlación del departamento j con Lima en un producto, "
      "z̄(j) su promedio en la escala transformada y r(j) la correlación promedio "
      "que se usa en la Figura 7 y en la Tabla 4.")
    p(doc,
      "La relación con la distancia y la zona natural se estima con la ecuación 5. La distancia "
      "entra en logaritmos porque el efecto de un kilómetro adicional no es lineal; las categorías "
      "de zona se incluyen como variables dicotómicas con Costa como referencia.")
    formula(doc,
            r"r_{j}=\alpha+\beta\ln(d_{j})+\gamma_{2}\mathrm{Selva}_{j}+\gamma_{3}\mathrm{Sierra}_{j}+u_{j}",
            5, ancho=6.0)
    p(doc,
      "donde d(j) es la distancia entre la capital departamental y Lima, y "
      "Selva(j), Sierra(j) identifican la zona natural. La ecuación 5 "
      "es descriptiva: explica cómo varía la integración entre departamentos, no cómo cambia un "
      "precio mes a mes.")
    figura(doc, "08_integracion_distancia.png", 7,
           "Integración con Lima y distancia, por departamento.", ancho=5.8)
    geo = pd.read_csv(AN / "04_estructura" / "geografia_modelos.csv")
    geo["modelo"] = geo["modelo"].replace({
        "distancia": "Distancia",
        "distancia_zona": "Distancia + zona"})
    geo["termino"] = geo["termino"].replace({
        "log_distancia": "log(distancia)",
        "Selva": "Selva",
        "Sierra": "Sierra"})
    geo = geo.rename(columns={
        "modelo": "Modelo", "termino": "Término", "coef": "Coeficiente",
        "p": "p", "r2": "R²"})
    tabla(doc, geo, 4, "Modelos de integración con Lima y geografía.",
          columnas=["Modelo", "Término", "Coeficiente", "p", "R²"],
          decimales={"Coeficiente": 3, "p": 3, "R²": 3},
          anchos=[1.4, 1.3, 0.9, 0.7, 0.6],
          font=7.8,
          nota="Variable dependiente: correlación promedio con Lima por departamento (23 departamentos sin Lima). Categoría de referencia: Costa.")
    p(doc,
      "La distancia es significativa (p = 0,022) y, por sí sola, explica 22 % de la variación. La "
      "prueba de Kruskal-Wallis no encuentra diferencias por zona (p = 0,196), pero al controlar "
      "por distancia las dummies de Selva y Sierra son significativas y el R² sube a 54 %. La "
      "geografía importa, aunque no explica todo: Arequipa está lejos y está integrada; Piura y "
      "Tumbes son costa y están desconectadas.")

    doc.add_heading("4.5. Síntesis del EDA y transición al modelado", level=2)
    p(doc,
      "El EDA deja cinco hechos que condicionan el modelo. Primero, los precios son asimétricos y "
      "el logaritmo estabiliza la escala; por eso el análisis trabaja con cambios logarítmicos. "
      "Segundo, la cobertura es heterogénea y el N cambia por producto y por etapa; por eso cada "
      "modelo declara su muestra efectiva. Tercero, la correlación con Lima no es uniforme entre "
      "productos ni entre departamentos; por eso no se impone un coeficiente único. Cuarto, la "
      "distancia y la zona explican una parte de la heterogeneidad, pero no toda; la geografía "
      "sirve para describir el contexto, no para reemplazar la información mensual. Quinto, los "
      "productos tienen dinámicas distintas; por eso se comparan modelos por producto y no se "
      "promete un modelo universal.")
    p(doc,
      "Estos hechos llevan de forma natural a un problema predictivo: si Lima y los rezagos "
      "contienen información sobre el cambio departamental, un modelo supervisado debería "
      "aprovecharla; si la relación cambia por producto, el desempeño también debería cambiar. La "
      "siguiente sección construye la matriz de predictores y las ventanas temporales que permiten "
      "contrastar esa idea.")


def variables(doc: Document) -> None:
    doc.add_heading("5. Variables y ventanas temporales", level=1)
    doc.add_heading("5.1. Variable objetivo y predictores", level=2)
    p(doc,
      "Para cada producto se construye una tabla mensual. La variable objetivo es el cambio "
      "logarítmico de Puno, Δp(P,t). Los predictores son el cambio de Lima "
      "contemporáneo y sus tres primeros rezagos, los dos primeros rezagos propios y dos armónicos "
      "anuales. La ecuación 6 formaliza la estimación del mes corriente.")
    formula_multilinea(doc, [
        r"\Delta p_{P,t}=\beta_{0}+\beta_{1}\Delta p_{L,t}+\sum_{k=1}^{3}\gamma_{k}\Delta p_{L,t-k}+\sum_{j=1}^{2}\delta_{j}\Delta p_{P,t-j}",
        r"+\alpha_{1}\sin\left(\frac{2\pi t}{12}\right)+\alpha_{2}\cos\left(\frac{2\pi t}{12}\right)+\varepsilon_{t}",
    ], 6, ancho=6.2)
    p(doc,
      "donde P identifica a Puno, L a Lima, k y j indexan los rezagos, los "
      "términos sin y cos capturan la estacionalidad anual y ε(t) es el "
      "error. El modelo se estima por producto; no se apilan los seis productos en una sola "
      "regresión porque los coeficientes pueden cambiar entre ellos.")
    p(doc,
      "La Tabla 5 resume las variables de las dos matrices de diseño. La matriz completa de papa "
      "en Puno, con sus 155 filas y sus ocho predictores, se reproduce en el Anexo A. Las matrices "
      "de los otros productos tienen la misma estructura y su tamaño se informa en la Tabla 6.")
    matrices = pd.DataFrame([
        ("Mes corriente", "dl0, dl1, dl2, dl3, dp1, dp2, sin1, cos1", "Δp(Puno, t)"),
        ("Pronóstico", "p0, p1, l0, l1, m0, sin1, cos1", "ln P(Puno, t+1)"),
    ], columns=["Tarea", "Predictores", "Objetivo"])
    tabla(doc, matrices, 5, "Variables de las dos matrices de diseño.",
          anchos=[1.2, 4.0, 1.3], font=7.5,
          nota="dl: cambio de Lima; dp: cambio propio de Puno; p: nivel propio; l: nivel de Lima; m: mediana de los otros 22 departamentos; sin y cos: armónicos anuales.")
    p(doc,
      "La matriz se construye antes de recortar la ventana, para que un mes faltante no se trate "
      "como si fuera el mes anterior. La matriz completa de papa en Puno tiene 155 filas: 67 de "
      "entrenamiento, 28 de validación y 60 de prueba final. Las filas del Anexo A son las que "
      "alimentan la ecuación 6 y la columna y es el cambio logarítmico de Puno.")

    doc.add_heading("5.2. Pronóstico a un mes", level=2)
    p(doc,
      "El pronóstico a un mes cambia de escala: la variable objetivo es el nivel del logaritmo "
      "del precio de Puno en t+1, no el cambio. Los predictores son el nivel propio, el nivel "
      "de Lima y la mediana de los otros 22 departamentos en t. La ecuación 7 resume el modelo.")
    formula_multilinea(doc, [
        r"\ln P_{P,t+1}=\beta_{0}+\beta_{1}\ln P_{P,t}+\beta_{2}\ln P_{L,t}+\beta_{3}\ln\bar{P}_{otras,t}",
        r"+\alpha_{1}\sin\left(\frac{2\pi t}{12}\right)+\alpha_{2}\cos\left(\frac{2\pi t}{12}\right)+\varepsilon_{t+1}",
    ], 7, ancho=6.2)
    p(doc,
      "donde P̄(otras,t) es la mediana de los precios de los departamentos distintos de "
      "Puno y Lima. La mediana funciona como una señal de red y evita que un departamento extremo "
      "domine el pronóstico. El modelo se evalúa contra la caminata aleatoria, que usa el precio "
      "de t como predicción de t+1.")

    doc.add_heading("5.3. Ventanas y tamaño de muestra", level=2)
    p(doc,
      "Los tres cortes temporales no son estudios independientes: son una partición de la misma "
      "serie. El pasado entrena, el período medio valida y el final prueba. El N cambia por "
      "faltantes, series truncadas, productos ausentes y rezagos.")
    n_etapas = pd.read_csv(AN / "06_modelos" / "n_etapas.csv")
    n_etapas["tarea"] = n_etapas["tarea"].replace({
        "estimación del mes corriente": "mes corriente",
        "pronóstico a un mes": "pronóstico"})
    n_etapas = n_etapas.rename(columns={
        "entrenamiento_2012_2017": "Entren. 2012–2017",
        "validacion_2018_2020": "Valid. 2018–2020",
        "prueba_2021_2026": "Prueba 2021–2026"})
    tabla(doc, n_etapas, 6, "Observaciones por producto y etapa.",
          columnas=["producto", "tarea", "Entren. 2012–2017", "Valid. 2018–2020", "Prueba 2021–2026"],
          decimales={"Entren. 2012–2017": 0, "Valid. 2018–2020": 0, "Prueba 2021–2026": 0},
          anchos=[0.9, 1.2, 1.25, 1.2, 1.2], font=8,
          nota="Entrenamiento 2012–2017; validación 2018–2020; prueba final 2021–octubre de 2026. Noviembre de 2026 permanece en el panel pero no se evalúa.")
    p(doc,
      "El N no es el número de meses calendario: es el número de meses con datos completos "
      "después de construir cambios y rezagos. Por eso papa y cebolla tienen más observaciones que "
      "arroz o aceite en la prueba final, y por eso el ranking de validación solo incluye los "
      "productos que alcanzan los mínimos de muestra.")


def modelos(doc: Document) -> None:
    doc.add_heading("6. Modelos y evaluación", level=1)
    doc.add_heading("6.1. De la heterogeneidad a la comparación de modelos", level=2)
    p(doc,
      "El EDA mostró que la relación con Lima cambia por producto y que la serie tiene "
      "estacionalidad, rezagos y valores atípicos. Esas características justifican comparar seis "
      "modelos de regresión: dos referencias lineales (OLS y Ridge) y cuatro modelos de árbol "
      "(Árbol de Decisión, Random Forest, XGBoost y LightGBM). La comparación no supone que el "
      "modelo más flexible sea mejor: el rendimiento debe medirse fuera de la muestra (James et "
      "al., 2021).")

    doc.add_heading("6.2. Modelos lineales: OLS y Ridge", level=2)
    p(doc,
      "OLS es la referencia lineal. Ridge añade una penalización sobre los coeficientes para "
      "reducir la varianza cuando los predictores están correlacionados (Hoerl y Kennard, 1970). "
      "Ambos tienen solución cerrada, que se escribe en la ecuación 8.")
    formula(doc,
            r"\hat{\beta}_{OLS}=(X'X)^{-1}X'y,\qquad \hat{\beta}_{Ridge}=(X'X+\lambda I)^{-1}X'y",
            8, ancho=6.0)
    p(doc,
      "donde X es la matriz de predictores, y el vector objetivo, I la identidad y "
      "λ la intensidad de la penalización. En este trabajo λ = 1 y Ridge "
      "estandariza los predictores antes de estimar; OLS es el caso sin penalización.")

    doc.add_heading("6.3. Árbol de Decisión", level=2)
    p(doc,
      "El Árbol de Decisión es la referencia interpretable de particiones recursivas (Breiman et "
      "al., 1984). Divide el espacio de predictores en regiones y predice una constante en cada "
      "hoja. Se usa profundidad máxima 5 y hoja mínima de 5 observaciones. Su ventaja es la "
      "transparencia; su límite, la alta varianza cuando el árbol crece demasiado.")

    doc.add_heading("6.4. Random Forest", level=2)
    p(doc,
      "Random Forest combina muchos árboles entrenados sobre muestras bootstrap y promedia sus "
      "predicciones (Breiman, 2001). La ecuación 9 representa el predictor agregado.")
    formula(doc,
            r"\hat{y}(x)=\frac{1}{B}\sum_{b=1}^{B}T_{b}(x)",
            9, ancho=3.8)
    p(doc,
      "donde B es el número de árboles y T(b,x) la predicción del árbol b. Se usan "
      "300 árboles y una hoja mínima de tres observaciones. El promedio reduce la varianza de un "
      "árbol individual, aunque no elimina el sesgo ni garantiza que el modelo generalize a otro "
      "período.")

    doc.add_heading("6.5. XGBoost", level=2)
    p(doc,
      "XGBoost es un boosting de gradiente con regularización (Chen y Guestrin, 2016). A diferencia "
      "del boosting clásico, penaliza la complejidad de los árboles y usa derivadas de segundo orden. "
      "Se usan 300 árboles, tasa de aprendizaje 0,05, profundidad 4 y submuestreo de filas y "
      "columnas. Es una alternativa a LightGBM: ambos optimizan una función de pérdida aditiva, "
      "pero difieren en la forma de construir y podar los árboles.")

    doc.add_heading("6.6. LightGBM", level=2)
    p(doc,
      "LightGBM usa boosting de árboles (Ke et al., 2017), una familia que construye el modelo "
      "corrigiendo errores de forma secuencial (Friedman, 2001). La ecuación 10 resume la "
      "actualización aditiva.")
    formula(doc,
            r"\hat{y}_{i}^{(m)}=\hat{y}_{i}^{(m-1)}+\nu f_{m}(x_{i})",
            10, ancho=4.2)
    p(doc,
      "donde m es la iteración, f(m) el árbol ajustado al gradiente del error y ν "
      "la tasa de aprendizaje. Se usan 300 árboles, tasa 0,05 y siete hojas. El modelo puede "
      "capturar no linealidades, pero también puede sobreajustar cuando el N es pequeño.")

    doc.add_heading("6.7. Métricas de evaluación", level=2)
    p(doc,
      "La métrica principal de la estimación contemporánea es el R², porque mide cuánta variación "
      "del cambio observado se explica. La ecuación 11 lo define.")
    formula(doc,
            r"R^{2}=1-\frac{\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^{2}}{\sum_{i=1}^{n}(y_{i}-\bar{y})^{2}}",
            11, ancho=5.0)
    p(doc,
      "donde y(i) es el valor observado, ŷ(i) la predicción y ȳ el "
      "promedio de la muestra de evaluación. Un R² negativo indica que el modelo predice peor que "
      "usar el promedio; no es un error de cálculo, sino un resultado posible en muestras "
      "volátiles.")
    p(doc,
      "Para el pronóstico se usa el MAE, porque está en la escala del precio y es fácil de "
      "interpretar. La ecuación 12 también define la mejora frente a una referencia.")
    formula(doc,
            r"\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_{i}-\hat{y}_{i}|,\qquad \mathrm{Mejora}=100\left(1-\frac{\mathrm{MAE}_{m}}{\mathrm{MAE}_{ref}}\right)",
            12, ancho=6.0)
    p(doc,
      "donde MAE(m) es el error del modelo y MAE(ref) el de la "
      "referencia. La referencia de la estimación contemporánea es predecir cambio cero; la del "
      "pronóstico es la caminata aleatoria. Un valor positivo de la mejora indica que el modelo "
      "reduce el error; uno negativo indica que lo aumenta. La lectura de errores y la comparación "
      "con una referencia simple siguen la práctica de evaluación de pronósticos (Hyndman y "
      "Athanasopoulos, 2021), y el MAE se reporta porque es interpretable en la escala del precio "
      "(Hyndman y Koehler, 2006).")
    p(doc,
      "El acierto direccional complementa las métricas anteriores. Se calcula sobre los meses con "
      "cambio distinto de cero y compara el signo de la predicción con el signo observado. Es una "
      "lectura binaria de la regresión, no un clasificador: no distingue un error de 0,1 % de uno "
      "de 5 %, por eso no se usa como criterio único de selección. La matriz de confusión de la "
      "sección 7.5 tampoco entrena una clase; solo cuenta coincidencias de signo entre la "
      "predicción continua y el valor observado. No debe interpretarse como un modelo de "
      "subida/bajada.")

    doc.add_heading("6.8. Protocolo de selección", level=2)
    p(doc,
      "Los seis modelos se entrenan con 2012–2017 y se evalúan en 2018–2020. El ganador se elige "
      "por el rango promedio de R² en validación, no por el mejor resultado de un solo producto. La "
      "prueba final se usa una sola vez para comprobar si la señal se sostiene. Cambiar el ganador "
      "después de ver la prueba final sería seleccionar sobre el conjunto de prueba.")


def resultados(doc: Document) -> None:
    doc.add_heading("7. Resultados", level=1)
    doc.add_heading("7.1. Validación: orden relativo de los modelos", level=2)
    ranking = cargar("tabla_08_torneo.csv").rename(columns={
        "modelo": "Modelo", "rank_promedio": "Rango", "r2_medio": "R² medio",
        "r2_mediano": "R² mediano", "acierto_medio": "Acierto de signo (%)",
        "productos_rankeados": "Productos"})
    ranking["Modelo"] = ranking["Modelo"].replace({"Arbol": "Árbol de Decisión"})
    tabla(doc, ranking, 7, "Ranking de estimación contemporánea en validación.",
          columnas=["Modelo", "Rango", "R² medio", "R² mediano", "Acierto de signo (%)", "Productos"],
          decimales={"Rango": 2, "R² medio": 3, "R² mediano": 3, "Acierto de signo (%)": 1, "Productos": 0},
          anchos=[1.2, 0.55, 0.75, 0.8, 1.0, 0.6], font=7.5)
    p(doc,
      "El ranking se calcula solo con los cuatro productos que alcanzan los mínimos de muestra en "
      "validación. En validación, todos los R² son negativos. Random Forest ocupa el primer lugar "
      "por rango promedio (2,00), seguido por Ridge (2,25), XGBoost (3,25), OLS (3,50), Árbol de "
      "Decisión (4,25) y LightGBM (5,75). La diferencia entre los dos primeros es pequeña y el "
      "nivel absoluto no permite hablar de un ajuste alto.")
    p(doc,
      "El resultado debe leerse como un orden relativo. La validación 2018–2020 fue un período "
      "volátil y con pocas observaciones por producto; en ese contexto, un R² negativo indica que "
      "los cambios mensuales son difíciles de superar con el promedio. Random Forest es el menos "
      "castigado, pero no un modelo con ajuste alto. La comparación sirve para elegir una familia, "
      "no para afirmar que el problema está resuelto.")
    figura(doc, "11_ranking_torneo.png", 8,
           "Ranking del torneo de estimación contemporánea.", ancho=5.6)

    doc.add_heading("7.2. Prueba final de la estimación contemporánea", level=2)
    holdout = cargar("tabla_09_ganador.csv")
    holdout = holdout[(holdout["periodo"] == "holdout") & (holdout["modelo"] == "Random Forest")]
    holdout = holdout.rename(columns={"producto": "Producto", "r2": "R²", "acierto": "Acierto", "n": "N"})
    tabla(doc, holdout, 8, "Prueba final de Random Forest por producto.",
          columnas=["Producto", "R²", "Acierto", "N"],
          decimales={"R²": 3, "Acierto": 1, "N": 0},
          anchos=[1.2, 0.8, 0.9, 0.6], font=7.5,
          nota="Acierto calculado solo sobre meses con cambio observado distinto de cero.")
    p(doc,
      "La prueba final es la que muestra si la señal se sostiene. Random Forest obtiene R² "
      "positivo en papa (0,270), cebolla (0,218) y zanahoria (0,080); en arroz, aceite y huevos el "
      "R² es negativo. La señal es contemporánea y específica por producto: el modelo encuentra "
      "información útil en los tres primeros, pero no en los segundos.")
    p(doc,
      "La diferencia entre productos es sustantiva. En papa y cebolla, Lima y los rezagos parecen "
      "resumir una parte relevante del movimiento mensual; en zanahoria la señal existe, pero es "
      "débil. En arroz, aceite y huevos el R² negativo sugiere que el movimiento local no se "
      "explica bien con Lima y la persistencia. Es una hipótesis, no una conclusión causal: esos "
      "productos pueden estar más expuestos a costos internacionales, regulación o estructuras de "
      "mercado distintas.")
    figura(doc, "12_ganador_vs_observado.png", 9,
           "Cambio observado y predicciones de los seis modelos en la prueba final.", ancho=6.2)

    doc.add_heading("7.3. Pronóstico a un mes", level=2)
    pronostico = cargar("tabla_23_pronostico_holdout.csv").rename(columns={
        "producto": "Producto", "n": "N", "OLS": "OLS (%)", "Ridge": "Ridge (%)",
        "Random Forest": "Random Forest (%)", "LightGBM": "LightGBM (%)"})
    tabla(doc, pronostico, 9, "Pronóstico a un mes: mejora porcentual del MAE frente a la caminata aleatoria.",
          columnas=["Producto", "N", "OLS (%)", "Ridge (%)", "Random Forest (%)", "LightGBM (%)"],
          decimales={"N": 0, "OLS (%)": 1, "Ridge (%)": 1, "Random Forest (%)": 1, "LightGBM (%)": 1},
          anchos=[1.0, 0.45, 0.7, 0.7, 1.15, 0.95], font=7,
          nota="Positivo = mejora; negativo = empeora. La referencia es la caminata aleatoria.")
    p(doc,
      "El pronóstico a un mes tiene otra lógica. Ridge y OLS mejoran el MAE en papa (12,7 % y "
      "14,5 %), cebolla (9,9 % y 8,3 %) y arroz (7,7 % y 3,9 %); ambos empeoran en zanahoria, "
      "aceite y huevos. Random Forest y LightGBM no superan la caminata aleatoria en los seis "
      "productos: sus errores son mucho mayores, sobre todo en arroz, aceite y huevos.")
    p(doc,
      "La explicación más plausible es que la caminata aleatoria es un rival fuerte cuando el "
      "precio tiene persistencia. Los modelos lineales aprovechan esa persistencia con pocos "
      "parámetros; los árboles, con muestras pequeñas, tienen más dificultad para estimar una "
      "función estable y tienden a sobreajustar el entrenamiento. El resultado es consistente con "
      "H2: el mejor modelo para el mes corriente no es el mejor para el mes siguiente.")
    p(doc,
      "La heterogeneidad por producto y horizonte también es consistente con la literatura de "
      "transmisión de precios, que documenta respuestas distintas según el mercado y el producto "
      "(Meyer y von Cramon-Taubadel, 2004). No hay un modelo único para los seis productos: la "
      "decisión práctica depende de la tarea y del producto.")

    doc.add_heading("7.4. Comparación de los seis modelos de regresión", level=2)
    p(doc,
      "El torneo reúne los seis modelos con los mismos predictores y las mismas ventanas: OLS, "
      "Ridge, Árbol de Decisión, Random Forest, XGBoost y LightGBM. OLS y Ridge son las "
      "referencias lineales; los cuatro restantes cubren desde un árbol único hasta boosting con "
      "regularización (Breiman et al., 1984; Chen y Guestrin, 2016; Ke et al., 2017). La tabla 10 "
      "resume magnitud y acierto de signo; no es una tabla de clasificación.")
    nowcast = pd.read_csv(AN / "06_modelos" / "torneo_nowcast_metricas.csv")
    nowcast = nowcast[nowcast.conjunto == "principal"]
    val = nowcast[nowcast.periodo == "desarrollo"].groupby("modelo").agg(
        r2_val=("r2", "mean"), mae_val=("mae", "mean"), acierto_val=("acierto", "mean"))
    prueba = nowcast[nowcast.periodo == "holdout"].groupby("modelo").agg(
        r2_prueba=("r2", "mean"), mae_prueba=("mae", "mean"), acierto_prueba=("acierto", "mean"))
    seis = val.join(prueba).reindex(["OLS", "Ridge", "Arbol", "Random Forest", "XGBoost", "LightGBM"])
    seis.index = ["OLS", "Ridge", "Árbol de Decisión", "Random Forest", "XGBoost", "LightGBM"]
    seis = seis.reset_index().rename(columns={
        "index": "Modelo", "r2_val": "R² validación", "mae_val": "MAE validación",
        "r2_prueba": "R² prueba", "mae_prueba": "MAE prueba",
        "acierto_val": "Acierto validación (%)", "acierto_prueba": "Acierto prueba (%)"})
    tabla(doc, seis, 10, "Comparación de los seis modelos de regresión.",
          decimales={"R² validación": 3, "MAE validación": 4, "R² prueba": 3,
                     "MAE prueba": 4, "Acierto validación (%)": 1,
                     "Acierto prueba (%)": 1},
          anchos=[1.25, 0.85, 0.85, 0.8, 0.8, 0.95, 0.9], font=7,
          nota="Promedios simples sobre los seis productos. El acierto de signo se calcula sobre meses con cambio distinto de cero; no es una clasificación entrenada.")
    p(doc,
      "En validación, Ridge tiene el mejor R² promedio (-0,661), seguido por Random Forest (-0,687) "
      "y OLS (-0,689); XGBoost logra el mejor acierto de signo (55,0 %). En la prueba final, Ridge "
      "(-0,020) y OLS (-0,024) conservan los mejores R² promedio, mientras que LightGBM alcanza el "
      "mejor acierto (61,7 %). El ranking por producto de la tabla 7 sigue favoreciendo a Random "
      "Forest porque promedia posiciones, no magnitudes: Ridge gana en promedio pero no es el más "
      "consistente en todos los productos.")
    p(doc,
      "La comparación no mezcla escalas: R² y MAE miden magnitud; el acierto de signo solo indica "
      "cuántas veces la predicción continua apunta en la dirección correcta. Un modelo puede acertar "
      "el signo y equivocarse en la magnitud, o explicar mejor la varianza y fallar en los meses de "
      "subida. Por eso el ganador se elige por R² en validación y el acierto se reporta como "
      "diagnóstico complementario.")

    doc.add_heading("7.5. Matriz de confusión de signo de la regresión", level=2)
    p(doc,
      "La matriz de confusión de la tabla 11 y la figura 10 no entrenan un clasificador. Se "
      "construyen tomando el signo de la predicción continua de cada modelo de regresión y "
      "comparándolo con el signo del cambio observado, después de excluir los meses con cambio "
      "exactamente cero. Sirven para leer cuántas subidas y bajadas acierta cada regresor, pero la "
      "decisión sigue siendo una predicción de magnitud, no una probabilidad de clase. No deben "
      "interpretarse como un modelo de subida/bajada.")
    confusion = pd.read_csv(AN / "06_modelos" / "matriz_confusion_modelos.csv")
    confusion = confusion.set_index("modelo").reindex(
        ["OLS", "Ridge", "Arbol", "Random Forest", "XGBoost", "LightGBM"]).reset_index()
    confusion = confusion.rename(columns={
        "modelo": "Modelo", "TP": "Acierta +", "FN": "Falla +",
        "FP": "Falso +", "TN": "Acierta −", "n": "N", "acierto": "Coincidencia (%)"})
    confusion["Modelo"] = confusion["Modelo"].replace({
        "Arbol": "Árbol de Decisión", "Random Forest": "Random Forest",
        "XGBoost": "XGBoost", "LightGBM": "LightGBM"})
    tabla(doc, confusion, 11, "Matriz de confusión de signo de los seis modelos de regresión.",
          columnas=["Modelo", "Acierta +", "Falla +", "Falso +",
                    "Acierta −", "N", "Coincidencia (%)"],
          decimales={"Coincidencia (%)": 1, "N": 0},
          anchos=[1.3, 0.85, 0.8, 0.8, 0.85, 0.5, 1.0], font=7,
          nota="Conteos sobre la prueba final, excluyendo cambios observados iguales a cero. Acierta +: predicción y observación positivas; Falla +: observación positiva no detectada; Falso +: predicción positiva con observación no positiva; Acierta −: ambas no positivas. No es un clasificador.")
    figura(doc, "24_matriz_confusion.png", 10,
           "Coincidencia de signo de los seis modelos de regresión en la prueba final.", ancho=6.2)
    p(doc,
      "La lectura es directa: LightGBM acierta 62,2 % de los signos, OLS y Ridge 59,2 %, Random "
      "Forest 58,0 %, XGBoost 57,6 % y el Árbol de Decisión 57,3 %. Esas diferencias no cambian el "
      "ganador del torneo, que se eligió por R² en validación: el acierto de signo trata igual un "
      "error de 0,1 % que uno de 5 % y no mide la calidad de la magnitud. La matriz sirve para "
      "interpretar la utilidad práctica del regresor, no para seleccionarlo.")

    doc.add_heading("7.6. Contraste de hipótesis y síntesis", level=2)
    p(doc,
      "H1 se sostiene parcialmente. Lima contiene información contemporánea sobre los "
      "departamentos, pero la magnitud y la estabilidad cambian por producto: la evidencia es "
      "clara en papa y cebolla, débil en zanahoria y negativa en arroz, aceite y huevos. H2 se "
      "sostiene con claridad: el desempeño predictivo es heterogéneo por producto y por horizonte. "
      "Un modelo bueno para el mes corriente no necesariamente sirve para pronosticar el mes "
      "siguiente.")
    p(doc,
      "La síntesis práctica es que no conviene desplegar un solo modelo para todos los productos. "
      "Para la estimación contemporánea, Random Forest es el mejor ordenamiento relativo, con "
      "señal en tres productos; OLS y Ridge quedan como referencias lineales con mejor R² promedio "
      "en la prueba, pero sin el mismo orden por producto. Para el pronóstico a un mes, Ridge u OLS "
      "son preferibles en papa, cebolla y arroz, y la caminata aleatoria sigue siendo una referencia "
      "difícil de superar en los demás. La selección debe hacerse por producto y por tarea.")


def conclusiones(doc: Document) -> None:
    doc.add_heading("8. Conclusiones y límites", level=1)
    doc.add_heading("8.1. Conclusiones", level=2)
    p(doc,
      "El proyecto construyó un panel auditado de precios mayoristas, documentó la normalización "
      "de unidades y la limpieza, y formuló de manera explícita las ecuaciones de transformación, "
      "de los modelos y de las métricas. Esa formalización permite reproducir el análisis y "
      "distinguir lo que se estima de lo que solo se describe.")
    p(doc,
      "La relación con Lima es real pero heterogénea. La geografía ayuda a entender por qué la "
      "integración no es uniforme: la distancia y la zona explican 54 % de la variación entre "
      "departamentos. Sin embargo, la geografía no reemplaza la información mensual: los modelos "
      "predictivos usan Lima, los rezagos y la estacionalidad, y su desempeño cambia por producto.")
    p(doc,
      "El resultado central es que no hay un modelo único. En el torneo de seis modelos de "
      "regresión, Random Forest obtiene el mejor ordenamiento en la estimación contemporánea, con "
      "R² positivo en papa, cebolla y zanahoria. Para el pronóstico a un mes, Ridge y OLS mejoran "
      "la caminata aleatoria en papa, cebolla y arroz, mientras que Random Forest y LightGBM no la "
      "superan. La evidencia respalda H1 de forma parcial y H2 de forma clara.")

    doc.add_heading("8.2. Límites y extensiones", level=2)
    p(doc,
      "Límites: datos mensuales, panel no balanceado, ausencia de volúmenes y costos de "
      "transporte, y falta de verificación del rezago de publicación. Por eso se habla de valor "
      "predictivo contemporáneo, no de una estimación operativa. El diseño es asociativo y "
      "predictivo, no causal. Los R² negativos de algunos productos y de la validación recuerdan "
      "que el poder predictivo es limitado y que el azar de la muestra importa.")
    p(doc,
      "Una extensión natural es incorporar costos de transferencia, volúmenes y precios "
      "internacionales para separar mejor integración de co-movimiento. Otra es evaluar modelos "
      "probabilísticos y horizontes múltiples, y repetir la comparación con más observaciones por "
      "producto. Mientras eso no ocurra, la recomendación es usar el modelo por producto y tarea, "
      "y reportar siempre la referencia y el tamaño de muestra.")


def referencias(doc: Document) -> None:
    doc.add_heading("Referencias", level=1)
    entradas = [
        "Baulch, B. (1997). Transfer costs, spatial arbitrage, and testing for food market integration. American Journal of Agricultural Economics, 79(2), 477–487.",
        "Bergmeir, C., y Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. Information Sciences, 191, 192–213.",
        "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.",
        "Breiman, L., Friedman, J. H., Olshen, R. A., y Stone, C. J. (1984). Classification and regression trees. Wadsworth.",
        "Chen, T., y Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.",
        "Fackler, P. L., y Goodwin, B. K. (2001). Spatial price analysis. En B. L. Gardner y G. C. Rausser (Eds.), Handbook of agricultural economics (Vol. 1, pp. 971–1024). Elsevier.",
        "Fayyad, U., Piatetsky-Shapiro, G., y Smyth, P. (1996). From data mining to knowledge discovery in databases. AI Magazine, 17(3), 37–54.",
        "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 29(5), 1189–1232.",
        "Hoerl, A. E., y Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. Technometrics, 12(1), 55–67.",
        "Hyndman, R. J., y Athanasopoulos, G. (2021). Forecasting: Principles and practice (3.ª ed.). OTexts.",
        "Hyndman, R. J., y Koehler, A. B. (2006). Another look at measures of forecast accuracy. International Journal of Forecasting, 22(4), 679–688.",
        "James, G., Witten, D., Hastie, T., y Tibshirani, R. (2021). An introduction to statistical learning: With applications in R (2.ª ed.). Springer.",
        "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., y Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30, 3146–3154.",
        "Meyer, J., y von Cramon-Taubadel, S. (2004). Asymmetric price transmission: A survey. Journal of Agricultural Economics, 55(3), 581–611.",
        "Ministerio de Desarrollo Agrario y Riego. (2026). Sistema de Información de Abastecimiento y Precios (SISAP): Precios de ciudades.",
        "Ravallion, M. (1986). Testing market integration. American Journal of Agricultural Economics, 68(1), 102–109.",
        "Tukey, J. W. (1977). Exploratory data analysis. Addison-Wesley.",
    ]
    for item in entradas:
        par = doc.add_paragraph()
        run = par.add_run(item)
        run.font.size = Pt(9.5)
        par.paragraph_format.left_indent = Inches(0.3)
        par.paragraph_format.first_line_indent = Inches(-0.3)
        par.paragraph_format.space_after = Pt(4)


def anexo_matriz(doc: Document) -> None:
    salto(doc)
    doc.add_heading("Anexo A. Matriz de diseño completa: papa en Puno", level=1)
    p(doc,
      "La tabla siguiente reproduce la matriz completa que se usa para estimar y evaluar el "
      "modelo de papa en Puno. Las columnas dl son los cambios de Lima, dp los cambios propios, "
      "sin1 y cos1 los armónicos anuales, y la columna y el cambio logarítmico de Puno. El "
      "entrenamiento usa las filas marcadas como entrenamiento; la validación y la prueba final "
      "se separan con las etiquetas correspondientes.")
    datos = pd.read_csv(AN / "06_modelos" / "matriz_diseno.csv")
    datos["split"] = datos["split"].replace({
        "entrenamiento": "entren.",
        "desarrollo": "valid.",
        "holdout": "prueba"})
    datos = datos.rename(columns={"split": "etapa", "y_cambio_log": "y"})
    tabla(doc, datos, "A1", "Matriz de diseño completa de papa en Puno.",
          columnas=["fecha", "etapa", "dl0", "dl1", "dl2", "dl3", "dp1", "dp2", "sin1", "cos1", "y"],
          decimales={"dl0": 3, "dl1": 3, "dl2": 3, "dl3": 3, "dp1": 3, "dp2": 3,
                     "sin1": 3, "cos1": 3, "y": 3},
          anchos=[0.72, 0.82, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.55],
          font=6,
          mantener_junta=False,
          nota="La matriz tiene 155 filas. Etapas: entren. = entrenamiento; valid. = validación; prueba = prueba final.")


def main() -> None:
    doc = Document()
    configurar(doc)
    portada(doc)
    salto(doc)
    resumen(doc)
    salto(doc)
    introduccion(doc)
    metodologia(doc)
    datos(doc)
    eda(doc)
    variables(doc)
    modelos(doc)
    resultados(doc)
    conclusiones(doc)
    referencias(doc)
    anexo_matriz(doc)
    salida = RAIZ / "informe.docx"
    doc.save(salida)
    print(f"Informe creado: {salida}")


if __name__ == "__main__":
    main()
