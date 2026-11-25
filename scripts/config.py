#!/usr/bin/env python3
"""Configuracion comun del proyecto.

Fuente unica de rutas, productos, ventanas, hiperparametros y estilo. Todos los
scripts la importan para evitar duplicar decisiones metodologicas.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns

try:
    from catalogo import EXCLUIDOS, cargar_catalogo
except ModuleNotFoundError:  # permite importar scripts.config desde la raiz
    from scripts.catalogo import EXCLUIDOS, cargar_catalogo

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"
RAW = DATA / "raw"
PROCESADOS = DATA / "processed"
SALIDAS = RAIZ / "outputs"
ANALISIS = SALIDAS / "analisis"
INFORME = SALIDAS / "informe"

DIRECTORIOS = {
    "fuente": ANALISIS / "01_fuente",
    "panel": ANALISIS / "02_panel",
    "calidad": ANALISIS / "03_calidad",
    "estructura": ANALISIS / "04_estructura",
    "ecm": ANALISIS / "05_ecm",
    "modelos": ANALISIS / "06_modelos",
    "robustez": ANALISIS / "07_robustez",
    "informe_figuras": INFORME / "figuras",
    "informe_tablas": INFORME / "tablas",
}

LARGO = PROCESADOS / "precios_ciudades_largo.csv"
PANEL_CRUDO = PROCESADOS / "panel_precios.csv"
PANEL_LIMPIO = PROCESADOS / "panel_precios_limpio.csv"
ATIPICOS = PROCESADOS / "atipicos_panel.csv"
UNIDADES = PROCESADOS / "unidades_panel.csv"

# --------------------------------------------------------------------------
# Unidad transversal
# --------------------------------------------------------------------------
# El catalogo es la unica fuente de verdad. Se excluyen Callao (provincia
# constitucional) y los mercados provinciales Andahuaylas, Chota y Jaen, que
# SISAP reporta aparte pero pertenecen a Apurimac y Cajamarca.
CATALOGO = cargar_catalogo()
DEPARTAMENTOS = CATALOGO["departamento_ascii"].tolist()
DEPARTAMENTOS_ID = CATALOGO["departamento_id"].tolist()
SUBDEPARTAMENTOS = {nombre: descripcion for nombre, descripcion in EXCLUIDOS.values()}

# --------------------------------------------------------------------------
# Productos
# --------------------------------------------------------------------------
PRODUCTOS = ["Papa", "Cebolla", "Zanahoria", "Azucar", "Arroz", "Limon",
             "Tomate", "Aceite", "Huevos"]
NUCLEO = ["Papa", "Cebolla", "Zanahoria", "Arroz", "Aceite", "Huevos"]

# --------------------------------------------------------------------------
# Ventanas
# --------------------------------------------------------------------------
ECM_INICIO = "2012-01-01"
ECM_FIN = "2023-12-01"
DEV_INICIO = "2018-01-01"
DEV_FIN = "2020-12-01"
HOLDOUT_INICIO = "2021-01-01"
HOLDOUT_FIN = "2026-10-01"
MIN_MESES = 60
MIN_NOWCAST_TRAIN = 60
MIN_NOWCAST_TEST = 20
MIN_FORECAST_TRAIN = 40
MIN_FORECAST_TEST = 20

# --------------------------------------------------------------------------
# Hiperparametros
# --------------------------------------------------------------------------
RIDGE_ALPHA = 1.0
RF = dict(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1)
LGBM = dict(n_estimators=300, learning_rate=0.05, num_leaves=7,
            min_child_samples=10, random_state=0, verbose=-1)
XGB = dict(n_estimators=300, learning_rate=0.05, max_depth=4,
           subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=0)
LAGS_ECM_MAX = 4

# --------------------------------------------------------------------------
# Geografia
# --------------------------------------------------------------------------
COORDENADAS = {
    fila.departamento_ascii: (fila.latitud, fila.longitud)
    for fila in CATALOGO.itertuples(index=False)
}
ZONAS = {
    fila.departamento_ascii: fila.zona_natural
    for fila in CATALOGO.itertuples(index=False)
}

# --------------------------------------------------------------------------
# Estilo
# --------------------------------------------------------------------------
COMA = FuncFormatter(lambda x, _: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))


def preparar_directorios() -> None:
    for ruta in DIRECTORIOS.values():
        ruta.mkdir(parents=True, exist_ok=True)


def aplicar_estilo() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 110,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "savefig.bbox": "tight",
    })


def cargar_panel(limpio: bool = True) -> pd.DataFrame:
    ruta = PANEL_LIMPIO if limpio else PANEL_CRUDO
    panel = pd.read_csv(ruta, parse_dates=["fecha"])
    panel["anio"] = panel["fecha"].dt.year
    return panel


def guardar(fig, nombre: str, carpeta: str = "calidad") -> None:
    preparar_directorios()
    fig.savefig(DIRECTORIOS[carpeta] / nombre, dpi=160, bbox_inches="tight")
    plt.close(fig)


def tabla(df: pd.DataFrame, nombre: str, carpeta: str = "calidad") -> None:
    preparar_directorios()
    df.to_csv(DIRECTORIOS[carpeta] / nombre, index=False, encoding="utf-8")
