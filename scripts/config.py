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
# 24 departamentos. Se excluyen Callao (provincia constitucional) y los
# mercados provinciales Andahuaylas, Chota y Jaen, que SISAP reporta aparte
# pero pertenecen a Apurimac y Cajamarca. Incluirlos como "departamentos"
# duplicaria la unidad de analisis y romperia la comparacion Lima-Puno.
DEPARTAMENTOS = [
    "Amazonas", "Ancash", "Apurimac", "Arequipa", "Ayacucho", "Cajamarca",
    "Cusco", "Huancavelica", "Huanuco", "Ica", "Junin", "La Libertad",
    "Lambayeque", "Lima", "Loreto", "Madre de Dios", "Moquegua", "Pasco",
    "Piura", "Puno", "San Martin", "Tacna", "Tumbes", "Ucayali",
]
SUBDEPARTAMENTOS = {
    "Andahuaylas": "Apurimac",
    "Chota": "Cajamarca",
    "Jaen": "Cajamarca",
}

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
    "Lima": (-12.05, -77.04), "Ica": (-14.07, -75.73), "Huanuco": (-9.93, -76.24),
    "Arequipa": (-16.41, -71.54), "La Libertad": (-8.11, -79.03),
    "Ayacucho": (-13.16, -74.22), "Cusco": (-13.53, -71.97), "Puno": (-15.84, -70.02),
    "Piura": (-5.19, -80.63), "Loreto": (-3.75, -73.25), "Tumbes": (-3.57, -80.45),
    "Junin": (-12.07, -75.21), "Ancash": (-9.53, -77.53), "Cajamarca": (-7.16, -78.51),
    "Amazonas": (-6.23, -77.87), "San Martin": (-6.03, -76.97),
    "Ucayali": (-8.38, -74.55), "Madre de Dios": (-12.59, -69.18),
    "Moquegua": (-17.19, -70.94), "Pasco": (-10.68, -76.26),
    "Huancavelica": (-12.79, -74.97), "Apurimac": (-13.63, -72.88),
    "Tacna": (-18.01, -70.25), "Lambayeque": (-6.77, -79.84),
}

ZONAS = {
    "Lima": "Costa", "Ica": "Costa", "Arequipa": "Costa", "La Libertad": "Costa",
    "Piura": "Costa", "Tumbes": "Costa", "Lambayeque": "Costa", "Moquegua": "Costa",
    "Tacna": "Costa", "Ancash": "Sierra", "Cajamarca": "Sierra",
    "Huancavelica": "Sierra", "Huanuco": "Sierra", "Junin": "Sierra",
    "Pasco": "Sierra", "Ayacucho": "Sierra", "Apurimac": "Sierra", "Cusco": "Sierra",
    "Puno": "Sierra", "Amazonas": "Sierra",
    "Loreto": "Selva", "San Martin": "Selva", "Ucayali": "Selva",
    "Madre de Dios": "Selva",
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
