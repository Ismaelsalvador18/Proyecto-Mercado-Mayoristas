#!/usr/bin/env python3
"""Descripcion de los datos: tabla del dataset, estadisticos, campanas y normalidad.

Corre después de la verificación de limpieza.
Genera en outputs/analisis/03_calidad:
  tabla_variables.csv, tabla_dataset_filas.csv, descriptivos_producto.csv,
  normalidad.csv, transformacion_log.csv,
  campanas_precio.png, campanas_log.png, boxplots.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import (DIRECTORIOS, NUCLEO, PANEL_LIMPIO, PRODUCTOS,
                    aplicar_estilo, guardar, preparar_directorios, tabla)

import seaborn as sns


def variables() -> pd.DataFrame:
    return pd.DataFrame([
        ("fecha", "fecha", "mes", "índice temporal", "SISAP"),
        ("anio", "entero", "año", "derivada", "fecha"),
        ("mes", "entero", "1-12", "derivada", "fecha"),
        ("region", "texto", "24 departamentos", "unidad transversal", "SISAP"),
        ("categoria", "texto", "9 categorías", "unidad de producto armonizado", "SISAP"),
        ("precio_kg", "decimal", "S/ por kg", "variable de análisis", "SISAP normalizado"),
        ("n_registros", "entero", "conteo", "control de agregación", "derivada"),
    ], columns=["variable", "tipo", "unidad", "rol", "fuente"])


def descriptivos(panel: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for cat in PRODUCTOS:
        d = panel.loc[panel.categoria == cat, "precio_kg"].dropna()
        filas.append({
            "producto": cat, "n": len(d), "media": d.mean(), "desv": d.std(),
            "min": d.min(), "p25": d.quantile(.25), "mediana": d.median(),
            "p75": d.quantile(.75), "max": d.max(),
            "asimetria": d.skew(), "curtosis": d.kurt(),
        })
    return pd.DataFrame(filas)


def normalidad(panel: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for cat in PRODUCTOS:
        d = panel.loc[panel.categoria == cat, "precio_kg"].dropna()
        ld = np.log(d[d > 0])
        jb_p = stats.jarque_bera(d)[1]
        jb_l = stats.jarque_bera(ld)[1]
        filas.append({
            "producto": cat, "n": len(d),
            "asimetria_precio": d.skew(), "curtosis_precio": d.kurt(),
            "jb_p_precio": jb_p,
            "asimetria_log": ld.skew(), "curtosis_log": ld.kurt(),
            "jb_p_log": jb_l,
        })
    return pd.DataFrame(filas)


def campanas(panel: pd.DataFrame, log: bool, nombre: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, cat in zip(axes.ravel(), PRODUCTOS):
        d = panel.loc[panel.categoria == cat, "precio_kg"].dropna()
        serie = np.log(d[d > 0]) if log else d
        inf, sup = (serie.quantile(0.005), serie.quantile(0.995)) if log else (0, d.quantile(0.995))
        sns.histplot(serie, bins=45, stat="density", color="#3b82a0", alpha=0.6, ax=ax)
        serie.plot(kind="kde", ax=ax, color="#b45309", linewidth=1.3)
        mu, sigma = serie.mean(), serie.std()
        x = np.linspace(inf, sup, 200)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), color="#111827", linestyle="--",
                linewidth=1.0, label="normal")
        ax.set_xlim(inf, sup)
        ax.set_title(cat)
        ax.set_xlabel("log S/ kg" if log else "S/ por kg")
        ax.set_ylabel("densidad")
    fig.suptitle("Campanas de Gauss: " + ("log-precio" if log else "precio"), y=1.01)
    plt.tight_layout()
    guardar(fig, nombre, "calidad")


def boxplots(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    sns.boxplot(data=panel, x="categoria", y="precio_kg", ax=ax,
                color="#bae6fd", fliersize=2)
    ax.set_yscale("log")
    ax.set_title("Distribucion del precio por producto (escala log)")
    ax.set_xlabel("")
    ax.set_ylabel("S/ por kg (log)")
    plt.xticks(rotation=30, ha="right")
    guardar(fig, "boxplots.png", "calidad")


def main() -> None:
    preparar_directorios()
    aplicar_estilo()
    panel = pd.read_csv(PANEL_LIMPIO, parse_dates=["fecha"])

    tabla(variables(), "tabla_variables.csv", "calidad")
    tabla(panel.head(8), "tabla_dataset_filas.csv", "calidad")

    desc = descriptivos(panel)
    tabla(desc, "descriptivos_producto.csv", "calidad")

    norm = normalidad(panel)
    tabla(norm, "normalidad.csv", "calidad")

    comparacion = norm[["producto", "asimetria_precio", "asimetria_log",
                        "curtosis_precio", "curtosis_log"]].copy()
    comparacion["reduce_asimetria"] = comparacion.asimetria_log.abs() < comparacion.asimetria_precio.abs()
    comparacion["reduce_curtosis"] = comparacion.curtosis_log < comparacion.curtosis_precio
    tabla(comparacion, "transformacion_log.csv", "calidad")

    campanas(panel, log=False, nombre="campanas_precio.png")
    campanas(panel, log=True, nombre="campanas_log.png")
    boxplots(panel)

    print("Descriptivos por producto:")
    print(desc.round(3).to_string(index=False))
    print("\nNormalidad (Jarque-Bera):")
    print(norm[["producto", "jb_p_precio", "jb_p_log", "asimetria_precio", "asimetria_log"]]
          .round(4).to_string(index=False))
    print("\nEl log reduce la asimetria en", int(comparacion.reduce_asimetria.sum()),
          "de", len(comparacion), "productos.")
    print("\nTablas y figuras de descripcion generadas en", DIRECTORIOS["calidad"])


if __name__ == "__main__":
    main()
