#!/usr/bin/env python3
"""Gráficos de predicciones por producto, con colores sólidos y distinguibles.

Lee las predicciones del holdout y genera:
  - ganador_vs_observado.png (2 x 3, para el informe)
  - predicciones_papa.png, predicciones_cebolla.png, ... (una por producto, para el PPT)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DIRECTORIOS, NUCLEO, aplicar_estilo, guardar, preparar_directorios

COLORES = {
    "OLS": "#1f77b4",
    "Ridge": "#ff7f0e",
    "Arbol": "#d62728",
    "Random Forest": "#2ca02c",
    "XGBoost": "#9467bd",
    "LightGBM": "#8c564b",
}
ETIQUETAS = {"Arbol": "Árbol de Decisión"}


def _etiqueta(modelo: str) -> str:
    return ETIQUETAS.get(modelo, modelo)


def _panel(ax, bloque: pd.DataFrame, titulo: str, leyenda: bool = False) -> None:
    observado = bloque.groupby("fecha")["observado"].first()
    ax.plot(observado.index, observado.values * 100, color="#111827",
            linewidth=2.0, label="Observado", zorder=3)
    for modelo, grupo in bloque.groupby("modelo"):
        grupo = grupo.sort_values("fecha")
        ax.plot(grupo.fecha, grupo.prediccion * 100,
                color=COLORES.get(modelo), linewidth=1.6, alpha=0.95,
                label=_etiqueta(modelo), zorder=2)
    ax.axhline(0, color="#9ca3af", linewidth=0.7)
    ax.set_title(titulo, fontsize=11)
    ax.set_ylabel("variación %")
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=8)
    if leyenda:
        ax.legend(ncol=4, fontsize=8.5, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), frameon=False)


def main() -> None:
    preparar_directorios()
    aplicar_estilo()
    predicciones = pd.read_csv(DIRECTORIOS["modelos"] / "predicciones_holdout.csv",
                               parse_dates=["fecha"])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, cat in zip(axes.ravel(), NUCLEO):
        _panel(ax, predicciones[predicciones.producto == cat], cat)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=8,
               frameon=False)
    fig.suptitle("Cambio observado y predicciones de los seis modelos por producto",
                 y=1.01)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    guardar(fig, "ganador_vs_observado.png", "modelos")

    for cat in NUCLEO:
        bloque = predicciones[predicciones.producto == cat]
        fig, ax = plt.subplots(figsize=(11, 6.5))
        _panel(ax, bloque, f"{cat}: observado y seis modelos", leyenda=True)
        plt.tight_layout()
        guardar(fig, f"predicciones_{cat.lower()}.png", "modelos")

    print("Predicciones por producto generadas en", DIRECTORIOS["modelos"])


if __name__ == "__main__":
    main()
