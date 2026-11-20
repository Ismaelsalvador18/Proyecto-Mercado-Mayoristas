#!/usr/bin/env python3
"""Evaluacion de los modelos: metricas y curvas de aprendizaje.

Corre despues de 05.1.
Genera en outputs/analisis/06_modelos:
  curvas_aprendizaje.csv, curvas_aprendizaje.png, metricas_modelos.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DEV_FIN, DEV_INICIO, DIRECTORIOS, NUCLEO, PANEL_LIMPIO, aplicar_estilo, guardar, preparar_directorios, tabla
from modelos import construir_nowcast, metricas, modelos_principales


def main() -> None:
    preparar_directorios()
    aplicar_estilo()
    panel = pd.read_csv(PANEL_LIMPIO, parse_dates=["fecha"])

    # 1. Curvas de aprendizaje sobre Cebolla
    X, y = construir_nowcast(panel, "Cebolla")
    train = X.index < pd.Timestamp(DEV_INICIO)
    test = (X.index >= pd.Timestamp(DEV_INICIO)) & (X.index <= pd.Timestamp(DEV_FIN))
    X_tr, y_tr = X[train], y[train]
    real = y[test].values
    filas = []
    for fraccion in [0.5, 0.7, 0.85, 1.0]:
        n = max(20, int(len(X_tr) * fraccion))
        Xf, yf = X_tr.iloc[:n], y_tr.iloc[:n]
        for nombre, modelo in modelos_principales().items():
            modelo.fit(Xf, yf)
            pred = modelo.predict(X[test])
            m = metricas(real, pred, np.zeros_like(real))
            filas.append({"fraccion": fraccion, "n_entrenamiento": n,
                          "modelo": nombre, "mae": m["mae"], "r2": m["r2"]})
    curvas = pd.DataFrame(filas)
    tabla(curvas, "curvas_aprendizaje.csv", "modelos")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for nombre, grupo in curvas.groupby("modelo"):
        axes[0].plot(grupo.n_entrenamiento, grupo.mae, marker="o", label=nombre)
        axes[1].plot(grupo.n_entrenamiento, grupo.r2, marker="o", label=nombre)
    axes[0].set_title("MAE vs. tamaño de entrenamiento")
    axes[0].set_xlabel("observaciones mensuales de entrenamiento")
    axes[0].set_ylabel("MAE (cambio log)")
    axes[1].set_title("R2 vs. tamaño de entrenamiento")
    axes[1].set_xlabel("observaciones mensuales de entrenamiento")
    axes[1].set_ylabel("R2")
    axes[1].axhline(0, color="#9ca3af", linewidth=0.8)
    axes[0].legend(fontsize=7)
    fig.suptitle("Curvas de aprendizaje: Cebolla", y=1.03)
    plt.tight_layout()
    guardar(fig, "curvas_aprendizaje.png", "modelos")

    # 2. Metricas del holdout por producto
    met = pd.read_csv(DIRECTORIOS["modelos"] / "torneo_nowcast_metricas.csv")
    hold = met[(met.periodo == "holdout") & (met.conjunto == "principal")]
    piv_r2 = hold.pivot(index="producto", columns="modelo", values="r2").loc[NUCLEO]
    piv_dir = hold.pivot(index="producto", columns="modelo", values="acierto").loc[NUCLEO]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    piv_r2.plot(kind="bar", ax=axes[0], width=0.75)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("R2 en holdout por producto")
    axes[0].set_ylabel("R2")
    axes[0].legend(fontsize=7, ncol=4)
    piv_dir.plot(kind="bar", ax=axes[1], width=0.75)
    axes[1].axhline(50, color="#6b7280", linestyle="--", linewidth=0.9)
    axes[1].set_title("Acierto direccional en holdout por producto")
    axes[1].set_ylabel("%")
    axes[1].set_xlabel("")
    axes[1].legend(fontsize=7, ncol=4)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    guardar(fig, "metricas_modelos.png", "modelos")

    print("Curvas de aprendizaje (Cebolla):")
    print(curvas.round(3).to_string(index=False))
    print("\nTablas y figuras de evaluacion generadas en", DIRECTORIOS["modelos"])


if __name__ == "__main__":
    main()
