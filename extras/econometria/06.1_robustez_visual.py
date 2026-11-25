#!/usr/bin/env python3
"""Visualizacion de la robustez del ECM.

Corre despues de 06. Genera:
  outputs/analisis/07_robustez/robustez_vida_media.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import pandas as pd

from config import DIRECTORIOS, aplicar_estilo, guardar, preparar_directorios

import seaborn as sns


def main() -> None:
    preparar_directorios()
    aplicar_estilo()
    tabla = pd.read_csv(DIRECTORIOS["robustez"] / "ecm_robustez.csv")
    piv = tabla.pivot(index="producto", columns="escenario", values="vida_media")
    orden = [c for c in ["base", "q0", "q1", "q2", "sin_shock", "ventana_2008",
                         "papa_amarilla"] if c in piv.columns]
    piv = piv[orden]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    sns.heatmap(piv, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5,
                linecolor="white", cbar_kws={"label": "vida media (meses)"}, ax=ax)
    ax.set_title("Robustez: vida media del desvio por escenario")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=30, ha="right")
    guardar(fig, "robustez_vida_media.png", "robustez")
    print("Figura de robustez generada en", DIRECTORIOS["robustez"])


if __name__ == "__main__":
    main()
