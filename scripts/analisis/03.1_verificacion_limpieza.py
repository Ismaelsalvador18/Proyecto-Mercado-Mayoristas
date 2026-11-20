#!/usr/bin/env python3
"""Verificacion de la limpieza: antes y despues.

Corre despues de 03. Genera:
  outputs/analisis/03_calidad/verificacion_limpieza.csv
  outputs/analisis/03_calidad/efecto_limpieza.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (ATIPICOS, DIRECTORIOS, PANEL_CRUDO, PANEL_LIMPIO,
                    aplicar_estilo, guardar, preparar_directorios, tabla)


def cobertura(panel: pd.DataFrame, ini: str, fin: str) -> float:
    sub = panel[(panel.fecha >= ini) & (panel.fecha <= fin)]
    esperado = 27 * 9 * sub.fecha.nunique()
    return 100 * len(sub) / esperado if esperado else np.nan


def main() -> None:
    preparar_directorios()
    aplicar_estilo()
    crudo = pd.read_csv(PANEL_CRUDO, parse_dates=["fecha"])
    limpio = pd.read_csv(PANEL_LIMPIO, parse_dates=["fecha"])
    atipicos = pd.read_csv(ATIPICOS, parse_dates=["fecha"])

    filas = [
        ("filas antes", len(crudo)),
        ("filas despues", len(limpio)),
        ("filas descartadas", len(crudo) - len(limpio)),
        ("atipicos marcados", len(atipicos)),
        ("nulos antes", int(crudo.isna().sum().sum())),
        ("nulos despues", int(limpio.isna().sum().sum())),
        ("duplicados antes", int(crudo.duplicated(["fecha", "region", "categoria"]).sum())),
        ("duplicados despues", int(limpio.duplicated(["fecha", "region", "categoria"]).sum())),
        ("cobertura 2012-2023 antes (%)", round(cobertura(crudo, "2012-01-01", "2023-12-01"), 1)),
        ("cobertura 2012-2023 despues (%)", round(cobertura(limpio, "2012-01-01", "2023-12-01"), 1)),
    ]
    tabla(pd.DataFrame(filas, columns=["metrica", "valor"]), "verificacion_limpieza.csv", "calidad")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (datos, titulo) in zip(axes, [(crudo, "Antes"), (limpio, "Despues")]):
        d = datos.loc[datos.categoria == "Cebolla", "precio_kg"].dropna()
        d = d[d < d.quantile(0.995)]
        ax.hist(d, bins=50, color="#0e7490", alpha=0.75)
        ax.set_title(f"Cebolla: precio por kg ({titulo.lower()})")
        ax.set_xlabel("S/ por kg")
        ax.set_ylabel("frecuencia")
    plt.tight_layout()
    guardar(fig, "efecto_limpieza.png", "calidad")

    print("Verificacion de limpieza:")
    for metrica, valor in filas:
        print(f"  {metrica:38s} {valor}")
    print("\nTabla y figura de limpieza generadas en", DIRECTORIOS["calidad"])


if __name__ == "__main__":
    main()
