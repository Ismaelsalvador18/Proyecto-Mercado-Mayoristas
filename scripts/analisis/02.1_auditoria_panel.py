#!/usr/bin/env python3
"""Auditoria del panel crudo: nulos, duplicados, unidades, truncamiento y atipicos.

Corre despues de 02 y antes de 03. No modifica datos: solo diagnostica.
Genera:
  outputs/analisis/02_panel/auditoria_panel.csv
  outputs/analisis/02_panel/unidades_cambio.csv
  outputs/analisis/02_panel/series_truncadas.csv
  outputs/analisis/02_panel/atipicos_detectados.csv
  outputs/analisis/02_panel/cobertura_anual.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (DIRECTORIOS, LARGO, PANEL_CRUDO, aplicar_estilo, guardar,
                    preparar_directorios, tabla)


def detectar_atipicos(panel: pd.DataFrame) -> pd.Series:
    panel = panel.sort_values(["region", "categoria", "fecha"]).reset_index(drop=True)
    nac = panel.groupby(["categoria", "fecha"])["precio_kg"].transform("median")
    loc = panel.groupby(["region", "categoria"])["precio_kg"].transform(
        lambda s: s.rolling(7, center=True, min_periods=3).median())
    return ((panel.precio_kg / loc > 4) | (panel.precio_kg / loc < 0.25) |
            (panel.precio_kg / nac > 5) | (panel.precio_kg / nac < 0.2) |
            (panel.precio_kg < 0.05))


def main() -> None:
    preparar_directorios()
    aplicar_estilo()
    panel = pd.read_csv(PANEL_CRUDO, parse_dates=["fecha"])
    crudo = pd.read_csv(LARGO)

    atipico = detectar_atipicos(panel)
    panel["atipico"] = atipico

    filas = [
        ("filas", len(panel)),
        ("regiones", panel.region.nunique()),
        ("productos", panel.categoria.nunique()),
        ("nulos", int(panel.isna().sum().sum())),
        ("duplicados fecha-region-producto", int(panel.duplicated(["fecha", "region", "categoria"]).sum())),
        ("atipicos detectados", int(atipico.sum())),
        ("atipicos (%)", round(100 * atipico.mean(), 3)),
        ("primer mes", panel.fecha.min().strftime("%Y-%m")),
        ("ultimo mes", panel.fecha.max().strftime("%Y-%m")),
    ]
    for ini, fin, etiqueta in [("2012-01-01", "2023-12-01", "2012-2023"),
                               ("2002-01-01", "2026-11-01", "2002-2026")]:
        sub = panel[(panel.fecha >= ini) & (panel.fecha <= fin)]
        esperado = 27 * 9 * sub.fecha.nunique()
        filas.append((f"cobertura {etiqueta} (%)", round(100 * len(sub) / esperado, 1)))
    tabla(pd.DataFrame(filas, columns=["metrica", "valor"]), "auditoria_panel.csv", "panel")

    unidades = crudo.groupby(["region", "producto"]).unidad.nunique().reset_index(name="unidades")
    tabla(unidades[unidades.unidades > 1].sort_values("unidades", ascending=False),
          "unidades_cambio.csv", "panel")

    serie = panel.groupby(["region", "categoria"]).fecha.agg(["min", "max", "size"]).reset_index()
    serie.columns = ["region", "categoria", "primer_mes", "ultimo_mes", "meses"]
    tabla(serie, "series_truncadas.csv", "panel")

    detalle = panel.loc[atipico, ["fecha", "region", "categoria", "precio_kg"]].copy()
    tabla(detalle, "atipicos_detectados.csv", "panel")

    anio = panel.groupby("anio").agg(regiones=("region", "nunique"),
                                     productos=("categoria", "nunique"),
                                     filas=("precio_kg", "size")).reset_index()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(anio.anio, anio.regiones, marker="o", label="regiones")
    ax.plot(anio.anio, anio.productos, marker="s", label="productos")
    ax.set_title("Cobertura anual del panel")
    ax.set_xlabel("año")
    ax.set_ylabel("cantidad")
    ax.legend()
    guardar(fig, "cobertura_anual.png", "panel")

    print("Auditoria del panel:")
    for metrica, valor in filas:
        print(f"  {metrica:40s} {valor}")
    print(f"\nCeldas con mas de una unidad: {int((unidades.unidades > 1).sum())} de {len(unidades)}")
    print(f"Series que terminan antes de 2026-11: {int((serie.ultimo_mes < pd.Timestamp('2026-11-01')).sum())} de {len(serie)}")
    print(f"Series con menos de 100 meses: {int((serie.meses < 100).sum())}")
    print("\nTablas y figuras del panel generadas en", DIRECTORIOS["panel"])


if __name__ == "__main__":
    main()
