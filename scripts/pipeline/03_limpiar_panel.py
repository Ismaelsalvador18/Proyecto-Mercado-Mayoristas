#!/usr/bin/env python3
"""Limpia valores incompatibles con el entorno del precio.

El SISAP tiene errores de transcripcion tipicos: un precio de 145,34 S/kg de
cebolla cuando la mediana nacional ese mes es 2,36 (desplazamiento de punto
decimal). No se imputan: se marcan como faltantes y se reportan.

Regla (doble, para no confundir un pico real con un error):
  1. Comparacion local: precio / mediana movil de +-3 meses. Se descarta si
     el cociente es > 4 o < 0,25.
  2. Comparacion nacional: precio / mediana del mismo producto y mes entre
     regiones. Se descarta si el cociente es > 5 o < 0,2.

Entrada : data/processed/panel_precios.csv
Salidas : data/processed/panel_precios_limpio.csv
          data/processed/atipicos_panel.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
ENTRADA = RAIZ / "data" / "processed" / "panel_precios.csv"
SALIDAS = RAIZ / "data" / "processed"


def main() -> None:
    panel = pd.read_csv(ENTRADA, parse_dates=["fecha"])
    panel = panel.sort_values(["region", "categoria", "fecha"]).reset_index(drop=True)

    panel["mediana_nacional"] = panel.groupby(["categoria", "fecha"])["precio_kg"].transform("median")
    panel["ratio_nacional"] = panel["precio_kg"] / panel["mediana_nacional"]

    grupo = panel.groupby(["region", "categoria"])["precio_kg"]
    panel["mediana_local"] = grupo.transform(
        lambda s: s.rolling(7, center=True, min_periods=3).median()
    )
    panel["ratio_local"] = panel["precio_kg"] / panel["mediana_local"]

    atipico = (
        (panel["ratio_local"] > 4)
        | (panel["ratio_local"] < 0.25)
        | (panel["ratio_nacional"] > 5)
        | (panel["ratio_nacional"] < 0.2)
        | (panel["precio_kg"] < 0.05)
    )

    panel["atipico"] = atipico
    panel["precio_limpio"] = panel["precio_kg"].where(~atipico)

    detalle = panel.loc[atipico, [
        "fecha", "region", "categoria", "precio_kg", "mediana_local",
        "mediana_nacional", "ratio_local", "ratio_nacional",
    ]].copy()
    detalle.to_csv(SALIDAS / "atipicos_panel.csv", index=False, encoding="utf-8")

    limpio = panel.drop(columns=["precio_kg", "mediana_local", "mediana_nacional",
                                 "ratio_local", "ratio_nacional", "atipico"])
    limpio = limpio.rename(columns={"precio_limpio": "precio_kg"})
    limpio = limpio.dropna(subset=["precio_kg"])
    limpio.to_csv(SALIDAS / "panel_precios_limpio.csv", index=False, encoding="utf-8")

    print(f"Filas originales : {len(panel):,}")
    print(f"Atipicos marcados: {int(atipico.sum()):,} ({100 * atipico.mean():.2f} %)")
    print("\nPor producto:")
    print(detalle.groupby("categoria").size().sort_values(ascending=False).to_string())
    print("\nEjemplos mas extremos:")
    print(
        detalle.assign(ratio_nacional=detalle["ratio_nacional"].round(1))
        .nlargest(8, "ratio_nacional")
        [["fecha", "region", "categoria", "precio_kg", "mediana_nacional", "ratio_nacional"]]
        .to_string(index=False)
    )
    print(f"\nArchivo limpio: {len(limpio):,} filas -> data/processed/panel_precios_limpio.csv")


if __name__ == "__main__":
    main()
