#!/usr/bin/env python3
"""Construye el panel mensual de precios mayoristas por departamento y producto.

Entrada : data/processed/precios_ciudades_largo.csv
Salidas : data/processed/panel_precios.csv       (formato largo tidy)
          data/processed/cobertura_panel.csv     (cobertura por producto-region)
          data/processed/unidades_panel.csv      (unidad dominante elegida)

Reglas:
  1. Se elige una variedad comercial por producto (la de mejor cobertura
     conjunta entre regiones y anos).
  2. Para cada region y producto se toma la unidad de medida dominante en el
     tiempo (kilogramo, arroba, saco...). Asi no se mezclan presentaciones.
  3. El precio se normaliza a soles por kilogramo dividiendo por la
     equivalencia en kg que publica el SISAP.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEPARTAMENTOS

RAIZ = Path(__file__).resolve().parent.parent.parent
ENTRADA = RAIZ / "data" / "processed" / "precios_ciudades_largo.csv"
SALIDAS = RAIZ / "data" / "processed"

VARIEDADES = {
    "Papa canchan": "Papa",
    "Cebolla cabeza roja": "Cebolla",
    "Tomate marzano": "Tomate",
    "Zanahoria": "Zanahoria",
    "Limon sutil bolsa": "Limon",
    "Arroz superior": "Arroz",
    "Huevos rosados": "Huevos",
    "Azucar rubia nacional (granel)": "Azucar",
    "Aceite clásico botella x 1l": "Aceite",
}


def unidad_dominante(datos: pd.DataFrame) -> pd.DataFrame:
    conteo = (
        datos.groupby(["region", "producto", "unidad", "equivalencia_kg"])
        .size()
        .reset_index(name="n")
    )
    # En empate, prefiere kilogramo (equivalencia 1).
    conteo["prioridad"] = np.where(
        (conteo["unidad"] == "Kilogramo") & (conteo["equivalencia_kg"] == 1), 0, 1
    )
    conteo = conteo.sort_values(
        ["region", "producto", "n", "prioridad"], ascending=[True, True, False, True]
    )
    return conteo.drop_duplicates(["region", "producto"])


def main() -> None:
    crudo = pd.read_csv(ENTRADA)
    datos = crudo[crudo["producto"].isin(VARIEDADES)].copy()
    datos = datos[datos["region"].isin(DEPARTAMENTOS)].copy()
    datos["categoria"] = datos["producto"].map(VARIEDADES)

    elegidas = unidad_dominante(datos)
    datos = datos.merge(
        elegidas[["region", "producto", "unidad", "equivalencia_kg"]],
        on=["region", "producto", "unidad", "equivalencia_kg"],
        how="inner",
    )

    datos["precio_kg"] = datos["valor"] / datos["equivalencia_kg"].replace(0, np.nan)
    datos["fecha"] = pd.to_datetime(
        dict(year=datos["anio"], month=datos["mes"], day=1)
    )

    # Duplicados residuales: promedio mensual.
    panel = (
        datos.groupby(["fecha", "anio", "mes", "region", "categoria"], as_index=False)
        .agg(precio_kg=("precio_kg", "mean"), n_registros=("precio_kg", "size"))
    )
    panel = panel.sort_values(["categoria", "region", "fecha"]).reset_index(drop=True)

    panel.to_csv(SALIDAS / "panel_precios.csv", index=False, encoding="utf-8")
    elegidas.to_csv(SALIDAS / "unidades_panel.csv", index=False, encoding="utf-8")

    cobertura = (
        panel.groupby(["categoria", "region"])
        .agg(
            meses=("precio_kg", "size"),
            primer_mes=("fecha", "min"),
            ultimo_mes=("fecha", "max"),
            precio_medio=("precio_kg", "mean"),
        )
        .reset_index()
    )
    cobertura["anios_cubiertos"] = cobertura["meses"] / 12
    cobertura.to_csv(SALIDAS / "cobertura_panel.csv", index=False, encoding="utf-8")

    print(f"Panel: {panel.shape[0]:,} filas | {panel.region.nunique()} departamentos "
          f"| {panel.categoria.nunique()} productos")
    print(f"Ventana: {panel.fecha.min():%Y-%m} a {panel.fecha.max():%Y-%m}")
    print("\nMeses por producto (resumen entre regiones):")
    resumen = (
        cobertura.groupby("categoria")["meses"]
        .agg(["min", "median", "max"])
        .sort_values("median", ascending=False)
    )
    print(resumen.round(0).to_string())
    print("\nCobertura de Puno:")
    print(
        cobertura[cobertura.region == "Puno"]
        .sort_values("meses", ascending=False)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
