#!/usr/bin/env python3
"""Estacionariedad de las series: ADF y KPSS.

Corre despues de 03.2 y antes de 03.4.
Genera:
  outputs/analisis/03_calidad/estacionariedad_producto.csv
  outputs/analisis/03_calidad/estacionariedad_region.csv
  outputs/analisis/03_calidad/adf_region.png
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from config import (DIRECTORIOS, ECM_FIN, ECM_INICIO, NUCLEO, PANEL_LIMPIO,
                    aplicar_estilo, guardar, preparar_directorios, tabla)

warnings.filterwarnings("ignore")


def pruebas(serie: pd.Series) -> dict:
    adf_c = adfuller(serie, autolag="AIC")[1]
    adf_ct = adfuller(serie, autolag="AIC", regression="ct")[1]
    kpss_c = kpss(serie, regression="c", nlags="auto")[1]
    adf_d = adfuller(serie.diff().dropna(), autolag="AIC")[1]
    if adf_c < 0.05 and kpss_c > 0.05:
        decision = "I(0)"
    elif adf_c >= 0.05 and kpss_c < 0.05:
        decision = "I(1)" if adf_d < 0.05 else "mixto"
    else:
        decision = "mixto"
    return {"adf_c": adf_c, "adf_ct": adf_ct, "kpss_c": kpss_c,
            "adf_diff": adf_d, "decision": decision}


def main() -> None:
    preparar_directorios()
    aplicar_estilo()
    panel = pd.read_csv(PANEL_LIMPIO, parse_dates=["fecha"])

    filas = []
    for cat in NUCLEO:
        d = panel[panel.categoria == cat].pivot_table(
            index="fecha", columns="region", values="precio_kg")[["Puno", "Lima"]]
        d = d.loc[ECM_INICIO:ECM_FIN].dropna()
        for col in ["Puno", "Lima"]:
            r = pruebas(np.log(d[col]))
            filas.append({"producto": cat, "serie": col, **r})
    producto = pd.DataFrame(filas)
    tabla(producto, "estacionariedad_producto.csv", "calidad")

    filas = []
    for cat in NUCLEO:
        d = panel[panel.categoria == cat].pivot_table(
            index="fecha", columns="region", values="precio_kg").loc[ECM_INICIO:ECM_FIN]
        for region in [c for c in d.columns if c != "Lima"]:
            serie = d[region].dropna()
            if len(serie) < 60:
                continue
            r = pruebas(np.log(serie))
            filas.append({"producto": cat, "region": region, **r})
    regiones = pd.DataFrame(filas)
    tabla(regiones, "estacionariedad_region.csv", "calidad")

    resumen = (regiones.groupby("producto")
               .agg(regiones=("region", "nunique"),
                    i0=("decision", lambda s: int((s == "I(0)").sum())),
                    i1=("decision", lambda s: int((s == "I(1)").sum())),
                    mixto=("decision", lambda s: int((s == "mixto").sum())),
                    adf_mediana=("adf_c", "median"))
               .reset_index())
    tabla(resumen, "estacionariedad_resumen.csv", "calidad")

    piv = regiones.pivot(index="producto", columns="region", values="adf_c")
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.2)
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=8)
    ax.set_title("ADF sobre log-precio por producto y departamento (p-valor)")
    fig.colorbar(im, ax=ax, label="p-valor ADF")
    plt.tight_layout()
    guardar(fig, "adf_region.png", "calidad")

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.bar(resumen.producto, resumen.i0, color="#166534", label="I(0)")
    ax.bar(resumen.producto, resumen.i1, bottom=resumen.i0, color="#b45309", label="I(1)")
    ax.bar(resumen.producto, resumen.mixto, bottom=resumen.i0 + resumen.i1,
           color="#9ca3af", label="Mixto")
    ax.set_ylabel("departamentos")
    ax.set_title("Orden de integración por producto")
    ax.legend(fontsize=8)
    plt.xticks(rotation=20, ha="right")
    guardar(fig, "adf_resumen.png", "calidad")

    print("Estacionariedad Puno/Lima:")
    print(producto.round(4).to_string(index=False))
    print("\nResumen regional:")
    print(resumen.round(2).to_string(index=False))
    print("\nTablas y figura de estacionariedad generadas en", DIRECTORIOS["calidad"])


if __name__ == "__main__":
    main()
