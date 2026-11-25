#!/usr/bin/env python3
"""Robustez del ECM: especificacion, ventana y variedad.

La construccion se hace sobre el calendario completo y la ventana se aplica
despues, igual que en 04_ecm.py.

Genera:
  outputs/analisis/07_robustez/ecm_robustez.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from config import (DIRECTORIOS, ECM_FIN, ECM_INICIO, LARGO, MIN_MESES, NUCLEO,
                    PANEL_LIMPIO, preparar_directorios, tabla)
from ecm import construir, estimar, vida_media

SELECCION = DIRECTORIOS["calidad"] / "seleccion_rezagos.csv"


def ecm(region: pd.Series, lima: pd.Series, q: int, inicio: str, fin: str):
    datos = construir(np.log(region), np.log(lima), q).loc[inicio:fin]
    if len(datos) < MIN_MESES:
        return None
    m = estimar(datos)
    g = float(m.params["ect"])
    return {"n": int(m.nobs), "beta0": float(m.params.get("dl0", np.nan)),
            "gamma": g, "vida_media": vida_media(g),
            "r2": float(m.rsquared), "p_gamma": float(m.pvalues["ect"])}


def panel_variedad(variedad: str) -> pd.DataFrame:
    crudo = pd.read_csv(LARGO)
    d = crudo[crudo.producto == variedad]
    filas = []
    for region in ["Puno", "Lima"]:
        bloque = d[d.region == region]
        if bloque.empty:
            continue
        conteo = bloque.groupby(["unidad", "equivalencia_kg"]).size().reset_index(name="n")
        conteo = conteo.sort_values("n", ascending=False).iloc[0]
        bloque = bloque[(bloque.unidad == conteo.unidad) &
                        (bloque.equivalencia_kg == conteo.equivalencia_kg)]
        bloque = bloque.assign(precio_kg=bloque.valor / bloque.equivalencia_kg,
                               fecha=pd.to_datetime(dict(year=bloque.anio, month=bloque.mes, day=1)))
        filas.append(bloque.groupby("fecha").precio_kg.mean().rename(region))
    return pd.concat(filas, axis=1, sort=True)


def main() -> None:
    preparar_directorios()
    panel = pd.read_csv(PANEL_LIMPIO, parse_dates=["fecha"])
    qs = dict(zip(*pd.read_csv(SELECCION)[["producto", "q_elegido"]].values.T))

    filas = []
    for cat in NUCLEO:
        q = int(qs.get(cat, 0))
        d = panel[panel.categoria == cat].pivot_table(
            index="fecha", columns="region", values="precio_kg")
        for escenario, qq, inicio, fin in [
            ("base", q, ECM_INICIO, ECM_FIN),
            ("q0", 0, ECM_INICIO, ECM_FIN),
            ("q1", 1, ECM_INICIO, ECM_FIN),
            ("q2", 2, ECM_INICIO, ECM_FIN),
            ("sin_shock", q, ECM_INICIO, "2020-12-01"),
            ("ventana_2008", q, "2008-01-01", ECM_FIN),
        ]:
            r = ecm(d["Puno"], d["Lima"], qq, inicio, fin)
            if r:
                filas.append({"producto": cat, "escenario": escenario, "q": qq, **r})

    try:
        alt = panel_variedad("Papa amarilla")
        if {"Puno", "Lima"}.issubset(alt.columns):
            r = ecm(alt["Puno"], alt["Lima"], 0, ECM_INICIO, ECM_FIN)
            if r:
                filas.append({"producto": "Papa", "escenario": "papa_amarilla", "q": 0, **r})
            else:
                solape = alt[["Puno", "Lima"]].dropna().loc[ECM_INICIO:ECM_FIN]
                filas.append({"producto": "Papa", "escenario": "papa_amarilla", "q": 0,
                              "n": len(solape), "beta0": np.nan, "gamma": np.nan,
                              "vida_media": np.nan, "r2": np.nan, "p_gamma": np.nan})
    except Exception as error:
        print("AVISO: no se pudo estimar papa amarilla:", error)

    tabla(pd.DataFrame(filas), "ecm_robustez.csv", "robustez")
    print("Robustez del ECM:")
    print(pd.DataFrame(filas).round(3).to_string(index=False))
    print("\nTabla de robustez generada en", DIRECTORIOS["robustez"])


if __name__ == "__main__":
    main()
