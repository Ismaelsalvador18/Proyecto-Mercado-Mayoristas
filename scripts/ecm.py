#!/usr/bin/env python3
"""Construccion y estimacion del ECM en un unico lugar.

Regla: las diferencias, rezagos y el termino de correccion se construyen sobre
el calendario mensual completo. La ventana de estimacion se aplica DESPUES.
Asi se evita tratar un mes despues de un hueco como si fuera el mes anterior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def calendario(series: list[pd.Series]) -> pd.DatetimeIndex:
    inicio = min(s.index.min() for s in series)
    fin = max(s.index.max() for s in series)
    return pd.date_range(inicio, fin, freq="MS")


def construir(lp: pd.Series, ll: pd.Series, q: int) -> pd.DataFrame:
    idx = calendario([lp, ll])
    lp = lp.reindex(idx)
    ll = ll.reindex(idx)
    dp, dl = lp.diff(), ll.diff()
    datos = pd.DataFrame({"dp": dp, "ect": lp.shift(1) - ll.shift(1)})
    for k in range(q + 1):
        datos[f"dl{k}"] = dl.shift(k)
    return datos.dropna()


def estimar(datos: pd.DataFrame, hac: bool = True):
    X = sm.add_constant(datos.drop(columns="dp"))
    modelo = sm.OLS(datos["dp"], X)
    if hac:
        return modelo.fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    return modelo.fit()


def vida_media(gamma: float) -> float:
    return float(np.log(0.5) / np.log(1 + gamma)) if -2 < gamma < 0 else np.nan


def bootstrap_ic(datos: pd.DataFrame, repeticiones: int = 500) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    valores = []
    for _ in range(repeticiones):
        idx = rng.integers(0, len(datos), len(datos))
        muestra = datos.iloc[idx]
        modelo = sm.OLS(muestra["dp"], sm.add_constant(muestra.drop(columns="dp"))).fit()
        v = vida_media(float(modelo.params["ect"]))
        if np.isfinite(v):
            valores.append(v)
    if not valores:
        return np.nan, np.nan
    return float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))


def estimar_producto(panel: pd.DataFrame, producto: str, region: str, referencia: str,
                     q: int, inicio: str, fin: str, hac: bool = True):
    """Devuelve (modelo, datos) con la ventana aplicada despues de construir."""
    tabla = panel[panel.categoria == producto].pivot_table(
        index="fecha", columns="region", values="precio_kg")
    lp, ll = np.log(tabla[region]), np.log(tabla[referencia])
    datos = construir(lp, ll, q).loc[inicio:fin]
    return estimar(datos, hac=hac), datos
