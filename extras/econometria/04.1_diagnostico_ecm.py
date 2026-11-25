#!/usr/bin/env python3
"""Diagnostico del ECM sobre la misma muestra que 04_ecm.py.

Construye los rezagos sobre el calendario completo y aplica la ventana despues,
igual que el ECM principal. Genera:
  diagnostico_ecm.csv, residuos_ecm.png, acf_ecm.png
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan

from config import (DIRECTORIOS, ECM_FIN, ECM_INICIO, MIN_MESES, NUCLEO,
                    PANEL_LIMPIO, aplicar_estilo, guardar, preparar_directorios, tabla)
from ecm import construir

warnings.filterwarnings("ignore")

SELECCION = DIRECTORIOS["calidad"] / "seleccion_rezagos.csv"


def main() -> None:
    preparar_directorios()
    aplicar_estilo()
    panel = pd.read_csv(PANEL_LIMPIO, parse_dates=["fecha"])
    qs = dict(zip(*pd.read_csv(SELECCION)[["producto", "q_elegido"]].values.T))

    filas, residuos = [], {}
    for cat in NUCLEO:
        d = panel[panel.categoria == cat].pivot_table(
            index="fecha", columns="region", values="precio_kg")
        datos = construir(np.log(d["Puno"]), np.log(d["Lima"]), int(qs.get(cat, 0))).loc[ECM_INICIO:ECM_FIN]
        if len(datos) < MIN_MESES:
            continue
        ols = sm.OLS(datos["dp"], sm.add_constant(datos.drop(columns="dp"))).fit()
        hac = sm.OLS(datos["dp"], sm.add_constant(datos.drop(columns="dp"))).fit(
            cov_type="HAC", cov_kwds={"maxlags": 4})
        lb4 = acorr_ljungbox(ols.resid, lags=[4], return_df=True)["lb_pvalue"].iloc[0]
        lb8 = acorr_ljungbox(ols.resid, lags=[8], return_df=True)["lb_pvalue"].iloc[0]
        bp = het_breuschpagan(ols.resid, ols.model.exog)[1]
        jb = stats.jarque_bera(ols.resid)[1]
        filas.append({
            "producto": cat, "n": int(ols.nobs),
            "ljung_box_4": float(lb4), "ljung_box_8": float(lb8),
            "breusch_pagan": float(bp), "jarque_bera": float(jb),
            "se_gamma_ols": float(ols.bse["ect"]), "se_gamma_hac": float(hac.bse["ect"]),
            "r2": float(ols.rsquared),
        })
        residuos[cat] = ols.resid

    tabla(pd.DataFrame(filas), "diagnostico_ecm.csv", "ecm")

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, cat in zip(axes.ravel(), NUCLEO):
        r = residuos[cat]
        ax.scatter(range(len(r)), r, s=8, color="#0e7490", alpha=0.7)
        ax.axhline(0, color="#111827", linewidth=0.8)
        ax.set_title(cat)
        ax.set_xlabel("observación")
        ax.set_ylabel("residuo")
    fig.suptitle("Residuos del ECM", y=1.01)
    plt.tight_layout()
    guardar(fig, "residuos_ecm.png", "ecm")

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, cat in zip(axes.ravel(), NUCLEO):
        sm.graphics.tsa.plot_acf(residuos[cat], lags=12, ax=ax, zero=False)
        ax.set_title(cat)
    fig.suptitle("Autocorrelación de residuos del ECM", y=1.01)
    plt.tight_layout()
    guardar(fig, "acf_ecm.png", "ecm")

    print("Diagnostico del ECM:")
    print(pd.DataFrame(filas).round(3).to_string(index=False))
    print("\nTablas y figuras de diagnostico generadas en", DIRECTORIOS["ecm"])


if __name__ == "__main__":
    main()
