#!/usr/bin/env python3
"""Seleccion de rezagos del ECM por AIC/BIC sobre la muestra del ECM.

Los rezagos se construyen sobre el calendario completo y la ventana 2012-2023
se aplica despues, igual que en 04_ecm.py.

Genera:
  seleccion_rezagos.csv, aic_bic.png
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

from config import (DIRECTORIOS, ECM_FIN, ECM_INICIO, LAGS_ECM_MAX, NUCLEO,
                    PANEL_LIMPIO, aplicar_estilo, guardar, preparar_directorios, tabla)
from ecm import construir, estimar

warnings.filterwarnings("ignore")


def main() -> None:
    preparar_directorios()
    aplicar_estilo()
    panel = pd.read_csv(PANEL_LIMPIO, parse_dates=["fecha"])

    filas, curvas = [], []
    for cat in NUCLEO:
        d = panel[panel.categoria == cat].pivot_table(
            index="fecha", columns="region", values="precio_kg")
        lp, ll = np.log(d["Puno"]), np.log(d["Lima"])
        mejor_aic, mejor_bic = None, None
        for q in range(LAGS_ECM_MAX + 1):
            datos = construir(lp, ll, q).loc[ECM_INICIO:ECM_FIN]
            m = estimar(datos, hac=False)
            curvas.append({"producto": cat, "q": q, "aic": m.aic, "bic": m.bic})
            if mejor_aic is None or m.aic < mejor_aic[1]:
                mejor_aic = (q, m.aic)
            if mejor_bic is None or m.bic < mejor_bic[1]:
                mejor_bic = (q, m.bic)
        elegido = mejor_bic[0]
        datos = construir(lp, ll, elegido).loc[ECM_INICIO:ECM_FIN]
        m = estimar(datos, hac=False)
        lb = acorr_ljungbox(m.resid, lags=[4], return_df=True)
        filas.append({"producto": cat, "q_aic": mejor_aic[0], "q_bic": mejor_bic[0],
                      "q_elegido": elegido, "aic": m.aic, "bic": m.bic,
                      "lb_p": float(lb["lb_pvalue"].iloc[0]), "r2": m.rsquared})
    seleccion = pd.DataFrame(filas)
    tabla(seleccion, "seleccion_rezagos.csv", "calidad")

    curvas = pd.DataFrame(curvas)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, cat in zip(axes.ravel(), NUCLEO):
        d = curvas[curvas.producto == cat]
        ax.plot(d.q, d.aic, marker="o", label="AIC", color="#0e7490")
        ax.plot(d.q, d.bic, marker="s", label="BIC", color="#b45309")
        ax.set_title(cat)
        ax.set_xlabel("rezagos q")
        ax.set_ylabel("criterio")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Seleccion de rezagos del ECM", y=1.01)
    plt.tight_layout()
    guardar(fig, "aic_bic.png", "calidad")

    print("Seleccion de rezagos:")
    print(seleccion.round(4).to_string(index=False))
    print("\nq elegido por producto:", dict(zip(seleccion.producto, seleccion.q_elegido)))
    print("\nTabla y figura de rezagos generadas en", DIRECTORIOS["calidad"])


if __name__ == "__main__":
    main()
