#!/usr/bin/env python3
"""Modelo de correccion de error en niveles, con q elegido por BIC.

Especificacion:
  dp(t) = a + b0*dl(t) + ... + bq*dl(t-q) + g*ect(t-1) + e
  ect(t-1) = log p(region,t-1) - log p(Lima,t-1)

La construccion se hace sobre el calendario mensual completo y la ventana
2012-2023 se aplica despues. Errores estandar HAC.

Genera en outputs/analisis/05_ecm:
  ecm_producto.csv, ecm_puno.csv, ecm_region.csv, asimetria.csv,
  vida_media_puno.png, vida_media_mapa.png
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
from statsmodels.tsa.stattools import coint

from config import (DIRECTORIOS, ECM_FIN, ECM_INICIO, MIN_MESES, NUCLEO,
                    PANEL_LIMPIO, aplicar_estilo, guardar, preparar_directorios, tabla)
from ecm import bootstrap_ic, construir, estimar, vida_media

warnings.filterwarnings("ignore")

SELECCION = DIRECTORIOS["calidad"] / "seleccion_rezagos.csv"


def main() -> None:
    preparar_directorios()
    aplicar_estilo()
    panel = pd.read_csv(PANEL_LIMPIO, parse_dates=["fecha"])
    seleccion = pd.read_csv(SELECCION)
    qs = dict(zip(seleccion.producto, seleccion.q_elegido))

    puno, regiones, asimetrias = [], [], []
    for cat in NUCLEO:
        q = int(qs.get(cat, 0))
        d = panel[panel.categoria == cat].pivot_table(
            index="fecha", columns="region", values="precio_kg")
        lp, ll = np.log(d["Puno"]), np.log(d["Lima"])

        datos = construir(lp, ll, q).loc[ECM_INICIO:ECM_FIN]
        if len(datos) < MIN_MESES:
            continue
        modelo = estimar(datos)
        g = float(modelo.params["ect"])
        try:
            par_coint = pd.concat([lp, ll], axis=1).dropna()
            p_coint = float(coint(par_coint.iloc[:, 0], par_coint.iloc[:, 1])[1])
        except Exception:
            p_coint = np.nan
        ic_inf, ic_sup = bootstrap_ic(datos, 500)
        puno.append({
            "producto": cat, "q": q, "n": int(modelo.nobs),
            "beta0": float(modelo.params.get("dl0", np.nan)),
            "beta0_p": float(modelo.pvalues.get("dl0", np.nan)),
            "gamma": g, "gamma_p": float(modelo.pvalues["ect"]),
            "vida_media": vida_media(g), "vida_ic_inf": ic_inf, "vida_ic_sup": ic_sup,
            "r2": float(modelo.rsquared), "p_coint": p_coint,
        })

        for region in [c for c in d.columns if c != "Lima"]:
            datos_r = construir(np.log(d[region]), np.log(d["Lima"]), q).loc[ECM_INICIO:ECM_FIN]
            if len(datos_r) < MIN_MESES:
                continue
            m = estimar(datos_r)
            gr = float(m.params["ect"])
            regiones.append({
                "producto": cat, "region": region, "q": q, "n": int(m.nobs),
                "beta0": float(m.params.get("dl0", np.nan)),
                "gamma": gr, "gamma_p": float(m.pvalues["ect"]),
                "vida_media": vida_media(gr), "r2": float(m.rsquared),
            })

        dl = ll.diff()
        asim = pd.DataFrame({
            "dp": lp.diff(), "sube": dl.clip(lower=0), "baja": dl.clip(upper=0),
            "ect": lp.shift(1) - ll.shift(1),
        }).loc[ECM_INICIO:ECM_FIN].dropna()
        # La asimetria usa una especificacion propia: sube, baja y ect.
        ma = sm.OLS(asim["dp"], sm.add_constant(asim[["sube", "baja", "ect"]])).fit()
        nombres = list(ma.params.index)
        R = np.zeros((1, len(nombres)))
        R[0, nombres.index("sube")] = 1
        R[0, nombres.index("baja")] = -1
        asimetrias.append({
            "producto": cat, "n": int(ma.nobs),
            "beta_sube": float(ma.params["sube"]), "beta_baja": float(ma.params["baja"]),
            "diferencia": float(ma.params["sube"] - ma.params["baja"]),
            "p_igualdad": float(ma.wald_test(R).pvalue),
        })

    puno = pd.DataFrame(puno)
    regiones = pd.DataFrame(regiones)
    asimetrias = pd.DataFrame(asimetrias)
    tabla(puno, "ecm_puno.csv", "ecm")
    tabla(regiones, "ecm_region.csv", "ecm")
    tabla(asimetrias, "asimetria.csv", "ecm")

    producto = (regiones.groupby("producto")
                .agg(pares=("region", "nunique"), n_mediana=("n", "median"),
                     beta0_mediana=("beta0", "median"), gamma_mediana=("gamma", "median"),
                     vida_mediana=("vida_media", "median"), r2_mediana=("r2", "median"),
                     pares_significativos=("gamma_p", lambda s: int((s < 0.05).sum())))
                .reset_index().sort_values("vida_mediana"))
    tabla(producto, "ecm_producto.csv", "ecm")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    orden = puno.sort_values("vida_media")
    axes[0].barh(orden.producto, orden.vida_media, color="#0e7490")
    axes[0].errorbar(orden.vida_media, orden.producto,
                     xerr=[orden.vida_media - orden.vida_ic_inf,
                           orden.vida_ic_sup - orden.vida_media],
                     fmt="none", ecolor="#111827", capsize=3)
    axes[0].set_xlabel("meses")
    axes[0].set_title("Puno: vida media del desvio (IC bootstrap 95 %)")

    orden_cats = producto.sort_values("vida_mediana").producto.tolist()
    datos_caja = [regiones[regiones.producto == c].vida_media.dropna().values for c in orden_cats]
    axes[1].boxplot(datos_caja, vert=False, tick_labels=orden_cats, patch_artist=True,
                    boxprops={"facecolor": "#bae6fd", "edgecolor": "#0e7490"},
                    medianprops={"color": "#b45309"})
    axes[1].set_xlabel("meses")
    axes[1].set_title("Vida media entre departamentos, por producto")
    axes[0].set_xlim(0, 12)
    axes[1].set_xlim(0, 12)
    tope = puno.loc[puno.producto == "Arroz", "vida_ic_sup"]
    if len(tope) and np.isfinite(tope.iloc[0]):
        fig.text(0.5, 0.01, f"Arroz: IC superior {tope.iloc[0]:.1f} meses, fuera de escala.",
                 ha="center", fontsize=8, color="#4b5563")
    plt.tight_layout()
    guardar(fig, "vida_media_puno.png", "ecm")

    piv = regiones.pivot(index="producto", columns="region", values="vida_media")
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=12)
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=8)
    ax.set_title("Vida media por producto y departamento (meses, recortada en 12)")
    fig.colorbar(im, ax=ax, label="meses")
    plt.tight_layout()
    guardar(fig, "vida_media_mapa.png", "ecm")

    print("ECM de Puno (q por BIC, HAC):")
    print(puno.round(3).to_string(index=False))
    print("\nResumen por producto:")
    print(producto.round(3).to_string(index=False))
    print("\nAsimetria (Puno):")
    print(asimetrias.round(3).to_string(index=False))
    print("\nTablas y figuras del ECM generadas en", DIRECTORIOS["ecm"])


if __name__ == "__main__":
    main()
