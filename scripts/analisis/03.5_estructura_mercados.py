#!/usr/bin/env python3
"""Estructura de mercados: correlaciones y geografia de la integracion.

Corre después de la limpieza y antes de los modelos.
Genera en outputs/analisis/04_estructura:
  correlacion_papa.csv, correlacion_lima.csv, correlacion_puno.csv,
  geografia_integracion.csv, geografia_modelos.csv,
  correlacion_papa.png, correlacion_lima.png, correlacion_puno.png,
  integracion_distancia.png
"""

from __future__ import annotations

import sys
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kruskal

from config import (COORDENADAS, DIRECTORIOS, NUCLEO, PANEL_LIMPIO, PRODUCTOS,
                    VENTANA_DENSA_FIN, VENTANA_DENSA_INICIO, ZONAS,
                    aplicar_estilo, guardar, preparar_directorios, tabla)

import seaborn as sns

REGIONES_PAPA = ["Lima", "Ica", "Arequipa", "La Libertad", "Huanuco",
                 "Ayacucho", "Cusco", "Puno", "Piura", "Loreto"]


def retornos(panel: pd.DataFrame, cat: str, regiones=None) -> pd.DataFrame:
    d = panel[panel.categoria == cat].pivot_table(
        index="fecha", columns="region", values="precio_kg")
    if regiones is not None:
        d = d[[r for r in regiones if r in d.columns]]
    return np.log(d).diff()


def haversine(a, b) -> float:
    la1, lo1 = map(radians, a)
    la2, lo2 = map(radians, b)
    h = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * asin(sqrt(h))


def main() -> None:
    preparar_directorios()
    aplicar_estilo()
    panel = pd.read_csv(PANEL_LIMPIO, parse_dates=["fecha"])

    # 1. Papa entre regiones
    c_papa = retornos(panel, "Papa", REGIONES_PAPA).loc[VENTANA_DENSA_INICIO:VENTANA_DENSA_FIN].corr()
    tabla(c_papa.reset_index(), "correlacion_papa.csv", "estructura")
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    sns.heatmap(c_papa, cmap="RdBu_r", vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title("Papa: correlacion de cambios mensuales entre departamentos (2012-2023)")
    guardar(fig, "correlacion_papa.png", "estructura")

    # 2. Correlacion con Lima por producto y region
    filas = []
    for cat in PRODUCTOS:
        r = retornos(panel, cat).loc[VENTANA_DENSA_INICIO:VENTANA_DENSA_FIN]
        if "Lima" not in r.columns:
            continue
        for region in r.columns:
            if region == "Lima":
                continue
            par = r[["Lima", region]].dropna()
            if len(par) >= 60:
                filas.append({"producto": cat, "region": region, "n": len(par),
                              "correlacion": par["Lima"].corr(par[region])})
    corr_lima = pd.DataFrame(filas)
    tabla(corr_lima, "correlacion_lima.csv", "estructura")

    piv = corr_lima.pivot(index="producto", columns="region", values="correlacion")
    fig, ax = plt.subplots(figsize=(14, 4.2))
    sns.heatmap(piv, cmap="RdBu_r", vmin=-0.2, vmax=0.9, linewidths=0.4,
                linecolor="white", cbar_kws={"label": "correlacion con Lima"}, ax=ax)
    ax.set_title("Correlacion de cambios mensuales con Lima, por producto y departamento")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    guardar(fig, "correlacion_lima.png", "estructura")

    # 3. Productos en Puno
    d_puno = panel[panel.region == "Puno"].pivot_table(
        index="fecha", columns="categoria", values="precio_kg")
    c_puno = np.log(d_puno).diff().loc[VENTANA_DENSA_INICIO:VENTANA_DENSA_FIN].corr()
    tabla(c_puno.reset_index(), "correlacion_puno.csv", "estructura")
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    sns.heatmap(c_puno, cmap="RdBu_r", vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title("Puno: correlacion de cambios mensuales entre productos")
    guardar(fig, "correlacion_puno.png", "estructura")

    # 4. Geografia
    zs: dict[str, list[float]] = {}
    for cat in NUCLEO:
        r = retornos(panel, cat).loc[VENTANA_DENSA_INICIO:VENTANA_DENSA_FIN]
        if "Lima" not in r.columns:
            continue
        for region in r.columns:
            if region == "Lima":
                continue
            par = r[["Lima", region]].dropna()
            if len(par) >= 60:
                c = float(np.clip(par["Lima"].corr(par[region]), -0.99, 0.99))
                zs.setdefault(region, []).append(float(np.arctanh(c)))
    faltan_geo = sorted(set(zs) - set(COORDENADAS))
    if faltan_geo:
        raise ValueError(f"Departamentos sin coordenadas: {faltan_geo}")
    faltan_zona = sorted(set(zs) - set(ZONAS))
    if faltan_zona:
        raise ValueError(f"Departamentos sin zona: {faltan_zona}")
    geo = pd.DataFrame([
        {"region": region, "r_lima": float(np.tanh(np.mean(z))),
         "distancia_km": haversine(COORDENADAS["Lima"], COORDENADAS[region]),
         "zona": ZONAS[region]}
        for region, z in zs.items()
    ]).sort_values("r_lima", ascending=False)
    if set(geo.region) != set(COORDENADAS) - {"Lima"}:
        raise ValueError("La geografia no cubre exactamente los 23 departamentos sin Lima")
    tabla(geo, "geografia_integracion.csv", "estructura")

    k = kruskal(*[g.r_lima.values for _, g in geo.groupby("zona")])
    x = sm.add_constant(np.log(geo.distancia_km).rename("log_distancia"))
    m1 = sm.OLS(geo.r_lima, x).fit()
    dummies = pd.get_dummies(geo.zona, drop_first=True).astype(float)
    x2 = sm.add_constant(pd.concat([np.log(geo.distancia_km).rename("log_distancia"), dummies], axis=1))
    m2 = sm.OLS(geo.r_lima, x2).fit()
    filas_modelo = [{"modelo": "distancia", "termino": "log_distancia",
                     "coef": m1.params["log_distancia"], "p": m1.pvalues["log_distancia"],
                     "r2": m1.rsquared}]
    for termino in m2.params.index:
        if termino != "const":
            filas_modelo.append({"modelo": "distancia_zona", "termino": termino,
                                 "coef": m2.params[termino], "p": m2.pvalues[termino],
                                 "r2": m2.rsquared})
    tabla(pd.DataFrame(filas_modelo), "geografia_modelos.csv", "estructura")

    fig, ax = plt.subplots(figsize=(9, 5.6))
    colores = {"Costa": "#0e7490", "Sierra": "#b45309", "Selva": "#166534"}
    for zona, grupo in geo.groupby("zona"):
        ax.scatter(grupo.distancia_km, grupo.r_lima, s=55, alpha=0.85,
                   color=colores[zona], label=zona, edgecolor="white", linewidth=0.6)
    for _, fila in geo.iterrows():
        if fila.region in {"Ica", "Huanuco", "Arequipa", "Puno", "Piura",
                           "Tumbes", "Loreto", "Cusco"}:
            ax.annotate(fila.region, (fila.distancia_km, fila.r_lima),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("distancia a Lima (km)")
    ax.set_ylabel("correlacion promedio con Lima")
    ax.set_title("Integracion con Lima y distancia, por departamento (2012-2023)")
    ax.legend(title="zona", fontsize=8)
    guardar(fig, "integracion_distancia.png", "estructura")

    print("Pares mas integrados con Lima:")
    print(corr_lima.sort_values("correlacion", ascending=False).head(8).round(3).to_string(index=False))
    print("\nPor zona:")
    print(geo.groupby("zona").r_lima.agg(["count", "mean", "median"]).round(3).to_string())
    print(f"\nKruskal-Wallis: H={k.statistic:.2f}, p={k.pvalue:.3f}")
    print(f"OLS r ~ log(distancia): R2={m1.rsquared:.2f}, p={m1.pvalues['log_distancia']:.3f}")
    print(f"OLS r ~ log(distancia) + zona: R2={m2.rsquared:.2f}")
    print("\nTablas y figuras de estructura generadas en", DIRECTORIOS["estructura"])


if __name__ == "__main__":
    main()
