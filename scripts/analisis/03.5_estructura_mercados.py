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
from matplotlib.patches import Rectangle
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
    ax.set_title("Papa: correlación de cambios mensuales entre departamentos (2012–2023)\n"
                 "Cada celda es una correlación; la diagonal es 1 porque es el mismo departamento",
                 fontsize=10)
    ax.set_xlabel("Departamento")
    ax.set_ylabel("Departamento")
    idx_puno = list(c_papa.columns).index("Puno")
    ax.add_patch(Rectangle((idx_puno, 0), 1, len(c_papa), fill=False,
                           edgecolor="#111827", linewidth=2.0, clip_on=False))
    ax.add_patch(Rectangle((0, idx_puno), len(c_papa), 1, fill=False,
                           edgecolor="#111827", linewidth=2.0, clip_on=False))
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
    fig, ax = plt.subplots(figsize=(14, 4.8))
    sns.heatmap(piv, cmap="RdBu_r", vmin=-0.2, vmax=0.9, linewidths=0.4,
                linecolor="white", cbar_kws={"label": "correlación con Lima"}, ax=ax)
    ax.set_title("Correlación de cambios mensuales con Lima, por producto y departamento\n"
                 "Cada columna es un departamento; Lima es la referencia, no una columna",
                 fontsize=11.5)
    ax.set_xlabel("Departamento (correlación con Lima)")
    ax.set_ylabel("Producto")
    ax.tick_params(axis="y", labelsize=9)
    plt.xticks(rotation=45, ha="right", fontsize=8.5)
    guardar(fig, "correlacion_lima.png", "estructura")

    # 2b. La misma información, ordenada para papa.
    papa = corr_lima[corr_lima.producto == "Papa"].sort_values(
        "correlacion", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    colores = ["#b91c1c" if valor >= 0 else "#1d4ed8" for valor in papa.correlacion]
    barras = ax.barh(papa.region, papa.correlacion, color=colores)
    ax.invert_yaxis()
    ax.axvline(0, color="#6b7280", linewidth=0.8)
    for barra, valor in zip(barras, papa.correlacion):
        ax.text(valor + (0.015 if valor >= 0 else -0.015),
                barra.get_y() + barra.get_height() / 2,
                f"{valor:.2f}", va="center",
                ha="left" if valor >= 0 else "right", fontsize=7)
    ax.set_xlim(-0.12, 0.95)
    ax.set_xlabel("Correlación de cambios mensuales con Lima")
    ax.set_title("Papa: cada barra es un departamento; Lima es la referencia\n"
                 "(no se grafica: su correlación consigo misma sería 1)",
                 fontsize=11)
    guardar(fig, "correlacion_papa_lima.png", "estructura")

    # 2c. Puno-Lima por alimento: la misma correlación, separada por producto.
    puno = corr_lima[corr_lima.region == "Puno"].sort_values(
        "correlacion", ascending=False).copy()
    etiquetas = {"Azucar": "Azúcar", "Cebolla": "Cebolla", "Papa": "Papa",
                 "Aceite": "Aceite", "Huevos": "Huevos", "Zanahoria": "Zanahoria",
                 "Arroz": "Arroz", "Limon": "Limón", "Tomate": "Tomate"}
    puno["etiqueta"] = puno.producto.replace(etiquetas)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colores = ["#b91c1c" if valor >= 0 else "#1d4ed8" for valor in puno.correlacion]
    barras = ax.barh(puno.etiqueta, puno.correlacion, color=colores)
    ax.invert_yaxis()
    ax.axvline(0, color="#6b7280", linewidth=0.8)
    for barra, fila in zip(barras, puno.itertuples(index=False)):
        ax.text(fila.correlacion + 0.012,
                barra.get_y() + barra.get_height() / 2,
                f"{fila.correlacion:.2f}  (n={int(fila.n)})",
                va="center", ha="left", fontsize=8)
    ax.set_xlim(0, 0.78)
    ax.set_xlabel("Correlación de cambios mensuales Puno–Lima")
    ax.set_title("Puno–Lima: correlación por alimento (2012–2023)\n"
                 "Cada barra es un alimento; Lima es la referencia",
                 fontsize=11)
    guardar(fig, "puno_lima_por_producto.png", "estructura")

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
    ax.set_xlabel("Distancia del departamento a Lima (km)")
    ax.set_ylabel("Correlación promedio con Lima\n(23 departamentos)")
    ax.set_title("Integración con Lima y distancia, por departamento (2012–2023)\n"
                 "Cada punto es un departamento; Lima es el origen y la referencia, no se grafica",
                 fontsize=10)
    ax.set_xlim(0, 1050)
    ax.legend(title="zona", fontsize=8)
    fig.text(0.01, 0.01,
             "Lima no es un punto: está en distancia 0 y su correlación consigo misma sería 1; "
             "se excluye para no distorsionar la escala.",
             fontsize=7.5, color="#374151")
    guardar(fig, "integracion_distancia.png", "estructura")

    # Mapa: Lima como estrella y cada departamento coloreado por su correlación.
    lima_lat, lima_lon = COORDENADAS["Lima"]
    fig, ax = plt.subplots(figsize=(11, 8))
    norm = plt.Normalize(vmin=-0.1, vmax=0.85)
    cmap = plt.cm.RdBu_r
    for _, fila in geo.iterrows():
        lat, lon = COORDENADAS[fila.region]
        color = cmap(norm(fila.r_lima))
        ax.plot([lima_lon, lon], [lima_lat, lat], color="#9ca3af",
                linewidth=0.6, alpha=0.55, zorder=1)
        ax.scatter(lon, lat, s=80 + 220 * max(fila.r_lima, 0),
                   color=color, edgecolor="white", linewidth=0.8, zorder=2)
        ax.annotate(fila.region, (lon, lat), textcoords="offset points",
                    xytext=(5, 4), fontsize=6.5, zorder=3)
    ax.scatter(lima_lon, lima_lat, marker="*", s=520, color="#111827",
               edgecolor="white", linewidth=0.8, zorder=4, label="Lima (referencia)")
    ax.annotate("Lima", (lima_lon, lima_lat), textcoords="offset points",
                xytext=(8, -12), fontsize=9, fontweight="bold")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_xlim(-82.0, -68.0)
    ax.set_ylim(-19.0, -2.5)
    ax.set_title("Cada línea sale de Lima; el color es la correlación promedio con Lima\n"
                 "Cada punto es un departamento; la estrella es Lima",
                 fontsize=11)
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.65, label="correlación promedio con Lima")
    ax.legend(loc="lower left", fontsize=8)
    guardar(fig, "mapa_integracion_lima.png", "estructura")

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
