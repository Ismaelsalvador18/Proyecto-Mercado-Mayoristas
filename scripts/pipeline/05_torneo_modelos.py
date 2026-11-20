#!/usr/bin/env python3
"""Torneo de modelos: nowcast contemporaneo y pronostico a un mes.

Cuarteto principal: OLS, Ridge, Random Forest, LightGBM.
Baselines: cambio cero (nowcast) y caminata aleatoria (pronostico).

Seleccion: rango promedio de R2 en la ventana de evaluación de desarrollo (2018-2020).
Entrenamiento para esa evaluación: observaciones anteriores a 2018.
Elegibilidad del ranking: al menos 60 observaciones de entrenamiento y 20 de evaluación.
Confirmacion: holdout 2021-01 a 2026-10, una sola vez.
Compuerta: R2 > 0 en >= 2 productos y acierto > 55 % en >= 2 productos.

Genera en outputs/analisis/06_modelos:
  torneo_nowcast_metricas.csv, torneo_nowcast_ranking.csv, torneo_nowcast_dm.csv,
  torneo_forecast_metricas.csv, torneo_forecast_ranking.csv,
  torneo_complemento.csv, torneo_ganador.md, hiperparametros.csv,
  ranking_torneo.png, ganador_vs_observado.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (DEV_FIN, DEV_INICIO, DIRECTORIOS, HOLDOUT_FIN, HOLDOUT_INICIO,
                    LGBM, MIN_FORECAST_TEST, MIN_FORECAST_TRAIN, MIN_NOWCAST_TEST,
                    MIN_NOWCAST_TRAIN, NUCLEO, PANEL_LIMPIO, RF, RIDGE_ALPHA, aplicar_estilo,
                    guardar, preparar_directorios, tabla)
from modelos import (construir_forecast, construir_nowcast, dm_test, metricas,
                     modelos_complemento, modelos_principales)


def evaluar_nowcast(panel: pd.DataFrame, conjunto: str) -> pd.DataFrame:
    filas = []
    for periodo, train_end, ini, fin in [
        ("desarrollo", DEV_INICIO, DEV_INICIO, DEV_FIN),
        ("holdout", HOLDOUT_INICIO, HOLDOUT_INICIO, HOLDOUT_FIN),
    ]:
        for cat in NUCLEO:
            X, y = construir_nowcast(panel, cat)
            train = X.index < pd.Timestamp(train_end)
            test = (X.index >= pd.Timestamp(ini)) & (X.index <= pd.Timestamp(fin))
            if train.sum() < MIN_NOWCAST_TRAIN or test.sum() < MIN_NOWCAST_TEST:
                continue
            real = y[test].values
            filas.append({"periodo": periodo, "producto": cat, "modelo": "Cambio cero",
                          "conjunto": "baseline", **metricas(real, np.zeros_like(real), np.zeros_like(real))})
            modelos = modelos_principales() if conjunto == "principal" else modelos_complemento()
            for nombre, modelo in modelos.items():
                modelo.fit(X[train], y[train])
                pred = modelo.predict(X[test])
                filas.append({"periodo": periodo, "producto": cat, "modelo": nombre,
                              "conjunto": conjunto, **metricas(real, pred, np.zeros_like(real))})
    return pd.DataFrame(filas)


def evaluar_forecast(panel: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for periodo, train_end, ini, fin in [
        ("desarrollo", DEV_INICIO, DEV_INICIO, DEV_FIN),
        ("holdout", HOLDOUT_INICIO, HOLDOUT_INICIO, HOLDOUT_FIN),
    ]:
        for cat in NUCLEO:
            X, y, p0 = construir_forecast(panel, cat)
            train = X.index < pd.Timestamp(train_end)
            test = (X.index >= pd.Timestamp(ini)) & (X.index <= pd.Timestamp(fin))
            if train.sum() < MIN_FORECAST_TRAIN or test.sum() < MIN_FORECAST_TEST:
                continue
            real = np.exp(y[test]).values
            base = np.exp(p0[test]).values
            filas.append({"periodo": periodo, "producto": cat, "modelo": "Caminata",
                          "conjunto": "baseline", **metricas(real, base, base)})
            for nombre, modelo in modelos_principales().items():
                modelo.fit(X[train], y[train])
                pred = np.exp(modelo.predict(X[test]))
                filas.append({"periodo": periodo, "producto": cat, "modelo": nombre,
                              "conjunto": "principal", **metricas(real, pred, base)})
    return pd.DataFrame(filas)


def main() -> None:
    preparar_directorios()
    aplicar_estilo()
    panel = pd.read_csv(PANEL_LIMPIO, parse_dates=["fecha"])

    nowcast = evaluar_nowcast(panel, "principal")
    tabla(nowcast, "torneo_nowcast_metricas.csv", "modelos")
    tabla(evaluar_nowcast(panel, "complemento"), "torneo_complemento.csv", "modelos")

    dev = nowcast[(nowcast.periodo == "desarrollo") & (nowcast.conjunto == "principal")].copy()
    productos_ranking = [cat for cat in NUCLEO if cat in dev.producto.unique()]
    productos_fuera_ranking = [cat for cat in NUCLEO if cat not in productos_ranking]
    dev["rank"] = dev.groupby("producto")["r2"].rank(ascending=False, method="average")
    ranking = (dev.groupby("modelo")
               .agg(rank_promedio=("rank", "mean"), r2_medio=("r2", "mean"),
                    r2_mediano=("r2", "median"), acierto_medio=("acierto", "mean"),
                    productos_rankeados=("producto", "nunique"))
               .sort_values("rank_promedio"))
    tabla(ranking.reset_index(), "torneo_nowcast_ranking.csv", "modelos")

    forecast = evaluar_forecast(panel)
    tabla(forecast, "torneo_forecast_metricas.csv", "modelos")
    fdev = forecast[(forecast.periodo == "desarrollo") & (forecast.conjunto == "principal")].copy()
    fdev["rank"] = fdev.groupby("producto")["mae"].rank(ascending=True, method="average")
    frank = (fdev.groupby("modelo").agg(rank_promedio=("rank", "mean"),
                                        mae_medio=("mae", "mean"))
             .sort_values("rank_promedio"))
    tabla(frank.reset_index(), "torneo_forecast_ranking.csv", "modelos")

    # Diebold-Mariano del ganador contra el baseline en holdout
    dm_filas = []
    for cat in NUCLEO:
        X, y = construir_nowcast(panel, cat)
        train = X.index < pd.Timestamp(HOLDOUT_INICIO)
        test = (X.index >= pd.Timestamp(HOLDOUT_INICIO)) & (X.index <= pd.Timestamp(HOLDOUT_FIN))
        if train.sum() < MIN_NOWCAST_TRAIN or test.sum() < MIN_NOWCAST_TEST:
            continue
        real = y[test].values
        for nombre, modelo in modelos_principales().items():
            modelo.fit(X[train], y[train])
            pred = modelo.predict(X[test])
            stat, p = dm_test(real - pred, real)
            dm_filas.append({"producto": cat, "modelo": nombre, "dm": stat, "p": p})
    tabla(pd.DataFrame(dm_filas), "torneo_nowcast_dm.csv", "modelos")

    ganador = ranking.index[0]
    hold = nowcast[(nowcast.periodo == "holdout") & (nowcast.modelo == ganador)].set_index("producto")
    productos_r2 = int((hold.r2 > 0).sum())
    productos_dir = int((hold.acierto > 55).sum())
    gate = productos_r2 >= 2 and productos_dir >= 2

    hiper = pd.DataFrame([
        ("OLS", "sin hiperparámetros", "lineal, interpretable"),
        ("Ridge", f"alpha={RIDGE_ALPHA}, estandarización", "lineal regularizado"),
        ("Random Forest", f"n={RF['n_estimators']}, leaf={RF['min_samples_leaf']}", "no lineal, bagging"),
        ("LightGBM", f"n={LGBM['n_estimators']}, lr={LGBM['learning_rate']}, leaves={LGBM['num_leaves']}", "no lineal, boosting"),
    ], columns=["modelo", "hiperparametros", "familia"])
    tabla(hiper, "hiperparametros.csv", "modelos")

    with (DIRECTORIOS["modelos"] / "torneo_ganador.md").open("w", encoding="utf-8") as fh:
        fh.write("# Resultado del ranking de nowcast\n\n")
        fh.write("**Objetivo principal:** anticipacion contemporanea del cambio mensual.\n\n")
        fh.write(f"- **Modelo adoptado:** {ganador}\n")
        fh.write(f"- Rango promedio por R2 en desarrollo: {ranking.loc[ganador, 'rank_promedio']:.2f}\n")
        fh.write(f"- R2 mediano en desarrollo: {ranking.loc[ganador, 'r2_mediano']:.3f}\n")
        fh.write(f"- Ranking calculado sobre {len(productos_ranking)} productos con muestras suficientes: "
                 f"{', '.join(productos_ranking)}\n")
        if productos_fuera_ranking:
            fh.write(f"- Excluidos del ranking por no alcanzar los mínimos muestrales: "
                     f"{', '.join(productos_fuera_ranking)}\n")
        fh.write(f"- Mínimos por producto para rankear: {MIN_NOWCAST_TRAIN} observaciones de entrenamiento "
                 f"y {MIN_NOWCAST_TEST} de evaluación\n")
        fh.write(f"- Holdout: {productos_r2} de {len(hold)} productos con R2 > 0; "
                 f"{productos_dir} con acierto > 55 %\n")
        fh.write(f"- Compuerta: {'superada' if gate else 'no superada'}\n\n")
        fh.write("### Holdout por producto\n\n| Producto | R2 | Acierto | n |\n|---|---:|---:|---:|\n")
        for producto, fila in hold.iterrows():
            fh.write(f"| {producto} | {fila.r2:.3f} | {fila.acierto:.1f} % | {int(fila.n)} |\n")
        fh.write("\n## Justificacion\n\n")
        fh.write("La ventana de desarrollo es 2018-2020 (36 meses calendario) y se usa para "
                 "evaluar/rankear; cada modelo se entrena con las observaciones disponibles antes "
                 "de 2018. Solo se rankean productos con al menos 60 observaciones de entrenamiento "
                 "y 20 de evaluación. El R2 es la metrica principal del nowcast porque el MAE sobre cambios "
                 "favorece predecir cero. El holdout 2021-01 a 2026-10 se uso una sola vez; noviembre "
                 "queda fuera porque no se verifico su fecha de cierre/publicacion.\n\n")
        fh.write("## Pronostico a un mes: holdout\n\n")
        fh.write("Cambio porcentual del MAE frente a la caminata aleatoria: positivo = mejora; "
                 "negativo = empeora.\n\n")
        forecast_hold = forecast[(forecast.periodo == "holdout") &
                                 (forecast.conjunto == "principal")]
        wide = forecast_hold.pivot(index="producto", columns="modelo", values="mejora_mae").reindex(NUCLEO)
        n_hold = forecast_hold.groupby("producto").n.first().reindex(NUCLEO)
        fh.write("| Producto | N | OLS | Ridge | Random Forest | LightGBM |\n")
        fh.write("|---|---:|---:|---:|---:|---:|\n")
        for producto in NUCLEO:
            valores = [f"{wide.loc[producto, modelo]:+.1f} %" for modelo in
                       ["OLS", "Ridge", "Random Forest", "LightGBM"]]
            fh.write(f"| {producto} | {int(n_hold.loc[producto])} | " + " | ".join(valores) + " |\n")
        fh.write("\nRidge y OLS mejoran en papa, arroz, cebolla y huevos; OLS mejora marginalmente en "
                 "zanahoria (+0,7 %), mientras Ridge empeora (−1,9 %); ambos empeoran en aceite. "
                 "Random Forest y LightGBM empeoran frente al baseline en los seis productos. "
                 "La referencia de Lima aporta valor contemporaneo, pero no ofrece un pronostico "
                 "general a un mes.\n")

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.barh(ranking.index, ranking.rank_promedio, color="#0e7490")
    ax.invert_yaxis()
    ax.set_xlabel("rango promedio por R2 en desarrollo (1 = mejor)")
    ax.set_title("Torneo de nowcast: cuarteto principal (4 productos elegibles)")
    guardar(fig, "ranking_torneo.png", "modelos")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.0))
    for ax, cat in zip(axes, ["Papa", "Cebolla"]):
        X, y = construir_nowcast(panel, cat)
        train = X.index < pd.Timestamp(HOLDOUT_INICIO)
        test = (X.index >= pd.Timestamp(HOLDOUT_INICIO)) & (X.index <= pd.Timestamp(HOLDOUT_FIN))
        modelo = modelos_principales()[ganador]
        modelo.fit(X[train], y[train])
        pred = modelo.predict(X[test])
        ax.plot(y[test].index, y[test].values * 100, color="#111827", linewidth=1.4, label="Observado")
        ax.plot(y[test].index, pred * 100, color="#b45309", linewidth=1.2, label=ganador)
        ax.axhline(0, color="#9ca3af", linewidth=0.8)
        ax.set_title(f"{cat} en Puno: cambio mensual observado y estimado, 2021-oct 2026")
        ax.set_ylabel("variacion %")
        ax.legend(fontsize=8)
    plt.tight_layout()
    guardar(fig, "ganador_vs_observado.png", "modelos")

    print("Ranking nowcast (desarrollo):")
    print(ranking.round(3).to_string())
    print("\nRanking pronostico (desarrollo):")
    print(frank.round(3).to_string())
    print(f"\nGanador: {ganador} | compuerta: {'superada' if gate else 'no superada'}")
    print("\nHoldout del ganador:")
    print(hold[["mae", "r2", "acierto", "n"]].round(3).to_string())
    print("\nTablas y figuras del torneo generadas en", DIRECTORIOS["modelos"])


if __name__ == "__main__":
    main()
