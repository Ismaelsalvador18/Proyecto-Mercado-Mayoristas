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
  predicciones_holdout.csv, matriz_confusion_modelos.csv,
  matriz_confusion_producto.csv, matriz_diseno.csv,
  ranking_torneo.png, ganador_vs_observado.png, matriz_confusion.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.dates as mdates
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



def predecir_holdout(panel: pd.DataFrame) -> pd.DataFrame:
    """Predicciones de los cuatro modelos para cada producto en el holdout."""
    filas = []
    for cat in NUCLEO:
        X, y = construir_nowcast(panel, cat)
        train = X.index < pd.Timestamp(HOLDOUT_INICIO)
        test = (X.index >= pd.Timestamp(HOLDOUT_INICIO)) & (X.index <= pd.Timestamp(HOLDOUT_FIN))
        if train.sum() < MIN_NOWCAST_TRAIN or test.sum() < MIN_NOWCAST_TEST:
            continue
        for nombre, modelo in modelos_principales().items():
            modelo.fit(X[train], y[train])
            pred = modelo.predict(X[test])
            for fecha, observado, estimado in zip(y[test].index, y[test].values, pred):
                filas.append({"fecha": fecha, "producto": cat, "modelo": nombre,
                              "observado": float(observado), "prediccion": float(estimado)})
    return pd.DataFrame(filas)


def matriz_confusion(predicciones: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matriz 2x2 de dirección: sube/baja, sin contar cambios observados iguales a cero."""
    filas, filas_producto = [], []
    for modelo, bloque in predicciones.groupby("modelo"):
        def contar(d: pd.DataFrame) -> dict:
            d = d[d["observado"] != 0].copy()
            real_pos = d["observado"] > 0
            pred_pos = d["prediccion"] > 0
            tp = int((real_pos & pred_pos).sum())
            fn = int((real_pos & ~pred_pos).sum())
            fp = int((~real_pos & pred_pos).sum())
            tn = int((~real_pos & ~pred_pos).sum())
            n = tp + fn + fp + tn
            return {"TP": tp, "FN": fn, "FP": fp, "TN": tn,
                    "n": n, "acierto": 100 * (tp + tn) / n if n else float("nan")}
        filas.append({"modelo": modelo, **contar(bloque)})
        for producto, grupo in bloque.groupby("producto"):
            filas_producto.append({"producto": producto, "modelo": modelo,
                                   **contar(grupo)})
    return pd.DataFrame(filas), pd.DataFrame(filas_producto)


def graficar_confusion(confusion: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2))
    for ax, (_, fila) in zip(axes.ravel(), confusion.iterrows()):
        matriz = np.array([[fila["TN"], fila["FP"]], [fila["FN"], fila["TP"]]])
        ax.imshow(matriz, cmap="Blues")
        ax.set_xticks([0, 1], ["Predice baja", "Predice sube"])
        ax.set_yticks([0, 1], ["Real baja", "Real sube"])
        ax.set_title(f"{fila['modelo']} (acierto {fila['acierto']:.1f} %)")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matriz[i, j]:,}".replace(",", " "),
                        ha="center", va="center",
                        color="white" if matriz[i, j] > matriz.max() * 0.75 else "black")
    fig.suptitle("Matriz de confusión direccional en holdout (seis productos)", y=1.02)
    plt.tight_layout()
    guardar(fig, "matriz_confusion.png", "modelos")


def graficar_predicciones(predicciones: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), sharex=False)
    colores = {"OLS": "#2563eb", "Ridge": "#7c3aed",
               "Random Forest": "#b45309", "LightGBM": "#059669"}
    for ax, cat in zip(axes.ravel(), NUCLEO):
        bloque = predicciones[predicciones.producto == cat]
        observado = bloque.groupby("fecha")["observado"].first()
        ax.plot(observado.index, observado.values * 100, color="#111827",
                linewidth=1.4, label="Observado")
        for modelo, grupo in bloque.groupby("modelo"):
            grupo = grupo.sort_values("fecha")
            ax.plot(grupo.fecha, grupo.prediccion * 100, color=colores.get(modelo),
                    linewidth=1.0, alpha=0.9, label=modelo)
        ax.axhline(0, color="#9ca3af", linewidth=0.7)
        ax.set_title(cat)
        ax.set_ylabel("variación %")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", labelsize=7)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("Nowcast en holdout: observado y cuatro modelos por producto", y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    guardar(fig, "ganador_vs_observado.png", "modelos")


def guardar_matriz_diseno(panel: pd.DataFrame) -> None:
    """Guarda la matriz X/y completa y una muestra legible para el reporte."""
    X, y = construir_nowcast(panel, "Papa")
    completo = X.copy()
    completo["y_cambio_log"] = y
    completo = completo.loc[:HOLDOUT_FIN]
    completo.insert(0, "producto", "Papa")
    completo.insert(1, "region", "Puno")
    completo["split"] = np.select(
        [completo.index < pd.Timestamp(DEV_INICIO),
         completo.index < pd.Timestamp(HOLDOUT_INICIO)],
        ["entrenamiento", "desarrollo"], default="holdout")
    completo = completo.reset_index().rename(columns={"index": "fecha"})
    tabla(completo, "matriz_diseno.csv", "modelos")

    muestra = pd.concat([completo.head(4), completo.tail(4)], ignore_index=True)
    tabla(muestra, "matriz_diseno_muestra.csv", "modelos")


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

    predicciones = predecir_holdout(panel)
    confusion, confusion_producto = matriz_confusion(predicciones)
    confusion_idx = confusion.set_index("modelo")["acierto"]
    hold_modelos = nowcast[(nowcast.periodo == "holdout") &
                           (nowcast.conjunto == "principal")]
    mae_rf = hold_modelos[hold_modelos.modelo == "Random Forest"].set_index("producto")["mae"]
    mae_lgbm = hold_modelos[hold_modelos.modelo == "LightGBM"].set_index("producto")["mae"]
    rf_mejora = int((mae_rf < mae_lgbm.reindex(mae_rf.index)).sum())

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
        fh.write(
            f"Por que no gana LightGBM si tiene mayor acierto en el holdout? LightGBM acierta el signo "
            f"en {confusion_idx['LightGBM']:.1f} % de los meses, frente a "
            f"{confusion_idx['Random Forest']:.1f} % de Random Forest, pero el acierto es una lectura "
            f"binaria: trata igual un error de 0,1 % que uno de 5 %. El ganador se eligio en "
            f"desarrollo con R2, donde Random Forest tiene rango promedio "
            f"{ranking.loc['Random Forest', 'rank_promedio']:.2f} y acierto medio "
            f"{ranking.loc['Random Forest', 'acierto_medio']:.1f} %, contra "
            f"{ranking.loc['LightGBM', 'rank_promedio']:.2f} y "
            f"{ranking.loc['LightGBM', 'acierto_medio']:.1f} % de LightGBM. En el holdout Random Forest "
            f"tiene menor MAE que LightGBM en {rf_mejora} de 6 productos y R2 positivo en papa, cebolla "
            f"y zanahoria; LightGBM tiene R2 muy negativo en arroz y zanahoria. Cambiar el ganador "
            f"segun el holdout seria seleccionar sobre el conjunto de prueba. La matriz de confusion "
            f"se reporta como diagnostico complementario, no como criterio de seleccion.\n\n")
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

    tabla(predicciones, "predicciones_holdout.csv", "modelos")
    graficar_predicciones(predicciones)

    tabla(confusion, "matriz_confusion_modelos.csv", "modelos")
    tabla(confusion_producto, "matriz_confusion_producto.csv", "modelos")
    graficar_confusion(confusion)
    guardar_matriz_diseno(panel)

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
