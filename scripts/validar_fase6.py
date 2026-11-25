#!/usr/bin/env python3
"""Valida la Fase 6: modelos, holdout, confusion y matriz de diseno.

Incluye una comprobacion directa de que ``mediana_otras`` se calcula con los
22 departamentos restantes, nunca con provincias ni con Callao.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalogo import EXCLUIDOS, cargar_catalogo  # noqa: E402
from config import HOLDOUT_FIN, HOLDOUT_INICIO, NUCLEO, PANEL_LIMPIO  # noqa: E402
from modelos import construir_forecast  # noqa: E402

MODELOS = RAIZ / "outputs" / "analisis" / "06_modelos"
MANIFIESTOS = RAIZ / "outputs" / "_manifests"
MANIFIESTO_MODELOS = MANIFIESTOS / "06_modelos.json"
MANIFIESTO_VALIDACION = MANIFIESTOS / "06_validacion.json"


def sha256_archivo(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def revisar_excluidos(datos: pd.DataFrame, etiqueta: str) -> None:
    nombres = {nombre.casefold() for nombre, _ in EXCLUIDOS.values()}
    for columna in datos.select_dtypes(include=["object", "string"]).columns:
        valores = set(datos[columna].dropna().astype(str).str.casefold())
        encontrados = valores & nombres
        if encontrados:
            raise AssertionError(f"{etiqueta}: mercados excluidos en {columna}: {sorted(encontrados)}")


def contar_direccion(d: pd.DataFrame) -> dict:
    d = d[d["observado"] != 0]
    real_pos = d["observado"] > 0
    pred_pos = d["prediccion"] > 0
    tp = int((real_pos & pred_pos).sum())
    fn = int((real_pos & ~pred_pos).sum())
    fp = int((~real_pos & pred_pos).sum())
    tn = int((~real_pos & ~pred_pos).sum())
    n = tp + fn + fp + tn
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "n": n, "acierto": 100 * (tp + tn) / n if n else np.nan}


def main() -> None:
    catalogo = cargar_catalogo()
    ids = set(catalogo["departamento_id"])
    permitidos = set(catalogo["departamento_ascii"])
    panel = pd.read_csv(PANEL_LIMPIO, parse_dates=["fecha"], dtype={"departamento_id": str})
    if set(panel["departamento_id"]) != ids:
        raise AssertionError("El panel limpio no tiene los 24 departamentos")
    if set(panel["region"]) != permitidos:
        raise AssertionError("El panel limpio no tiene los alias departamentales esperados")

    archivos = sorted(MODELOS.glob("*.csv")) + [MODELOS / "torneo_ganador.md"]
    for ruta in archivos:
        if ruta.suffix == ".csv":
            revisar_excluidos(pd.read_csv(ruta), ruta.name)
        else:
            texto = ruta.read_text(encoding="utf-8").casefold()
            for nombre, _ in EXCLUIDOS.values():
                if nombre.casefold() in texto:
                    raise AssertionError(f"{ruta.name}: contiene el mercado excluido {nombre}")

    # Ranking de nowcast: recalcular desde las metricas de desarrollo.
    nowcast = pd.read_csv(MODELOS / "torneo_nowcast_metricas.csv")
    dev = nowcast[(nowcast["periodo"] == "desarrollo") &
                  (nowcast["conjunto"] == "principal")].copy()
    dev["rank"] = dev.groupby("producto")["r2"].rank(ascending=False, method="average")
    ranking_calc = dev.groupby("modelo")["rank"].mean().sort_values()
    ranking = pd.read_csv(MODELOS / "torneo_nowcast_ranking.csv").set_index("modelo")
    if not np.allclose(ranking_calc.reindex(ranking.index).values,
                       ranking["rank_promedio"].values, atol=1e-12):
        raise AssertionError("El ranking de nowcast no coincide con las metricas")
    if ranking.index[0] != "Random Forest":
        raise AssertionError("El ganador de nowcast cambio inesperadamente")

    # Ranking de pronostico: recalcular desde MAE de desarrollo.
    forecast = pd.read_csv(MODELOS / "torneo_forecast_metricas.csv")
    fdev = forecast[(forecast["periodo"] == "desarrollo") &
                    (forecast["conjunto"] == "principal")].copy()
    fdev["rank"] = fdev.groupby("producto")["mae"].rank(ascending=True, method="average")
    frank_calc = fdev.groupby("modelo")["rank"].mean().sort_values()
    frank = pd.read_csv(MODELOS / "torneo_forecast_ranking.csv").set_index("modelo")
    if not np.allclose(frank_calc.reindex(frank.index).values,
                       frank["rank_promedio"].values, atol=1e-12):
        raise AssertionError("El ranking de pronostico no coincide con las metricas")

    # Holdout: seis productos y cuatro modelos.
    hold = nowcast[(nowcast["periodo"] == "holdout") &
                   (nowcast["conjunto"] == "principal")]
    if set(hold["producto"]) != set(NUCLEO) or hold["modelo"].nunique() != 4:
        raise AssertionError("El holdout de nowcast no cubre seis productos y cuatro modelos")
    if len(forecast[(forecast["periodo"] == "holdout") &
                    (forecast["conjunto"] == "principal")]) != 24:
        raise AssertionError("El holdout de pronostico no tiene 6 productos x 4 modelos")

    pred = pd.read_csv(MODELOS / "predicciones_holdout.csv", parse_dates=["fecha"])
    if set(pred["producto"]) != set(NUCLEO) or set(pred["modelo"]) != {
            "OLS", "Ridge", "Random Forest", "LightGBM"}:
        raise AssertionError("Las predicciones del holdout no cubren el cuarteto")
    if pred["fecha"].min() < pd.Timestamp(HOLDOUT_INICIO) or pred["fecha"].max() > pd.Timestamp(HOLDOUT_FIN):
        raise AssertionError("Las predicciones se salen del holdout")
    for (producto, modelo), grupo in pred.groupby(["producto", "modelo"]):
        if grupo["observado"].duplicated().sum() and grupo.groupby("fecha")["observado"].nunique().max() > 1:
            raise AssertionError(f"Observado inconsistente en {producto}/{modelo}")

    confusion = pd.read_csv(MODELOS / "matriz_confusion_modelos.csv").set_index("modelo")
    calculada = {modelo: contar_direccion(grupo) for modelo, grupo in pred.groupby("modelo")}
    for modelo, fila in confusion.iterrows():
        esperado = calculada[modelo]
        for columna in ["TP", "FN", "FP", "TN", "n"]:
            if int(fila[columna]) != int(esperado[columna]):
                raise AssertionError(f"Confusion inconsistente en {modelo}/{columna}")
        if not np.isclose(float(fila["acierto"]), float(esperado["acierto"]), atol=1e-9):
            raise AssertionError(f"Acierto inconsistente en {modelo}")

    confusion_prod = pd.read_csv(MODELOS / "matriz_confusion_producto.csv")
    calculada_prod = {
        (producto, modelo): contar_direccion(grupo)
        for (producto, modelo), grupo in pred.groupby(["producto", "modelo"])
    }
    if len(confusion_prod) != 24:
        raise AssertionError("La confusion por producto no tiene 24 filas")
    for fila in confusion_prod.itertuples(index=False):
        esperado = calculada_prod[(fila.producto, fila.modelo)]
        for columna in ["TP", "FN", "FP", "TN", "n"]:
            if int(getattr(fila, columna)) != int(esperado[columna]):
                raise AssertionError("Confusion por producto inconsistente")

    # Matriz de diseno: estructura X/y y splits.
    diseno = pd.read_csv(MODELOS / "matriz_diseno.csv", parse_dates=["fecha"])
    columnas = {"dl0", "dl1", "dl2", "dl3", "dp1", "dp2", "sin1", "cos1", "y_cambio_log", "split"}
    if not columnas <= set(diseno.columns):
        raise AssertionError("La matriz de diseno no tiene las columnas esperadas")
    if set(diseno["split"]) != {"entrenamiento", "desarrollo", "holdout"}:
        raise AssertionError("La matriz de diseno tiene splits invalidos")
    if diseno["fecha"].min() < pd.Timestamp("2012-01-01"):
        raise AssertionError("La matriz de diseno empieza antes de la ventana")
    if diseno.isnull().any().any():
        raise AssertionError("La matriz de diseno tiene nulos")

    # Mediana de otras regiones: 24 - Puno - Lima = 22.
    otras = set(panel["region"]) - {"Puno", "Lima"}
    if len(otras) != 22:
        raise AssertionError("La mediana de otras regiones no tiene 22 departamentos")
    for cat in NUCLEO:
        X, _, _ = construir_forecast(panel, cat)
        fecha = X.index[0]
        esperado = np.log(
            panel[(panel["fecha"] == fecha) & (panel["categoria"] == cat) &
                  (~panel["region"].isin(["Puno", "Lima"]))]["precio_kg"].median()
        )
        if not np.isclose(float(X.loc[fecha, "m0"]), float(esperado), atol=1e-12):
            raise AssertionError(f"mediana_otras incorrecta para {cat} en {fecha}")

    # Archivos de representacion y evaluacion.
    if len(pd.read_csv(MODELOS / "curvas_aprendizaje.csv")) != 16:
        raise AssertionError("Las curvas de aprendizaje no tienen 16 filas")
    if len(pd.read_csv(MODELOS / "coeficientes.csv")) != 9:
        raise AssertionError("Los coeficientes no tienen 9 filas")
    if len(pd.read_csv(MODELOS / "hiperparametros.csv")) != 4:
        raise AssertionError("Los hiperparametros no tienen 4 modelos")
    if len(pd.read_csv(MODELOS / "model_cards.csv")) != 4:
        raise AssertionError("Las model cards no tienen 4 modelos")
    if len(pd.read_csv(MODELOS / "torneo_nowcast_dm.csv")) != 24:
        raise AssertionError("Diebold-Mariano no tiene 24 filas")

    hashes = {str(ruta.relative_to(RAIZ)): sha256_archivo(ruta) for ruta in archivos}
    manifiesto = {
        "fase": "06_modelos",
        "catalogo_sha256": sha256_archivo(
            RAIZ / "data" / "reference" / "departamentos.csv"
        ),
        "panel_limpio_sha256": sha256_archivo(PANEL_LIMPIO),
        "departamentos": 24,
        "productos_holdout": 6,
        "modelos": 4,
        "predicciones_holdout": int(len(pred)),
        "archivos": hashes,
    }
    MANIFIESTOS.mkdir(parents=True, exist_ok=True)
    MANIFIESTO_MODELOS.write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    validacion = {
        "fase": "06_validacion",
        "ok": True,
        "departamentos": 24,
        "productos_holdout": 6,
        "modelos": 4,
        "predicciones_holdout": int(len(pred)),
        "matriz_confusion_filas": int(len(confusion)),
        "matriz_confusion_producto_filas": int(len(confusion_prod)),
        "ganador_nowcast": "Random Forest",
        "ganador_pronostico": str(frank.index[0]),
        "archivos_validados": len(archivos),
        "hashes": hashes,
    }
    MANIFIESTO_VALIDACION.write_text(
        json.dumps(validacion, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Fase 6 valida.")
    print(f"  Panel: 24 departamentos | 216 series")
    print(f"  Holdout: 6 productos x 4 modelos = {len(pred):,} predicciones")
    print(f"  Ganador nowcast: {validacion['ganador_nowcast']}")
    print(f"  Ganador pronostico: {validacion['ganador_pronostico']}")
    print(f"  Archivos validados: {len(archivos)}")
    print(f"  Manifiesto: {MANIFIESTO_VALIDACION.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
