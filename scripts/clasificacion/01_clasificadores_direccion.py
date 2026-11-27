#!/usr/bin/env python3
"""Clasificación de la dirección del precio al mes siguiente.

Se usa el mismo universo de Puno y los seis productos del núcleo. El target
es 1 si el log-precio del mes siguiente es mayor que el actual. Se comparan
DecisionTreeClassifier con criterio gini y entropía, y LGBMClassifier.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))
from config import NUCLEO, PANEL_LIMPIO  # noqa: E402
from modelos import construir_forecast  # noqa: E402

SALIDA = RAIZ / "outputs" / "clasificacion_direccion"
TABLAS = SALIDA / "tablas"
FIGURAS = SALIDA / "figuras"
FIN_TEST = pd.Timestamp("2026-09-01")
SEMILLA = 0


def metricas(y, pred, prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
        "n": int(len(y)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main() -> None:
    TABLAS.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    panel = pd.read_csv(PANEL_LIMPIO, parse_dates=["fecha"])

    partes = []
    for cat in NUCLEO:
        X, y, p0 = construir_forecast(panel, cat)
        X = X.loc[:FIN_TEST].copy()
        y = y.loc[X.index]
        p0 = p0.loc[X.index]
        X["producto"] = cat
        X["target"] = (y > p0).astype(int)
        partes.append(X)
    datos = pd.concat(partes).reset_index().rename(columns={"index": "fecha"})
    datos = datos.dropna().sort_values("fecha").reset_index(drop=True)

    dummies = pd.get_dummies(datos["producto"], prefix="producto", dtype=float)
    features = [c for c in datos.columns if c not in {"fecha", "producto", "target"}]
    X = pd.concat([datos[features], dummies], axis=1)
    y = datos["target"]

    train = datos.fecha < pd.Timestamp("2018-01-01")
    val = (datos.fecha >= pd.Timestamp("2018-01-01")) & (datos.fecha <= pd.Timestamp("2020-12-01"))
    test = datos.fecha >= pd.Timestamp("2021-01-01")

    modelos = {
        "Árbol de Decisión (gini)": DecisionTreeClassifier(
            criterion="gini", max_depth=4, min_samples_leaf=20,
            class_weight="balanced", random_state=SEMILLA),
        "Árbol de Decisión (entropía)": DecisionTreeClassifier(
            criterion="entropy", max_depth=4, min_samples_leaf=20,
            class_weight="balanced", random_state=SEMILLA),
        "LightGBM": LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=15,
            class_weight="balanced", random_state=SEMILLA,
            n_jobs=-1, verbose=-1),
    }

    filas_val, filas_test, filas_conf, filas_pred = [], [], [], []
    for nombre, modelo in modelos.items():
        modelo.fit(X[train], y[train])
        for etapa, mascara, destino in [
            ("validacion", val, filas_val), ("prueba", test, filas_test)]:
            prob = modelo.predict_proba(X[mascara])[:, 1]
            pred = (prob >= 0.5).astype(int)
            met = metricas(y[mascara].values, pred, prob)
            met["modelo"] = nombre
            destino.append(met)
        prob = modelo.predict_proba(X[test])[:, 1]
        pred = (prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y[test], pred, labels=[0, 1]).ravel()
        filas_conf.append({"modelo": nombre, "tn": int(tn), "fp": int(fp),
                           "fn": int(fn), "tp": int(tp)})
        filas_pred.append(pd.DataFrame({
            "modelo": nombre, "fecha": datos.loc[test, "fecha"].values,
            "target": y[test].values, "probabilidad": prob, "prediccion": pred,
        }))

    val_df = pd.DataFrame(filas_val).sort_values("f1", ascending=False)
    test_df = pd.DataFrame(filas_test).sort_values("f1", ascending=False)
    val_df.to_csv(TABLAS / "metricas_validacion.csv", index=False, encoding="utf-8")
    test_df.to_csv(TABLAS / "metricas_prueba.csv", index=False, encoding="utf-8")
    pd.DataFrame(filas_conf).to_csv(TABLAS / "matriz_confusion.csv", index=False, encoding="utf-8")
    pd.concat(filas_pred).to_csv(TABLAS / "predicciones.csv", index=False, encoding="utf-8")
    pd.DataFrame([
        ("filas", len(datos)), ("productos", datos.producto.nunique()),
        ("target_sube", int(y.sum())), ("target_sube_pct", round(100 * y.mean(), 2)),
        ("entrenamiento", int(train.sum())), ("validacion", int(val.sum())),
        ("prueba", int(test.sum())),
    ], columns=["metrica", "valor"]).to_csv(TABLAS / "resumen.csv", index=False, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    base = test_df.set_index("modelo")[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    base.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("valor")
    ax.set_title("Clasificación de la dirección al mes siguiente: prueba final")
    ax.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    plt.xticks(rotation=12, fontsize=7)
    fig.savefig(FIGURAS / "comparativa_clasificadores.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (_, fila) in zip(axes.ravel(), pd.DataFrame(filas_conf).iterrows()):
        matriz = np.array([[fila.tn, fila.fp], [fila.fn, fila.tp]])
        sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Predice no sube", "Predice sube"],
                    yticklabels=["Real no sube", "Real sube"], ax=ax)
        ax.set_title(fila.modelo, fontsize=8)
    fig.suptitle("Matrices de confusión en la prueba final", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURAS / "matrices_confusion_clasificadores.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("Validación:")
    print(val_df[["modelo", "accuracy", "f1", "roc_auc"]].round(3).to_string(index=False))
    print("\nPrueba final:")
    print(test_df[["modelo", "accuracy", "precision", "recall", "f1", "roc_auc"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
