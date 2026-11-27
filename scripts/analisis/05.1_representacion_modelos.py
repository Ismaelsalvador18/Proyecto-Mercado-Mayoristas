#!/usr/bin/env python3
"""Representacion de los modelos: coeficientes, arboles, importancia y dependencia.

Corre despues de 05. Usa Cebolla, el producto con mayor senal, y entrena con
datos hasta 2020 para no tocar la prueba final.
Genera en outputs/analisis/06_modelos:
  coeficientes.csv, model_cards.csv, ridge_path.png, arbol_rf.png,
  importancia_rf.png, importancia_lgbm.png, pdp_cebollla.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree

from config import (DIRECTORIOS, HOLDOUT_INICIO, LGBM, PANEL_LIMPIO, RF,
                    RIDGE_ALPHA, aplicar_estilo, guardar, preparar_directorios, tabla)
from modelos import construir_nowcast, modelos_principales

PRODUCTO = "Cebolla"


def main() -> None:
    preparar_directorios()
    aplicar_estilo()
    panel = pd.read_csv(PANEL_LIMPIO, parse_dates=["fecha"])
    X, y = construir_nowcast(panel, PRODUCTO)
    train = X.index < pd.Timestamp(HOLDOUT_INICIO)
    X_tr, y_tr = X[train], y[train]

    modelos = modelos_principales()
    ols = modelos["OLS"].fit(X_tr, y_tr)
    ridge = modelos["Ridge"].fit(X_tr, y_tr)
    rf = modelos["Random Forest"].fit(X_tr, y_tr)
    lgbm = modelos["LightGBM"].fit(X_tr, y_tr)

    # 1. Coeficientes OLS y Ridge estandarizado
    ols_sm = sm.OLS(y_tr, sm.add_constant(X_tr)).fit()
    coef = pd.DataFrame({
        "variable": ols_sm.params.index,
        "ols": ols_sm.params.values,
        "ols_p": ols_sm.pvalues.values,
    })
    escalador = StandardScaler().fit(X_tr)
    ridge_esc = Ridge(alpha=RIDGE_ALPHA).fit(escalador.transform(X_tr), y_tr)
    coef["ridge_estandarizado"] = np.r_[0.0, ridge_esc.coef_][:len(coef)]
    tabla(coef, "coeficientes.csv", "modelos")

    # 2. Camino de Ridge
    alphas = np.logspace(-2, 3, 40)
    coefs = []
    for alpha in alphas:
        m = Ridge(alpha=alpha).fit(escalador.transform(X_tr), y_tr)
        coefs.append(m.coef_)
    coefs = np.array(coefs)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for i, col in enumerate(X.columns):
        ax.plot(alphas, coefs[:, i], label=col, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("coeficiente estandarizado")
    ax.set_title(f"Ridge: camino de coeficientes ({PRODUCTO})")
    ax.legend(fontsize=6, ncol=2)
    guardar(fig, "ridge_path.png", "modelos")

    # 3. Arbol de Random Forest
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_tree(rf.estimators_[0], max_depth=3, feature_names=list(X.columns),
              filled=True, rounded=True, fontsize=7, impurity=False, proportion=True, ax=ax)
    ax.set_title(f"Random Forest: primer arbol ({PRODUCTO}, profundidad 3)")
    guardar(fig, "arbol_rf.png", "modelos")

    # 5. Importancias
    for nombre, modelo, archivo in [("Random Forest", rf, "importancia_rf.png"),
                                    ("LightGBM", lgbm, "importancia_lgbm.png")]:
        imp = pd.Series(modelo.feature_importances_, index=X.columns).sort_values()
        fig, ax = plt.subplots(figsize=(8, 4.6))
        imp.plot(kind="barh", color="#0e7490", ax=ax)
        ax.set_title(f"{nombre}: importancia de variables ({PRODUCTO})")
        ax.set_xlabel("importancia")
        guardar(fig, archivo, "modelos")

    # 6. Dependencia parcial
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    PartialDependenceDisplay.from_estimator(
        rf, X_tr, ["dl0", "dp1", "cos1"], ax=axes, kind="average")
    fig.suptitle(f"Dependencia parcial: Random Forest ({PRODUCTO})", y=1.02)
    guardar(fig, "pdp_cebolla.png", "modelos")

    # 7. Model cards
    cards = pd.DataFrame([
        ("OLS", "lineal", "sin hiperparámetros", "relación lineal y aditiva",
         "alta", "sensible a colinealidad"),
        ("Ridge", "lineal regularizado", f"alpha={RIDGE_ALPHA}", "linealidad + penalización",
         "alta", "alpha fijado a priori"),
        ("Random Forest", "no lineal (bagging)", f"n={RF['n_estimators']}, leaf={RF['min_samples_leaf']}",
         "interacciones y umbrales", "media", "no extrapola; requiere n grande"),
        ("LightGBM", "no lineal (boosting)", f"n={LGBM['n_estimators']}, lr={LGBM['learning_rate']}, leaves={LGBM['num_leaves']}",
         "interacciones y no linealidad", "baja", "sobreajuste con n pequeño"),
    ], columns=["modelo", "familia", "hiperparametros", "supuesto", "interpretabilidad", "riesgo"])
    tabla(cards, "model_cards.csv", "modelos")

    print(f"Representacion de modelos para {PRODUCTO} (entrenamiento hasta 2020).")
    print("\nCoeficientes OLS:")
    print(coef.round(3).to_string(index=False))
    print("\nTablas y figuras de representacion generadas en", DIRECTORIOS["modelos"])


if __name__ == "__main__":
    main()
