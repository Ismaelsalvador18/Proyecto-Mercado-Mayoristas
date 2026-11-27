#!/usr/bin/env python3
"""Dibuja un árbol representativo de cada modelo basado en árboles.

Usa el mismo producto (cebolla), la misma matriz de predictores y el mismo corte
temporal que el resto del análisis. En Random Forest, XGBoost y LightGBM se
muestra uno de los árboles que componen el ensemble, no el ensemble completo.

Genera en outputs/analisis/06_modelos:
  arboles_modelos.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import plot_tree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import HOLDOUT_INICIO, PANEL_LIMPIO, aplicar_estilo, guardar, preparar_directorios
from modelos import construir_nowcast, modelos_torneo

PRODUCTO = "Cebolla"


def _layout(nodes: dict, root: str, max_depth: int) -> dict:
    """Asigna una posición horizontal a cada nodo hasta max_depth."""
    pos: dict[str, tuple[float, int]] = {}

    def recorrer(nodo: str, profundidad: int, xmin: float, xmax: float) -> None:
        x = (xmin + xmax) / 2
        pos[nodo] = (x, -profundidad)
        hijos = nodes[nodo]
        if (profundidad < max_depth and hijos["left"] in nodes
                and hijos["right"] in nodes):
            recorrer(hijos["left"], profundidad + 1, xmin, x)
            recorrer(hijos["right"], profundidad + 1, x, xmax)

    recorrer(root, 0, 0.02, 0.98)
    return pos


def _dibujar_arbol(ax, nodes: dict, root: str, titulo: str, max_depth: int = 2) -> None:
    pos = _layout(nodes, root, max_depth)
    for nodo, (x, y) in pos.items():
        hijos = nodes[nodo]
        for etiqueta, clave in [("sí", "left"), ("no", "right")]:
            hijo = hijos[clave]
            if hijo not in pos:
                continue
            xh, yh = pos[hijo]
            ax.plot([x, xh], [y, yh], color="#6B7280", linewidth=0.9)
            ax.text((x + xh) / 2, (y + yh) / 2 + 0.04, etiqueta,
                    ha="center", va="bottom", fontsize=5.5, color="#6B7280")
    for nodo, (x, y) in pos.items():
        info = nodes[nodo]
        profundidad = int(-y)
        if info["feature"] is None:
            texto = f"ŷ = {info['value']:.3g}"
            color = "#DCFCE7"
        elif profundidad >= max_depth and info["left"] in nodes:
            texto = "más niveles"
            color = "#E5E7EB"
        else:
            texto = f"{info['feature']} ≤ {info['split']:.3g}"
            color = "#DBEAFE"
        ax.text(x, y, texto, ha="center", va="center", fontsize=6.2,
                bbox=dict(boxstyle="round,pad=0.28", facecolor=color,
                          edgecolor="#374151", linewidth=0.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(-max_depth - 0.45, 0.45)
    ax.axis("off")
    ax.set_title(titulo, fontsize=9.5, pad=4)


def _nodos_xgboost(modelo, arbol: int = 0) -> tuple[dict, str]:
    tabla = modelo.get_booster().trees_to_dataframe()
    tabla = tabla[tabla.Tree == arbol]
    nodes = {}
    for fila in tabla.itertuples(index=False):
        hoja = fila.Feature == "Leaf"
        nodes[str(fila.ID)] = {
            "feature": None if hoja else str(fila.Feature),
            "split": None if hoja else float(fila.Split),
            "left": None if hoja else str(fila.Yes),
            "right": None if hoja else str(fila.No),
            "value": float(fila.Gain) if hoja else None,
        }
    return nodes, "0-0"


def _nodos_lightgbm(modelo, arbol: int = 0) -> tuple[dict, str]:
    tabla = modelo.booster_.trees_to_dataframe()
    tabla = tabla[tabla.tree_index == arbol]
    nodes = {}
    for fila in tabla.itertuples(index=False):
        hoja = pd.isna(fila.split_feature)
        nodes[str(fila.node_index)] = {
            "feature": None if hoja else str(fila.split_feature),
            "split": None if hoja else float(fila.threshold),
            "left": None if hoja else str(fila.left_child),
            "right": None if hoja else str(fila.right_child),
            "value": float(fila.value) if hoja else None,
        }
    return nodes, "0-S0"


def main() -> None:
    preparar_directorios()
    aplicar_estilo()
    panel = pd.read_csv(PANEL_LIMPIO, parse_dates=["fecha"])
    X, y = construir_nowcast(panel, PRODUCTO)
    train = X.index < pd.Timestamp(HOLDOUT_INICIO)
    X_tr, y_tr = X[train], y[train]

    modelos = modelos_torneo()
    arbol = modelos["Arbol"].fit(X_tr, y_tr)
    rf = modelos["Random Forest"].fit(X_tr, y_tr)
    xgbm = modelos["XGBoost"].fit(X_tr, y_tr)
    lgbm = modelos["LightGBM"].fit(X_tr, y_tr)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    plot_tree(arbol, max_depth=2, feature_names=list(X.columns), filled=True,
              rounded=True, fontsize=6, impurity=False, proportion=True,
              ax=axes[0, 0])
    axes[0, 0].set_title("Árbol de Decisión", fontsize=10)

    plot_tree(rf.estimators_[0], max_depth=2, feature_names=list(X.columns),
              filled=True, rounded=True, fontsize=6, impurity=False,
              proportion=True, ax=axes[0, 1])
    axes[0, 1].set_title("Random Forest (uno de sus 300 árboles)", fontsize=10)

    # Los nodos que quedan por debajo del corte se marcan como "más niveles".
    for ax in (axes[0, 0], axes[0, 1]):
        for texto in ax.texts:
            if texto.get_text().strip() == "(...)":
                texto.set_text("más niveles")
                texto.set_fontsize(5.5)

    nodos_xgb, raiz_xgb = _nodos_xgboost(xgbm)
    _dibujar_arbol(axes[1, 0], nodos_xgb, raiz_xgb,
                   "XGBoost (uno de sus 300 árboles)", max_depth=2)
    nodos_lgb, raiz_lgb = _nodos_lightgbm(lgbm)
    _dibujar_arbol(axes[1, 1], nodos_lgb, raiz_lgb,
                   "LightGBM (uno de sus 300 árboles)", max_depth=2)

    fig.suptitle("Un árbol representativo de cada modelo · cebolla · entrenamiento hasta 2020",
                 fontsize=13, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    guardar(fig, "arboles_modelos.png", "modelos")

    # Versiones individuales, para una diapositiva por modelo.
    for modelo, nombre, titulo in [
        (arbol, "arbol_decision_modelo.png", "Árbol de Decisión · cebolla"),
        (rf.estimators_[0], "arbol_rf_modelo.png", "Random Forest · un árbol del bosque · cebolla"),
    ]:
        fig, ax = plt.subplots(figsize=(13, 7.5))
        plot_tree(modelo, max_depth=3, feature_names=list(X.columns), filled=True,
                  rounded=True, fontsize=7, impurity=False, proportion=True, ax=ax)
        for texto in ax.texts:
            if texto.get_text().strip() == "(...)":
                texto.set_text("más niveles")
                texto.set_fontsize(6.5)
        ax.set_title(titulo + " · tres primeros niveles", fontsize=12)
        guardar(fig, nombre, "modelos")

    for nodos, raiz, nombre, titulo in [
        (nodos_xgb, raiz_xgb, "arbol_xgboost_modelo.png", "XGBoost · un árbol del boosting · cebolla"),
        (nodos_lgb, raiz_lgb, "arbol_lightgbm_modelo.png", "LightGBM · un árbol del boosting · cebolla"),
    ]:
        fig, ax = plt.subplots(figsize=(13, 7.5))
        _dibujar_arbol(ax, nodos, raiz, titulo + " · tres primeros niveles", max_depth=3)
        ax.set_title(titulo + " · tres primeros niveles", fontsize=12)
        guardar(fig, nombre, "modelos")

    print("Árboles representativos generados en", "outputs/analisis/06_modelos/")


if __name__ == "__main__":
    main()
