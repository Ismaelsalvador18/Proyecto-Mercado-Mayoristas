#!/usr/bin/env python3
"""Cuenta observaciones por producto y etapa de validación.

Genera una tabla que explica por qué el N cambia entre productos y modelos.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (DEV_FIN, DEV_INICIO, DIRECTORIOS, HOLDOUT_FIN, HOLDOUT_INICIO,
                    NUCLEO, PANEL_LIMPIO, preparar_directorios, tabla)
from modelos import construir_forecast, construir_nowcast


def contar(X: pd.DataFrame) -> dict:
    train = X.index < pd.Timestamp(DEV_INICIO)
    val = (X.index >= pd.Timestamp(DEV_INICIO)) & (X.index <= pd.Timestamp(DEV_FIN))
    test = (X.index >= pd.Timestamp(HOLDOUT_INICIO)) & (X.index <= pd.Timestamp(HOLDOUT_FIN))
    return {
        "entrenamiento_2012_2017": int(train.sum()),
        "validacion_2018_2020": int(val.sum()),
        "prueba_2021_2026": int(test.sum()),
        "total": int(len(X)),
    }


def main() -> None:
    preparar_directorios()
    panel = pd.read_csv(PANEL_LIMPIO, parse_dates=["fecha"])
    filas = []
    for cat in NUCLEO:
        Xn, _ = construir_nowcast(panel, cat)
        Xf, _, _ = construir_forecast(panel, cat)
        filas.append({"producto": cat, "tarea": "estimación del mes corriente",
                      **contar(Xn)})
        filas.append({"producto": cat, "tarea": "pronóstico a un mes",
                      **contar(Xf)})
    datos = pd.DataFrame(filas)
    tabla(datos, "n_etapas.csv", "modelos")
    print(datos.to_string(index=False))
    print("\nTabla de N por etapa generada en", DIRECTORIOS["modelos"])


if __name__ == "__main__":
    main()
