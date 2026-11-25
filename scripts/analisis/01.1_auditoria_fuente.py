#!/usr/bin/env python3
"""Auditoria de la fuente: que se descargo y que cubre.

Corre despues de 01 y antes de 02.
Genera:
  outputs/analisis/01_fuente/auditoria_fuente.csv
  outputs/analisis/01_fuente/cobertura_anual.png
  outputs/analisis/01_fuente/filas_por_region.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import pandas as pd

from config import DIRECTORIOS, LARGO, aplicar_estilo, guardar, preparar_directorios, tabla

import seaborn as sns


def main() -> None:
    preparar_directorios()
    aplicar_estilo()
    crudo = pd.read_csv(LARGO)
    # codigo_producto contiene la variedad; los cuatro primeros digitos identifican la categoria.
    crudo["categoria_codigo"] = crudo.codigo_producto.astype(str).str.zfill(6).str[:4]

    filas = []
    filas.append(("filas", len(crudo)))
    filas.append(("departamentos con datos", crudo.region.nunique()))
    filas.append(("categorias con datos", crudo.categoria_codigo.nunique()))
    filas.append(("variedades", crudo.producto.nunique()))
    filas.append(("anio minimo", int(crudo.anio.min())))
    filas.append(("anio maximo", int(crudo.anio.max())))
    filas.append(("unidades distintas", crudo.unidad.nunique()))
    filas.append(("filas con precio", int(crudo.valor.notna().sum())))
    auditoria = pd.DataFrame(filas, columns=["metrica", "valor"])
    tabla(auditoria, "auditoria_fuente.csv", "fuente")

    por_region = crudo.groupby("region").agg(filas=("valor", "size"),
                                             variedades=("producto", "nunique"),
                                             anio_min=("anio", "min"),
                                             anio_max=("anio", "max")).reset_index()
    tabla(por_region, "filas_por_region.csv", "fuente")

    anio = crudo.groupby("anio").agg(regiones=("region", "nunique"),
                                     categorias=("categoria_codigo", "nunique"),
                                     filas=("valor", "size")).reset_index()
    tabla(anio, "cobertura_anual.csv", "fuente")

    print("Auditoria de fuente:")
    print(auditoria.to_string(index=False))
    print("\nDepartamentos con datos:", crudo.region.nunique())
    print("Categorias con datos:", crudo.categoria_codigo.nunique(),
          "| variedades:", crudo.producto.nunique())
    print("Rango:", crudo.anio.min(), "-", crudo.anio.max())

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(anio.anio, anio.regiones, marker="o", label="departamentos")
    ax.plot(anio.anio, anio.categorias, marker="s", label="categorias")
    ax.set_title("Cobertura anual de la fuente")
    ax.set_xlabel("año")
    ax.set_ylabel("cantidad")
    ax.legend()
    guardar(fig, "cobertura_anual.png", "fuente")

    fig, ax = plt.subplots(figsize=(10, 5))
    orden = por_region.sort_values("filas", ascending=False)
    sns.barplot(data=orden, y="region", x="filas", color="#0e7490", ax=ax)
    ax.set_title("Filas descargadas por departamento")
    ax.set_xlabel("filas")
    ax.set_ylabel("")
    guardar(fig, "filas_por_region.png", "fuente")

    print("\nTablas y figuras de fuente generadas en", DIRECTORIOS["fuente"])


if __name__ == "__main__":
    main()
