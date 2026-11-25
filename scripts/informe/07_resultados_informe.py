#!/usr/bin/env python3
"""Artefactos de resultados: flujo KDD, tablas numeradas y figuras numeradas.

Corre despues de los modelos. No recalcula analisis: copia y numera resultados.
Genera:
  outputs/informe/tablas/  (17 tablas)
  outputs/informe/figuras/ (16 figuras)
  outputs/informe/INDICE.md
"""

from __future__ import annotations

import shutil
import sys
import platform
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd

from config import (DIRECTORIOS, INFORME, MIN_NOWCAST_TEST, MIN_NOWCAST_TRAIN, NUCLEO,
                    aplicar_estilo, preparar_directorios)

TAB = INFORME / "tablas"
FIG = INFORME / "figuras"


def flujo_kdd() -> None:
    """Copia la figura KDD del template usado como referencia visual."""
    origen = INFORME.parent.parent / "assets" / "figura_kdd_template.png"
    if not origen.exists():
        raise FileNotFoundError(f"No existe la figura KDD: {origen}")
    shutil.copy2(origen, FIG / "01_flujo_kdd.png")

def copiar_figuras() -> None:
    mapa = {
        "02_cobertura.png": ("panel", "cobertura_anual.png"),
        "03_campanas_precio.png": ("calidad", "campanas_precio.png"),
        "04_campanas_log.png": ("calidad", "campanas_log.png"),
        "06_correlacion_papa.png": ("estructura", "correlacion_papa.png"),
        "07_correlacion_lima.png": ("estructura", "correlacion_lima.png"),
        "08_integracion_distancia.png": ("estructura", "integracion_distancia.png"),
        "11_ranking_torneo.png": ("modelos", "ranking_torneo.png"),
        "12_ganador_vs_observado.png": ("modelos", "ganador_vs_observado.png"),
        "13_curvas_aprendizaje.png": ("modelos", "curvas_aprendizaje.png"),
        "15_arbol_rf.png": ("modelos", "arbol_rf.png"),
        "16_arbol_lgbm.png": ("modelos", "arbol_lgbm_sustituto.png"),
        "17_importancia_rf.png": ("modelos", "importancia_rf.png"),
        "18_pdp.png": ("modelos", "pdp_cebolla.png"),
        "23_verificacion_limpieza.png": ("calidad", "efecto_limpieza.png"),
        "24_matriz_confusion.png": ("modelos", "matriz_confusion.png"),
    }
    for destino, (carpeta, origen) in mapa.items():
        ruta = DIRECTORIOS[carpeta] / origen
        if ruta.exists():
            shutil.copy2(ruta, FIG / destino)


def copiar_tablas() -> None:
    mapa = {
        "tabla_01_variables.csv": ("calidad", "tabla_variables.csv"),
        "tabla_01b_ejemplo.csv": ("calidad", "tabla_dataset_filas.csv"),
        "tabla_02_auditoria.csv": ("panel", "auditoria_panel.csv"),
        "tabla_03_descriptivos.csv": ("calidad", "descriptivos_producto.csv"),
        "tabla_08_torneo.csv": ("modelos", "torneo_nowcast_ranking.csv"),
        "tabla_09_ganador.csv": ("modelos", "torneo_nowcast_metricas.csv"),
        "tabla_12_correlacion_lima.csv": ("estructura", "correlacion_lima.csv"),
        "tabla_13_geografia.csv": ("estructura", "geografia_integracion.csv"),
        "tabla_15_complemento.csv": ("modelos", "torneo_complemento.csv"),
        "tabla_20_transformacion_log.csv": ("calidad", "transformacion_log.csv"),
        "tabla_21_normalidad.csv": ("calidad", "normalidad.csv"),
        "tabla_22_verificacion_limpieza.csv": ("calidad", "verificacion_limpieza.csv"),
        "tabla_24_matriz_diseno.csv": ("modelos", "matriz_diseno_muestra.csv"),
        "tabla_25_confusion_modelos.csv": ("modelos", "matriz_confusion_modelos.csv"),
    }
    for destino, (carpeta, origen) in mapa.items():
        ruta = DIRECTORIOS[carpeta] / origen
        if ruta.exists():
            shutil.copy2(ruta, TAB / destino)


def protocolo() -> None:
    filas = [
        ("Validación", "2018-01–2020-12 (36 meses)",
         "entrenamiento con datos previos a 2018; ranking con mínimos muestrales"),
        ("Prueba final", "2021-01–2026-10", "evaluación final; modelos entrenados hasta 2020"),
        ("Elegibilidad del ranking de estimación contemporánea",
         f"≥ {MIN_NOWCAST_TRAIN} entrenamiento; ≥ {MIN_NOWCAST_TEST} evaluación",
         "si no alcanza, se excluye de la Tabla 15"),
        ("Referencia de estimación contemporánea", "cambio cero", "umbral de valor predictivo"),
        ("Baseline pronóstico", "caminata aleatoria", "umbral de pronóstico"),
        ("Métrica principal de estimación contemporánea", "R²", "captura de señal, no shrinkage"),
        ("Métrica principal pronóstico", "MAE", "error en S/ kg"),
        ("Métricas secundarias", "MAE, mejora relativa, acierto direccional", "diagnóstico complementario"),
        ("Selección", "rango promedio de R2", "no un solo producto"),
        ("Contraste", "Diebold-Mariano (HLN)", "diferencia de precisión"),
        ("Compuerta", "R2 > 0 y acierto > 55 % en >= 2 productos", "utilidad mínima"),
    ]
    pd.DataFrame(filas, columns=["elemento", "definicion", "rol"]).to_csv(
        TAB / "tabla_16_protocolo.csv", index=False, encoding="utf-8")


def software() -> None:
    paquetes = ["pandas", "numpy", "scikit-learn", "statsmodels", "scipy",
                "matplotlib", "seaborn", "lightgbm", "xgboost", "python-docx", "jinja2"]
    filas = [("python", platform.python_version())]
    for p in paquetes:
        try:
            filas.append((p, version(p)))
        except Exception:
            filas.append((p, "no instalado"))
    pd.DataFrame(filas, columns=["software", "version"]).to_csv(
        TAB / "tabla_17_software.csv", index=False, encoding="utf-8")


def pronostico_holdout() -> None:
    """Mejora de MAE del pronóstico a un mes frente a la caminata aleatoria."""
    ruta = DIRECTORIOS["modelos"] / "torneo_forecast_metricas.csv"
    datos = pd.read_csv(ruta)
    datos = datos[(datos.periodo == "holdout") & (datos.conjunto == "principal")]
    mejora = datos.pivot(index="producto", columns="modelo", values="mejora_mae")
    n = datos.groupby("producto").n.first()
    modelos = ["OLS", "Ridge", "Random Forest", "LightGBM"]
    mejora = mejora.reindex(index=NUCLEO, columns=modelos)
    mejora.insert(0, "n", n.reindex(NUCLEO))
    mejora.index.name = "producto"
    mejora.reset_index().to_csv(
        TAB / "tabla_23_pronostico_holdout.csv", index=False, encoding="utf-8")


def indice() -> None:
    lineas = ["# Indice de artefactos de resultados\n",
              "| Figura | Archivo |", "|---|---|"]
    for ruta in sorted(FIG.glob("*.png")):
        lineas.append(f"| {ruta.stem} | `{ruta.name}` |")
    lineas += ["\n| Tabla | Archivo |", "|---|---|"]
    for ruta in sorted(TAB.glob("*.csv")):
        lineas.append(f"| {ruta.stem} | `{ruta.name}` |")
    (INFORME / "INDICE.md").write_text("\n".join(lineas), encoding="utf-8")


def main() -> None:
    preparar_directorios()
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    flujo_kdd()
    copiar_figuras()
    copiar_tablas()
    protocolo()
    software()
    pronostico_holdout()
    indice()
    print(f"Figuras de resultados: {len(list(FIG.glob('*.png')))}")
    print(f"Tablas de resultados:  {len(list(TAB.glob('*.csv')))}")
    print("Indice:", INFORME / "INDICE.md")


if __name__ == "__main__":
    main()
