#!/usr/bin/env python3
"""Artefactos del informe: flujo KDD, tablas numeradas y figuras numeradas.

Corre despues de 06.1. No recalcula analisis: copia y numera resultados.
Genera:
  outputs/informe/tablas/tabla_01...tabla_23
  outputs/informe/figuras/01...18
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
    aplicar_estilo()
    fases = ["Selección", "Preprocesamiento", "Transformación", "Minería", "Evaluación"]
    detalles = ["SISAP:\n27 regiones\n9 productos", "Auditoría:\nnulos, unidades,\natípicos",
                "Log-precio,\nrezagos,\nventanas", "ECM + torneo\nde 4 modelos",
                "HAC, bootstrap,\nR2, MAE, DM"]
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")
    ancho, alto, y = 1.7, 0.9, 0.75
    for i, (fase, detalle) in enumerate(zip(fases, detalles)):
        x = 0.2 + i * 1.95
        caja = FancyBboxPatch((x, y), ancho, alto, boxstyle="round,pad=0.05",
                              facecolor="#e0f2fe", edgecolor="#0e7490", linewidth=1.2)
        ax.add_patch(caja)
        ax.text(x + ancho / 2, y + alto - 0.18, fase, ha="center", va="center",
                fontsize=9, fontweight="bold")
        ax.text(x + ancho / 2, y + 0.22, detalle, ha="center", va="center", fontsize=6.5)
        if i < len(fases) - 1:
            ax.add_patch(FancyArrowPatch((x + ancho, y + alto / 2),
                                         (x + 1.95, y + alto / 2),
                                         arrowstyle="-|>", mutation_scale=14, color="#374151"))
    ax.set_title("Flujo KDD aplicado al estudio", fontsize=11)
    fig.savefig(FIG / "01_flujo_kdd.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def copiar_figuras() -> None:
    mapa = {
        "02_cobertura.png": ("panel", "cobertura_anual.png"),
        "03_campanas_precio.png": ("calidad", "campanas_precio.png"),
        "04_campanas_log.png": ("calidad", "campanas_log.png"),
        "05_qq_log.png": ("calidad", "qq_log.png"),
        "06_correlacion_papa.png": ("estructura", "correlacion_papa.png"),
        "07_correlacion_lima.png": ("estructura", "correlacion_lima.png"),
        "08_integracion_distancia.png": ("estructura", "integracion_distancia.png"),
        "09_vida_media_puno.png": ("ecm", "vida_media_puno.png"),
        "10_vida_media_mapa.png": ("ecm", "vida_media_mapa.png"),
        "11_ranking_torneo.png": ("modelos", "ranking_torneo.png"),
        "12_ganador_vs_observado.png": ("modelos", "ganador_vs_observado.png"),
        "13_curvas_aprendizaje.png": ("modelos", "curvas_aprendizaje.png"),
        "14_robustez.png": ("robustez", "robustez_vida_media.png"),
        "15_arbol_rf.png": ("modelos", "arbol_rf.png"),
        "16_arbol_lgbm.png": ("modelos", "arbol_lgbm_sustituto.png"),
        "17_importancia_rf.png": ("modelos", "importancia_rf.png"),
        "18_pdp.png": ("modelos", "pdp_cebolla.png"),
        "19_residuos_ecm.png": ("ecm", "residuos_ecm.png"),
        "20_acf_ecm.png": ("ecm", "acf_ecm.png"),
        "21_adf_region.png": ("calidad", "adf_region.png"),
        "07_adf_resumen.png": ("calidad", "adf_resumen.png"),
        "22_aic_bic.png": ("calidad", "aic_bic.png"),
        "23_verificacion_limpieza.png": ("calidad", "efecto_limpieza.png"),
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
        "tabla_04_estacionariedad.csv": ("calidad", "estacionariedad_producto.csv"),
        "tabla_05_rezagos.csv": ("calidad", "seleccion_rezagos.csv"),
        "tabla_06_ecm_puno.csv": ("ecm", "ecm_puno.csv"),
        "tabla_07_ecm_producto.csv": ("ecm", "ecm_producto.csv"),
        "tabla_08_torneo.csv": ("modelos", "torneo_nowcast_ranking.csv"),
        "tabla_09_ganador.csv": ("modelos", "torneo_nowcast_metricas.csv"),
        "tabla_10_robustez.csv": ("robustez", "ecm_robustez.csv"),
        "tabla_11_asimetria.csv": ("ecm", "asimetria.csv"),
        "tabla_12_correlacion_lima.csv": ("estructura", "correlacion_lima.csv"),
        "tabla_13_geografia.csv": ("estructura", "geografia_integracion.csv"),
        "tabla_14_ecm_region.csv": ("ecm", "ecm_region.csv"),
        "tabla_15_complemento.csv": ("modelos", "torneo_complemento.csv"),
        "tabla_18_diagnostico.csv": ("ecm", "diagnostico_ecm.csv"),
        "tabla_19_estacionariedad_resumen.csv": ("calidad", "estacionariedad_resumen.csv"),
        "tabla_20_transformacion_log.csv": ("calidad", "transformacion_log.csv"),
        "tabla_21_normalidad.csv": ("calidad", "normalidad.csv"),
        "tabla_22_verificacion_limpieza.csv": ("calidad", "verificacion_limpieza.csv"),
    }
    for destino, (carpeta, origen) in mapa.items():
        ruta = DIRECTORIOS[carpeta] / origen
        if ruta.exists():
            shutil.copy2(ruta, TAB / destino)


def protocolo() -> None:
    filas = [
        ("Desarrollo (evaluación)", "2018-01–2020-12 (36 meses)",
         "entrenamiento con datos previos a 2018; ranking con mínimos muestrales"),
        ("Holdout", "2021-01–2026-10", "evaluación final; modelos entrenados hasta 2020"),
        ("Elegibilidad del ranking de nowcast",
         f"≥ {MIN_NOWCAST_TRAIN} entrenamiento; ≥ {MIN_NOWCAST_TEST} evaluación",
         "si no alcanza, se excluye de la Tabla 15"),
        ("Baseline nowcast", "cambio cero", "umbral de valor predictivo"),
        ("Baseline pronóstico", "caminata aleatoria", "umbral de pronóstico"),
        ("Métrica principal nowcast", "R2", "captura de señal, no shrinkage"),
        ("Métrica principal pronóstico", "MAE", "error en S/ kg"),
        ("Métricas secundarias", "MAE, mejora relativa, acierto direccional", "robustez"),
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
    lineas = ["# Indice de artefactos del informe\n",
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
    print(f"Figuras del informe: {len(list(FIG.glob('*.png')))}")
    print(f"Tablas del informe:  {len(list(TAB.glob('*.csv')))}")
    print("Indice:", INFORME / "INDICE.md")


if __name__ == "__main__":
    main()
