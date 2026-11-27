#!/usr/bin/env python3
"""Valida la Fase 7: artefactos, documentos, cuadernos y consistencia narrativa."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
INFORME = RAIZ / "outputs" / "informe"
TABLAS = INFORME / "tablas"
FIGURAS = INFORME / "figuras"
MANIFIESTOS = RAIZ / "outputs" / "_manifests"
MANIFIESTO_RESULTADOS = MANIFIESTOS / "07_resultados.json"
MANIFIESTO_VALIDACION = MANIFIESTOS / "07_validacion.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalogo import EXCLUIDOS, cargar_catalogo  # noqa: E402

TABLAS_ESPERADAS = {
    "tabla_01_variables.csv": 7,
    "tabla_01b_ejemplo.csv": 8,
    "tabla_02_auditoria.csv": 11,
    "tabla_03_descriptivos.csv": 9,
    "tabla_08_torneo.csv": 6,
    "tabla_09_ganador.csv": 70,
    "tabla_12_correlacion_lima.csv": 155,
    "tabla_13_geografia.csv": 23,
    "tabla_16_protocolo.csv": 11,
    "tabla_17_software.csv": 12,
    "tabla_20_transformacion_log.csv": 9,
    "tabla_21_normalidad.csv": 9,
    "tabla_22_verificacion_limpieza.csv": 10,
    "tabla_23_pronostico_holdout.csv": 6,
    "tabla_24_matriz_diseno.csv": 8,
    "tabla_25_confusion_modelos.csv": 6,
}

PATRONES_ANTIGUOS = [
    r"\b27 regiones\b",
    r"\b24 regiones\b",
    r"(?<![\d.,])243(?![\d.,])",
    r"(?<![\d.,])599(?![\d.,])",
    r"\b49[ .,]?041\b",
    r"\b49[ .,]?113\b",
    r"\b115[ .,]?294\b",
    r"\b72 atípicos\b",
    r"\b72 atipicos\b",
    r"\b72 casos\b",
    r"\b85,5\b",
    r"\b65,8\b",
    r"\b0,017\b",
    r"\b0,247\b",
    r"\b53 %\b",
    r"\b28 consultas\b",
    r"papa, arroz, cebolla y huevos",
    r"Ridge y OLS mejoran en papa, arroz",
]


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


def strings_notebook(ruta: Path) -> list[str]:
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    salida: list[str] = []

    def recorrer(valor) -> None:
        if isinstance(valor, str):
            salida.append(valor)
        elif isinstance(valor, dict):
            for item in valor.values():
                recorrer(item)
        elif isinstance(valor, list):
            for item in valor:
                recorrer(item)

    recorrer(datos)
    return salida


def main() -> None:
    catalogo = cargar_catalogo()
    ids = set(catalogo["departamento_id"])
    permitidos = set(catalogo["departamento_ascii"])
    sin_lima = permitidos - {"Lima"}

    faltan = sorted(set(TABLAS_ESPERADAS) - {p.name for p in TABLAS.glob("*.csv")})
    sobran = sorted({p.name for p in TABLAS.glob("*.csv")} - set(TABLAS_ESPERADAS))
    if faltan or sobran:
        raise AssertionError(f"Tablas inesperadas. Faltan={faltan}; sobran={sobran}")
    for nombre, filas in TABLAS_ESPERADAS.items():
        datos = pd.read_csv(TABLAS / nombre)
        if len(datos) != filas:
            raise AssertionError(f"{nombre}: {len(datos)} filas; se esperaban {filas}")
        revisar_excluidos(datos, nombre)

    figuras = sorted(p.name for p in FIGURAS.glob("*.png"))
    if len(figuras) != 16:
        raise AssertionError(f"Se esperaban 16 figuras; hay {len(figuras)}")
    if "24_matriz_confusion.png" not in figuras or "01_flujo_kdd.png" not in figuras:
        raise AssertionError("Faltan figuras clave del informe")

    indice = (INFORME / "INDICE.md").read_text(encoding="utf-8")
    for archivo in list(TABLAS_ESPERADAS) + figuras:
        if archivo not in indice:
            raise AssertionError(f"El indice no referencia {archivo}")

    variables = pd.read_csv(TABLAS / "tabla_01_variables.csv")
    if variables.loc[variables["variable"] == "region", "unidad"].iloc[0] != "24 departamentos":
        raise AssertionError("La tabla de variables no declara 24 departamentos")

    auditoria = pd.read_csv(TABLAS / "tabla_02_auditoria.csv").set_index("metrica")["valor"]
    if int(float(auditoria["filas"])) != 43921:
        raise AssertionError("La auditoria no tiene 43 921 filas")
    if int(float(auditoria["departamentos"])) != 24:
        raise AssertionError("La auditoria no tiene 24 departamentos")
    if int(float(auditoria["atipicos detectados"])) != 66:
        raise AssertionError("La auditoria no tiene 66 atipicos")

    limpieza = pd.read_csv(TABLAS / "tabla_22_verificacion_limpieza.csv").set_index("metrica")["valor"]
    if float(limpieza["filas antes"]) != 43921 or float(limpieza["filas despues"]) != 43855:
        raise AssertionError("La verificacion de limpieza no coincide")
    if float(limpieza["atipicos marcados"]) != 66:
        raise AssertionError("La verificacion no tiene 66 atipicos")

    corr = pd.read_csv(TABLAS / "tabla_12_correlacion_lima.csv")
    if not set(corr["region"]) <= sin_lima:
        raise AssertionError("La correlacion con Lima contiene regiones no oficiales")
    if corr.duplicated(["producto", "region"]).any():
        raise AssertionError("La correlacion con Lima tiene duplicados")

    geo = pd.read_csv(TABLAS / "tabla_13_geografia.csv")
    if len(geo) != 23 or set(geo["region"]) != sin_lima:
        raise AssertionError("La geografia no cubre los 23 departamentos sin Lima")

    # Documentación y cuadernos.
    docs = (RAIZ / "README.md").read_text(encoding="utf-8")
    for ruta in sorted((RAIZ / "docs").glob("*.md")):
        docs += "\n" + ruta.read_text(encoding="utf-8")
    obligatorios = ["24 departamentos", "43 855", "66 atípicos", "216", "85,7",
                    "66,5", "p = 0,022", "p = 0,196", "54 %", "papa, cebolla y arroz"]
    for texto in obligatorios:
        if texto not in docs:
            raise AssertionError(f"Falta en documentacion: {texto!r}")
    for patron in PATRONES_ANTIGUOS:
        if re.search(patron, docs):
            raise AssertionError(f"Documentacion conserva un valor antiguo: {patron}")

    for nombre in ["01_resultados.ipynb", "02_guia_detallada.ipynb"]:
        ruta = RAIZ / "notebooks" / nombre
        textos = strings_notebook(ruta)
        contenido = "\n".join(textos)
        for patron in PATRONES_ANTIGUOS:
            if re.search(patron, contenido):
                raise AssertionError(f"{nombre} conserva un valor antiguo: {patron}")
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        code = [c for c in datos.get("cells", []) if c.get("cell_type") == "code"]
        if not code or any(c.get("execution_count") is None for c in code):
            raise AssertionError(f"{nombre} tiene celdas sin ejecutar")
        for c in code:
            for output in c.get("outputs", []):
                if output.get("output_type") == "error":
                    raise AssertionError(f"{nombre} tiene un error de ejecucion")

    archivos = [TABLAS / nombre for nombre in TABLAS_ESPERADAS] + [FIGURAS / nombre for nombre in figuras]
    hashes = {str(ruta.relative_to(RAIZ)): sha256_archivo(ruta) for ruta in archivos}
    manifiesto = {
        "fase": "07_resultados",
        "catalogo_sha256": sha256_archivo(
            RAIZ / "data" / "reference" / "departamentos.csv"
        ),
        "tablas": len(TABLAS_ESPERADAS),
        "figuras": len(figuras),
        "departamentos": 24,
        "departamentos_sin_lima": 23,
        "archivos": hashes,
    }
    MANIFIESTOS.mkdir(parents=True, exist_ok=True)
    MANIFIESTO_RESULTADOS.write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    validacion = {
        "fase": "07_validacion",
        "ok": True,
        "tablas": len(TABLAS_ESPERADAS),
        "figuras": len(figuras),
        "departamentos": 24,
        "departamentos_sin_lima": 23,
        "notebooks": 2,
        "documentos_revisados": 1 + len(list((RAIZ / "docs").glob("*.md"))),
        "archivos_validados": len(archivos),
        "hashes": hashes,
    }
    MANIFIESTO_VALIDACION.write_text(
        json.dumps(validacion, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Fase 7 valida.")
    print(f"  Tablas: {len(TABLAS_ESPERADAS)} | Figuras: {len(figuras)}")
    print("  Documentacion: sin cifras antiguas")
    print("  Cuadernos: ejecutados y sin errores")
    print(f"  Manifiesto: {MANIFIESTO_VALIDACION.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
