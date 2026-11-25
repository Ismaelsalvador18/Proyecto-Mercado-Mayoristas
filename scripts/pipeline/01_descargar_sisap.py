#!/usr/bin/env python3
"""Descarga los precios mayoristas mensuales por departamento desde SISAP.

Fuente: SISAP - MIDAGRI, modulo "Precios de ciudades".
  http://sistemas.midagri.gob.pe/sisap/portal2/ciudades

El portal no ofrece una descarga unica: hay que consultar departamento por departamento.
Este script reproduce la consulta que hace el formulario web (GET a
/resumenes/filtrar con ajax=true) y guarda:

  data/raw/ciudades_<codigo>.html   respuesta cruda de cada departamento
  data/processed/precios_ciudades_largo.csv
  outputs/_manifests/01_fuente.json  manifiesto de la fase

Variables descargadas:
  may_precio_prom = precio promedio mayorista

Uso:
  .venv/bin/python scripts/01_descargar_sisap.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

RAIZ = Path(__file__).resolve().parent.parent.parent
RAW = RAIZ / "data" / "raw"
PROCESADOS = RAIZ / "data" / "processed"
MANIFIESTOS = RAIZ / "outputs" / "_manifests"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from catalogo import cargar_catalogo, mapa_codigos  # noqa: E402

CATALOGO = cargar_catalogo()
REGIONES = mapa_codigos(CATALOGO)

BASE = "http://sistemas.midagri.gob.pe/sisap/portal2/ciudades"
VARIABLE = "may_precio_prom"
ANIOS = [str(a) for a in range(1997, 2027)]
MESES = [f"{m:02d}" for m in range(1, 13)]

# El catalogo define los 24 codigos SISAP permitidos. No se consultan Callao
# ni los mercados provinciales Andahuaylas, Chota y Jaen.

# Codigos de genero (categoria amplia). El portal devuelve todas las variedades.
PRODUCTOS = {
    "0104": "Papa",
    "0212": "Cebolla",
    "0228": "Tomate",
    "0230": "Zanahoria",
    "0611": "Limon",
    "0401": "Arroz",
    "1305": "Pollo",
    "1105": "Huevos",
    "1001": "Aceite",
    "1005": "Azucar",
}

CABECERA = [
    "departamento_id",
    "departamento",
    "region_codigo",
    "region",
    "codigo_producto",
    "producto",
    "unidad",
    "equivalencia_kg",
    "anio",
    "mes",
    "valor",
    "precio_por_kg",
]


def crear_sesion() -> requests.Session:
    sesion = requests.Session()
    sesion.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120 Safari/537.36"
            )
        }
    )
    return sesion


def obtener_postid(sesion: requests.Session) -> str:
    respuesta = sesion.get(BASE, timeout=30)
    respuesta.raise_for_status()
    sopa = BeautifulSoup(respuesta.text, "html.parser")
    campo = sopa.find("input", {"name": "postID"})
    if campo is None:
        raise RuntimeError("No se encontro el token postID en el portal")
    return campo["value"]


def descargar_region(sesion: requests.Session, postid: str, region: str) -> str:
    parametros = {
        "__ajax_carga_final": "consulta",
        "postID": postid,
        "fecha": time.strftime("%d/%m/%Y"),
        "desde": "01/01/1997",
        "hasta": "31/12/2026",
        "region": region,
        "meses[]": MESES,
        "anios[]": ANIOS,
        "periodicidad": "mensual",
        "productos[]": list(PRODUCTOS),
        "variables[]": VARIABLE,
    }
    respuesta = sesion.get(
        BASE + "/resumenes/filtrar",
        params=parametros,
        headers={"X-Requested-With": "XMLHttpRequest", "Referer": BASE},
        timeout=180,
    )
    respuesta.raise_for_status()
    return respuesta.text


def numero(texto: str) -> float | None:
    texto = (texto or "").strip()
    if texto in {"", "-", "..", "..."}:
        return None
    texto = texto.replace(" ", "")
    try:
        return float(texto)
    except ValueError:
        return None


def parsear(
    html: str,
    region_codigo: str,
    region: str,
    departamento_id: str,
    departamento: str,
) -> list[dict]:
    sopa = BeautifulSoup(html, "lxml")
    tabla = sopa.find("table")
    if tabla is None:
        return []

    filas = tabla.find_all("tr")
    registros: list[dict] = []
    codigo = None
    producto = None
    unidad = None
    equivalencia = 1.0

    for fila in filas:
        celdas = [c.get_text(" ", strip=True) for c in fila.find_all(["td", "th"])]
        if not celdas:
            continue

        primera = celdas[0] if celdas else ""
        if re.fullmatch(r"\d{6}", primera or ""):
            codigo = primera
            producto = celdas[1] if len(celdas) > 1 else ""
            unidad = None
            equivalencia = 1.0
            continue

        if codigo is None:
            continue

        anio = celdas[3] if len(celdas) > 3 else ""
        if not re.fullmatch(r"\d{4}", anio or ""):
            # fila de unidad de medida: producto vacio, unidad y equivalencia
            if len(celdas) > 2 and celdas[1] and not celdas[3]:
                unidad = celdas[1]
                equivalencia = numero(celdas[2]) or 1.0
            continue

        equiv = equivalencia if equivalencia and equivalencia > 0 else 1.0
        for indice, valor_texto in enumerate(celdas[6:18], start=1):
            valor = numero(valor_texto)
            if valor is None:
                continue
            registros.append(
                {
                    "departamento_id": departamento_id,
                    "departamento": departamento,
                    "region_codigo": region_codigo,
                    "region": region,
                    "codigo_producto": codigo,
                    "producto": producto,
                    "unidad": unidad or "Kilogramo",
                    "equivalencia_kg": equiv,
                    "anio": int(anio),
                    "mes": indice,
                    "valor": valor,
                    "precio_por_kg": round(valor / equiv, 6) if equiv else None,
                }
            )
    return registros


def sha256_archivo(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESADOS.mkdir(parents=True, exist_ok=True)
    MANIFIESTOS.mkdir(parents=True, exist_ok=True)

    sesion = crear_sesion()
    postid = obtener_postid(sesion)
    print(f"postID obtenido: {postid}")

    todos: list[dict] = []
    for region_codigo, meta in REGIONES.items():
        region = meta["departamento_ascii"]
        departamento_id = meta["departamento_id"]
        departamento = meta["departamento"]
        destino = RAW / f"ciudades_{region_codigo}.html"
        if destino.exists() and destino.stat().st_size > 5000:
            html = destino.read_text(encoding="utf-8", errors="ignore")
            print(f"{departamento:16s} cache")
        else:
            html = descargar_region(sesion, postid, region_codigo)
            destino.write_text(html, encoding="utf-8")
            print(f"{departamento:16s} descargado ({len(html):,} bytes)")
            time.sleep(0.6)

        registros = parsear(html, region_codigo, region, departamento_id, departamento)
        todos.extend(registros)
        print(f"{'':16s} {len(registros):5d} filas")
        if not registros:
            print(f"  AVISO: sin datos para {departamento}")

    codigos_observados = {fila["region_codigo"] for fila in todos}
    ids_observados = {fila["departamento_id"] for fila in todos}
    if codigos_observados != set(REGIONES):
        raise RuntimeError(
            "La descarga no cubre exactamente los codigos del catalogo: "
            f"faltan={sorted(set(REGIONES) - codigos_observados)}, "
            f"sobran={sorted(codigos_observados - set(REGIONES))}"
        )
    if ids_observados != set(CATALOGO["departamento_id"]):
        raise RuntimeError(
            "La descarga no cubre exactamente los 24 departamentos: "
            f"faltan={sorted(set(CATALOGO['departamento_id']) - ids_observados)}, "
            f"sobran={sorted(ids_observados - set(CATALOGO['departamento_id']))}"
        )

    salida = PROCESADOS / "precios_ciudades_largo.csv"
    with salida.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=CABECERA, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(todos)

    conteo = Counter(fila["departamento_id"] for fila in todos)
    categorias_datos = len({str(fila["codigo_producto"]).zfill(6)[:4] for fila in todos})
    variedades_datos = len({fila["producto"] for fila in todos})
    manifiesto = {
        "fase": "01_fuente",
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "catalogo": str((RAIZ / "data" / "reference" / "departamentos.csv").relative_to(RAIZ)),
        "catalogo_sha256": sha256_archivo(RAIZ / "data" / "reference" / "departamentos.csv"),
        "departamentos": int(len(CATALOGO)),
        "departamentos_ids": sorted(ids_observados),
        "codigos_sisap": sorted(codigos_observados),
        "filas": len(todos),
        "categorias": categorias_datos,
        "variedades": variedades_datos,
        "anio_min": min(fila["anio"] for fila in todos),
        "anio_max": max(fila["anio"] for fila in todos),
        "filas_por_departamento": dict(sorted(conteo.items())),
        "salida": str(salida.relative_to(RAIZ)),
        "salida_sha256": sha256_archivo(salida),
    }
    destino_manifiesto = MANIFIESTOS / "01_fuente.json"
    destino_manifiesto.write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nTotal: {len(todos):,} filas -> {salida.relative_to(RAIZ)}")
    print(f"Consultas: {len(REGIONES)} departamentos, {len(PRODUCTOS)} categorías solicitadas")
    print(f"Con datos: {len(ids_observados)} departamentos | {categorias_datos} categorías | "
          f"{variedades_datos} variedades")
    print(f"Manifiesto: {destino_manifiesto.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
