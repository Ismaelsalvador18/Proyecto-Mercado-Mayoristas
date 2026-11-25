#!/usr/bin/env bash
# Ejecuta el pipeline completo en el orden metodologico.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"

$PY scripts/pipeline/01_descargar_sisap.py
$PY scripts/validar_fase1.py
$PY scripts/analisis/01.1_auditoria_fuente.py

$PY scripts/pipeline/02_construir_panel.py
$PY scripts/validar_fase2.py
$PY scripts/analisis/02.1_auditoria_panel.py

$PY scripts/pipeline/03_limpiar_panel.py
$PY scripts/validar_fase3.py
$PY scripts/analisis/03.1_verificacion_limpieza.py
$PY scripts/analisis/03.2_descripcion_datos.py
$PY scripts/analisis/03.5_estructura_mercados.py
$PY scripts/validar_fase4.py

$PY scripts/pipeline/05_torneo_modelos.py
$PY scripts/analisis/05.1_representacion_modelos.py
$PY scripts/analisis/05.2_evaluacion_modelos.py
$PY scripts/analisis/05.3_n_etapas.py
$PY scripts/validar_fase6.py

$PY scripts/informe/07_resultados_informe.py
$PY scripts/validar_fase7.py
$PY scripts/informe/08_generar_informe.py
$PY scripts/validar_fase8.py

echo "Pipeline completo. Artefactos en outputs/."
