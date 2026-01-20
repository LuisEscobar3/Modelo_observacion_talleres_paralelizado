import os
import sys
import time
import logging
import tempfile
import math  # <--- CRITICO: Necesario para la división matemática
from pathlib import Path

from google.cloud import storage
from langchain_core.globals import set_debug

# TUS MODULOS (Asegúrate que estén en la imagen Docker)
from services.llm_manager import load_llms
from Functions.read_csv import read_csv_with_polars
from Functions.Process_indibitual_observations import procesar_observacion_individual

# =========================
# CONFIGURACIÓN
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

storage_client = storage.Client()
BUCKET_NAME = "bucket-aux-ia-modelo-seguimiento-talleres"


# =========================
# HELPERS
# =========================
def get_gemini():
    set_debug(False)
    os.environ["APP_ENV"] = os.environ.get("APP_ENV", "sbx")
    llms = load_llms()
    if "gemini_pro" not in llms:
        raise RuntimeError("❌ Error: 'gemini_pro' no encontrado en load_llms()")
    return llms["gemini_pro"]


def descargar_csv_desde_gcs(blob_name: str) -> str:
    logging.info(f"⬇️ Descargando Blob: {blob_name}")
    if not BUCKET_NAME: raise ValueError("❌ BUCKET_NAME vacío")

    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    if not blob.exists(): raise FileNotFoundError(f"No existe: {blob_name}")

    local_path = Path(tempfile.gettempdir()) / Path(blob_name).name
    blob.download_to_filename(str(local_path))
    logging.info(f"✅ Descargado en: {local_path}")
    return str(local_path)


# =========================
# PIPELINE (CORE LOGIC)
# =========================
def pipeline(csv_local_path: str):
    logging.info(f"🚀 Iniciando Pipeline: {csv_local_path}")

    # 1. LEER CSV (Polars es eficiente)
    df_data = read_csv_with_polars(csv_local_path)
    if df_data.is_empty():
        logging.warning("⚠️ CSV vacío. Terminando.")
        return

    # 2. SHARDING (DIVISIÓN DE DATOS)
    # Obtenemos el índice de esta máquina (0, 1, 2... 9)
    try:
        task_index = int(os.environ.get("CLOUD_RUN_TASK_INDEX", 0))
        task_count = int(os.environ.get("CLOUD_RUN_TASK_COUNT", 1))
    except ValueError:
        task_index = 0;
        task_count = 1

    total_rows = len(df_data)

    # Calculamos el tamaño exacto del pedazo
    chunk_size_per_task = math.ceil(total_rows / task_count)

    # Definimos INICIO y FIN para ESTA máquina
    start_idx = task_index * chunk_size_per_task
    end_idx = start_idx + chunk_size_per_task

    # CORTAMOS EL DATAFRAME (Slicing)
    # Esto asegura que ninguna máquina toque los datos de otra
    df_sharded = df_data[start_idx: min(end_idx, total_rows)]

    logging.info(
        f"🔢 TAREA {task_index + 1}/{task_count} | Rango: {start_idx} a {end_idx} | Total filas: {len(df_sharded)}")

    if df_sharded.is_empty():
        logging.warning("⚠️ Worker sin filas asignadas. Finalizando.")
        return

    # 3. CONFIGURACIÓN DE WORKERS (HILOS)
    # Estrategia: 4 CPUs Físicos + Latencia de BD alta = 16 Hilos.
    # Esto mantiene ocupada la máquina mientras espera respuesta de la BD.
    max_workers = 16

    # Chunksize interno para los hilos
    chunksize = max(5, len(df_sharded) // (max_workers * 2))

    logging.info(f"🔥 PROCESANDO: 4 CPUs | {max_workers} Hilos Simultáneos | Chunksize={chunksize}")

    gemini = get_gemini()

    # 4. EJECUCIÓN
    procesar_observacion_individual(
        df_observacion=df_sharded,  # <-- Pasamos solo el pedazo recortado
        prompt_sistema="",
        cliente_llm=gemini,
        max_workers=max_workers,
        chunksize=chunksize,
    )

    logging.info(f"🏁 Tarea {task_index} completada con éxito.")


# =========================
# MAIN
# =========================
def main():
    # Parseo de argumentos key=value
    args = dict(arg.split("=", 1) for arg in sys.argv[1:] if "=" in arg)
    blob_name = args.get("blob")
    request_id = args.get("request_id", "N/A")

    logging.info(f"🆔 Request ID: {request_id}")

    if not blob_name:
        raise RuntimeError("Falta argumento 'blob'")

    csv_local_path = descargar_csv_desde_gcs(blob_name)
    pipeline(csv_local_path)


if __name__ == "__main__":
    main()