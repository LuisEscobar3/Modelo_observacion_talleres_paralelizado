import os
import sys
import time
import logging
import tempfile
from pathlib import Path

from google.cloud import storage
from langchain_core.globals import set_debug

from services.llm_manager import load_llms
from Functions.read_csv import read_csv_with_polars
from Functions.Process_indibitual_observations import procesar_observacion_individual


# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# =========================
# GCS CONFIG (BUCKET QUEMADO)
# =========================
storage_client = storage.Client()

BUCKET_NAME = os.getenv(
    "GCS_BUCKET_NAME",
    "bucket-aux-ia-modelo-seguimiento-talleres"
)


# =========================
# LLM
# =========================
def get_gemini():
    set_debug(False)
    os.environ["APP_ENV"] = os.environ.get("APP_ENV", "sbx")

    llms = load_llms()
    if "gemini_pro" not in llms:
        raise RuntimeError("No se encontró gemini_pro en load_llms()")

    return llms["gemini_pro"]


# =========================
# DESCARGA CSV (MISMO PATRÓN QUE IMAGEN)
# =========================
def descargar_csv_desde_gcs(blob_name: str) -> str:
    """
    MISMA lógica que tu función de imagen:
    1. bucket fijo
    2. blob(blob_name)
    3. download_as_bytes()
    4. escribir en /tmp
    """
    logging.info(f"📄 Blob CSV recibido: {blob_name}")
    logging.info(f"🪣 Bucket usado: {BUCKET_NAME}")

    if not blob_name:
        raise ValueError("blob_name vacío")

    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        logging.error(f"❌ CSV no encontrado en GCS: {blob_name}")
        raise FileNotFoundError(blob_name)

    # 1️⃣ Descargar a RAM (IGUAL que tu código de imagen)
    contenido_bytes = blob.download_as_bytes()

    # 2️⃣ Escribir a /tmp (Cloud Run friendly)
    local_path = Path(tempfile.gettempdir()) / Path(blob_name).name
    local_path.write_bytes(contenido_bytes)

    logging.info(f"✅ CSV descargado en: {local_path}")
    return str(local_path)


# =========================
# PIPELINE PRINCIPAL
# =========================
def pipeline(csv_local_path: str):
    logging.info(f"🚀 Iniciando procesamiento CSV: {csv_local_path}")

    df_data = read_csv_with_polars(csv_local_path)

    gemini = get_gemini()

    cpu_count = os.cpu_count() or 2
    max_workers = max(1, cpu_count - 1)
    chunksize = max(10, len(df_data) // max_workers)

    logging.info(
        f"⚙️ CPU={cpu_count} | workers={max_workers} | chunksize={chunksize}"
    )

    procesar_observacion_individual(
        df_observacion=df_data,
        prompt_sistema="",
        cliente_llm=gemini,
        max_workers=max_workers,
        chunksize=chunksize,
    )

    logging.info("🏁 Procesamiento finalizado correctamente")


# =========================
# MAIN (ENTRYPOINT DEL JOB)
# =========================
def main():
    """
    Espera argumentos:
      blob=inputs/xxxx.csv
      request_id=xxxx
    """
    args = dict(arg.split("=", 1) for arg in sys.argv[1:] if "=" in arg)

    blob_name = args.get("blob")
    request_id = args.get("request_id")

    logging.info(f"🆔 request_id: {request_id}")
    logging.info(f"📎 blob recibido: {blob_name}")

    if not blob_name:
        raise RuntimeError("Falta argumento blob")

    csv_local_path = descargar_csv_desde_gcs(blob_name)

    pipeline(csv_local_path)


if __name__ == "__main__":
    main()
