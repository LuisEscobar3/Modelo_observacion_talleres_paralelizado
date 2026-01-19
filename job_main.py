import os
import sys
import time
import logging
import tempfile
from pathlib import Path

from google.cloud import storage
from langchain_core.globals import set_debug

# Asegúrate de que estos módulos existan en tu estructura de carpetas
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
# GCS CONFIG
# =========================
storage_client = storage.Client()

# Asegúrate de que este nombre sea correcto y no tenga espacios extra
BUCKET_NAME = "bucket-aux-ia-modelo-seguimiento-talleres"


# =========================
# LLM
# =========================
def get_gemini():
    set_debug(False)
    os.environ["APP_ENV"] = os.environ.get("APP_ENV", "sbx")

    llms = load_llms()
    # Validación extra por seguridad
    if "gemini_pro" not in llms:
        raise RuntimeError("❌ No se encontró 'gemini_pro' en la configuración de load_llms()")

    return llms["gemini_pro"]


# =========================
# DESCARGA CSV
# =========================
def descargar_csv_desde_gcs(blob_name: str) -> str:
    """
    Descarga el blob especificado desde el bucket configurado a la carpeta temporal.
    """
    logging.info(f"📄 Blob CSV solicitado: {blob_name}")

    # --- VALIDACIÓN CRÍTICA PARA EVITAR IndexError ---
    if not BUCKET_NAME:
        raise ValueError("❌ Error Crítico: La variable BUCKET_NAME está vacía.")

    logging.info(f"🪣 Usando Bucket: '{BUCKET_NAME}'")

    if not blob_name:
        raise ValueError("❌ El argumento blob_name está vacío.")

    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            logging.error(f"❌ El archivo no existe en GCS: gs://{BUCKET_NAME}/{blob_name}")
            raise FileNotFoundError(f"Blob no encontrado: {blob_name}")

        # 1️⃣ Descargar a RAM
        contenido_bytes = blob.download_as_bytes()

        # 2️⃣ Escribir a /tmp (Compatible con Cloud Run)
        local_path = Path(tempfile.gettempdir()) / Path(blob_name).name
        local_path.write_bytes(contenido_bytes)

        logging.info(f"✅ CSV descargado exitosamente en: {local_path}")
        return str(local_path)

    except Exception as e:
        logging.error(f"💥 Error descargando desde GCS: {e}")
        raise e


# =========================
# PIPELINE PRINCIPAL
# =========================
def pipeline(csv_local_path: str):
    logging.info(f"🚀 Iniciando procesamiento del archivo local: {csv_local_path}")

    # Lectura del CSV
    df_data = read_csv_with_polars(csv_local_path)

    if df_data.is_empty():
        logging.warning("⚠️ El CSV descargado está vacío. Finalizando pipeline.")
        return

    # Carga del modelo
    gemini = get_gemini()

    # Configuración de concurrencia
    cpu_count = os.cpu_count() or 2
    max_workers = max(1, cpu_count - 1)
    chunksize = max(10, len(df_data) // max_workers)

    logging.info(
        f"⚙️ Configuración: CPU={cpu_count} | Workers={max_workers} | Chunksize={chunksize}"
    )

    # Procesamiento
    procesar_observacion_individual(
        df_observacion=df_data,
        prompt_sistema="",  # Asegúrate de definir esto si es necesario
        cliente_llm=gemini,
        max_workers=max_workers,
        chunksize=chunksize,
    )

    logging.info("🏁 Procesamiento finalizado correctamente")


# =========================
# MAIN (ENTRYPOINT)
# =========================
def main():
    """
    Punto de entrada.
    Espera argumentos tipo: blob=inputs/archivo.csv request_id=12345
    """
    # Parseo simple de argumentos clave=valor
    args = dict(arg.split("=", 1) for arg in sys.argv[1:] if "=" in arg)

    blob_name = args.get("blob")
    request_id = args.get("request_id", "N/A")

    logging.info("========================================")
    logging.info(f"🆔 Request ID: {request_id}")
    logging.info(f"📎 Blob recibido: {blob_name}")
    logging.info("========================================")

    if not blob_name:
        logging.error("❌ Falta el argumento 'blob'. Uso: python job_main.py blob=ruta/archivo.csv")
        raise RuntimeError("Falta argumento blob")

    # Ejecución
    csv_local_path = descargar_csv_desde_gcs(blob_name)
    pipeline(csv_local_path)


if __name__ == "__main__":
    main()