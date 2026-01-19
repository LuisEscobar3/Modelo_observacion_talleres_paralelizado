import sys
import os
import time
import dotenv
from functools import lru_cache
from pathlib import Path

from google.cloud import storage
from langchain_core.globals import set_debug

from services.llm_manager import load_llms
from Functions.read_csv import read_csv_with_polars
from Functions.Process_indibitual_observations import procesar_observacion_individual


# ======================================================
# LLM (GEMINI SOLO EN EL JOB)
# ======================================================
@lru_cache(maxsize=1)
def get_gemini():
    dotenv.load_dotenv()
    set_debug(False)

    os.environ["APP_ENV"] = os.environ.get("APP_ENV", "sbx")

    llms = load_llms()
    if "gemini_pro" not in llms:
        raise RuntimeError("❌ No se encontró 'gemini_pro' en load_llms()")

    return llms["gemini_pro"]


# ======================================================
# UTILIDADES
# ======================================================
def get_arg(name: str) -> str:
    """
    Obtiene argumentos tipo: key=value
    """
    for arg in sys.argv[1:]:
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    raise ValueError(f"❌ Argumento requerido faltante: {name}")


def download_from_gcs(gcs_path: str) -> str:
    """
    Descarga un archivo desde GCS (gs://bucket/path)
    y lo guarda en /tmp/input.csv
    """
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"❌ Ruta GCS inválida: {gcs_path}")

    # Eliminar gs://
    path = gcs_path.replace("gs://", "", 1)

    # Separar bucket y blob
    try:
        bucket_name, blob_name = path.split("/", 1)
    except ValueError:
        raise ValueError(f"❌ Ruta GCS incompleta: {gcs_path}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    local_path = Path("/tmp/input.csv")
    blob.download_to_filename(local_path)

    return str(local_path)


# ======================================================
# MAIN JOB
# ======================================================
def main():
    print("🚀 Job iniciado")
    print("🧾 Argumentos recibidos:", sys.argv)

    # Leer argumentos
    csv_gcs_path = get_arg("csv_path")
    request_id = get_arg("request_id")

    print(f"🆔 Request ID: {request_id}")
    print(f"📥 CSV en GCS: {csv_gcs_path}")

    # Descargar CSV
    csv_path = download_from_gcs(csv_gcs_path)
    print(f"📄 CSV descargado en: {csv_path}")

    start = time.perf_counter()

    # Leer CSV
    df_data = read_csv_with_polars(csv_path)
    total_registros = len(df_data)
    print(f"📊 Registros a procesar: {total_registros}")

    # Cargar Gemini
    gemini = get_gemini()
    print("🤖 Gemini cargado correctamente")

    # Concurrencia controlada
    cpu_count = os.cpu_count() or 2
    max_workers = min(4, max(1, cpu_count - 1))
    chunksize = max(50, total_registros // max_workers)

    print(f"⚙️ CPU: {cpu_count}")
    print(f"⚙️ max_workers: {max_workers}")
    print(f"⚙️ chunksize: {chunksize}")

    # Procesamiento principal
    procesar_observacion_individual(
        df_observacion=df_data,
        prompt_sistema="",
        cliente_llm=gemini,
        max_workers=max_workers,
        chunksize=chunksize,
    )

    elapsed = round(time.perf_counter() - start, 2)
    print(f"✅ Job terminado correctamente")
    print(f"⏱️ Tiempo total: {elapsed} segundos")


# ======================================================
# ENTRYPOINT
# ======================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ Error fatal en el Job")
        print(e)
        sys.exit(1)
