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


# =========================
# LLM
# =========================
@lru_cache(maxsize=1)
def get_gemini():
    dotenv.load_dotenv()
    set_debug(False)
    os.environ["APP_ENV"] = os.environ.get("APP_ENV", "sbx")

    llms = load_llms()
    if "gemini_pro" not in llms:
        raise RuntimeError("No se encontró 'gemini_pro'")

    return llms["gemini_pro"]


# =========================
# ARG PARSER
# =========================
def get_arg(name: str) -> str:
    for arg in sys.argv[1:]:
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    raise ValueError(f"Argumento requerido faltante: {name}")


# =========================
# GCS DOWNLOAD (ULTRA LOGUEADO)
# =========================
def download_from_gcs(gcs_path: str) -> str:
    print(f"🔍 csv_gcs_path recibido: '{gcs_path}'")

    if not gcs_path or not isinstance(gcs_path, str):
        raise ValueError("csv_gcs_path vacío o inválido")

    gcs_path = gcs_path.strip()

    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Ruta GCS inválida: {gcs_path}")

    path = gcs_path[len("gs://"):]
    if "/" not in path:
        raise ValueError(f"Ruta GCS incompleta: {gcs_path}")

    bucket_name, blob_name = path.split("/", 1)

    print(f"🪣 Bucket parseado: '{bucket_name}'")
    print(f"📄 Blob parseado: '{blob_name}'")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    local_path = Path("/tmp/input.csv")
    blob.download_to_filename(local_path)

    print(f"✅ Archivo descargado en {local_path}")
    return str(local_path)


# =========================
# MAIN
# =========================
def main():
    print("🚀 Job iniciado")
    print("🧾 sys.argv recibido:")
    for arg in sys.argv:
        print(f"   {arg}")

    csv_gcs_path = get_arg("csv_path")
    request_id = get_arg("request_id")

    print(f"🆔 request_id = {request_id}")
    print(f"📥 csv_gcs_path = {csv_gcs_path}")

    csv_path = download_from_gcs(csv_gcs_path)

    start = time.perf_counter()

    df_data = read_csv_with_polars(csv_path)
    print(f"📊 Registros leídos: {len(df_data)}")

    gemini = get_gemini()
    print("🤖 Gemini cargado")

    cpu_count = os.cpu_count() or 2
    max_workers = min(4, max(1, cpu_count - 1))
    chunksize = max(50, len(df_data) // max_workers)

    procesar_observacion_individual(
        df_observacion=df_data,
        prompt_sistema="",
        cliente_llm=gemini,
        max_workers=max_workers,
        chunksize=chunksize,
    )

    elapsed = round(time.perf_counter() - start, 2)
    print(f"✅ Job finalizado en {elapsed} segundos")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ ERROR FATAL EN EL JOB")
        print(e)
        sys.exit(1)
