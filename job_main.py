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
# LLM (GEMINI SOLO AQUÍ)
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
# UTILS
# =========================
def get_arg(name: str) -> str:
    for arg in sys.argv[1:]:
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    raise ValueError(f"Argumento faltante: {name}")


def download_from_gcs(gcs_path: str) -> str:
    client = storage.Client()

    _, bucket_name, *blob_parts = gcs_path.split("/")
    blob_name = "/".join(blob_parts)

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    local_path = Path("/tmp/input.csv")
    blob.download_to_filename(local_path)

    return str(local_path)


# =========================
# MAIN JOB
# =========================
def main():
    csv_gcs_path = get_arg("csv_path")
    request_id = get_arg("request_id")

    print(f"▶️ Iniciando job {request_id}")
    print(f"📥 Descargando CSV desde {csv_gcs_path}")

    csv_path = download_from_gcs(csv_gcs_path)

    start = time.perf_counter()

    df_data = read_csv_with_polars(csv_path)
    gemini = get_gemini()

    cpu_count = os.cpu_count() or 2
    max_workers = min(4, max(1, cpu_count - 1))
    chunksize = 500

    procesar_observacion_individual(
        df_observacion=df_data,
        prompt_sistema="",
        cliente_llm=gemini,
        max_workers=max_workers,
        chunksize=chunksize,
    )

    elapsed = round(time.perf_counter() - start, 2)
    print(f"✅ Job {request_id} terminado en {elapsed} segundos")


if __name__ == "__main__":
    main()
