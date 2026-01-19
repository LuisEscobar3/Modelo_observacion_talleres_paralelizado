import sys
import os
import json
import time
import dotenv
from functools import lru_cache
from pathlib import Path

from langchain_core.globals import set_debug

from services.llm_manager import load_llms
from Functions.read_csv import read_csv_with_polars
from Functions.Process_indibitual_observations import procesar_observacion_individual


# =========================
# LLM (MISMO QUE TU CÓDIGO)
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


def get_arg(name: str):
    for arg in sys.argv[1:]:
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    raise ValueError(f"Argumento faltante: {name}")


def main():
    csv_path = get_arg("csv_path")
    request_id = get_arg("request_id")

    start = time.perf_counter()

    df_data = read_csv_with_polars(csv_path)
    gemini = get_gemini()

    cpu_count = os.cpu_count() or 2
    max_workers = min(4, max(1, cpu_count - 1))
    chunksize = 500

    resultado_json = procesar_observacion_individual(
        df_observacion=df_data,
        prompt_sistema="",
        cliente_llm=gemini,
        max_workers=max_workers,
        chunksize=chunksize,
    )

    elapsed = round(time.perf_counter() - start, 2)

    output = {
        "request_id": request_id,
        "elapsed_seconds": elapsed,
        "resultado_json": resultado_json
    }

    out_path = Path("/tmp") / f"{request_id}.json"
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"✅ Job terminado: {out_path}")


if __name__ == "__main__":
    main()
