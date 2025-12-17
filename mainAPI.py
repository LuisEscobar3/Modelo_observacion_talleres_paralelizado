import os
import json
import time
import uuid
import dotenv
from functools import lru_cache
from pathlib import Path

from langchain.globals import set_debug
from services.llm_manager import load_llms
from Functions.read_csv import read_csv_with_polars
from Functions.Process_indibitual_observations import procesar_observacion_individual

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool


app = FastAPI(title="Observaciones Talleres API", version="1.0.0")

# Cloud Run: /tmp es el lugar estándar para archivos temporales
BASE_DIR = Path(os.environ.get("WORKDIR", "/tmp/obs_service"))
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_gemini():
    dotenv.load_dotenv()
    set_debug(False)
    os.environ["APP_ENV"] = os.environ.get("APP_ENV", "sbx")
    llms = load_llms()
    if "gemini_pro" not in llms:
        raise RuntimeError("No se encontró 'gemini_pro' en load_llms()")
    return llms["gemini_pro"]


def pipeline_por_ruta(ruta_csv: str, max_workers: int = 0, chunksize: int = 0) -> dict:
    df_data = read_csv_with_polars(ruta_csv)

    start = time.perf_counter()
    gemini = get_gemini()

    resultado_json = procesar_observacion_individual(
        df_observacion=df_data,
        prompt_sistema="",
        cliente_llm=gemini,
        max_workers=max_workers,
        chunksize=chunksize,
    )

    elapsed = time.perf_counter() - start
    return {"resultado_json": resultado_json, "elapsed_seconds": round(elapsed, 2)}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/process")
async def process_csv(
    file: UploadFile = File(...),
    max_workers: int = 0,
    chunksize: int = 0
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "El archivo debe ser .csv")

    request_id = uuid.uuid4().hex
    safe_name = Path(file.filename).name
    csv_path = UPLOAD_DIR / f"{request_id}_{safe_name}"

    content = await file.read()
    if not content:
        raise HTTPException(400, "CSV vacío")

    csv_path.write_bytes(content)

    try:
        result = await run_in_threadpool(pipeline_por_ruta, str(csv_path), max_workers, chunksize)
    except Exception as e:
        raise HTTPException(500, f"Error procesando: {e}")

    out_path = OUTPUT_DIR / f"{request_id}.json"
    try:
        out_path.write_text(
            json.dumps(result["resultado_json"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        raise HTTPException(500, f"Error guardando JSON: {e}")

    return {
        "ok": True,
        "request_id": request_id,
        "elapsed_seconds": result["elapsed_seconds"],
        "total_registros": len(result["resultado_json"].get("registros", [])),
        "resultado": result["resultado_json"],
    }
