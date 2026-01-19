import os
import json
import time
import uuid
import dotenv
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from langchain_core.globals import set_debug

from services.llm_manager import load_llms
from Functions.read_csv import read_csv_with_polars
from Functions.Process_indibitual_observations import procesar_observacion_individual


# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Observaciones Talleres API", version="1.0.0")


# =========================
# DIRECTORIOS (LOCAL / CLOUD RUN)
# =========================
BASE_DIR = Path(os.environ.get("WORKDIR", "/tmp/obs_service"))
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# LLM (CACHEADO)
# =========================
@lru_cache(maxsize=1)
def get_gemini():
    dotenv.load_dotenv()
    set_debug(False)
    os.environ["APP_ENV"] = os.environ.get("APP_ENV", "sbx")

    llms = load_llms()
    if "gemini_pro" not in llms:
        raise RuntimeError("No se encontró 'gemini_pro' en load_llms()")

    return llms["gemini_pro"]


# =========================
# PIPELINE PRINCIPAL
# =========================
def pipeline_por_ruta(ruta_csv: str) -> dict:
    df_data = read_csv_with_polars(ruta_csv)

    start = time.perf_counter()
    gemini = get_gemini()

    # 👉 Concurrencia se calcula INTERNAMENTE
    cpu_count = os.cpu_count() or 2
    max_workers = max(1, cpu_count - 1)
    chunksize = max(10, len(df_data) // max_workers)

    resultado_json = procesar_observacion_individual(
        df_observacion=df_data,
        prompt_sistema="",
        cliente_llm=gemini,
        max_workers=max_workers,
        chunksize=chunksize,
    )

    elapsed = time.perf_counter() - start

    return {
        "resultado_json": resultado_json,
        "elapsed_seconds": round(elapsed, 2)
    }


# =========================
# BACKGROUND TASK
# =========================
def process_csv_background(csv_path: str, request_id: str):
    try:
        result = pipeline_por_ruta(csv_path)

        out_path = OUTPUT_DIR / f"{request_id}.json"
        out_path.write_text(
            json.dumps(result["resultado_json"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"✅ Procesamiento terminado: {out_path}")

    except Exception as e:
        print(f"❌ Error en background ({request_id}): {e}")


# =========================
# HEALTH
# =========================
@app.get("/health")
def health():
    return {"ok": True}


# =========================
# PROCESS (ACK INMEDIATO)
# =========================
@app.post("/process")
async def process_csv(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
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

    # 🚀 SE MANDA A BACKGROUND
    background_tasks.add_task(
        process_csv_background,
        str(csv_path),
        request_id
    )

    # ✅ RESPUESTA INMEDIATA
    return {
        "ok": True,
        "status": "recibido",
        "request_id": request_id
    }
