import os
import uuid
import tempfile
import dotenv
from functools import lru_cache

from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_core.globals import set_debug

from google.auth import default
from google.auth.transport.requests import Request
import requests

from services.llm_manager import load_llms


# =========================
# FASTAPI APP (SERVICE)
# =========================
app = FastAPI(title="Observaciones Talleres API", version="1.0.0")


# =========================
# LLM (CACHEADO - IGUAL A TU CÓDIGO)
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
# CLOUD RUN JOB CONFIG
# =========================
PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
REGION = os.environ.get("REGION", "europe-west1")
JOB_NAME = "ia-mv-modelo-observacion-talleres-paralelizado"


def get_access_token():
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())
    return credentials.token


def launch_cloud_run_job(args: dict):
    token = get_access_token()

    url = (
        f"https://{REGION}-run.googleapis.com/apis/run.googleapis.com/v1/"
        f"namespaces/{PROJECT_ID}/jobs/{JOB_NAME}:run"
    )

    payload = {
        "overrides": {
            "containerOverrides": [
                {
                    "args": [f"{k}={v}" for k, v in args.items()]
                }
            ]
        }
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    r = requests.post(url, headers=headers, json=payload)
    if not r.ok:
        raise RuntimeError(r.text)


# =========================
# HEALTH
# =========================
@app.get("/health")
def health():
    return {"ok": True}


# =========================
# PROCESS CSV (EL QUE CONSUMES)
# =========================
@app.post("/process")
async def process_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "El archivo debe ser .csv")

    request_id = uuid.uuid4().hex

    # Guardado temporal SOLO para pasarlo al Job
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(await file.read())
        csv_path = tmp.name

    # 🔥 AQUÍ SE ACTIVA EL JOB
    launch_cloud_run_job({
        "csv_path": csv_path,
        "request_id": request_id
    })

    return {
        "ok": True,
        "status": "job_started",
        "request_id": request_id
    }
