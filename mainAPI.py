import uuid
import requests

from fastapi import FastAPI, UploadFile, File, HTTPException
from google.cloud import storage
from google.auth import default
from google.auth.transport.requests import Request

# =========================
# FASTAPI APP (SERVICE)
# =========================
app = FastAPI(title="Observaciones Talleres API", version="1.0.0")

# =========================
# CONFIG
# =========================
PROJECT_ID = "sb-iapatrimoniales-dev"
REGION = "europe-west1"
JOB_NAME = "ia-mv-modelo-observacion-talleres-paralelizado"
BUCKET_NAME = "bucket-aux-ia-modelo-seguimiento-talleres"


# =========================
# AUTH
# =========================
def get_access_token():
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())
    return credentials.token


# =========================
# GCS
# =========================
def upload_csv_to_gcs(file_bytes: bytes, request_id: str) -> str:
    """
    Sube el archivo a GCS y devuelve la ruta relativa (blob path)
    necesaria para la librería de storage client.
    """
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # Ruta relativa dentro del bucket
    blob_path = f"inputs/{request_id}.csv"

    blob = bucket.blob(blob_path)
    blob.upload_from_string(file_bytes, content_type="text/csv")

    print(f"📤 CSV subido a GCS (URI): gs://{BUCKET_NAME}/{blob_path}")

    # RETORNAMOS LA RUTA RELATIVA (sin gs://)
    return blob_path


# =========================
# CLOUD RUN JOB
# =========================
def launch_cloud_run_job(args: dict):
    token = get_access_token()

    url = (
        f"https://{REGION}-run.googleapis.com/apis/run.googleapis.com/v1/"
        f"namespaces/{PROJECT_ID}/jobs/{JOB_NAME}:run"
    )

    print("🚀 Lanzando Job con argumentos:")
    for k, v in args.items():
        print(f"   - {k} = {v}")

    payload = {
        "overrides": {
            "containerOverrides": [
                {
                    # Convertimos el diccionario args en una lista de argumentos "--key=value" o "key=value"
                    # Asumiendo que job_main.py parsea con formato "key=value"
                    "args": ["job_main.py"] + [f"{k}={v}" for k, v in args.items()]
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
        print("❌ Error al lanzar Job:")
        print(r.text)
        raise RuntimeError(r.text)

    print("✅ Job lanzado correctamente")


# =========================
# HEALTH
# =========================
@app.get("/health")
def health():
    return {"ok": True}


# =========================
# PROCESS CSV
# =========================
@app.post("/process")
async def process_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "El archivo debe ser .csv")

    request_id = uuid.uuid4().hex
    content = await file.read()

    # 1️⃣ Subir CSV a GCS (Obtenemos ruta relativa)
    blob_relative_path = upload_csv_to_gcs(content, request_id)

    # 2️⃣ Lanzar Job
    # IMPORTANTE: Usamos la clave "blob" para que coincida con job_main.py
    launch_cloud_run_job({
        "blob": blob_relative_path,
        "request_id": request_id
    })

    return {
        "ok": True,
        "status": "job_started",
        "request_id": request_id
    }