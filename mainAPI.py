import uuid
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from google.cloud import storage
from google.auth import default
from google.auth.transport.requests import Request

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Observaciones Talleres API", version="1.0.0")

# =========================
# CONFIGURACIÓN
# =========================
PROJECT_ID = "sb-iapatrimoniales-dev"
REGION = "europe-west1"
JOB_NAME = "ia-mv-modelo-observacion-talleres-paralelizado"
BUCKET_NAME = "bucket-aux-ia-modelo-seguimiento-talleres"


# =========================
# AUTENTICACIÓN
# =========================
def get_access_token():
    """Obtiene el token de identidad para invocar Cloud Run."""
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())
    return credentials.token


# =========================
# GCS UPLOAD
# =========================
def upload_csv_to_gcs(file_bytes: bytes, request_id: str) -> str:
    """Sube el CSV a GCS y devuelve la ruta relativa."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # Ruta: inputs/ID.csv
    blob_path = f"inputs/{request_id}.csv"

    blob = bucket.blob(blob_path)
    blob.upload_from_string(file_bytes, content_type="text/csv")

    print(f"📤 Archivo subido: gs://{BUCKET_NAME}/{blob_path}")
    return blob_path


# =========================
# LANZADOR DEL JOB (LIMPIO)
# =========================
def launch_cloud_run_job(args: dict, task_count: int = 10):
    """
    Lanza el Job solicitando 10 tareas (máquinas) en paralelo.
    NOTA: La CPU, RAM y Reintentos deben estar configurados
    en la definición del Job en la consola de GCP.
    """
    token = get_access_token()

    url = (
        f"https://{REGION}-run.googleapis.com/apis/run.googleapis.com/v1/"
        f"namespaces/{PROJECT_ID}/jobs/{JOB_NAME}:run"
    )

    print(f"🚀 Solicitando ejecución con {task_count} tareas paralelas...")

    # Payload simplificado (Sin resources ni maxRetries para evitar error 400)
    payload = {
        "overrides": {
            "taskCount": task_count,
            "containerOverrides": [
                {
                    # Pasamos los argumentos: job_main.py blob=... request_id=...
                    "args": ["job_main.py"] + [f"{k}={v}" for k, v in args.items()]
                }
            ]
        }
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        r = requests.post(url, headers=headers, json=payload)

        if not r.ok:
            print("❌ Error de Cloud Run API:")
            print(r.text)
            raise RuntimeError(f"Fallo al lanzar job: {r.status_code} - {r.text}")

        response_data = r.json()
        execution_name = response_data.get("metadata", {}).get("name", "Desconocido")
        print(f"✅ Job lanzado correctamente. Ejecución: {execution_name}")
        return execution_name

    except Exception as e:
        print(f"💥 Excepción al conectar con GCP: {e}")
        raise e


# =========================
# ENDPOINTS
# =========================
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/process")
async def process_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "El archivo debe ser un CSV.")

    request_id = uuid.uuid4().hex
    print(f"🆔 Nuevo Request recibido: {request_id}")

    try:
        content = await file.read()

        # 1. Subir a Storage
        blob_relative_path = upload_csv_to_gcs(content, request_id)

        # 2. Lanzar Job (10 Tareas)
        launch_cloud_run_job({
            "blob": blob_relative_path,
            "request_id": request_id
        }, task_count=10)

        return {
            "ok": True,
            "status": "Job lanzado en paralelo (10 workers)",
            "request_id": request_id
        }

    except Exception as e:
        raise HTTPException(500, detail=str(e))