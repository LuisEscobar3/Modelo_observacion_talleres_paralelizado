import uuid
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from google.cloud import storage
from google.auth import default
from google.auth.transport.requests import Request

app = FastAPI(title="Observaciones Talleres API", version="1.0.0")

# CONFIG
PROJECT_ID = "sb-iapatrimoniales-dev"
REGION = "europe-west1"
JOB_NAME = "ia-mv-modelo-observacion-talleres-paralelizado"
BUCKET_NAME = "bucket-aux-ia-modelo-seguimiento-talleres"


# AUTH
def get_access_token():
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())
    return credentials.token


# UPLOAD
def upload_csv_to_gcs(file_bytes: bytes, request_id: str) -> str:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob_path = f"inputs/{request_id}.csv"
    blob = bucket.blob(blob_path)
    blob.upload_from_string(file_bytes, content_type="text/csv")
    print(f"📤 Uploaded: gs://{BUCKET_NAME}/{blob_path}")
    return blob_path


# =========================
# LANZAR JOB (CONFIGURACIÓN OPTIMIZADA)
# =========================
def launch_cloud_run_job(args: dict, task_count: int = 10):  # <-- 10 MAQUINAS
    token = get_access_token()
    url = f"https://{REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/{PROJECT_ID}/jobs/{JOB_NAME}:run"

    print(f"🚀 Lanzando Cluster: {task_count} Máquinas x 4 CPUs...")

    payload = {
        "overrides": {
            "taskCount": task_count,
            "containerOverrides": [
                {
                    "args": ["job_main.py"] + [f"{k}={v}" for k, v in args.items()],

                    # --- LIMITES DE RECURSOS ---
                    # 4 CPUs y 4GB RAM: El equilibrio perfecto
                    "resources": {
                        "limits": {
                            "cpu": "4000m",
                            "memory": "4Gi"
                        }
                    }
                }
            ],
            # --- CERO REINTENTOS ---
            # Si una tarea falla, NO se reinicia. Evita duplicados en BD.
            "maxRetries": 0
        }
    }

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    r = requests.post(url, headers=headers, json=payload)
    if not r.ok:
        print(f"❌ Error lanzando Job: {r.text}")
        raise RuntimeError(r.text)

    print("✅ Job lanzado correctamente")


# ENDPOINTS
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/process")
async def process_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Archivo debe ser .csv")

    request_id = uuid.uuid4().hex
    content = await file.read()

    # 1. Subir CSV
    blob_relative_path = upload_csv_to_gcs(content, request_id)

    # 2. Lanzar Job Paralelizado (10 Tareas)
    # Esto divide los 75k registros en bloques de 7,500
    launch_cloud_run_job({
        "blob": blob_relative_path,
        "request_id": request_id
    }, task_count=10)

    return {
        "ok": True,
        "request_id": request_id,
        "strategy": "parallel_10_nodes_4cpu"
    }