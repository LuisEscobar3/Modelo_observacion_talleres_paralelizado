# =========================
# Imagen base
# =========================
FROM python:3.10-slim

# Evita buffering y .pyc
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# =========================
# Directorio de trabajo
# =========================
WORKDIR /app

# =========================
# Dependencias
# =========================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =========================
# Código
# =========================
COPY . .

# =========================
# Puerto Cloud Run
# =========================
EXPOSE 8080

# =========================
# ENTRYPOINT NEUTRO
# =========================
# Permite:
# - Service → uvicorn
# - Job → python job_main.py ...
ENTRYPOINT []

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# =========================
# CMD por defecto (SERVICE)
# =========================
CMD ["-m", "uvicorn", "mainAPI:app", "--host", "0.0.0.0", "--port", "8080"]
