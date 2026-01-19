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
# CMD por defecto (SERVICE)
# =========================
CMD ["uvicorn", "mainAPI:app", "--host", "0.0.0.0", "--port", "8080"]
