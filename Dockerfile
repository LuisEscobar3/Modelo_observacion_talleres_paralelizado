# Imagen base
FROM python:3.11-slim

# Evita .pyc y asegura logs inmediatos
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cloud Run suele usar PORT; dejamos default 8080 para local
ENV PORT=8080

WORKDIR /app

# Dependencias
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copia el código
COPY . /app

# (Opcional) Documentativo. Cloud Run igual enruta por $PORT
EXPOSE 8080

# IMPORTANTE: usamos sh -c para que $PORT se expanda
CMD ["sh", "-c", "python -m uvicorn mainAPI:app --host 0.0.0.0 --port $PORT"]
