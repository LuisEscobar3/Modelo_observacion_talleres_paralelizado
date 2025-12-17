# Usa una imagen base ligera de Python
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia los requerimientos e instálalos
COPY requirements.txt .
# Instalamos las dependencias.
# Nota: Si tienes dependencias que requieren compilación (como psycopg2 no-binary),
# podrías necesitar instalar gcc y otras librerías antes de pip install.
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código
COPY . .

# Expone el puerto 8080 (Puerto estándar de Cloud Run)
EXPOSE 8080

# Ejecuta la aplicación usando Uvicorn directamente para producción.
# Usamos main_firebase:app ya que es donde está definida tu instancia de FastAPI.
CMD ["uvicorn", "mainAPI:app", "--host", "0.0.0.0", "--port", "8080"]
