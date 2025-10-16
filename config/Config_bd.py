import psycopg2
from psycopg2 import OperationalError

# --- CONFIGURACIÓN DE LA CONEXIÓN ---
DB_CONFIG = {
    "host": "54.205.173.107",
    "port": 8081,
    "database": "Piloto_obsercaciones__talleres",
    "user": "postgres",
    "password": "1234"
}


def create_connection():
    """
    Crea y devuelve una conexión a la base de datos PostgreSQL.
    Si falla, lanza una excepción detallada.
    """
    try:
        connection = psycopg2.connect(**DB_CONFIG)
        print("✅ Conexión exitosa a la base de datos PostgreSQL.")
        return connection
    except OperationalError as e:
        print("❌ Error al conectar a la base de datos:", e)
        return None


# Si quieres probar la conexión directamente ejecutando este archivo
if __name__ == "__main__":
    conn = create_connection()
    if conn:
        conn.close()
        print("🔌 Conexión cerrada correctamente.")
