from typing import List, Dict, Any
from datetime import datetime
from psycopg2.extras import execute_values
from psycopg2 import sql

from config.Config_bd import create_connection

# -----------------------------
# Configuración de la tabla
# -----------------------------
TABLE_NAME = "observaciones_taller"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id BIGSERIAL PRIMARY KEY,
    nit_taller           VARCHAR(32)   NOT NULL,
    nombre_taller        TEXT          NOT NULL,
    numero_aviso         BIGINT        NOT NULL,
    numero_siniestro     BIGINT        NOT NULL,
    placa                VARCHAR(16)   NOT NULL,
    fecha_observacion    TIMESTAMP     NOT NULL,
    usuario              TEXT          NOT NULL,
    rol_analista         TEXT,
    observacion          TEXT,
    clasificacion        TEXT,
    explicacion          TEXT,
    confianza            NUMERIC(5,3),

    -- Evita duplicados típicos (ajústalo si tu negocio requiere otra clave)
    CONSTRAINT uq_obs UNIQUE (numero_aviso, numero_siniestro, fecha_observacion, usuario)
);
"""

# -----------------------------
# Utilidades
# -----------------------------
def _parse_fecha(valor: str) -> datetime:
    """
    Normaliza y parsea fechas tipo '2025/01/13  2:18:43 PM ' (doble espacio).
    Acepta 'YYYY/MM/DD hh:mm:ss AM/PM'
    """
    if not valor:
        raise ValueError("fecha_observacion vacía")
    # Normaliza espacios duplicados y recorta
    clean = " ".join(valor.split())
    # Formato esperado: 2025/01/13 2:18:43 PM
    return datetime.strptime(clean, "%Y/%m/%d %I:%M:%S %p")


def _normalize_record(r: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convierte el dict de entrada (con claves con espacios) a snake_case
    y tipifica campos.
    """
    return {
        "nit_taller":          str(r.get("NIT TALLER", "")).strip(),
        "nombre_taller":       str(r.get("NOMBRE TALLER", "")).strip(),
        "numero_aviso":        int(r.get("numero_aviso")),
        "numero_siniestro":    int(r.get("numero_siniestro")),
        "placa":               str(r.get("placa", "")).strip().upper(),
        "fecha_observacion":   _parse_fecha(str(r.get("fecha_observacion", ""))),
        "usuario":             str(r.get("usuario", "")).strip(),
        "rol_analista":        (str(r.get("rol_analista")).strip()
                                if r.get("rol_analista") is not None else None),
        "observacion":         (str(r.get("observacion")).strip()
                                if r.get("observacion") is not None else None),
        "clasificacion":       (str(r.get("clasificacion")).strip()
                                if r.get("clasificacion") is not None else None),
        "explicacion":         (str(r.get("explicacion")).strip()
                                if r.get("explicacion") is not None else None),
        "confianza":           float(r.get("confianza")) if r.get("confianza") is not None else None,
    }


# -----------------------------
# API pública
# -----------------------------
def ensure_table():
    """
    Crea la tabla si no existe.
    """
    conn = create_connection()
    if not conn:
        raise RuntimeError("No se pudo obtener conexión a la BD.")

    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
        conn.commit()
    finally:
        conn.close()


def insert_observaciones(registros: List[Dict[str, Any]]) -> int:
    """
    Inserta en lote los registros. Si ya existe (según UNIQUE),
    hace UPSERT (actualiza campos no clave).
    Retorna la cantidad de filas insertadas/actualizadas.
    """
    if not registros:
        return 0

    # Normaliza y tipifica
    parsed = [_normalize_record(r) for r in registros]

    columns = [
        "nit_taller",
        "nombre_taller",
        "numero_aviso",
        "numero_siniestro",
        "placa",
        "fecha_observacion",
        "usuario",
        "rol_analista",
        "observacion",
        "clasificacion",
        "explicacion",
        "confianza",
    ]

    values = [
        (
            p["nit_taller"],
            p["nombre_taller"],
            p["numero_aviso"],
            p["numero_siniestro"],
            p["placa"],
            p["fecha_observacion"],
            p["usuario"],
            p["rol_analista"],
            p["observacion"],
            p["clasificacion"],
            p["explicacion"],
            p["confianza"],
        )
        for p in parsed
    ]

    # UPSERT: si ya existe una fila con esa combinación, actualiza campos
    upsert_sql = sql.SQL("""
        INSERT INTO {table} ({fields})
        VALUES %s
        ON CONFLICT (numero_aviso, numero_siniestro, fecha_observacion, usuario)
        DO UPDATE SET
            nit_taller = EXCLUDED.nit_taller,
            nombre_taller = EXCLUDED.nombre_taller,
            placa = EXCLUDED.placa,
            rol_analista = EXCLUDED.rol_analista,
            observacion = EXCLUDED.observacion,
            clasificacion = EXCLUDED.clasificacion,
            explicacion = EXCLUDED.explicacion,
            confianza = EXCLUDED.confianza;
    """).format(
        table=sql.Identifier(TABLE_NAME),
        fields=sql.SQL(", ").join(map(sql.Identifier, columns))
    )

    conn = create_connection()
    if not conn:
        raise RuntimeError("No se pudo obtener conexión a la BD.")

    rows_affected = 0
    try:
        with conn.cursor() as cur:
            execute_values(cur, upsert_sql.as_string(conn), values, page_size=500)
            rows_affected = cur.rowcount if cur.rowcount is not None else len(values)
        conn.commit()
    finally:
        conn.close()

    return rows_affected


# -----------------------------
# Prueba rápida (opcional)
# -----------------------------
if __name__ == "__main__":
    # Ejemplo del payload que enviaste:
    registros = [
        {
            "NIT TALLER": "901044640-1",
            "NOMBRE TALLER": "JORGE CORTES Y CIA S.A.S - MAZDA",
            "numero_aviso": 219861,
            "numero_siniestro": 10000106468,
            "placa": "DWM896",
            "fecha_observacion": "2025/01/13  2:18:43 PM ",
            "usuario": "DIEGO BAREÑO",
            "rol_analista": "ANALISTA ASEGURADORA",
            "observacion": "CLIENTE ... PROGRAMAR LA VALORACIÓN EN LOS PRÓXIMOS DÍAS",
            "clasificacion": "comunicacion_cliente",
            "explicacion": "El cliente informa su situación y confirma futuro contacto para programar la cita.",
            "confianza": 0.9
        },
        {
            "NIT TALLER": "901044640-1",
            "NOMBRE TALLER": "JORGE CORTES Y CIA S.A.S - MAZDA",
            "numero_aviso": 219861,
            "numero_siniestro": 10000106468,
            "placa": "DWM896",
            "fecha_observacion": "2025/01/23  10:09:50 AM ",
            "usuario": "DIEGO BAREÑO",
            "rol_analista": "ANALISTA ASEGURADORA",
            "observacion": "EN ESPERA DEL AGENDAMIENTO POR PARTE DEL CLIENTE PARA EL PROCESO CORRESPONDIENTE",
            "clasificacion": "sin_cambio",
            "explicacion": "El vehículo está a la espera de la acción del cliente para agendar.",
            "confianza": 0.9
        }
    ]

    ensure_table()
    n = insert_observaciones(registros)
    print(f"✅ Filas afectadas (insert/ups): {n}")
