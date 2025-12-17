import os
import re
import csv
import psycopg2
from psycopg2 import OperationalError, sql
from datetime import datetime

# =========================
# CONEXIÓN A POSTGRES
# =========================
DB_CONFIG = {
    "host": "54.205.173.107",
    "port": 8081,  # cambia a 5432 si tu servidor usa el puerto estándar
    "database": "Seguimiento_talleres",
    "user": "postgres",
    "password": "1234",
}

def create_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Conexión exitosa a PostgreSQL.")
        return conn
    except OperationalError as e:
        print("❌ Error al conectar a la base de datos:", e)
        return None

# =========================
# UTILIDADES
# =========================
def sanitize_identifier(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^\w]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "col"
    if name[0].isdigit():
        name = f"col_{name}"
    return name

def ensure_unique(names):
    seen = {}
    out = []
    for n in names:
        base = n
        i = 1
        while n in seen:
            i += 1
            n = f"{base}_{i}"
        seen[n] = True
        out.append(n)
    return out

def _is_int(s: str) -> bool:
    try:
        if s.strip() == "":
            return False
        # Evitar formatos con coma/punto decimal
        if re.search(r"[.,]", s):
            return False
        int(s)
        return True
    except:
        return False

def _is_float(s: str) -> bool:
    try:
        if s.strip() == "":
            return False
        float(s)
        # Si es entero puro, lo tratamos como int
        return not _is_int(s)
    except:
        return False

def _is_bool(s: str) -> bool:
    return s.lower() in {"true", "false", "t", "f", "yes", "no", "y", "n", "1", "0"}

def _is_date(s: str) -> bool:
    fmts = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"]
    for f in fmts:
        try:
            datetime.strptime(s, f).date()
            return True
        except:
            pass
    return False

def _is_time(s: str) -> bool:
    fmts = ["%H:%M", "%H:%M:%S"]
    for f in fmts:
        try:
            datetime.strptime(s, f).time()
            return True
        except:
            pass
    return False

def _is_timestamp(s: str) -> bool:
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
    ]
    for f in fmts:
        try:
            datetime.strptime(s, f)
            return True
        except:
            pass
    # ISO flexible con milisegundos/Z
    try:
        s2 = s.replace("Z", "")
        if "." in s2:
            s2 = s2.split(".")[0]
        datetime.fromisoformat(s2.replace("T", " "))
        return True
    except:
        return False

def _fits_int32(v: int) -> bool:
    return -2147483648 <= v <= 2147483647

def _fits_int64(v: int) -> bool:
    return -9223372036854775808 <= v <= 9223372036854775807

def guess_pg_type(values):
    """
    Devuelve el tipo PostgreSQL más específico considerando rangos:
      - INTEGER si cabe en int32
      - BIGINT si no cabe en int32 pero sí en int64
      - DOUBLE PRECISION para floats
      - BOOLEAN, DATE, TIME, TIMESTAMP
      - TEXT por defecto
    'values' debe ser una lista de strings no vacíos.
    """
    if all(_is_int(v) for v in values):
        ints = [int(v) for v in values]
        min_v, max_v = min(ints), max(ints)
        if _fits_int32(min_v) and _fits_int32(max_v):
            return "INTEGER"
        if _fits_int64(min_v) and _fits_int64(max_v):
            return "BIGINT"
        # Si excede BIGINT, a TEXT
        return "TEXT"

    if all(_is_float(v) or _is_int(v) for v in values):
        return "DOUBLE PRECISION"

    if all(_is_bool(v) for v in values):
        return "BOOLEAN"
    if all(_is_date(v) for v in values):
        return "DATE"
    if all(_is_time(v) for v in values):
        return "TIME"
    if all(_is_timestamp(v) for v in values):
        return "TIMESTAMP"
    return "TEXT"

def infer_schema_from_csv(csv_path, delimiter=",", encoding="utf-8", sample_rows=5000):
    """
    Lee encabezados y una muestra (hasta sample_rows) y devuelve:
      (col_names_saneados, col_types)
    """
    with open(csv_path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        original_cols = reader.fieldnames
        if not original_cols:
            raise ValueError("El CSV no tiene encabezados o no se pudieron leer.")
        sane_cols = [sanitize_identifier(c) for c in original_cols]
        sane_cols = ensure_unique(sane_cols)

        buckets = {c: [] for c in sane_cols}
        count = 0
        for row in reader:
            for orig, sane in zip(original_cols, sane_cols):
                val = (row.get(orig) or "").strip()
                if val != "":
                    if len(buckets[sane]) < sample_rows:
                        buckets[sane].append(val)
            count += 1
            if count >= sample_rows:
                # seguimos leyendo luego para COPY, pero no para inferencia
                pass

    types = []
    for c in sane_cols:
        non_empty = buckets[c]
        if not non_empty:
            types.append("TEXT")
        else:
            types.append(guess_pg_type(non_empty))
    return sane_cols, types

def table_exists(conn, table_name: str, schema: str = "public") -> bool:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
        """, (schema, table_name))
        return cur.fetchone() is not None

def drop_table(conn, table_name: str, schema: str = "public"):
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE;")
            .format(sql.Identifier(schema), sql.Identifier(table_name))
        )
    conn.commit()

def create_table(conn, table_name: str, col_names, col_types, schema: str = "public"):
    cols = [
        sql.SQL("{} {}").format(sql.Identifier(n), sql.SQL(t))
        for n, t in zip(col_names, col_types)
    ]
    stmt = sql.SQL("CREATE TABLE {}.{} ({});").format(
        sql.Identifier(schema),
        sql.Identifier(table_name),
        sql.SQL(", ").join(cols)
    )
    with conn.cursor() as cur:
        cur.execute(stmt)
    conn.commit()
    print(f"🆕 Tabla creada: {schema}.{table_name}")

def copy_csv(conn, csv_path, table_name: str, delimiter=",", encoding="utf-8", schema: str = "public"):
    """
    Inserta datos vía COPY usando el encabezado real del CSV -> columnas saneadas.
    """
    with open(csv_path, "r", encoding=encoding, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader)
        sane_header = ensure_unique([sanitize_identifier(h) for h in header])

    columns_sql = sql.SQL(", ").join(sql.Identifier(c) for c in sane_header)

    copy_sql = sql.SQL("""
        COPY {}.{} ({})
        FROM STDIN WITH (
            FORMAT CSV,
            HEADER TRUE,
            DELIMITER {},
            QUOTE {},
            ESCAPE {}
        );
    """).format(
        sql.Identifier(schema),
        sql.Identifier(table_name),
        columns_sql,
        sql.Literal(delimiter),
        sql.Literal('"'),
        sql.Literal('"'),
    )

    with conn.cursor() as cur, open(csv_path, "r", encoding=encoding, newline="") as f_data:
        cur.copy_expert(copy_sql.as_string(conn), f_data, size=1024 * 1024)
    conn.commit()
    print(f"📥 Datos insertados en {schema}.{table_name} mediante COPY.")

# =========================
# FLUJO PRINCIPAL
# =========================
def export_csv_to_postgres(
    csv_path: str,
    table_name: str = None,
    delimiter: str = ",",
    encoding: str = "utf-8",
    schema: str = "public",
    if_exists: str = "append",  # 'append' | 'replace' | 'fail'
):
    """
    Exporta un CSV a PostgreSQL:
      - Infiera tipos (usa BIGINT si el entero no cabe en int32).
      - Crea la tabla si no existe.
      - Inserta con COPY.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No existe el archivo CSV: {csv_path}")

    # Derivar nombre de tabla si no se especifica
    if not table_name:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        table_name = sanitize_identifier(base)

    conn = create_connection()
    if not conn:
        raise RuntimeError("No se pudo establecer la conexión a la base de datos.")

    try:
        print("🔎 Infiriendo esquema desde el CSV (muestra)…")
        col_names, col_types = infer_schema_from_csv(csv_path, delimiter=delimiter, encoding=encoding)
        print("   - Columnas:", col_names)
        print("   - Tipos   :", col_types)

        exists = table_exists(conn, table_name, schema)
        if exists:
            if if_exists == "fail":
                raise RuntimeError(f"La tabla {schema}.{table_name} ya existe y if_exists='fail'.")
            elif if_exists == "replace":
                print(f"♻️ Reemplazando tabla {schema}.{table_name}…")
                drop_table(conn, table_name, schema)
                create_table(conn, table_name, col_names, col_types, schema)
            else:
                print(f"➕ Insertando en tabla existente {schema}.{table_name} (append).")
        else:
            print(f"🛠️ Creando tabla {schema}.{table_name}…")
            create_table(conn, table_name, col_names, col_types, schema)

        copy_csv(conn, csv_path, table_name, delimiter=delimiter, encoding=encoding, schema=schema)
        print("✅ Proceso completado.")
    finally:
        try:
            conn.close()
            print("🔌 Conexión cerrada correctamente.")
        except:
            pass

# =========================
# EJECUCIÓN DIRECTA
# =========================
if __name__ == "__main__":
    # 👉 Ajusta esta ruta a tu CSV:
    CSV_PATH = r"C:\Users\1032497498\PycharmProjects\Modelo_observacion_talleres_paralelizado\services\Clasificacion_onservacion.csv"
    # 👉 Si quieres un nombre específico para la tabla, descomenta y ajusta:
    # TABLE_NAME = "clasificacion_observacion"
    # export_csv_to_postgres(CSV_PATH, table_name=TABLE_NAME, delimiter=",", encoding="utf-8", if_exists="append")

    # Sin nombre de tabla explícito: usa el nombre del archivo como base
    export_csv_to_postgres(CSV_PATH, delimiter=",", encoding="utf-8", if_exists="append")
