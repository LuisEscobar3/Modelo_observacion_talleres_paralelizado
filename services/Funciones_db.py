# app/services/Funciones_db.py

import psycopg2
from psycopg2 import OperationalError, IntegrityError
from psycopg2.extras import Json
from typing import Optional, Dict, Any

from app.config.conexionbd import create_connection, close_connection


# ============================================================
#                       TABLA: CASOS
# ============================================================

def create_casos_table() -> None:
    """Crea/verifica la tabla 'casos'."""
    conn = create_connection()
    if conn is None:
        return
    cursor = conn.cursor()
    try:
        ddl = """
        CREATE TABLE IF NOT EXISTS casos (
            id BIGSERIAL PRIMARY KEY,
            numero_caso VARCHAR(255) UNIQUE NOT NULL,
            agente_usuario VARCHAR(255) NOT NULL,
            fecha_creacion TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(ddl)
        conn.commit()
        print("✅ Tabla 'casos' verificada/creada exitosamente.")
    except OperationalError as e:
        print(f"❌ Ocurrió un error al crear la tabla 'casos': '{e}'")
        conn.rollback()
    finally:
        cursor.close()
        close_connection(conn)


def insert_caso(numero_caso: str, agente_usuario: str) -> Optional[int]:
    """
    Inserta un caso y retorna su id.
    Si ya existe por UNIQUE(numero_caso), retorna el id existente.
    """
    conn = create_connection()
    if conn is None:
        return None

    cursor = conn.cursor()
    caso_id: Optional[int] = None

    try:
        sql = """
        INSERT INTO casos (numero_caso, agente_usuario)
        VALUES (%s, %s)
        RETURNING id;
        """
        cursor.execute(sql, (numero_caso, agente_usuario))
        caso_id = cursor.fetchone()[0]
        conn.commit()
        print(f"✅ Caso '{numero_caso}' insertado con ID: {caso_id}")

    except IntegrityError as e:
        conn.rollback()
        try:
            cursor.execute("SELECT id FROM casos WHERE numero_caso = %s;", (numero_caso,))
            row = cursor.fetchone()
            if row:
                caso_id = row[0]
                print(f"ℹ️ Caso '{numero_caso}' ya existía. Usando ID existente: {caso_id}")
            else:
                print(f"⚠️ Error de integridad y no se encontró el caso '{numero_caso}'. Detalle: '{e}'")
        except OperationalError as e2:
            print(f"❌ Error al recuperar ID existente de 'casos': '{e2}'")
            conn.rollback()

    except OperationalError as e:
        print(f"❌ Ocurrió un error en la base de datos (insert_caso): '{e}'")
        conn.rollback()

    finally:
        cursor.close()
        close_connection(conn)

    return caso_id


# ============================================================
#                 TABLA: TRANSCRIPCIONES (1:N)
# ============================================================

def create_transcripciones_table() -> None:
    """
    Crea/verifica la tabla 'transcripciones' (modelo híbrido con JSONB).
    Relación 1:N con 'casos' por caso_id.
    """
    conn = create_connection()
    if conn is None:
        return

    cursor = conn.cursor()
    try:
        ddl = """
        CREATE TABLE IF NOT EXISTS transcripciones (
          id                        BIGSERIAL PRIMARY KEY,
          caso_id                   BIGINT NOT NULL REFERENCES casos(id) ON DELETE CASCADE,
          segmento_orden            INTEGER NOT NULL,
          transcripcion_text        TEXT NOT NULL,
          resumen                   TEXT NOT NULL,
          cliente                   JSONB NOT NULL,
          vehiculo                  JSONB NOT NULL,
          poliza                    JSONB NOT NULL,
          relato_siniestro          TEXT NOT NULL,
          analisis_siniestro        JSONB NOT NULL,
          analisis_negociacion      JSONB NOT NULL,
          agente                    JSONB NOT NULL,
          analisis_conversacional   JSONB NOT NULL,
          seguimiento_guion         JSONB NOT NULL,
          analisis_inconsistencias  JSONB NOT NULL,
          created_at                TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
          UNIQUE (caso_id, segmento_orden)
        );

        CREATE INDEX IF NOT EXISTS idx_transcripciones_caso
          ON transcripciones(caso_id);

        CREATE INDEX IF NOT EXISTS idx_transcripciones_cliente_gin
          ON transcripciones USING GIN (cliente);

        CREATE INDEX IF NOT EXISTS idx_transcripciones_seg_guion_gin
          ON transcripciones USING GIN (seguimiento_guion);
        """
        cursor.execute(ddl)
        conn.commit()
        print("✅ Tabla 'transcripciones' verificada/creada exitosamente.")
    except OperationalError as e:
        print(f"❌ Ocurrió un error al crear la tabla 'transcripciones': '{e}'")
        conn.rollback()
    finally:
        cursor.close()
        close_connection(conn)


def insert_transcripcion(caso_id: int, segmento_orden: int, data: Dict[str, Any]) -> Optional[int]:
    """
    Inserta/actualiza una transcripción validada (dict de Transcripcion.model_dump()).
    Upsert por (caso_id, segmento_orden). Retorna el id de la fila.
    """
    conn = create_connection()
    if conn is None:
        return None

    cursor = conn.cursor()
    new_id: Optional[int] = None

    try:
        sql = """
        INSERT INTO transcripciones(
          caso_id, segmento_orden,
          transcripcion_text, resumen,
          cliente, vehiculo, poliza, relato_siniestro,
          analisis_siniestro, analisis_negociacion, agente,
          analisis_conversacional, seguimiento_guion, analisis_inconsistencias
        )
        VALUES (
          %(caso_id)s, %(segmento_orden)s,
          %(transcripcion_text)s, %(resumen)s,
          %(cliente)s, %(vehiculo)s, %(poliza)s, %(relato_siniestro)s,
          %(analisis_siniestro)s, %(analisis_negociacion)s, %(agente)s,
          %(analisis_conversacional)s, %(seguimiento_guion)s, %(analisis_inconsistencias)s
        )
        ON CONFLICT (caso_id, segmento_orden) DO UPDATE SET
          transcripcion_text       = EXCLUDED.transcripcion_text,
          resumen                  = EXCLUDED.resumen,
          cliente                  = EXCLUDED.cliente,
          vehiculo                 = EXCLUDED.vehiculo,
          poliza                   = EXCLUDED.poliza,
          relato_siniestro         = EXCLUDED.relato_siniestro,
          analisis_siniestro       = EXCLUDED.analisis_siniestro,
          analisis_negociacion     = EXCLUDED.analisis_negociacion,
          agente                   = EXCLUDED.agente,
          analisis_conversacional  = EXCLUDED.analisis_conversacional,
          seguimiento_guion        = EXCLUDED.seguimiento_guion,
          analisis_inconsistencias = EXCLUDED.analisis_inconsistencias
        RETURNING id;
        """
        params = {
            "caso_id": caso_id,
            "segmento_orden": segmento_orden,
            "transcripcion_text": data["transcripcion_text"],
            "resumen": data["resumen"],
            "cliente": Json(data["cliente"]),
            "vehiculo": Json(data["vehiculo"]),
            "poliza": Json(data["poliza"]),
            "relato_siniestro": data["relato_siniestro"],
            "analisis_siniestro": Json(data["analisis_siniestro"]),
            "analisis_negociacion": Json(data["analisis_negociacion"]),
            "agente": Json(data["agente"]),
            "analisis_conversacional": Json(data["analisis_conversacional"]),
            "seguimiento_guion": Json(data["seguimiento_guion"]),
            "analisis_inconsistencias": Json(data["analisis_inconsistencias"]),
        }
        cursor.execute(sql, params)
        new_id = cursor.fetchone()[0]
        conn.commit()
        print(f"💾 Transcripción guardada/actualizada (caso_id={caso_id}, segmento={segmento_orden}) id={new_id}")

    except (IntegrityError, OperationalError, KeyError) as e:
        print(f"❌ Error al insertar/actualizar transcripción (caso_id={caso_id}, segmento={segmento_orden}): '{e}'")
        conn.rollback()
        new_id = None
    finally:
        cursor.close()
        close_connection(conn)

    return new_id


# ============================================================
#                 TABLA: CONSOLIDADOS (1:1)
# ============================================================

def create_consolidados_table() -> None:
    """
    Crea/verifica la tabla 'consolidados' (relación 1:1 con 'casos').
    Guarda:
      - columnas proyectadas clave
      - el JSON completo en 'payload'
      - el guion consolidado dentro de la misma tabla como JSONB
    """
    conn = create_connection()
    if conn is None:
        return

    cursor = conn.cursor()
    try:
        # Crear base (si no existe)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS consolidados (
          id         BIGSERIAL PRIMARY KEY,
          caso_id    BIGINT NOT NULL UNIQUE REFERENCES casos(id) ON DELETE CASCADE,
          -- Proyecciones clave
          resumen_negociacion_completa   TEXT,
          resultado_final_negociacion    TEXT,
          monto_acordado_final           NUMERIC(18,2),
          pasos_siguientes_generales     TEXT,
          sospecha_fraude_consolidado    TEXT,
          justificacion_fraude_consolidado TEXT,
          datos_clave_extraidos          JSONB,
          tono_general_negociacion       JSONB,
          revision_inconsistencias       JSONB,
          seguimiento_guion_consolidado  JSONB,
          numero_caso                    TEXT,
          agente_usuario                 TEXT,
          -- JSON completo por trazabilidad/flexibilidad
          payload    JSONB NOT NULL,
          created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Índices útiles (idempotentes)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_consolidados_caso ON consolidados(caso_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_consolidados_resultado ON consolidados(resultado_final_negociacion);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_consolidados_fraude ON consolidados(sospecha_fraude_consolidado);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_consolidados_payload_gin ON consolidados USING GIN (payload);")

        conn.commit()
        print("✅ Tabla 'consolidados' verificada/creada exitosamente.")
    except OperationalError as e:
        print(f"❌ Error creando 'consolidados': '{e}'")
        conn.rollback()
    finally:
        cursor.close()
        close_connection(conn)


def upsert_consolidado(caso_id: int, payload: Dict[str, Any]) -> Optional[int]:
    """
    Inserta o actualiza el consolidado (1:1 por caso) en UNA SOLA TABLA:
      - Columnas proyectadas (estado, montos, identificadores)
      - Campos flexibles como JSONB (datos_clave, tono, revisión, guion)
      - 'payload' con el JSON completo
    Retorna el id del registro.
    """
    conn = create_connection()
    if conn is None:
        return None

    cursor = conn.cursor()
    new_id: Optional[int] = None

    # Extraer campos del payload (seguros y tolerantes)
    resumen = payload.get("resumen_negociacion_completa")
    resultado = payload.get("resultado_final_negociacion")
    monto_final_raw = payload.get("monto_acordado_final")
    pasos_sig = payload.get("pasos_siguientes_generales")
    fraude = payload.get("sospecha_fraude_consolidado")
    just_fraude = payload.get("justificacion_fraude_consolidado")
    datos_clave = payload.get("datos_clave_extraidos")
    tono = payload.get("tono_general_negociacion")
    revision = payload.get("revision_inconsistencias")
    guion = payload.get("seguimiento_guion_consolidado")
    numero_caso = payload.get("numero_caso")
    agente_usuario = payload.get("agente_usuario")

    # Cast numérico amigable
    def _to_num(x):
        if x is None:
            return None
        try:
            s = str(x).replace(",", "").strip()
            return None if s == "" or s.lower() == "no especificado" else float(s)
        except Exception:
            return None

    monto_final = _to_num(monto_final_raw)

    try:
        sql = """
        INSERT INTO consolidados (
          caso_id,
          resumen_negociacion_completa,
          resultado_final_negociacion,
          monto_acordado_final,
          pasos_siguientes_generales,
          sospecha_fraude_consolidado,
          justificacion_fraude_consolidado,
          datos_clave_extraidos,
          tono_general_negociacion,
          revision_inconsistencias,
          seguimiento_guion_consolidado,
          numero_caso,
          agente_usuario,
          payload
        )
        VALUES (
          %(caso_id)s,
          %(resumen)s,
          %(resultado)s,
          %(monto_final)s,
          %(pasos_sig)s,
          %(fraude)s,
          %(just_fraude)s,
          %(datos_clave)s,
          %(tono)s,
          %(revision)s,
          %(guion)s,
          %(numero_caso)s,
          %(agente_usuario)s,
          %(payload)s
        )
        ON CONFLICT (caso_id) DO UPDATE SET
          resumen_negociacion_completa   = EXCLUDED.resumen_negociacion_completa,
          resultado_final_negociacion    = EXCLUDED.resultado_final_negociacion,
          monto_acordado_final           = EXCLUDED.monto_acordado_final,
          pasos_siguientes_generales     = EXCLUDED.pasos_siguientes_generales,
          sospecha_fraude_consolidado    = EXCLUDED.sospecha_fraude_consolidado,
          justificacion_fraude_consolidado = EXCLUDED.justificacion_fraude_consolidado,
          datos_clave_extraidos          = EXCLUDED.datos_clave_extraidos,
          tono_general_negociacion       = EXCLUDED.tono_general_negociacion,
          revision_inconsistencias       = EXCLUDED.revision_inconsistencias,
          seguimiento_guion_consolidado  = EXCLUDED.seguimiento_guion_consolidado,
          numero_caso                    = EXCLUDED.numero_caso,
          agente_usuario                 = EXCLUDED.agente_usuario,
          payload                        = EXCLUDED.payload
        RETURNING id;
        """
        params = {
            "caso_id": caso_id,
            "resumen": resumen,
            "resultado": resultado,
            "monto_final": monto_final,
            "pasos_sig": pasos_sig,
            "fraude": fraude,
            "just_fraude": just_fraude,
            "datos_clave": Json(datos_clave) if datos_clave is not None else None,
            "tono": Json(tono) if tono is not None else None,
            "revision": Json(revision) if revision is not None else None,
            "guion": Json(guion) if guion is not None else None,
            "numero_caso": numero_caso,
            "agente_usuario": agente_usuario,
            "payload": Json(payload),
        }
        cursor.execute(sql, params)
        new_id = cursor.fetchone()[0]
        conn.commit()
        print(f"💾 Consolidado guardado/actualizado (caso_id={caso_id}) id={new_id}")

    except (IntegrityError, OperationalError) as e:
        print(f"❌ Error guardando consolidado (caso_id={caso_id}): '{e}'")
        conn.rollback()
    finally:
        cursor.close()
        close_connection(conn)

    return new_id

# ============================================================
#           TABLA: TRANSCRIPCIONES_SIMPLES (1:N)
# ============================================================

def create_transcripciones_simples_table() -> None:
    """Crea la tabla 'transcripciones_simples' como una tabla independiente."""
    conn = create_connection()
    if conn is None: return
    cursor = conn.cursor()
    try:
        ddl = """
        CREATE TABLE IF NOT EXISTS transcripciones_simples (
            id BIGSERIAL PRIMARY KEY,
            caso_id VARCHAR(255) NOT NULL,
            usuario_id VARCHAR(255) NOT NULL,
            texto_transcripcion TEXT NOT NULL,
            fecha_creacion TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(ddl)
        conn.commit()
        print("✅ Tabla 'transcripciones_simples' (independiente) verificada/creada exitosamente.")
    except OperationalError as e:
        print(f"❌ Ocurrió un error al crear la tabla 'transcripciones_simples': '{e}'")
        conn.rollback()
    finally:
        cursor.close()
        close_connection(conn)


from typing import Optional
from psycopg2 import OperationalError, IntegrityError

# Asumiendo que estas funciones y clases están definidas
# en alguna parte de tu código.
# class Transcripcion: ...
# def create_connection(): ...
# def close_connection(conn): ...
# def insert_caso(numero_caso: str, agente_usuario: str) -> Optional[int]: ...

def guardar_transcripcion(data: "Transcripcion") -> Optional[int]:
    """
    Función principal que orquesta el guardado de una transcripción simple.
    """
    if not data.numero_caso or not data.agente_usuario:
        print("❌ Error: 'numero_caso' y 'agente_usuario' son requeridos para guardar.")
        return None

    # Paso 1: Obtener el ID del caso y el ID del usuario.
    conn = create_connection()
    if conn is None: return
    cursor = conn.cursor()


    transcripcion_id: Optional[int] = None
    try:
        sql = """
        INSERT INTO transcripciones_simples (caso_id, usuario_id, texto_transcripcion)
        VALUES (%s, %s, %s)
        RETURNING id;
        """
        # Se corrigió la lista de valores, usando los IDs numéricos
        cursor.execute(sql, (data.numero_caso, data.agente_usuario, data.texto_transcripcion))
        transcripcion_id = cursor.fetchone()[0]
        conn.commit()
    except (OperationalError, IntegrityError) as e:
        print(f"❌ Error al insertar en 'transcripciones_simples': '{e}'")
        conn.rollback()
        return None
    except TypeError:
        print("❌ Error: La inserción no devolvió un ID. No se pudo guardar la transcripción.")
        conn.rollback()
        return None
    finally:
        cursor.close()
        close_connection(conn)

    return transcripcion_id

