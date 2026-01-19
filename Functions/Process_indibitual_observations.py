# -*- coding: utf-8 -*-
"""
Procesa observaciones (LLM) y:
1) Devuelve {"registros": [...]} (para tu main que guarda JSON).
2) Persiste en PostgreSQL (create_connection de Config_bd.py).

Tabla destino: public.clasificacion_onservacion
Columnas exactas en PostgreSQL:
nit_taller, nombre_taller, numero_aviso, numero_siniestro, placa, fecha_observacion,
usuario, rol_analista, observacion, clasificacion, explicacion, confianza,
explicacion_clasificacion, confianza_clasificacion, calidad_comunicativa_score,
explicacion_calidad, elementos_faltantes (JSONB)

Columnas exactas en tu CSV:
"NUMERO AVISO","NUMERO SINIESTRO","NIT TALLER","NOMBRE TALLER","PLACA",
"FECHA OBSERVACION","USUARIO","ROL ANALISTA","OBSERVACION"
"""

import os
import re
import ast
import json
import time
import math
import unicodedata
import traceback
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
from langchain_core.messages import SystemMessage, HumanMessage

# === Importa tus funciones reales ===
from config.Config_bd import create_connection
from config.Config_bd import insertar_clasificacion# Debe devolver conexión psycopg2
from services.miscelaneous import load_prompts_generales  # Lee prompt desde tu YAML/servicio

# ================================
# DEBUG
# ================================
DEBUG = True
def set_debug(value: bool) -> None:
    global DEBUG
    DEBUG = bool(value)

def dprint(msg: str) -> None:
    if DEBUG:
        print(msg)

# ================================
# LOGGING DETALLADO
# ================================
def _log_exception_detallado(e: BaseException, contexto: str = "") -> None:
    print(f"❌ [{contexto}] {type(e).__name__}: {e}")
    tb = e.__traceback__
    if tb:
        for i, frame in enumerate(traceback.extract_tb(tb), start=1):
            archivo = frame.filename
            linea = frame.lineno
            funcion = frame.name
            codigo = (frame.line or "").strip()
            print(f"   #{i} File \"{archivo}\", line {linea}, in {funcion}")
            if codigo:
                print(f"       → {codigo}")
        print("--- TRACEBACK COMPLETO ---")
        print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        print("--------------------------")

# ================================
# HELPERS
# ================================
def _coerce_scalar(v: Any) -> Any:
    """
    Convierte Series/list/tuple/str-list -> escalar limpio, sin usar 'or' booleano.
    """
    try:
        if isinstance(v, pl.Series):
            # si viene una serie (no debería con to_dict en fila, pero por seguridad)
            return v.item() if len(v) > 0 else None
        if isinstance(v, (list, tuple)):
            return v[0] if v else None
        if isinstance(v, str):
            s = v.strip()
            if s.startswith('[') and s.endswith(']'):
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple)) and parsed:
                        return parsed[0]
                except Exception:
                    pass
            return s or None
        return v
    except Exception:
        return v

def _parse_llm_json(raw: str) -> Optional[Dict[str, Any]]:
    """Parsea robustamente JSON desde respuesta LLM; acepta ```json ...``` o JSON plano."""
    if not isinstance(raw, str):
        return None
    txt = unicodedata.normalize('NFKC', raw)
    txt = re.sub(r'[\x00-\x1f\x7f]', '', txt)
    txt = re.sub(r'(?<!\\)\n|\r|\t', ' ', txt)
    txt = txt.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", txt, re.IGNORECASE)
    if m:
        txt = m.group(1).strip()
    try:
        return json.loads(txt)
    except Exception:
        pass
    start = txt.find("{")
    end = txt.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(txt[start:end + 1])
        except Exception:
            return None
    return None

# ================================
# CONFIGURACIÓN DE BD
# ================================
SCHEMA = "public"
TABLE_NAME = "clasificacion_onservacion"

# Orden EXACTO en PostgreSQL
COLUMNS_SQL_ORDER = [
    "nit_taller", "nombre_taller", "numero_aviso", "numero_siniestro", "placa",
    "fecha_observacion", "usuario", "rol_analista", "observacion",
    "clasificacion", "explicacion", "confianza", "explicacion_clasificacion",
    "confianza_clasificacion", "calidad_comunicativa_score",
    "explicacion_calidad", "elementos_faltantes"
]

# DDL 100% PostgreSQL
DDL_PG = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE_NAME} (
    id BIGSERIAL PRIMARY KEY,
    nit_taller TEXT,
    nombre_taller TEXT,
    numero_aviso TEXT,
    numero_siniestro TEXT,
    placa TEXT,
    fecha_observacion TEXT,
    usuario TEXT,
    rol_analista TEXT,
    observacion TEXT,
    clasificacion TEXT,
    explicacion TEXT,
    confianza DOUBLE PRECISION,
    explicacion_clasificacion TEXT,
    confianza_clasificacion DOUBLE PRECISION,
    calidad_comunicativa_score INTEGER,
    explicacion_calidad TEXT,
    elementos_faltantes JSONB
);
"""

def _ensure_table_exists(conn) -> None:
    """Crea la tabla si no existe; rollback en caso de error."""
    try:
        with conn.cursor() as cur:
            cur.execute(DDL_PG)
        conn.commit()
        dprint(f"🗄️ Tabla {SCHEMA}.{TABLE_NAME} verificada/creada.")
    except Exception as e:
        _log_exception_detallado(e, contexto="DDL")
        try:
            conn.rollback()
        except Exception:
            pass
        # Propaga para que capas superiores decidan (pero no detiene el JSON)
        raise

def _insert_batch_pg(conn, registros: List[Dict[str, Any]]) -> None:
    """
    Inserta registros con placeholders %s.
    elementos_faltantes -> ::jsonb (serializado desde list/obj).
    """
    if not registros:
        dprint("⚠️ No hay registros para insertar.")
        return

    cols = ", ".join(COLUMNS_SQL_ORDER)
    ph = ", ".join(["%s::jsonb" if c == "elementos_faltantes" else "%s" for c in COLUMNS_SQL_ORDER])
    sql = f"INSERT INTO {SCHEMA}.{TABLE_NAME} ({cols}) VALUES ({ph})"

    try:
        with conn.cursor() as cur:
            for i, rec in enumerate(registros, start=1):
                values: List[Any] = []
                for c in COLUMNS_SQL_ORDER:
                    val = rec.get(c)
                    if c == "elementos_faltantes":
                        val = json.dumps(val or [], ensure_ascii=False)
                    values.append(val)
                cur.execute(sql, values)
                dprint(f"💾 INSERT {i:04d}: taller='{rec.get('nombre_taller')}', placa='{rec.get('placa')}', clasif='{rec.get('clasificacion')}'")
        conn.commit()
        print(f"💾 Guardados {len(registros)} registros en BD.")
    except Exception as e:
        _log_exception_detallado(e, contexto="INSERT BATCH")
        try:
            conn.rollback()
        except Exception:
            pass
        # No re-raise: seguimos retornando JSON al main

# ================================
# WORKER (procesa 1 registro con el LLM)
# ================================
def _worker_un_registro(
    df: pl.DataFrame,
    idx: int,
    cliente_llm: Any,
    system_prompt: str,
    max_retries: int = 2,
    backoff_sec: float = 2.0,
) -> Dict[str, Any]:
    """
    Devuelve un dict con claves 1:1 con PostgreSQL.
    Lee únicamente las columnas REALES del CSV (evitamos usar 'or' con Series).
    """
    try:
        # Tomamos la fila como dict plano:
        registro = df.row(idx, named=True)  # evita sorpresas con Series

        # CSV -> BD (nombres exactos)
        nit_taller          = _coerce_scalar(registro.get("NIT TALLER"))
        nombre_taller       = _coerce_scalar(registro.get("NOMBRE TALLER"))
        numero_aviso        = _coerce_scalar(registro.get("NUMERO AVISO"))
        numero_siniestro    = _coerce_scalar(registro.get("NUMERO SINIESTRO"))
        placa               = _coerce_scalar(registro.get("PLACA"))
        fecha_observacion   = _coerce_scalar(registro.get("FECHA OBSERVACION"))
        usuario             = _coerce_scalar(registro.get("USUARIO"))
        rol_analista        = _coerce_scalar(registro.get("ROL ANALISTA"))
        observacion         = _coerce_scalar(registro.get("OBSERVACION"))
        estado_aviso        = _coerce_scalar(registro.get("ESTADO AVISO"))
        base_out = {
            "nit_taller": nit_taller,
            "nombre_taller": nombre_taller,
            "numero_aviso": numero_aviso,
            "estado_aviso":estado_aviso,
            "numero_siniestro": numero_siniestro,
            "placa": placa,
            "fecha_observacion": fecha_observacion,
            "usuario": usuario,
            "rol_analista": rol_analista,
            "observacion": observacion,
        }

        if not system_prompt:
            base_out.update({
                "clasificacion": "sin_clasificar",
                "explicacion": "prompt no encontrado",
                "confianza": 0.0,
                "explicacion_clasificacion": "prompt no encontrado",
                "confianza_clasificacion": 0.0,
                "calidad_comunicativa_score": 0,
                "explicacion_calidad": "Sin evaluación de calidad comunicativa.",
                "elementos_faltantes": [],
            })
            dprint(f"⚠️ [{idx + 1}] Sin prompt. Taller='{nombre_taller}', Placa='{placa}'")
            return base_out

        ALLOWED = {"comunicacion_cliente", "cambio_estado", "comunicacion_interna", "sin_cambio", "sin_clasificar","gestion_repuestos"}

        intento = 0
        last_error_reason = ""

        # Defaults (por si falla el LLM)
        clasificacion = "sin_clasificar"
        explicacion = "no se pudo parsear salida LLM"
        confianza = 0.0
        explicacion_clasificacion = explicacion
        confianza_clasificacion = confianza
        calidad_comunicativa_score = 0
        explicacion_calidad = "La salida no fue utilizable."
        elementos_faltantes: List[str] = []

        while intento <= max_retries:
            intento += 1
            try:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Observación: {observacion}")
                ]
                resp = cliente_llm.invoke(messages)
                raw = resp.content if hasattr(resp, "content") else str(resp)
                parsed = _parse_llm_json(raw)

                if not parsed:
                    last_error_reason = "no se pudo parsear JSON"
                    raise ValueError(last_error_reason)

                c = (parsed.get("clasificacion") or "").strip()
                print(c)
                if c not in ALLOWED:
                    last_error_reason = f"clasificacion inválida: {c!r}"
                    raise ValueError(last_error_reason)

                expl = (parsed.get("explicacion_clasificacion") or parsed.get("explicacion") or "sin explicación").strip()

                conf_val = parsed.get("confianza_clasificacion", parsed.get("confianza", 0.0))
                try:
                    conf_f = float(conf_val)
                except Exception:
                    last_error_reason = f"confianza inválida: {conf_val!r}"
                    raise ValueError(last_error_reason)

                # Asignar resultados
                clasificacion = c
                explicacion = expl
                confianza = max(0.0, min(1.0, float(conf_f)))
                explicacion_clasificacion = expl
                confianza_clasificacion = confianza

                try:
                    calidad_comunicativa_score = int(round(float(parsed.get("calidad_comunicativa_score", 0))))
                except Exception:
                    calidad_comunicativa_score = 0
                calidad_comunicativa_score = max(0, min(100, calidad_comunicativa_score))

                explicacion_calidad = (parsed.get("explicacion_calidad") or "").strip() or "Sin evaluación de calidad comunicativa."

                elems = parsed.get("elementos_faltantes") or []
                if isinstance(elems, (str, int, float)):
                    elementos_faltantes = [str(elems)]
                elif isinstance(elems, list):
                    elementos_faltantes = [str(x).strip() for x in elems if str(x).strip()]
                else:
                    elementos_faltantes = []

                break  # éxito

            except Exception as e_llm:
                _log_exception_detallado(e_llm, contexto=f"WORKER LLM idx={idx + 1} intento={intento}")
                if intento <= max_retries:
                    wait_s = backoff_sec * (2 ** (intento - 1))
                    dprint(f"⏳ [{idx + 1}] Reintento en {wait_s:.1f}s… Motivo: {last_error_reason}")
                    time.sleep(wait_s)
                else:
                    dprint(f"⚠️ [{idx + 1}] Reintentos agotados. Motivo: {last_error_reason}")

        base_out.update({
            "clasificacion": clasificacion,
            "explicacion": explicacion,
            "confianza": confianza,
            "explicacion_clasificacion": explicacion_clasificacion,
            "confianza_clasificacion": confianza_clasificacion,
            "calidad_comunicativa_score": calidad_comunicativa_score,
            "explicacion_calidad": explicacion_calidad,
            "elementos_faltantes": elementos_faltantes,
        })

        insertar_clasificacion(base_out)

        print(f"✅ [{idx + 1}] Taller='{nombre_taller}', Placa='{placa}', Clasif='{clasificacion}', Conf={confianza:.2f}")
        return base_out

    except Exception as e:
        _log_exception_detallado(e, contexto=f"WORKER idx={idx + 1}")
        out_err = {c: None for c in COLUMNS_SQL_ORDER}
        out_err.update({"clasificacion": "sin_clasificar", "explicacion": f"{type(e).__name__}: {e}", "confianza": 0.0, "elementos_faltantes": []})
        return out_err

# ================================
# ORQUESTADOR (procesa + guarda + devuelve JSON)
# ================================
def procesar_observacion_individual(
    df_observacion: pl.DataFrame,
    prompt_sistema: str,
    cliente_llm: Any,
    max_workers: Optional[int] = 0,
    chunksize: Optional[int] = 0,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    - Usa prompt_sistema si llega; si no, carga 'observaciones_clasificacion_prompt' con load_prompts_generales.
    - Ejecuta en paralelo con hilos.
    - Intenta persistir en PostgreSQL (si create_connection() entrega conexión).
    - Devuelve {"registros": [...]} para tu main.
    """
    if df_observacion is None or df_observacion.height == 0:
        return {"registros": []}

    system_prompt = (prompt_sistema or "").strip() or load_prompts_generales("observaciones_clasificacion_prompt")
    if not system_prompt:
        print("⚠️ No se encontró el prompt 'observaciones_clasificacion_prompt'. Se marcará 'sin_clasificar'.")

    # Paralelismo
    workers = max_workers if (max_workers and max_workers > 0) else (os.cpu_count() or 1)
    n = df_observacion.height

    # (Solo para logging) estimación de lotes
    if not chunksize or chunksize <= 0:
        chunk_size_eff = max(1, math.ceil(n / workers))
        marks = sorted(set(min((i + 1) * chunk_size_eff, n) for i in range(workers)))
        total_lotes = len(marks)
    else:
        chunk_size_eff = min(max(1, chunksize), n)
        marks = list(range(chunk_size_eff, n, chunk_size_eff)) + [n]
        total_lotes = len(marks)

    print(f"⚙️ Procesando {n} observaciones con {workers} hilos (lotes={total_lotes}, tamaño_lote≈{chunk_size_eff})...")

    resultados: List[Optional[Dict[str, Any]]] = [None] * n
    start_global = time.perf_counter()
    completed, lot_start_time, next_mark_idx = 0, start_global, 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(_worker_un_registro, df_observacion, i, cliente_llm, system_prompt): i
            for i in range(n)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                res = future.result()
            except Exception as e:
                _log_exception_detallado(e, contexto=f"ORQUESTADOR idx={idx}")
                res = {c: None for c in COLUMNS_SQL_ORDER}
                res.update({
                    "clasificacion": "sin_clasificar",
                    "explicacion": f"{type(e).__name__}: {e}",
                    "confianza": 0.0,
                    "elementos_faltantes": [],
                })

            resultados[idx] = res
            completed += 1

            while next_mark_idx < len(marks) and completed >= marks[next_mark_idx]:
                lote_num = next_mark_idx + 1
                elapsed = time.perf_counter() - lot_start_time
                print(f"⏱️ Lote {lote_num}/{total_lotes} completado ({completed}/{n}) en {elapsed:.2f} s")
                lot_start_time = time.perf_counter()
                next_mark_idx += 1

    total_elapsed = time.perf_counter() - start_global
    print(f"✅ Finalizado. Registros: {n} — Tiempo total: {total_elapsed:.2f} s")

    registros_finales: List[Dict[str, Any]] = [r for r in resultados if r is not None]

    # Resumen útil (debug)
    try:
        by_clasif: Dict[str, int] = {}
        for r in registros_finales:
            by_clasif[r.get("clasificacion", "sin_clasificar")] = by_clasif.get(r.get("clasificacion", "sin_clasificar"), 0) + 1
        print(f"📊 Resumen por clasificación: {by_clasif}")
    except Exception:
        pass

    return {"registros": registros_finales}
