import os
import re
import ast
import json
import time
import math
import unicodedata
import traceback
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
from langchain.schema import SystemMessage, HumanMessage

# Si ya los tienes importados en el archivo, puedes dejar estas líneas:
# from .payloads import build_json_para_n8n_registro  # (no se usa aquí)
from services.miscelaneous import load_prompts_generales


# =========================
# Helpers genéricos
# =========================

def _coerce_scalar(v: Any) -> Any:
    try:
        if isinstance(v, pl.Series):
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
            return s
        return v
    except Exception:
        return v


def _parse_llm_json(raw: str) -> Optional[Dict[str, Any]]:
    """
    Intenta parsear un JSON robusto desde una respuesta de un LLM:
    - Limpia controles
    - Soporta bloque ```json ... ```
    - Busca primer {...} si falla
    """
    if not isinstance(raw, str):
        return None
    txt = unicodedata.normalize('NFKC', raw)
    txt = re.sub(r'[\x00-\x1f\x7f]', '', txt)
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
        candidate = txt[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def _norm_col(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.strip().lower().replace("  ", " ")


def _find_col(df: pl.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Busca una columna por variaciones (case/acentos).
    """
    norm_map = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        real = norm_map.get(_norm_col(cand))
        if real:
            return real
    # búsqueda laxa por contiene
    for key_norm, real in norm_map.items():
        for cand in candidates:
            if _norm_col(cand) in key_norm:
                return real
    return None


def _to_date_safe(s: Any) -> Optional[str]:
    """
    Convierte varias formas a 'YYYY-MM-DD' si es posible; si no, devuelve None.
    """
    if s is None:
        return None
    ss = str(s).strip()
    if not ss:
        return None
    fmts = [
        "%Y-%m-%d", "%Y/%m/%d",
        "%d/%m/%Y", "%d-%m-%Y",
        "%m/%d/%Y", "%m-%d-%Y",
        "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"
    ]
    from datetime import datetime
    for f in fmts:
        try:
            return datetime.strptime(ss, f).date().strftime("%Y-%m-%d")
        except Exception:
            continue
    # último intento: si viene estilo ISO extendido
    try:
        return datetime.fromisoformat(ss).date().strftime("%Y-%m-%d")
    except Exception:
        return None


def _sanitize_obs_text(x: Any) -> str:
    """
    Limpia texto de observación para evitar romper el separador '|'.
    """
    s = str(x) if x is not None else ""
    s = s.replace("|", " / ")  # evita romper el join
    s = re.sub(r"\s+", " ", s).strip()
    return s


# =========================
# Armado de "observaciones_unidas"
# =========================

def _build_observaciones_unidas_para_idx(df: pl.DataFrame, idx: int) -> Tuple[str, int, Dict[str, Any]]:
    """
    Intenta obtener 'observaciones_unidas' y total_eventos para el caso en idx.
    - Si ya existe columna 'observaciones_unidas' en la fila → la usa.
    - Si no, agrupa TODAS las filas del mismo caso y concatena "YYYY-MM-DD - texto" con '|'.
    Devuelve: (observaciones_unidas, total_eventos, meta_identificacion_dict)
    meta_identificacion: {'numero_aviso', 'numero_siniestro', 'placa', 'usuario', 'rol_analista'}
    """
    row = df.row(idx, named=True)

    # Identificadores posibles
    col_aviso = _find_col(df, ["NUMERO AVISO", "numero_aviso", "aviso"])
    col_sini = _find_col(df, ["NUMERO SINIESTRO", "numero_siniestro", "siniestro"])
    col_placa = _find_col(df, ["PLACA", "placa"])
    col_user = _find_col(df, ["USUARIO", "usuario"])
    col_rol = _find_col(df, ["ROL ANALISTA", "rol_analista", "rol"])

    numero_aviso = _coerce_scalar(row.get(col_aviso)) if col_aviso else None
    numero_siniestro = _coerce_scalar(row.get(col_sini)) if col_sini else None
    placa = _coerce_scalar(row.get(col_placa)) if col_placa else None
    usuario = _coerce_scalar(row.get(col_user)) if col_user else None
    rol_analista = _coerce_scalar(row.get(col_rol)) if col_rol else None

    meta = {
        "numero_aviso": numero_aviso,
        "numero_siniestro": numero_siniestro,
        "placa": placa,
        "usuario": usuario,
        "rol_analista": rol_analista
    }

    # Si ya viene observaciones_unidas en la fila, úsala
    col_obs_unidas = _find_col(df, ["observaciones_unidas", "OBSERVACIONES UNIDAS"])
    if col_obs_unidas and row.get(col_obs_unidas):
        s = _coerce_scalar(row[col_obs_unidas])
        s = str(s).strip()
        total = len([p for p in s.split("|") if p.strip()]) if s else 0
        return (s, total, meta)

    # Si NO viene, tratamos de agrupar por identificador (prioridad: aviso > siniestro > placa)
    key_col = None
    key_val = None
    for c, v in [(col_aviso, numero_aviso), (col_sini, numero_siniestro), (col_placa, placa)]:
        if c and v is not None and str(v).strip() != "":
            key_col, key_val = c, v
            break

    if key_col is None:
        # No hay llave para agrupar → usar solo esta fila si tiene fecha/observación
        col_fecha = _find_col(df, ["FECHA OBSERVACION", "FECHA_OBSERVACION", "fecha_observacion", "FECHA", "Fecha"])
        col_obs = _find_col(df, ["OBSERVACION", "observacion", "OBSERVACIONES", "observaciones"])
        fecha = _to_date_safe(row.get(col_fecha)) if col_fecha else None
        texto = _sanitize_obs_text(row.get(col_obs)) if col_obs else ""
        pieza = f"{fecha or '0000-00-00'} - {texto}".strip(" -")
        piezas = [pieza] if texto else []
    else:
        # Agrupar todas las filas del caso
        sub = df.filter(pl.col(key_col) == key_val)

        col_fecha = _find_col(df, ["FECHA OBSERVACION", "FECHA_OBSERVACION", "fecha_observacion", "FECHA", "Fecha"])
        col_obs = _find_col(df, ["OBSERVACION", "observacion", "OBSERVACIONES", "observaciones"])

        piezas = []
        if sub.height > 0 and col_obs:
            # construimos lista (fecha_norm, texto_sanitizado)
            tmp = []
            for i in range(sub.height):
                r = sub.row(i, named=True)
                f = _to_date_safe(r.get(col_fecha)) if col_fecha else None
                t = _sanitize_obs_text(r.get(col_obs))
                if t:
                    tmp.append((f or "0000-00-00", t))
            # ordenar por fecha
            tmp.sort(key=lambda x: x[0])
            piezas = [f"{f} - {t}".strip(" -") for f, t in tmp]

    obs_unidas = " | ".join(piezas)
    total = len(piezas)
    return (obs_unidas, total, meta)


# =========================
# Worker (1 caso)
# =========================

def _worker_un_caso(
        df: pl.DataFrame,
        idx: int,
        cliente_llm: Any,
        max_retries: int = 2,
        backoff_sec: float = 2.0,
        verbose: bool = False,
) -> Dict[str, Any]:
    start_ts = time.perf_counter()
    try:
        obs_unidas, total_evt, meta = _build_observaciones_unidas_para_idx(df, idx)

        print(f"▶️ Iniciando caso idx={idx + 1} — aviso={meta['numero_aviso']} siniestro={meta['numero_siniestro']} placa={meta['placa']} eventos={total_evt}")

        # Si no hay contenido, devolvemos estructura mínima
        if not obs_unidas:
            out = {
                "numero_aviso": meta["numero_aviso"],
                "numero_siniestro": meta["numero_siniestro"],
                "placa": meta["placa"],
                "total_eventos": total_evt,
                # Salida del LLM vacía/controlada
                "resumen_general": "sin datos",
                "informatividad_score_0a100": 0,
                "informatividad_justificacion": "sin datos",
                "oportunidad_score_0a100": 0,
                "oportunidad_justificacion": "sin datos",
                "orientacion_score_0a100": 0,
                "orientacion_justificacion": "sin datos",
                "claridad_score_0a100": 0,
                "claridad_justificacion": "sin datos",
                "consistencia_score_0a100": 0,
                "consistencia_justificacion": "sin datos",
                "score_final_0a100": 0,
                "evidencias": [],
                "observaciones": ["Nota/Supuesto: no se encontraron observaciones para evaluar."],
                "confianza": "baja"
            }
            print(f"⚠️ Caso idx={idx + 1} sin observaciones.")
            print(f"🧾 Final [{idx + 1}]: {json.dumps(out, ensure_ascii=False)}")
            return out

        # System prompt del evaluador de calidad CX
        system_prompt = load_prompts_generales("observaciones_calidad_prompt")
        if not system_prompt:
            raise RuntimeError("Prompt 'observaciones_calidad_prompt' no encontrado.")

        # ENTRADA exactamente como especifica el prompt
        entrada = {
            "numero_aviso": "" if meta["numero_aviso"] is None else str(meta["numero_aviso"]),
            "numero_siniestro": "" if meta["numero_siniestro"] is None else str(meta["numero_siniestro"]),
            "placa": "" if meta["placa"] is None else str(meta["placa"]),
            "usuario": "" if meta["usuario"] is None else str(meta["usuario"]),
            "rol_analista": "" if meta["rol_analista"] is None else str(meta["rol_analista"]),
            "total_eventos": total_evt,
            "observaciones_unidas": obs_unidas
        }

        # FORMATO_DE_SALIDA (para reforzar comportamiento del modelo)
        formato_salida_str = """{
  "resumen_general": "string",
  "informatividad_score_0a100": 0,
  "informatividad_justificacion": "string",
  "oportunidad_score_0a100": 0,
  "oportunidad_justificacion": "string",

  "orientacion_score_0a100": 0,
  "orientacion_justificacion": "string",

  "claridad_score_0a100": 0,
  "claridad_justificacion": "string",

  "consistencia_score_0a100": 0,
  "consistencia_justificacion": "string",

  "score_final_0a100": 0,

  "evidencias": [
    "<cita recortada con fecha>",
    "<cita recortada con fecha>"
  ],

  "observaciones": [
    "Recomendación: <acción concreta en imperativo>",
    "Nota/Supuesto: <texto breve explicando limitación>"
  ],

  "confianza": "baja|media|alta"
}"""

        # HumanMessage: ENTRADA + exigencia de salida única JSON
        human_content = (
            "INSTRUCCIONES:\n"
            "- Usa EXCLUSIVAMENTE los datos de la ENTRADA.\n"
            "- Devuelve SOLO un JSON único, válido y parseable EXACTAMENTE en el FORMATO_DE_SALIDA indicado (sin texto adicional).\n\n"
            "<ENTRADA>\n"
            f"{json.dumps(entrada, ensure_ascii=False, indent=2)}\n"
            "</ENTRADA>\n\n"
            "FORMATO_DE_SALIDA:\n"
            f"{formato_salida_str}"
        )

        # Invocación con reintentos
        last_err = None
        llm_json = None
        for attempt in range(max_retries + 1):
            try:
                print(f"🚀 LLM calidad idx {idx + 1} intento {attempt + 1}/{max_retries + 1}")
                msgs = [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
                resp = cliente_llm.invoke(msgs)
                raw = resp.content if hasattr(resp, "content") else str(resp)

                print(f"📥 LLM RAW [{idx + 1}, intento {attempt + 1}]: {raw}")

                parsed = _parse_llm_json(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("Respuesta del modelo no contiene JSON válido.")

                llm_json = parsed
                break
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    wait_s = backoff_sec * (2 ** attempt)
                    print(f"⏳ JSON inválido/erróneo ({e}). Reintentando en {wait_s:.1f}s...")
                    time.sleep(wait_s)
                else:
                    print(f"⚠️ Reintentos agotados en idx {idx + 1}. Motivo: {e}")
                    # fallback mínimo
                    llm_json = {
                        "resumen_general": "No fue posible evaluar por error del modelo.",
                        "informatividad_score_0a100": 0,
                        "informatividad_justificacion": "Error del modelo",
                        "oportunidad_score_0a100": 0,
                        "oportunidad_justificacion": "Error del modelo",
                        "orientacion_score_0a100": 0,
                        "orientacion_justificacion": "Error del modelo",
                        "claridad_score_0a100": 0,
                        "claridad_justificacion": "Error del modelo",
                        "consistencia_score_0a100": 0,
                        "consistencia_justificacion": "Error del modelo",
                        "score_final_0a100": 0,
                        "evidencias": [],
                        "observaciones": ["Nota/Supuesto: no se pudo obtener salida del LLM."],
                        "confianza": "baja"
                    }

        # Ensamblaje final con trazabilidad del caso
        out = {
            "numero_aviso": entrada["numero_aviso"],
            "numero_siniestro": entrada["numero_siniestro"],
            "placa": entrada["placa"],
            "total_eventos": entrada["total_eventos"],
        }
        if isinstance(llm_json, dict):
            out.update(llm_json)

        print(f"✅ Procesado caso {idx + 1}")
        print(f"🧾 Final [{idx + 1}]: {json.dumps(out, ensure_ascii=False)}")

        return out

    except Exception as e:
        print(f"❌ Error catastrófico en idx {idx + 1}: {e}")
        print("--- TRACEBACK COMPLETO ---")
        print(traceback.format_exc())
        print("-------------------------")
        # salida segura
        out = {
            "numero_aviso": None,
            "numero_siniestro": None,
            "placa": None,
            "total_eventos": 0,
            "resumen_general": "error",
            "informatividad_score_0a100": 0,
            "informatividad_justificacion": f"error: {e}",
            "oportunidad_score_0a100": 0,
            "oportunidad_justificacion": "error",
            "orientacion_score_0a100": 0,
            "orientacion_justificacion": "error",
            "claridad_score_0a100": 0,
            "claridad_justificacion": "error",
            "consistencia_score_0a100": 0,
            "consistencia_justificacion": "error",
            "score_final_0a100": 0,
            "evidencias": [],
            "observaciones": ["Nota/Supuesto: error en worker."],
            "confianza": "baja"
        }
        print(f"🧾 Final [{idx + 1}]: {json.dumps(out, ensure_ascii=False)}")
        return out


# =========================
# Orquestación
# =========================

def procesar_calidad_por_caso(
        df_casos: pl.DataFrame,
        prompt_sistema: str,  # compatibilidad, no usado (el prompt se carga por nombre)
        cliente_llm: Any,
        max_workers: Optional[int] = 0,
        chunksize: Optional[int] = 0,
        verbose: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Evalúa la calidad de comunicación de CADA CASO (fila o grupo) usando el prompt `observaciones_calidad_prompt`.
    Devuelve: {"registros": [ {resultado_por_caso}, ... ]}
    """
    if df_casos is None or df_casos.height == 0:
        return {"registros": []}

    workers = max_workers if (max_workers and max_workers > 0) else (os.cpu_count() or 1)
    n = df_casos.height

    # Marcas de progreso
    if not chunksize or chunksize <= 0:
        chunk_size_eff = max(1, math.ceil(n / workers))
        marks = sorted(set(min((i + 1) * chunk_size_eff, n) for i in range(workers)))
        total_lotes = len(marks)
    else:
        chunk_size_eff = min(max(1, chunksize), n)
        marks = list(range(chunk_size_eff, n, chunk_size_eff)) + [n]
        total_lotes = len(marks)

    print(f"⚙️ Casos: {n} | hilos={workers} | lotes={total_lotes} | tamaño_lote≈{chunk_size_eff}")

    resultados: List[Optional[Dict[str, Any]]] = [None] * n
    start_global = time.perf_counter()
    completed, lot_start_time, next_mark_idx = 0, start_global, 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {executor.submit(_worker_un_caso, df_casos, i, cliente_llm, 2, 2.0, verbose): i for i in range(n)}

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                res = future.result()
            except Exception as e:
                print(f"❌ Futuro falló idx={idx}: {e}")
                print("--- TRACEBACK ORQ ---")
                print(traceback.format_exc())
                print("---------------------")
                res = {
                    "numero_aviso": None,
                    "numero_siniestro": None,
                    "placa": None,
                    "total_eventos": 0,
                    "resumen_general": "error",
                    "informatividad_score_0a100": 0,
                    "informatividad_justificacion": f"excepción en futuro: {e}",
                    "oportunidad_score_0a100": 0,
                    "oportunidad_justificacion": "error",
                    "orientacion_score_0a100": 0,
                    "orientacion_justificacion": "error",
                    "claridad_score_0a100": 0,
                    "claridad_justificacion": "error",
                    "consistencia_score_0a100": 0,
                    "consistencia_justificacion": "error",
                    "score_final_0a100": 0,
                    "evidencias": [],
                    "observaciones": ["Nota/Supuesto: excepción en orquestación."],
                    "confianza": "baja"
                }

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

    return {"registros": [r for r in resultados if r is not None]}
