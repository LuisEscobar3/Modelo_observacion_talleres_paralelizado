
# === BEGIN ROBUST LLM JSON HELPERS (added) ===
import re as _re
import json as _json
import ast as _ast
import unicodedata as _unicodedata

def _strip_control_chars(text: str) -> str:
    return _re.sub(r"[\x00-\x1f\x7f]", "", text)

def _desmart_quotes(text: str) -> str:
    trans = str.maketrans({
        '“': '"', '”': '"', '„': '"', '‟': '"', '«': '"', '»': '"',
        '‘': "'", '’': "'", '‚': "'"
    })
    return text.translate(trans)

def _remove_trailing_commas(text: str) -> str:
    return _re.sub(r",\s*(\]|\})", r"\1", text)

def _extract_json_candidate(txt: str):
    if not isinstance(txt, str):
        return None
    t = _unicodedata.normalize('NFKC', txt)
    t = _strip_control_chars(t).strip()
    m = _re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, _re.IGNORECASE)
    if m:
        t = m.group(1).strip()
    start = t.find('{')
    end = t.rfind('}')
    if start != -1 and end != -1 and end > start:
        return t[start:end+1]
    return None

def _pythonish_to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _pythonish_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_pythonish_to_jsonable(v) for v in obj]
    if obj is True or obj is False or obj is None:
        return obj
    if isinstance(obj, (int, float, str)):
        return obj
    return str(obj)

# Defaults & normalization
DEFAULT_OUTPUT = {
    "resumen_general": "No fue posible evaluar.",
    "informatividad_score_0a100": 0,
    "informatividad_justificacion": "Sin datos",
    "oportunidad_score_0a100": 0,
    "oportunidad_justificacion": "Sin datos",
    "orientacion_score_0a100": 0,
    "orientacion_justificacion": "Sin datos",
    "claridad_score_0a100": 0,
    "claridad_justificacion": "Sin datos",
    "consistencia_score_0a100": 0,
    "consistencia_justificacion": "Sin datos",
    "score_final_0a100": 0,
    "evidencias": [],
    "observaciones": ["Nota/Supuesto: salida por defecto."],
    "confianza": "baja",
}

NUMERIC_KEYS = [
    "informatividad_score_0a100",
    "oportunidad_score_0a100",
    "orientacion_score_0a100",
    "claridad_score_0a100",
    "consistencia_score_0a100",
    "score_final_0a100",
]

def _to_int_0_100(x) -> int:
    try:
        v = int(round(float(x)))
        return max(0, min(100, v))
    except Exception:
        return 0

def _ensure_list_str(x, max_items: int = 20) -> list:
    if isinstance(x, list):
        out = [str(i) for i in x if i is not None]
    elif isinstance(x, str) and x.strip():
        out = [x.strip()]
    else:
        out = []
    return out[:max_items]

def _sanitize_llm_output(d: dict | None) -> dict:
    if not isinstance(d, dict):
        d = {}
    out = {**DEFAULT_OUTPUT, **d}
    for k in NUMERIC_KEYS:
        out[k] = _to_int_0_100(out.get(k))
    out["evidencias"] = _ensure_list_str(out.get("evidencias"), max_items=10)
    out["observaciones"] = _ensure_list_str(out.get("observaciones"), max_items=20)
    conf = str(out.get("confianza", "baja")).lower()
    if conf not in {"baja", "media", "alta"}:
        conf = "baja"
    out["confianza"] = conf
    if not d.get("score_final_0a100"):
        subs = [out["informatividad_score_0a100"], out["oportunidad_score_0a100"],
                out["orientacion_score_0a100"], out["claridad_score_0a100"], out["consistencia_score_0a100"]]
        out["score_final_0a100"] = _to_int_0_100(sum(subs) / max(1, len(subs)))
    for k in [
        "resumen_general",
        "informatividad_justificacion",
        "oportunidad_justificacion",
        "orientacion_justificacion",
        "claridad_justificacion",
        "consistencia_justificacion",
    ]:
        out[k] = str(out.get(k, ""))
    return out
# === END ROBUST LLM JSON HELPERS (added) ===


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
    Parser robusto para extraer un JSON válido desde la respuesta del LLM.
    Estrategia por intentos:
      1) json.loads directo
      2) limpiar comillas tipográficas y comas colgantes; extraer bloque {...} y reintentar
      3) ast.literal_eval -> Python -> JSON serializable
    """
    if not isinstance(raw, str):
        return None

    # Intento 1: literal
    try:
        obj = _json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Limpiezas + candidato
    cand = _extract_json_candidate(raw) or raw
    cand = _desmart_quotes(cand)
    cand = _remove_trailing_commas(cand)
    cand = _strip_control_chars(cand).strip()

    # Intento 2: JSON tras limpieza
    try:
        obj = _json.loads(cand)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Intento 3: literal_eval "pythonish" → JSONable
    try:
        py_obj = _ast.literal_eval(cand)
        py_obj = _pythonish_to_jsonable(py_obj)
        obj = _json.loads(_json.dumps(py_obj, ensure_ascii=False))
        return obj if isinstance(obj, dict) else None
    except Exception:
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

        try:
            if verbose:
                print(
                    f"▶️ Iniciando caso idx={idx + 1} — aviso={meta.get('numero_aviso')} "
                    f"siniestro={meta.get('numero_siniestro')} placa={meta.get('placa')} eventos={total_evt}"
                )
        except Exception:
            pass

        # Sin contenido → salida mínima controlada
        if not obs_unidas:
            out = {
                "numero_aviso": meta.get("numero_aviso"),
                "numero_siniestro": meta.get("numero_siniestro"),
                "placa": meta.get("placa"),
                "total_eventos": total_evt,
                **DEFAULT_OUTPUT,
                "resumen_general": "sin datos",
                "observaciones": [
                    "Nota/Supuesto: no se encontraron observaciones para evaluar."
                ],
            }
            if verbose:
                print(f"⚠️ Caso idx={idx + 1} sin observaciones.")
                print(f"🧾 Final [{idx + 1}]: {json.dumps(out, ensure_ascii=False)}")
            return out

        # Prompt del evaluador de calidad
        system_prompt = load_prompts_generales("observaciones_calidad_prompt")
        if not system_prompt:
            raise RuntimeError("Prompt 'observaciones_calidad_prompt' no encontrado.")

        entrada = {
            "numero_aviso": "" if meta.get("numero_aviso") is None else str(meta.get("numero_aviso")),
            "numero_siniestro": "" if meta.get("numero_siniestro") is None else str(meta.get("numero_siniestro")),
            "placa": "" if meta.get("placa") is None else str(meta.get("placa")),
            "usuario": "" if meta.get("usuario") is None else str(meta.get("usuario")),
            "rol_analista": "" if meta.get("rol_analista") is None else str(meta.get("rol_analista")),
            "total_eventos": total_evt,
            "observaciones_unidas": obs_unidas,
        }

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

        # Invocación con reintentos simples
        last_err = None
        llm_json = None
        for attempt in range(max_retries + 1):
            try:
                if verbose:
                    print(f"🚀 LLM calidad idx {idx + 1} intento {attempt + 1}/{max_retries + 1}")
                msgs = [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
                # Mantener compat: invoke/predict/generate/callable
                if hasattr(cliente_llm, "invoke"):
                    resp = cliente_llm.invoke(msgs)
                    raw = resp.content if hasattr(resp, "content") else str(resp)
                elif hasattr(cliente_llm, "predict"):
                    raw = str(cliente_llm.predict(msgs))
                elif hasattr(cliente_llm, "generate"):
                    gen = cliente_llm.generate(msgs)
                    raw = getattr(gen, "text", str(gen))
                elif callable(cliente_llm):
                    raw = str(cliente_llm(msgs))
                else:
                    raise TypeError("cliente_llm no es invocable de forma conocida")

                if verbose:
                    preview = raw if len(str(raw)) < 1000 else str(raw)[:1000] + "...<truncated>"
                    print(f"📥 LLM RAW [{idx + 1}, intento {attempt + 1}]: {preview}")

                parsed = _parse_llm_json(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("Respuesta del modelo no contiene JSON válido")

                llm_json = parsed
                break
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    wait_s = backoff_sec * (2 ** attempt)
                    if verbose:
                        print(f"⏳ JSON inválido/erróneo ({e}). Reintentando en {wait_s:.1f}s...")
                    import time as _time
                    _time.sleep(wait_s)
                else:
                    if verbose:
                        print(f"⚠️ Reintentos agotados en idx {idx + 1}. Motivo: {e}")
                    llm_json = None

        # Ensamblaje final con normalización robusta
        out = {
            "numero_aviso": entrada["numero_aviso"],
            "numero_siniestro": entrada["numero_siniestro"],
            "placa": entrada["placa"],
            "total_eventos": entrada["total_eventos"],
        }
        out.update(_sanitize_llm_output(llm_json))

        if verbose:
            print(f"✅ Procesado caso {idx + 1}")
            print(f"🧾 Final [{idx + 1}]: {json.dumps(out, ensure_ascii=False)}")
        return out

    except Exception as e:
        if verbose:
            print(f"❌ Error catastrófico en idx {idx + 1}: {e}")
            import traceback as _traceback
            print("--- TRACEBACK COMPLETO ---")
            print(_traceback.format_exc())
            print("-------------------------")
        out = {
            "numero_aviso": None,
            "numero_siniestro": None,
            "placa": None,
            "total_eventos": 0,
            **DEFAULT_OUTPUT,
            "resumen_general": "error",
            "informatividad_justificacion": f"error: {e}",
            "oportunidad_justificacion": "error",
            "orientacion_justificacion": "error",
            "claridad_justificacion": "error",
            "consistencia_justificacion": "error",
            "observaciones": ["Nota/Supuesto: error en worker."],
            "confianza": "baja",
        }
        if verbose:
            print(f"🧾 Final [{idx + 1}]: {json.dumps(out, ensure_ascii=False)}")
        return out

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


