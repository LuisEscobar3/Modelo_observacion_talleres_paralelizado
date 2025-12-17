import os
import re
import ast
import json
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
from langchain.schema import SystemMessage, HumanMessage

from services.miscelaneous import load_prompts_generales

# ========================================
#   MODO DEBUG PARA VER RESPUESTAS COMPLETAS DEL LLM
# ========================================
DEBUG_LLM_JSON = os.getenv("DEBUG_LLM_JSON", "0") == "1"


# ========================================
#            HELPERS GENERALES
# ========================================

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
    if not isinstance(raw, str):
        if DEBUG_LLM_JSON:
            print(f"_parse_llm_json: raw no es str → {type(raw)}", flush=True)
        return None

    # Limpieza básica de control chars
    txt = unicodedata.normalize('NFKC', raw)
    txt = re.sub(r'[\x00-\x1f\x7f]', '', txt).strip()

    # Si viene envuelto en ``` ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", txt, re.IGNORECASE)
    if m:
        txt = m.group(1).strip()

    # Intento directo
    try:
        return json.loads(txt)
    except Exception as e:
        if DEBUG_LLM_JSON:
            print(f"Parser: json.loads directo falló → {e}", flush=True)

    # Intento recortando desde la primera llave
    start = txt.find("{")
    end = txt.rfind("}")
    if start != -1 and end != -1 and end > start:
        fragment = txt[start:end + 1]
        try:
            return json.loads(fragment)
        except Exception as e:
            if DEBUG_LLM_JSON:
                print(f"Parser: json.loads recortado falló → {e}", flush=True)
                print(f"Fragmento analizado (primeros 500 chars):\n{fragment[:500]}\n--- FIN FRAG ---", flush=True)
            return None

    if DEBUG_LLM_JSON:
        print("Parser: No se encontró JSON válido", flush=True)

    return None


def _norm_col(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.strip().lower().replace("  ", " ")


def _find_col(df: pl.DataFrame, candidates: List[str]) -> Optional[str]:
    norm = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        real = norm.get(_norm_col(cand))
        if real:
            return real
    for key, real in norm.items():
        for cand in candidates:
            if _norm_col(cand) in key:
                return real
    return None


def _sanitize_obs_text(x: Any) -> str:
    s = str(x) if x is not None else ""
    s = s.replace("|", " / ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _to_mapping(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (list, tuple)):
        return {str(i): v for i, v in enumerate(obj)}
    if isinstance(obj, set):
        return {str(i): v for i, v in enumerate(sorted(list(obj), key=str))}
    return {"llm_raw": obj}


def _safe_merge(a: Dict[str, Any], b: Any) -> Dict[str, Any]:
    out = dict(a) if isinstance(a, dict) else {}
    out.update(_to_mapping(b))
    return out


# ========================================
#     ARMADO DE OBSERVACIONES UNIDAS
# ========================================

def _build_observaciones_unidas_para_idx(df: pl.DataFrame, idx: int) -> Tuple[str, int, Dict[str, Any]]:
    row = df.row(idx, named=True)

    col_aviso = _find_col(df, ["numero aviso", "numero_aviso", "aviso"])
    col_sini = _find_col(df, ["numero siniestro", "numero_siniestro", "siniestro"])
    col_placa = _find_col(df, ["placa"])
    col_user = _find_col(df, ["usuario"])
    col_rol = _find_col(df, ["rol analista", "rol_analista", "rol"])
    col_nit = _find_col(df, ["nit taller", "nit_taller"])
    col_nom = _find_col(df, ["nombre taller", "nombre_taller"])

    meta = {
        "nit_taller": _coerce_scalar(row.get(col_nit)) if col_nit else None,
        "nombre_taller": _coerce_scalar(row.get(col_nom)) if col_nom else None,
        "numero_aviso": _coerce_scalar(row.get(col_aviso)) if col_aviso else None,
        "numero_siniestro": _coerce_scalar(row.get(col_sini)) if col_sini else None,
        "placa": _coerce_scalar(row.get(col_placa)) if col_placa else None,
        "usuario": _coerce_scalar(row.get(col_user)) if col_user else None,
        "rol_analista": _coerce_scalar(row.get(col_rol)) if col_rol else None,
    }

    col_obs_unidas = _find_col(df, ["observaciones unidas", "observaciones_unidas"])
    if col_obs_unidas and row.get(col_obs_unidas):
        s = str(_coerce_scalar(row[col_obs_unidas])).strip()
        total = len([p for p in s.split("|") if p.strip()])
        return (s, total, meta)

    # Si no existe columna de observaciones_unidas, armarla desde detalle
    key_col, key_val = None, None
    for c, v in [
        (col_aviso, meta["numero_aviso"]),
        (col_sini, meta["numero_siniestro"]),
        (col_placa, meta["placa"])
    ]:
        if c and v:
            key_col, key_val = c, v
            break

    piezas: List[str] = []
    if key_col:
        sub = df.filter(pl.col(key_col) == key_val)
        col_fecha = _find_col(df, ["fecha observacion", "fecha"])
        col_obs = _find_col(df, ["observacion", "observaciones"])

        if col_obs:
            tmp = []
            for i in range(sub.height):
                r = sub.row(i, named=True)
                f = r.get(col_fecha) if col_fecha else None
                t = _sanitize_obs_text(r.get(col_obs))
                if t:
                    tmp.append((f or "0000-00-00", t))

            tmp.sort(key=lambda x: x[0])
            piezas = [f"{f} - {t}".strip(" -") for f, t in tmp]

    obs_unidas = " | ".join(piezas)
    return (obs_unidas, len(piezas), meta)


# ========================================
#               WORKER
# ========================================

def _worker_un_caso(
    df: pl.DataFrame,
    idx: int,
    system_prompt: str,
    cliente_llm: Any,
    max_retries: int = 2,
    backoff_sec: float = 2.0
) -> Dict[str, Any]:
    try:
        print(f"▶️  Worker start idx={idx + 1}/{df.height}", flush=True)

        obs_unidas, total_evt, meta = _build_observaciones_unidas_para_idx(df, idx)

        if not obs_unidas:
            print(f"⚪ idx={idx + 1}: sin observaciones", flush=True)
            # Alineado con el nuevo esquema (resumen_ejecutivo)
            return {
                **meta,
                "total_eventos": 0,
                "resumen_ejecutivo": "sin datos",
                "alertas_comunicacion_cliente": {},
                "evaluacion_calidad_observaciones": {},
                "evidencias_representativas": [],
                "recomendaciones_priorizadas": [],
                "estadisticas_caso": {},
                "metadata_analisis": {}
            }

        entrada = {**meta, "total_eventos": total_evt, "observaciones_unidas": obs_unidas}
        human_content = f"<ENTRADA>\n{json.dumps(entrada, ensure_ascii=False, indent=2)}\n</ENTRADA>"

        llm_json: Optional[Dict[str, Any]] = None

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"🔁 Reintento {attempt} idx={idx + 1}", flush=True)
                    time.sleep(backoff_sec)

                msgs = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_content)
                ]

                resp = cliente_llm.invoke(msgs)
                raw = resp.content if hasattr(resp, "content") else str(resp)

                if DEBUG_LLM_JSON:
                    print(f"\n📨 RAW LLM idx={idx + 1} (primeros 800 chars):\n{raw[:800]}\n--- FIN RAW ---", flush=True)

                parsed = _parse_llm_json(raw)

                if isinstance(parsed, dict):
                    llm_json = parsed
                    print(f"✅ idx={idx + 1}: LLM OK", flush=True)
                    break
                else:
                    print(f"⚠️ idx={idx + 1}: salida no JSON", flush=True)
                    if DEBUG_LLM_JSON:
                        cleaned = unicodedata.normalize('NFKC', raw)
                        cleaned = re.sub(r'[\x00-\x1f\x7f]', '', cleaned).strip()
                        print(f"🧾 Limpio idx={idx + 1} (primeros 800 chars):\n{cleaned[:800]}\n--- FIN LIMPIO ---", flush=True)

            except Exception as e:
                print(f"❌ idx={idx + 1}: excepción {e}", flush=True)
                if attempt == max_retries:
                    # Fallback de error alineado al nuevo esquema (resumen_ejecutivo)
                    llm_json = {
                        "resumen_ejecutivo": "error en invocacion LLM",
                        "metadata_analisis": {
                            "nivel_confianza": "baja",
                            "factores_confianza": f"Excepcion en LLM: {e}",
                            "limitaciones": ["Fallo en la llamada al modelo de lenguaje"],
                            "total_eventos_procesados": int(total_evt)
                        }
                    }

        # Si tras todos los intentos sigue sin JSON, generamos uno mínimo de error
        if llm_json is None:
            llm_json = {
                "resumen_ejecutivo": "error: el modelo no devolvio JSON valido",
                "metadata_analisis": {
                    "nivel_confianza": "baja",
                    "factores_confianza": "No se pudo parsear la salida del modelo a JSON.",
                    "limitaciones": ["Salida del modelo fuera del formato JSON esperado"],
                    "total_eventos_procesados": int(total_evt)
                }
            }

        print(f"🏁 Worker end idx={idx + 1}", flush=True)
        return _safe_merge(entrada, llm_json)

    except Exception as e:
        print(f"❌ [WORKER idx={idx + 1}] {e}", flush=True)
        return {
            "resumen_ejecutivo": "error inesperado en worker",
            "metadata_analisis": {
                "nivel_confianza": "baja",
                "factores_confianza": f"Excepcion general en worker: {e}",
                "limitaciones": ["Error interno en procesamiento del caso"],
                "total_eventos_procesados": 0
            }
        }


# ========================================
#            ORQUESTADOR
# ========================================

def procesar_calidad_por_caso(
        df_casos: pl.DataFrame,
        prompt_sistema: str,
        cliente_llm: Any,
        max_workers: Optional[int] = 0,
        chunksize: Optional[int] = 0) -> Dict[str, List[Dict[str, Any]]]:

    if df_casos is None or df_casos.height == 0:
        print("⚪ DataFrame vacío", flush=True)
        return {"registros": []}

    print("🚀 Procesando...", flush=True)

    # Si no se pasó prompt_sistema, cargar desde repositorio de prompts
    if not prompt_sistema:
        system_prompt = load_prompts_generales("observaciones_calidad_prompt")
    else:
        system_prompt = prompt_sistema

    workers = max_workers if max_workers and max_workers > 0 else (os.cpu_count() or 1)

    resultados: List[Optional[Dict[str, Any]]] = [None] * df_casos.height
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_worker_un_caso, df_casos, i, system_prompt, cliente_llm): i
            for i in range(df_casos.height)
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                resultados[idx] = future.result()
                done += 1
                print(f"📦 Progreso: {done}/{df_casos.height}", flush=True)
            except Exception as e:
                print(f"❌ Error orquestador idx={idx}: {e}", flush=True)
                resultados[idx] = {
                    "resumen_ejecutivo": "error en orquestador",
                    "metadata_analisis": {
                        "nivel_confianza": "baja",
                        "factores_confianza": f"Excepcion en orquestador: {e}",
                        "limitaciones": ["Error interno en procesamiento global"],
                        "total_eventos_procesados": 0
                    }
                }

    print("✅ Orquestación completada", flush=True)
    return {"registros": [r for r in resultados if r is not None]}
