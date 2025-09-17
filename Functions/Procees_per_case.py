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
    if not isinstance(raw, str):
        return None
    txt = unicodedata.normalize('NFKC', raw)
    txt = re.sub(r'[\x00-\x1f\x7f]', '', txt).strip()

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


def _norm_col(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.strip().lower().replace("  ", " ")


def _find_col(df: pl.DataFrame, candidates: List[str]) -> Optional[str]:
    norm_map = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        real = norm_map.get(_norm_col(cand))
        if real:
            return real
    for key_norm, real in norm_map.items():
        for cand in candidates:
            if _norm_col(cand) in key_norm:
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
        return {str(i): v for i, v in enumerate(sorted(list(obj), key=lambda x: str(x)))}
    return {"llm_raw": obj}


def _safe_merge(a: Dict[str, Any], b: Any) -> Dict[str, Any]:
    out = dict(a) if isinstance(a, dict) else {}
    out.update(_to_mapping(b))
    return out


# =========================
# Armado de "observaciones_unidas"
# =========================

def _build_observaciones_unidas_para_idx(df: pl.DataFrame, idx: int) -> Tuple[str, int, Dict[str, Any]]:
    row = df.row(idx, named=True)

    col_aviso = _find_col(df, ["NUMERO AVISO", "numero_aviso", "aviso"])
    col_sini = _find_col(df, ["NUMERO SINIESTRO", "numero_siniestro", "siniestro"])
    col_placa = _find_col(df, ["PLACA", "placa"])
    col_user = _find_col(df, ["USUARIO", "usuario"])
    col_rol = _find_col(df, ["ROL ANALISTA", "rol_analista", "rol"])
    col_nit_taller = _find_col(df, ["NIT TALLER", "nit_taller", "nit taller"])
    col_nombre_taller = _find_col(df, ["NOMBRE TALLER", "nombre_taller", "nombre taller"])

    meta = {
        "nit_taller": _coerce_scalar(row.get(col_nit_taller)) if col_nit_taller else None,
        "nombre_taller": _coerce_scalar(row.get(col_nombre_taller)) if col_nombre_taller else None,
        "numero_aviso": _coerce_scalar(row.get(col_aviso)) if col_aviso else None,
        "numero_siniestro": _coerce_scalar(row.get(col_sini)) if col_sini else None,
        "placa": _coerce_scalar(row.get(col_placa)) if col_placa else None,
        "usuario": _coerce_scalar(row.get(col_user)) if col_user else None,
        "rol_analista": _coerce_scalar(row.get(col_rol)) if col_rol else None,
    }

    col_obs_unidas = _find_col(df, ["observaciones_unidas", "OBSERVACIONES UNIDAS"])
    if col_obs_unidas and row.get(col_obs_unidas):
        s = str(_coerce_scalar(row[col_obs_unidas])).strip()
        total = len([p for p in s.split("|") if p.strip()]) if s else 0
        return (s, total, meta)

    key_col, key_val = None, None
    for c, v in [(col_aviso, meta["numero_aviso"]), (col_sini, meta["numero_siniestro"]), (col_placa, meta["placa"])]:
        if c and v not in (None, ""):
            key_col, key_val = c, v
            break

    piezas = []
    if key_col:
        sub = df.filter(pl.col(key_col) == key_val)
        col_fecha = _find_col(df, ["FECHA OBSERVACION", "fecha_observacion", "FECHA"])
        col_obs = _find_col(df, ["OBSERVACION", "observacion", "OBSERVACIONES"])
        if sub.height > 0 and col_obs:
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


# =========================
# Worker
# =========================

def _worker_un_caso(df: pl.DataFrame, idx: int, cliente_llm: Any,
                    max_retries: int = 2, backoff_sec: float = 2.0) -> Dict[str, Any]:
    try:
        print(f"▶️  Worker start idx={idx + 1}/{df.height}", flush=True)

        obs_unidas, total_evt, meta = _build_observaciones_unidas_para_idx(df, idx)

        if not obs_unidas:
            print(f"⚪ idx={idx + 1}: sin observaciones", flush=True)
            return {**meta, "resumen_general": "sin datos", "total_eventos": 0}

        system_prompt = load_prompts_generales("observaciones_calidad_prompt")
        entrada = {**meta, "total_eventos": total_evt, "observaciones_unidas": obs_unidas}
        human_content = f"<ENTRADA>\n{json.dumps(entrada, ensure_ascii=False, indent=2)}\n</ENTRADA>"

        llm_json = None
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"🔁 Reintento {attempt} idx={idx + 1}", flush=True)
                    time.sleep(backoff_sec)

                msgs = [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
                resp = cliente_llm.invoke(msgs)
                parsed = _parse_llm_json(resp.content if hasattr(resp, "content") else str(resp))
                if isinstance(parsed, dict):
                    llm_json = parsed
                    print(f"✅ idx={idx + 1}: LLM OK", flush=True)
                    break
                else:
                    print(f"⚠️ idx={idx + 1}: salida no JSON", flush=True)
            except Exception as e:
                print(f"❌ idx={idx + 1}: excepción {e}", flush=True)
                if attempt == max_retries:
                    llm_json = {"resumen_general": "error", "observaciones": [str(e)]}

        print(f"🏁 Worker end idx={idx + 1}", flush=True)
        return _safe_merge(entrada, llm_json)

    except Exception as e:
        print(f"❌ [WORKER idx={idx + 1}] {e}", flush=True)
        return {"resumen_general": "error", "observaciones": [str(e)]}


# =========================
# Orquestación
# =========================

def procesar_calidad_por_caso(
        df_casos: pl.DataFrame,
        prompt_sistema: str,
        cliente_llm: Any,
        max_workers: Optional[int] = 0,
        chunksize: Optional[int] = 0) -> Dict[str, List[Dict[str, Any]]]:
    if df_casos is None or df_casos.height == 0:
        print("⚪ DataFrame vacío", flush=True)
        return {"registros": []}

    print("🚀 Procesando", flush=True)
    workers = max_workers if max_workers and max_workers > 0 else (os.cpu_count() or 1)

    resultados: List[Optional[Dict[str, Any]]] = [None] * df_casos.height
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {executor.submit(_worker_un_caso, df_casos, i, cliente_llm): i for i in range(df_casos.height)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                resultados[idx] = future.result()
                done += 1
                print(f"📦 Progreso: {done}/{df_casos.height}", flush=True)
            except Exception as e:
                print(f"❌ [ORQUESTADOR idx={idx}] {e}", flush=True)
                resultados[idx] = {"resumen_general": "error", "observaciones": [str(e)]}

    print("✅ Orquestación completada", flush=True)
    return {"registros": [r for r in resultados if r is not None]}
