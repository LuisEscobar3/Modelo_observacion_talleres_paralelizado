import os
import re
import ast
import json
import time
import math
import unicodedata
import traceback  # <-- 1. IMPORTADO EL MÓDULO

from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
from langchain.schema import SystemMessage, HumanMessage

# Importa tu helper para construir el registro base
from .payloads import build_json_para_n8n_registro
# Carga del prompt desde tu servicio real
from services.miscelaneous import load_prompts_generales


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _coerce_scalar(v: Any) -> Any:
    """
    Convierte Series/list/tuple/str-list -> escalar limpio.
    """
    # (El código de esta función no se modifica)
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
    Intenta parsear un JSON robusto desde una respuesta de un LLM.
    """
    # (El código de esta función no se modifica)
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
        candidate = txt[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


# -------------------------------------------------------------------
# Worker (procesa 1 registro)
# -------------------------------------------------------------------

def _worker_un_registro(
        df: pl.DataFrame,
        idx: int,
        cliente_llm: Any,
        max_retries: int = 2,
        backoff_sec: float = 2.0,
) -> Dict[str, Any]:
    """
    Procesa 1 observación con el LLM y devuelve el resultado.
    """
    try:
        payload = build_json_para_n8n_registro(df, idx)
        registro = payload.get("registro")
        if not registro:
            print(f"⚠️ Registro vacío en índice {idx}")
            base_out = {
                "numero_aviso": None, "numero_siniestro": None, "placa": None,
                "fecha_observacion": None, "usuario": None, "rol_analista": None,
                "observacion": None, "clasificacion": "sin_clasificar",
                "explicacion": "registro vacío", "confianza": 0.0, "idx": idx
            }
            print(f"🧾 Final [{idx + 1}]: {json.dumps(base_out, ensure_ascii=False)}")
            return base_out

        base_out = {
            "numero_aviso": _coerce_scalar(registro.get("NUMERO AVISO", "")),
            "numero_siniestro": _coerce_scalar(registro.get("NUMERO SINIESTRO", "")),
            "placa": _coerce_scalar(registro.get("PLACA", "")),
            "fecha_observacion": _coerce_scalar(registro.get("FECHA OBSERVACION", "")),
            "usuario": _coerce_scalar(registro.get("USUARIO", "")),
            "rol_analista": _coerce_scalar(registro.get("ROL ANALISTA", "")),
            "observacion": _coerce_scalar(registro.get("OBSERVACION", "")),
        }

        system_prompt = load_prompts_generales("observaciones_clasificacion_prompt")
        if not system_prompt:
            base_out.update({"clasificacion": "sin_clasificar", "explicacion": "prompt no encontrado", "confianza": 0.0})
            print(f"✅ Procesado registro {idx + 1}")
            print(f"🧾 Final [{idx + 1}]: {json.dumps(base_out, ensure_ascii=False)}")
            return base_out

        ALLOWED = {"comunicacion_cliente", "cambio_estado", "sin_cambio", "sin_clasificar", "comunicacion_interna"}
        intento = 0
        clasificacion, explicacion, confianza = "sin_clasificar", "no se pudo parsear salida LLM", 0.0
        last_error_reason = ""

        while intento <= max_retries:
            intento += 1
            raw_content = None
            try:
                messages_for_llm = [SystemMessage(content=system_prompt), HumanMessage(content=f"Observación: {base_out['observacion']}")]
                print(f"🚀 LLM idx {idx + 1} intento {intento}/{max_retries + 1}")
                response_obj = cliente_llm.invoke(messages_for_llm)
                raw_content = response_obj.content if hasattr(response_obj, "content") else str(response_obj)

                llm_json = _parse_llm_json(raw_content)
                if llm_json is None:
                    last_error_reason = "no se pudo parsear JSON"
                    print(f"📥 LLM RAW [{idx + 1}, intento {intento}] (error): {raw_content}")
                    raise ValueError(last_error_reason)

                c = (llm_json.get("clasificacion") or "").strip()
                e = (llm_json.get("explicacion") or "").strip() or "sin explicación"
                conf_val = llm_json.get("confianza", 0.0)

                try:
                    conf_f = float(conf_val)
                except Exception:
                    last_error_reason = f"confianza inválida: {conf_val!r}"
                    print(f"📥 LLM RAW [{idx + 1}, intento {intento}] (error): {raw_content}")
                    raise ValueError(last_error_reason)

                if c not in ALLOWED:
                    last_error_reason = f"clasificacion inválida: {c!r}"
                    print(f"📥 LLM RAW [{idx + 1}, intento {intento}] (error): {raw_content}")
                    raise ValueError(last_error_reason)

                clasificacion, explicacion, confianza = c, e, max(0.0, min(1.0, conf_f))
                break

            except Exception as e_llm:
                if not last_error_reason: last_error_reason = str(e_llm)
                if intento <= max_retries:
                    wait_s = backoff_sec * (2 ** (intento - 1))
                    print(f"⏳ JSON inválido/erróneo ({last_error_reason}). Reintentando en {wait_s:.1f}s...")
                    time.sleep(wait_s)
                else:
                    print(f"⚠️ Reintentos agotados en idx {idx + 1}. Motivo: {last_error_reason}")

        base_out.update({"clasificacion": clasificacion, "explicacion": explicacion, "confianza": confianza})
        print(f"✅ Procesado registro {idx + 1}")
        print(f"🧾 Final [{idx + 1}]: {json.dumps(base_out, ensure_ascii=False)}")
        return base_out

    except Exception as e:
        # --- 2. CAPTURA DE ERROR MEJORADA EN EL WORKER ---
        print(f"❌ Error catastrófico en registro {idx + 1}: {e}")
        print("--- TRACEBACK COMPLETO (WORKER) ---")
        print(traceback.format_exc()) # Imprime el archivo y la línea exacta del error.
        print("---------------------------------")
        base_out = {
            "numero_aviso": None, "numero_siniestro": None, "placa": None,
            "fecha_observacion": None, "usuario": None, "rol_analista": None,
            "observacion": None, "clasificacion": "sin_clasificar",
            "explicacion": f"error en worker: {e}", "confianza": 0.0, "idx": idx
        }
        print(f"🧾 Final [{idx + 1}]: {json.dumps(base_out, ensure_ascii=False)}")
        return base_out


# -------------------------------------------------------------------
# Orquestación
# -------------------------------------------------------------------

def procesar_observacion_individual(
        df_observacion: pl.DataFrame,
        prompt_sistema: str,
        cliente_llm: Any,
        max_workers: Optional[int] = 0,
        chunksize: Optional[int] = 0,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Procesa todas las observaciones en paralelo.
    """
    if df_observacion is None or df_observacion.height == 0:
        return {"registros": []}

    workers = max_workers if (max_workers and max_workers > 0) else (os.cpu_count() or 1)
    n = df_observacion.height

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
        future_to_idx = {executor.submit(_worker_un_registro, df_observacion, i, cliente_llm): i for i in range(n)}

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                res = future.result()
            except Exception as e:
                # --- 3. CAPTURA DE ERROR MEJORADA EN EL ORQUESTADOR ---
                print(f"❌ Futuro falló catastróficamente en idx {idx}: {e}")
                print("--- TRACEBACK COMPLETO (ORQUESTADOR) ---")
                print(traceback.format_exc()) # Imprime si el worker tuvo un error no capturado.
                print("---------------------------------------")
                res = {
                    "numero_aviso": None, "numero_siniestro": None, "placa": None,
                    "fecha_observacion": None, "usuario": None, "rol_analista": None,
                    "observacion": None, "clasificacion": "sin_clasificar",
                    "explicacion": f"excepción en futuro: {e}", "confianza": 0.0, "idx": idx
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