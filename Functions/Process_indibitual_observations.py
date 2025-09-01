import os
import re
import ast
import json
import time
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
    - "['ABC']"   -> "ABC"
    - [123]       -> 123
    - pl.Series   -> primer item
    - str         -> strip()
    """
    try:
        if isinstance(v, pl.Series):
            return v.item() if len(v) > 0 else None
        if isinstance(v, (list, tuple)):
            return v[0] if v else None
        if isinstance(v, str):
            s = v.strip()
            # Si viene como "['ABC']" o "[123]" en string, intenta evaluarlo
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
    Intenta parsear JSON robusto desde una respuesta del LLM:
    - Elimina fences ```json ... ```
    - Si falla, intenta extraer el primer bloque {...}
    """
    if not isinstance(raw, str):
        return None
    txt = raw.strip()

    # 1) Remover fences tipo ```json ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", txt, re.IGNORECASE)
    if m:
        txt = m.group(1).strip()

    # 2) Intento directo
    try:
        return json.loads(txt)
    except Exception:
        pass

    # 3) Extraer primer bloque {...}
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
    cliente_llm: Any
) -> Dict[str, Any]:
    """
    Procesa 1 observación con el LLM y devuelve SIEMPRE:
    {
      numero_aviso, numero_siniestro, placa, fecha_observacion,
      usuario, rol_analista, observacion,
      clasificacion, explicacion, confianza
    }
    Imprime el JSON final resultante por registro.
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
            print(f"🧾 Final [{idx+1}]: {json.dumps(base_out, ensure_ascii=False)}")
            return base_out

        # --- Campos base SIEMPRE presentes (limpios) ---
        base_out = {
            "numero_aviso": _coerce_scalar(registro.get("NUMERO AVISO", "")),
            "numero_siniestro": _coerce_scalar(registro.get("NUMERO SINIESTRO", "")),
            "placa": _coerce_scalar(registro.get("PLACA", "")),
            "fecha_observacion": _coerce_scalar(registro.get("FECHA OBSERVACION", "")),
            "usuario": _coerce_scalar(registro.get("USUARIO", "")),
            "rol_analista": _coerce_scalar(registro.get("ROL ANALISTA", "")),
            "observacion": _coerce_scalar(registro.get("OBSERVACION", "")),
        }

        # --- Prompt del sistema ---
        system_prompt = load_prompts_generales("observaciones_clasificacion_prompt")
        if not system_prompt:
            base_out.update({
                "clasificacion": "sin_clasificar",
                "explicacion": "prompt no encontrado",
                "confianza": 0.0
            })
            print(f"✅ Procesado registro {idx+1}")
            print(f"🧾 Final [{idx+1}]: {json.dumps(base_out, ensure_ascii=False)}")
            return base_out

        # --- Mensajes y llamada al LLM ---
        messages_for_llm = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Observación: {base_out['observacion']}")
        ]
        response_obj = cliente_llm.invoke(messages_for_llm)
        raw_content = response_obj.content if hasattr(response_obj, "content") else str(response_obj)

        # --- Parseo robusto ---
        llm_json = _parse_llm_json(raw_content)

        # --- Validación & Fallbacks ---
        ALLOWED = {"comunicacion_cliente", "cambio_estado", "sin_cambio", "sin_clasificar"}

        if llm_json is None:
            # Falló el parseo → fallback
            clasificacion = "sin_clasificar"
            explicacion = "no se pudo parsear salida LLM"
            confianza = 0.0
        else:
            # Usar 'or' para cubrir None o cadenas vacías
            clasificacion = (llm_json.get("clasificacion") or "").strip()
            explicacion = (llm_json.get("explicacion") or "").strip() or "sin explicación"
            confianza = llm_json.get("confianza", 0.0)

            # Validar clasificacion
            if clasificacion not in ALLOWED:
                clasificacion = "sin_clasificar"
                if not explicacion or explicacion == "sin explicación":
                    explicacion = "clasificación ausente o inválida"

            # Normalizar confianza a float en [0, 1]
            try:
                confianza = float(confianza)
            except Exception:
                confianza = 0.0
            if confianza < 0.0:
                confianza = 0.0
            elif confianza > 1.0:
                confianza = 1.0

        # --- Fusión final ---
        base_out.update({
            "clasificacion": clasificacion,
            "explicacion": explicacion,
            "confianza": confianza
        })

        print(f"✅ Procesado registro {idx+1}")
        print(f"🧾 Final [{idx+1}]: {json.dumps(base_out, ensure_ascii=False)}")
        return base_out

    except Exception as e:
        print(f"❌ Error en registro {idx+1}: {e}")
        base_out = {
            "numero_aviso": None, "numero_siniestro": None, "placa": None,
            "fecha_observacion": None, "usuario": None, "rol_analista": None,
            "observacion": None, "clasificacion": "sin_clasificar",
            "explicacion": f"error en worker: {e}", "confianza": 0.0, "idx": idx
        }
        print(f"🧾 Final [{idx+1}]: {json.dumps(base_out, ensure_ascii=False)}")
        return base_out


# -------------------------------------------------------------------
# Orquestación (paralelo “cola continua” + tiempos por lote lógico)
# -------------------------------------------------------------------

def procesar_observacion_individual(
    df_observacion: pl.DataFrame,
    prompt_sistema: str,   # no usado; compatibilidad
    cliente_llm: Any,
    max_workers: Optional[int] = 0,
    chunksize: int = 1000,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Procesa todas las observaciones en paralelo con un pool de hilos:
    - workers = núcleos si no se especifica
    - el pool toma el siguiente índice en cuanto un hilo queda libre
    - imprime tiempo por 'lote lógico' cada `chunksize` completados
    - devuelve resultados en el MISMO orden del DataFrame
    """
    if df_observacion is None or df_observacion.height == 0:
        return {"registros": []}

    workers = max_workers if (max_workers and max_workers > 0) else (os.cpu_count() or 1)
    n = df_observacion.height
    print(f"⚙️ Procesando {n} observaciones con {workers} hilos (alimentación continua)...")

    # Prealocar para mantener orden
    resultados: List[Optional[Dict[str, Any]]] = [None] * n

    start_global = time.perf_counter()
    completed = 0
    chunk_start = start_global
    next_mark = chunksize if chunksize > 0 else n

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(_worker_un_registro, df_observacion, i, cliente_llm): i
            for i in range(n)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                res = future.result()
            except Exception as e:
                print(f"❌ Futuro falló en idx {idx}: {e}")
                res = {
                    "numero_aviso": None, "numero_siniestro": None, "placa": None,
                    "fecha_observacion": None, "usuario": None, "rol_analista": None,
                    "observacion": None, "clasificacion": "sin_clasificar",
                    "explicacion": f"excepción en futuro: {e}", "confianza": 0.0, "idx": idx
                }

            resultados[idx] = res
            completed += 1

            # Corte por lote lógico (cada 'chunksize' completados o al final)
            if completed >= next_mark or completed == n:
                elapsed = time.perf_counter() - chunk_start
                if chunksize > 0:
                    lote_fin = completed
                    lote_ini = max(1, lote_fin - (chunksize - 1))
                else:
                    lote_ini, lote_fin = 1, completed
                print(f"⏱️ Lote {lote_ini}-{lote_fin} completado en {elapsed:.2f} s (progreso {completed}/{n})")
                chunk_start = time.perf_counter()
                next_mark += chunksize

    total_elapsed = time.perf_counter() - start_global
    print(f"✅ Finalizado. Registros: {n} — Tiempo total: {total_elapsed:.2f} s")

    # Compactar (ya están en orden)
    return {"registros": [r for r in resultados if r is not None]}
