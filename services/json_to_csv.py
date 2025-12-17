import json
import csv
import os
from typing import List, Dict, Any

# Orden preferido de columnas (incluye los nuevos campos)
PREFERRED_HEADERS = [
    "NIT TALLER",
    "NOMBRE TALLER",
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
    # 👇 nuevos campos solicitados
    "explicacion_clasificacion",
    "confianza_clasificacion",
    "calidad_comunicativa_score",
    "explicacion_calidad",
    "elementos_faltantes",
]


def _string_sanitize(s: str) -> str:
    """Quita tabs, múltiple espacio y saltos; conserva un espacio simple."""
    return " ".join(s.split())


def _clean_value(v: Any, key: str = "") -> Any:
    """
    Limpia valores para CSV:
    - str: sanea espacios
    - list: une por ' | ' (para 'elementos_faltantes'); si no es esa clave, serializa JSON compacto
    - dict: serializa JSON compacto
    - otros tipos: tal cual
    """
    if isinstance(v, str):
        return _string_sanitize(v)

    if isinstance(v, list):
        # Caso especial: elementos_faltantes como 'item1 | item2 | item3'
        if key == "elementos_faltantes":
            items = []
            for it in v:
                if it is None:
                    continue
                items.append(_string_sanitize(str(it)))
            return " | ".join([x for x in items if x])
        # Para otras listas, serializa JSON compacto
        try:
            return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return " | ".join(_string_sanitize(str(it)) for it in v)

    if isinstance(v, dict):
        try:
            return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            # Fallback tosco si hubiese tipos no serializables
            return _string_sanitize(str(v))

    return v


def _resolve_headers(rows: List[Dict[str, Any]]) -> List[str]:
    """Devuelve los headers en orden preferido + cualquier extra al final."""
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    # Mantener el orden preferido si existe en los datos
    headers = [h for h in PREFERRED_HEADERS if h in all_keys]
    # Agregar claves extra (no incluidas en preferidos), ordenadas
    extras = sorted(k for k in all_keys if k not in headers)
    return headers + extras


def json_registros_to_csv(json_path: str, csv_path: str) -> None:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No se encontró el archivo: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "registros" not in data:
        raise ValueError("El JSON debe ser un objeto con la clave 'registros' que contenga una lista.")

    registros = data.get("registros", [])
    if not isinstance(registros, list) or not registros:
        raise ValueError("La clave 'registros' debe ser una lista no vacía.")

    # Limpieza básica de valores (aware de clave para 'elementos_faltantes')
    cleaned: List[Dict[str, Any]] = []
    for row in registros:
        new_row: Dict[str, Any] = {}
        for k, v in row.items():
            new_row[k] = _clean_value(v, key=k)
        cleaned.append(new_row)

    headers = _resolve_headers(cleaned)

    # Escribir CSV (utf-8-sig para Excel)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in cleaned:
            # Asegurar que todas las claves existan (si faltan, quedan vacías)
            full_row = {h: row.get(h, "") for h in headers}
            writer.writerow(full_row)

    print(f"✅ CSV generado: {csv_path}")


# -----------------------------
# Ejemplo de uso
# -----------------------------
if __name__ == "__main__":
    # JSON de entrada y CSV de salida
    json_path = r"C:\Users\1032497498\PycharmProjects\Modelo_observacion_talleres_paralelizado\salida_registros.json"
    csv_path = "Clasificacion_onservacion.csv"  # (usa tu ruta/nombre preferido)

    json_registros_to_csv(json_path, csv_path)
