#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Any, Dict, List

import polars as pl

# ====== Configura aquí tus rutas ======
INPUT_PATH  = Path(r"C:\Users\1032497498\PycharmProjects\Modelo_observacion_talleres_paralelizado\Functions\auditoria_calidad.json")
OUTPUT_NAME = "Observaciones_por_caso.csv"   # si no es absoluto, se guarda junto al JSON
LIST_SEP    = " | "                          # separador para listas


def _join_if_list(v: Any, sep: str) -> Any:
    """Si v es lista/tupla, la une con sep. Si es dict, lo serializa. Si es None, devuelve ''. Si es str/num, lo deja."""
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return sep.join("" if x is None else str(x) for x in v)
    if isinstance(v, dict):
        try:
            return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(v)
    if isinstance(v, str):
        # Evitar que Excel interprete fórmulas si empiezan con '='
        return v.strip().lstrip("=")
    return v


def convert_calidad_json_to_csv(input_path: Path, output_path: Path, list_sep: str = " | ") -> None:
    """
    Convierte el JSON de auditoría de calidad (clave 'registros') a CSV.
    Aplana 'evidencias' y 'observaciones' si son listas.
    """
    if not input_path.is_file():
        print(f"❌ No existe el archivo: {input_path}")
        return

    data = json.loads(input_path.read_text(encoding="utf-8"))
    registros: List[Dict[str, Any]] = []
    if isinstance(data, dict) and isinstance(data.get("registros"), list):
        registros = data["registros"]
    elif isinstance(data, list):
        registros = data
    else:
        print("⚠️ JSON sin 'registros' válidos. Se creará CSV vacío.")
        pl.DataFrame({"info": ["sin registros"]}).write_csv(output_path)
        print(f"📄 CSV creado: {output_path}")
        return

    if not registros:
        print("⚠️ JSON sin registros. CSV vacío.")
        pl.DataFrame({"info": ["sin registros"]}).write_csv(output_path)
        print(f"📄 CSV creado: {output_path}")
        return

    # Normalizar cada registro de forma simple (Python) antes de crear el DataFrame
    norm_rows: List[Dict[str, Any]] = []
    for r in registros:
        if not isinstance(r, dict):
            norm_rows.append({"value": _join_if_list(r, list_sep)})
            continue

        # Copia superficial y aplana campos de lista conocidos
        out = dict(r)
        out["evidencias"]    = _join_if_list(out.get("evidencias"), list_sep)
        out["observaciones"] = _join_if_list(out.get("observaciones"), list_sep)
        # Si quieres aplanar otras listas futuras, añade más líneas como las de arriba
        norm_rows.append(out)

    df = pl.DataFrame(norm_rows)

    # (Opcional) Forzar algunos tipos a texto para evitar notación científica / ceros perdidos:
    # for col in ("numero_aviso", "numero_siniestro", "placa"):
    #     if col in df.columns:
    #         df = df.with_columns(pl.col(col).cast(pl.Utf8))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)
    print(f"✅ {len(df)} filas exportadas a: {output_path}")


def main():
    out_path = Path(OUTPUT_NAME)
    if not out_path.is_absolute():
        out_path = INPUT_PATH.parent / OUTPUT_NAME
    convert_calidad_json_to_csv(INPUT_PATH, out_path, list_sep=LIST_SEP)


if __name__ == "__main__":
    main()
