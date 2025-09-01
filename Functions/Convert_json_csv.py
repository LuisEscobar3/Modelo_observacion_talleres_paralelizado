#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Any, Dict, List

import polars as pl

# === Configura aquí tus rutas ===
INPUT_PATH  = Path(r"C:\Users\1032497498\PycharmProjects\Modelo_observacion_talleres_paralelizado\Functions\auditoria_calidad.json")
OUTPUT_NAME = "Observaciones_por_caso.csv"  # si no es absoluto, se guarda junto al JSON
LIST_SEP    = " | "  # separador para listas (evidencias/observaciones)


def _load_registros(path_json: Path) -> List[Dict[str, Any]]:
    """Lee el JSON y devuelve la lista bajo 'registros' (o lista directa)."""
    data = json.loads(path_json.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("registros"), list):
        return data["registros"]
    if isinstance(data, list):
        return data
    return []


def convert_calidad_json_to_csv(input_path: Path, output_path: Path, list_sep: str = " | ") -> None:
    """
    Convierte el JSON de auditoría de calidad (clave 'registros') a CSV.
    - Aplana listas ('evidencias', 'observaciones') con list_sep.
    """
    if not input_path.is_file():
        print(f"❌ No existe el archivo: {input_path}")
        return

    registros = _load_registros(input_path)
    if not registros:
        print("⚠️ JSON sin registros. CSV vacío con una fila informativa.")
        pl.DataFrame({"info": ["sin registros"]}).write_csv(output_path)
        print(f"📄 CSV creado: {output_path}")
        return

    df = pl.DataFrame(registros)

    exprs = []
    if "evidencias" in df.columns:
        exprs.append(
            pl.when(pl.col("evidencias").is_not_null())
              .then(pl.col("evidencias").cast(pl.List(pl.Utf8)).list.join(list_sep))
              .otherwise("")
              .alias("evidencias")
        )
    if "observaciones" in df.columns:
        exprs.append(
            pl.when(pl.col("observaciones").is_not_null())
              .then(pl.col("observaciones").cast(pl.List(pl.Utf8)).list.join(list_sep))
              .otherwise("")
              .alias("observaciones")
        )

    if exprs:
        df = df.with_columns(exprs)

    # (Opcional) Forzar estos como texto si quieres evitar notaciones científicas / ceros perdidos:
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
