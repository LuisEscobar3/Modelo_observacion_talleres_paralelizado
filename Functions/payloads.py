# App/Functions/payloads.py
from __future__ import annotations
import polars as pl
from typing import Dict, Any

import polars as pl
from typing import Dict, Any


def build_json_para_n8n_registro(df: pl.DataFrame, idx: int = 0) -> Dict[str, Any]:
    """
    Devuelve un único registro del DataFrame en formato diccionario.
    Por defecto toma el primero (idx=0).
    """
    columnas = [
        "NIT TALLER",
        "NOMBRE TALLER",
        "NUMERO AVISO",
        "NUMERO SINIESTRO",
        "PLACA",
        "FECHA OBSERVACION",
        "USUARIO",
        "ROL ANALISTA",
        "OBSERVACION",
    ]

    df_sel = df.select(columnas)

    if 0 <= idx < df_sel.height:
        registro = df_sel[idx].to_dict(as_series=False)
        return {"registro": registro}
    else:
        return {"registro": None}
