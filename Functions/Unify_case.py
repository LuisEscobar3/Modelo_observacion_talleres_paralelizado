import json
import sys

import dotenv
import os
import polars as pl
from pathlib import Path
from langchain.globals import set_debug
from Procees_per_case import procesar_calidad_por_caso
from services.llm_manager import load_llms


# --- Helpers estilo JS ---

def _clean_expr(col: str) -> pl.Expr:
    # v == null ? "" : String(v).trim().replace(/^=/, "")
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .fill_null("")
        .str.strip_chars()
        .str.replace(r"^=", "", literal=False)
    )


def _evento_fmt_expr() -> pl.Expr:
    # "YYYY-MM-DD - TEXTO [clasificacion]" (omitiendo tag si no hay clasif.)
    return (
        pl.when(pl.col("clasificacion").is_null() | (pl.col("clasificacion") == ""))
        .then(pl.lit(""))
        .otherwise(pl.lit(" [") + pl.col("clasificacion") + pl.lit("]"))
        .alias("tag_clasif")
    )


def _non_empty_event_expr() -> pl.Expr:
    # Filtrado: si texto vacío -> None (para poder drop_nulls en el groupby)
    return (
        pl.when(pl.col("observacion").str.strip_chars() == "")
        .then(pl.lit(None))
        .otherwise(pl.col("evento_fmt"))
        .alias("evento_fmt_nonempty")
    )


# --- Pipeline principal ---

def cargar_json_a_polars(path_json: str | Path) -> pl.DataFrame:
    """Lee el archivo y devuelve un DataFrame con los registros."""
    data = json.loads(Path(path_json).read_text(encoding="utf-8"))
    registros = data.get("registros", [])
    return pl.DataFrame(registros)


import polars as pl


def unir_observaciones_por_caso(df: pl.DataFrame, separador: str = " | ") -> pl.DataFrame:
    """
    1 fila por (numero_siniestro, placa) con eventos unidos:
    "YYYY-MM-DD - OBSERVACION [clasificacion]" en orden de fecha asc.
    """
    # 1) Limpiar textos (trim y quitar '=' inicial si viene de Excel)
    cols_texto = [
        "numero_aviso", "numero_siniestro", "placa", "usuario",
        "rol_analista", "observacion", "clasificacion", "fecha_observacion"
    ]
    df1 = df.with_columns([
        pl.col(c).cast(pl.Utf8).fill_null("").str.strip_chars().str.replace(r"^=", "", literal=False).alias(c)
        for c in cols_texto
    ])

    # 2) Normalizar fecha a texto y colapsar espacios -> fecha_obs_norm
    df2 = df1.with_columns(
        pl.col("fecha_observacion")
        .cast(pl.Utf8).fill_null("").str.strip_chars()
        .str.replace_all(r"\s+", " ", literal=False)
        .alias("fecha_obs_norm")
    )

    # 3) Parsear fecha a datetime en varios formatos -> fecha_dt
    df3 = df2.with_columns(
        pl.coalesce([
            pl.col("fecha_obs_norm").str.strptime(pl.Datetime, format="%Y/%m/%d %I:%M:%S %p", strict=False),
            pl.col("fecha_obs_norm").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False),
            pl.col("fecha_obs_norm").str.strptime(pl.Datetime, format="%Y-%m-%d", strict=False),
        ]).alias("fecha_dt")
    )

    # 4) Formato final de fecha printable -> fecha_fmt
    df4 = df3.with_columns(
        pl.when(pl.col("fecha_dt").is_not_null())
        .then(pl.col("fecha_dt").dt.strftime("%Y-%m-%d"))
        .otherwise(pl.col("fecha_obs_norm"))
        .alias("fecha_fmt")
    )

    # 5a) Tag opcional de clasificación: " [clasificacion]"
    df5a = df4.with_columns(
        pl.when(pl.col("clasificacion").is_null() | (pl.col("clasificacion") == ""))
        .then(pl.lit(""))
        .otherwise(pl.lit(" [") + pl.col("clasificacion") + pl.lit("]"))
        .alias("tag_clasif")
    )

    # 5b) Texto del evento: "YYYY-MM-DD - OBSERVACION[ tag ]"
    df5b = df5a.with_columns(
        (pl.col("fecha_fmt") + pl.lit(" - ") + pl.col("observacion").str.strip_chars() + pl.col("tag_clasif"))
        .alias("evento_fmt")
    )

    # 5c) Versión filtrable (None si observación vacía)
    df5 = df5b.with_columns(
        pl.when(pl.col("observacion").str.strip_chars() == "")
        .then(pl.lit(None))
        .otherwise(pl.col("evento_fmt"))
        .alias("evento_fmt_nonempty")
    )

    # 6) Ordenar por caso y fecha (usa fecha_dt; si falta, fecha_obs_norm)
    df_sorted = df5.sort(["numero_siniestro", "placa", "fecha_dt", "fecha_obs_norm"])

    # 7) Agrupar por caso y unir eventos no vacíos
    out = (
        df_sorted
        .group_by(["numero_siniestro", "placa"], maintain_order=True)
        .agg([
            pl.col("numero_aviso").first().alias("numero_aviso"),
            pl.col("usuario").first().alias("usuario"),
            pl.col("rol_analista").first().alias("rol_analista"),
            pl.len().alias("total_eventos"),
            # recolectar eventos a lista (sin nulos) y luego unir
            pl.col("evento_fmt_nonempty").drop_nulls().implode().alias("eventos_list"),
        ])
        .with_columns(
            pl.col("eventos_list").list.join(separador).fill_null("").alias("observaciones_unidas")
        )
        .select([
            "numero_aviso",
            "numero_siniestro",
            "placa",
            "usuario",
            "rol_analista",
            "total_eventos",
            "observaciones_unidas",
        ])
    )
    return out


# --- Ejemplo de uso ---
if __name__ == "__main__":
    ruta = r"C:\Users\1032497498\PycharmProjects\Modelo_observacion_talleres_paralelizado\salida_registros.json"
    dotenv.load_dotenv()
    set_debug(False)
    os.environ["APP_ENV"] = os.environ.get("APP_ENV", "sbx")
    llms = load_llms()
    gemini = llms["gemini_pro"]
    df = cargar_json_a_polars(ruta)
    df_casos = unir_observaciones_por_caso(df, separador=" | ")  # o "\n"
    print(df_casos.head(10))
    nfilas = len(df_casos)
    print("dsadfaFSAFAFS")
    resultado_json = procesar_calidad_por_caso(
        df_casos=df_casos,
        prompt_sistema="",  # no se usa aquí, se deja por compatibilidad
        cliente_llm=gemini,  # tu cliente LLM
        max_workers=0,  # 0 => usa todos los CPUs
        chunksize=0,  # 0 => nº de lotes = nº de CPUs
    )
    with open("auditoria_calidad.json", "w", encoding="utf-8") as f:
        json.dump(resultado_json, f, ensure_ascii=False, indent=2)
    print(nfilas)
    # Para guardar:
    df_casos.write_csv(r"C:\Users\1032497498\PycharmProjects\Modelo_observacion_talleres_paralelizado\observaciones_por_caso.csv")
