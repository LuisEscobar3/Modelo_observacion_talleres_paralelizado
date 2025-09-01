import polars as pl
from typing import Optional

def read_csv_with_polars(file_path: str) -> Optional[pl.DataFrame]:
    """
    Lee un archivo CSV de forma eficiente usando la API "lazy" de Polars.

    Esta aproximación es ideal para archivos grandes, ya que Polars primero
    escanea el archivo para construir un plan de ejecución optimizado y solo

    carga los datos en memoria al final con el comando `.collect()`.

    Args:
        file_path: La ruta al archivo CSV que se va a leer.

    Returns:
        Un DataFrame de Polars con el contenido del archivo, o None si el
        archivo no se encuentra o ocurre un error de lectura.
    """
    try:
        df = pl.scan_csv(file_path).collect()
        return df
    except FileNotFoundError:
        # En una librería real, podríamos loguear el error o relanzar la excepción.
        # Devolver None es una opción simple para el manejo por parte del llamador.
        return None
    except Exception:
        # Captura otros posibles errores de Polars durante la lectura/parseo.
        return None