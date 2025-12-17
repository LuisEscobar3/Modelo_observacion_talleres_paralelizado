import polars as pl
from typing import Optional

def read_csv_with_polars(file_path: str) -> Optional[pl.DataFrame]:
    """
    Lee un CSV normal con Polars y devuelve todo el DataFrame (sin filtros ni límites).
    """
    try:
        df = pl.read_csv(file_path)

        if "NOMBRE TALLER" not in df.columns:
            print("❌ La columna 'NOMBRE TALLER' no existe en el archivo.")
            return None

        return df

    except FileNotFoundError as e:
        print(f"Archivo no encontrado: {e}")
        return None
    except Exception as e:
        print(f"Ocurrió un error al leer con Polars: {e}")
        return None
