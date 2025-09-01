import os
import json
import time
import dotenv
from langchain.globals import set_debug
import polars as pl
from services.llm_manager import load_llms  # (no lo usamos aquí, se mantiene por compatibilidad)
from Functions.read_csv import read_csv_with_polars
from Functions.Process_indibitual_observations import procesar_observacion_individual


def main():
    """
    Ejecuta la configuración inicial del entorno y procesa el DataFrame
    para devolver un JSON con todos los registros.
    """
    dotenv.load_dotenv()
    set_debug(False)
    os.environ["APP_ENV"] = os.environ.get("APP_ENV", "sbx")

    # Ruta de entrada (ajústala a tu entorno)
    ruta_csv = r"C:\Users\1032497498\PycharmProjects\Modelo_observacion_talleres_paralelizado\datos_procesar.csv"
    llms = load_llms()
    gemini = llms["gemini_pro"]
    # 1) Leer datos con Polars
    df_data = read_csv_with_polars(ruta_csv)
    print(df_data.head())

    # 2) Cronometrar el procesamiento
    start_time = time.perf_counter()

    resultado_json = procesar_observacion_individual(
        df_observacion=df_data,
        prompt_sistema="",  # no se usa en esta función
        cliente_llm=gemini,  # no se usa en esta función
        max_workers=0,  # usa número de núcleos disponibles
        chunksize=1000,
    )

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # 3) Mostrar resumen y guardar salida
    print(f"✅ Total registros procesados: {len(resultado_json.get('registros', []))}")
    print(f"⏱️ Tiempo de ejecución: {elapsed_time:.2f} segundos")

    salida_json = r"C:\Users\1032497498\PycharmProjects\Modelo_observacion_talleres_paralelizado\salida_registros.json"
    with open(salida_json, "w", encoding="utf-8") as f:
        json.dump(resultado_json, f, ensure_ascii=False, indent=2)
    print(f"📂 JSON generado en: {salida_json}")



    df = cargar_json_a_polars(salida_json)
    df_casos = unir_observaciones_por_caso(df, separador=" | ")  # o "\n"
    print(df_casos.head(10))

if __name__ == "__main__":
    print("xddd")
    main()
