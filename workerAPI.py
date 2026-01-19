import time
import json
from fastapi import FastAPI, Request

from services.llm_manager import load_llms
from Functions.Process_indibitual_observations import procesar_observacion_individual

app = FastAPI()


@app.post("/process-chunk")
async def process_chunk(request: Request):
    payload = await request.json()

    request_id = payload["request_id"]
    chunk_index = payload["chunk_index"]
    registros = payload["registros"]

    start = time.perf_counter()

    llm = load_llms()["gemini_pro"]

    # 🚫 SIN paralelización interna
    resultado = procesar_observacion_individual(
        df_observacion=registros,
        prompt_sistema="",
        cliente_llm=llm,
        max_workers=1,
        chunksize=len(registros)
    )

    elapsed = round(time.perf_counter() - start, 2)

    print(f"✅ Chunk {chunk_index} procesado ({len(registros)}) en {elapsed}s")

    # 👉 aquí puedes guardar a BD o GCS

    return {"ok": True}
