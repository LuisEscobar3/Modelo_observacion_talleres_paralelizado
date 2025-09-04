import subprocess
from datetime import datetime

def exportar_librerias():
    # Fecha y hora actual
    fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo = f"requirements_{fecha}.txt"

    # Ejecuta pip freeze para listar librerías
    with open(archivo, "w", encoding="utf-8") as f:
        subprocess.run(["pip", "freeze"], stdout=f)

    print(f"✅ Librerías exportadas en: {archivo}")
    return archivo

if __name__ == "__main__":
    exportar_librerias()
