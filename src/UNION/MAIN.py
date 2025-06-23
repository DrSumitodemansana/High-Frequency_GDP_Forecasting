import subprocess
import time
from tqdm import tqdm

scripts = [
    'generar_factores.py',
    'PREDICCION.py',
    'PREDINDI.py',
    'Desagregacion.py',
    'Conciliacion.py'
]

print("🚀 Iniciando ejecución secuencial de scripts...\n")
start_global = time.time()

for script in tqdm(scripts, desc="📜 Ejecutando scripts", unit="script"):
    print(f"\n🔁 Ejecutando: {script}")
    start_time = time.time()
    subprocess.run(['python', script], check=True)
    elapsed = time.time() - start_time
    print(f"✅ Finalizado: {script} en {elapsed:.2f} segundos")

total_time = time.time() - start_global
print(f"\n🎉 Todos los scripts se ejecutaron correctamente en {total_time:.2f} segundos.")
