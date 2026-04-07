import os
from pathlib import Path

# Obtiene la ruta de la raíz del proyecto
# Esto busca el directorio padre de 'src'
BASE_DIR = Path(__file__).resolve().parent.parent.parent

CONFIG_FILE_PATH = BASE_DIR / "src" / "config" / "config.yaml"