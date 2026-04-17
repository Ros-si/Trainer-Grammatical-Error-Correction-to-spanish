import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging  
from dataclasses import dataclass
from pathlib import Path

import os
from datasets import load_dataset
from src.entity.config_entity import DataIngestionConfig

class DataIngestion:
    """
    Clase responsable de descargar el dataset desde Hugging Face y guardarlo localmente en formato Apache Arrow
    """
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_dataset(self):
        """
        Descarga y guarda el dataset en formato Arrow
        """
        if not os.path.exists(self.config.dataset_cache_dir):
            logging.info(f"Descargando dataset desde el Hub: {self.config.source_URL}...")
            
            # Descargamos el dataset
            ds = load_dataset(self.config.source_URL)

            # Se elimina el split y las columnas que no se usaran durante el entrenamiento
            del(ds["test"])
            dataset =ds.remove_columns(['sentence','tokens', 'error_tags', 'error_type', 'span', 'annotation', 'corrupted_tagged'])
            del(ds)
            
            # Guardamos directamente en disco en formato Arrow
            dataset.save_to_disk(self.config.dataset_cache_dir)
            
            logging.info(f"Dataset guardado en: {self.config.dataset_cache_dir}")
        else:
            logging.info("El dataset ya existe localmente, omitiendo descarga")