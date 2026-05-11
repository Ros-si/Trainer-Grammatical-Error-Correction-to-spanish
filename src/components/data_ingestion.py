from sklearn.model_selection import train_test_split
from src.logger import logging  
import os
from datasets import load_dataset, concatenate_datasets, DatasetDict
from src.entity.config_entity import DataIngestionConfig

class DataIngestion:
    """
    Clase responsable de descargar el dataset desde Hugging Face y guardarlo localmente en formato Apache Arrow
    """
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_test_datasets(self):
        """
        Descarga y guarda los datasets de prueba en formato Arrow
        """
        ds_synthetic = load_dataset(self.config.source_URL, split="test")
        ds_cowsl2h = load_dataset(self.config.source_cowsl2h, split="test")

        ds_cowsl2h = ds_cowsl2h.rename_columns({"input_text":"corrupted", "target_text":"sentence"})
        ds_synthetic.remove_columns(['tokens', 'error_tags', 'error_type', 'span', 'annotation', 'corrupted_tagged'])
        
        ds_synthetic.save_to_disk(os.path.join(self.config.dataset_test_cache_dir, "synthetic"))
        ds_cowsl2h.save_to_disk(os.path.join(self.config.dataset_test_cache_dir, "cowsl2h"))
        
        logging.info(f"Datasets de prueba guardados en: {self.config.dataset_test_cache_dir}")        

                  
    def download_dataset(self):
        """
        Descarga y guarda el dataset en formato Arrow segun el modo definido en la configuración (synthetic, cowsl2h o hybrid)
        """
        if self.config.mode == "synthetic":
            logging.info("Modo de ingesta: synthetic...")
            dataset = self.get_data_synthetic()       
        elif self.config.mode == "cowsl2h":
            logging.info("Modo de ingesta: COWSL-2H...")
            dataset = self.get_data_cowsl2h()
        else:  
            logging.info(f"Modo de ingesta: synthetic + COWSL-2H...")
            dataset_synthetic = self.get_data_synthetic()
            dataset_cowsl2h = self.get_data_cowsl2h()
            #dataset = concatenate_datasets(dataset_synthetic, dataset_cowsl2h)
            dataset= DatasetDict()
            for split in ['train', 'validation']:
                dataset[split] = concatenate_datasets([
                    dataset_cowsl2h[split], 
                    dataset_synthetic[split]
                ])
            dataset = dataset.shuffle(seed=42)

        # Guardamos directamente en disco en formato Arrow
        dataset.save_to_disk(self.config.dataset_cache_dir)
            
        logging.info(f"Dataset guardado en: {self.config.dataset_cache_dir}")
        
    def get_data_synthetic(self):
        """
        Método para obtener el dataset sintético

        Returns
        -------
        Dataset
            El dataset sintético generado 
        """
        if not os.path.exists(self.config.dataset_cache_dir):
            logging.info(f"Descargando dataset desde el Hub: {self.config.source_URL}...")
            
            # Descargamos el dataset
            ds = load_dataset(self.config.source_URL)

            # Se elimina el split y las columnas que no se usaran durante el entrenamiento
            del(ds["test"])
            dataset =ds.remove_columns(['tokens', 'error_tags', 'error_type', 'span', 'annotation', 'corrupted_tagged'])
            del(ds)
        else:
            logging.info("El dataset ya existe localmente, omitiendo descarga")

        return dataset
    

    def get_data_cowsl2h(self):
        """
        Método para obtener el dataset COWSL-2H

        Returns
        -------
        Dataset
            El dataset COWSL-2H  
        """
        if not os.path.exists(self.config.dataset_cache_dir):
            logging.info(f"Descargando dataset desde el Hub: {self.config.source_cowsl2h}...")
            
            # Descargamos el dataset
            ds = load_dataset(self.config.source_cowsl2h)
            del ds['test']
        else:
            logging.info("El dataset ya existe localmente, omitiendo descarga")

        return ds
    

   

