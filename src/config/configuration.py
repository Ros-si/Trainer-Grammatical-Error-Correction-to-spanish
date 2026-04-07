import os
from src.utils import create_directories, read_yaml
from src.logger import logging
from src.entity.config_entity import (DataIngestionConfig, 
                                      DataTransformationConfig, 
                                      ModelTrainerConfig)
from pathlib import Path
from src.constants import *


class ConfigurationManager:
    """
    Clase para gestionar la configuración del proyecto. Se encarga de leer el archivo config.yaml, crear los directorios necesarios y proporcionar métodos para obtener objetos de configuración específicos para cada etapa del pipeline (ingesta, transformación, entrenamiento)
    """
    def __init__(self, config_filepath = CONFIG_FILE_PATH):
        # Leer config.yaml
        self.config = read_yaml(config_filepath)
        
        # Crear la carpeta raíz general (artifacts/)
        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Método para obtener la configuración de la etapa de ingesta de datos. Lee los parámetros necesarios del config.yaml, crea los directorios necesarios y devuelve un objeto DataIngestionConfig con las rutas y parámetros configurados

        Returns
        -------
        DataIngestionConfig
            Configuración para la etapa de ingesta de datos
        """
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            dataset_cache_dir=Path(config.dataset_cache_dir)
        )

        return data_ingestion_config


    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Método para obtener la configuración de la etapa de transformación de datos. Lee los parámetros necesarios del config.yaml, crea los directorios necesarios y devuelve un objeto DataTransformationConfig con las rutas y parámetros configurados

        Returns
        -------
        DataTransformationConfig
            Configuración para la etapa de transformación de datos
        """
        config = self.config.data_transformation
        
        # Definir la subcarpeta específica para el modelo (ej: artifacts/data_transformation/marian_mt)
        model_specific_dir = os.path.join(config.root_dir, config.model_id)
        create_directories([model_specific_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            dataset_cache_dir=Path(self.config.data_ingestion.dataset_cache_dir),
            tokenizer_name=config.tokenizer_name,
            max_input_length=config.max_input_length,
            max_target_length=config.max_target_length,
            # Rutas de salida para los datasets tokenizados (Apache Arrow)
            transformed_train_path=Path(os.path.join(model_specific_dir, "train")),
            transformed_test_path=Path(os.path.join(model_specific_dir, "test")),
            transformed_validation_path=Path(os.path.join(model_specific_dir, "validation")),
            # Ruta para guardar el tokenizador con .save_pretrained()
            preprocessor_obj_file_path=Path(os.path.join(model_specific_dir, "tokenizer")),
            save_to_disk=config.save_to_disk
        )
        return data_transformation_config

        

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Método para obtener la configuración de la etapa de entrenamiento del modelo. Lee los parámetros necesarios del config.yaml, crea los directorios necesarios y devuelve un objeto ModelTrainerConfig con las rutas y parámetros configurados

        Returns
        -------
        ModelTrainerConfig
            Configuración para la etapa de entrenamiento del modelo

        """
        config = self.config.model_trainer
        
        # Crear el directorio raíz del entrenamiento
        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            model_ckpt=config.model_ckpt,
            run_name=config.run_name,
            epochs=config.num_train_epochs,
            lr=float(config.lr),
            train_batch_size=config.per_device_train_batch_size,
            eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            optim=config.optim,
            fp16=config.fp16,
            load_best_model=config.load_best_model,
            push_to_hub=config.push_to_hub
        )
        return model_trainer_config
