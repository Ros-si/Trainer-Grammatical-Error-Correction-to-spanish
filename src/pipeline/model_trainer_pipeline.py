from src.config.configuration import ConfigurationManager
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from transformers import AutoTokenizer
from datasets import load_from_disk
import sys
import wandb
from src.exception import CustomException

class ModelTrainerTrainingPipeline:
    """
    Pipeline para la etapa de entrenamiento del modelo. Se encarga de orquestar la ejecución del componente de entrenamiento, gestionando la configuración, la carga de recursos y el flujo de ejecución.
     1. Gestiona la configuración utilizando ConfigurationManager para obtener los parámetros necesarios para el entrenamiento del modelo.
     2. Carga los datasets tokenizados desde disco, evitando re-descargas innecesarias.
     3. Carga el tokenizador utilizado para el preprocesamiento desde disco.
     4. Inicia el componente de ModelTrainer, pasando los datasets y el tokenizador cargados, y ejecuta el proceso de entrenamiento.
    """
    def __init__(self):
        pass

    def main(self):
        try:
            # 1. Gestionar configuración
            config_manager = ConfigurationManager()
            model_trainer_config = config_manager.get_model_trainer_config()
            data_transformation_config = config_manager.get_data_transformation_config()

            # 2. Cargar recursos desde artifacts 
            logging.info("Cargando datasets tokenizados desde artifacts...")
            train_dataset = load_from_disk(data_transformation_config.transformed_train_path)
            eval_dataset = load_from_disk(data_transformation_config.transformed_validation_path)

            logging.info(f"Cargando tokenizador desde: {data_transformation_config.preprocessor_obj_file_path}")
            tokenizer = AutoTokenizer.from_pretrained(data_transformation_config.preprocessor_obj_file_path)

            # 3. Iniciar entrenamiento 
            logging.info("Iniciando componente ModelTrainer...")
            model_trainer = ModelTrainer(config=model_trainer_config)
            
            trainer = model_trainer.initiate_model_training(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer
            )            
            logging.info("Etapa de Model Trainer finalizada")

        except Exception as e:
            raise CustomException(e, sys)