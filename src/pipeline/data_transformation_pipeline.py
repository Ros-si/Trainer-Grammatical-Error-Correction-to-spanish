from src.config.configuration import ConfigurationManager
from src.components.data_transformation import DataTransformation
from src.logger import logging
from src.exception import CustomException
import sys

class DataTransformationTrainingPipeline:
    """
    Pipeline para la etapa de transformación de datos. Se encarga de orquestar la ejecución del componente de transformación, gestionando la configuración y el flujo de ejecución.
    """
    def __init__(self):
        pass

    def main(self):
        try:
            logging.info("Obteniendo configuración para Data Transformation...")
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            logging.info("Iniciando el componente de Data Transformation...")
            data_transformation = DataTransformation(config=data_transformation_config)
            data_transformation.initiate_data_transformation()
            logging.info("Etapa de Data Transformation finalizada")
        except Exception as e:
            raise CustomException(e, sys)