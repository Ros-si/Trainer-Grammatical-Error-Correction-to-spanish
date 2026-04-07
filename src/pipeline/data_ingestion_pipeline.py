import sys
from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.logger import logging
from src.exception import CustomException

class DataIngestionTrainingPipeline:
    """
    Pipeline para la etapa de ingesta de datos. Se encarga de orquestar la ejecución del componente de ingesta, gestionando la configuración y el flujo de ejecución.
    """
    def __init__(self):
        pass

    def main(self):
        """
        Metodo principal que orquesta la ejecucion de la etapa de ingesta.
        1. Gestiona la configuración utilizando ConfigurationManager para obtener los parámetros necesarios para la ingesta de datos.
        2. Inicia el componente de DataIngestion, pasando la configuración obtenida.
        """
        try:
            logging.info("Obteniendo configuración para Data Ingestion...")
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            
            logging.info("Iniciando el componente de Data Ingestion...")
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_dataset()            
            logging.info("Etapa de Data Ingestion finalizada.")
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    try:
        obj = DataIngestionTrainingPipeline()
        obj.main()
    except Exception as e:
        logging.exception(e)
        raise e