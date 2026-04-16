from src.config.configuration import ConfigurationManager
from src.components.hypertuning import HyperparameterTuner
from src.logger import logging

class HypertuningPipeline:
    """
    Pipeline para la etapa de optimización de busqueda de hiperparametros de los modelos. Gestiona la configuración, ejecuta la búsqueda.
     1. Gestiona la configuración utilizando ConfigurationManager para obtener los parámetros necesarios para la búsqueda de hiperparametros.
     2. Ejecuta la búsqueda de hiperparametros para cada ckeckpoint definido
    """
    def __init__(self):
        pass

    def main(self):                

        config_manager = ConfigurationManager()
        hypertune_config =config_manager.get_hypertuning_config()
        model_trainer_config = config_manager.get_model_trainer_config() 
        data_transformation_config = config_manager.get_data_transformation_config()

        checkpoints = hypertune_config.models_ckpt
        for checkpoint in checkpoints :
            logging.info(f"Busqueda de hiperparametros para {checkpoint} iniciada....")
            tuner = HyperparameterTuner(checkpoint, hypertune_config, model_trainer_config, data_transformation_config)
            best_params = tuner.run_tuning() 
            logging.info(f"Mejores parametros encontrados para {checkpoint}: {best_params}")


if __name__ == "__main__":
    obj = HypertuningPipeline()
    obj.main()



            
       