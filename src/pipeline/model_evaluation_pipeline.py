from src.config.configuration import ConfigurationManager
from src.components.model_evaluation import ModelEvaluation
from src.logger import logging
import wandb

class ModelEvaluationTrainingPipeline:
    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_model_evaluation_config()
        evaluation = ModelEvaluation(config=eval_config)
        
        # Generar archivos
        evaluation.generate_predictions()
        
        # Obtener métricas ERRANT
        metrics = evaluation.run_errant_pipeline()
        
        # Guardar metricas en WandB si la sesion existe
        if wandb.run is not None:
            wandb.log(metrics)
            logging.info("Métricas enviadas a WandB:", metrics)