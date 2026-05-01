import pandas as pd
from src.config.configuration import ConfigurationManager
from src.components.model_evaluation import ModelEvaluation
from src.logger import logging
import wandb

class ModelEvaluationPipeline:
    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_model_evaluation_config()
        evaluation = ModelEvaluation(config=eval_config)
        
        # Generar archivos
        #evaluation.generate_predictions()
        
        # Obtener métricas ERRANT
        metrics = evaluation.run_full_evaluation()
        evaluation.save_metrics_to_local(metrics)
        df = pd.DataFrame([metrics])
        
        # CreaR la tabla de WandB a partir del DataFrame
        metrics_table = wandb.Table(columns=["Dataset", "F0.5", "Precision", "Recall", "TP", "FP", "FN"])
            
        for set_name, m in metrics.items():
            metrics_table.add_data(
                set_name, 
                m["errant_F0.5"], 
                m["errant_Precision"], 
                m["errant_Recall"],
                m["errant_TP"], 
                m["errant_FP"], 
                m["errant_FN"]
            )
        
        # Guardar metricas en WandB si la sesion existe
        wandb.log({"final_evaluation_summary": metrics_table})
        logging.info("Métricas enviadas a WandB:", metrics)

if __name__ == '__main__':
    try:
        obj = ModelEvaluationPipeline()
        obj.main()
    except Exception as e:
        logging.exception(e)
        raise e