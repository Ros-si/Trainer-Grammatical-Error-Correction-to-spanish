from src.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.pipeline.model_trainer_pipeline import ModelTrainerTrainingPipeline
from src.pipeline.model_evaluation_pipeline import ModelEvaluationPipeline
import sys
from src.logger import logging
from src.exception import CustomException
import wandb

try:
   # Etapa 1: DATA INGESTION
   STAGE_NAME = "Etapa: Data Ingestion"
   logging.info(f"--- {STAGE_NAME} iniciada ---") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logging.info(f"--- {STAGE_NAME} completada ---\n\n============")

   # Etapa 2: DATA TRANSFORMATION
   STAGE_NAME = "Etapa: Data Transformation"
   logging.info(f"--- {STAGE_NAME} iniciada ---") 
   data_transformation = DataTransformationTrainingPipeline()
   data_transformation.main()
   logging.info(f"--- {STAGE_NAME} completada ---\n\n============")

   # Etapa 3: MODEL TRAINING
   STAGE_NAME = "Etapa: Model Trainer"
   logging.info(f"--- {STAGE_NAME} iniciada ---") 
   model_trainer = ModelTrainerTrainingPipeline()
   model_trainer.main()
   logging.info(f"--- {STAGE_NAME} completada ---\n\n============")

   # Etapa 4: MODEL EVALUATION
   STAGE_NAME = "Etapa: Model Evaluation"
   logging.info(f"--- {STAGE_NAME} iniciada ---") 
   model_evaluation = ModelEvaluationPipeline()
   model_evaluation.main()
   logging.info(f"--- {STAGE_NAME} completada ---\n\n============")
   
   if wandb.run is not None:
       wandb.finish() 

except Exception as e:
    logging.exception(e)
    raise CustomException(e, sys)