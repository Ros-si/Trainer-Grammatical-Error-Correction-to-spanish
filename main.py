from src.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.pipeline.model_trainer_pipeline import ModelTrainerTrainingPipeline
import sys
from src.logger import logging
from src.exception import CustomException

# Etapa 1: DATA INGESTION
STAGE_NAME = "Etapa: Data Ingestion"
try:
   logging.info(f"--- {STAGE_NAME} iniciada ---") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logging.info(f"--- {STAGE_NAME} completada ---\n\n============")
except Exception as e:
    logging.exception(e)
    raise CustomException(e, sys)


# Etapa 2: DATA TRANSFORMATION
STAGE_NAME = "Etapa: Data Transformation"
try:
   logging.info(f"--- {STAGE_NAME} iniciada ---") 
   data_transformation = DataTransformationTrainingPipeline()
   data_transformation.main()
   logging.info(f"--- {STAGE_NAME} completada ---\n\n============")
except Exception as e:
    logging.exception(e)
    raise CustomException(e, sys)


# Etapa 3: MODEL TRAINING
STAGE_NAME = "Etapa: Model Trainer"
try:
   logging.info(f"--- {STAGE_NAME} iniciada ---") 
   model_trainer = ModelTrainerTrainingPipeline()
   model_trainer.main()
   logging.info(f"--- {STAGE_NAME} completada ---\n\n============")
except Exception as e:
    logging.exception(e)
    raise CustomException(e, sys)