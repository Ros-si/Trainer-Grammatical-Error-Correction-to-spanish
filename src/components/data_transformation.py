import sys
from datasets import load_from_disk
from transformers import AutoTokenizer
from src.exception import CustomException
from src.logger import logging
from datasets import load_from_disk
from src.entity.config_entity import DataTransformationConfig

class DataTransformation:
    """
    Clase responsable de transformar el dataset descargado en la etapa de ingesta, aplicando tokenización y guardando los resultados
    """
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        try:
            logging.info(f"Cargando tokenizer para: {self.config.tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name, use_fast=True)
            
            # Configuraciones arquitectura
            model_name = str(self.config.tokenizer_name).lower()
            
            if "mbart" in model_name:
                self.tokenizer.src_lang = "es_XX"
                self.tokenizer.tgt_lang = "es_XX"
            elif "m2m100" in model_name:
                self.tokenizer.src_lang = "es"
                self.tokenizer.tgt_lang = "es"
                
        except Exception as e:
            raise CustomException(e, sys)

    def preprocess_function(self, examples):
        """
        Función de tokenizacion que se aplicará a cada ejemplo del dataset. Se encarga de tokenizar tanto las entradas como las salidas (inputs y targets).

        Parameters
        ----------
        examples : dict
            un batch de ejemplos del dataset, con claves 'corrupted' para las entradas y 'sentence' para las salidas
        """
        try:
            inputs = examples['corrupted']
            targets = examples['sentence']
            
            model_inputs = self.tokenizer(
                inputs, 
                text_target=targets, 
                max_length=self.config.max_input_length, 
                truncation=True,
                padding="max_length" 
            )
            return model_inputs
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self):
        """
        Función principal que orquesta la transformación de datos: carga el dataset, aplica la tokenización y guarda los resultados en disco, en caso de estar configurado para hacerlo, en modo hypertuning no se guarda. Devuelve las rutas de los datasets transformados y del tokenizador.
        """
        try:
            logging.info("Iniciando transformación de datos...")
            
            # 1. Cargar dataset desde la etapa de Ingestión
            ds = load_from_disk(self.config.dataset_cache_dir)
            
            # 2. Aplicar el preprocesamiento
            logging.info("Aplicando tokenizacion al dataset...")
            tokenized_dataset = ds.map(
                self.preprocess_function,
                batched=True,
                remove_columns=ds['train'].column_names 
            )

            # 3. Guardar dataset tokenizado
            if self.config.save_to_disk:
                logging.info("Guardando datasets tokenizados en artifacts (Full Dataset Mode)...")
                tokenized_dataset['train'].save_to_disk(self.config.transformed_train_path)
                #tokenized_dataset['test'].save_to_disk(self.config.transformed_test_path)
                tokenized_dataset['validation'].save_to_disk(self.config.transformed_validation_path)
            else:
                logging.info("Omitiendo guardado en disco (Hypertuning Mode)")

            # 4. Guardar el tokenizador (para Inferencia)
            self.tokenizer.save_pretrained(self.config.preprocessor_obj_file_path)

            logging.info("Transformación completada")

            return (
                self.config.transformed_train_path,
                self.config.transformed_test_path,
                self.config.transformed_validation_path,
                self.config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)