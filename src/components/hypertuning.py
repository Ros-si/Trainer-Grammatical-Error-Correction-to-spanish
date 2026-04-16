import gc
import torch
import json
import os
import sys
import optuna
import wandb
from transformers import AutoTokenizer
from datasets import load_dataset
from src.exception import CustomException
from src.entity.config_entity import HypertuningConfig, ModelTrainerConfig, DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class HyperparameterTuner:
    def __init__(self, model_checkpoint:str, config: HypertuningConfig, trainer_config: ModelTrainerConfig, data_transformation_config: DataTransformationConfig):
        self.model_checkpoint = model_checkpoint
        self.config = config
        self.trainer_config = trainer_config
        self.data_transformation_config = data_transformation_config    
        self.dataset =None
    
    
    def prepare_data(self):
        """Preprocesa el dataset una sola vez por checkpoint"""
        # Actualizar el config de DataTransformation
        self.data_transformation_config.tokenizer_name = self.model_checkpoint
        self.data_transformation_config.save_to_disk = False

        data_transformation = DataTransformation(config=self.data_transformation_config)

        ds = load_dataset(self.config.source_data_URL)
        self.dataset = ds.map(data_transformation.preprocess_function, batched=True, remove_columns=ds['train'].column_names)


    def objective(self, trial):
        try:
            lr = trial.suggest_float("learning_rate", self.config.lr[0], self.config.lr[-1], log=True)
            #wd = trial.suggest_float("weight_decay", self.config.wd[0], self.config.wd[-1])
            #bs = trial.suggest_categorical("batch_size", self.config.bs)
            self.trainer_config.lr =lr

            # Actualizar el config de ModelTrainer para la búsqueda de hiperpárametros 
            self.trainer_config.project_name = self.config.project_name
            self.trainer_config.model_ckpt = self.model_checkpoint
            self.trainer_config.run_name = f"{self.config.run_name}-trial-{trial.number}"
            self.trainer_config.load_best_model = False
            self.trainer_config.push_to_hub = False
            self.trainer_config.lr =lr

            config_wb = {
            "lr": lr,
            #"weight_decay": wd,
            "checkpoint": self.model_checkpoint,
            #"batch_size": bs
            }

            tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

            model_trainer = ModelTrainer(self.trainer_config)
            trainer = model_trainer.initiate_model_training(self.dataset['train'], self.dataset['validation'], tokenizer, config_wb=config_wb)
            print(trainer)
            metrics = trainer.evaluate() 
            print("metrics:", metrics)
            eval_loss = metrics['eval_loss']

            # Ahora usamos eval_loss para Optuna
            trial.set_user_attr("eval_loss", eval_loss)
            
            trial.set_user_attr("eval_loss", eval_loss)
            wandb.log({
                "eval_loss": eval_loss
            })

            wandb.finish()
        except Exception as e:
            raise CustomException(e, sys)
        finally:          
            # Limpieza de memoria
            if 'model_trainer' in locals():
                del model_trainer
            if 'trainer_obj' in locals():
                del trainer_obj
            self.cleanup()

        return trainer["eval_loss"]

    def run_tuning(self):
        self.prepare_data()
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.config.n_trials)
        
        # Guardar resultados en un JSON
        model_name = self.model_checkpoint.split("/")[-1]
        output_path = os.path.join(self.config.root_dir, f"best_params_{model_name}-{self.config.run_name}.json")
        
        with open(output_path, "w") as f:
            json.dump(study.best_params, f, indent=4)
        
        return study.best_params
    

    def cleanup(self):
        # 1. Eliminar variables pesadas si existen en el scope global
        if 'model' in globals(): del globals()['model']
        if 'trainer' in globals(): del globals()['trainer']
            
        # 2. Recolector de basura de Python
        gc.collect()
            
        # 3. Vaciar caché de PyTorch
        torch.cuda.empty_cache()
            
        # 4.Resetear estadísticas de memoria
        torch.cuda.reset_peak_memory_stats()        