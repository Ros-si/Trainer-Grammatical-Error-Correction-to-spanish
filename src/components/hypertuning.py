import gc
import torch
import json
import os
import sys
import optuna
import wandb
from datasets import DatasetDict, concatenate_datasets
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
        self.raw_dataset=None   
        self.dataset =None
        self.tokenizer=None
    
    def load_dataset_custom(self):
        """
        Carga el dataset segun el tipo de escenario (sintético, COWS-L2H o híbrido)
        """
        mode = self.config.mode 
    
        def get_cows():
            ds = load_dataset(self.config.source_cowsl2h, split={'train': 'train[:30%]', 'validation': 'validation[:20%]'})
            return ds.rename_columns({"input_text": "corrupted", "target_text": "sentence"})

        def get_synthetic():
            return load_dataset(self.config.source_synthetic)

        # logica de selección de dataset según el modo configurado
        if mode == "hybrid":
            ds_cows = get_cows()
            ds_synth = get_synthetic()
            
            dataset_hybrid = DatasetDict()
            for split in ['train', 'validation']:
                dataset_hybrid[split] = concatenate_datasets([
                    ds_cows[split], 
                    ds_synth[split]
                ])
            return dataset_hybrid.shuffle(seed=42)

        elif mode == "cows":
            return get_cows()
        
        else: # mode == "synthetic" o por defecto
            return get_synthetic()
            
        
    def prepare_data(self):
        """Preprocesa el dataset una sola vez por checkpoint"""
        # Actualizar el config de DataTransformation
        self.data_transformation_config.tokenizer_name = self.model_checkpoint
        self.data_transformation_config.save_to_disk = False

        data_transformation = DataTransformation(config=self.data_transformation_config)

        self.raw_dataset = self.load_dataset_custom()        
        
        self.dataset = self.raw_dataset.map(data_transformation.preprocess_function, batched=True, remove_columns=self.raw_dataset['train'].column_names)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)


    def objective(self, trial):
        """

        """
        try:
            lr = trial.suggest_float("learning_rate", self.config.lr[0], self.config.lr[-1], log=True)
            wd = trial.suggest_float("weight_decay", self.config.wd[0], self.config.wd[-1])
            bs = trial.suggest_categorical("batch_size", self.config.bs)
            gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", self.config.gradient_accumulation_steps[0], self.config.gradient_accumulation_steps[-1])
            # Actualizar el config de ModelTrainer para la búsqueda de hiperpárametros 
            self.trainer_config.project_name = self.config.project_name
            self.trainer_config.model_ckpt = self.model_checkpoint
            self.trainer_config.run_name = f"{self.config.mode}-trial-{trial.number}"
            self.trainer_config.epochs = self.config.epochs
            self.trainer_config.load_best_model = False
            self.trainer_config.push_to_hub = False
            self.trainer_config.train_batch_size= bs
            self.trainer_config.weight_decay = wd
            self.trainer_config.lr =lr
            self.trainer_config.gradient_accumulation_steps=gradient_accumulation_steps
            if "mt5" in self.trainer_config.model_ckpt:
                self.trainer_config.fp16 = False

            if "m2m100" in self.trainer_config.model_ckpt:
                self.trainer_config.optim= "adafactor"
            
            config_wb = {
            "lr": lr,
            "weight_decay": wd,
            "checkpoint": self.model_checkpoint,
            "per_device_train_batch_size": bs
            }

            if self.config.use_lora:
                #r = trial.suggest_categorical("r", self.config.lora_config.r)
                alpha = trial.suggest_categorical("lora_alpha", self.config.lora_config.lora_alpha)
                dropout = trial.suggest_categorical("lora_dropout", self.config.lora_config.lora_dropout)
                
                self.trainer_config.use_lora = True
                self.trainer_config.lora_config.lora_alpha = alpha
                self.trainer_config.lora_config.r = alpha #r
                self.trainer_config.lora_config.lora_dropout = dropout
                config_lora = {
                    "r":alpha, #r,
                    "alpha":alpha,
                    "dropout":dropout
                }
                config_wb.update(config_lora)
            

            model_trainer = ModelTrainer(self.trainer_config)
            trainer = model_trainer.initiate_model_training(self.dataset['train'], self.dataset['validation'], self.tokenizer, eval_dataset_raw=self.raw_dataset['validation']['corrupted'], config_wb=config_wb)
            print(trainer)
            metrics = trainer.evaluate() 
            print("metrics:", metrics)
            eval_loss = metrics['eval_loss']
            gleu = metrics['eval_gleu']

            # Ahora usamos eval_loss para Optuna
            trial.set_user_attr("eval_loss", eval_loss)            
            trial.set_user_attr("gleu", gleu)

            wandb.log({"trial_gleu": gleu, "trial_loss": eval_loss})

            wandb.finish()
            # Limpieza
            del trainer
            del model_trainer
            
        except Exception as e:
            raise CustomException(e, sys)
        finally:          
            # Limpieza de memoria
            gc.collect()
            torch.cuda.empty_cache()
            self.cleanup()

        return gleu

    def run_tuning(self):
        self.prepare_data()
        model_name = self.model_checkpoint.split("/")[-1]

        tuning_dir = self.config.root_dir 
        os.makedirs(tuning_dir, exist_ok=True)
        db_path = os.path.join(tuning_dir, f"{model_name}_hypertune.db")     
        storage_name = f"sqlite:///{os.path.abspath(db_path)}"   
        print(f"[*] Iniciando/Retomando estudio en: {db_path}")

        
        study = optuna.create_study(
            study_name="gec-tuning",
            storage=storage_name,
            load_if_exists=True,
            direction="maximize"
            )
        print(f"Trials completados: {len(study.trials)}")
        study.optimize(self.objective, n_trials=self.config.n_trials)
        
        # Guardar resultados en un JSON        
        output_path = os.path.join(self.config.root_dir, f"best_params_{model_name}-{self.config.mode}.json")
        
        with open(output_path, "w") as f:
            json.dump(study.best_params, f, indent=4)
        
        return study.best_params
    

    def cleanup(self):
        """
        Libera los recursos de hardware y memoria del sistema. Eliminando referencias a modelos y trainers, forzando la recolección de basura de Python y liberando la caché de memoria CUDA para evitar errores de Out-of-Memory entre experimentos.
        """
        if 'model' in globals(): del globals()['model']
        if 'trainer' in globals(): del globals()['trainer']
            
        # 2. Recolector de basura de Python
        gc.collect()            
        # 3. Vaciar caché de PyTorch
        torch.cuda.empty_cache()            
        # 4.Resetear estadísticas de memoria
        torch.cuda.reset_peak_memory_stats()        