import os
import sys
from pathlib import Path
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from src.entity.config_entity import ModelTrainerConfig
import wandb

class ModelTrainer:
    """
    Clase responsable de entrenar el modelo. Se encarga de cargar el modelo, configurar el entrenamiento y ejecutar el proceso de entrenamiento
    """
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    def initiate_model_training(self, train_dataset, eval_dataset, tokenizer):
        """
        Inicia el proceso de entrenamiento del modelo. Carga el modelo desde el checkpoint especificado en el config.yaml, configura los argumentos de entrenamiento y ejecuta el entrenamiento utilizando Seq2SeqTrainer, finaliza subiendo el modelo al Hugging Face Hub si así se ha configurado. El proceso de entrenamiento se monitorea con WandB

        Parameters
        ----------
        train_dataset : Dataset
            El dataset de entrenamiento ya tokenizado
        eval_dataset : Dataset
            El dataset de evaluación ya tokenizado
        tokenizer : AutoTokenizer
            El tokenizador utilizado para el preprocesamiento
        """
        # El modelo se carga desde el config.yaml
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt)
        model_name = self.config.model_ckpt.split("/")[-1]
        run_name = self.config.run_name

        wandb.init(
            project="GEC-Spanish-trainer", 
            group=f"{model_name}-experiments", 
            name=f"{model_name}-{run_name}",
            reinit=True
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        # Configuración de entrenamiento
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.root_dir,            
            learning_rate=self.config.lr,
            num_train_epochs=self.config.epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            gradient_checkpointing=True,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            logging_strategy='steps',
            logging_steps=100,
            optim=self.config.optim,               
            max_grad_norm=1.0,
            weight_decay=0.01,
            load_best_model_at_end=self.config.load_best_model,      
            metric_for_best_model="eval_loss",# La métrica para decidir cuál es el mejor
            greater_is_better=False,          # Queremos que el Loss sea lo más bajo posible
            save_total_limit=1,
            predict_with_generate=True,
            fp16=self.config.fp16, # Si usas GPU con soporte
            push_to_hub=self.config.push_to_hub, 
            hub_model_id=f"Ro551/{model_name}-GEC-spanish-{run_name}", 
            hub_strategy="end", # Sube el checkpoint en cada época -> every_save
            report_to="wandb"            
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator
        )

        trainer.train()
        if self.config.push_to_hub:
            trainer.push_to_hub(commit_message="Model trained and pushed to Hugging Face Hub")
        wandb.finish()