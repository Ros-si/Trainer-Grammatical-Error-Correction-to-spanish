import os
import subprocess
import json
from seaborn import load_dataset
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, concatenate_datasets, load_from_disk
import wandb
from src.logger import logging
from src.entity.config_entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def save_metrics_to_local(self, metrics):
        """
        Guarda las métricas calculadas en un archivo JSON local para su posterior análisis
        Parameters
        ----------
        metrics : dict
            Diccionario con las métricas calculadas 
        """
        path = os.path.join(self.config.root_dir, self.config.metric_file_name)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metricas guardadas localmente en: {path}")    

    def _parse_metrics(self, output):
        """
        Extrae las métricas del output de ERRANT

        Parameters
        ----------
        output : str
            La salida de ERRANT que contiene el resultado de las métricas 
        Return
        ------
            Diccionario con las métricas extraídas (TP, FP, FN, Precision, Recall, F0.5)
        """
        for line in output.splitlines():
            if line.strip() and line[0].isdigit():
                parts = line.split("\t")
                metrics = {
                    "errant_TP": int(parts[0]),
                    "errant_FP": int(parts[1]),
                    "errant_FN": int(parts[2]),
                    "errant_Precision": float(parts[3]),
                    "errant_Recall": float(parts[4]),
                    "errant_F0.5": float(parts[5]),
                }
                return metrics
        return None
    

    def run_errant_pipeline(self, source_path, gold_path, pred_path, set_name):
        """
        Ejecuta el pipeline de ERRANT para un conjunto específico
        Parameters
        ----------
        source_path : str
            Ruta al archivo de texto con las oraciones originales (corrupted)
        gold_path : str
            Ruta al archivo de texto con las oraciones corregidas de referencia (sentence)
        pred_path : str
            Ruta al archivo de texto con las oraciones corregidas por el modelo (predicción)
        set_name : str
            Nombre del conjunto evaluado ("synthetic", "cowsl2h", "merged") 
        
        Return
        ------
            Diccionario con las métricas calculadas para el conjunto evaluado
        """
        logging.info(f"Iniciando pipeline de ERRANT para: {set_name}...")
        
        gold_m2 = os.path.join(self.config.root_dir, f"{set_name}_gold.m2")
        pred_m2 = os.path.join(self.config.root_dir, f"{set_name}_pred.m2")
        
        # Crear archivos M2
        logging.info("Generando archivos M2...")
        subprocess.run(["errant_parallel", "-orig", source_path, "-cor", gold_path, "-out", gold_m2])
        subprocess.run(["errant_parallel", "-orig", source_path, "-cor", pred_path, "-out", pred_m2])
        
        # Comparar hipotesis vs referencia
        logging.info("Comparando resultados...")
        result = subprocess.run(
            ["errant_compare", "-hyp", pred_m2, "-ref", gold_m2],
            capture_output=True, text=True
        )
        
        return self._parse_metrics(result.stdout)

    
    def evaluate_single_dataset(self, dataset, set_name, model, tokenizer):
        """
        Genera predicciones y calcula métricas para un dataset específico 
        Return
        ------
            Diccionario con las métricas calculadas para el dataset evaluado
        """
        logging.info(f"Evaluando sobre el conjunto: {set_name} ({len(dataset)} ejemplos)")
        
        src_file = os.path.join(self.config.root_dir, f"{set_name}_src.txt")
        gold_file = os.path.join(self.config.root_dir, f"{set_name}_gold.txt")
        pred_file = os.path.join(self.config.root_dir, f"{set_name}_pred.txt")

        with open(src_file, "w", encoding="utf-8") as f_src, \
             open(gold_file, "w", encoding="utf-8") as f_gold, \
             open(pred_file, "w", encoding="utf-8") as f_pred:

            for example in dataset:
                # Generación de la corrección
                #inputs = tokenizer(example['corrupted'], return_tensors="pt", truncation=True).to(self.device)
                inputs = {
                "input_ids": torch.tensor([example["input_ids"]]).to(self.device),
                "attention_mask": torch.tensor([example["attention_mask"]]).to(self.device)
            }
                with torch.no_grad():
                    output_tokens = model.generate(
                        **inputs, 
                        max_length=128, 
                        num_beams=4
                    )
                
                prediction = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
                
                # Guardar para ERRANT
                f_src.write(example['corrupted'].strip() + "\n")
                f_gold.write(example['sentence'].strip() + "\n")
                f_pred.write(prediction.strip() + "\n")

        # Ejecutar ERRANT para este set
        return self.run_errant_pipeline(src_file, gold_file, pred_file, set_name)


    def run_full_evaluation(self):
        """
        Carga los datos, realiza la evaluación triple y loguea a WandB.

        Return
        -------      
            Diccionario con las métricas para cada conjunto evaluado
        """
        logging.info("Iniciando Evaluación Triple (Sintético, COWSL2H, Combinado)...")
        
        # Cargar Modelo y Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(self.device)

        evaluation_map = {
            "synthetic": load_from_disk(os.path.join(self.config.data_transformed_test_path,"synthetic")),
            "cowsl2h": load_from_disk(os.path.join(self.config.data_transformed_test_path,"cowsl2h")),
            "merged": load_from_disk(os.path.join(self.config.data_transformed_test_path,"merged")), 
        }

        # Ejecutar bucle de evaluación
        all_metrics = {}
        for name, ds in evaluation_map.items():
            metrics = self.evaluate_single_dataset(ds, name, model, tokenizer)
            
            if metrics:
                all_metrics[name] = metrics
            
        logging.info("Evaluación triple completada")
        return all_metrics