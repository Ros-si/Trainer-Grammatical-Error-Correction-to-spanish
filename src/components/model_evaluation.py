import os
import subprocess

from seaborn import load_dataset
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, load_from_disk
from src.logger import logging
from src.entity.config_entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def generate_predictions(self):
        """Genera las correcciones del modelo y las guarda en archivos de texto
        """
        logging.info("Cargando modelo y datos para generación de predicciones...")
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path)
        dataset = load_dataset(self.config.data_path, split="test")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # Rutas de archivos temporales para ERRANT
        src_path = os.path.join(self.config.root_dir, self.config.source_file)
        gold_path = os.path.join(self.config.root_dir, self.config.gold_file)
        pred_path = os.path.join(self.config.root_dir, self.config.pred_file)

        logging.info(f"Procesando {len(dataset)} ejemplos en {device}...")
        
        with open(src_path, "w", encoding="utf-8") as f_src, \
             open(gold_path, "w", encoding="utf-8") as f_gold, \
             open(pred_path, "w", encoding="utf-8") as f_pred:

            for example in dataset:
                inputs = tokenizer(example['sentence'], return_tensors="pt", truncation=True).to(device)
                
                with torch.no_grad():
                    output_tokens = model.generate(
                        **inputs, 
                        max_length=128, 
                        num_beams=4
                    )
                
                prediction = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
                
                f_src.write(example['sentence'].strip() + "\n")
                f_gold.write(example['target'].strip() + "\n")
                f_pred.write(prediction.strip() + "\n")


    def _parse_metrics(self, output):
        """
        Extrae las métricas del output de ERRANT

        Parameters
        ----------
        output : str
            La salida de ERRANT que contiene el resultado de las métricas 
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
    

    def run_errant_pipeline(self):
        """Ejecuta errant_parallel y errant_compare"""
        logging.info("Iniciando pipeline de ERRANT...")
        root = self.config.root_dir
        
        # 1. Crear archivos M2 (anotaciones de edición)
        logging.info("Generando archivos M2...")
        subprocess.run(["errant_parallel", "-orig", f"{root}/source.txt", "-cor", f"{root}/gold.txt", "-out", f"{root}/gold.m2"])
        subprocess.run(["errant_parallel", "-orig", f"{root}/source.txt", "-cor", f"{root}/pred.txt", "-out", f"{root}/pred.m2"])
        
        # 2. Comparar Hyp vs Ref
        logging.info("Comparando resultados...")
        result = subprocess.run(
            ["errant_compare", "-hyp", f"{root}/pred.m2", "-ref", f"{root}/gold.m2"],
            capture_output=True, text=True
        )
        
        return self._parse_metrics(result.stdout)