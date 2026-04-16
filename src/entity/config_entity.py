from dataclasses import dataclass
from pathlib import Path

@dataclass()
class ModelTrainerConfig:
    root_dir: Path       
    model_ckpt: str      
    run_name: str         
    project_name: str    
    epochs: int           
    lr: float             
    train_batch_size: int 
    eval_batch_size: int
    gradient_accumulation_steps: int
    weight_decay: float
    optim: str            
    fp16: bool            
    load_best_model: bool 
    push_to_hub: bool     

@dataclass()
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    dataset_cache_dir: Path


@dataclass()    
class DataTransformationConfig:
    root_dir: Path
    dataset_cache_dir: Path
    tokenizer_name: str
    max_input_length: int
    max_target_length: int
    
    transformed_train_path: Path
    transformed_test_path: Path
    transformed_validation_path: Path
    preprocessor_obj_file_path: Path # Para guardar el tokenizer.save_pretrained()
    save_to_disk: bool


@dataclass()
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path

    tokenizer_path: Path
    source_file: Path
    gold_file: Path
    pred_file: Path
    metric_file_name: Path

@dataclass()
class HypertuningConfig():
    root_dir: Path
    source_data_URL: str
    models_ckpt: list[str]
    project_name: str   
    run_name: str 
    n_trials: int
    lr: list[float]
    wd: list[float]
    bs: list[int]