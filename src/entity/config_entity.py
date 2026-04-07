from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path       
    model_ckpt: str      
    run_name: str         
    epochs: int           
    lr: float             
    train_batch_size: int 
    eval_batch_size: int
    gradient_accumulation_steps: int
    optim: str            
    fp16: bool            
    load_best_model: bool 
    push_to_hub: bool     

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    dataset_cache_dir: Path


@dataclass(frozen=True)    
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