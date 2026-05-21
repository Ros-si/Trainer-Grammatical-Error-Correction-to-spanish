
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig

MODEL_CONFIGS = {
    "marianMT_merged-lora-best": {
        "base_model": "Helsinki-NLP/opus-mt-es-en" ,
        "adapter_model": "Ro551/opus-mt-es-en-GEC-spanish-LORA-merged",
        "is_lora": True
    },
    "mT5_small_dsSint": {
        "base_model": "google/mt5-small",
        "adapter_model": "Ro551/mt5-small-GEC-spanish-dsSint",
        "is_lora": False
    },
    "mT5_small_dsmergeg": {
        "base_model": "google/mt5-small",
        "adapter_model": "Ro551/mt5-small-GEC-spanish-merged",
        "is_lora": False
    },
    "MarianMT_dsSint": {
        "base_model": "Helsinki-NLP/opus-mt-es-es", 
        "adapter_model": "Ro551/opus-mt-es-en-GEC-spanish-dsSint",
        "is_lora": False
    }
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CACHE = {
    "tokenizer": None,
    "model": None,
    "current_name": None
}

def load_model_and_tokenizer(model_key):
    """
    Carga de forma dinámica el modelo seleccionado y su tokenizador
    dentro de la cache global del entorno de ejecución.
    """
    global MODEL_CACHE
    
    # Si el modelo solicitado ya se encuentra en caché, entonces reutilizarlo 
    if MODEL_CACHE["current_name"] == model_key:
        return MODEL_CACHE["model"], MODEL_CACHE["tokenizer"]
   
    # Liberar memoria de modelos anteriores para evitar Out-of-Memory (OOM)
    if MODEL_CACHE["model"] is not None:
        del MODEL_CACHE["model"]
        del MODEL_CACHE["tokenizer"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    config_info = MODEL_CONFIGS[model_key]
    base_path = config_info["base_model"]
    adapter_path = config_info["adapter_model"]
    
    #print(f"[INFO] Cargando Tokenizador para: {base_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    
    #print(f"[INFO] Cargando Modelo Base en {DEVICE}: {base_path}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        adapter_path, 
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Si el modelo requiere acoplar la matriz de adaptación de bajo rango (LoRA)
    if config_info["is_lora"] and adapter_path:
        #print(f"[INFO] Inyectando Adaptador LoRA desde: {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.to(DEVICE)
    model.eval() 
    
    # Guardar en la caché global 
    MODEL_CACHE["model"] = model
    MODEL_CACHE["tokenizer"] = tokenizer
    MODEL_CACHE["current_name"] = model_key    
    return model, tokenizer

def execute_inference(text, model_name):
    """
    Orquesta la tokenización, el forward pass del modelo,
    el proceso de decodificación y el formateo de salida.
    """
    try:
        model, tokenizer = load_model_and_tokenizer(model_name)
        config_info = MODEL_CONFIGS[model_name]
        
        # Tratamiento especial de inicialización de tokens requerido nativamente por M2M100
        if "m2m100" in config_info:
            tokenizer.src_lang = "es"
            tokenizer.tgt_lang = "es"   
        elif "mbart" in config_info:
            tokenizer.src_lang = "es_XX"
            tokenizer.tgt_lang = "es_XX"

        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=5
            )
                
        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected_text
        
    except Exception as e:
        print(f"[ERROR] Error crítico durante la inferencia: {str(e)}")
        return f"[Error del Servidor] No se pudo procesar el modelo. Detalles: {str(e)}"