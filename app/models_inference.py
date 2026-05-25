
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
        "base_model": "Helsinki-NLP/opus-mt-es-en", 
        "adapter_model": "Ro551/opus-mt-es-en-GEC-spanish-dsSint",
        "is_lora": False
    }
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CACHE = {}

def load_model_and_tokenizer(model_key):
    """
    Devuelve el modelo y tokenizador desde la caché global.
    Si por alguna razón no se precargó, lo añade a la caché.
    """
    global MODEL_CACHE
    # Si ya está en la caché persistente, se entrega de inmediato
    if model_key in MODEL_CACHE:
        return MODEL_CACHE[model_key]["model"], MODEL_CACHE[model_key]["tokenizer"]
   
    config_info = MODEL_CONFIGS[model_key]
    base_path = config_info["base_model"]
    adapter_path = config_info["adapter_model"]
    
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        adapter_path, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    if config_info["is_lora"] and adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.to(DEVICE)
    model.eval() 
    
    MODEL_CACHE[model_key] = {
        "model": model,
        "tokenizer": tokenizer
    }
    return model, tokenizer

def preload_all_models():
    """
    Función de calentamiento (Warmup) que descarga y estructura 
    todos los modelos en memoria en el arranque del servidor.
    """
    print("\n" + "="*50)
    print(f"[INFO] Iniciando la pre-carga de TODOS los modelos GEC en {DEVICE}...")
    print("="*50)
    
    for model_key in MODEL_CONFIGS.keys():
        print(f"[INFO] Cargando: {model_key} ...")
        try:
            load_model_and_tokenizer(model_key)
            print(f"[OK] {model_key} listo.")
        except Exception as e:
            print(f"[ERROR] No se pudo precargar {model_key}: {str(e)}")
            
    print("="*50)
    print("[SUCCESS] ¡Todos los modelos están calientes en memoria e inferencia lista!")
    print("="*50 + "\n")

def execute_inference(text, model_name):
    """
    Orquesta la tokenización, el forward pass del modelo,
    el proceso de decodificación y el formateo de salida.
    """
    try:
        model, tokenizer = load_model_and_tokenizer(model_name)
        config_info = MODEL_CONFIGS[model_name]
        
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

preload_all_models()