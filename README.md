# Trainer & Hyperparameter Tuning for Spanish GEC
Proyecto para entrenar y evaluar modelos  SequenceToSequence para la  corrección de errores gramaticales en español. Se automatiza la ingesta y el procesamiento de datos, el entrenamiento de modelos mediante Hugging Face y la evaluación  utilizando ERRANT. Además, integra Weights &amp; Biases (WandB) para el monitoreo de experimentos.

El sistema está diseñado separando de forma modular la fase de búsqueda de hiperparámetros (mediante Optuna) y la fase de entrenamiento (Fine-Tuning y PEFT/LoRA).

## Características clave
* **Ingesta y preprocesamientode datos:** Soporte para la ingesta de los datos de entrenamiento y evaluación sintéticos y de COWSL2. Ademas de su transformación para los modelos seq2seq.
* **Optimización Eficiente de Parámetros (PEFT):** Soporte para la integración de adaptadores **LoRA**, permitiendo el entrenamiento de modelos grandes mitigando el consumo de VRAM y el olvido catastrófico.
* **Búsqueda de Hiperparámetros:** Integración con **Optuna** para la exploración automatizada del *learning rate*, *weight decay* y *warmup ratio*, ádemas de *rank*, *alpha* y *dropout* para el entrenamiento con LoRA.
* **Entrenamiento:** Soporte para entrenamiento de modelos seq2seq utilizando la librería Transformers para un ajuste fino completo y eficiente (LoRA).
* **Evaluación:**  Evaluación final del modelo utilizando el marco de trabajo ERRANT.
* **Monitoreo:** Registro de métricas de pérdida (Loss Train y Loss Eval), GLEU y ERRANT exportables a plataformas de *tracking* como Weights & Biases (W&B).
* **Soporte para modelos Seq2Seq:** El flujo de transformación y entrenamiento esta diseñado para los siguientes modelos encoder-decoder y configurados para el español: 
   * mt5-small
   * mt5-large
   * m2m100_418M
   * opus-mt-es-en
   * mbart-large-50

* **Conjuntos de datos utilizados:**
   * Síntético: https://huggingface.co/datasets/Ro551/WikiCorrupted-spanish_to_GEC-GED
   * COWSL2H: https://github.com/ucdaviscl/cowsl2h
* **Interfaz de inferencia:** Despliegue de un entorno interactivo basado en **Gradio** para realizar pruebas manuales y visualizar las correcciones generadas por el modelo en tiempo real.
---

## Modelos soportados y estrategia de hardware

El pipeline está diseñado para operar con arquitecturas del tipo Encoder-Decoder (*seq2seq*). Para maximizar el costo-beneficio de la infraestructura, se recomienda la siguiente segmentación de hardware según la escala del modelo:

| Categoría | Modelos | Arquitectura Recomendada | Estrategia |
| :--- | :--- | :--- | :--- |
| **Modelos Masivos** | `mT5-large`, `mbart-large-50`, `m2m100_418M` | **NVIDIA A100 (40GB/80GB)** | Full Fine-tuning |
| **Modelos Ligeros** | `mT5-small`, `opus-mt-es-en` | **NVIDIA P100 (16GB)** | Full Fine-Tuning |
| **Modelos Masivos** | `mT5-large` | **NVIDIA A100 (40GB/80GB)** | PEFT LoRA |
| **Modelos Ligeros** | `mT5-small`, `opus-mt-es-en`, `m2m100_418M`, `mbart-large-50` | **NVIDIA P100 (16GB)** | PEFT LoRA |

---

## Requisitos e instalación

Este entorno está optimizado para **Python 3.10+** y sistemas con soporte **CUDA**.

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/Ros-si/Trainer-Grammatical-Error-Correction-to-spanish.git](https://github.com/Ros-si/Trainer-Grammatical-Error-Correction-to-spanish.git)
   cd Trainer-Grammatical-Error-Correction-to-spanish
   ```
2. **Instalar dependencias necesarias:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configuración de Credenciales:**
Inicia sesión en Hugging Face (para guardar los checkpoints) y en Weights & Biases (para el monitoreo):
   ```bash
   huggingface-cli login
   wandb login
   ```
---
## Ejecución del pipeline de búsqueda de hiperparámetros
Para ejecutar la búsqueda de hiperparámetros antes del entrenamiento, ejecuta el script:

   ```bash
   python -m src.pipeline.hypertuning_pipeline
   ```
> Los rangos de búsqueda definidos se encuentran en el archivo /src/config/config.yaml

## Ejecución del pipeline de entrenamiento
Los hiperparámetros se definen en el archivo /src/config/config.yaml

> Para entrenar usando la estrategia LoRA (PEFT) se establece en la configuración la variable use_lora=True

Para ejecutar el pipeline de entrenamiento, que contempla la ingesta, transformacion, entrenamiento y evaluación se ejecuta el comando en terminal:

   ```bash
   python main.py
   ```
El resultado final sera:
* Artefactos correspondientes a cada etapa en la ruta local /artifacts.
* Checkpoint del modelo entrenado en el hub de Hugging Face.
* Monitoreo  de entrenamiento y métricas de evaluación en la plataforma Weights &amp; Biases. 
