import gradio as gr
import torch
import html
import difflib
import spacy
from models_inference import execute_inference, preload_all_models, MODEL_CONFIGS

nlp = spacy.load("es_core_news_md")

print("=" * 60)
print("CUDA disponible:", torch.cuda.is_available())
print("Dispositivo:", "cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)

# Cache global para la última inferencia realizada
LAST_PREDICTION_CACHE = {
    "text": None,
    "model_name": None,
    "prediction": None
}

nlp = spacy.load("es_core_news_md")
TITLE = "Corrector de Errores Gramaticales del idioma Español"

DESCRIPTION = """
**Demo interactiva para la correcion de errores gramaticales en el idioma español.**

**Funcionamiento:**
1. Introduce un texto en el cuadro **Texto de entrada**.
2. Selecciona el modelo para la correción.
3. Presiona **Procesar** para observar el **Resultado** de la versión corregida por el modelo seleccionado.
"""

EXAMPLES = [
    "Pepito jugar en el parque",
    "muchos perros juegan en la, parque tambien muchos niño salen de la escuels",
    "Las gata pasear por el jardin bello", 
    "Mañana Tomy viajarear a Londres"
    ]

MODELS =list(MODEL_CONFIGS.keys())
COLOR_ERROR="#ff3333"
COLOR_CORRECT="#D0E6A5"

CSS = f"""
.correction-box-error,
.correction-word {{
    display: inline-flex;
    align-items: center;
    vertical-align: middle;
    border-radius: 6px;
    padding: 2px 6px;
    margin: 1px 2px;
}}
.correction-box-error {{
    background-color: {COLOR_ERROR};
}}
.correction-word {{
    background-color: {COLOR_CORRECT};
}}
/* Palabra original tachada */
.src-word {{
    padding: 1px 4px;
    text-decoration: line-through;
    margin-right: 4px;
    opacity: 0.8;
    color:#ffffff;
    font-weight:bold;
}}
"""

CSS += """
/* Contenedor principal al 70% del ancho */
.gradio-container {
    width: 70% !important;
    margin: auto;
}
/* Título centrado y grande */
.gradio-title {
    text-align: center !important;
    font-size: 2.3em !important;
    font-weight: bold;
    margin-bottom: 20px;
}
/* Subtítulos Markdown centrados */
.gradio-markdown {
    text-align: center;
}
/* Ajuste de filas y spacing */
.gr-row {
    margin-bottom: 15px;
    justify-content: center; /* centra los elementos de la fila */
}
/* TextBoxes uniformes */
.gr-textbox {
    width: 48% !important; /* dos cajas lado a lado */
}
/* Botones centrados y uniformes */
.gr-button {
    min-width: 150px;
    margin: 0 5px;
}
/* HTML output centrado */
.gr-html {
    width: 100%;
    text-align: left;
    margin-top: 10px;
}
/* Leyenda de colores */
.leyenda {
    margin-top: 10px;
    font-size: 0.9em;
    text-align: center;
}
"""

LEYEND=f"""
<div style="margin: 8px 0 18px; display:flex; gap:10px; font-size:14px; align-items:center; flex-wrap:wrap;">
    <div style="display:flex; align-items:center; gap:6px;">
        <span style="width:16px; height:16px; background:{COLOR_ERROR}; border-radius:4px; display:inline-block;"></span>
        <span>Error</span>
    </div>

    <div style="display:flex; align-items:center; gap:6px;">
        <span style="width:16px; height:16px; background:{COLOR_CORRECT}; border-radius:4px; display:inline-block;"></span>
        <span>Corrección</span>
    </div>
</div>
"""
def render_diff_html(orig_text, pred_text, mode="Corrección"):
    """
        Compara el texto original y el texto predicho, token por token y genera el HTML
    """
    orig_doc = nlp(orig_text)
    pred_doc = nlp(pred_text)
    
    orig_words = [t.text for t in orig_doc]
    pred_words = [t.text for t in pred_doc]
    
    # autojunk=False evita que difflib desalinee palabras o puntuaciones repetidas
    matcher = difflib.SequenceMatcher(None, orig_words, pred_words, autojunk=False)
    html_out = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        # Reconstruir fragmentos preservando espacios originales
        src_str = "".join([t.text_with_ws for t in orig_doc[i1:i2]])
        tgt_str = "".join([t.text_with_ws for t in pred_doc[j1:j2]])
        
        src_clean = src_str.strip()
        tgt_clean = tgt_str.strip()
        
        # Obtener el espaciado correcto
        ws = ""
        if tag in ('equal', 'replace', 'insert') and j2 > 0:
            ws = pred_doc[j2 - 1].whitespace_
        elif tag == 'delete' and i2 > 0:
            ws = orig_doc[i2 - 1].whitespace_

        if tag == 'equal':
            html_out.append(html.escape(tgt_str))
            
        elif tag == 'replace':
            if mode == "Original":
                html_out.append(f"<span class='correction-box-error'><span class='src-word'>{html.escape(src_clean)}</span></span>{ws}")
            elif mode == "Corrección":
                html_out.append(f"<span class='correction-word'>{html.escape(tgt_clean)}</span>{ws}")
            else:  # "Ambos"
                html_out.append(f"<span class='correction-box-error'><span class='src-word'>{html.escape(src_clean)}</span></span> <span class='correction-word'>{html.escape(tgt_clean)}</span>{ws}")
                
        elif tag == 'delete':
            if mode in ("Original", "Ambos"):
                html_out.append(f"<span class='correction-box-error'><span class='src-word'>{html.escape(src_clean)}</span></span>{ws}")
                
        elif tag == 'insert':
            if mode in ("Corrección", "Ambos"):
                html_out.append(f"<span class='correction-word'>{html.escape(tgt_clean)}</span>{ws}")
                
    return "".join(html_out)


def show_correction(text, model_name, type_draw):
    global LAST_PREDICTION_CACHE
    
    if not text or not text.strip():
        return ""

    # Si el texto y el modelo son idénticos a los almacenados, reutiliza la predicción
    if (LAST_PREDICTION_CACHE["text"] == text and 
        LAST_PREDICTION_CACHE["model_name"] == model_name and 
        LAST_PREDICTION_CACHE["prediction"] is not None):
        
        predict = LAST_PREDICTION_CACHE["prediction"]
    else:
        # Si cambiaron, ejecuta la inferencia y actualiza la caché
        predict = execute_inference(text, model_name)
        LAST_PREDICTION_CACHE["text"] = text
        LAST_PREDICTION_CACHE["model_name"] = model_name
        LAST_PREDICTION_CACHE["prediction"] = predict
        
    return render_diff_html(text, predict, mode=type_draw)

def clear():
    global LAST_PREDICTION_CACHE
    LAST_PREDICTION_CACHE = {"text": None, "model_name": None, "prediction": None}
    return "", ""

with gr.Blocks() as demo:
    gr.Markdown(f"<h1 class='gradio-title'>{TITLE}</h1>") 
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        text = gr.Textbox(label="Texto de entrada", placeholder="Texto con errores...", lines=3)

    with gr.Row():
        model_select = gr.Dropdown(MODELS, label="Seleccionar modelo", value=MODELS[0])
        
    with gr.Row():
        btn_correct = gr.Button("Procesar", variant="primary")
        btn_clear = gr.Button("Limpiar", variant="secondary")
        
    type_draw = gr.Radio(
        choices=["Original", "Corrección", "Ambos"],
        label="Tipo de visualización",
        value="Corrección"   
    )
    
    gr.Markdown("### Resultado")
    output_text = gr.HTML(label="Correción")
    gr.HTML(LEYEND)
   
    # Evento al presionar el botón "Procesar"
    btn_correct.click(
        fn=show_correction, 
        inputs=[text, model_select, type_draw], 
        outputs=[output_text]
    )
    
    # Evento reactivo al cambiar de radio button (cambio instantáneo sin re-ejecutar el modelo)
    type_draw.change(
        fn=show_correction, 
        inputs=[text, model_select, type_draw], 
        outputs=[output_text]
    )
    
    btn_clear.click(fn=clear, outputs=[text, output_text])

    gr.Examples(
        examples=[[ex] for ex in EXAMPLES],
        inputs=[text, model_select, type_draw],
        outputs=[output_text],
        fn=show_correction,
        cache_examples=False,
    )

if __name__ == "__main__":
    print("[INFO] Inicializando la interfaz y cargando modelos...")
    preload_all_models()
    demo.launch(css=CSS)