import gradio as gr
import html
import difflib
from models_inference import execute_inference, MODEL_CONFIGS
import spacy
from spacy.tokens import Doc

nlp = spacy.load("es_core_news_md")
TITLE = "Corrector de Errores Gramaticales del Español"

DESCRIPTION = """
**Demo interactiva para la correcion de errores gramaticales en el idioma español.**

**Funcionamiento:**
1. Introduce un texto en el cuadro **Texto de entrada**.
2. Selecciona el modelo para la correción.
3. Presiona **Procesar** para observar el **Resultado** de la versión corregida por el modelo seleccionado.
"""

EXAMPLES = [
    "Pepito jugar en el parque",
    "muchos perros juegan en la parque tambien muchos niño salen. de la escuels",
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
    color:#686868;
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

def draw_corrected(error):
    html_out = f"<span class='correction-word'>{html.escape(error['target'])}</span>"
    return html_out

def draw_error(error):
     html_out = f"""
        <span class='correction-box-error'>
            <span class='src-word'>{html.escape(error['source'])}</span>
        </span>
        """
     return html_out

def draw_merge(error):
    if not error['source']:
        return draw_corrected(error)
    if not error['target']:
        return draw_error(error)
    return draw_error(error) + " " + draw_corrected(error)

def get_errors(orig_text, pred_text):
    """
    Genera una lista de ediciones (spans) con sus coordenadas exactas de palabras
    utilizando difflib, sin incluir etiquetas (labels) ni categorías (tags).
    """
    orig_words = orig_text.split()
    pred_words = pred_text.split()
    
    matcher = difflib.SequenceMatcher(None, orig_words, pred_words)
    spans = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        # Filtrar solo los bloques que contengan modificaciones
        if tag != 'equal':
            spans.append({
                "source": " ".join(orig_words[i1:i2]),
                "target": " ".join(pred_words[j1:j2]),
                "src_start": i1,
                "src_end": i2,
                "tgt_start": j1,
                "tgt_end": j2
            })
            print("spnas:", spans)
            
    return spans


def render_with_draw(orig_text, pred_text, draw_fn):
    doc = nlp(pred_text)
    tokens = [token.text for token in doc]
    spaces = [token.whitespace_ for token in doc]
    errors = get_errors(orig_text, pred_text)

    for error in reversed(errors):
        start = error['tgt_start']
        end = error['tgt_end']

        if start == end:
            if 0 <= start <= len(tokens):
                tokens.insert(start, draw_fn(error))
                spaces.insert(start, spaces[start] if start < len(spaces) else "")
        else:
            tokens[start] = draw_fn(error)
            if start + 1 < end:
                del tokens[start+1:end]
                del spaces[start+1:end]

    return "".join([t + s for t, s in zip(tokens, spaces)])

def render_corrected(orig_text, pred_text):
    return render_with_draw(orig_text, pred_text, draw_corrected)

def render_original(orig_text, pred_text):
    return render_with_draw(orig_text, pred_text, draw_error)

def render_merge(orig_text, pred_text):
    return render_with_draw(orig_text, pred_text, draw_merge)

def get_predict(text, model_name):
    predict = execute_inference(text, model_name)
    return predict

def show_correction(text, model_name, type_draw):
    predict = get_predict(text, model_name)
    if type_draw == "Original":
        output = render_original(text, predict)
    elif type_draw == "Corrección":
        output = render_corrected(text, predict)
    else:
        output = render_merge(text, predict)
    return output

def clear():
    return "", "", ""

with gr.Blocks(css=CSS) as demo:
    gr.Markdown(f"<h1 class='gradio-title'>{TITLE}</h1>") 
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        text = gr.Textbox(label="Texto de entrada", placeholder="Texto con errores...", lines=3)

    with gr.Row():
        model_select = gr.Dropdown(MODELS, label="Seleccionar modelo")
    with gr.Row():
        btn_correct = gr.Button("Procesar", variant="primary")
        btn_clear = gr.Button("Limpiar", variant="secondary")
    type_draw= gr.Radio(choices=["Original", "Corrección", "Ambos"],
                                label="Tipo de visualización",
                                value="Corrección"   
                               )
    gr.Markdown("### Resultado")
    output_text=gr.HTML(label="Correción")
    gr.HTML(LEYEND)
   
   
    btn_correct.click(fn=show_correction, inputs=[text, model_select, type_draw], outputs=[output_text])
    btn_clear.click(fn=clear, outputs=[text, output_text])

    # Formateo correcto de la matriz de ejemplos para evitar desajustes de dimensiones
    formatted_examples = [[ex] for ex in EXAMPLES]
       
    gr.Examples(
        examples=formatted_examples,
        inputs=[text, model_select, type_draw],
        outputs=[output_text],
        fn=show_correction,
        cache_examples=False,
    )
demo.launch(server_name="0.0.0.0", server_port=7860)
