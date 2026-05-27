import gradio as gr
from transformers import pipeline

# Load NER pipeline
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

# Entity color mapping
ENTITY_COLORS = {
    "PER": "#FFD700",   # Person - Gold
    "ORG": "#90EE90",   # Organization - Green
    "LOC": "#87CEEB",   # Location - Blue
    "MISC": "#FFB6C1",  # Miscellaneous - Pink
}

ENTITY_LABELS = {
    "PER": "👤 Person",
    "ORG": "🏢 Organization",
    "LOC": "📍 Location",
    "MISC": "🔖 Miscellaneous",
}

def run_ner(text):
    if not text.strip():
        return "Please enter some text.", ""

    entities = ner_pipeline(text)

    # Build highlighted HTML
    html = "<div style='font-size:16px; line-height:2; font-family:Arial'>"
    last_idx = 0
    for ent in entities:
        start, end = ent["start"], ent["end"]
        label = ent["entity_group"]
        color = ENTITY_COLORS.get(label, "#E0E0E0")
        display = ENTITY_LABELS.get(label, label)

        html += text[last_idx:start]
        html += (
            f"<mark style='background:{color}; padding:2px 6px; border-radius:4px; margin:2px;'>"
            f"{text[start:end]} <sup style='font-size:10px'>{display}</sup></mark>"
        )
        last_idx = end
    html += text[last_idx:]
    html += "</div>"

    # Build entity table
    table = "| Entity | Type | Confidence |\n|--------|------|------------|\n"
    for ent in entities:
        label = ENTITY_LABELS.get(ent["entity_group"], ent["entity_group"])
        score = f"{ent['score']:.2%}"
        table += f"| {ent['word']} | {label} | {score} |\n"

    return html, table


# Sample texts
examples = [
    ["Apple was founded by Steve Jobs in Cupertino, California in 1976."],
    ["Elon Musk visited NASA headquarters in Washington D.C. last Tuesday."],
    ["The United Nations held a summit in Paris where Emmanuel Macron spoke about climate change."],
]

# Gradio UI
with gr.Blocks(title="NER App", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 Named Entity Recognition (NER)
    ### Detect People, Organizations, Locations & More from any text!
    Powered by `bert-large-cased-finetuned-conll03` via Hugging Face 🤗
    """)

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Enter Text",
                placeholder="Type or paste any sentence here...",
                lines=5
            )
            submit_btn = gr.Button("🔎 Analyze", variant="primary")

    gr.Examples(examples=examples, inputs=input_text)

    with gr.Row():
        output_html = gr.HTML(label="Highlighted Entities")

    with gr.Row():
        output_table = gr.Markdown(label="Entity Table")

    submit_btn.click(fn=run_ner, inputs=input_text, outputs=[output_html, output_table])

    gr.Markdown("""
    ---
    **Entity Types:**
    🟡 `Person` &nbsp; 🟢 `Organization` &nbsp; 🔵 `Location` &nbsp; 🩷 `Miscellaneous`
    """)

demo.launch()