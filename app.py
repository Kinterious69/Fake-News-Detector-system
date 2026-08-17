import html
import os
import re
from functools import lru_cache

import gradio as gr
import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HF_MODEL_ID = "kinterious-69/fake-news-distilbert"
LABEL_NAMES = ["Real", "Fake"]
MAX_LENGTH = 128
LIME_SAMPLES = 60
LIME_FEATURES = 10
PREDICTION_BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(min(4, os.cpu_count() or 1))

_model = None
_tokenizer = None
_explainer = None


def load_model():
    global _model, _tokenizer, _explainer
    if _model is None:
        print(f"Loading {HF_MODEL_ID} on {DEVICE}...")
        _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        _model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
        _model.to(DEVICE)
        _model.eval()
        _explainer = LimeTextExplainer(class_names=LABEL_NAMES, random_state=42)
        print("Model loaded successfully.")
    return _tokenizer, _model, _explainer


def predict_proba(texts):
    """Batched inference; LIME perturbations are processed in batches."""
    tokenizer, model, _ = load_model()
    clean_texts = [str(t)[:4000] for t in texts]
    results = []
    for start in range(0, len(clean_texts), PREDICTION_BATCH_SIZE):
        batch = clean_texts[start:start + PREDICTION_BATCH_SIZE]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=MAX_LENGTH, padding=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        results.append(probs.detach().cpu().numpy())
    return np.vstack(results)


@lru_cache(maxsize=32)
def cached_prediction(text):
    return tuple(float(x) for x in predict_proba([text])[0])


def normalize_token(token):
    return re.sub(r"^[^\w]+|[^\w]+$", "", token.lower())


def build_highlights(text, explanation):
    weights = {normalize_token(w): float(s) for w, s in explanation if normalize_token(w)}
    parts = []
    for raw in text.split():
        clean = normalize_token(raw)
        weight = weights.get(clean, 0.0)
        if weight > 0.01:
            opacity = min(0.65, 0.12 + abs(weight) * 3.5)
            style = f"background:rgba(220,53,69,{opacity:.2f});border-bottom:2px solid #dc3545;"
        elif weight < -0.01:
            opacity = min(0.65, 0.12 + abs(weight) * 3.5)
            style = f"background:rgba(25,135,84,{opacity:.2f});border-bottom:2px solid #198754;"
        else:
            style = ""
        parts.append(f'<span style="display:inline-block;margin:2px 2px;padding:2px 4px;border-radius:4px;{style}">{html.escape(raw)}</span>')
    return '<div class="highlight-box">' + " ".join(parts) + '</div><div class="legend"><span class="fake-key">■</span> pushes toward FAKE &nbsp;&nbsp;<span class="real-key">■</span> pushes toward REAL</div>'


def build_bar_chart(explanation):
    max_abs = max((abs(float(w)) for _, w in explanation), default=1.0)
    rows = []
    for word, weight in explanation:
        weight = float(weight)
        width = min(100, abs(weight) / max_abs * 100)
        color = "#dc3545" if weight >= 0 else "#198754"
        direction = "FAKE" if weight >= 0 else "REAL"
        rows.append(f'<div class="bar-row"><div class="bar-label">{html.escape(str(word))}</div><div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%;background:{color};"></div></div><div class="bar-value" style="color:{color};">{weight:+.3f} ({direction})</div></div>')
    return '<div class="bars">' + ''.join(rows) + '</div>'


def analyze(text):
    if not text or not text.strip():
        return "⚠️ Please enter a news claim or headline.", "", "", ""
    text = text.strip()[:500]
    try:
        _, _, explainer = load_model()
        probs = np.array(cached_prediction(text))
        pred_label = int(np.argmax(probs))
        confidence = float(probs[pred_label])
        label = LABEL_NAMES[pred_label]
        real_pct, fake_pct = float(probs[0])*100, float(probs[1])*100

        exp = explainer.explain_instance(
            text, predict_proba, num_features=LIME_FEATURES,
            num_samples=LIME_SAMPLES, labels=[pred_label]
        )
        explanation = exp.as_list(label=pred_label)

        if label == "Fake":
            verdict = f'<div class="verdict fake"><div class="verdict-title">⚠️ LIKELY FAKE</div><div class="confidence">Model confidence: {confidence*100:.1f}%</div></div>'
        else:
            verdict = f'<div class="verdict real"><div class="verdict-title">✅ LIKELY REAL</div><div class="confidence">Model confidence: {confidence*100:.1f}%</div></div>'

        gauge = f'<div class="gauge-wrap"><div class="gauge-title">Credibility Score — {real_pct:.1f}% Real</div><div class="gauge-track"><div class="gauge-real" style="width:{real_pct:.2f}%"></div></div><div class="gauge-labels"><span>0% Fake</span><span>50%</span><span>100% Real</span></div><div class="probabilities"><span>Real: <b>{real_pct:.1f}%</b></span><span>Fake: <b>{fake_pct:.1f}%</b></span></div></div>'
        return verdict, gauge, build_highlights(text, explanation), build_bar_chart(explanation)
    except Exception as exc:
        return f'<div class="error">Analysis failed: {html.escape(str(exc))}</div>', "", "", ""


SAMPLES = [
    "The president signed a new infrastructure bill worth 1 trillion dollars.",
    "Scientists confirm that drinking bleach cures all known diseases.",
    "NASA announced plans to return astronauts to the moon.",
    "The government secretly banned all elections and free speech permanently.",
]

CSS = """
body{background:#f7f8fa}.gradio-container{max-width:950px!important}.title{text-align:center;font-size:2.5rem;font-weight:800}.subtitle{text-align:center;color:#667085;margin-bottom:25px}.verdict{padding:18px 22px;border-radius:12px;margin-top:10px}.verdict.fake{background:#fff1f2;border-left:6px solid #dc3545}.verdict.real{background:#ecfdf3;border-left:6px solid #198754}.verdict-title{font-size:1.6rem;font-weight:800}.confidence{margin-top:5px;color:#667085}.gauge-wrap{padding:14px 4px}.gauge-title{font-weight:700;margin-bottom:9px}.gauge-track{height:22px;background:#f8d7da;border-radius:999px;overflow:hidden}.gauge-real{height:100%;background:#198754;border-radius:999px}.gauge-labels,.probabilities{display:flex;justify-content:space-between;margin-top:7px;color:#667085}.probabilities{margin-top:14px;color:#344054}.highlight-box{line-height:2;padding:14px;border:1px solid #e4e7ec;border-radius:10px;background:#fff}.legend{margin-top:10px;color:#667085;font-size:.9rem}.fake-key{color:#dc3545}.real-key{color:#198754}.bars{background:#fff;padding:12px;border-radius:10px;border:1px solid #e4e7ec}.bar-row{display:grid;grid-template-columns:120px 1fr 120px;gap:10px;align-items:center;margin:8px 0}.bar-label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.bar-track{height:14px;background:#f2f4f7;border-radius:8px;overflow:hidden}.bar-fill{height:100%;border-radius:8px}.bar-value{text-align:right;font-size:.85rem}.error{padding:14px;background:#fff1f2;color:#b42318;border-radius:10px}@media(max-width:650px){.title{font-size:2rem}.bar-row{grid-template-columns:80px 1fr}.bar-value{grid-column:2;text-align:left}}
"""

with gr.Blocks(title="Fake News Detector", css=CSS) as demo:
    gr.HTML('<div class="title">🕵️ Fake News Detector</div><div class="subtitle">Powered by DistilBERT + LIME Explainability · AI Final Group Project</div>')
    gr.Markdown("### 📝 Enter a news claim or headline")
    text = gr.Textbox(lines=5, max_lines=8, max_length=500, placeholder='e.g. "The government secretly banned all elections permanently."', label="News claim")
    analyze_btn = gr.Button("🔍 Analyze Claim", variant="primary")
    gr.Examples(examples=SAMPLES, inputs=text, label="💡 Try a sample claim")
    gr.Markdown("### 🏷️ Verdict")
    verdict = gr.HTML()
    gr.Markdown("### 📊 Credibility Score")
    gauge = gr.HTML()
    gr.Markdown("### 🔍 Why did the model decide this?")
    highlights = gr.HTML()
    gr.Markdown("### 📈 Top Influencing Words")
    bars = gr.HTML()
    gr.Markdown("⚠️ This tool is a decision-support aid — not an arbiter of truth. Always verify important claims with trusted primary sources.")
    analyze_btn.click(analyze, inputs=text, outputs=[verdict, gauge, highlights, bars], show_progress="full")
    text.submit(analyze, inputs=text, outputs=[verdict, gauge, highlights, bars], show_progress="full")

if __name__ == "__main__":
    load_model()
    demo.launch()
