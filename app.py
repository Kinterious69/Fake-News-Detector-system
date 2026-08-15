"""
app.py — Fake News Detector
Uses HuggingFace Inference API — no local model loading needed
Works on any free hosting platform
"""

import os
import json
import requests
import gradio as gr

# ── HuggingFace Inference API ──────────────────────────────────
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
HF_MODEL_ID  = "kinterious-69/fake-news-distilbert"
API_URL      = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

FAKE_WORDS = ["secret","secretly","banned","ban","permanently","shocking",
              "explosive","hoax","conspiracy","hidden","exposed","miracle",
              "cure","cures","guaranteed","confirmed","proves","undeniable",
              "never","always","everybody","nobody","100%","absolutely"]

REAL_WORDS = ["according","reported","announced","said","study","research",
              "scientists","researchers","experts","university","published",
              "percent","data","evidence","signed","approved","official",
              "spokesperson","conference","institute","journal"]

def query_model(text):
    """Call HuggingFace Inference API."""
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    return response.json()

def get_highlights(text):
    """Generate word highlights based on indicator words."""
    fake_found = []
    real_found = []
    html_parts = []

    for word in text.split():
        clean = word.lower().strip(".,!?;:'\"")
        if clean in FAKE_WORDS:
            fake_found.append(clean)
            html_parts.append(f"🔴{word}")
        elif clean in REAL_WORDS:
            real_found.append(clean)
            html_parts.append(f"🟢{word}")
        else:
            html_parts.append(word)

    highlights  = " ".join(html_parts)
    highlights += "\n\n🔴 = pushes toward FAKE   🟢 = pushes toward REAL"

    explanation = ""
    if fake_found:
        explanation += f"Fake indicators: {', '.join(set(fake_found))}\n"
    if real_found:
        explanation += f"Real indicators: {', '.join(set(real_found))}\n"
    if not fake_found and not real_found:
        explanation = "No strong indicator words — prediction based on context."

    return highlights, explanation

def analyze(text):
    if not text or not text.strip():
        return "Please enter a claim.", "", ""

    try:
        result = query_model(text)

        # Parse API response
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                scores = result[0]
            else:
                scores = result

            prob_real = 0.0
            prob_fake = 0.0

            for item in scores:
                if isinstance(item, dict):
                    label = item.get("label", "").upper()
                    score = item.get("score", 0.0)
                    if label in ["LABEL_0", "REAL", "0"]:
                        prob_real = score
                    elif label in ["LABEL_1", "FAKE", "1"]:
                        prob_fake = score

            # If model returns opposite labels swap
            if prob_real == 0.0 and prob_fake == 0.0:
                prob_real = 0.5
                prob_fake = 0.5

            pred      = 1 if prob_fake > prob_real else 0
            label     = "Fake" if pred == 1 else "Real"
            confidence = prob_fake if pred == 1 else prob_real

            if label == "Fake":
                verdict = (
                    f"⚠️ LIKELY FAKE\n"
                    f"Confidence: {confidence*100:.1f}%\n"
                    f"Real: {prob_real*100:.1f}%  |  Fake: {prob_fake*100:.1f}%"
                )
            else:
                verdict = (
                    f"✅ LIKELY REAL\n"
                    f"Confidence: {confidence*100:.1f}%\n"
                    f"Real: {prob_real*100:.1f}%  |  Fake: {prob_fake*100:.1f}%"
                )

        elif isinstance(result, dict) and "error" in result:
            # Model is loading — retry message
            verdict = f"⏳ Model is loading, please wait 20 seconds and try again.\nDetails: {result['error']}"
            return verdict, "", ""
        else:
            verdict = f"Unexpected response: {result}"
            return verdict, "", ""

    except Exception as e:
        verdict = f"Error: {str(e)}"
        return verdict, "", ""

    highlights, explanation = get_highlights(text)
    return verdict, highlights, explanation


# ── Gradio Interface ───────────────────────────────────────────
demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(
        lines=4,
        placeholder='e.g. "The government secretly banned all elections permanently."',
        label="📝 News Claim or Headline"
    ),
    outputs=[
        gr.Textbox(label="🏷️ Verdict & Score",          lines=4),
        gr.Textbox(label="🔍 Word Highlights",            lines=4),
        gr.Textbox(label="📊 Key Indicators",             lines=3),
    ],
    title="🕵️ Fake News Detector",
    description=(
        "AI-powered fake news detection using fine-tuned DistilBERT. "
        "Enter any news claim to get a real-time credibility verdict.\n"
        "Built with DistilBERT + HuggingFace Transformers · AI Final Group Project"
    ),
    allow_flagging="never",
)

demo.launch()
