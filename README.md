---
title: Fake News Detector
emoji: 🕵️
colorFrom: blue
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
---

# 🕵️ Fake News Detector

A DistilBERT-based fake-news classifier with LIME word-level explainability.

The Space downloads the trained model from `kinterious-69/fake-news-distilbert`; the 250 MB model is not duplicated inside this Space.

### Deployment optimizations
- Batched model inference for LIME perturbations.
- LIME reduced to 60 samples instead of 150–300 for shared CPU responsiveness.
- Lightweight HTML visualizations instead of Plotly.
- Model loaded once and reused.
- Exact repeated predictions cached.
- Input capped at 500 characters and model sequence length at 128 tokens.

This is a decision-support tool, not a factuality oracle. Verify important claims with reliable primary sources.
