from flask import Flask, request, jsonify, render_template_string
import requests
import os

app = Flask(__name__)

HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
API_URL = "https://api-inference.huggingface.co/models/kinterious-69/fake-news-distilbert"

FAKE_WORDS = ["secret","secretly","banned","ban","permanently","shocking",
              "explosive","hoax","conspiracy","hidden","exposed","miracle",
              "cure","cures","guaranteed","confirmed","proves","undeniable"]

REAL_WORDS = ["according","reported","announced","said","study","research",
              "scientists","researchers","experts","university","published",
              "percent","data","evidence","signed","approved","official"]

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fake News Detector</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Arial,sans-serif;background:#f0f2f5;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px}
.card{background:#fff;border-radius:16px;padding:2rem;max-width:680px;width:100%;box-shadow:0 4px 24px rgba(0,0,0,0.1)}
h1{color:#1F4E79;text-align:center;font-size:1.8rem;margin-bottom:4px}
.sub{color:#888;text-align:center;font-size:.85rem;margin-bottom:1.5rem}
textarea{width:100%;padding:12px;border:2px solid #e0e0e0;border-radius:8px;font-size:1rem;resize:vertical;min-height:90px;outline:none;font-family:Arial}
textarea:focus{border-color:#2E75B6}
.btn{width:100%;padding:12px;background:#1F4E79;color:#fff;border:none;border-radius:8px;font-size:1rem;font-weight:bold;cursor:pointer;margin-top:10px;transition:.2s}
.btn:hover{background:#2E75B6}
.samples{margin-top:10px}
.samples p{font-size:.8rem;color:#888;margin-bottom:6px}
.sbtn{background:#f0f2f5;border:1px solid #ddd;border-radius:6px;padding:5px 10px;font-size:.78rem;cursor:pointer;margin:2px;display:inline-block}
.sbtn:hover{background:#ddd}
.result{margin-top:1.2rem;display:none}
.fake-box{background:#fdedec;border-left:5px solid #c0392b;padding:1rem;border-radius:8px;margin-bottom:10px}
.real-box{background:#eafaf1;border-left:5px solid #27ae60;padding:1rem;border-radius:8px;margin-bottom:10px}
.vlabel{font-size:1.3rem;font-weight:bold}
.fake-lbl{color:#c0392b}.real-lbl{color:#1e8449}
.vscore{color:#555;font-size:.9rem;margin-top:4px}
.bars{margin-top:8px}
.bar-row{display:flex;align-items:center;gap:8px;margin:4px 0}
.bar-label{font-size:.8rem;width:30px;color:#555}
.bar-bg{flex:1;background:#e0e0e0;border-radius:8px;height:10px;overflow:hidden}
.bar-fill-r{background:#27ae60;height:100%;border-radius:8px;transition:.5s}
.bar-fill-f{background:#c0392b;height:100%;border-radius:8px;transition:.5s}
.section{background:#f8f9fa;border-radius:8px;padding:1rem;margin-bottom:10px}
.section h3{color:#1F4E79;font-size:.95rem;margin-bottom:6px}
.highlights{line-height:2.2;font-size:.95rem}
.fw{background:rgba(192,57,43,.2);border-radius:3px;padding:1px 4px}
.rw{background:rgba(39,174,96,.2);border-radius:3px;padding:1px 4px}
.legend{font-size:.75rem;color:#888;margin-top:6px}
.loading{text-align:center;color:#888;padding:1rem;display:none}
.disclaimer{font-size:.72rem;color:#aaa;text-align:center;margin-top:1rem;border-top:1px solid #eee;padding-top:10px}
</style>
</head>
<body>
<div class="card">
  <h1>🕵️ Fake News Detector</h1>
  <p class="sub">Powered by DistilBERT · AI Final Group Project</p>
  <textarea id="txt" placeholder='Enter a news claim e.g. "The government secretly banned all elections permanently."'></textarea>
  <div class="samples">
    <p>💡 Try a sample:</p>
    <span class="sbtn" onclick="set('The government secretly banned all elections and free speech permanently.')">Fake sample 1</span>
    <span class="sbtn" onclick="set('Scientists confirm drinking bleach cures all known diseases.')">Fake sample 2</span>
    <span class="sbtn" onclick="set('NASA announced plans to return astronauts to the moon by 2026.')">Real sample 1</span>
    <span class="sbtn" onclick="set('The president signed a new infrastructure bill worth 1 trillion dollars.')">Real sample 2</span>
  </div>
  <button class="btn" onclick="analyze()">🔍 Analyze Claim</button>
  <div class="loading" id="loading">⏳ Analyzing... please wait</div>
  <div class="result" id="result">
    <div id="vbox">
      <div class="vlabel" id="vlbl"></div>
      <div class="vscore" id="vscore"></div>
      <div class="bars">
        <div class="bar-row"><span class="bar-label">Real</span><div class="bar-bg"><div class="bar-fill-r" id="rbar"></div></div></div>
        <div class="bar-row"><span class="bar-label">Fake</span><div class="bar-bg"><div class="bar-fill-f" id="fbar"></div></div></div>
      </div>
    </div>
    <div class="section" style="margin-top:10px">
      <h3>🔍 Word-Level Explanation</h3>
      <div class="highlights" id="hl"></div>
      <div class="legend">🔴 Red = pushes toward FAKE &nbsp; 🟢 Green = pushes toward REAL</div>
    </div>
    <div class="section">
      <h3>📊 Key Indicators</h3>
      <div id="exp" style="font-size:.9rem;color:#444"></div>
    </div>
  </div>
  <p class="disclaimer">⚠️ Decision-support tool only. Always verify with trusted sources.</p>
</div>
<script>
function set(t){document.getElementById('txt').value=t}
async function analyze(){
  const text=document.getElementById('txt').value.trim();
  if(!text){alert('Please enter a claim.');return}
  document.getElementById('loading').style.display='block';
  document.getElementById('result').style.display='none';
  const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
  const d=await r.json();
  document.getElementById('loading').style.display='none';
  document.getElementById('result').style.display='block';
  const vbox=document.getElementById('vbox');
  const vlbl=document.getElementById('vlbl');
  if(d.label==='Fake'){
    vbox.className='fake-box';
    vlbl.innerHTML='<span class="fake-lbl">⚠️ LIKELY FAKE</span>';
  }else{
    vbox.className='real-box';
    vlbl.innerHTML='<span class="real-lbl">✅ LIKELY REAL</span>';
  }
  document.getElementById('vscore').textContent='Confidence: '+d.confidence+'% | Real: '+d.prob_real+'% | Fake: '+d.prob_fake+'%';
  document.getElementById('rbar').style.width=d.prob_real+'%';
  document.getElementById('fbar').style.width=d.prob_fake+'%';
  document.getElementById('hl').innerHTML=d.highlights;
  document.getElementById('exp').textContent=d.explanation;
}
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text"})
    try:
        headers  = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        response = requests.post(API_URL, headers=headers, json={"inputs": text}, timeout=30)
        result   = response.json()
        prob_real, prob_fake = 0.5, 0.5
        if isinstance(result, list):
            scores = result[0] if isinstance(result[0], list) else result
            for item in scores:
                if isinstance(item, dict):
                    lbl = item.get("label","").upper()
                    sc  = item.get("score", 0.0)
                    if lbl in ["LABEL_0","REAL","0"]:
                        prob_real = sc
                    elif lbl in ["LABEL_1","FAKE","1"]:
                        prob_fake = sc
        elif isinstance(result, dict) and "error" in result:
            return jsonify({"label":"Loading","confidence":0,"prob_real":50,"prob_fake":50,
                           "highlights":"Model loading, wait 20 seconds and retry.",
                           "explanation": result["error"]})
        pred  = 1 if prob_fake > prob_real else 0
        label = "Fake" if pred == 1 else "Real"
        conf  = round((prob_fake if pred==1 else prob_real)*100, 1)
        fake_found, real_found, html_parts = [], [], []
        for word in text.split():
            clean = word.lower().strip(".,!?;:'\"")
            if clean in FAKE_WORDS:
                fake_found.append(clean)
                html_parts.append(f'<span class="fw">{word}</span>')
            elif clean in REAL_WORDS:
                real_found.append(clean)
                html_parts.append(f'<span class="rw">{word}</span>')
            else:
                html_parts.append(word)
        highlights  = " ".join(html_parts)
        explanation = ""
        if fake_found: explanation += f"Fake indicators: {', '.join(set(fake_found))}\n"
        if real_found: explanation += f"Real indicators: {', '.join(set(real_found))}\n"
        if not fake_found and not real_found:
            explanation = "No strong indicator words — prediction based on overall context."
        return jsonify({"label":label,"confidence":conf,
                       "prob_real":round(prob_real*100,1),
                       "prob_fake":round(prob_fake*100,1),
                       "highlights":highlights,"explanation":explanation})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)