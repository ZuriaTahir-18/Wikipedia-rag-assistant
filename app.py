import os
import sys
import pandas as pd
import faiss
import numpy as np
from flask import Flask, request, render_template_string
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from pyngrok import ngrok

# -------------------------
HF_TOKEN = os.getenv("HF_TOKEN") or "your_hf_token_here"

NGROK_TOKEN = os.getenv("NGROK_AUTHTOKEN") or "your ngrok token here"
# -------------------------

app = Flask(__name__)

DATA_PATH = "/content/drive/MyDrive/data"

# ---------- Load resources ----------
try:
    print("Loading FAISS index and metadata from Drive...", file=sys.stderr)
    index = faiss.read_index(os.path.join(DATA_PATH, "wiki_index.faiss"))
    metadata = pd.read_parquet(os.path.join(DATA_PATH, "metadata.parquet"))
except Exception as e:
    print("ERROR loading index/metadata:", e, file=sys.stderr)
    raise

try:
    print("Loading embedding model (SentenceTransformer)...", file=sys.stderr)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print("ERROR loading embedding model:", e, file=sys.stderr)
    raise

try:
    print("Initializing HuggingFace InferenceClient (Mistral-7B-Instruct)...", file=sys.stderr)
    client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=HF_TOKEN)  
except Exception as e:
    print("ERROR initializing InferenceClient (check HF token / access):", e, file=sys.stderr)
    raise

# ---------- Retrieval & LLM wrapper ----------
def retrieve(query, k=5):
    q_emb = model.encode([query]).astype("float32")
    D, I = index.search(q_emb, k)
    return metadata.iloc[I[0]]

def ask_llm(question, contexts):
    ctx = "\n\n".join([f"- {c}" for c in contexts["text"].tolist()])
    prompt = (
        "You are a concise assistant. Answer in 2-3 short sentences (very brief). "
        "Do NOT add extra explanations. Use only the provided context passages.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    )
    try:
        res = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.2", 
            messages=[
                {"role": "system", "content": "Be concise and return a short summary (2-3 sentences)."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=120
        )
        text = res.choices[0].message["content"].strip()
        return text
    except Exception as e:
        return f"[LLM error] {e}"

# ---------- HTML frontend ----------
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Wikipedia RAG Assistant</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body{padding:30px; background-color:#f8f9fa;}
    .card{box-shadow:0 4px 10px rgba(0,0,0,0.08);}
    .passage{margin-bottom:10px;}
    .passage-title{font-weight:600;}
    pre{white-space:pre-wrap;}
  </style>
  <script>
    function showPassages(n){
      const container = document.getElementById('passage-container');
      if(!container) return;
      container.style.display = 'block';
      const items = container.querySelectorAll('.retrieved-passage');
      items.forEach((it, idx) => {
        it.style.display = (idx < n) ? 'block' : 'none';
      });
      container.scrollIntoView({behavior:'smooth', block:'start'});
    }
    function hidePassages(){
      const container = document.getElementById('passage-container');
      if(container) container.style.display = 'none';
    }
  </script>
</head>
<body>
  <div class="container">
    <h2 class="mb-3 text-primary">📖 Wikipedia RAG Assistant</h2>
    <p class="text-muted">Ask a question — answer will be short. Click a button to view retrieved passages (top 3 or top 5).</p>

    <!-- Main input -->
    <form method="post" action="/" class="mb-3">
      <div class="input-group">
        <input type="text" name="query" class="form-control" placeholder="Ask a question" required>
        <button type="submit" class="btn btn-primary">Ask</button>
      </div>
    </form>

    {% if answer %}
      <!-- Answer card -->
      <div class="card mb-3">
        <div class="card-header bg-success text-white"><strong>Answer</strong></div>
        <div class="card-body">
          <p>{{ answer }}</p>

          <!-- Buttons to show/hide passages -->
          <div class="mb-2">
            <button type="button" class="btn btn-sm btn-outline-primary" onclick="showPassages(3)">Show top 3 passages</button>
            <button type="button" class="btn btn-sm btn-outline-secondary" onclick="showPassages(5)">Show top 5 passages</button>
            <button type="button" class="btn btn-sm btn-outline-danger" onclick="hidePassages()">Hide passages</button>
          </div>

          <!-- Ask another (convenience) -->
          <form method="post" action="/" class="mt-3">
            <div class="input-group">
              <input type="text" name="query" class="form-control" placeholder="Ask another question..." required>
              <button type="submit" class="btn btn-outline-primary">Ask</button>
            </div>
          </form>
        </div>
      </div>

      <!-- Hidden passage container -->
      <div id="passage-container" style="display:none;">
        <div class="card mb-3">
          <div class="card-header bg-info text-white"><strong>Retrieved Passages (top {{ passages|length }})</strong></div>
          <div class="card-body">
            {% for item in passages %}
              <div class="retrieved-passage passage" style="display:none;">
                <div class="passage-title">{{ loop.index }}. {{ item[0] }}</div>
                <div class="passage-text">{{ item[1][:800] }}{% if item[1]|length > 800 %}...{% endif %}</div>
                <hr/>
              </div>
            {% endfor %}
            <div class="text-muted small">Showing only the number of passages you request. Click "Show top 3" or "Show top 5".</div>
          </div>
        </div>
      </div>
    {% endif %}

    <footer class="mt-4"><small>Powered by Wikipedia subset + Mistral-7B-Instruct</small></footer>
  </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    passages = None
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            ctxs = retrieve(query, k=5)
            answer = ask_llm(query, ctxs)
            passages = list(zip(ctxs["title"].tolist(), ctxs["text"].tolist()))
    return render_template_string(HTML_TEMPLATE, answer=answer, passages=passages)

# ---------- Run with pyngrok ----------
if __name__ == "__main__":
    try:
        ngrok.set_auth_token(NGROK_TOKEN)
        public_url = ngrok.connect(5000)
        print(" * ngrok tunnel:", public_url)
    except Exception as e:
        print("WARNING: ngrok tunnel could not be started:", e, file=sys.stderr)
    app.run(host="0.0.0.0", port=5000)
