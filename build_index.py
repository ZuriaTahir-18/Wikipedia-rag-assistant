import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# Path to Google Drive data folder
DATA_PATH = "/content/drive/MyDrive/data"
os.makedirs(DATA_PATH, exist_ok=True)

# 1. Load small subset of English Wikipedia
print("Loading dataset...")
dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train[:1000]",                # ✅ FIX: use fixed number of rows instead of percentage
    cache_dir=os.path.join(DATA_PATH, "hf_cache") # HF cache in Drive
)

# 2. Chunking function
def chunk_text(text, max_len=800):
    sentences = text.split(". ")
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= max_len:
            cur += s + ". "
        else:
            if cur.strip(): chunks.append(cur.strip())
            cur = s + ". "
    if cur.strip(): chunks.append(cur.strip())
    return chunks

# 3. Prepare chunks with metadata
print("Preparing chunks...")
records = []
for i, row in enumerate(tqdm(dataset)):
    text = row["text"]
    title = row["title"]
    for c in chunk_text(text):
        records.append({"title": title, "text": c})

df = pd.DataFrame(records)
print(f"Total chunks: {len(df)}")
df.to_parquet(os.path.join(DATA_PATH, "metadata.parquet"))

# 4. Create embeddings
print("Creating embeddings...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embs = model.encode(df["text"].tolist(), show_progress_bar=True, batch_size=64)
embs = np.array(embs).astype("float32")
np.save(os.path.join(DATA_PATH, "embeddings.npy"), embs)

# 5. Build FAISS index
print("Building FAISS index...")
d = embs.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embs)
faiss.write_index(index, os.path.join(DATA_PATH, "wiki_index.faiss"))
print("Index built and saved in Drive.")
