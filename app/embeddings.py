from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(texts):
    embeddings = model.encode(texts)
    return np.array(embeddings, dtype=np.float32)  # Fix: FAISS requires float32 explicitly