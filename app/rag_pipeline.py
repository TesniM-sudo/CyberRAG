import os
import requests
from dotenv import load_dotenv
from app.embeddings import embed_text
import app.vector_store as vector_store

from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # <-- THIS IS THE NAME OF THE VARIABLE, NOT THE KEY
MODEL = "llama-3.3-70b-versatile"

def load_documents():
    docs = []
    folder = "data/docs"

    for file in os.listdir(folder):
        filepath = os.path.join(folder, file)
        if os.path.isfile(filepath):  # Fix: skip subdirectories
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                chunks = [text[i:i+300] for i in range(0, len(text), 300)]
                docs.extend(chunks)

    return docs

def build_index():
    vector_store.documents = load_documents()
    embeddings = embed_text(vector_store.documents)
    vector_store.create_index(embeddings)

def retrieve_context(query, k=3):
    query_embedding = embed_text([query])
    results = vector_store.search(query_embedding, k)
    return "\n".join(results)

def generate_answer(query, context):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in your .env file.")

    prompt = f"""You are a cybersecurity expert.

Context:
{context}

Question:
{query}

Answer clearly and professionally:
"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def ask_rag(query):
    context = retrieve_context(query)
    answer = generate_answer(query, context)
    return answer