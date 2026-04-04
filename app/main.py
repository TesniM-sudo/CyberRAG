# app/main.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI
from .rag_pipeline import build_index, ask_rag  # <-- relative import
from dotenv import load_dotenv
import os
load_dotenv()  # must be called before accessing os.getenv
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is not set. Check your .env file.")

executor = ThreadPoolExecutor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, build_index)
    yield

app = FastAPI(title="CyberRAG", lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "CyberRAG is running"}

@app.get("/ask")
def ask_endpoint(query: str):
    answer = ask_rag(query)
    return {"query": query, "answer": answer}