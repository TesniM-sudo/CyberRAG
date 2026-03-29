from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.rag_pipeline import build_index, ask_rag

@asynccontextmanager
async def lifespan(app: FastAPI):
    build_index()
    yield


app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "CyberRAG is running"}

@app.get("/ask")
def ask(query: str):
    answer = ask_rag(query)
    return {"query": query, "answer": answer}