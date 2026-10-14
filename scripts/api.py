"""FastAPI backend.  Run: uvicorn api:app --app-dir scripts --reload"""
from fastapi import FastAPI
from pydantic import BaseModel
from core import predict_next_day, search_knowledge
from agent import run_agent

app = FastAPI(title="Delhi Monsoon Weather Intelligence API")

class Q(BaseModel):
    question: str
@app.get("/")
def home():
    return {"status": "ok", "docs": "/docs", "endpoints": ["/forecast", "/search?q=monsoon", "/ask"]}
@app.get("/forecast")
def forecast(): return predict_next_day()

@app.get("/search")
def search(q: str): return {"results": search_knowledge(q)}

@app.post("/ask")
def ask(body: Q): return {"answer": run_agent(body.question)}
